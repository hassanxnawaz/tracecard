"""Deterministic mock tools for the 20-task battery.

Every tool is pure: same arguments, same result. Two harness configs change
only how the tools present themselves to the model and how they fail:

- baseline: terse descriptions, loose schemas, bare error strings, full results.
- candidate: enums and examples in the schema, errors that name the next valid
  call, long results truncated with a page hint.
"""
from __future__ import annotations

import json

WEATHER = {"paris": 18, "tokyo": 24, "austin": 31, "oslo": 7, "lagos": 29}
RATES = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09, ("USD", "JPY"): 151.0, ("GBP", "USD"): 1.27}
ORDERS = {"A-1001": {"status": "shipped", "items": 3, "total": 84.5},
          "A-1002": {"status": "processing", "items": 1, "total": 19.0},
          "A-1003": {"status": "delivered", "items": 2, "total": 42.25}}
USERS = {"ana@example.com": {"name": "Ana Ruiz", "plan": "pro", "seats": 12},
         "li@example.com": {"name": "Li Wei", "plan": "free", "seats": 1}}
DOCS = {
    "doc-rate-limits": "API rate limits: 600 requests per minute per key. Burst 50.",
    "doc-refunds": "Refunds are issued within 5 business days to the original method.",
    "doc-sso": "SSO supports SAML 2.0 and OIDC. Enable it under Admin > Security.",
    "doc-retention": "Logs are retained for 30 days on Pro and 7 days on Free.",
    "doc-changelog": "\n".join(f"v1.{i}: maintenance release" for i in range(400)) + "\nv2.0: streaming tool calls",
}
CALENDAR = {"2026-11-02": ["09:00 standup", "14:00 design review"], "2026-11-03": []}


class ToolError(Exception):
    pass


def _need(args, key, kind=str):
    if key not in args:
        raise ToolError(f"missing required argument '{key}'")
    if not isinstance(args[key], kind):
        raise ToolError(f"argument '{key}' must be a " + ("number" if kind != str else "string"))
    return args[key]


def get_weather(args, cfg):
    city = _need(args, "city").lower()
    unit = args.get("unit", "C")
    if unit not in ("C", "F"):
        hint = " Use unit 'C' or 'F'." if cfg["hints"] else ""
        raise ToolError(f"invalid unit '{unit}'.{hint}")
    if city not in WEATHER:
        raise ToolError(f"unknown city '{city}'")
    c = WEATHER[city]
    return {"city": city.title(), "temp": c if unit == "C" else round(c * 9 / 5 + 32), "unit": unit}


def convert_currency(args, cfg):
    amt = _need(args, "amount", (int, float))
    src, dst = _need(args, "from").upper(), _need(args, "to").upper()
    if (src, dst) not in RATES:
        hint = f" Supported pairs: {', '.join(a + '>' + b for a, b in RATES)}." if cfg["hints"] else ""
        raise ToolError(f"unsupported pair {src}>{dst}.{hint}")
    return {"amount": round(amt * RATES[(src, dst)], 2), "currency": dst}


def calculator(args, cfg):
    expr = _need(args, "expression")
    if not set(expr) <= set("0123456789+-*/(). "):
        raise ToolError("invalid expression: digits and + - * / ( ) only")
    return {"result": round(eval(expr, {"__builtins__": {}}), 6)}  # noqa: S307 - charset checked above


def lookup_order(args, cfg):
    oid = _need(args, "order_id")
    if oid not in ORDERS:
        hint = " Order ids look like 'A-1001'." if cfg["hints"] else ""
        raise ToolError(f"order not found: {oid}.{hint}")
    return {"order_id": oid, **ORDERS[oid]}


def lookup_user(args, cfg):
    email = _need(args, "email").lower()
    if email not in USERS:
        raise ToolError(f"no user with email {email}")
    return USERS[email]


def search_docs(args, cfg):
    q = _need(args, "query").lower()
    words = [w for w in q.replace("?", "").split() if len(w) > 2]
    hits = [k for k, v in DOCS.items() if any(w in v.lower() or w in k for w in words)]
    if not hits and cfg["hints"]:
        return {"matches": [], "note": "No matches. Try one keyword, e.g. 'refund' or 'sso'."}
    return {"matches": hits}


def read_doc(args, cfg):
    did = _need(args, "doc_id")
    if did not in DOCS:
        hint = " Call search_docs first to get a doc_id." if cfg["hints"] else ""
        raise ToolError(f"unknown doc_id '{did}'.{hint}")
    text = DOCS[did]
    page = int(args.get("page", 0))
    if cfg["truncate"] and len(text) > 800:
        chunk = text[page * 800:(page + 1) * 800]
        return {"doc_id": did, "text": chunk, "page": page, "more": (page + 1) * 800 < len(text),
                "hint": "Pass page=N for more. The last page is " + str((len(text) - 1) // 800)}
    return {"doc_id": did, "text": text}


def get_calendar(args, cfg):
    day = _need(args, "date")
    return {"date": day, "events": CALENDAR.get(day, [])}


def create_event(args, cfg, state):
    title, day, time = _need(args, "title"), _need(args, "date"), _need(args, "time")
    if len(time) != 5 or time[2] != ":":
        hint = " Use 24h HH:MM, e.g. '15:30'." if cfg["hints"] else ""
        raise ToolError(f"invalid time '{time}'.{hint}")
    state.setdefault("events", []).append({"title": title, "date": day, "time": time})
    return {"created": True, "date": day, "time": time}


REGISTRY = {
    "get_weather": (get_weather, {"city": "string", "unit": ["C", "F"]}),
    "convert_currency": (convert_currency, {"amount": "number", "from": "string", "to": "string"}),
    "calculator": (calculator, {"expression": "string"}),
    "lookup_order": (lookup_order, {"order_id": "string"}),
    "lookup_user": (lookup_user, {"email": "string"}),
    "search_docs": (search_docs, {"query": "string"}),
    "read_doc": (read_doc, {"doc_id": "string", "page": "integer"}),
    "get_calendar": (get_calendar, {"date": "string"}),
    "create_event": (create_event, {"title": "string", "date": "string", "time": "string"}),
}

DESCRIPTIONS = {
    "baseline": {
        "get_weather": "Weather.", "convert_currency": "Convert money.", "calculator": "Math.",
        "lookup_order": "Order info.", "lookup_user": "User info.", "search_docs": "Search.",
        "read_doc": "Read a doc.", "get_calendar": "Calendar.", "create_event": "Make an event.",
    },
    "candidate": {
        "get_weather": "Current temperature for a city. unit is 'C' or 'F' (default C).",
        "convert_currency": "Convert an amount between ISO currency codes, e.g. from='USD', to='EUR'.",
        "calculator": "Evaluate arithmetic with digits and + - * / ( ). Use for any math.",
        "lookup_order": "Status, item count and total for an order id like 'A-1001'.",
        "lookup_user": "Plan and seat count for a user by email.",
        "search_docs": "Find doc ids by keyword. Returns ids only; then call read_doc.",
        "read_doc": "Read a doc by id from search_docs. Long docs are paged: pass page=N.",
        "get_calendar": "Events on a date (YYYY-MM-DD).",
        "create_event": "Create a calendar event. date YYYY-MM-DD, time 24h HH:MM.",
    },
}

CONFIGS = {
    "baseline": {"name": "baseline", "hints": False, "truncate": False, "enums": False},
    "candidate": {"name": "candidate", "hints": True, "truncate": True, "enums": True},
}


def schemas(cfg) -> list[dict]:
    out = []
    for name, (_, params) in REGISTRY.items():
        props = {}
        for k, t in params.items():
            if isinstance(t, list):
                props[k] = {"type": "string", "enum": t} if cfg["enums"] else {"type": "string"}
            else:
                props[k] = {"type": t}
        required = [k for k in params if k not in ("unit", "page")]
        out.append({"type": "function", "function": {
            "name": name, "description": DESCRIPTIONS[cfg["name"]][name],
            "parameters": {"type": "object", "properties": props, "required": required}}})
    return out


def call(name: str, args: dict, cfg: dict, state: dict) -> tuple[str, str | None]:
    """Run a tool. Returns (result_json, error_text)."""
    if name not in REGISTRY:
        return json.dumps({"error": f"no such tool '{name}'"}), f"no such tool '{name}'"
    fn = REGISTRY[name][0]
    try:
        res = fn(args, cfg, state) if name == "create_event" else fn(args, cfg)
        return json.dumps(res), None
    except ToolError as e:
        return json.dumps({"error": str(e)}), str(e)
