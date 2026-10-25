"""Readers for the trace formats agent teams already produce.

Supported inputs
- NeMo Relay ATOF (``*.atof.jsonl``): one file per run. Scope start/end events.
- OpenTelemetry spans with OpenInference attributes: OTLP JSON
  (``{"resourceSpans": ...}``), or a flat list of spans as exported by Phoenix.
- NeMo Agent Toolkit intermediate steps: ``all_requests_profiler_traces.json``
  or ``workflow_output.json`` from ``nat eval``.

Run metadata (task, arm, rep, success) comes from the trace when present, or
from a sidecar ``meta.jsonl`` with one JSON object per run (``run_id``, ``task``,
``arm``, ``rep``, ``model``, optional ``success``, ``wall_s``, ``tail``).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .model import Run, Step

Verifier = Callable[[Run, dict], "bool | None"]

_ERR_RE = re.compile(r'"error":\s*"(?!null)')
_EXIT_RE = re.compile(r'"exit_code":\s*[1-9-]')


def _ts(value: Any) -> float:
    """Accept ISO-8601 (with nanoseconds), epoch seconds, or epoch nanoseconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e17:          # nanoseconds
            return v / 1e9
        if v > 1e14:          # microseconds
            return v / 1e6
        if v > 1e11:          # milliseconds
            return v / 1e3
        return v
    s = str(value)
    if s.isdigit():
        return _ts(int(s))
    m = re.match(r"(.*T\d\d:\d\d:\d\d)(\.\d+)?(.*)", s)
    if m:
        base, frac, tz = m.groups()
        tz = tz.replace("Z", "+00:00") or "+00:00"
        dt = datetime.fromisoformat(base + tz)
        return dt.timestamp() + (float(frac) if frac else 0.0)
    return datetime.fromisoformat(s).timestamp()


def _as_text(data: Any) -> str:
    if data is None:
        return ""
    return data if isinstance(data, str) else json.dumps(data)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _looks_like_error(text: str) -> str | None:
    """Heuristic used by Hermes ToolPerf's ab_eval.py: an "error" key with a
    non-null value near the start, or a nonzero exit code."""
    if _ERR_RE.search(text[:1500]) or _EXIT_RE.search(text[:200]):
        m = re.search(r'"error":\s*"([^"]{0,160})', text[:1500])
        return m.group(1) if m else "nonzero exit code"
    head = text.lstrip()[:200].lower()
    if head.startswith(("error", "traceback", "exception")):
        return text.strip()[:160]
    return None


# ---------------------------------------------------------------- ATOF ----

def load_atof(path: Path) -> Run:
    starts: dict[str, dict] = {}
    steps: list[Step] = []
    model = ""
    output = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("kind") != "scope":
            continue
        cat, phase, uid = ev.get("category"), ev.get("scope_category"), ev.get("uuid")
        if cat not in ("llm", "tool"):
            continue
        if phase == "start":
            starts[uid] = ev
            continue
        if phase != "end":
            continue
        st = starts.pop(uid, ev)
        meta = ev.get("metadata") or {}
        data = ev.get("data")
        text = _as_text(data)
        if cat == "llm":
            usage = (data or {}).get("usage") if isinstance(data, dict) else None
            usage = usage or {}
            model = (data or {}).get("model", model) if isinstance(data, dict) else model
            try:
                msg = data["choices"][0]["message"]
                if msg.get("content") and not msg.get("tool_calls"):
                    output = msg["content"]
            except (TypeError, KeyError, IndexError):
                pass
            status = meta.get("otel.status_code")
            steps.append(Step(
                kind="llm", name=ev.get("name") or "llm",
                start=_ts(st.get("timestamp")), end=_ts(ev.get("timestamp")),
                error=None if status in (None, "OK", "UNSET") else str(status),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                span_id=uid or "", parent_id=ev.get("parent_uuid") or ""))
        else:
            status = meta.get("status") or meta.get("otel.status_code")
            err = None
            if status not in (None, "ok", "OK", "UNSET"):
                err = str(status)
            err = err or _looks_like_error(text)
            steps.append(Step(
                kind="tool", name=ev.get("name") or "tool",
                start=_ts(st.get("timestamp")), end=_ts(ev.get("timestamp")),
                args=st.get("data"), error=err, result_bytes=len(text),
                result_preview=text[:240],
                span_id=uid or "", parent_id=ev.get("parent_uuid") or ""))
    steps.sort(key=lambda s: s.start)
    name = Path(path).name.replace(".atof.jsonl", "").replace(".jsonl", "")
    return Run(run_id=name, task=re.sub(r"-r\d+$", "", name), model=model,
               steps=steps, output=output, source=str(path))


# ------------------------------------------------------- OpenTelemetry ----

def _otlp_value(v: dict) -> Any:
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return int(v[k]) if k == "intValue" else v[k]
    if "arrayValue" in v:
        return [_otlp_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def _flatten_spans(doc: Any) -> list[dict]:
    """Return spans as dicts with flat ``attributes``."""
    spans: list[dict] = []
    if isinstance(doc, dict) and "resourceSpans" in doc:
        for rs in doc["resourceSpans"]:
            for ss in rs.get("scopeSpans", rs.get("instrumentationLibrarySpans", [])):
                for sp in ss.get("spans", []):
                    attrs = {a["key"]: _otlp_value(a["value"]) for a in sp.get("attributes", [])}
                    status = sp.get("status") or {}
                    code = status.get("code")
                    spans.append({
                        "trace_id": sp.get("traceId"), "span_id": sp.get("spanId"),
                        "parent_id": sp.get("parentSpanId") or "", "name": sp.get("name"),
                        "start": _ts(int(sp.get("startTimeUnixNano", 0))),
                        "end": _ts(int(sp.get("endTimeUnixNano", 0))),
                        "attributes": attrs,
                        "error": (status.get("message") or "error")
                        if code in (2, "STATUS_CODE_ERROR", "ERROR") else None,
                    })
        return spans
    items = doc if isinstance(doc, list) else doc.get("spans", doc.get("data", []))
    for sp in items:
        ctx = sp.get("context", {})
        status = sp.get("status_code") or (sp.get("status") or {}).get("status_code")
        attrs = dict(sp.get("attributes") or {})
        spans.append({
            "trace_id": ctx.get("trace_id") or sp.get("trace_id"),
            "span_id": ctx.get("span_id") or sp.get("span_id"),
            "parent_id": sp.get("parent_id") or "", "name": sp.get("name"),
            "start": _ts(sp.get("start_time")), "end": _ts(sp.get("end_time")),
            "attributes": attrs,
            "error": (sp.get("status_message") or "error") if str(status).upper() == "ERROR" else None,
        })
    return spans


def load_otel(path: Path) -> list[Run]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    by_trace: dict[str, list[dict]] = {}
    for sp in _flatten_spans(doc):
        by_trace.setdefault(sp["trace_id"] or "trace", []).append(sp)
    runs = []
    for trace_id, spans in by_trace.items():
        spans.sort(key=lambda s: s["start"])
        root = next((s for s in spans if not s["parent_id"]), spans[0])
        ra = root["attributes"]
        steps = []
        model = ""
        for sp in spans:
            a = sp["attributes"]
            kind = str(a.get("openinference.span.kind", "")).upper()
            if kind == "LLM":
                model = a.get("llm.model_name", model)
                steps.append(Step(
                    kind="llm", name=a.get("llm.model_name") or sp["name"],
                    start=sp["start"], end=sp["end"], error=sp["error"],
                    prompt_tokens=a.get("llm.token_count.prompt"),
                    completion_tokens=a.get("llm.token_count.completion"),
                    cost_usd=a.get("llm.cost.total"),
                    span_id=sp["span_id"] or "", parent_id=sp["parent_id"]))
            elif kind == "TOOL":
                out = _as_text(a.get("output.value"))
                args = _maybe_json(a.get("tool.parameters") or a.get("input.value"))
                steps.append(Step(
                    kind="tool", name=a.get("tool.name") or sp["name"],
                    start=sp["start"], end=sp["end"], args=args,
                    error=sp["error"] or _looks_like_error(out),
                    result_bytes=int(a.get("tracecard.result_bytes", len(out))), result_preview=out[:240],
                    span_id=sp["span_id"] or "", parent_id=sp["parent_id"]))
        meta = _maybe_json(ra.get("metadata")) or {}
        meta = meta if isinstance(meta, dict) else {}
        get = lambda k, d=None: ra.get(f"tracecard.{k}", meta.get(k, d))
        success = get("success")
        expected = _maybe_json(get("expected_tools"))
        runs.append(Run(
            run_id=str(get("run_id", trace_id)), task=str(get("task", root["name"])),
            arm=str(get("arm", "default")), rep=int(get("rep", 0)),
            model=str(get("model", model)),
            success=None if success is None else bool(success),
            steps=steps, wall_s=root["end"] - root["start"],
            output=_as_text(ra.get("output.value")), source=str(path),
            expected_tools=expected if isinstance(expected, list) else None))
    return runs


# ------------------------------------------------- NeMo Agent Toolkit ----

def _nat_requests(doc: Any) -> list[tuple[dict, list[dict]]]:
    """Yield (item metadata, steps) pairs from NAT eval/profiler outputs."""
    out = []
    if isinstance(doc, dict):
        doc = doc.get("requests") or doc.get("items") or [doc]
    for i, item in enumerate(doc):
        if isinstance(item, list):
            out.append(({"id": i}, item))
        elif isinstance(item, dict) and "intermediate_steps" in item:
            out.append((item, item["intermediate_steps"]))
        elif isinstance(item, dict) and "payload" in item:
            out.append(({"id": 0}, doc))
            break
    return out


def load_nat(path: Path) -> list[Run]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = []
    for item, events in _nat_requests(doc):
        starts: dict[str, dict] = {}
        steps = []
        model = ""
        for ev in events:
            p = ev.get("payload", ev)
            et = str(p.get("event_type", ""))
            uid = p.get("UUID") or p.get("uuid")
            if et in ("LLM_START", "TOOL_START"):
                starts[uid] = p
                continue
            if et not in ("LLM_END", "TOOL_END"):
                continue
            st = starts.pop(uid, p)
            data = p.get("data") or {}
            start = _ts(p.get("span_event_timestamp") or st.get("event_timestamp"))
            end = _ts(p.get("event_timestamp"))
            if et == "LLM_END":
                tu = ((p.get("usage_info") or {}).get("token_usage") or {})
                model = p.get("name") or model
                steps.append(Step(kind="llm", name=p.get("name") or "llm", start=start, end=end,
                                  prompt_tokens=tu.get("prompt_tokens"),
                                  completion_tokens=tu.get("completion_tokens"),
                                  span_id=uid or "", parent_id=ev.get("parent_id", "")))
            else:
                out = _as_text(data.get("output"))
                steps.append(Step(kind="tool", name=p.get("name") or "tool", start=start, end=end,
                                  args=_maybe_json(data.get("input") or (st.get("data") or {}).get("input")),
                                  error=_looks_like_error(out), result_bytes=len(out),
                                  result_preview=out[:240],
                                  span_id=uid or "", parent_id=ev.get("parent_id", "")))
        steps.sort(key=lambda s: s.start)
        rid = str(item.get("id", len(runs)))
        runs.append(Run(run_id=rid, task=rid, model=model, steps=steps,
                        output=_as_text(item.get("generated_answer", "")), source=str(path)))
    return runs


# ------------------------------------------------------------ loading ----

def detect(path: Path) -> str:
    p = Path(path)
    if p.name.endswith(".atof.jsonl"):
        return "atof"
    head = p.read_text(encoding="utf-8")[:4000]
    if '"atof_version"' in head:
        return "atof"
    if '"resourceSpans"' in head or "openinference.span.kind" in head:
        return "otel"
    if '"event_type"' in head or '"intermediate_steps"' in head:
        return "nat"
    raise ValueError(f"unrecognized trace format: {p}")


def load_file(path: Path) -> list[Run]:
    kind = detect(path)
    if kind == "atof":
        return [load_atof(path)]
    if kind == "otel":
        return load_otel(path)
    return load_nat(path)


def _read_meta(path: Path) -> dict[str, dict]:
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            meta[str(m["run_id"])] = m
    return meta


def load_path(path: str | Path, arm: str | None = None,
              verifier: Verifier | None = None) -> list[Run]:
    """Load every trace under ``path``. Applies sidecar metadata and a verifier."""
    root = Path(path)
    files: Iterable[Path]
    if root.is_dir():
        files = sorted(f for f in root.rglob("*")
                       if f.is_file() and f.suffix in (".json", ".jsonl") and f.name != "meta.jsonl")
    else:
        files = [root]
    runs: list[Run] = []
    for f in files:
        try:
            loaded = load_file(f)
        except ValueError:
            continue
        side = f.parent / "meta.jsonl"
        meta = _read_meta(side) if side.exists() else {}
        for r in loaded:
            m = meta.get(r.run_id, {})
            r.task = str(m.get("task", r.task))
            r.arm = arm or str(m.get("arm", r.arm))
            r.rep = int(m.get("rep", r.rep))
            r.model = str(m.get("model", r.model))
            if "wall_s" in m:
                r.wall_s = float(m["wall_s"])
            if m.get("tail") and not r.output:
                r.output = m["tail"]
            if "success" in m:
                r.success = bool(m["success"])
            if verifier is not None:
                v = verifier(r, m)
                if v is not None:
                    r.success = bool(v)
            runs.append(r)
    return runs


def load_verifier(spec: str) -> Verifier:
    """Load ``path/to/file.py:function`` as a verifier ``f(run, meta) -> bool | None``."""
    import importlib.util
    file, _, fn = spec.partition(":")
    mod_spec = importlib.util.spec_from_file_location("tracecard_verifier", file)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, fn or "verify")
