"""Tool-use agent loop that records OpenTelemetry spans with OpenInference attributes.

Two model backends share one loop:
- ``OpenAICompatClient``: any OpenAI-compatible ``/chat/completions`` endpoint
  (NVIDIA NIM at https://integrate.api.nvidia.com/v1, vLLM, a local server).
- ``SimClient``: an offline stand-in so the pipeline runs with no keys. It
  follows each task's reference plan and injects failure modes at rates tied to
  the harness config. It measures the toolkit, not any real model.
"""
from __future__ import annotations

import json
import random
import time
import urllib.request
import uuid
from pathlib import Path

from . import tools as T

SYSTEM = "You are a careful assistant. Use the tools to answer. Reply with a short final answer."


# ------------------------------------------------------------ recording ----

class Recorder:
    def __init__(self, run_meta: dict):
        self.trace_id = uuid.uuid4().hex
        self.root_id = uuid.uuid4().hex[:16]
        self.spans: list[dict] = []
        self.meta = run_meta

    def span(self, name, kind, start, end, attrs, error=None, parent=None):
        a = {"openinference.span.kind": kind, **attrs}
        self.spans.append({
            "traceId": self.trace_id, "spanId": uuid.uuid4().hex[:16],
            "parentSpanId": parent if parent is not None else self.root_id, "name": name,
            "startTimeUnixNano": str(int(start * 1e9)), "endTimeUnixNano": str(int(end * 1e9)),
            "attributes": [_attr(k, v) for k, v in a.items() if v is not None],
            "status": {"code": 2, "message": error} if error else {"code": 1},
        })

    def otlp(self, start, end, output) -> dict:
        attrs = {f"tracecard.{k}": v for k, v in self.meta.items()}
        attrs["output.value"] = output
        root = {"traceId": self.trace_id, "spanId": self.root_id, "parentSpanId": "", "name": self.meta["task"],
                "startTimeUnixNano": str(int(start * 1e9)), "endTimeUnixNano": str(int(end * 1e9)),
                "attributes": [_attr("openinference.span.kind", "AGENT")] + [_attr(k, v) for k, v in attrs.items()],
                "status": {"code": 1}}
        return {"resourceSpans": [{"resource": {"attributes": [_attr("service.name", "tracecard-harness")]},
                                   "scopeSpans": [{"scope": {"name": "tracecard.harness"},
                                                   "spans": [root] + self.spans}]}]}


def _attr(k, v):
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": str(v)}}
    if isinstance(v, float):
        return {"key": k, "value": {"doubleValue": v}}
    if not isinstance(v, str):
        v = json.dumps(v)
    return {"key": k, "value": {"stringValue": v}}


# -------------------------------------------------------------- clients ----

class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.timeout = model, api_key, timeout

    def chat(self, messages, tools):
        body = json.dumps({"model": self.model, "messages": messages, "tools": tools,
                           "tool_choice": "auto", "temperature": 0}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        t1 = time.time()
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return msg, usage.get("prompt_tokens"), usage.get("completion_tokens"), t0, t1


RATES = {  # failure-mode probabilities for the simulator
    "baseline": {"bad_args": 0.5, "wrong_tool": 0.45, "recover": 0.35, "retry_same": 0.6, "skip_tool": 0.35, "lost": 0.45},
    "candidate": {"bad_args": 0.12, "wrong_tool": 0.1, "recover": 0.92, "retry_same": 0.1, "skip_tool": 0.08, "lost": 0.0},
}


class SimClient:
    """Deterministic simulated model. Seeded per task and rep."""

    def __init__(self, task: dict, config: str, seed: int, clock: list):
        self.task, self.cfg = task, config
        self.rng = random.Random(f"{seed}:{task['id']}:{config}")
        self.r = RATES[config]
        self.i = 0                 # plan step index
        self.pending = None        # what we sent last
        self.failed_once = False
        self.gave_up = False
        self.skipped = None
        self.lost = False
        self.tried_wrong: set = set()
        self.tried_bad: set = set()
        self.clock = clock
        self.model = "simulated-agent"

    def _tokens(self, messages, tools, out):
        p = len(json.dumps(messages)) // 4 + len(json.dumps(tools)) // 4
        return p, len(out) // 4 + 12

    def _call(self, name, args):
        self.pending = (name, args)
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex[:8], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}

    def _final(self, text):
        return {"role": "assistant", "content": text}

    def _decide(self, messages):
        plan = self.task["plan"]
        last = messages[-1]
        if last["role"] == "tool":
            res = json.loads(last["content"])
            if "error" in res:
                hinted = any(w in res["error"] for w in ("Use ", "Supported", "look like", "Call search_docs", "24h"))
                if self.failed_once and self.pending and self.rng.random() < self.r["retry_same"] and not hinted:
                    self.failed_once = False
                    return self._call(*self.pending)            # identical retry: the loop pattern
                if self.rng.random() < self.r["recover"]:
                    self.failed_once = False
                    step = plan[self.i]
                    return self._call(step["tool"], step["args"])
                if not self.failed_once:
                    self.failed_once = True
                    return self._call(*self.pending)
                return self._final("I could not complete this: the tool kept failing.")
            # success on a plan step
            step = plan[self.i]
            if step.get("long_doc") and res.get("more"):
                last_page = int(res["hint"].rsplit(" ", 1)[-1])
                if res.get("page") != last_page:
                    return self._call("read_doc", {**step["args"], "page": last_page})
            if step.get("long_doc") and not res.get("more") and "page" not in res and self.rng.random() < self.r["lost"]:
                self.lost = True
            self.i += 1
        if self.i >= len(plan):
            if self.skipped is not None:
                return self._final(f"About {self.skipped}.")
            return self._final(self.task["lost_answer"] if self.lost else self.task["answer"])
        step = plan[self.i]
        if "skippable" in step and self.rng.random() < self.r["skip_tool"]:
            self.skipped = step["skippable"]
            self.i = len(plan)
            return self._final(f"About {self.skipped}.")
        if "wrong_tool" in step and self.i not in self.tried_wrong:
            self.tried_wrong.add(self.i)
            if self.rng.random() < self.r["wrong_tool"]:
                return self._call(step["wrong_tool"]["tool"], step["wrong_tool"]["args"])
        if "bad_args" in step and self.i not in self.tried_bad:
            self.tried_bad.add(self.i)
            if self.rng.random() < self.r["bad_args"]:
                return self._call(step["tool"], step["bad_args"])
        if False:
            return self._call(step["tool"], step["bad_args"])
        return self._call(step["tool"], step["args"])

    def chat(self, messages, tools):
        msg = self._decide(messages)
        out = json.dumps(msg)
        p, c = self._tokens(messages, tools, out)
        t0 = self.clock[0]
        dt = 0.35 + p / 9000 + c / 90 + self.rng.random() * 0.25
        self.clock[0] = t0 + dt
        return msg, p, c, t0, t0 + dt


# ---------------------------------------------------------------- loop ----

def check(task: dict, output: str, state: dict) -> bool:
    c = task["check"]
    ok = all(s.lower() in (output or "").lower() for s in c.get("contains", []))
    if "event" in c:
        ok = ok and any(all(e.get(k) == v for k, v in c["event"].items()) for e in state.get("events", []))
    return ok


def run_task(task: dict, config: str, client, rep: int, max_turns: int = 8, sim_clock: list | None = None):
    cfg = T.CONFIGS[config]
    schemas = T.schemas(cfg)
    rec = Recorder({"task": task["id"], "arm": config, "rep": rep, "run_id": f"{task['id']}-r{rep}",
                    "model": getattr(client, "model", ""), "expected_tools": task["expected_tools"]})
    now = (lambda: sim_clock[0]) if sim_clock is not None else time.time
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    state: dict = {}
    output = ""
    start = now()
    for _ in range(max_turns):
        msg, pt, ct, t0, t1 = client.chat(messages, schemas)
        rec.span("chat", "LLM", t0, t1, {"llm.model_name": getattr(client, "model", ""),
                                         "llm.token_count.prompt": pt, "llm.token_count.completion": ct})
        calls = msg.get("tool_calls") or []
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        if not calls:
            output = msg.get("content") or ""
            break
        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            a0 = now()
            result, err = T.call(name, args, cfg, state)
            if sim_clock is not None:
                sim_clock[0] += 0.04 + len(result) / 200000
            a1 = now() if sim_clock is not None else time.time()
            rec.span(name, "TOOL", a0, a1, {"tool.name": name, "tool.parameters": args, "output.value": result}, error=err)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    end = now()
    ok = check(task, output, state)
    rec.meta["success"] = ok
    return rec.otlp(start, end, output), ok


def run_battery(config: str, out_dir: Path, reps: int = 3, client_factory=None, tasks_path: Path | None = None,
                seed: int = 11) -> dict:
    tasks = json.loads((tasks_path or Path(__file__).with_name("tasks.json")).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    passed = 0
    bundle: list = []
    for rep in range(reps):
        for t in tasks:
            if client_factory is None:
                clock = [1_790_000_000.0 + rep * 10_000 + tasks.index(t) * 100]
                client = SimClient(t, config, seed + rep, clock)
                doc, ok = run_task(t, config, client, rep, sim_clock=clock)
            else:
                doc, ok = run_task(t, config, client_factory(), rep)
            passed += ok
            bundle += doc["resourceSpans"]
    # One OTLP JSON file per arm, the way an exporter batches spans. Runs are split by traceId on load.
    (out_dir / "traces.otel.json").write_text(json.dumps({"resourceSpans": bundle}))
    return {"config": config, "runs": reps * len(tasks), "passed": passed}
