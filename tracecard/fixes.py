"""Turn trace patterns into ranked fixes.

Each detector looks for one failure class, counts the runs it touches, and
estimates impact in two currencies a team can act on:
- failed runs that show the pattern (completion you could win back), and
- wasted steps or data in passing runs (latency and cost you could win back).

Impact = failed_runs_touched + 0.25 * wasted_steps_share * runs. It is a
ranking heuristic, not a forecast. The scorecard says so next to the number.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, asdict

from .model import Run, Step, comparable

RAW_TOOL_MARKUP = re.compile(r"<function=|<tool_call>|\"tool_calls\"\s*:", re.I)
EMPTY_RESULT = re.compile(r'("total_count":\s*0\b|"matches":\s*\[\]|^\s*\[\]\s*$|no (matches|results) found)', re.I)


@dataclass
class Finding:
    id: str
    title: str
    change: str            # what a team would change
    watch: str             # the metric that should move
    runs_touched: int
    failed_touched: int
    wasted_steps: int
    impact: float
    example: dict          # {"run_id", "step_index", "why"}

    def to_dict(self) -> dict:
        return asdict(self)


def _first(run: Run, pred) -> int | None:
    for i, s in enumerate(run.steps):
        if pred(i, s):
            return i
    return None


def _idx(run: Run, step: Step) -> int:
    return run.steps.index(step)


def diagnose(runs: list[Run], big_result_kb: float = 8.0) -> list[Finding]:
    if not runs:
        return []
    found: dict[str, dict] = {}

    def hit(fid, run, step_i, why, wasted=0):
        f = found.setdefault(fid, {"runs": set(), "failed": set(), "wasted": 0, "example": None})
        f["runs"].add(run.run_id)
        if run.success is False:
            f["failed"].add(run.run_id)
        f["wasted"] += wasted
        # prefer a failed run as the example, it tells the better story
        if f["example"] is None or (run.success is False and f["example"]["success"] is not False):
            f["example"] = {"run_id": run.run_id, "step_index": step_i, "why": why, "success": run.success}

    turns_by_task: dict[str, list[int]] = {}
    for r in runs:
        turns_by_task.setdefault(r.task, []).append(len(r.llm_steps))

    for r in runs:
        tools = r.tool_steps
        # 1. gave up after a tool error: failed run, last tool call errored
        if r.success is False and tools and tools[-1].error:
            hit("give_up", r, _idx(r, tools[-1]), f"last call to {tools[-1].name} failed, then the run ended")
        # 2. retry loops: the same failing call repeated
        for a, b in zip(tools, tools[1:]):
            if a.error and a.name == b.name and a.args == b.args and comparable(b.args):
                hit("retry_loop", r, _idx(r, b), f"{b.name} retried with identical arguments after an error", wasted=1)
        # 3. argument errors
        for s in tools:
            if s.error and re.search(r"(invalid|missing|required|unexpected|unknown|must be|not one of|schema|argument)", s.error, re.I):
                hit("arg_error", r, _idx(r, s), f"{s.name}: {s.error[:90]}", wasted=1)
                break
        # 4. wrong tool for the task
        if r.expected_tools is not None:
            for s in tools:
                if s.name not in r.expected_tools:
                    hit("wrong_tool", r, _idx(r, s), f"called {s.name}; task expects {', '.join(r.expected_tools)}", wasted=1)
                    break
        # 5. oversized tool results
        for s in tools:
            if s.result_bytes / 1024 > big_result_kb:
                hit("big_result", r, _idx(r, s), f"{s.name} returned {s.result_bytes/1024:.0f} KB into context",
                    wasted=int(s.result_bytes / 1024 // big_result_kb))
                break
        # 6. empty results followed by more searching
        for a, b in zip(tools, tools[1:]):
            if not a.error and EMPTY_RESULT.search(a.result_preview or "") and b.name == a.name:
                extra = sum(1 for s in tools[tools.index(b):] if s.name == a.name)
                hit("empty_probe", r, _idx(r, a), f"{a.name} came back empty, agent searched {extra} more time(s)", wasted=extra)
                break
        # 7. turn blow-up versus the task's own median
        med = statistics.median(turns_by_task[r.task])
        if med and len(r.llm_steps) >= 2 * med and len(r.llm_steps) - med >= 3:
            hit("turn_blowup", r, _idx(r, r.llm_steps[int(med)]),
                f"{len(r.llm_steps)} model calls vs task median {med:g}", wasted=int(len(r.llm_steps) - med))
        # 8. provider broke the tool-call template (not your harness)
        if RAW_TOOL_MARKUP.search(r.output or ""):
            hit("raw_markup", r, len(r.steps) - 1 if r.steps else 0, "final answer contains raw tool-call markup")

    catalog = {
        "give_up": ("Agent gives up after a tool error",
                    "Make tool errors recoverable: say what failed and the next valid call (recovery recipe).",
                    "completion on tasks with tool errors"),
        "retry_loop": ("Identical retries after an error",
                       "Return an actionable error and cap identical retries at one.",
                       "retries per run, wall time"),
        "arg_error": ("Bad tool arguments",
                      "Tighten the schema: enums, examples, required fields; validate before execution.",
                      "argument errors per run"),
        "wrong_tool": ("Wrong tool picked",
                       "De-duplicate overlapping tools and sharpen descriptions around when to use each.",
                       "wrong-tool calls per run"),
        "big_result": ("Oversized tool results",
                       "Paginate or truncate results and spill the full output to a file the agent can page through.",
                       "tool-result KB, input tokens"),
        "empty_probe": ("Empty results trigger blind re-searching",
                        "When a search returns nothing, say why (case, hidden files, path) and suggest the next query.",
                        "tool calls per run on search tasks"),
        "turn_blowup": ("Runs far longer than the task median",
                        "Read these traces first. Usually one probe or tool output sends the model exploring.",
                        "model calls per run, p95 wall time"),
        "raw_markup": ("Provider returned raw tool-call markup",
                       "Fix or pin the provider chat template. This deflates completion on both arms and is not a harness effect.",
                       "completion, after excluding provider-broken runs"),
    }
    n = len(runs)
    out = []
    for fid, f in found.items():
        title, change, watch = catalog[fid]
        share = f["wasted"] / max(1, sum(len(r.steps) for r in runs))
        impact = len(f["failed"]) + 0.25 * share * n
        ex = dict(f["example"])
        ex.pop("success", None)
        out.append(Finding(id=fid, title=title, change=change, watch=watch,
                           runs_touched=len(f["runs"]), failed_touched=len(f["failed"]),
                           wasted_steps=f["wasted"], impact=round(impact, 2), example=ex))
    out.sort(key=lambda x: (-x.impact, -x.runs_touched))
    return out


def first_anomaly(run: Run) -> int | None:
    """Index of the step where the run first went wrong, for the viewer."""
    i = _first(run, lambda i, s: bool(s.error))
    if i is not None:
        return i
    tools = run.tool_steps
    for a, b in zip(tools, tools[1:]):
        if EMPTY_RESULT.search(a.result_preview or "") and a.name == b.name:
            return _idx(run, a)
    return None


def step_flags(run: Run, big_result_kb: float = 8.0) -> dict[int, str]:
    """Per-step labels for the trace viewer."""
    flags: dict[int, str] = {}
    tools = run.tool_steps
    for a, b in zip(tools, tools[1:]):
        if a.error and a.name == b.name and a.args == b.args and comparable(b.args):
            flags[_idx(run, b)] = "identical retry after an error"
        if not a.error and EMPTY_RESULT.search(a.result_preview or "") and a.name == b.name:
            flags.setdefault(_idx(run, a), "empty result, then more searching")
    for s in tools:
        i = _idx(run, s)
        if s.error:
            flags.setdefault(i, "tool error: " + s.error[:80])
        elif s.result_bytes / 1024 > big_result_kb:
            flags.setdefault(i, f"{s.result_bytes/1024:.0f} KB result fed back to the model")
        if run.expected_tools is not None and s.name not in run.expected_tools:
            flags[i] = f"wrong tool: {s.name} (expected {', '.join(run.expected_tools)})"
    if RAW_TOOL_MARKUP.search(run.output or "") and run.llm_steps:
        flags.setdefault(_idx(run, run.llm_steps[-1]), "final answer is raw tool-call markup (provider template)")
    return flags
