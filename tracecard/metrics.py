"""Per-run and per-arm metrics.

The scorecard keeps three tiers on purpose:
- Outcome: verified task completion. The only number that decides.
- Guardrails: wall time, tokens or tool-result data, cost. A change may not
  make these worse than the budget you set.
- Diagnostics: tool errors, retries, repeated calls, wrong tools, argument
  errors. These explain the outcome. They never decide it on their own.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass

from .model import Run, comparable

ARG_ERR = re.compile(r"(invalid|missing|required|unexpected|unknown|must be|not one of|validation|schema|argument)", re.I)


@dataclass
class Price:
    """USD per million tokens. Zero means cost is not reported."""
    input_per_m: float = 0.0
    output_per_m: float = 0.0


def _args_key(args) -> str:
    try:
        return json.dumps(args, sort_keys=True)
    except TypeError:
        return str(args)


def run_metrics(run: Run, price: Price | None = None) -> dict:
    tools = run.tool_steps
    llms = run.llm_steps
    errs = retries = repeats = wrong = arg_errs = 0
    last_err_tool = None
    prev_key = None
    for s in tools:
        key = s.name + _args_key(s.args)
        if last_err_tool == s.name:
            retries += 1                      # same tool right after it failed
        if key == prev_key and comparable(s.args):
            repeats += 1                      # identical call, identical args
        prev_key = key
        if run.expected_tools is not None and s.name not in run.expected_tools:
            wrong += 1
        if s.error:
            errs += 1
            last_err_tool = s.name
            if ARG_ERR.search(s.error or ""):
                arg_errs += 1
        else:
            last_err_tool = None
    pt = [s.prompt_tokens for s in llms if s.prompt_tokens is not None]
    ct = [s.completion_tokens for s in llms if s.completion_tokens is not None]
    # only trust token totals when every model call reported usage
    tokens_known = bool(llms) and len(pt) == len(llms)
    tin, tout = sum(pt), sum(ct)
    cost = None
    step_costs = [s.cost_usd for s in llms if s.cost_usd is not None]
    if llms and len(step_costs) == len(llms):
        cost = sum(step_costs)                  # provider-reported, preferred over a price table
    elif tokens_known and price and (price.input_per_m or price.output_per_m):
        cost = tin / 1e6 * price.input_per_m + tout / 1e6 * price.output_per_m
    return {
        "run_id": run.run_id, "task": run.task, "arm": run.arm, "rep": run.rep,
        "success": run.success,
        "llm_calls": len(llms), "tool_calls": len(tools),
        "tool_errors": errs, "retries": retries, "repeat_calls": repeats,
        "wrong_tool": wrong, "arg_errors": arg_errs,
        "wall_s": round(run.duration, 3),
        "llm_s": round(sum(s.duration for s in llms), 3),
        "tool_s": round(sum(s.duration for s in tools), 3),
        "result_kb": sum(s.result_bytes for s in tools) / 1024,
        "tokens_in": tin if tokens_known else None,
        "tokens_out": tout if tokens_known else None,
        "cost_usd": cost,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
    return xs[i]


def arm_summary(runs: list[Run], price: Price | None = None) -> dict:
    rows = [run_metrics(r, price) for r in runs]
    verified = [r for r in rows if r["success"] is not None]
    k = sum(1 for r in verified if r["success"])
    n = len(verified)
    lo, hi = wilson(k, n)
    keys = ["llm_calls", "tool_calls", "tool_errors", "retries", "repeat_calls",
            "wrong_tool", "arg_errors", "wall_s", "result_kb", "tokens_in", "tokens_out", "cost_usd"]
    per_tool: dict[str, list[float]] = {}
    for r in runs:
        for s in r.tool_steps:
            per_tool.setdefault(s.name, []).append(s.duration)
    llm_lat = [s.duration for r in runs for s in r.llm_steps]
    return {
        "runs": len(rows), "verified": n, "passed": k,
        "success_rate": k / n if n else None, "success_ci95": [lo, hi],
        "mean": {key: _mean([r[key] for r in rows]) for key in keys},
        "total_cost_usd": sum(r["cost_usd"] for r in rows if r["cost_usd"] is not None) if any(
            r["cost_usd"] is not None for r in rows) else None,
        "tokens_recorded": any(r["tokens_in"] is not None for r in rows),
        "token_coverage": (sum(1 for r in runs for s in r.llm_steps if s.prompt_tokens is not None)
                           / max(1, sum(len(r.llm_steps) for r in runs))),
        "latency": {
            "llm_p50": _pct(llm_lat, .5), "llm_p95": _pct(llm_lat, .95),
            "tools": {name: {"calls": len(d), "p50": _pct(d, .5), "p95": _pct(d, .95),
                             "total_s": sum(d)} for name, d in sorted(per_tool.items())},
        },
        "rows": rows,
    }
