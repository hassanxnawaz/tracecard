"""Baseline vs candidate comparison and the decision rule.

The rule follows the standard NVIDIA describes for harness changes (NeMo Relay
post, Sep 30, 2026): call the candidate an improvement only when it produces a
repeatable increase in task completion, or keeps completion while improving the
reliability, latency, or cost measure you meant to change.

"Repeatable" is made concrete with a task-clustered bootstrap: resample tasks,
then reps inside each task, and require the 90% interval of the completion
difference to clear zero. Seeds are fixed so the same traces give the same call.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

from .metrics import Price, arm_summary, run_metrics
from .model import Run

LOWER_IS_BETTER = ("llm_calls", "tool_calls", "tool_errors", "retries", "wall_s",
                   "result_kb", "tokens_in", "cost_usd")


@dataclass
class Policy:
    target: str = "wall_s"            # the measure the change was meant to improve
    margin: float = 0.05              # max completion drop still called "kept"
    max_wall_increase: float = 0.25   # guardrail budgets, as fractions
    max_cost_increase: float = 0.25   # applies to cost_usd, else tokens_in, else result_kb
    n_boot: int = 4000
    seed: int = 7
    confidence: float = 0.90


@dataclass
class Comparison:
    verdict: str                      # SHIP | CHANGE | RERUN | STOP
    headline: str
    reasons: list[str]
    success: dict
    deltas: dict
    tasks: list[dict]
    baseline: dict
    candidate: dict
    next_step: str
    flags: list[str] = field(default_factory=list)


def _by_task(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["task"], []).append(r)
    return out


def _boot(bt, ct, tasks, key, n, rng, conf, relative=False):
    """Task-clustered bootstrap of mean(candidate) - mean(baseline)."""
    stats = []
    for _ in range(n):
        pick = [rng.choice(tasks) for _ in tasks]
        b_vals, c_vals = [], []
        for t in pick:
            b = bt[t]
            c = ct[t]
            b_vals += [rng.choice(b)[key] for _ in b]
            c_vals += [rng.choice(c)[key] for _ in c]
        b_vals = [float(v) for v in b_vals if v is not None]
        c_vals = [float(v) for v in c_vals if v is not None]
        if not b_vals or not c_vals:
            continue
        mb, mc = statistics.fmean(b_vals), statistics.fmean(c_vals)
        stats.append((mc - mb) / mb if relative and mb else mc - mb)
    if not stats:
        return None
    stats.sort()
    a = (1 - conf) / 2
    return [stats[int(a * (len(stats) - 1))], stats[int((1 - a) * (len(stats) - 1))]]


def compare(baseline: list[Run], candidate: list[Run], policy: Policy | None = None,
            price: Price | None = None) -> Comparison:
    p = policy or Policy()
    rng = random.Random(p.seed)
    bs, cs = arm_summary(baseline, price), arm_summary(candidate, price)
    brows = [dict(r, success=1.0 if r["success"] else 0.0) for r in bs["rows"] if r["success"] is not None]
    crows = [dict(r, success=1.0 if r["success"] else 0.0) for r in cs["rows"] if r["success"] is not None]
    bt, ct = _by_task(brows), _by_task(crows)
    tasks = sorted(set(bt) & set(ct))
    if not tasks:
        raise ValueError("baseline and candidate share no verified tasks")

    sb = statistics.fmean(r["success"] for t in tasks for r in bt[t])
    sc = statistics.fmean(r["success"] for t in tasks for r in ct[t])
    s_ci = _boot(bt, ct, tasks, "success", p.n_boot, rng, p.confidence)

    deltas = {}
    for key in LOWER_IS_BETTER:
        mb = bs["mean"].get(key)
        mc = cs["mean"].get(key)
        if mb is None or mc is None:
            deltas[key] = None
            continue
        rel = (mc - mb) / mb if mb else None
        ci = _boot(bt, ct, tasks, key, p.n_boot // 2, rng, p.confidence, relative=True) if mb else None
        deltas[key] = {"baseline": mb, "candidate": mc, "rel": rel, "rel_ci": ci}

    # Per-task view: where did the change actually move things?
    trows = []
    for t in tasks:
        b, c = bt[t], ct[t]
        b_ok, c_ok = sum(r["success"] for r in b), sum(r["success"] for r in c)
        b_turns = statistics.fmean(r["llm_calls"] for r in b)
        c_turns = statistics.fmean(r["llm_calls"] for r in c)
        note = []
        if c_ok / len(c) - b_ok / len(b) >= 0.5:
            note.append("flipped to pass")
        elif b_ok / len(b) - c_ok / len(c) >= 0.34:
            note.append("completion regressed")
        if b_turns and c_turns >= 2 * b_turns and c_turns - b_turns >= 2:
            if c_ok > b_ok:
                note.append(f"recovered: more turns ({b_turns:.1f} to {c_turns:.1f}) bought passes")
            else:
                note.append(f"turns {b_turns:.1f} to {c_turns:.1f}")
        if b_ok == 0 and c_ok == 0:
            note.append("fails on both arms")
        trows.append({"task": t, "baseline_pass": f"{int(b_ok)}/{len(b)}",
                      "candidate_pass": f"{int(c_ok)}/{len(c)}",
                      "baseline_turns": round(b_turns, 1), "candidate_turns": round(c_turns, 1),
                      "baseline_kb": round(statistics.fmean(r["result_kb"] for r in b), 1),
                      "candidate_kb": round(statistics.fmean(r["result_kb"] for r in c), 1),
                      "notes": note})

    # Guardrails
    reasons, flags = [], []
    cost_key = next((k for k in ("cost_usd", "tokens_in", "result_kb") if deltas.get(k)), None)
    wall = deltas.get("wall_s")
    over_wall = bool(wall and wall["rel"] is not None and wall["rel"] > p.max_wall_increase)
    over_cost = bool(cost_key and deltas[cost_key]["rel"] is not None
                     and deltas[cost_key]["rel"] > p.max_cost_increase)
    if over_wall:
        flags.append(f"wall time +{wall['rel']:.0%} (budget +{p.max_wall_increase:.0%})")
    if over_cost:
        label = {"cost_usd": "cost", "tokens_in": "input tokens", "result_kb": "tool-result data"}[cost_key]
        flags.append(f"{label} +{deltas[cost_key]['rel']:.0%} (budget +{p.max_cost_increase:.0%})")
    regressed_tasks = [t["task"] for t in trows if "completion regressed" in t["notes"] or
                       any(n.startswith("turns") for n in t["notes"])]
    if regressed_tasks:
        flags.append("task-level regressions: " + ", ".join(regressed_tasks))

    n_runs = len(brows) + len(crows)
    lo, hi = s_ci or (0.0, 0.0)
    target = deltas.get(p.target)
    target_better = bool(target and target["rel_ci"] and target["rel_ci"][1] < 0)
    k = (f"Completion {sb:.0%} to {sc:.0%} ({(sc - sb) * 100:+.0f} pts), {int(p.confidence * 100)}% interval "
         f"[{lo * 100:+.0f}, {hi * 100:+.0f}] pts")

    if lo > 0:
        if over_wall or over_cost or regressed_tasks:
            verdict, headline = "CHANGE", "Completion went up, but the price is over budget"
            nxt = "Keep the change. Work the flagged guardrails or task regressions, then rerun the same battery."
        else:
            verdict, headline = "SHIP", "Repeatable completion gain inside budget"
            nxt = "Ship behind a flag and rerun on the next model or workload it should support."
        reasons.append(k)
    elif hi < -p.margin:
        verdict, headline = "STOP", "Completion got worse"
        nxt = "Revert. Read the traces of the tasks that regressed before trying a variant."
        reasons.append(k)
    elif lo >= -p.margin and target_better and not over_cost:
        verdict, headline = "SHIP", f"Completion held and {p.target} improved"
        nxt = "Ship behind a flag. Watch completion on live traffic."
        reasons.append(k)
        reasons.append(f"{p.target} {target['rel']:+.0%}, interval [{target['rel_ci'][0]:+.0%}, {target['rel_ci'][1]:+.0%}]")
    elif abs(sc - sb) * n_runs / 2 <= 1.5 and hi - lo < 0.25:
        verdict, headline = "STOP", "No measurable effect on this model"
        nxt = "Do not claim a win here. If the change is cheap, keep it for the models where it helps."
        reasons.append(k)
    else:
        verdict, headline = "RERUN", "Promising, not yet repeatable"
        moved = [t["task"] for t in trows if t["notes"]]
        nxt = ("Add reps on the tasks that moved" + (f" ({', '.join(moved)})" if moved else "") +
               ". The interval still crosses zero at this sample size.")
        reasons.append(k)
    return Comparison(verdict=verdict, headline=headline, reasons=reasons,
                      success={"baseline": sb, "candidate": sc, "delta": sc - sb, "ci": s_ci,
                               "baseline_n": len(brows), "candidate_n": len(crows)},
                      deltas=deltas, tasks=trows, baseline=bs, candidate=cs,
                      next_step=nxt, flags=flags)
