"""Markdown and JSON scorecards."""
from __future__ import annotations

from .compare import Comparison
from .fixes import Finding


def _f(v, fmt="{:.1f}", none="n/a"):
    return none if v is None else fmt.format(v)


def _pct(v):
    return "n/a" if v is None else f"{v:+.0%}"


METRICS = [
    ("llm_calls", "Model calls / run"), ("tool_calls", "Tool calls / run"),
    ("tool_errors", "Tool errors / run"), ("retries", "Retries / run"),
    ("wall_s", "Wall time / run (s)"), ("result_kb", "Tool-result data / run (KB)"),
    ("tokens_in", "Input tokens / run"), ("cost_usd", "Cost / run (USD)"),
]


def scorecard_md(title: str, summary: dict, findings: list[Finding]) -> str:
    s = summary
    lines = [f"# {title}", ""]
    if s["success_rate"] is not None:
        lo, hi = s["success_ci95"]
        lines.append(f"**Verified completion: {s['passed']}/{s['verified']} ({s['success_rate']:.0%}),"
                     f" 95% interval {lo:.0%} to {hi:.0%}.**")
    else:
        lines.append("**No verifier results. Completion unknown; add a verifier before trusting any other number.**")
    lines += ["", "| Metric | Mean |", "|---|---|"]
    for key, label in METRICS:
        lines.append(f"| {label} | {_f(s['mean'].get(key), '{:.3f}' if key == 'cost_usd' else '{:.1f}')} |")
    if s.get("token_coverage", 1) < 0.9:
        lines.append(f"\nTokens were reported on {s['token_coverage']:.0%} of model calls, so token and cost"
                     " numbers are left out. Tool-result KB is the context-cost proxy.")
    lines += ["", "## Top fixes, ranked by impact", ""]
    if not findings:
        lines.append("No known failure pattern found in these traces.")
    for i, f in enumerate(findings[:3], 1):
        lines += [f"{i}. **{f.title}** (impact {f.impact}: {f.failed_touched} failed / {f.runs_touched} runs touched)",
                  f"   - Change: {f.change}",
                  f"   - Watch: {f.watch}",
                  f"   - Example: `{f.example['run_id']}` step {f.example['step_index']}: {f.example['why']}"]
    lines += ["", "_Impact = failed runs touched + 0.25 x wasted-step share x runs. A ranking aid, not a forecast._"]
    return "\n".join(lines) + "\n"


def comparison_md(title: str, c: Comparison) -> str:
    s = c.success
    lines = [f"# {title}", "", f"## Decision: {c.verdict}. {c.headline}", ""]
    lines += [f"- {r}" for r in c.reasons]
    lines += [f"- Guardrail: {fl}" for fl in c.flags]
    lines += [f"- Next: {c.next_step}", "",
              "| Metric | Baseline | Candidate | Change |", "|---|---|---|---|",
              f"| Verified completion | {s['baseline']:.0%} (n={s['baseline_n']}) | "
              f"{s['candidate']:.0%} (n={s['candidate_n']}) | {s['delta']*100:+.0f} pts |"]
    for key, label in METRICS:
        d = c.deltas.get(key)
        if d:
            fmt = "{:.3f}" if key == "cost_usd" else "{:.1f}"
            lines.append(f"| {label} | {_f(d['baseline'], fmt)} | {_f(d['candidate'], fmt)} | {_pct(d['rel'])} |")
    lines += ["", "## Per task", "", "| Task | Pass (base to cand) | Turns | Tool KB | Notes |", "|---|---|---|---|---|"]
    for t in c.tasks:
        lines.append(f"| {t['task']} | {t['baseline_pass']} to {t['candidate_pass']} | "
                     f"{t['baseline_turns']} to {t['candidate_turns']} | {t['baseline_kb']} to {t['candidate_kb']} | "
                     f"{'; '.join(t['notes'])} |")
    return "\n".join(lines) + "\n"


def comparison_json(c: Comparison) -> dict:
    strip = lambda s: {k: v for k, v in s.items() if k != "rows"}
    return {"verdict": c.verdict, "headline": c.headline, "reasons": c.reasons, "flags": c.flags,
            "next_step": c.next_step, "success": c.success, "deltas": c.deltas, "tasks": c.tasks,
            "baseline": strip(c.baseline), "candidate": strip(c.candidate)}
