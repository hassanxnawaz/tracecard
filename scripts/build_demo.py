"""Build docs/index.html, the static demo published on GitHub Pages.

  python scripts/fetch_hermes.sh            # once: public Hermes ToolPerf traces (~28 MB)
  python scripts/build_demo.py --hermes data/hermes/results

Tabs:
1-2. Hermes ToolPerf Aug 6, 2026 rerun (Nous Research), real public traces,
     the case NVIDIA's NeMo Relay post walks through. Step metadata only:
     tool names, timing, sizes, error flags. No prompts, arguments or results.
3.   The 20-task harness in this repo, simulated agent, labeled as such.
4.   Dogfood: the trace of the agent session that built this repo (if exported).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracecard.adapters import load_path, load_verifier  # noqa: E402
from tracecard.compare import Policy, compare  # noqa: E402
from tracecard.fixes import RAW_TOOL_MARKUP, diagnose  # noqa: E402
from tracecard.metrics import Price, arm_summary  # noqa: E402
from tracecard.viewer import dataset, write_html  # noqa: E402

REPO = "https://github.com/hassanxnawaz/tracecard"
CREDIT = "Designed and directed by Hassan Nawaz. Built by AI coding agents working under his direction. MIT licensed. Not affiliated with or endorsed by NVIDIA or Nous Research."

METHOD = """<h4>How the call is made</h4>
<p>Completion comes from a deterministic verifier. The interval is a task-clustered bootstrap
(resample tasks, then reps), 4,000 draws, fixed seed, so the same traces always give the same call.</p>
<p><b>SHIP</b>: the completion interval clears zero inside the budgets, or completion holds within 5 pts and the
target metric improves. <b>CHANGE</b>: completion is up but a guardrail or a task regressed.
<b>RERUN</b>: the interval still crosses zero. <b>STOP</b>: worse, or no measurable effect.</p>
<p class="src">The rule is the one NVIDIA states for harness changes in
<a href="https://developer.nvidia.com/blog/tracing-agent-harness-behavior-with-nvidia-nemo-relay">Tracing Agent Harness
Behavior with NeMo Relay</a> (Sep 30, 2026): call it an improvement only when completion rises repeatably, or holds while the
measure you meant to change gets better.</p>"""


def hermes(root: Path):
    v = load_verifier(str(ROOT / "examples/hermes_toolperf/verifier.py") + ":verify")
    out = []
    models = [("anthropic_claude-sonnet-4.5", "Hermes: Sonnet 4.5"), ("qwen_qwen3-coder-30b-a3b-instruct", "Hermes: Qwen3 Coder 30B")]
    for key, label in models:
        b = load_path(root / key / "baseline", arm="baseline", verifier=v)
        c = load_path(root / key / "fixes", arm="candidate", verifier=v)
        cmp = compare(b, c, Policy(target="wall_s"))
        sens = []
        bad_b = [r for r in b if RAW_TOOL_MARKUP.search(r.output or "")]
        bad_c = [r for r in c if RAW_TOOL_MARKUP.search(r.output or "")]
        sens.append(f"Reproduces the published table: {cmp.success['baseline']*27:.0f}/27 to {cmp.success['candidate']*27:.0f}/27, "
                    f"model calls {cmp.deltas['llm_calls']['baseline']:.1f} to {cmp.deltas['llm_calls']['candidate']:.1f}, "
                    f"tool data {cmp.deltas['result_kb']['baseline']:.0f} to {cmp.deltas['result_kb']['candidate']:.0f} KB, "
                    f"wall {cmp.deltas['wall_s']['baseline']:.0f} to {cmp.deltas['wall_s']['candidate']:.0f} s.")
        if bad_b or bad_c:
            keep = lambda rs: [r for r in rs if not RAW_TOOL_MARKUP.search(r.output or "")]
            c2 = compare(keep(b), keep(c), Policy(target="wall_s"))
            tasks_b = sorted({r.task for r in bad_b})
            sens.append(f"{len(bad_b)} baseline and {len(bad_c)} candidate runs ended in raw tool-call markup (provider template breakage)"
                        + (f", baseline ones all on {', '.join(tasks_b)}. " if bad_b else ". ") +
                        f"Without them: {c2.success['baseline']:.0%} to {c2.success['candidate']:.0%}, interval "
                        f"[{c2.success['ci'][0]*100:+.0f}, {c2.success['ci'][1]*100:+.0f}] pts, call {c2.verdict}.")
        if "Sonnet" in label:
            pm = """<h4>What a PM takes from this</h4>
<p>No effect on the strong model. Completion moved by one run of 27 and every trace measure is flat except wall time,
which is not a harness effect worth claiming either way.</p>
<p>The useful output is permission to stop: the fixes cost Sonnet nothing, so keep them for the models where they help,
and don't report a Sonnet win.</p>
<p>One task, hidden-file search, fails on both arms. That's the open product gap, and it's where the next change should aim.</p>"""
        else:
            pm = """<h4>What a PM takes from this</h4>
<p>The fixes look like they did their job. Inline-script and big-output runs that the baseline abandoned now finish,
and the extra turns are the price of recovering. That's the reading NVIDIA's post gives too.</p>
<p>The traces raise one question before anyone calls it a win. All 5 baseline runs that ended in raw tool-call markup
(a provider template problem) sit on exactly those two tasks. Drop the broken runs from both arms and completion goes
86% to 88%. The traces can't say whether the fixes avoided the breakage or the breakage was chance.</p>
<p>Separately, case-insensitive search went from 3.3 to 9.3 model calls with no completion gain. That one's a real regression.</p>
<p>What I'd do next: fix or pin the provider template, rerun the two credited tasks with more reps, and rewrite the
zero-match probe text. Then read completion again.</p>"""
        note = "Nous Research, hermes-toolperf-evals, results/2026-08-06_rerun: 54 runs per model, NeMo Relay ATOF traces."
        out.append(dataset(key, label, "real", note, b, c, cmp, redact=True, pm_html=pm, method_html=METHOD, sensitivity=sens))
    return out


def harness():
    b = load_path(ROOT / "runs/harness/baseline", arm="baseline")
    c = load_path(ROOT / "runs/harness/candidate", arm="candidate")
    price = Price(0.5, 1.5)
    cmp = compare(b, c, Policy(target="wall_s"), price)
    pm = """<h4>What a team would change</h4>
<p>The baseline's top three fixes are all about how tools fail: identical retries, bad arguments, and giving up after an error.
None of them need a better model.</p>
<p>The candidate config changes only the tool layer: enums and examples in schemas, errors that name the next valid call,
and paged results for long documents. Completion rises and every guardrail improves, so the rule says ship.</p>
<p>Paging is not free: the changelog task needs one more call to reach the last page. The scorecard shows that cost
instead of hiding it in the average.</p>"""
    note = ("Simulated agent: it follows each task's reference plan and injects failure modes at fixed rates per config. "
            "It tests the pipeline, not a model. Cost uses an illustrative $0.50 / $1.50 per 1M tokens.")
    return dataset("harness", "20-task harness (simulated)", "simulated", note, b, c, cmp, pm_html=pm, method_html=METHOD)


def dogfood(path: Path):
    if not path.exists():
        return None
    runs = load_path(path)
    if not runs:
        return None
    s = arm_summary(runs)
    d = dataset("dogfood", "We scored the agent that built this", "dogfood", runs[0].source and
                "One real agent session: the build of this repo. Content stripped; names, timing, tokens and cost kept.",
                runs, [], None, redact=True, summary=s, method_html=(path.parent / "method.html").read_text()
                if (path.parent / "method.html").exists() else "",
                pm_html=(path.parent / "pm.html").read_text() if (path.parent / "pm.html").exists() else "")
    d["headline"] = (path.parent / "headline.txt").read_text().strip() if (path.parent / "headline.txt").exists() else ""
    d["sensitivity"] = [l for l in ((path.parent / "facts.txt").read_text().splitlines()
                                    if (path.parent / "facts.txt").exists() else []) if l.strip()]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hermes", default=str(ROOT / "data/hermes/results"))
    ap.add_argument("--out", default=str(ROOT / "docs/index.html"))
    a = ap.parse_args()
    sets = []
    if Path(a.hermes).exists():
        sets += hermes(Path(a.hermes))[::-1]          # Qwen first: it's the interesting one
    else:
        print("Hermes traces not found; run scripts/fetch_hermes.sh. Building without them.")
    sets.append(harness())
    df = dogfood(ROOT / "examples/dogfood/session.otel.json")
    if df:
        sets.append(df)
    lede = ("tracecard reads the traces your agent already emits (NeMo Relay ATOF, OpenTelemetry/OpenInference, "
            "NeMo Agent Toolkit profiler output), checks completion with a verifier, and makes the call: ship, change, "
            "rerun or stop. Then it ranks the three fixes most likely to pay off and shows where each run went wrong.")
    write_html(a.out, sets, title="tracecard", lede=lede, credit=CREDIT, repo=REPO)
    print("wrote", a.out, "with", len(sets), "datasets")


if __name__ == "__main__":
    main()
