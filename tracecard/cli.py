"""tracecard command line.

  tracecard score PATH                         one arm: scorecard + top fixes
  tracecard compare BASELINE CANDIDATE         two arms: decision + per-task view
Options shared by both: --verifier file.py:fn, --price-in/--price-out (USD per 1M tokens),
--md FILE, --json FILE, --html FILE.
"""
from __future__ import annotations

import argparse
import json
import sys

from .adapters import load_path, load_verifier
from .compare import Policy, compare
from .fixes import diagnose
from .metrics import Price, arm_summary
from .report import comparison_json, comparison_md, scorecard_md
from .viewer import dataset, write_html


def _common(p):
    p.add_argument("--verifier", help="path/to/verifier.py:function -> f(run, meta) -> bool | None")
    p.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    p.add_argument("--md"), p.add_argument("--json"), p.add_argument("--html")
    p.add_argument("--redact", action="store_true", help="drop argument values and tool results from the HTML")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracecard", description="Agent traces in, a decision out.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score one set of runs")
    s.add_argument("path")
    _common(s)
    c = sub.add_parser("compare", help="baseline vs candidate decision")
    c.add_argument("baseline"), c.add_argument("candidate")
    c.add_argument("--target", default="wall_s", help="metric the change was meant to improve (default wall_s)")
    c.add_argument("--margin", type=float, default=0.05, help="largest completion drop still called 'kept'")
    c.add_argument("--budget-wall", type=float, default=0.25, help="max wall-time increase, fraction")
    c.add_argument("--budget-cost", type=float, default=0.25, help="max cost/tokens/tool-data increase, fraction")
    _common(c)
    a = ap.parse_args(argv)
    v = load_verifier(a.verifier) if a.verifier else None
    price = Price(a.price_in, a.price_out)

    if a.cmd == "score":
        runs = load_path(a.path, verifier=v)
        if not runs:
            sys.exit(f"no traces found under {a.path}")
        summ = arm_summary(runs, price)
        md = scorecard_md(f"Scorecard: {a.path}", summ, diagnose(runs))
        print(md)
        if a.md:
            open(a.md, "w").write(md)
        if a.json:
            json.dump({k: v for k, v in summ.items()} | {"findings": [f.to_dict() for f in diagnose(runs)]},
                      open(a.json, "w"), indent=1, default=str)
        if a.html:
            write_html(a.html, [dataset("score", a.path, "real", a.path, runs, [], None,
                                        redact=a.redact, summary=summ)])
        return
    b = load_path(a.baseline, arm="baseline", verifier=v)
    k = load_path(a.candidate, arm="candidate", verifier=v)
    if not b or not k:
        sys.exit("no traces found for one of the arms")
    res = compare(b, k, Policy(target=a.target, margin=a.margin, max_wall_increase=a.budget_wall,
                               max_cost_increase=a.budget_cost), price)
    md = comparison_md(f"{a.baseline} vs {a.candidate}", res)
    print(md)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump(comparison_json(res), open(a.json, "w"), indent=1, default=str)
    if a.html:
        write_html(a.html, [dataset("compare", "baseline vs candidate", "real", f"{a.baseline} vs {a.candidate}",
                                    b, k, res, redact=a.redact)])


if __name__ == "__main__":
    main()
