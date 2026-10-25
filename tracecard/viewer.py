"""Self-contained HTML scorecard and trace viewer (no network, no build step)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .compare import Comparison
from .fixes import diagnose, first_anomaly, step_flags
from .model import Run
from .report import comparison_json

TEMPLATE = Path(__file__).with_name("templates") / "viewer.html"


def run_view(run: Run, redact: bool) -> dict:
    flags = step_flags(run)
    anomaly = first_anomaly(run)
    if anomaly is None and flags:
        anomaly = min(flags)
    d = run.to_dict(redact=redact)
    d["flags"] = {str(k): v for k, v in flags.items()}
    d["anomaly"] = anomaly if (run.success is False or flags) else None
    return d


def dataset(id: str, label: str, kind: str, source_note: str, baseline: list[Run], candidate: list[Run],
            comparison: Comparison | None, redact: bool = False, pm_html: str = "", method_html: str = "",
            sensitivity: list[str] | None = None, summary: dict | None = None) -> dict:
    return {
        "id": id, "label": label, "kind": kind, "source_note": source_note,
        "comparison": comparison_json(comparison) if comparison else None,
        "summary": {k: v for k, v in (summary or {}).items() if k != "rows"} or None,
        "findings_baseline": [f.to_dict() for f in diagnose(baseline)],
        "runs": {"baseline": [run_view(r, redact) for r in baseline],
                 "candidate": [run_view(r, redact) for r in candidate]},
        "pm_html": pm_html, "method_html": method_html, "sensitivity": sensitivity or [],
    }


def write_html(out: str | Path, datasets: list[dict], title: str = "tracecard", lede: str = "",
               credit: str = "", repo: str = "") -> Path:
    payload = {"meta": {"title": title, "lede": lede, "credit": credit, "repo": repo,
                        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
               "datasets": datasets}
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", data)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
