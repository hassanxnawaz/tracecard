"""Normalized run and step records. Every adapter converts its input into these."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Step:
    kind: str                      # "llm" or "tool"
    name: str                      # tool name, or model/provider for llm steps
    start: float                   # seconds since epoch (or since run start)
    end: float
    args: Any = None               # tool arguments as parsed JSON when available
    error: str | None = None       # short error text when the step failed
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None   # provider-reported cost of this call, when present
    result_bytes: int = 0          # size of the tool result fed back to the model
    result_preview: str = ""       # first characters of the result (empty when redacted)
    span_id: str = ""
    parent_id: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Run:
    run_id: str
    task: str
    arm: str = "default"
    model: str = ""
    rep: int = 0
    success: bool | None = None    # None means no verifier result was available
    steps: list[Step] = field(default_factory=list)
    wall_s: float | None = None
    output: str = ""
    source: str = ""
    expected_tools: list[str] | None = None

    @property
    def llm_steps(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "llm"]

    @property
    def tool_steps(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def duration(self) -> float:
        if self.wall_s is not None:
            return self.wall_s
        if not self.steps:
            return 0.0
        return max(s.end for s in self.steps) - min(s.start for s in self.steps)

    def to_dict(self, redact: bool = False) -> dict:
        d = asdict(self)
        t0 = min((s.start for s in self.steps), default=0.0)
        for s in d["steps"]:
            s["start"] = round(s["start"] - t0, 4)
            s["end"] = round(s["end"] - t0, 4)
            if redact:
                s["args"] = _shape(s["args"])
                s["result_preview"] = ""
                if s["error"]:
                    s["error"] = "error"
        if redact:
            d["output"] = ""
        return d


def comparable(args: Any) -> bool:
    """False when argument values were redacted, so identical-call checks can't be trusted."""
    if isinstance(args, dict) and args and all(v in ("…", "...") for v in args.values()):
        return False
    return args is not None


def _shape(args: Any) -> Any:
    """Keep argument keys, drop values. Used when traces may hold private data."""
    if isinstance(args, dict):
        return {k: "…" for k in args}
    return None
