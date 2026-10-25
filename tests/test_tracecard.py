import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.agent import OpenAICompatClient, run_battery, run_task  # noqa: E402
from tracecard.adapters import load_path, load_verifier  # noqa: E402
from tracecard.compare import Policy, compare  # noqa: E402
from tracecard.fixes import diagnose  # noqa: E402
from tracecard.metrics import Price, arm_summary, run_metrics, wilson  # noqa: E402
from tracecard.model import Run, Step  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def mk(task, ok, arm="a", rep=0, n_llm=2, err=False):
    steps = [Step("llm", "m", 0, 1, prompt_tokens=100, completion_tokens=10)]
    steps.append(Step("tool", "t", 1, 1.5, args={"x": 1}, error="boom" if err else None, result_bytes=100))
    steps += [Step("llm", "m", 2 + i, 3 + i, prompt_tokens=100, completion_tokens=10) for i in range(n_llm - 1)]
    return Run(run_id=f"{task}-{arm}-{rep}", task=task, arm=arm, rep=rep, success=ok, steps=steps)


class Adapters(unittest.TestCase):
    def test_atof(self):
        (r,) = load_path(FIX / "sample-r0.atof.jsonl")
        self.assertEqual(r.task, "sample")
        self.assertEqual(len(r.llm_steps), 2)
        self.assertEqual(len(r.tool_steps), 2)
        self.assertIsNotNone(r.tool_steps[0].error)       # exit_code 1 + error key
        self.assertIsNone(r.tool_steps[1].error)
        m = run_metrics(r)
        self.assertEqual((m["tool_errors"], m["retries"], m["repeat_calls"]), (1, 1, 1))
        self.assertEqual(m["tokens_in"], 220)
        self.assertEqual(r.output, "VALUE=42")

    def test_nat_profiler(self):
        (r,) = load_path(FIX / "all_requests_profiler_traces.json")
        self.assertEqual([s.kind for s in r.steps], ["llm", "tool", "llm"])
        self.assertEqual(r.tool_steps[0].args, {"q": "x"})
        self.assertTrue(r.tool_steps[0].error)
        self.assertEqual(run_metrics(r)["tokens_in"], 110)

    def test_otel_roundtrip_from_harness(self):
        with tempfile.TemporaryDirectory() as d:
            run_battery("baseline", Path(d), reps=1)
            runs = load_path(d)
            self.assertEqual(len(runs), 20)
            self.assertTrue(all(r.expected_tools for r in runs))
            self.assertTrue(all(r.success is not None for r in runs))
            self.assertTrue(any(s.error for r in runs for s in r.tool_steps))


class Metrics(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(19, 27)
        self.assertTrue(0.5 < lo < 0.7 < hi < 0.9)

    def test_cost_from_price(self):
        s = arm_summary([mk("a", True)], Price(1.0, 2.0))
        self.assertAlmostEqual(s["mean"]["cost_usd"], (200 / 1e6) * 1 + (20 / 1e6) * 2)

    def test_partial_tokens_are_not_summed(self):
        r = mk("a", True)
        r.steps[0].prompt_tokens = None
        self.assertIsNone(run_metrics(r)["tokens_in"])


class Decisions(unittest.TestCase):
    def arms(self, b_pass, c_pass, tasks=10, reps=3, c_llm=2):
        b = [mk(f"t{t}", (t * reps + r) < b_pass, "b", r) for t in range(tasks) for r in range(reps)]
        c = [mk(f"t{t}", (t * reps + r) < c_pass, "c", r, n_llm=c_llm) for t in range(tasks) for r in range(reps)]
        return b, c

    def test_ship_on_clear_gain(self):
        self.assertEqual(compare(*self.arms(12, 27)).verdict, "SHIP")

    def test_change_when_gain_costs_too_much(self):
        self.assertEqual(compare(*self.arms(12, 27, c_llm=6)).verdict, "CHANGE")

    def test_stop_on_regression(self):
        self.assertEqual(compare(*self.arms(27, 9)).verdict, "STOP")

    def test_stop_when_identical(self):
        c = compare(*self.arms(20, 20))
        self.assertEqual(c.verdict, "STOP")
        self.assertIn("No measurable effect", c.headline)

    def test_rerun_when_underpowered(self):
        b = [mk("a", r < 1, "b", r) for r in range(3)] + [mk("b", True, "b", r) for r in range(3)]
        c = [mk("a", r < 3, "c", r) for r in range(3)] + [mk("b", r < 2, "c", r) for r in range(3)]
        self.assertEqual(compare(b, c).verdict, "RERUN")

    def test_deterministic(self):
        b, c = self.arms(15, 21)
        self.assertEqual(compare(b, c).success["ci"], compare(b, c).success["ci"])


class Findings(unittest.TestCase):
    def test_give_up_and_retry(self):
        r = Run("x", "x", success=False, steps=[
            Step("llm", "m", 0, 1), Step("tool", "t", 1, 2, args={"a": 1}, error="invalid unit"),
            Step("tool", "t", 2, 3, args={"a": 1}, error="invalid unit")])
        ids = [f.id for f in diagnose([r])]
        self.assertIn("give_up", ids)
        self.assertIn("retry_loop", ids)
        self.assertIn("arg_error", ids)

    def test_redacted_args_are_not_called_identical(self):
        steps = [Step("tool", "bash", i, i + 1, args={"command": "…"}, error="exit 1" if i == 0 else None) for i in range(3)]
        r = Run("x", "x", success=True, steps=steps)
        self.assertNotIn("retry_loop", [f.id for f in diagnose([r])])
        self.assertEqual(run_metrics(r)["repeat_calls"], 0)

    def test_ranked_by_failed_runs(self):
        good = Run("g", "x", success=True, steps=[Step("tool", "t", 0, 1, result_bytes=20000)])
        bad = Run("b", "y", success=False, steps=[Step("tool", "t", 0, 1, error="missing required argument 'x'")])
        f = diagnose([good, bad])
        self.assertEqual(f[0].failed_touched, 1)


class _Mock(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        msg = ({"role": "assistant", "content": "It is 64 F in Paris."} if has_tool_result else
               {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                   "name": "get_weather", "arguments": json.dumps({"city": "Paris", "unit": "F"})}}]})
        out = json.dumps({"choices": [{"message": msg}], "usage": {"prompt_tokens": 300, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class LiveMode(unittest.TestCase):
    def test_openai_compatible_endpoint(self):
        srv = HTTPServer(("127.0.0.1", 0), _Mock)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            task = json.loads((ROOT / "harness" / "tasks.json").read_text())[0]
            client = OpenAICompatClient(f"http://127.0.0.1:{srv.server_port}/v1", "mock-model")
            doc, ok = run_task(task, "candidate", client, 0)
            self.assertTrue(ok)
            with tempfile.TemporaryDirectory() as d:
                Path(d, "r.otel.json").write_text(json.dumps(doc))
                (r,) = load_path(d)
                self.assertEqual(run_metrics(r)["tokens_in"], 600)
                self.assertEqual([s.name for s in r.tool_steps], ["get_weather"])
        finally:
            srv.shutdown()
            srv.server_close()


HERMES = ROOT / "data" / "hermes" / "results"


@unittest.skipUnless(HERMES.exists(), "run scripts/fetch_hermes.sh to enable")
class HermesReproduction(unittest.TestCase):
    """Matches Table 2 of NVIDIA's NeMo Relay post (Sep 30, 2026) from the raw public traces."""

    def test_table(self):
        v = load_verifier(str(ROOT / "examples/hermes_toolperf/verifier.py") + ":verify")
        want = {"anthropic_claude-sonnet-4.5": (24, 23, 2.9, 2.8), "qwen_qwen3-coder-30b-a3b-instruct": (19, 22, 3.8, 4.9)}
        for model, (b_ok, c_ok, b_llm, c_llm) in want.items():
            b = load_path(HERMES / model / "baseline", verifier=v)
            c = load_path(HERMES / model / "fixes", verifier=v)
            self.assertEqual(sum(r.success for r in b), b_ok)
            self.assertEqual(sum(r.success for r in c), c_ok)
            self.assertAlmostEqual(arm_summary(b)["mean"]["llm_calls"], b_llm, delta=0.05)
            self.assertAlmostEqual(arm_summary(c)["mean"]["llm_calls"], c_llm, delta=0.05)


if __name__ == "__main__":
    unittest.main()
