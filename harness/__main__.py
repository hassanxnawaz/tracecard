"""Run the 20-task battery.

Offline (no keys):   python -m harness --config baseline --out runs/harness/baseline
Live (any OpenAI-compatible endpoint, e.g. NVIDIA NIM):
  NVIDIA_API_KEY=... python -m harness --config candidate --out runs/live/candidate \\
      --endpoint https://integrate.api.nvidia.com/v1 --model <model-id> --api-key-env NVIDIA_API_KEY
"""
import argparse
import os

from .agent import OpenAICompatClient, run_battery


def main():
    ap = argparse.ArgumentParser(prog="python -m harness")
    ap.add_argument("--config", choices=["baseline", "candidate"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL. Omit to use the offline simulator.")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Name of the env var holding the key (never pass the key itself).")
    a = ap.parse_args()
    factory = None
    if a.endpoint:
        key = os.environ.get(a.api_key_env)
        factory = lambda: OpenAICompatClient(a.endpoint, a.model, key)
    res = run_battery(a.config, a.out, reps=a.reps, client_factory=factory)
    mode = "live" if a.endpoint else "simulated"
    print(f"{res['config']} ({mode}): {res['passed']}/{res['runs']} verified. Traces in {a.out}")


if __name__ == "__main__":
    main()
