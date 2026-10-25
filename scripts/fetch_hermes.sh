#!/usr/bin/env sh
# Download the public Hermes ToolPerf Aug 6, 2026 rerun traces (Nous Research) into data/hermes/.
# Not redistributed in this repo; fetched from the source.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/hermes
curl -sL -o data/hermes/atof-traces.tgz \
  https://raw.githubusercontent.com/NousResearch/hermes-toolperf-evals/main/results/2026-08-06_rerun/atof-traces.tgz
tar xzf data/hermes/atof-traces.tgz -C data/hermes
echo "Traces in data/hermes/results"
