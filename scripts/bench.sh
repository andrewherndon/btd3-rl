#!/usr/bin/env bash
# Submit the per-node benchmark, wait for it to finish, and print every node's
# result to the terminal. Run from anywhere:  bash scripts/bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

echo ">> submitting benchmark (blocks until all nodes finish, ~1-2 min)..."
jobid=$(sbatch --wait --parsable agent/bench.sbatch)

echo
echo "=== per-node benchmark results (job $jobid) ==="
log="logs/bench_${jobid}.out"
if grep -qE "Traceback|Error|FAIL" "$log"; then
  echo "!! problems found — full log below:"
  cat "$log"
else
  # one line per node: python/versions + "env OK" + "RESULT: N steps/s PASS"
  grep -E "python |torch |env OK|RESULT" "$log" | sort
fi
