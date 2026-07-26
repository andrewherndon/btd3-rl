#!/usr/bin/env bash
# Submit the per-node benchmark, wait for it to finish, and print every node's
# result to the terminal. Run from anywhere:  bash scripts/bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# Optional: --backend rust (default python). Picks which sim bench.py measures.
backend=python
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) backend="$2"; shift 2 ;;
    --backend=*) backend="${1#*=}"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo ">> submitting benchmark (backend=$backend, blocks until all nodes finish, ~1-2 min)..."
jobid=$(sbatch --wait --parsable --export=ALL,BACKEND="$backend" agent/bench.sbatch)

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
