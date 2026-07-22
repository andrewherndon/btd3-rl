# Running on an HPC (SLURM)

Parallelize by running **many trainings at once** (a hyperparameter sweep), one
per node — not one faster run (each run is single-core; the vec-env is IPC-bound).

## One-time install (login node, shared home)
```bash
bash scripts/install.sh        # miniforge + conda env "btd" + deps (RHEL-safe)
```

## Quick check first (per-node perf + sanity)
```bash
bash scripts/bench.sh          # submits, waits, prints every node's result to the terminal
```
Runs a 30k-step benchmark on every node and reports, per node: python/torch
versions, `env OK` (builds + steps end-to-end), and `RESULT: N steps/s PASS`.
Surfaces a broken node (missing dep / wrong Python) immediately, before you
commit to real runs. (An "env step" is one agent *decision*, not a round — some
steps fast-forward a whole round, so steps/s varies with how well the agent plays.)

## Submit the sweep (one job per config)
```bash
sbatch agent/sweep.sbatch
# if you changed the grid size, match the array range:
sbatch --array=0-$(($(python agent/sweep_configs.py --count)-1)) agent/sweep.sbatch
```

## Watch it
```bash
conda activate btd
python agent/sweep_dashboard.py        # live table (status/steps/reward); --once = snapshot
```

## Collect results
```bash
for d in agent/models/sweep_*/; do
  echo "$d"; python agent/evaluate.py --model ${d}best_model --difficulty hard --episodes 20 \
    | grep -E "win rate|round reached"
done
```

## Editing the sweep
- **Grid** — edit the lists at the top of `agent/sweep_configs.py`
  (`GAMMA / ENT_COEF / LR / SEED`); every combination is one run.
  Check the count: `python agent/sweep_configs.py --count`.
- **Steps per run** — `--timesteps` in `agent/sweep.sbatch` (default **5M**,
  ~3–7 h/run). Use `1000000` for a quick pass, `10000000` for a deep one.
- **Concurrency / RAM** — `--array=…%N` caps how many run at once; `--mem` is
  per-job RAM (~2 fit per 3.5 GB node at 1500M). Check nodes: `sinfo -o "%n %c %m"`.
- **Freeplay sweep** — add `--freeplay` to the `python agent/train.py` line in
  `sweep.sbatch` to chase past round 50.

## Notes
- SLURM auto-queues configs that don't fit and backfills as jobs finish, so a
  sweep bigger than the cluster just runs in a couple of waves.
- Headless: no `watch.py` there. Copy a `best_model.zip` back to a desktop to watch.
