# Running on the SLURM cluster

**Cluster shape:** an ARM Pi head node (NFS server + SLURM controller + login,
orchestrator only) and **3 x86_64 compute nodes** (`node01-03`, 4 cores / 3.5 GB,
Rocky Linux 9). `/clusterfs` is shared NFS (writable by the `hpc` user, visible on
every node) — the repo and the Python env live there. Do everything as `hpc`.

Parallelize by running **many trainings at once** (a sweep), one per node — not
one faster run (each run is single-core; ~6 concurrent across the 3 nodes).

## One-time install
```bash
sudo -iu hpc git clone https://github.com/andrewherndon/btd3-rl.git /clusterfs/btd3-rl
sudo -iu hpc bash /clusterfs/btd3-rl/scripts/install.sh
```
Installs an x86_64 miniforge env + deps into `/clusterfs/miniforge3` (the installer
runs on a compute node so the binaries are x86_64).

## Quick check first (per-node perf + sanity)
```bash
sudo -iu hpc bash /clusterfs/btd3-rl/scripts/bench.sh   # submits, waits, prints each node's result
```
Reports per node: python/torch versions, `env OK`, and `RESULT: N steps/s PASS`.

## Submit the sweep (one job per config)
```bash
sudo -iu hpc sbatch /clusterfs/btd3-rl/agent/sweep.sbatch
```

## Watch it (runs on the Pi; the dashboard needs no torch)
```bash
sudo -iu hpc python3 /clusterfs/btd3-rl/agent/sweep_dashboard.py   # live table; --once = snapshot
```

## Collect results
```bash
sudo -iu hpc bash -c 'source /clusterfs/miniforge3/etc/profile.d/conda.sh; conda activate btd
for d in /clusterfs/btd3-rl/agent/models/sweep_*/; do echo "$d"
  srun -N1 python /clusterfs/btd3-rl/agent/evaluate.py --model ${d}best_model --difficulty hard --episodes 20 \
    | grep -E "win rate|round reached"; done'
```

## Editing the sweep
- **Grid** — `GAMMA/ENT_COEF/LR/SEED` lists at the top of `agent/sweep_configs.py`;
  every combination is one run. Count: `python agent/sweep_configs.py --count`.
- **Steps per run** — `--timesteps` in `agent/sweep.sbatch` (default 5M).
- **Concurrency / RAM** — `--array=…%N` caps concurrent tasks; `--mem` is per-job RAM
  (~2 fit per 3.5 GB node at 1500M). Nodes: `sinfo -o "%n %c %m"`.
- **Freeplay sweep** — add `--freeplay` to the `train.py` line in `sweep.sbatch`.

## Notes
- SLURM auto-queues configs that don't fit and backfills as jobs finish.
- Headless: no `watch.py` on the cluster — copy a `best_model.zip` off `/clusterfs`
  to a desktop to watch.
- The 2015-era i5 cores are slower per-core than a modern laptop; the cluster's
  value is *parallel* runs, not single-run speed.
