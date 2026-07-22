# Setup

Train the BTD3 RL agent on a fresh machine (Linux or macOS).

## Requirements
Python **3.11+** (torch needs it). Check with `python3 --version`.
On RHEL/HPC the system Python is often older — load or install a newer one, e.g.
`module load python/3.11`, or use conda / pyenv / uv.

## Install
```bash
git clone https://github.com/andrewherndon/btd3-rl.git
cd btd3-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[rl]"
```

## Train (headless)
```bash
python agent/train.py --timesteps 10000000 --n-envs 8 \
  --difficulties easy,medium,hard --eval-difficulty hard \
  --save-path agent/models/run1/model
```
`best_model.zip` is written to the save dir whenever eval improves.

## Evaluate
```bash
python agent/evaluate.py --model agent/models/run1/best_model --difficulty hard --episodes 30
python agent/trace.py   --model agent/models/run1/best_model --seed 0   # per-action log
```

## Notes
- **Rendering** (`watch.py`, `play.py`) needs a display — not available on a
  headless HPC. Train there, copy the model `.zip` back to a desktop to watch.
- **Many cores don't speed one run** (the vec-env is IPC-bound). Use them to run
  several experiments in parallel — one training process per core, different
  seeds/configs — rather than more `--n-envs` on a single run.
