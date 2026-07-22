"""Live CLI dashboard for a running SLURM sweep. Reads each task's log +
squeue state and prints a refreshing table.

    python agent/sweep_dashboard.py            # live, refresh every 5s
    python agent/sweep_dashboard.py --once     # print once and exit
    python agent/sweep_dashboard.py --interval 10

Pure stdlib; safe to run over SSH on the login node.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_configs as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODELS = ROOT / "agent" / "models"

_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def _last(pattern: str, text: str):
    m = re.findall(pattern, text)
    return m[-1] if m else None


def parse_log(idx: int) -> dict:
    """Latest steps / train-reward / eval-reward from task idx's log file."""
    files = sorted(glob.glob(str(LOG_DIR / f"sweep_*_{idx}.out")), key=os.path.getmtime)
    if not files:
        return {}
    text = Path(files[-1]).read_text(errors="ignore")
    steps = _last(r"total_timesteps\s*\|\s*(\d+)", text)
    return {
        "steps": int(steps) if steps else None,
        "ep_rew": _last(r"ep_rew_mean\s*\|\s*" + _NUM, text),
        "eval": _last(r"mean_reward\s*\|\s*" + _NUM, text),
        "mtime": os.path.getmtime(files[-1]),
    }


def squeue_states() -> dict[int, str]:
    """{array_task_idx: 'run'|'pend'} from squeue, best-effort (empty if squeue absent)."""
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%i %T"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    states: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or "_" not in parts[0]:
            continue
        tid, state = parts[0].split("_", 1)[1], parts[1]
        st = "run" if state.startswith("R") else "pend"
        if tid.startswith("["):                         # pending range e.g. [4-23] or [4,6]
            for chunk in tid.strip("[]").split(","):
                if "-" in chunk:
                    a, b = chunk.split("-"); rng = range(int(a), int(b) + 1)
                else:
                    rng = [int(chunk)] if chunk.isdigit() else []
                for i in rng:
                    states[i] = "pend"
        elif tid.isdigit():
            states[int(tid)] = st
    return states


def status(idx: int, sq: dict[int, str], has_model: bool) -> str:
    if idx in sq:
        return sq[idx].upper()
    if has_model:
        return "DONE"
    return "-"


def render(once: bool) -> str:
    sq = squeue_states()
    rows = []
    counts = {"RUN": 0, "PEND": 0, "DONE": 0, "-": 0}
    for i in range(len(sc.CONFIGS)):
        c = sc.config(i)
        log = parse_log(i)
        has_model = (MODELS / f"sweep_{i:03d}" / "best_model.zip").exists()
        st = status(i, sq, has_model)
        counts[st if st in counts else "-"] = counts.get(st if st in counts else "-", 0) + 1
        steps = f"{log['steps']/1e6:.1f}M" if log.get("steps") else "-"
        rows.append((i, c["gamma"], c["ent_coef"], c["lr"], c["seed"], st,
                     steps, log.get("ep_rew") or "-", log.get("eval") or "-"))
    lines = [f"BTD sweep — {len(sc.CONFIGS)} configs      {time.strftime('%H:%M:%S')}", ""]
    lines.append(f"{'idx':>3} {'gamma':>6} {'ent':>6} {'lr':>7} {'seed':>4} "
                 f"{'status':>6} {'steps':>7} {'ep_rew':>8} {'eval':>8}")
    for r in rows:
        lines.append(f"{r[0]:>3} {r[1]:>6} {r[2]:>6} {r[3]:>7} {r[4]:>4} "
                     f"{r[5]:>6} {r[6]:>7} {str(r[7]):>8} {str(r[8]):>8}")
    lines.append("")
    lines.append(f"DONE {counts.get('DONE',0)}   RUN {counts.get('RUN',0)}   "
                 f"PEND {counts.get('PEND',0)}   idle {counts.get('-',0)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=5)
    args = p.parse_args()
    if args.once:
        print(render(True))
        return
    try:
        while True:
            print("\033[2J\033[H" + render(False), flush=True)   # clear + home
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
