"""Convert .npy path data to Rust-native .bin format.

Usage:
    python convert_paths.py

Reads from ../sim/paths/*.npy, writes to paths/*.bin.
"""

import struct
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "sim" / "paths"
DST = Path(__file__).resolve().parent / "paths"


def convert_npy(name: str) -> None:
    src = SRC / f"{name}.npy"
    if not src.exists():
        print(f"  SKIP {name} (not found)")
        return
    data: np.ndarray = np.load(src)  # shape (N, 2), float64
    dst = DST / f"{name}.bin"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(struct.pack("<I", len(data)))  # little-endian u32 count
        f.write(data.astype(np.float64).tobytes())
    print(f"  OK  {name}: {len(data)} points -> {dst}")


def main() -> None:
    print("Converting path data...")
    convert_npy("track_3")
    convert_npy("boomerang_arc")
    print("Done.")


if __name__ == "__main__":
    main()
