"""
Extract per-frame bloon (x, y) coordinates from a JPEXS-exported SWF XML.

The bloon's path on each track is baked into the timeline of a DefineSprite
MovieClip (one per track-rank-branch). We only need rank-1 for each track since
all ranks share the same path. We walk PlaceObject2/3 + ShowFrame tags and
record the inner object's translate matrix at each ShowFrame.

Output: paths/track_<n>.npy of shape (frame_count, 2) in pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TWIPS_PER_PIXEL = 20.0

# Per-track stage offset applied to the bloon MovieClip in BloonsTD.NewBloon
# (the _loc11_, _loc12_ values). Path coordinates inside each MovieClip's
# timeline are in *local* space; on-stage position = local + offset.
# Branched tracks (4, 6, 8) have one offset per branch.
TRACK_OFFSETS: dict[int, tuple[float, float] | dict[int, tuple[float, float]]] = {
    1: (-54.0, 14.0),
    2: (-48.0, -133.0),
    3: (47.0, -174.0),
    4: {1: (337.0, -164.0), 2: (82.0, -178.0)},
    5: (-35.0, -15.0),
    6: {1: (140.0, -156.0), 2: (-65.0, 135.0), 3: (-85.0, -135.0)},
    7: (240.0, 378.0),  # reuses track-4 MovieClip geometry, different stage offset
    8: {1: (250.0, -175.0), 2: (-66.0, -66.0), 3: (-72.0, 212.0)},
}


def get_track_offset(track: int, branch: int = 1) -> tuple[float, float]:
    entry = TRACK_OFFSETS[track]
    if isinstance(entry, dict):
        return entry[branch]
    return entry


def load_symbol_map(symbols_csv: Path) -> dict[str, int]:
    """Returns {linkage_name: symbol_id}."""
    mapping: dict[str, int] = {}
    with symbols_csv.open() as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        for row in reader:
            if len(row) != 2:
                continue
            sid, name = row
            mapping[name] = int(sid)
    return mapping


def resolve_track_sprite_id(
    track: int, branch: int, symbol_map: dict[str, int]
) -> tuple[int, str]:
    """Pick the rank-1 bloon sprite for a given track (+ branch). Track 7
    reuses track-4 MovieClips, so we resolve to Bloon_1_4 there."""
    if track == 7:
        name = "Bloon_1_4"
    elif isinstance(TRACK_OFFSETS[track], dict):
        name = f"Bloon_1_{track}_{branch}"
    else:
        name = f"Bloon_1_{track}"
    if name not in symbol_map:
        raise KeyError(f"No symbol for {name} in symbols.csv")
    return symbol_map[name], name


def extract_path(xml_path: Path, sprite_id: int, depth_filter: int | None = 1) -> np.ndarray:
    """Stream the XML, find DefineSpriteTag with sprite_id, walk its subTags,
    and emit one (tx, ty) row per ShowFrameTag, in twips.

    depth_filter: only track placements at this Flash depth (bloons are at depth 1).
    Set to None to accept any depth.
    """
    sprite_id_str = str(sprite_id)
    in_target = False
    cur_x: float | None = None
    cur_y: float | None = None
    frames: list[tuple[float, float]] = []

    # Only clear top-level <item> elements; clearing inner <matrix> children
    # would zero their attributes before the PlaceObject end-event reads them
    # (ElementTree.Element.clear() removes attributes too).
    for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
        if elem.tag != "item":
            continue
        t = elem.get("type")

        if event == "start":
            if t == "DefineSpriteTag" and elem.get("spriteId") == sprite_id_str:
                in_target = True
            continue

        # event == "end"
        if in_target:
            if t in ("PlaceObject2Tag", "PlaceObject3Tag"):
                if depth_filter is None or elem.get("depth") == str(depth_filter):
                    m = elem.find("matrix")
                    if m is not None:
                        tx = m.get("translateX")
                        ty = m.get("translateY")
                        if tx is not None:
                            cur_x = float(tx)
                        if ty is not None:
                            cur_y = float(ty)
            elif t == "ShowFrameTag":
                if cur_x is None or cur_y is None:
                    raise RuntimeError(
                        "ShowFrameTag encountered before any placement set a position."
                    )
                frames.append((cur_x, cur_y))
            elif t == "DefineSpriteTag" and elem.get("spriteId") == sprite_id_str:
                in_target = False
                elem.clear()
                break

        elem.clear()

    if not frames:
        raise RuntimeError(f"No frames extracted for sprite {sprite_id}.")

    arr = np.array(frames, dtype=np.float64) / TWIPS_PER_PIXEL
    return arr


def peek_frame_count(xml_path: Path, sprite_id: int) -> int | None:
    """Cheap pre-scan: find the DefineSpriteTag opening line and read frameCount."""
    pattern = re.compile(
        rf'DefineSpriteTag"[^>]*frameCount="(\d+)"[^>]*spriteId="{sprite_id}"'
    )
    with xml_path.open() as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", type=Path, required=True, help="JPEXS-exported SWF XML")
    ap.add_argument("--symbols", type=Path, required=True, help="symbols.csv path")
    ap.add_argument("--track", type=int, required=True, help="Track number (1-8)")
    ap.add_argument("--branch", type=int, default=1, help="Branch (1-3) for tracks 4,6,8")
    ap.add_argument("--out-dir", type=Path, default=Path("paths"))
    ap.add_argument(
        "--local-coords",
        action="store_true",
        help="Save in MovieClip-local coords (skip stage offset).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    symbol_map = load_symbol_map(args.symbols)
    sprite_id, name = resolve_track_sprite_id(args.track, args.branch, symbol_map)
    expected_frames = peek_frame_count(args.xml, sprite_id)
    offset = get_track_offset(args.track, args.branch)
    print(
        f"track {args.track} (branch {args.branch}) -> {name} "
        f"(sprite {sprite_id}, frameCount={expected_frames}, offset={offset})"
    )

    print("streaming XML...")
    path = extract_path(args.xml, sprite_id)
    print(f"extracted {len(path)} frames")

    if not args.local_coords:
        path = path + np.array(offset)

    if expected_frames is not None and len(path) != expected_frames:
        print(
            f"WARNING: frameCount mismatch: header says {expected_frames}, "
            f"got {len(path)}"
        )

    out_npy = args.out_dir / f"track_{args.track}.npy"
    out_json = args.out_dir / f"track_{args.track}.json"
    np.save(out_npy, path)
    with out_json.open("w") as f:
        json.dump(
            {
                "track": args.track,
                "sprite_id": sprite_id,
                "linkage_name": name,
                "frame_count": len(path),
                "units": "pixels",
                "stage_size_px": [640, 480],
                "frames": path.round(3).tolist(),
            },
            f,
            indent=None,
        )

    print(
        f"x range: [{path[:, 0].min():.1f}, {path[:, 0].max():.1f}]\n"
        f"y range: [{path[:, 1].min():.1f}, {path[:, 1].max():.1f}]\n"
        f"start:   ({path[0, 0]:.1f}, {path[0, 1]:.1f})\n"
        f"end:     ({path[-1, 0]:.1f}, {path[-1, 1]:.1f})\n"
        f"wrote:   {out_npy}\n         {out_json}"
    )


if __name__ == "__main__":
    main()
