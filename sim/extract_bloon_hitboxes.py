"""Extract per-rank bloon hitbox bounds from the JPEXS SWF XML.

The body of each rank is a DefineSprite placed as `name="inner"` inside the
per-track Bloon MovieClip. Its on-screen AABB (what AS hitTestObject uses for
collision) is the union of its descendants' bounds, with each contribution
transformed by the chain of PlaceObject matrices. We compute this AABB in the
inner's *local* coordinate space so it can be applied as a fixed offset around
the bloon's stage position (`bloon.x, bloon.y` in the sim).

Output: paths/bloon_hitboxes.json — for each rank:
  {
    "rank": int,
    "name": "red" | ...,
    "half_w": float, "half_h": float,      # pixels (twips / 20)
    "cx": float, "cy": float,              # AABB center offset from inner origin
    "radius": float,                       # max distance origin -> AABB corner (px)
  }
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

TWIPS_PER_PX = 20.0

# Rank -> canonical name. Matches NOTES.md.
RANK_NAMES = {
    1: "red", 2: "blue", 3: "green", 4: "yellow",
    5: "black", 6: "white", 7: "lead", 8: "rainbow",
    9: "ceramic", 10: "MOAB",
}


def load_symbol_map(path: Path) -> dict[str, int]:
    m: dict[str, int] = {}
    with path.open() as f:
        for row in csv.reader(f, delimiter=";", quotechar='"'):
            if len(row) == 2:
                m[row[1]] = int(row[0])
    return m


def parse_matrix(m_elem) -> dict[str, float]:
    """SWF 2D affine matrix:
      x' = a*x + c*y + tx
      y' = b*x + d*y + ty
    XML attribute aliases: scaleX=a, scaleY=d, rotateSkew0=c, rotateSkew1=b."""
    if m_elem is None:
        return {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "tx": 0.0, "ty": 0.0}
    has_scale = m_elem.get("hasScale") == "true"
    has_rotate = m_elem.get("hasRotate") == "true"
    return {
        "a": float(m_elem.get("scaleX", 1.0)) if has_scale else 1.0,
        "d": float(m_elem.get("scaleY", 1.0)) if has_scale else 1.0,
        "c": float(m_elem.get("rotateSkew0", 0.0)) if has_rotate else 0.0,
        "b": float(m_elem.get("rotateSkew1", 0.0)) if has_rotate else 0.0,
        "tx": float(m_elem.get("translateX", 0.0)),
        "ty": float(m_elem.get("translateY", 0.0)),
    }


def identity():
    return {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "tx": 0.0, "ty": 0.0}


def compose(A, B):
    # Returns A * B (apply B first, then A — the usual nested transform order).
    return {
        "a": A["a"] * B["a"] + A["c"] * B["b"],
        "b": A["b"] * B["a"] + A["d"] * B["b"],
        "c": A["a"] * B["c"] + A["c"] * B["d"],
        "d": A["b"] * B["c"] + A["d"] * B["d"],
        "tx": A["a"] * B["tx"] + A["c"] * B["ty"] + A["tx"],
        "ty": A["b"] * B["tx"] + A["d"] * B["ty"] + A["ty"],
    }


def transform_pt(M, x, y):
    return (M["a"] * x + M["c"] * y + M["tx"],
            M["b"] * x + M["d"] * y + M["ty"])


def transform_bbox(bbox, M):
    xmin, ymin, xmax, ymax = bbox
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    pts = [transform_pt(M, x, y) for x, y in corners]
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def build_index(xml_path: Path) -> dict:
    """One streaming pass over the SWF XML. For each character ID we store:
      ('shape', (xmin, ymin, xmax, ymax))  — from DefineShape*Tag's shapeBounds
      ('sprite', [(child_id, name, matrix), ...])  — first-frame post-state of
        each depth slot, as if the sprite was just at the end of frame 1.
    We also track simple PlaceObject moves and RemoveObject2 within frame 1."""
    index: dict[int, tuple[str, object]] = {}
    in_sprite_id: int | None = None
    sprite_first_frame_done = False
    # depth -> {char_id, name, matrix, depth}
    placements: dict[int, dict] = {}
    SHAPE_TAGS = {"DefineShapeTag", "DefineShape2Tag", "DefineShape3Tag", "DefineShape4Tag"}

    for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
        if elem.tag != "item":
            continue
        t = elem.get("type")

        if event == "start":
            if t == "DefineSpriteTag":
                in_sprite_id = int(elem.get("spriteId"))
                sprite_first_frame_done = False
                placements = {}
            continue

        # event == "end"
        if t == "DefineSpriteTag":
            ordered = sorted(placements.values(), key=lambda p: p["depth"])
            index[in_sprite_id] = (
                "sprite",
                [(p["char"], p["name"], p["matrix"]) for p in ordered],
            )
            in_sprite_id = None
            placements = {}
            elem.clear()
            continue

        if t in SHAPE_TAGS:
            sid = int(elem.get("shapeId"))
            sb = elem.find("shapeBounds")
            if sb is not None:
                index[sid] = (
                    "shape",
                    (
                        float(sb.get("Xmin")), float(sb.get("Ymin")),
                        float(sb.get("Xmax")), float(sb.get("Ymax")),
                    ),
                )
            elem.clear()
            continue

        if in_sprite_id is not None:
            if t == "ShowFrameTag":
                sprite_first_frame_done = True
            elif not sprite_first_frame_done and t in ("PlaceObject2Tag", "PlaceObject3Tag"):
                depth = int(elem.get("depth"))
                has_char = elem.get("placeFlagHasCharacter") == "true"
                is_move = elem.get("placeFlagMove") == "true"
                has_matrix = elem.get("placeFlagHasMatrix") == "true"
                name = elem.get("name") or ""
                matrix = parse_matrix(elem.find("matrix")) if has_matrix else identity()
                if has_char:
                    placements[depth] = {
                        "char": int(elem.get("characterId")),
                        "name": name,
                        "matrix": matrix,
                        "depth": depth,
                    }
                elif is_move and depth in placements:
                    if has_matrix:
                        placements[depth]["matrix"] = matrix
                    if name:
                        placements[depth]["name"] = name
            elif not sprite_first_frame_done and t == "RemoveObject2Tag":
                placements.pop(int(elem.get("depth")), None)
        elem.clear()

    return index


def compute_bbox(char_id: int, index: dict, acc=None) -> tuple | None:
    """Return AABB of character `char_id` in its own local frame, after
    composing all descendant transforms. None if unknown character."""
    if acc is None:
        acc = identity()
    entry = index.get(char_id)
    if entry is None:
        return None
    kind, data = entry
    if kind == "shape":
        return transform_bbox(data, acc)
    bbox = None
    for child_id, _name, mat in data:
        bbox = union_bbox(bbox, compute_bbox(child_id, index, compose(acc, mat)))
    return bbox


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--symbols", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("paths/bloon_hitboxes.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    symbols = load_symbol_map(args.symbols)
    print(f"streaming index from {args.xml.name} ...")
    index = build_index(args.xml)
    print(f"indexed {len(index)} characters")

    results: dict[int, dict] = {}
    for rank in range(1, 11):
        name = f"Bloon_{rank}_1"
        if name not in symbols:
            print(f"  rank {rank}: no {name} in symbols.csv — skipping")
            continue
        bloon_sid = symbols[name]
        entry = index.get(bloon_sid)
        if entry is None or entry[0] != "sprite":
            print(f"  rank {rank}: {name} sprite {bloon_sid} not in index — skipping")
            continue
        inner = next(((c, m) for c, n, m in entry[1] if n == "inner"), None)
        if inner is None:
            print(f"  rank {rank}: no 'inner' placement — skipping")
            continue
        inner_id, _ = inner
        bbox_twips = compute_bbox(inner_id, index)
        if bbox_twips is None:
            print(f"  rank {rank}: inner {inner_id} bbox could not be computed")
            continue
        xmin, ymin, xmax, ymax = (b / TWIPS_PER_PX for b in bbox_twips)
        half_w = (xmax - xmin) / 2.0
        half_h = (ymax - ymin) / 2.0
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        # Conservative circle: farthest corner from inner origin (0, 0).
        radius = max(
            (abs(xmin) ** 2 + abs(ymin) ** 2) ** 0.5,
            (abs(xmax) ** 2 + abs(ymin) ** 2) ** 0.5,
            (abs(xmax) ** 2 + abs(ymax) ** 2) ** 0.5,
            (abs(xmin) ** 2 + abs(ymax) ** 2) ** 0.5,
        )
        results[rank] = {
            "rank": rank,
            "name": RANK_NAMES.get(rank, f"r{rank}"),
            "inner_sprite_id": inner_id,
            "bloon_sprite_id": bloon_sid,
            "bbox_px": [round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3)],
            "half_w": round(half_w, 3),
            "half_h": round(half_h, 3),
            "cx": round(cx, 3),
            "cy": round(cy, 3),
            "radius": round(radius, 3),
        }
        print(
            f"  rank {rank:2d} {RANK_NAMES.get(rank, ''):>8}: "
            f"box {2*half_w:.1f} x {2*half_h:.1f} px, "
            f"center ({cx:+.1f}, {cy:+.1f}), radius {radius:.1f}"
        )

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
