"""Round generation. Ports BloonsTD.BuildLevels() from BloonsTD.as.

Rounds 1-50 are a hardcoded list of `(count, round, rank)` calls — `ABSTL`
in the AS — that append batches of bloons to a round in the order they're
declared. Some rounds get batches appended multiple times across the call
list (e.g. round 6 gets a tail-end addition at index 23, round 26 gets two
late additions at indices 76-77). The call order IS the spawn order, so we
preserve it verbatim.

Rounds 51-149 are procedural: 7 + (round - 50) batches per round, each batch's
rank chosen by a random roll biased by difficulty. The RNG used here is the
sim's RNG, consumed at sim construction — meaning rounds 51+ are fixed for
a given seed + difficulty pair.
"""

from __future__ import annotations

import numpy as np

# (count, round, rank). Order is significant. Direct transcription of
# BloonsTD.as lines 1536-1691.
_HARDCODED_ABSTL: list[tuple[int, int, int]] = [
    (14, 1, 1),
    (30, 2, 1),
    (10, 3, 1),
    (4, 3, 2),
    (5, 3, 1),
    (4, 3, 2),
    (5, 4, 1),
    (12, 4, 2),
    (5, 4, 1),
    (12, 4, 2),
    (10, 5, 1),
    (8, 5, 2),
    (12, 5, 1),
    (20, 5, 2),
    (13, 6, 1),
    (7, 6, 3),
    (50, 7, 2),
    (9, 8, 1),
    (16, 8, 2),
    (9, 8, 1),
    (7, 8, 2),
    (9, 8, 1),
    (7, 8, 2),
    (8, 6, 3),
    (20, 9, 2),
    (15, 9, 3),
    (12, 9, 2),
    (32, 10, 3),
    (12, 11, 3),
    (7, 11, 4),
    (1, 12, 8),
    (4, 11, 4),
    (18, 13, 2),
    (18, 13, 1),
    (30, 13, 3),
    (20, 13, 2),
    (1, 14, 8),
    (12, 14, 4),
    (8, 15, 4),
    (6, 15, 3),
    (8, 15, 4),
    (8, 15, 3),
    (5, 15, 4),
    (35, 16, 3),
    (15, 16, 4),
    (9, 16, 2),
    (7, 16, 4),
    (20, 17, 2),
    (55, 17, 3),
    (10, 17, 4),
    (30, 18, 2),
    (25, 18, 4),
    (28, 18, 3),
    (45, 19, 3),
    (25, 19, 4),
    (5, 20, 7),
    (17, 21, 4),
    (10, 21, 2),
    (27, 21, 4),
    (10, 21, 3),
    (30, 21, 3),
    (50, 22, 4),
    (30, 23, 4),
    (35, 23, 3),
    (30, 23, 4),
    (30, 24, 3),
    (45, 24, 4),
    (26, 24, 3),
    (20, 24, 2),
    (20, 25, 4),
    (15, 25, 5),
    (22, 25, 4),
    (80, 26, 4),
    (15, 26, 5),
    (35, 27, 5),
    (19, 28, 5),
    (16, 28, 6),
    (20, 26, 4),
    (14, 26, 7),
    (6, 29, 7),
    (12, 29, 5),
    (14, 29, 6),
    (60, 30, 4),
    (28, 30, 5),
    (2, 31, 9),
    (20, 32, 4),
    (16, 32, 6),
    (22, 32, 5),
    (60, 33, 5),
    (3, 33, 9),
    (25, 34, 5),
    (25, 34, 6),
    (50, 34, 4),
    (4, 34, 9),
    (12, 35, 8),
    (11, 36, 5),
    (12, 36, 4),
    (10, 36, 5),
    (10, 36, 7),
    (12, 36, 6),
    (9, 36, 5),
    (1, 37, 10),
    (1, 38, 9),
    (60, 38, 4),
    (50, 38, 5),
    (4, 38, 9),
    (50, 39, 4),
    (22, 39, 5),
    (22, 39, 6),
    (10, 39, 7),
    (9, 39, 8),
    (64, 40, 5),
    (5, 40, 9),
    (25, 39, 6),
    (18, 41, 6),
    (14, 41, 7),
    (16, 41, 8),
    (10, 42, 9),
    (100, 42, 4),
    (54, 42, 5),
    (23, 43, 8),
    (20, 43, 7),
    (5, 43, 9),
    (5, 44, 9),
    (130, 44, 5),
    (1, 44, 10),
    (12, 46, 8),
    (11, 45, 9),
    (90, 45, 6),
    (8, 46, 9),
    (38, 46, 7),
    (18, 46, 8),
    (20, 47, 5),
    (40, 47, 6),
    (6, 47, 9),
    (18, 47, 7),
    (15, 47, 8),
    (6, 47, 9),
    (25, 48, 8),
    (30, 48, 6),
    (30, 48, 5),
    (25, 48, 7),
    (12, 48, 8),
    (5, 49, 9),
    (34, 49, 8),
    (17, 49, 9),
    (8, 50, 9),
    (13, 50, 8),
    (6, 50, 7),
    (5, 50, 9),
    (7, 50, 8),
    (6, 50, 7),
    (9, 50, 8),
    (4, 50, 7),
    (9, 50, 8),
    (2, 50, 10),
]

_DIFF_BIAS: dict[str, int] = {"easy": 0, "medium": 3, "hard": 7}


def _procedural_batch(round_num: int, roll: int) -> tuple[int, int]:
    """Return `(rank, count)` for a single batch in a round-51+ procedural round,
    given the (already-biased) random roll. Mirrors the switch on _loc4_ in
    BuildLevels."""
    if roll > 47:
        rank = 10
        count = int(round((round_num - 50) / 3.0))
    elif roll > 39:
        rank = 9
        count = round_num - 42
    elif roll > 29:
        rank = 8
        count = round_num - 40
    elif roll > 16:
        rank = 7
        count = 10
    elif roll > 10:
        rank = 6
        count = 10
    else:
        rank = 5
        count = 10
    return rank, max(0, count)


def build_levels(
    rng: np.random.Generator,
    difficulty: str,
    last_round: int = 149,
) -> dict[int, list[int]]:
    """Build {round: [rank, ...]} for rounds 1..last_round. Consumes RNG state
    for rounds 51+ — call once at sim init."""
    levels: dict[int, list[int]] = {}
    for count, round_num, rank in _HARDCODED_ABSTL:
        levels.setdefault(round_num, []).extend([rank] * count)

    if last_round <= 50:
        return levels

    bias = _DIFF_BIAS[difficulty]
    for round_num in range(51, last_round + 1):
        n_batches = 7 + round_num - 50
        ranks_list = levels.setdefault(round_num, [])
        for _ in range(n_batches):
            roll = int(rng.integers(0, round_num)) + bias
            rank, count = _procedural_batch(round_num, roll)
            if count > 0:
                ranks_list.extend([rank] * count)
    return levels
