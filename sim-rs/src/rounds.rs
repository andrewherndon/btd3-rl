/// Round generation. Ports BloonsTD.BuildLevels() and sim/btd/rounds.py.
///
/// Rounds 1-50 hardcoded from the AS ABSTL calls (order = spawn order).
/// Rounds 51-149 procedural: 7 + (round - 50) batches per round, rank chosen
/// by a random roll biased by difficulty.

use crate::rng::GameRng;
use std::collections::HashMap;

// (count, round, rank). Direct transcription of BloonsTD.as lines 1536-1691.
const HARDCODED_ABSTL: &[(u16, u16, u8)] = &[
    (14, 1, 1), (30, 2, 1),
    (10, 3, 1), (4, 3, 2), (5, 3, 1), (4, 3, 2),
    (5, 4, 1), (12, 4, 2), (5, 4, 1), (12, 4, 2),
    (10, 5, 1), (8, 5, 2), (12, 5, 1), (20, 5, 2),
    (13, 6, 1), (7, 6, 3),
    (50, 7, 2),
    (9, 8, 1), (16, 8, 2), (9, 8, 1), (7, 8, 2), (9, 8, 1), (7, 8, 2),
    (8, 6, 3),
    (20, 9, 2), (15, 9, 3), (12, 9, 2),
    (32, 10, 3),
    (12, 11, 3), (7, 11, 4),
    (1, 12, 8),
    (4, 11, 4),
    (18, 13, 2), (18, 13, 1), (30, 13, 3), (20, 13, 2),
    (1, 14, 8), (12, 14, 4),
    (8, 15, 4), (6, 15, 3), (8, 15, 4), (8, 15, 3), (5, 15, 4),
    (35, 16, 3), (15, 16, 4), (9, 16, 2), (7, 16, 4),
    (20, 17, 2), (55, 17, 3), (10, 17, 4),
    (30, 18, 2), (25, 18, 4), (28, 18, 3),
    (45, 19, 3), (25, 19, 4),
    (5, 20, 7),
    (17, 21, 4), (10, 21, 2), (27, 21, 4), (10, 21, 3), (30, 21, 3),
    (50, 22, 4),
    (30, 23, 4), (35, 23, 3), (30, 23, 4),
    (30, 24, 3), (45, 24, 4), (26, 24, 3), (20, 24, 2),
    (20, 25, 4), (15, 25, 5), (22, 25, 4),
    (80, 26, 4), (15, 26, 5),
    (35, 27, 5),
    (19, 28, 5), (16, 28, 6),
    (20, 26, 4), (14, 26, 7),
    (6, 29, 7), (12, 29, 5), (14, 29, 6),
    (60, 30, 4), (28, 30, 5),
    (2, 31, 9),
    (20, 32, 4), (16, 32, 6), (22, 32, 5),
    (60, 33, 5), (3, 33, 9),
    (25, 34, 5), (25, 34, 6), (50, 34, 4), (4, 34, 9),
    (12, 35, 8),
    (11, 36, 5), (12, 36, 4), (10, 36, 5), (10, 36, 7), (12, 36, 6), (9, 36, 5),
    (1, 37, 10),
    (1, 38, 9), (60, 38, 4), (50, 38, 5), (4, 38, 9),
    (50, 39, 4), (22, 39, 5), (22, 39, 6), (10, 39, 7), (9, 39, 8),
    (64, 40, 5), (5, 40, 9), (25, 39, 6),
    (18, 41, 6), (14, 41, 7), (16, 41, 8),
    (10, 42, 9), (100, 42, 4), (54, 42, 5),
    (23, 43, 8), (20, 43, 7), (5, 43, 9),
    (5, 44, 9), (130, 44, 5), (1, 44, 10),
    (12, 46, 8), (11, 45, 9), (90, 45, 6),
    (8, 46, 9), (38, 46, 7), (18, 46, 8),
    (20, 47, 5), (40, 47, 6), (6, 47, 9), (18, 47, 7), (15, 47, 8), (6, 47, 9),
    (25, 48, 8), (30, 48, 6), (30, 48, 5), (25, 48, 7), (12, 48, 8),
    (5, 49, 9), (34, 49, 8), (17, 49, 9),
    (8, 50, 9), (13, 50, 8), (6, 50, 7), (5, 50, 9), (7, 50, 8), (6, 50, 7),
    (9, 50, 8), (4, 50, 7), (9, 50, 8), (2, 50, 10),
];

const DIFF_BIAS_EASY: u32 = 0;
const DIFF_BIAS_MEDIUM: u32 = 3;
const DIFF_BIAS_HARD: u32 = 7;

fn diff_bias(difficulty: &str) -> u32 {
    match difficulty {
        "medium" => DIFF_BIAS_MEDIUM,
        "hard" => DIFF_BIAS_HARD,
        _ => DIFF_BIAS_EASY,
    }
}

fn procedural_batch(round_num: u16, roll: u32) -> (u8, u32) {
    // Mirrors the switch on _loc4_ in BuildLevels.
    if roll > 47 {
        (10, ((round_num as f64 - 50.0) / 3.0).round() as u32)
    } else if roll > 39 {
        (9, (round_num - 42) as u32)
    } else if roll > 29 {
        (8, (round_num - 40) as u32)
    } else if roll > 16 {
        (7, 10)
    } else if roll > 10 {
        (6, 10)
    } else {
        (5, 10)
    }
}

/// Build {round: [rank, ...]} for rounds 1..last_round. Consumes RNG state
/// for rounds 51+ — call once at sim init.
pub(crate) fn build_levels(
    rng: &mut GameRng,
    difficulty: &str,
    last_round: u16,
) -> HashMap<u16, Vec<u8>> {
    let mut levels: HashMap<u16, Vec<u8>> = HashMap::new();

    for &(count, round_num, rank) in HARDCODED_ABSTL {
        if round_num > last_round { break; }
        let entry = levels.entry(round_num).or_default();
        for _ in 0..count {
            entry.push(rank);
        }
    }

    if last_round <= 50 {
        return levels;
    }

    let bias = diff_bias(difficulty);
    for round_num in 51..=last_round {
        let n_batches = 7 + (round_num - 50);
        let entry = levels.entry(round_num).or_default();
        for _ in 0..n_batches {
            let roll = rng.int(round_num as u32) + bias;
            let (rank, count) = procedural_batch(round_num, roll);
            if count > 0 {
                for _ in 0..count {
                    entry.push(rank);
                }
            }
        }
    }

    levels
}
