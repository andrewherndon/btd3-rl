/// Deterministic RNG wrapper using ChaCha12 (via rand_chacha).
///
/// Mirrors the Python approach: a main SeedSequence spawns sub-streams, each
/// consumed by a different consumer (main game RNG, round generation, etc.).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

#[derive(Clone, Debug)]
pub(crate) struct GameRng {
    inner: ChaCha12Rng,
}

impl GameRng {
    /// Create from a seed. Two sub-streams are derived internally:
    ///   stream 0 — main game RNG (jitter, pop money, snap freeze, etc.)
    ///   stream 1 — round generation (rounds 51+)
    pub fn from_seed(seed: u64) -> (GameRng, GameRng) {
        let mut main = ChaCha12Rng::seed_from_u64(seed);
        let stream1_seed: u64 = main.gen();
        (
            GameRng { inner: main },
            GameRng { inner: ChaCha12Rng::seed_from_u64(stream1_seed) },
        )
    }

    /// Uniform integer in [0, max). Mirrors `rng.integers(0, max)` in Python.
    pub fn int(&mut self, max: u32) -> u32 {
        if max <= 1 { return 0; }
        self.inner.gen::<u64>() as u32 % max
    }

    /// Uniform integer in [0, 100).
    pub fn int_100(&mut self) -> u32 {
        self.int(100)
    }

    /// Uniform integer in [0, 3).
    pub fn int_3(&mut self) -> u32 {
        self.int(3)
    }

    /// Uniform integer in [0, 5).
    pub fn int_5(&mut self) -> u32 {
        self.int(5)
    }
}
