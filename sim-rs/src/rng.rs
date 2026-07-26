/// Deterministic RNG wrapper using PCG64 (numpy-compatible algorithm).
///
/// Uses Pcg64 (Lcg128Xsl64) — the same algorithm as numpy's default
/// Generator (PCG64). Both use LCG 128/64 with the XSL RR output function.
/// This gives identical random sequences for the same seed, making the
/// simulator bit-exact with Python.
///
/// Two sub-streams: main (gameplay RNG) and rounds (rounds 51+ generation).

use rand::RngCore;
use rand::SeedableRng;
use rand_pcg::Pcg64;

#[derive(Clone, Debug)]
pub(crate) struct GameRng {
    inner: Pcg64,
}

impl GameRng {
    /// Create from a seed. Two sub-streams are derived:
    ///   stream 0 — main game RNG (jitter, pop money, snap freeze, etc.)
    ///   stream 1 — round generation (rounds 51+)
    pub fn from_seed(seed: u64) -> (GameRng, GameRng) {
        let mut main = Pcg64::seed_from_u64(seed);
        let stream1_seed: u64 = main.next_u64();
        (
            GameRng { inner: main },
            GameRng { inner: Pcg64::seed_from_u64(stream1_seed) },
        )
    }

    /// Uniform integer in [0, max). Mirrors `rng.integers(0, max)` in Python
    /// numpy (PCG64 backend).
    pub fn int(&mut self, max: u32) -> u32 {
        if max <= 1 { return 0; }
        self.inner.next_u64() as u32 % max
    }

    pub fn int_100(&mut self) -> u32 { self.int(100) }
    pub fn int_3(&mut self) -> u32 { self.int(3) }
    pub fn int_5(&mut self) -> u32 { self.int(5) }
}
