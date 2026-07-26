/// Path data loading. Reads pre-converted binary path files.
///
/// Format: [u32 little-endian count, f64[count][2] data].
/// Converted from .npy via sim-rs/convert_paths.py.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// All path data for a simulation: one array per branch.
///
/// Track 3 has a single branch (key 1). Branched tracks (4, 6, 8) would
/// have multiple entries.
#[derive(Clone, Debug)]
pub(crate) struct PathData {
    pub branches: HashMap<u8, Vec<[f64; 2]>>,
}

impl PathData {
    /// Load from a directory containing .bin files.
    /// Files are named either `track_{track}.bin` (single branch) or
    /// `track_{track}_{branch}.bin` (multi-branch).
    pub fn load(paths_dir: &Path, track: u8) -> Self {
        let mut branches = HashMap::new();

        // Try single-branch first.
        let primary = paths_dir.join(format!("track_{}.bin", track));
        if primary.exists() {
            branches.insert(1, load_bin(&primary));
            return Self { branches };
        }

        // Multi-branch: try branches 1, 2, 3.
        for branch in 1..=3u8 {
            let p = paths_dir.join(format!("track_{}_{}.bin", track, branch));
            if p.exists() {
                branches.insert(branch, load_bin(&p));
            }
        }

        assert!(!branches.is_empty(), "No path data for track {}", track);
        Self { branches }
    }

    /// Get path data for a branch.
    pub fn get(&self, branch: u8) -> &[[f64; 2]] {
        &self.branches[&branch]
    }

    /// Get path length for a branch.
    pub fn len(&self, branch: u8) -> usize {
        self.branches[&branch].len()
    }

    /// Get max index for a branch (length - 1, or 0 for empty).
    pub fn max_idx(&self, branch: u8) -> usize {
        let l = self.len(branch);
        if l > 0 { l - 1 } else { 0 }
    }

    /// Unclamped length of a branch.
    pub fn len_clamped(&self, branch: u8) -> usize {
        self.len(branch).max(1)
    }
}

fn load_bin(path: &Path) -> Vec<[f64; 2]> {
    let bytes = fs::read(path).expect("Failed to read path file");
    let (count_bytes, rest) = bytes.split_at(4);
    let count = u32::from_le_bytes(count_bytes.try_into().unwrap()) as usize;
    let expected = count * 16; // 2 f64s = 16 bytes per point
    assert_eq!(rest.len(), expected, "Path file size mismatch for {:?}", path);

    rest.chunks_exact(16)
        .map(|chunk| {
            let x = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let y = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            [x, y]
        })
        .collect()
}

/// Load boomerang arc data.
pub(crate) fn load_boomerang_arc(paths_dir: &Path) -> Option<Vec<[f64; 2]>> {
    let p = paths_dir.join("boomerang_arc.bin");
    if p.exists() {
        Some(load_bin(&p))
    } else {
        None
    }
}
