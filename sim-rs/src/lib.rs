//! btd_rs — BTD3 simulator core, exposed to Python via PyO3.
//!
//! Usage from Python:
//! ```python
//! from btd_rs import BloonsSim, SimConfig
//! sim = BloonsSim(SimConfig(track=3, difficulty="easy", seed=0))
//! sim.place_tower("dart", 350.0, 100.0)
//! sim.start_round()
//! sim.step()
//! ```

#![allow(dead_code)]

mod constants;
mod entity;
mod game;
mod path;
mod rng;
mod rounds;
mod upgrades;

use std::collections::HashMap;
use pyo3::prelude::*;

use crate::constants::TOWER_TYPE_NAMES;
use crate::game::{BloonsSim as CoreSim, SimConfig as CoreConfig};

// ---- SimConfig (Python class) -----------------------------------------------

/// Configuration for a single simulator instance.
#[pyclass(name = "SimConfig")]
#[derive(Clone, Debug)]
pub struct PySimConfig {
    #[pyo3(get, set)]
    pub track: u8,
    #[pyo3(get, set)]
    pub difficulty: String,
    #[pyo3(get, set)]
    pub seed: u64,
    #[pyo3(get, set)]
    pub freeplay: bool,
    #[pyo3(get, set)]
    pub paths_dir: String,
}

#[pymethods]
impl PySimConfig {
    #[new]
    #[pyo3(signature = (track=3, difficulty="easy".to_string(), seed=0, freeplay=false, paths_dir=None))]
    fn new(
        track: u8,
        difficulty: String,
        seed: u64,
        freeplay: bool,
        paths_dir: Option<String>,
    ) -> Self {
        let pd = paths_dir.unwrap_or_else(|| "paths".to_string());
        Self { track, difficulty, seed, freeplay, paths_dir: pd }
    }

    fn __repr__(&self) -> String {
        format!("SimConfig(track={}, difficulty={:?}, seed={}, freeplay={})",
                self.track, self.difficulty, self.seed, self.freeplay)
    }
}

// ---- BloonsSim (Python class) -----------------------------------------------

/// BTD3 game simulation. Call step() to advance one game frame (1/40 s).
#[pyclass(name = "BloonsSim")]
pub struct PyBloonsSim {
    inner: CoreSim,
}

#[pymethods]
impl PyBloonsSim {
    #[new]
    fn new(config: &PySimConfig) -> Self {
        let core_config = CoreConfig {
            track: config.track,
            difficulty: config.difficulty.clone(),
            seed: config.seed,
            freeplay: config.freeplay,
            paths_dir: config.paths_dir.clone(),
        };
        Self { inner: CoreSim::new(core_config) }
    }

    /// Advance one game frame (1/40 s).
    fn step(&mut self) {
        self.inner.step();
    }

    /// Start the next round. Returns False if a round is already active.
    fn start_round(&mut self) -> bool {
        self.inner.start_round()
    }

    /// Place a tower. Returns the tower ID (>=0) or -1 on failure.
    fn place_tower(&mut self, type_: &str, x: f64, y: f64) -> i32 {
        self.inner.place_tower(type_, x, y)
    }

    /// Check if a position is valid for placement.
    fn is_placement_valid(&self, x: f64, y: f64) -> bool {
        self.inner.is_placement_valid(x, y)
    }

    /// Distance from (x, y) to the nearest path centerline.
    fn distance_to_path(&self, x: f64, y: f64) -> f64 {
        self.inner.distance_to_path(x, y)
    }

    /// Sell a tower by ID. Returns True on success.
    fn sell_tower(&mut self, tower_id: u32) -> bool {
        self.inner.sell_tower(tower_id)
    }

    /// Buy a specific upgrade by name (e.g. "dart1"). Returns True on success.
    fn upgrade_tower(&mut self, tower_id: u32, upgrade_name: &str) -> bool {
        self.inner.upgrade_tower(tower_id, upgrade_name)
    }

    /// Buy the next upgrade on `path` (1 or 2) for the given tower.
    fn upgrade_path(&mut self, tower_id: u32, path: u8) -> bool {
        self.inner.upgrade_path(tower_id, path)
    }

    /// Get available upgrades as {1: (name, price)|None, 2: (name, price)|None}.
    fn available_upgrades(&self, tower_id: u32) -> HashMap<u8, Option<(String, u32)>> {
        self.inner.available_upgrades(tower_id)
    }

    // -- debug helpers ---------------------------------------------------------

    fn debug_add_money(&mut self, amount: i32) {
        self.inner.debug_add_money(amount);
    }

    fn debug_add_lives(&mut self, amount: i32) {
        self.inner.debug_add_lives(amount);
    }

    fn debug_set_round(&mut self, round_num: u16) -> bool {
        self.inner.debug_set_round(round_num)
    }

    fn debug_clear_bloons(&mut self) {
        self.inner.debug_clear_bloons();
    }

    // -- observation snapshot --------------------------------------------------

    fn observe<'py>(&self, py: Python<'py>) -> PyResult<HashMap<String, PyObject>> {
        let mut d = HashMap::new();
        d.insert("frame".to_string(), self.inner.frame_count.to_object(py));
        d.insert("round".to_string(), self.inner.round.to_object(py));
        d.insert("in_round".to_string(), self.inner.in_round.to_object(py));
        d.insert("money".to_string(), self.inner.money.to_object(py));
        d.insert("lives".to_string(), self.inner.lives.to_object(py));
        d.insert("game_over".to_string(), self.inner.game_over.to_object(py));
        d.insert("won".to_string(), self.inner.won.to_object(py));
        d.insert("n_bloons".to_string(),
                 (self.inner.bloons.iter().filter(|b| b.alive()).count()).to_object(py));
        d.insert("n_bullets".to_string(), self.inner.bullets.len().to_object(py));
        d.insert("n_towers".to_string(), self.inner.towers.len().to_object(py));
        d.insert("pops_this_round".to_string(), self.inner.bloons_popped_this_round.to_object(py));
        Ok(d)
    }

    // -- property access for env -------------------------------------------------

    #[getter]
    fn n_bloons(&self) -> usize {
        self.inner.bloons.iter().filter(|b| b.alive()).count()
    }

    #[getter] fn round(&self) -> u16 { self.inner.round }
    #[getter] fn money(&self) -> u32 { self.inner.money }
    #[getter] fn lives(&self) -> u32 { self.inner.lives }
    #[getter] fn n_towers(&self) -> usize { self.inner.towers.len() }
    #[getter] fn in_round(&self) -> bool { self.inner.in_round }
    #[getter] fn game_over(&self) -> bool { self.inner.game_over }
    #[getter] fn won(&self) -> bool { self.inner.won }

    fn get_towers<'py>(&self, py: Python<'py>) -> Vec<HashMap<String, PyObject>> {
        self.inner.towers.iter().map(|t| {
            let mut d = HashMap::new();
            d.insert("id".to_string(), t.id.to_object(py));
            d.insert("type".to_string(), TOWER_TYPE_NAMES[t.type_ as usize].to_object(py));
            d.insert("x".to_string(), t.x.to_object(py));
            d.insert("y".to_string(), t.y.to_object(py));
            d.insert("upgrade1".to_string(), t.upgrade1.to_object(py));
            d.insert("upgrade2".to_string(), t.upgrade2.to_object(py));
            d.insert("upgrade3".to_string(), t.upgrade3.to_object(py));
            d.insert("upgrade4".to_string(), t.upgrade4.to_object(py));
            d.insert("pop_count".to_string(), t.pop_count.to_object(py));
            d
        }).collect()
    }

    fn get_next_round_bloons(&self) -> Vec<u8> {
        self.inner.round_table.get(&(self.inner.round + 1))
            .cloned().unwrap_or_default()
    }

    fn __repr__(&self) -> String {
        format!("BloonsSim(round={}, money={}, lives={}, game_over={}, won={})",
                self.inner.round, self.inner.money, self.inner.lives,
                self.inner.game_over, self.inner.won)
    }
}

// ---- PyO3 module registration -----------------------------------------------

/// BTD3 simulator — Rust backend.
#[pymodule]
fn btd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimConfig>()?;
    m.add_class::<PyBloonsSim>()?;
    Ok(())
}
