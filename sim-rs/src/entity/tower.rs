/// Tower entity. Mirrors Tower.as + sim/btd/tower.py.

use crate::constants::{TowerType, TOWER_STATS};

#[derive(Clone, Debug)]
pub(crate) struct Tower {
    pub id: u32,
    pub type_: TowerType,
    pub x: f64,
    pub y: f64,
    pub attack_rate: u16,
    pub attack_radius: f64,
    pub shoot_power: f64,
    pub pierce_max: u16,
    pub spent_on_me: u32,
    pub icebreak: bool,
    pub leadbreak: bool,
    pub is_spread: bool,
    pub is_attacker: bool,
    pub freeze_len: u16,
    pub bullet_scale: f64,
    // Upgrade flags.
    pub upgrade1: bool,
    pub upgrade2: bool,
    pub upgrade3: bool,
    pub upgrade4: bool,
    // Visual / behavioural flags.
    pub transformed: bool,
    pub laser: bool,
    // Beacon buffs (refreshed each frame).
    pub beacon_radius_active: bool,
    pub beacon_rate_active: bool,
    // Stats.
    pub pop_count: u32,
    pub time_since_last_shot: u16,
    pub placed_round: u16,
}

impl Tower {
    pub fn from_type(tower_id: u32, type_: TowerType, x: f64, y: f64) -> Self {
        let stats = TOWER_STATS.iter().find(|(t, _)| *t == type_).map(|(_, s)| s).unwrap();
        Self {
            id: tower_id,
            type_,
            x,
            y,
            attack_rate: stats.attack_rate,
            attack_radius: stats.attack_radius,
            shoot_power: stats.shoot_power,
            pierce_max: stats.pierce_max,
            spent_on_me: stats.cost,
            icebreak: stats.icebreak,
            leadbreak: stats.leadbreak,
            is_spread: stats.is_spread,
            is_attacker: stats.is_attacker,
            freeze_len: stats.freeze_len,
            bullet_scale: 1.0,
            upgrade1: false,
            upgrade2: false,
            upgrade3: false,
            upgrade4: false,
            transformed: false,
            laser: false,
            beacon_radius_active: false,
            beacon_rate_active: false,
            pop_count: 0,
            time_since_last_shot: 0,
            placed_round: 0,
        }
    }
}
