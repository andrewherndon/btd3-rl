/// Bullet entity. Mirrors Bullet.as + sim/btd/bullet.py.

use crate::constants::{BulletType, BULLET_STATS};

#[derive(Clone, Debug)]
pub(crate) struct Bullet {
    pub type_: BulletType,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub pierce_max: u16,
    pub radius: f64,
    pub lifespan: u16,
    pub shooter_id: u32,
    pub icebreak: bool,
    pub leadbreak: bool,
    pub freeze_len: u16,       // forwarded from ice tower
    // Two-stage bullets (bomb).
    pub explosion_radius: f64,
    pub hashit: bool,
    // Boomerang-only fields.
    pub arc_anchor_x: f64,
    pub arc_anchor_y: f64,
    pub arc_angle: f64,
    // Tracking.
    pub pierce_count: u16,
    pub time_alive: u16,
    pub is_dead: bool,
}

impl Bullet {
    pub fn from_type(
        type_: BulletType,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        pierce_max: u16,
        shooter_id: u32,
        icebreak: bool,
        leadbreak: bool,
        freeze_len: u16,
        scale: f64,
    ) -> Self {
        let stats = BULLET_STATS.iter().find(|(t, _)| *t == type_).map(|(_, s)| s).unwrap();
        Self {
            type_,
            x,
            y,
            vx,
            vy,
            pierce_max,
            radius: stats.radius * scale,
            lifespan: stats.lifespan,
            shooter_id,
            icebreak,
            leadbreak,
            freeze_len,
            explosion_radius: stats.explosion_radius * scale,
            hashit: false,
            arc_anchor_x: 0.0,
            arc_anchor_y: 0.0,
            arc_angle: 0.0,
            pierce_count: 0,
            time_alive: 0,
            is_dead: false,
        }
    }
}
