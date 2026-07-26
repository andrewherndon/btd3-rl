/// Game balance constants. Mirrors sim/btd/constants.py + BloonsTD.as.

pub const FPS: u32 = 40;
pub const STAGE_W: f64 = 640.0;
pub const STAGE_H: f64 = 480.0;

pub const STARTING_MONEY: u32 = 650;
pub const SELL_RATE: f64 = 0.8;

pub const LIVES_EASY: u32 = 100;
pub const LIVES_MEDIUM: u32 = 75;
pub const LIVES_HARD: u32 = 50;

pub const COST_MULT_EASY: f64 = 0.85;
pub const COST_MULT_MEDIUM: f64 = 1.02;
pub const COST_MULT_HARD: f64 = 1.08;

// Per-rank max path-frames-per-game-frame, from Bloon.Init switch.
pub const BLOON_MAXSPEED: [f64; 11] = [
    0.0,   // 0 unused
    1.0,   // 1 red
    1.4,   // 2 blue
    1.8,   // 3 green
    3.2,   // 4 yellow
    1.8,   // 5 black
    2.5,   // 6 white
    1.0,   // 7 lead
    2.2,   // 8 rainbow
    2.5,   // 9 ceramic
    1.0,   // 10 MOAB
];

// Lives lost on escape, from BloonsTD.Escaped.
pub const BLOON_ESCAPE_DAMAGE: [u32; 11] = [
    0,    // 0 unused
    1,    // 1
    2,    // 2
    3,    // 3
    4,    // 4
    9,    // 5
    9,    // 6
    19,   // 7
    37,   // 8
    38,   // 9
    100,  // 10
];

// Hits to pop. Everything not listed is 1 hit.
pub const BLOON_HITS: [u8; 11] = [
    1,  // 0 unused
    1,  // 1 red
    1,  // 2 blue
    1,  // 3 green
    1,  // 4 yellow
    1,  // 5 black
    1,  // 6 white
    1,  // 7 lead
    1,  // 8 rainbow
    8,  // 9 ceramic
    130, // 10 MOAB
];

// Pop hierarchy from Bloon.RemoveMe: rank -> [(child_rank, frame_offset), ...]
pub const BLOON_CHILDREN: &[(u8, &[(u8, i16)])] = &[
    (2, &[(1, 0)]),
    (3, &[(2, 0)]),
    (4, &[(3, 0)]),
    (5, &[(4, 5), (4, -5)]),
    (6, &[(4, 5), (4, -5)]),
    (7, &[(5, 4), (5, -4)]),
    (8, &[(5, 5), (5, 1), (6, -1), (6, -5)]),
    (9, &[(8, 6), (8, -6)]),
    (10, &[(9, 5), (9, 2), (9, -2), (9, -5)]),
];

// Collision radii in px. Calibrated from SWF bbox data.
pub const BLOON_RADIUS: [f64; 11] = [
    0.0,  // 0 unused
    19.0, // 1 red
    21.0, // 2 blue
    22.0, // 3 green
    24.0, // 4 yellow
    10.0, // 5 black
    19.0, // 6 white
    21.0, // 7 lead
    23.0, // 8 rainbow
    23.0, // 9 ceramic
    57.0, // 10 MOAB
];

// Tower type IDs, in the same order as Python's TOWER_TYPES.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TowerType {
    Dart = 0,
    Tack = 1,
    Ice = 2,
    Bomb = 3,
    Spikeopult = 4,
    Super = 5,
    Boomerang = 6,
    Beacon = 7,
}

pub const TOWER_TYPE_NAMES: &[&str] = &[
    "dart", "tack", "ice", "bomb", "spikeopult", "super", "boomerang", "beacon",
];

impl TowerType {
    pub fn from_name(name: &str) -> Option<TowerType> {
        match name {
            "dart" => Some(TowerType::Dart),
            "tack" => Some(TowerType::Tack),
            "ice" => Some(TowerType::Ice),
            "bomb" => Some(TowerType::Bomb),
            "spikeopult" => Some(TowerType::Spikeopult),
            "super" => Some(TowerType::Super),
            "boomerang" => Some(TowerType::Boomerang),
            "beacon" => Some(TowerType::Beacon),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        TOWER_TYPE_NAMES[*self as usize]
    }
}

// Bullet type IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BulletType {
    Dart,
    Tack,
    Ice,
    Bomb,
    Frag,
    Spikeopult,
    Super,
    Boomerang,
}

impl BulletType {
    pub fn from_name(name: &str) -> Option<BulletType> {
        match name {
            "dart" => Some(BulletType::Dart),
            "tack" => Some(BulletType::Tack),
            "ice" => Some(BulletType::Ice),
            "bomb" => Some(BulletType::Bomb),
            "frag" => Some(BulletType::Frag),
            "spikeopult" => Some(BulletType::Spikeopult),
            "super" => Some(BulletType::Super),
            "boomerang" => Some(BulletType::Boomerang),
            _ => None,
        }
    }
}

// Tower initial stats.
#[derive(Clone, Copy, Debug)]
pub struct TowerStats {
    pub attack_rate: u16,
    pub attack_radius: f64,
    pub shoot_power: f64,
    pub pierce_max: u16,
    pub cost: u32,
    pub icebreak: bool,
    pub leadbreak: bool,
    pub is_spread: bool,
    pub is_attacker: bool,
    pub freeze_len: u16,
}

pub const TOWER_STATS: &[(TowerType, TowerStats)] = &[
    (TowerType::Dart, TowerStats {
        attack_rate: 33, attack_radius: 100.0, shoot_power: 23.0, pierce_max: 1,
        cost: 250, icebreak: false, leadbreak: false, is_spread: false,
        is_attacker: true, freeze_len: 0,
    }),
    (TowerType::Tack, TowerStats {
        attack_rate: 54, attack_radius: 70.0, shoot_power: 15.0, pierce_max: 8,
        cost: 360, icebreak: false, leadbreak: false, is_spread: true,
        is_attacker: true, freeze_len: 0,
    }),
    (TowerType::Ice, TowerStats {
        attack_rate: 93, attack_radius: 60.0, shoot_power: 6.0, pierce_max: 50,
        cost: 425, icebreak: false, leadbreak: false, is_spread: true,
        is_attacker: true, freeze_len: 50,
    }),
    (TowerType::Bomb, TowerStats {
        attack_rate: 54, attack_radius: 120.0, shoot_power: 13.0, pierce_max: 18,
        cost: 725, icebreak: true, leadbreak: true, is_spread: false,
        is_attacker: true, freeze_len: 0,
    }),
    (TowerType::Spikeopult, TowerStats {
        attack_rate: 63, attack_radius: 110.0, shoot_power: 10.0, pierce_max: 6,
        cost: 600, icebreak: false, leadbreak: false, is_spread: false,
        is_attacker: true, freeze_len: 0,
    }),
    (TowerType::Super, TowerStats {
        attack_rate: 2, attack_radius: 140.0, shoot_power: 20.0, pierce_max: 1,
        cost: 4000, icebreak: false, leadbreak: false, is_spread: false,
        is_attacker: true, freeze_len: 0,
    }),
    (TowerType::Boomerang, TowerStats {
        attack_rate: 50, attack_radius: 130.0, shoot_power: 0.0, pierce_max: 2,
        cost: 515, icebreak: false, leadbreak: false, is_spread: false,
        is_attacker: true, freeze_len: 0,
    }),
    (TowerType::Beacon, TowerStats {
        attack_rate: 60, attack_radius: 120.0, shoot_power: 0.0, pierce_max: 0,
        cost: 1000, icebreak: false, leadbreak: false, is_spread: false,
        is_attacker: false, freeze_len: 0,
    }),
];

// Bullet initial stats.
#[derive(Clone, Copy, Debug)]
pub struct BulletStats {
    pub lifespan: u16,
    pub radius: f64,
    pub explosion_radius: f64,
}

pub const BULLET_STATS: &[(BulletType, BulletStats)] = &[
    (BulletType::Dart, BulletStats { lifespan: 7, radius: 4.0, explosion_radius: 0.0 }),
    (BulletType::Tack, BulletStats { lifespan: 5, radius: 4.0, explosion_radius: 0.0 }),
    (BulletType::Ice, BulletStats { lifespan: 10, radius: 4.0, explosion_radius: 0.0 }),
    (BulletType::Boomerang, BulletStats { lifespan: 24, radius: 6.0, explosion_radius: 0.0 }),
    (BulletType::Bomb, BulletStats { lifespan: 18, radius: 6.0, explosion_radius: 30.0 }),
    (BulletType::Frag, BulletStats { lifespan: 5, radius: 4.0, explosion_radius: 0.0 }),
    (BulletType::Spikeopult, BulletStats { lifespan: 20, radius: 6.0, explosion_radius: 0.0 }),
    (BulletType::Super, BulletStats { lifespan: 20, radius: 4.0, explosion_radius: 0.0 }),
];

// Spread towers fire SPREAD_SHARDS projectiles in a uniform fan.
pub const SPREAD_SHARDS: u32 = 8;

// Spawn-time jitter range (uniform 0..9 inclusive on each axis).
pub const SPAWN_JITTER_RANGE: f64 = 10.0;

// Radius around path centerline where towers cannot be placed.
pub const PATH_PLACEMENT_BUFFER: f64 = 16.0;

// Round-end timing.
pub const ROUND_END_GRACE_FRAMES: u16 = 20;
pub const ROUND_END_TIMEOUT_FRAMES: u16 = 5 * FPS as u16; // 200

// Beacon buff multipliers.
pub const BEACON_RANGE_FACTOR: f64 = 1.2;
pub const BEACON_RATE_FACTOR: f64 = 0.85;
