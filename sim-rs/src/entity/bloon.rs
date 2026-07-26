/// Bloon entity. Mirrors Bloon.as + sim/btd/bloon.py.

#[derive(Clone, Debug)]
pub(crate) struct Bloon {
    pub rank: u8,
    pub frame: f64,         // advances by `speed` each tick
    pub maxspeed: f64,
    pub speed: f64,
    pub jitter_x: f64,
    pub jitter_y: f64,
    pub branch: u8,         // branch index (track 3 = 1)
    pub hits_remaining: u8, // only meaningful for MOAB/ceramic
    pub radius: f64,
    // Stage position cached each tick.
    pub x: f64,
    pub y: f64,
    // Lifecycle.
    pub popped: bool,
    pub escaped: bool,
    pub hit_this_frame: bool, // bloon absorbs at most one bullet per tick
    // Freeze state.
    pub frozen: bool,
    pub time_frozen: u16,
    pub freeze_duration: u16,
    pub freezer_id: i32,       // -1 = no freezer
    pub snap_frozen: bool,
}

impl Bloon {
    pub fn new(
        rank: u8,
        frame: f64,
        maxspeed: f64,
        speed: f64,
        jitter_x: f64,
        jitter_y: f64,
        branch: u8,
        radius: f64,
    ) -> Self {
        let hits = crate::constants::BLOON_HITS[rank as usize];
        Self {
            rank,
            frame: frame.max(0.0),
            maxspeed,
            speed,
            jitter_x,
            jitter_y,
            branch,
            hits_remaining: hits,
            radius,
            x: 0.0,
            y: 0.0,
            popped: false,
            escaped: false,
            hit_this_frame: false,
            frozen: false,
            time_frozen: 0,
            freeze_duration: 0,
            freezer_id: -1,
            snap_frozen: false,
        }
    }

    pub fn alive(&self) -> bool {
        !self.popped && !self.escaped
    }
}
