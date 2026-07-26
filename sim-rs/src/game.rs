/// BloonsSim — the top-level simulator. Mirrors sim/btd/game.py and
/// BloonsTD.as (Init, StartLevel, EnterFrame, NewBloon, ShootBullet, etc.).

use std::collections::HashMap;
use std::f64::consts;
use std::path::Path;

use crate::constants::*;
use crate::entity::bloon::Bloon;
use crate::entity::bullet::Bullet;
use crate::entity::tower::Tower;
use crate::path::PathData;
use crate::rng::GameRng;
use crate::rounds;
use crate::upgrades;

// ---- SimConfig ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct SimConfig {
    pub track: u8,
    pub difficulty: String,
    pub seed: u64,
    pub freeplay: bool,
    pub paths_dir: String,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            track: 3,
            difficulty: "easy".to_string(),
            seed: 0,
            freeplay: false,
            paths_dir: "paths".to_string(),
        }
    }
}

// Free function helpers (defined before impl so they can be called from it).

fn apply_upgrade(tower: &mut Tower, upgrade_name: &str, spec: &upgrades::UpgradeSpec) {
    let prefix = tower.type_.name();
    let suffix = &upgrade_name[prefix.len()..];
    match suffix {
        "1" => tower.upgrade1 = true,
        "2" => tower.upgrade2 = true,
        "3" => tower.upgrade3 = true,
        "4" => tower.upgrade4 = true,
        _ => {}
    }

    for &(attr, delta) in spec.additive {
        match attr {
            "attack_radius" => tower.attack_radius += delta,
            "pierce_max" => tower.pierce_max = (tower.pierce_max as f64 + delta) as u16,
            "attack_rate" => tower.attack_rate = ((tower.attack_rate as f64 + delta).max(1.0)) as u16,
            "freeze_len" => tower.freeze_len = (tower.freeze_len as f64 + delta) as u16,
            _ => {}
        }
    }

    for &(attr, value) in spec.absolute {
        match attr {
            "bullet_scale" => tower.bullet_scale = value,
            "shoot_power" => tower.shoot_power = value,
            "attack_rate" => tower.attack_rate = value as u16,
            "pierce_max" => tower.pierce_max = value as u16,
            _ => {}
        }
    }

    for &(attr, val) in spec.flags {
        match attr {
            "transformed" => tower.transformed = val,
            "is_spread" => tower.is_spread = val,
            "icebreak" => tower.icebreak = val,
            "leadbreak" => tower.leadbreak = val,
            "laser" => tower.laser = val,
            _ => {}
        }
    }

    if spec.reset_tsls { tower.time_since_last_shot = 0; }
}

fn tower_bullet_type(tt: TowerType) -> BulletType {
    match tt {
        TowerType::Dart => BulletType::Dart,
        TowerType::Tack => BulletType::Tack,
        TowerType::Ice => BulletType::Ice,
        TowerType::Bomb => BulletType::Bomb,
        TowerType::Spikeopult => BulletType::Spikeopult,
        TowerType::Super => BulletType::Super,
        TowerType::Boomerang => BulletType::Boomerang,
        TowerType::Beacon => BulletType::Dart,
    }
}

// ---- BloonsSim ---------------------------------------------------------------

pub(crate) struct BloonsSim {
    pub config: SimConfig,

    // RNG
    pub rng: GameRng,

    // Path data, keyed by branch.
    paths: PathData,
    // Pre-computed path lengths for hot loops.
    path_len: HashMap<u8, usize>,
    path_max_idx: HashMap<u8, usize>,
    path_len_clamped: HashMap<u8, usize>,
    // Boomerang arc.
    boomerang_arc: Option<Vec<[f64; 2]>>,

    // Round table (built once at init).
    pub round_table: HashMap<u16, Vec<u8>>,
    pub(crate) max_round: u16,

    // Game state.
    pub money: u32,
    pub lives: u32,
    pub(crate) cost_mult: f64,
    pub round: u16,
    glob_speed_mod: f64,

    // Entities.
    pub bloons: Vec<Bloon>,
    pub towers: Vec<Tower>,
    pub bullets: Vec<Bullet>,
    next_tower_id: u32,

    // Frame / round state.
    pub frame_count: u64,
    pub in_round: bool,
    spawn_queue: Vec<u8>,
    spawn_counter: u16,
    bloon_interval: u16,
    frames_since_last_bloon: u16,
    end_round_count: u16,
    pub game_over: bool,
    pub won: bool,
    pub bloons_popped_this_round: u32,
}

impl BloonsSim {
    pub fn new(config: SimConfig) -> Self {
        let (rng, mut rounds_rng) = GameRng::from_seed(config.seed);

        let paths_dir = Path::new(&config.paths_dir);
        let paths = PathData::load(paths_dir, config.track);
        let boomerang_arc = crate::path::load_boomerang_arc(paths_dir);

        // Pre-compute path lengths.
        let mut path_len = HashMap::new();
        let mut path_max_idx = HashMap::new();
        let mut path_len_clamped = HashMap::new();
        for (&br, p) in &paths.branches {
            let l = p.len();
            path_len.insert(br, l);
            path_max_idx.insert(br, if l > 0 { l - 1 } else { 0 });
            path_len_clamped.insert(br, l.max(1));
        }

        let last_round = if config.freeplay { 149 } else { 50 };
        let round_table = rounds::build_levels(&mut rounds_rng, &config.difficulty, last_round);
        let max_round = round_table.keys().copied().max().unwrap_or(0);

        let lives = match config.difficulty.as_str() {
            "medium" => LIVES_MEDIUM,
            "hard" => LIVES_HARD,
            _ => LIVES_EASY,
        };
        let cost_mult = match config.difficulty.as_str() {
            "medium" => COST_MULT_MEDIUM,
            "hard" => COST_MULT_HARD,
            _ => COST_MULT_EASY,
        };

        Self {
            config,
            rng,
            paths,
            path_len,
            path_max_idx,
            path_len_clamped,
            boomerang_arc,
            round_table,
            max_round,
            money: STARTING_MONEY,
            lives,
            cost_mult,
            round: 0,
            glob_speed_mod: 0.0,
            bloons: Vec::new(),
            towers: Vec::new(),
            bullets: Vec::new(),
            next_tower_id: 0,
            frame_count: 0,
            in_round: false,
            spawn_queue: Vec::new(),
            spawn_counter: 0,
            bloon_interval: 20,
            frames_since_last_bloon: 0,
            end_round_count: 0,
            game_over: false,
            won: false,
            bloons_popped_this_round: 0,
        }
    }

    // -- public API ------------------------------------------------------------

    pub fn step(&mut self) {
        if self.game_over { return; }
        self.frame_count += 1;
        if self.in_round { self.tick_spawns(); }
        self.tick_towers();
        self.tick_bullets();
        self.tick_bloons();
        self.tick_collisions();
        self.cleanup();
        self.tick_round_end();
    }

    pub fn start_round(&mut self) -> bool {
        if self.in_round || self.game_over { return false; }
        self.round += 1;
        self.in_round = true;
        self.spawn_queue = self.round_data(self.round);
        self.spawn_counter = 0;
        self.bloon_interval = self.round_interval(self.round);
        self.frames_since_last_bloon = 0;
        self.end_round_count = 0;
        self.bloons_popped_this_round = 0;
        self.glob_speed_mod = if self.round > 50 {
            let base = (self.round as f64 - 50.0) / 15.0;
            let bonus = match self.config.difficulty.as_str() {
                "medium" => 0.1, "hard" => 0.25, _ => 0.0,
            };
            base + bonus
        } else { 0.0 };
        true
    }

    pub fn place_tower(&mut self, type_name: &str, x: f64, y: f64) -> i32 {
        let tt = match TowerType::from_name(type_name) { Some(t) => t, None => return -1 };
        if !self.is_placement_valid(x, y) { return -1; }
        let cost = TOWER_STATS.iter().find(|(t, _)| *t == tt).map(|(_, s)| s.cost).unwrap();
        let price = self.price(cost);
        if price > self.money { return -1; }
        self.money -= price;
        let tid = self.next_tower_id;
        self.next_tower_id += 1;
        let mut tower = Tower::from_type(tid, tt, x, y);
        tower.placed_round = self.round;
        self.towers.push(tower);
        tid as i32
    }

    pub fn is_placement_valid(&self, x: f64, y: f64) -> bool {
        self.distance_to_path(x, y) > PATH_PLACEMENT_BUFFER
    }

    pub fn distance_to_path(&self, x: f64, y: f64) -> f64 {
        let mut best = f64::INFINITY;
        for branch_path in self.paths.branches.values() {
            for pair in branch_path.windows(2) {
                let a = &pair[0];
                let b = &pair[1];
                let seg_x = b[0] - a[0];
                let seg_y = b[1] - a[1];
                let dx = x - a[0];
                let dy = y - a[1];
                let len_sq = (seg_x * seg_x + seg_y * seg_y).max(1.0);
                let t = ((dx * seg_x + dy * seg_y) / len_sq).clamp(0.0, 1.0);
                let d = (x - (a[0] + t * seg_x)).hypot(y - (a[1] + t * seg_y));
                if d < best { best = d; }
            }
        }
        best
    }

    pub fn sell_tower(&mut self, tower_id: u32) -> bool {
        if let Some(pos) = self.towers.iter().position(|t| t.id == tower_id) {
            self.money += (SELL_RATE * self.towers[pos].spent_on_me as f64).floor() as u32;
            self.towers.remove(pos);
            true
        } else { false }
    }

    // -- upgrade API ------------------------------------------------------------

    pub fn upgrade_tower(&mut self, tower_id: u32, upgrade_name: &str) -> bool {
        let (tower_type, u1, u2, u3, u4) = {
            let t = match self.tower_by_id(tower_id) { Some(t) => t, None => return false };
            (t.type_, t.upgrade1, t.upgrade2, t.upgrade3, t.upgrade4)
        };
        let spec = match upgrades::UPGRADES.iter().find(|(n, _)| *n == upgrade_name) {
            Some((_, s)) => s,
            None => return false,
        };
        if !upgrades::can_upgrade(tower_type, upgrade_name, u1, u2, u3, u4) { return false; }
        let price = self.price(spec.cost);
        if price > self.money { return false; }
        self.money -= price;
        let tower = self.tower_by_id_mut(tower_id).unwrap();
        tower.spent_on_me += price;
        apply_upgrade(tower, upgrade_name, spec);
        true
    }

    pub fn upgrade_path(&mut self, tower_id: u32, path: u8) -> bool {
        let (tt, u1, u2, u3, u4) = {
            let t = match self.tower_by_id(tower_id) { Some(t) => t.clone(), None => return false };
            (t.type_, t.upgrade1, t.upgrade2, t.upgrade3, t.upgrade4)
        };
        match upgrades::next_path_upgrade(tt, path, u1, u2, u3, u4) {
            Some(n) => self.upgrade_tower(tower_id, n),
            None => false,
        }
    }

    pub fn available_upgrades(&self, tower_id: u32) -> HashMap<u8, Option<(String, u32)>> {
        let mut out = HashMap::new();
        let tower = match self.tower_by_id(tower_id) {
            Some(t) => t,
            None => { out.insert(1u8, None); out.insert(2u8, None); return out; }
        };
        for &path in &[1u8, 2u8] {
            let n = upgrades::next_path_upgrade(
                tower.type_, path, tower.upgrade1, tower.upgrade2, tower.upgrade3, tower.upgrade4,
            );
            let entry = n.and_then(|name| {
                let cost = upgrades::UPGRADES.iter().find(|(n, _)| *n == name)?.1.cost;
                Some((name.to_string(), self.price(cost)))
            });
            out.insert(path, entry);
        }
        out
    }

    // -- debug helpers ---------------------------------------------------------

    pub fn debug_add_money(&mut self, amount: i32) {
        if amount >= 0 { self.money += amount as u32; }
        else { self.money = self.money.saturating_sub(amount.unsigned_abs() as u32); }
    }

    pub fn debug_add_lives(&mut self, amount: i32) {
        if amount >= 0 { self.lives += amount as u32; }
        else { self.lives = self.lives.saturating_sub(amount.unsigned_abs() as u32); }
        if self.lives > 0 && self.game_over && !self.won { self.game_over = false; }
    }

    pub fn debug_set_round(&mut self, round_num: u16) -> bool {
        if self.in_round { return false; }
        self.round = round_num.saturating_sub(1).min(self.max_round).max(0);
        true
    }

    pub fn debug_clear_bloons(&mut self) {
        for b in &mut self.bloons { b.popped = true; }
    }

    // -- internal helpers -------------------------------------------------------

    pub fn price(&self, base_cost: u32) -> u32 {
        // Python's round() uses banker's rounding (round half to even),
        // while Rust's f64::round() uses round half away from zero.
        // We implement Python's semantics to match exactly.
        let v = base_cost as f64 * self.cost_mult / 5.0;
        let v_floor = v.floor();
        let frac = v - v_floor;
        let rounded = if frac > 0.5 {
            v_floor + 1.0
        } else if frac < 0.5 {
            v_floor
        } else {
            // Exactly 0.5: round to even
            if v_floor as i64 % 2 == 0 { v_floor } else { v_floor + 1.0 }
        };
        (rounded as u32) * 5
    }

    fn round_data(&mut self, round_num: u16) -> Vec<u8> {
        self.round_table.get(&round_num).cloned().unwrap_or_default()
    }

    fn round_interval(&self, round_num: u16) -> u16 {
        let v = 20i32 - round_num as i32;
        let v = if v < 7 { (7.0 - round_num as f64 / 20.0).ceil() as i32 } else { v };
        v.max(1) as u16
    }

    fn tower_by_id(&self, tower_id: u32) -> Option<&Tower> {
        self.towers.iter().find(|t| t.id == tower_id)
    }

    fn tower_by_id_mut(&mut self, tower_id: u32) -> Option<&mut Tower> {
        self.towers.iter_mut().find(|t| t.id == tower_id)
    }

    fn tower_has_leadbreak(&self, tt: TowerType, u4: bool) -> bool {
        // Bomb always leadbreak. Super upgrade4 adds leadbreak (laser/plasma).
        // Boomerang upgrade4 adds leadbreak via glaive.
        if tt == TowerType::Bomb { return true; }
        if tt == TowerType::Super && u4 { return true; }
        // U4 boomerang has glaive which adds leadbreak.
        if tt == TowerType::Boomerang && u4 { return true; }
        false
    }

    fn tower_has_icebreak(&self, tt: TowerType, u3: bool, u4: bool) -> bool {
        // Bomb always icebreak.
        if tt == TowerType::Bomb { return true; }
        // Super upgrade3 (laser) and upgrade4 (plasma) add icebreak.
        if tt == TowerType::Super && (u3 || u4) { return true; }
        // Boomerang upgrade3 adds icebreak.
        if tt == TowerType::Boomerang && u3 { return true; }
        false
    }

    fn lookup_position(&self, branch: u8, frame: f64) -> (f64, f64) {
        let p = self.paths.get(branch);
        let idx = (frame.round() as usize).min(self.path_max_idx[&branch]);
        (p[idx][0], p[idx][1])
    }

    // -- spawn ------------------------------------------------------------------

    fn tick_spawns(&mut self) {
        if self.spawn_queue.is_empty() { return; }
        self.spawn_counter += 1;
        if self.spawn_counter > self.bloon_interval {
            self.spawn_counter = 0;
            let rank = self.spawn_queue.remove(0);
            self.spawn_bloon(rank, 1, 0.0, None);
            self.frames_since_last_bloon = 0;
        }
    }

    fn spawn_bloon(&mut self, rank: u8, branch: u8, frame: f64, jitter: Option<(f64, f64)>) {
        let (jx, jy) = match jitter {
            Some(j) => j,
            None => (self.rng.int(10) as f64, self.rng.int(10) as f64),
        };
        let maxspeed = BLOON_MAXSPEED[rank as usize] + self.glob_speed_mod;
        let mut bloon = Bloon::new(rank, frame.max(0.0), maxspeed, maxspeed,
                                    jx, jy, branch, BLOON_RADIUS[rank as usize]);
        // Set position
        let (x, y) = self.lookup_position(bloon.branch, bloon.frame);
        bloon.x = x + bloon.jitter_x;
        bloon.y = y + bloon.jitter_y;
        self.bloons.push(bloon);
    }

    // -- tower tick --------------------------------------------------------------

    fn tick_towers(&mut self) {
        self.refresh_beacon_buffs();
        for i in 0..self.towers.len() {
            // Read fields BEFORE mutable borrow.
            let (rate, is_attacker, rate_active, is_spread, radius_in, beacon_active,
                 x, y, tt, id, freeze, scale, pierce) = {
                let t = &self.towers[i];
                (t.attack_rate, t.is_attacker, t.beacon_rate_active,
                 t.is_spread, t.attack_radius, t.beacon_radius_active,
                 t.x, t.y, t.type_, t.id, t.freeze_len, t.bullet_scale, t.pierce_max)
            };

            self.towers[i].time_since_last_shot += 1;

            if !is_attacker { continue; }

            let effective = if rate_active { ((rate as f64 * BEACON_RATE_FACTOR).ceil().max(1.0)) as u16 } else { rate };
            if self.towers[i].time_since_last_shot <= effective { continue; }

            // Read upgrade flags for this tower.
            let (actual_icebreak, actual_leadbreak) = {
                let t = &self.towers[i];
                let ib = t.icebreak || self.tower_has_icebreak(tt, t.upgrade3, t.upgrade4);
                let lb = t.leadbreak || self.tower_has_leadbreak(tt, t.upgrade4);
                (ib, lb)
            };

            // Acquire target using the stored fields.
            let mut ar_sq = radius_in * radius_in;
            if beacon_active { ar_sq *= BEACON_RANGE_FACTOR; }
            let target_idx = self.acquire_target_idx(ar_sq, x, y, actual_icebreak);

            if target_idx.is_none() { continue; }
            let target_idx = target_idx.unwrap();

            self.towers[i].time_since_last_shot = 0;

            if is_spread {
                self.shoot_spread(x, y, tt, self.towers[i].shoot_power, id,
                                   actual_icebreak, actual_leadbreak, freeze, scale);
            } else {
                let tgt_x = self.bloons[target_idx].x;
                let tgt_y = self.bloons[target_idx].y;
                self.shoot(x, y, tt, self.towers[i].shoot_power, id,
                            actual_icebreak, actual_leadbreak, freeze, scale, pierce,
                            tgt_x, tgt_y);
            }
        }
    }

    fn refresh_beacon_buffs(&mut self) {
        for t in &mut self.towers {
            if t.type_ != TowerType::Beacon {
                t.beacon_radius_active = false;
                t.beacon_rate_active = false;
            }
        }
        // Collect beacon data first to avoid borrow conflicts.
        let beacon_data: Vec<(f64, f64, f64)> = self.towers.iter()
            .filter(|t| t.type_ == TowerType::Beacon)
            .map(|t| (t.x, t.y, t.attack_radius))
            .collect();

        for (bx, by, radius) in &beacon_data {
            let ar_sq = radius * radius;
            for t in &mut self.towers {
                if t.type_ == TowerType::Beacon { continue; }
                let dx = t.x - bx;
                let dy = t.y - by;
                if dx * dx + dy * dy < ar_sq {
                    t.beacon_radius_active = true;
                }
            }
        }
    }

    fn acquire_target_idx(&self, ar_sq: f64, tx: f64, ty: f64, icebreak: bool) -> Option<usize> {
        let mut best = None;
        let mut best_progress = -1.0_f64;
        for (i, b) in self.bloons.iter().enumerate() {
            if b.popped || b.escaped { continue; }
            if b.frozen && !icebreak { continue; }
            let dx = b.x - tx;
            let dy = b.y - ty;
            if dx * dx + dy * dy >= ar_sq { continue; }
            let progress = b.frame / self.path_len_clamped[&b.branch] as f64;
            if progress > best_progress {
                best_progress = progress;
                best = Some(i);
            }
        }
        best
    }

    fn shoot_spread(&mut self, tx: f64, ty: f64, tt: TowerType, shoot_power: f64,
                     shooter_id: u32, icebreak: bool, leadbreak: bool,
                     freeze_len: u16, scale: f64) {
        let bt = tower_bullet_type(tt);
        for i in 0..SPREAD_SHARDS {
            let angle = (2.0 * consts::PI * i as f64) / SPREAD_SHARDS as f64;
            let (ux, uy) = (angle.cos(), angle.sin());
            self.bullets.push(Bullet::from_type(
                bt, tx + ux * 10.0, ty + uy * 10.0,
                ux * shoot_power, uy * shoot_power, 1, shooter_id,
                icebreak, leadbreak, freeze_len, scale,
            ));
        }
    }

    fn shoot(&mut self, tx: f64, ty: f64, tt: TowerType, shoot_power: f64,
              shooter_id: u32, icebreak: bool, leadbreak: bool,
              freeze_len: u16, scale: f64, pierce_max: u16,
              target_x: f64, target_y: f64) {
        let dx = target_x - tx;
        let dy = target_y - ty;
        let dist = dx.hypot(dy).max(1.0);
        let (ux, uy) = (dx / dist, dy / dist);
        let bt = tower_bullet_type(tt);

        if tt == TowerType::Boomerang {
            if let Some(ref arc) = self.boomerang_arc {
                let angle = ux.atan2(-uy);
                let (c, s) = (angle.cos(), angle.sin());
                let (x0, y0) = (tx + arc[0][0] * c - arc[0][1] * s,
                                ty + arc[0][0] * s + arc[0][1] * c);
                let mut b = Bullet::from_type(bt, x0, y0, 0.0, 0.0, pierce_max, shooter_id,
                                               icebreak, leadbreak, freeze_len, scale);
                b.arc_anchor_x = tx;
                b.arc_anchor_y = ty;
                b.arc_angle = angle;
                self.bullets.push(b);
                return;
            }
        }

        self.bullets.push(Bullet::from_type(
            bt, tx + ux * 10.0, ty + uy * 10.0,
            ux * shoot_power, uy * shoot_power, pierce_max, shooter_id,
            icebreak, leadbreak, freeze_len, scale,
        ));
    }

    // -- bullet / bloon ticks ----------------------------------------------------

    fn tick_bullets(&mut self) {
        for i in 0..self.bullets.len() {
            let b = &mut self.bullets[i];
            if b.is_dead { continue; }
            b.time_alive += 1;
            if b.time_alive > b.lifespan { b.is_dead = true; continue; }
            if b.type_ == BulletType::Boomerang {
                if let Some(ref arc) = self.boomerang_arc {
                    let idx = (b.time_alive as usize).min(arc.len() - 1);
                    let (lx, ly) = (arc[idx][0], arc[idx][1]);
                    let (c, s) = (b.arc_angle.cos(), b.arc_angle.sin());
                    b.x = b.arc_anchor_x + lx * c - ly * s;
                    b.y = b.arc_anchor_y + lx * s + ly * c;
                    continue;
                }
            }
            b.x += b.vx;
            b.y += b.vy;
        }
    }

    fn tick_bloons(&mut self) {
        let mut escapes: Vec<(usize, u32)> = Vec::new();
        for i in 0..self.bloons.len() {
            let b = &mut self.bloons[i];
            if b.popped || b.escaped { continue; }
            b.hit_this_frame = false;
            if b.frozen {
                b.time_frozen += 1;
                if b.time_frozen > b.freeze_duration {
                    b.frozen = false;
                    b.time_frozen = 0;
                }
                // Refresh position via inline lookup (no self borrow conflict).
                let p = self.paths.get(b.branch);
                let idx = (b.frame.round() as usize).min(self.path_max_idx[&b.branch]);
                b.x = p[idx][0] + b.jitter_x;
                b.y = p[idx][1] + b.jitter_y;
                continue;
            }
            b.frame += b.speed;
            if b.frame.round() as usize >= self.path_len[&b.branch] {
                b.escaped = true;
                escapes.push((i, BLOON_ESCAPE_DAMAGE[b.rank as usize]));
                continue;
            }
            // Refresh position inline.
            let p = self.paths.get(b.branch);
            let idx = (b.frame.round() as usize).min(self.path_max_idx[&b.branch]);
            b.x = p[idx][0] + b.jitter_x;
            b.y = p[idx][1] + b.jitter_y;
        }
        // Process escapes (after the mutable borrow of bloons is released).
        for (_, dmg) in escapes {
            if self.in_round {
                if dmg >= self.lives {
                    self.lives = 0;
                    self.game_over = true;
                    self.won = false;
                } else {
                    self.lives -= dmg;
                }
            }
        }
    }

    // -- collision detection ----------------------------------------------------

    fn tick_collisions(&mut self) {
        // We must be careful with borrows: on_hit may mutate bullets[] radii
        // (bomb two-stage) and bloons[] state (pop/freeze).
        // Strategy: collect hit events as (bullet_idx, bloon_idx) pairs,
        // process them. But bomb radius change mid-loop affects subsequent
        // collision checks in the same frame, so we MUST process inline.
        //
        // Use raw index access to satisfy the borrow checker: split the struct
        // accesses by never holding two mutable references simultaneously.

        for bi in 0..self.bullets.len() {
            if self.bullets[bi].is_dead { continue; }

            let mut bullet_exhausted = false;

            for bj in 0..self.bloons.len() {
                // Read bloon state (immutable borrow)
                let (b_popped, b_escaped, b_hit, bx, by, br) = {
                    let bl = &self.bloons[bj];
                    (bl.popped, bl.escaped, bl.hit_this_frame, bl.x, bl.y, bl.radius)
                };
                if b_popped || b_escaped || b_hit { continue; }

                // Read bullet state (immutable borrow)
                let (bul_x, bul_y, bul_rad) = {
                    let bu = &self.bullets[bi];
                    (bu.x, bu.y, bu.radius)
                };

                let dx = bul_x - bx;
                let dy = bul_y - by;
                let r_sum = bul_rad + br;
                if dx * dx + dy * dy > r_sum * r_sum { continue; }

                // HIT! Mark bloon hit and increment pierce.
                self.bloons[bj].hit_this_frame = true;
                self.bullets[bi].pierce_count += 1;

                // Process hit effects (may mutate bloon/bullet state).
                self.on_hit(bi, bj);

                if self.bullets[bi].pierce_count >= self.bullets[bi].pierce_max {
                    bullet_exhausted = true;
                    break;
                }
            }

            if bullet_exhausted {
                self.bullets[bi].is_dead = true;
            }
        }
    }

    // -- hit processing ----------------------------------------------------------

    fn on_hit(&mut self, bullet_idx: usize, bloon_idx: usize) {
        // Extract ALL bullet data up front.
        let b_type = self.bullets[bullet_idx].type_;
        let b_leadbreak = self.bullets[bullet_idx].leadbreak;
        let b_icebreak = self.bullets[bullet_idx].icebreak;
        let b_hashit = self.bullets[bullet_idx].hashit;
        let b_explosion_rad = self.bullets[bullet_idx].explosion_radius;
        let b_shooter = self.bullets[bullet_idx].shooter_id;
        let b_x = self.bullets[bullet_idx].x;
        let b_y = self.bullets[bullet_idx].y;
        let b_freeze_len = self.bullets[bullet_idx].freeze_len;

        // Extract bloon data.
        let bloon_rank = self.bloons[bloon_idx].rank;
        let bloon_frozen = self.bloons[bloon_idx].frozen;

        // --- Step 1: Lead clink ---
        if bloon_rank == 7 && !b_leadbreak && b_type != BulletType::Ice {
            self.bullets[bullet_idx].pierce_count += 4;
            return;
        }

        // --- Step 2: Bomb two-stage ---
        if b_type == BulletType::Bomb && !b_hashit {
            self.bullets[bullet_idx].hashit = true;
            self.bullets[bullet_idx].vx = 0.0;
            self.bullets[bullet_idx].vy = 0.0;
            if b_explosion_rad > 0.0 {
                self.bullets[bullet_idx].radius = b_explosion_rad;
            }
            // Frag check: get upgrade2 status from shooter before any mutation.
            let has_frags = self.tower_by_id(b_shooter).map_or(false, |t| t.upgrade2);
            if has_frags {
                self.spawn_frags(b_x, b_y, b_shooter);
            }
        }

        // --- Step 3: Frozen clink ---
        if bloon_frozen && !b_icebreak && b_type != BulletType::Ice {
            return;
        }

        // --- Step 4: Black bomb-immunity ---
        if (b_type == BulletType::Bomb || b_type == BulletType::Frag) && bloon_rank == 5 {
            return;
        }

        // --- Step 5: Ice freeze ---
        if b_type == BulletType::Ice {
            self.try_freeze(bloon_idx, b_shooter, b_freeze_len);
            return;
        }

        // --- Step 6: Pop ---
        self.bloons[bloon_idx].hits_remaining -= 1;
        if self.bloons[bloon_idx].hits_remaining <= 0 {
            self.pop(bloon_idx, b_shooter);
        }
    }

    fn try_freeze(&mut self, bloon_idx: usize, shooter_id: u32, freeze_len: u16) {
        let rank = self.bloons[bloon_idx].rank;
        if self.bloons[bloon_idx].frozen { return; }
        if rank == 6 || rank == 9 || rank == 10 { return; }

        self.bloons[bloon_idx].frozen = true;
        self.bloons[bloon_idx].time_frozen = 0;
        self.bloons[bloon_idx].freeze_duration = freeze_len.min(100);
        self.bloons[bloon_idx].freezer_id = shooter_id as i32;

        // Get shooter upgrade info upfront.
        let (has_permafrost, has_snap_freeze) = self.tower_by_id(shooter_id)
            .map(|t| (t.upgrade2, t.upgrade4))
            .unwrap_or((false, false));

        if has_permafrost {
            if self.bloons[bloon_idx].speed == self.bloons[bloon_idx].maxspeed && rank != 10 {
                self.bloons[bloon_idx].speed /= 2.0;
            }
        }

        if has_snap_freeze && self.rng.int_100() > 60 {
            self.bloons[bloon_idx].snap_frozen = true;
            self.bloons[bloon_idx].hits_remaining -= 1;
            if self.bloons[bloon_idx].hits_remaining <= 0 {
                self.pop(bloon_idx, shooter_id);
            }
        }
    }

    fn spawn_frags(&mut self, x: f64, y: f64, shooter_id: u32) {
        for i in 0..SPREAD_SHARDS {
            let angle = (2.0 * consts::PI * i as f64) / SPREAD_SHARDS as f64;
            let (ux, uy) = (angle.cos(), angle.sin());
            self.bullets.push(Bullet::from_type(
                BulletType::Frag, x + ux * 4.0, y + uy * 4.0,
                ux * 10.0, uy * 10.0, 1, shooter_id, false, false, 0, 1.0,
            ));
        }
    }

    fn pop(&mut self, bloon_idx: usize, shooter_id: u32) {
        self.bloons[bloon_idx].popped = true;
        self.bloons_popped_this_round += 1;
        self.award_pop_money();

        // Credit tower.
        if let Some(t) = self.towers.iter_mut().find(|t| t.id == shooter_id) {
            t.pop_count += 1;
        }

        // Capture parent bloon state.
        let parent_rank = self.bloons[bloon_idx].rank;
        let parent_frame = self.bloons[bloon_idx].frame;
        let parent_branch = self.bloons[bloon_idx].branch;
        let parent_jx = self.bloons[bloon_idx].jitter_x;
        let parent_jy = self.bloons[bloon_idx].jitter_y;
        let parent_snapped = self.bloons[bloon_idx].snap_frozen;
        let parent_freeze_dur = self.bloons[bloon_idx].freeze_duration;
        let parent_freezer = self.bloons[bloon_idx].freezer_id;

        // Check if freezer has permafrost (for inheritance).
        let freezer_has_permafrost = if parent_freezer >= 0 {
            self.tower_by_id(parent_freezer as u32).map_or(false, |t| t.upgrade2)
        } else { false };

        // Spawn children.
        if let Some((_, children_list)) = BLOON_CHILDREN.iter().find(|&&(r, _)| r == parent_rank) {
            for &(child_rank, frame_offset) in *children_list {
                let child_frame = parent_frame + frame_offset as f64;
                self.spawn_bloon(child_rank, parent_branch, child_frame,
                                  Some((parent_jx, parent_jy)));

                // Snap-freeze inheritance.
                if parent_snapped && child_rank != 6 && child_rank != 9 && child_rank != 10 {
                    if let Some(child) = self.bloons.last_mut() {
                        child.frozen = true;
                        child.time_frozen = 0;
                        child.freeze_duration = parent_freeze_dur;
                        child.freezer_id = parent_freezer;
                        if freezer_has_permafrost && child.speed == child.maxspeed && child_rank != 10 {
                            child.speed /= 2.0;
                        }
                    }
                }
            }
        }
    }

    fn award_pop_money(&mut self) {
        if self.round < 51 { self.money += 1; }
        else if self.round < 60 { if self.rng.int_3() == 0 { self.money += 1; } }
        else if self.rng.int_5() == 0 { self.money += 1; }
    }

    // -- round end / cleanup ----------------------------------------------------

    fn tick_round_end(&mut self) {
        if !self.in_round || !self.spawn_queue.is_empty() {
            self.frames_since_last_bloon = 0;
            return;
        }
        if self.bloons.iter().any(|b| b.alive()) {
            self.frames_since_last_bloon = 0;
            self.end_round_count = 0;
            return;
        }
        self.end_round_count += 1;
        self.frames_since_last_bloon += 1;
        if self.end_round_count > ROUND_END_GRACE_FRAMES
            || self.frames_since_last_bloon > ROUND_END_TIMEOUT_FRAMES {
            self.finish_round();
        }
    }

    fn finish_round(&mut self) {
        self.in_round = false;
        for b in &mut self.bullets { b.is_dead = true; }
        let win_round = if self.config.freeplay { self.max_round } else { 50 };
        if self.round >= win_round {
            self.game_over = true;
            self.won = true;
            return;
        }
        self.money += 99 + self.round as u32;
    }

    fn cleanup(&mut self) {
        if !self.bullets.is_empty() { self.bullets.retain(|b| !b.is_dead); }
        if !self.bloons.is_empty() { self.bloons.retain(|b| b.alive()); }
    }
}
