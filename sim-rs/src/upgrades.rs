/// Tower upgrade definitions. Ports BloonsTD.GetUpgrade() and sim/btd/upgrades.py.
///
/// Two paths of two upgrades each:
///   Path 1: upgrade1 -> upgrade2  (flags upgrade1, upgrade2)
///   Path 2: upgrade3 -> upgrade4  (flags upgrade3, upgrade4)
/// Paths are fully independent — both can be maxed in any interleaved order.

use crate::constants::TowerType;

#[derive(Clone, Debug)]
pub(crate) struct UpgradeSpec {
    pub cost: u32,
    /// Stat deltas applied with +=.
    pub additive: &'static [(&'static str, f64)],
    /// Stat values applied with =.
    pub absolute: &'static [(&'static str, f64)],
    /// Boolean flags flipped to True.
    pub flags: &'static [(&'static str, bool)],
    /// Transform upgrades reset the firing cooldown.
    pub reset_tsls: bool,
}

// All upgrades keyed by name. Same structure as Python's UPGRADES dict.
pub(crate) const UPGRADES: &[(&str, UpgradeSpec)] = &[
    // Dart Monkey
    ("dart1", UpgradeSpec { cost: 90, additive: &[("attack_radius", 25.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("dart2", UpgradeSpec { cost: 90, additive: &[("attack_radius", 25.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("dart3", UpgradeSpec { cost: 140, additive: &[("pierce_max", 1.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("dart4", UpgradeSpec { cost: 120, additive: &[("pierce_max", 1.0)], absolute: &[], flags: &[], reset_tsls: false }),

    // Tack Shooter
    ("tack1", UpgradeSpec { cost: 200, additive: &[("attack_rate", -15.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("tack2", UpgradeSpec { cost: 180, additive: &[("attack_rate", -5.0)], absolute: &[], flags: &[("transformed", true)], reset_tsls: true }),
    ("tack3", UpgradeSpec { cost: 100, additive: &[("attack_radius", 10.0)], absolute: &[("bullet_scale", 1.3)], flags: &[], reset_tsls: false }),
    ("tack4", UpgradeSpec { cost: 100, additive: &[("attack_radius", 10.0)], absolute: &[("bullet_scale", 1.5)], flags: &[], reset_tsls: false }),

    // Boomerang
    ("boomerang1", UpgradeSpec { cost: 270, additive: &[("pierce_max", 3.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("boomerang2", UpgradeSpec { cost: 280, additive: &[("pierce_max", 3.0)], absolute: &[], flags: &[("transformed", true)], reset_tsls: true }),
    ("boomerang3", UpgradeSpec { cost: 150, additive: &[], absolute: &[], flags: &[("icebreak", true)], reset_tsls: false }),
    ("boomerang4", UpgradeSpec { cost: 120, additive: &[], absolute: &[], flags: &[("leadbreak", true)], reset_tsls: false }),

    // Bomb (Cannon)
    ("bomb1", UpgradeSpec { cost: 430, additive: &[], absolute: &[("bullet_scale", 1.5)], flags: &[], reset_tsls: false }),
    ("bomb2", UpgradeSpec { cost: 220, additive: &[], absolute: &[], flags: &[], reset_tsls: false }),
    ("bomb3", UpgradeSpec { cost: 200, additive: &[("attack_radius", 20.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("bomb4", UpgradeSpec { cost: 210, additive: &[("attack_rate", -8.0)], absolute: &[("shoot_power", 25.0)], flags: &[("transformed", true)], reset_tsls: true }),

    // Ice Ball
    ("ice1", UpgradeSpec { cost: 250, additive: &[("freeze_len", 20.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("ice2", UpgradeSpec { cost: 250, additive: &[], absolute: &[], flags: &[], reset_tsls: false }),
    ("ice3", UpgradeSpec { cost: 200, additive: &[("attack_radius", 15.0)], absolute: &[("bullet_scale", 1.0)], flags: &[], reset_tsls: false }),
    ("ice4", UpgradeSpec { cost: 290, additive: &[], absolute: &[], flags: &[], reset_tsls: false }),

    // Super Monkey
    ("super1", UpgradeSpec { cost: 1000, additive: &[("attack_radius", 50.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("super2", UpgradeSpec { cost: 1400, additive: &[("attack_radius", 50.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("super3", UpgradeSpec { cost: 3500, additive: &[("pierce_max", 1.0)], absolute: &[], flags: &[("icebreak", true), ("laser", true)], reset_tsls: false }),
    ("super4", UpgradeSpec { cost: 4000, additive: &[("pierce_max", 1.0)], absolute: &[("attack_rate", 1.0)], flags: &[("icebreak", true), ("leadbreak", true), ("laser", true)], reset_tsls: false }),

    // Spike-o-pult
    ("spikeopult1", UpgradeSpec { cost: 250, additive: &[("attack_radius", 20.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("spikeopult2", UpgradeSpec { cost: 825, additive: &[], absolute: &[("pierce_max", 20.0)], flags: &[], reset_tsls: false }),
    ("spikeopult3", UpgradeSpec { cost: 250, additive: &[("attack_rate", -8.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("spikeopult4", UpgradeSpec { cost: 575, additive: &[], absolute: &[], flags: &[("transformed", true), ("is_spread", true)], reset_tsls: true }),

    // Beacon (drums only; storm upgrades deferred)
    ("beacon1", UpgradeSpec { cost: 500, additive: &[("attack_radius", 30.0)], absolute: &[], flags: &[], reset_tsls: false }),
    ("beacon2", UpgradeSpec { cost: 1500, additive: &[], absolute: &[], flags: &[], reset_tsls: false }),
];

/// Return the upgrade name the given path button would buy next, or None
/// if the path is fully maxed.
pub(crate) fn next_path_upgrade(
    tower_type: TowerType,
    path: u8,
    u1: bool, u2: bool, u3: bool, u4: bool,
) -> Option<&'static str> {
    let prefix = tower_type.name();
    if path == 1 {
        if !u1 {
            // e.g. "dart1"
            // We need to construct the name. Since UPGRADES is keyed by name,
            // we can look it up directly. But we need a static string.
            // The upgrade name is "{prefix}{suffix}" where suffix is "1" or "2".
            // We iterate UPGRADES to find matching entries.
            return find_upgrade_name(prefix, "1");
        }
        if !u2 {
            return find_upgrade_name(prefix, "2");
        }
        return None;
    }
    if path == 2 {
        if !u3 {
            return find_upgrade_name(prefix, "3");
        }
        if !u4 {
            return find_upgrade_name(prefix, "4");
        }
        return None;
    }
    None
}

fn find_upgrade_name(prefix: &str, suffix: &str) -> Option<&'static str> {
    // Linear scan avoids allocation. UPGRADES is small (≤32 entries).
    let expected_len = prefix.len() + suffix.len();
    UPGRADES.iter()
        .find(|(name, _)| name.len() == expected_len
              && name.starts_with(prefix)
              && name.ends_with(suffix))
        .map(|(name, _)| *name)
}

/// Check whether an upgrade can be bought (prerequisite flags).
pub(crate) fn can_upgrade(tower_type: TowerType, upgrade_name: &str, u1: bool, u2: bool, u3: bool, u4: bool) -> bool {
    let prefix = tower_type.name();
    if !upgrade_name.starts_with(prefix) {
        return false;
    }
    let suffix = &upgrade_name[prefix.len()..];
    match suffix {
        "1" => !u1,
        "2" => u1 && !u2,
        "3" => !u3,
        "4" => u3 && !u4,
        _ => false,
    }
}
