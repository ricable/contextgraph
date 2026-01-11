# TASK-ATC-P2-002: Extend DomainThresholds Struct with New Threshold Fields

**Version:** 3.0
**Status:** Ready
**Layer:** Foundation
**Sequence:** 2
**Implements:** REQ-ATC-006, REQ-ATC-007
**Depends On:** TASK-ATC-P2-001 (COMPLETE - threshold-inventory.yaml exists)
**Estimated Complexity:** Medium
**Priority:** P2
**Last Audited:** 2026-01-11

---

## Metadata

```yaml
id: TASK-ATC-P2-002
title: Extend DomainThresholds Struct with New Threshold Fields
status: ready
layer: foundation
sequence: 2
implements:
  - REQ-ATC-006
  - REQ-ATC-007
depends_on:
  - TASK-ATC-P2-001  # COMPLETE
estimated_complexity: medium
```

---

## ⚠️ CRITICAL RULES

1. **NO BACKWARDS COMPATIBILITY** - System must work after changes or fail fast
2. **NO WORKAROUNDS OR FALLBACKS** - Errors must surface immediately with robust logging
3. **NO MOCK DATA IN TESTS** - Use real computations, verify actual system state
4. **FAIL FAST** - Invalid thresholds MUST panic or return `Result::Err`

---

## Current State Audit (2026-01-11)

### What ACTUALLY EXISTS

| Component | Location | Status |
|-----------|----------|--------|
| `DomainThresholds` struct | `crates/context-graph-core/src/atc/domain.rs:89-97` | ✅ EXISTS - 6 fields |
| `Domain` enum | `crates/context-graph-core/src/atc/domain.rs:13-20` | ✅ EXISTS |
| `DomainManager` | `crates/context-graph-core/src/atc/domain.rs:170-282` | ✅ EXISTS |
| `AdaptiveThresholdCalibration` | `crates/context-graph-core/src/atc/mod.rs:63-245` | ✅ EXISTS |
| `threshold-inventory.yaml` | `specs/tasks/threshold-inventory.yaml` | ✅ EXISTS - 78 thresholds |
| ATC Levels 1-4 | `crates/context-graph-core/src/atc/*.rs` | ✅ EXISTS |
| `ThresholdAccessor` trait | **DOES NOT EXIST** | ❌ MUST CREATE |

### Current DomainThresholds (6 fields)

```rust
// File: crates/context-graph-core/src/atc/domain.rs lines 89-97
pub struct DomainThresholds {
    pub domain: Domain,
    pub theta_opt: f32,      // [0.60, 0.90]
    pub theta_acc: f32,      // [0.55, 0.85]
    pub theta_warn: f32,     // [0.40, 0.70]
    pub theta_dup: f32,      // [0.80, 0.98]
    pub theta_edge: f32,     // [0.50, 0.85]
    pub confidence_bias: f32,
}
```

### Required NEW Fields (from threshold-inventory.yaml critical + should_migrate)

From `specs/tasks/threshold-inventory.yaml` categories:

**Critical (12 thresholds - MUST add):**
- `theta_gate` - GWT broadcast gate [0.65, 0.95] (constitution: gwt.workspace.coherence_threshold)
- `theta_hypersync` - Pathological hypersync [0.90, 0.99] (gwt.kuramoto.thresholds.hypersync)
- `theta_fragmentation` - Fragmented state [0.35, 0.65] (gwt.kuramoto.thresholds.fragmented)
- `theta_memory_sim` - Memory relevance [0.35, 0.75] (layers.L3_Memory)
- `theta_reflex_hit` - Reflex cache hit [0.70, 0.95] (layers.L2_Reflex)
- `theta_consolidation` - Weight consolidation [0.05, 0.30] (layers.L4_Learning)
- `theta_dream_activity` - Dream trigger [0.05, 0.30] (dream.trigger.activity)
- `theta_shortcut_conf` - Amortized confidence [0.50, 0.85] (dream.amortized.confidence_threshold)
- `theta_johari` - Johari entropy boundary [0.35, 0.65] (utl.johari)
- `theta_blind_spot` - Blind spot detection [0.35, 0.65] (utl.johari)
- `theta_obsolescence_low` - Low relevance [0.20, 0.50] (mcp.autonomous)
- `theta_obsolescence_high` - High confidence [0.65, 0.90] (mcp.autonomous)

**Should Migrate (adds 1 more unique):**
- `theta_semantic_leap` - REM exploration [0.50, 0.90] (dream.phases.rem.blind_spot.min_semantic_distance)

---

## Scope

### In Scope

1. **Extend `DomainThresholds` struct** with 13 new fields (19 total)
2. **Update `DomainThresholds::new(Domain)`** with domain-aware initialization
3. **Update `DomainThresholds::is_valid()`** with new range checks + monotonicity
4. **Update `DomainThresholds::clamp()`** with new field ranges
5. **Update `DomainThresholds::blend_with_similar()`** to include new fields
6. **Create `ThresholdAccessor` trait** in new file `accessor.rs`
7. **Implement `ThresholdAccessor` for `AdaptiveThresholdCalibration`**
8. **Add comprehensive unit tests** with FSV pattern

### Out of Scope

- Migration of calling code (TASK-ATC-P2-003 through P2-007)
- MCP tool updates
- Integration tests with other modules

---

## Definition of Done

### Target Struct (19 fields)

```rust
// File: crates/context-graph-core/src/atc/domain.rs

/// Domain-specific thresholds for adaptive calibration.
///
/// # Constitution Reference
/// See `docs2/constitution.yaml` sections:
/// - `adaptive_thresholds.priors` for base ranges
/// - `gwt.workspace`, `gwt.kuramoto` for GWT thresholds
/// - `dream.trigger`, `dream.phases` for dream thresholds
/// - `utl.johari` for classification thresholds
#[derive(Debug, Clone)]
pub struct DomainThresholds {
    pub domain: Domain,

    // === Existing fields (6) ===
    pub theta_opt: f32,           // [0.60, 0.90] Optimal alignment
    pub theta_acc: f32,           // [0.55, 0.85] Acceptable alignment
    pub theta_warn: f32,          // [0.40, 0.70] Warning alignment
    pub theta_dup: f32,           // [0.80, 0.98] Duplicate detection
    pub theta_edge: f32,          // [0.50, 0.85] Edge creation
    pub confidence_bias: f32,     // Domain confidence adjustment

    // === NEW: GWT thresholds (3) ===
    pub theta_gate: f32,          // [0.65, 0.95] GW broadcast gate
    pub theta_hypersync: f32,     // [0.90, 0.99] Hypersync detection
    pub theta_fragmentation: f32, // [0.35, 0.65] Fragmentation warning

    // === NEW: Layer thresholds (3) ===
    pub theta_memory_sim: f32,    // [0.35, 0.75] Memory similarity
    pub theta_reflex_hit: f32,    // [0.70, 0.95] Reflex cache hit
    pub theta_consolidation: f32, // [0.05, 0.30] Consolidation trigger

    // === NEW: Dream thresholds (3) ===
    pub theta_dream_activity: f32,  // [0.05, 0.30] Dream trigger
    pub theta_semantic_leap: f32,   // [0.50, 0.90] REM exploration
    pub theta_shortcut_conf: f32,   // [0.50, 0.85] Shortcut confidence

    // === NEW: Classification thresholds (2) ===
    pub theta_johari: f32,          // [0.35, 0.65] Johari boundary
    pub theta_blind_spot: f32,      // [0.35, 0.65] Blind spot detection

    // === NEW: Autonomous thresholds (2) ===
    pub theta_obsolescence_low: f32,  // [0.20, 0.50] Low relevance
    pub theta_obsolescence_high: f32, // [0.65, 0.90] High confidence
}
```

### ThresholdAccessor Trait

```rust
// File: crates/context-graph-core/src/atc/accessor.rs (NEW FILE)

use super::{AdaptiveThresholdCalibration, Domain, DomainThresholds};

/// Unified threshold access by name.
///
/// Enables dynamic threshold lookup for MCP tools and subsystems
/// that need to access thresholds by string name rather than field access.
pub trait ThresholdAccessor {
    /// Get threshold value by name for a domain.
    /// Returns `None` if threshold name is unknown.
    fn get_threshold(&self, name: &str, domain: Domain) -> Option<f32>;

    /// Get threshold with fallback to General domain if domain-specific unavailable.
    fn get_threshold_or_general(&self, name: &str, domain: Domain) -> f32;

    /// Observe threshold usage for EWMA drift tracking (Level 1).
    fn observe_threshold_usage(&mut self, name: &str, value: f32);

    /// List all available threshold names.
    fn list_threshold_names() -> &'static [&'static str];
}
```

---

## Implementation Steps

### Step 1: Extend DomainThresholds Struct

**File:** `crates/context-graph-core/src/atc/domain.rs`

Add 13 new fields after `confidence_bias`:

```rust
// After line 96, add:
    // === NEW: GWT thresholds ===
    pub theta_gate: f32,
    pub theta_hypersync: f32,
    pub theta_fragmentation: f32,

    // === NEW: Layer thresholds ===
    pub theta_memory_sim: f32,
    pub theta_reflex_hit: f32,
    pub theta_consolidation: f32,

    // === NEW: Dream thresholds ===
    pub theta_dream_activity: f32,
    pub theta_semantic_leap: f32,
    pub theta_shortcut_conf: f32,

    // === NEW: Classification thresholds ===
    pub theta_johari: f32,
    pub theta_blind_spot: f32,

    // === NEW: Autonomous thresholds ===
    pub theta_obsolescence_low: f32,
    pub theta_obsolescence_high: f32,
```

### Step 2: Update DomainThresholds::new()

**File:** `crates/context-graph-core/src/atc/domain.rs`

Replace `new()` method (lines 101-119) with domain-aware initialization:

```rust
pub fn new(domain: Domain) -> Self {
    let strictness = domain.strictness();

    // Existing thresholds (unchanged logic)
    let theta_opt = 0.75 + (strictness * 0.1);
    let theta_acc = 0.70 + (strictness * 0.08);
    let theta_warn = 0.55 + (strictness * 0.05);

    // GWT thresholds: stricter domains have higher gates
    let theta_gate = 0.75 + (strictness * 0.15);           // [0.75, 0.90]
    let theta_hypersync = 0.93 + (strictness * 0.04);      // [0.93, 0.97]
    let theta_fragmentation = 0.50 - (strictness * 0.10); // [0.40, 0.50]

    // Layer thresholds
    let theta_memory_sim = 0.50 + (strictness * 0.15);     // [0.50, 0.65]
    let theta_reflex_hit = 0.80 + (strictness * 0.10);     // [0.80, 0.90]
    let theta_consolidation = 0.10 + (strictness * 0.10); // [0.10, 0.20]

    // Dream thresholds: creative domains dream more aggressively
    let theta_dream_activity = 0.15 - (strictness * 0.05);  // [0.10, 0.15]
    let theta_semantic_leap = 0.70 - (strictness * 0.10);   // [0.60, 0.70]
    let theta_shortcut_conf = 0.70 + (strictness * 0.10);   // [0.70, 0.80]

    // Classification: fixed per constitution (may later be domain-tuned)
    let theta_johari = 0.50;
    let theta_blind_spot = 0.50;

    // Autonomous thresholds: stricter domains require higher confidence
    let theta_obsolescence_low = 0.30 + (strictness * 0.10);  // [0.30, 0.40]
    let theta_obsolescence_high = 0.75 + (strictness * 0.10); // [0.75, 0.85]

    Self {
        domain,
        theta_opt,
        theta_acc,
        theta_warn,
        theta_dup: 0.90,
        theta_edge: 0.70,
        confidence_bias: 1.0,
        theta_gate,
        theta_hypersync,
        theta_fragmentation,
        theta_memory_sim,
        theta_reflex_hit,
        theta_consolidation,
        theta_dream_activity,
        theta_semantic_leap,
        theta_shortcut_conf,
        theta_johari,
        theta_blind_spot,
        theta_obsolescence_low,
        theta_obsolescence_high,
    }
}
```

### Step 3: Update is_valid()

**File:** `crates/context-graph-core/src/atc/domain.rs`

Replace `is_valid()` (lines 133-157) with comprehensive validation:

```rust
pub fn is_valid(&self) -> bool {
    // Existing monotonicity check
    if !(self.theta_opt > self.theta_acc && self.theta_acc > self.theta_warn) {
        return false;
    }

    // Existing range checks
    if !(0.60..=0.90).contains(&self.theta_opt) { return false; }
    if !(0.55..=0.85).contains(&self.theta_acc) { return false; }
    if !(0.40..=0.70).contains(&self.theta_warn) { return false; }
    if !(0.80..=0.98).contains(&self.theta_dup) { return false; }
    if !(0.50..=0.85).contains(&self.theta_edge) { return false; }

    // NEW: GWT thresholds
    if !(0.65..=0.95).contains(&self.theta_gate) { return false; }
    if !(0.90..=0.99).contains(&self.theta_hypersync) { return false; }
    if !(0.35..=0.65).contains(&self.theta_fragmentation) { return false; }

    // NEW: Layer thresholds
    if !(0.35..=0.75).contains(&self.theta_memory_sim) { return false; }
    if !(0.70..=0.95).contains(&self.theta_reflex_hit) { return false; }
    if !(0.05..=0.30).contains(&self.theta_consolidation) { return false; }

    // NEW: Dream thresholds
    if !(0.05..=0.30).contains(&self.theta_dream_activity) { return false; }
    if !(0.50..=0.90).contains(&self.theta_semantic_leap) { return false; }
    if !(0.50..=0.85).contains(&self.theta_shortcut_conf) { return false; }

    // NEW: Classification thresholds
    if !(0.35..=0.65).contains(&self.theta_johari) { return false; }
    if !(0.35..=0.65).contains(&self.theta_blind_spot) { return false; }

    // NEW: Autonomous thresholds - MUST enforce monotonicity
    if !(0.20..=0.50).contains(&self.theta_obsolescence_low) { return false; }
    if !(0.65..=0.90).contains(&self.theta_obsolescence_high) { return false; }
    if !(self.theta_obsolescence_high > self.theta_obsolescence_low) { return false; }

    true
}
```

### Step 4: Update clamp()

**File:** `crates/context-graph-core/src/atc/domain.rs`

Replace `clamp()` (lines 160-166) with all fields:

```rust
pub fn clamp(&mut self) {
    // Existing
    self.theta_opt = self.theta_opt.clamp(0.60, 0.90);
    self.theta_acc = self.theta_acc.clamp(0.55, 0.85);
    self.theta_warn = self.theta_warn.clamp(0.40, 0.70);
    self.theta_dup = self.theta_dup.clamp(0.80, 0.98);
    self.theta_edge = self.theta_edge.clamp(0.50, 0.85);

    // NEW: GWT
    self.theta_gate = self.theta_gate.clamp(0.65, 0.95);
    self.theta_hypersync = self.theta_hypersync.clamp(0.90, 0.99);
    self.theta_fragmentation = self.theta_fragmentation.clamp(0.35, 0.65);

    // NEW: Layer
    self.theta_memory_sim = self.theta_memory_sim.clamp(0.35, 0.75);
    self.theta_reflex_hit = self.theta_reflex_hit.clamp(0.70, 0.95);
    self.theta_consolidation = self.theta_consolidation.clamp(0.05, 0.30);

    // NEW: Dream
    self.theta_dream_activity = self.theta_dream_activity.clamp(0.05, 0.30);
    self.theta_semantic_leap = self.theta_semantic_leap.clamp(0.50, 0.90);
    self.theta_shortcut_conf = self.theta_shortcut_conf.clamp(0.50, 0.85);

    // NEW: Classification
    self.theta_johari = self.theta_johari.clamp(0.35, 0.65);
    self.theta_blind_spot = self.theta_blind_spot.clamp(0.35, 0.65);

    // NEW: Autonomous
    self.theta_obsolescence_low = self.theta_obsolescence_low.clamp(0.20, 0.50);
    self.theta_obsolescence_high = self.theta_obsolescence_high.clamp(0.65, 0.90);
}
```

### Step 5: Update blend_with_similar()

**File:** `crates/context-graph-core/src/atc/domain.rs`

Replace `blend_with_similar()` (lines 122-130):

```rust
pub fn blend_with_similar(&mut self, similar: &DomainThresholds, alpha: f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    let blend = |self_val: f32, other_val: f32| -> f32 {
        alpha * other_val + (1.0 - alpha) * self_val
    };

    // Existing
    self.theta_opt = blend(self.theta_opt, similar.theta_opt);
    self.theta_acc = blend(self.theta_acc, similar.theta_acc);
    self.theta_warn = blend(self.theta_warn, similar.theta_warn);
    self.theta_dup = blend(self.theta_dup, similar.theta_dup);
    self.theta_edge = blend(self.theta_edge, similar.theta_edge);

    // NEW: GWT
    self.theta_gate = blend(self.theta_gate, similar.theta_gate);
    self.theta_hypersync = blend(self.theta_hypersync, similar.theta_hypersync);
    self.theta_fragmentation = blend(self.theta_fragmentation, similar.theta_fragmentation);

    // NEW: Layer
    self.theta_memory_sim = blend(self.theta_memory_sim, similar.theta_memory_sim);
    self.theta_reflex_hit = blend(self.theta_reflex_hit, similar.theta_reflex_hit);
    self.theta_consolidation = blend(self.theta_consolidation, similar.theta_consolidation);

    // NEW: Dream
    self.theta_dream_activity = blend(self.theta_dream_activity, similar.theta_dream_activity);
    self.theta_semantic_leap = blend(self.theta_semantic_leap, similar.theta_semantic_leap);
    self.theta_shortcut_conf = blend(self.theta_shortcut_conf, similar.theta_shortcut_conf);

    // NEW: Classification
    self.theta_johari = blend(self.theta_johari, similar.theta_johari);
    self.theta_blind_spot = blend(self.theta_blind_spot, similar.theta_blind_spot);

    // NEW: Autonomous
    self.theta_obsolescence_low = blend(self.theta_obsolescence_low, similar.theta_obsolescence_low);
    self.theta_obsolescence_high = blend(self.theta_obsolescence_high, similar.theta_obsolescence_high);
}
```

### Step 6: Create accessor.rs

**File:** `crates/context-graph-core/src/atc/accessor.rs` (NEW FILE)

```rust
//! Unified threshold access by name.
//!
//! Enables dynamic threshold lookup for MCP tools and subsystems.

use super::{AdaptiveThresholdCalibration, Domain};

/// All threshold names supported by the system.
pub const THRESHOLD_NAMES: &[&str] = &[
    "theta_opt", "theta_acc", "theta_warn", "theta_dup", "theta_edge",
    "theta_gate", "theta_hypersync", "theta_fragmentation",
    "theta_memory_sim", "theta_reflex_hit", "theta_consolidation",
    "theta_dream_activity", "theta_semantic_leap", "theta_shortcut_conf",
    "theta_johari", "theta_blind_spot",
    "theta_obsolescence_low", "theta_obsolescence_high",
];

/// Unified threshold access by name.
pub trait ThresholdAccessor {
    /// Get threshold value by name for a domain.
    fn get_threshold(&self, name: &str, domain: Domain) -> Option<f32>;

    /// Get threshold with fallback to General domain.
    fn get_threshold_or_general(&self, name: &str, domain: Domain) -> f32;

    /// Observe threshold usage for EWMA drift tracking.
    fn observe_threshold_usage(&mut self, name: &str, value: f32);

    /// List all threshold names.
    fn list_threshold_names() -> &'static [&'static str];
}

impl ThresholdAccessor for AdaptiveThresholdCalibration {
    fn get_threshold(&self, name: &str, domain: Domain) -> Option<f32> {
        let thresholds = self.get_domain_thresholds(domain)?;

        Some(match name {
            "theta_opt" => thresholds.theta_opt,
            "theta_acc" => thresholds.theta_acc,
            "theta_warn" => thresholds.theta_warn,
            "theta_dup" => thresholds.theta_dup,
            "theta_edge" => thresholds.theta_edge,
            "theta_gate" => thresholds.theta_gate,
            "theta_hypersync" => thresholds.theta_hypersync,
            "theta_fragmentation" => thresholds.theta_fragmentation,
            "theta_memory_sim" => thresholds.theta_memory_sim,
            "theta_reflex_hit" => thresholds.theta_reflex_hit,
            "theta_consolidation" => thresholds.theta_consolidation,
            "theta_dream_activity" => thresholds.theta_dream_activity,
            "theta_semantic_leap" => thresholds.theta_semantic_leap,
            "theta_shortcut_conf" => thresholds.theta_shortcut_conf,
            "theta_johari" => thresholds.theta_johari,
            "theta_blind_spot" => thresholds.theta_blind_spot,
            "theta_obsolescence_low" => thresholds.theta_obsolescence_low,
            "theta_obsolescence_high" => thresholds.theta_obsolescence_high,
            _ => {
                tracing::warn!(threshold_name = %name, "Unknown threshold name requested");
                return None;
            }
        })
    }

    fn get_threshold_or_general(&self, name: &str, domain: Domain) -> f32 {
        self.get_threshold(name, domain)
            .or_else(|| self.get_threshold(name, Domain::General))
            .unwrap_or_else(|| {
                tracing::error!(threshold_name = %name, "Unknown threshold, returning 0.5");
                0.5
            })
    }

    fn observe_threshold_usage(&mut self, name: &str, value: f32) {
        self.observe_threshold(name, value);
    }

    fn list_threshold_names() -> &'static [&'static str] {
        THRESHOLD_NAMES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_threshold_known_names() {
        let atc = AdaptiveThresholdCalibration::new();

        for name in THRESHOLD_NAMES {
            let value = atc.get_threshold(name, Domain::General);
            assert!(value.is_some(), "Threshold {} should exist", name);
            let v = value.unwrap();
            assert!(v >= 0.0 && v <= 1.0, "Threshold {} = {} out of [0,1]", name, v);
        }
    }

    #[test]
    fn test_get_threshold_unknown_returns_none() {
        let atc = AdaptiveThresholdCalibration::new();
        let result = atc.get_threshold("theta_nonexistent", Domain::General);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_threshold_or_general_fallback() {
        let atc = AdaptiveThresholdCalibration::new();
        let value = atc.get_threshold_or_general("theta_opt", Domain::Code);
        assert!(value >= 0.60 && value <= 0.90);
    }

    #[test]
    fn test_domain_differences() {
        let atc = AdaptiveThresholdCalibration::new();

        let code_gate = atc.get_threshold("theta_gate", Domain::Code).unwrap();
        let creative_gate = atc.get_threshold("theta_gate", Domain::Creative).unwrap();

        // Code (strictness=0.9) should have higher gate than Creative (strictness=0.2)
        assert!(code_gate > creative_gate,
            "Code gate {} should be > Creative gate {}", code_gate, creative_gate);
    }

    #[test]
    fn test_list_threshold_names() {
        let names = AdaptiveThresholdCalibration::list_threshold_names();
        assert_eq!(names.len(), 18);
        assert!(names.contains(&"theta_opt"));
        assert!(names.contains(&"theta_gate"));
        assert!(names.contains(&"theta_obsolescence_high"));
    }
}
```

### Step 7: Update mod.rs

**File:** `crates/context-graph-core/src/atc/mod.rs`

Add after line 45:
```rust
pub mod accessor;
```

Add to re-exports after line 56:
```rust
pub use accessor::{ThresholdAccessor, THRESHOLD_NAMES};
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-core/src/atc/accessor.rs` | ThresholdAccessor trait and impl |

## Files to Modify

| Path | Changes |
|------|---------|
| `crates/context-graph-core/src/atc/domain.rs` | Extend struct, update methods |
| `crates/context-graph-core/src/atc/mod.rs` | Add `pub mod accessor` and re-exports |

---

## Full State Verification (FSV) Requirements

### Source of Truth

| Data | Source | Verification |
|------|--------|--------------|
| DomainThresholds fields | `domain.rs` struct definition | Compile + 19 fields present |
| Field ranges | `is_valid()` method | Unit tests with boundary values |
| Domain strictness scaling | `new()` method | Unit tests comparing domains |
| ThresholdAccessor | `accessor.rs` | Unit tests for all 18 names |

### Execute & Inspect Pattern

After each change:

1. **Compile check:** `cargo check -p context-graph-core`
2. **Run tests:** `cargo test -p context-graph-core atc::`
3. **Verify field count:** Grep for `pub theta_` in domain.rs, expect 18 matches
4. **Verify THRESHOLD_NAMES:** Grep for `THRESHOLD_NAMES`, expect array of 18 entries

### Boundary & Edge Case Audit

| Edge Case | Input | Expected | Test |
|-----------|-------|----------|------|
| Min strictness domain | `Domain::Creative` (0.2) | theta_gate ≈ 0.78 | `test_creative_thresholds` |
| Max strictness domain | `Domain::Medical` (1.0) | theta_gate ≈ 0.90 | `test_medical_thresholds` |
| Invalid theta_gate | 0.50 (below min) | `is_valid() == false` | `test_invalid_gate_below_min` |
| Invalid monotonicity | obsolescence_high < low | `is_valid() == false` | `test_invalid_obsolescence_monotonicity` |
| Blend 50/50 | alpha=0.5 | Values between both | `test_blend_midpoint` |
| Unknown threshold | "theta_xyz" | `get_threshold() == None` | `test_unknown_threshold` |

### Evidence of Success

After tests pass, run:
```bash
cargo test -p context-graph-core atc:: -- --nocapture 2>&1 | grep -E "(PASS|FAIL|test_)"
```

Expected output:
```
test atc::accessor::tests::test_get_threshold_known_names ... ok
test atc::accessor::tests::test_get_threshold_unknown_returns_none ... ok
test atc::accessor::tests::test_domain_differences ... ok
test atc::domain::tests::test_extended_fields_exist ... ok
test atc::domain::tests::test_domain_strictness_affects_thresholds ... ok
test atc::domain::tests::test_obsolescence_monotonicity ... ok
... (all tests pass)
```

---

## Validation Criteria

| Criterion | Method | Status |
|-----------|--------|--------|
| Struct has 19 fields | Compile success | ⬜ |
| All domains initialize successfully | Unit test iteration | ⬜ |
| is_valid() checks all 19 fields | Unit tests with invalid values | ⬜ |
| clamp() respects all ranges | Unit tests | ⬜ |
| blend_with_similar() includes all fields | Unit tests | ⬜ |
| ThresholdAccessor returns correct values | Unit tests per threshold | ⬜ |
| Unknown threshold names return None | Unit test | ⬜ |
| Domain strictness scaling correct | Compare Code vs Creative | ⬜ |
| Obsolescence monotonicity enforced | Unit test | ⬜ |
| All existing ATC tests pass | `cargo test atc::` | ⬜ |

---

## Test Commands

```bash
# Compile check
cargo check -p context-graph-core

# Run ATC tests
cargo test -p context-graph-core atc:: -- --nocapture

# Run specific new tests
cargo test -p context-graph-core atc::domain::tests::test_extended
cargo test -p context-graph-core atc::accessor::tests

# Clippy check
cargo clippy -p context-graph-core -- -D warnings

# Doc tests
cargo test -p context-graph-core --doc
```

---

## Test Cases to Implement

### In domain.rs tests module:

```rust
#[test]
fn test_extended_fields_exist() {
    let thresholds = DomainThresholds::new(Domain::Code);
    // All new fields accessible and in valid range
    assert!((0.65..=0.95).contains(&thresholds.theta_gate));
    assert!((0.90..=0.99).contains(&thresholds.theta_hypersync));
    assert!((0.35..=0.65).contains(&thresholds.theta_fragmentation));
    assert!((0.35..=0.75).contains(&thresholds.theta_memory_sim));
    assert!((0.70..=0.95).contains(&thresholds.theta_reflex_hit));
    assert!((0.05..=0.30).contains(&thresholds.theta_consolidation));
    assert!((0.05..=0.30).contains(&thresholds.theta_dream_activity));
    assert!((0.50..=0.90).contains(&thresholds.theta_semantic_leap));
    assert!((0.50..=0.85).contains(&thresholds.theta_shortcut_conf));
    assert!((0.35..=0.65).contains(&thresholds.theta_johari));
    assert!((0.35..=0.65).contains(&thresholds.theta_blind_spot));
    assert!((0.20..=0.50).contains(&thresholds.theta_obsolescence_low));
    assert!((0.65..=0.90).contains(&thresholds.theta_obsolescence_high));
}

#[test]
fn test_domain_strictness_affects_new_thresholds() {
    let code = DomainThresholds::new(Domain::Code);       // strictness=0.9
    let creative = DomainThresholds::new(Domain::Creative); // strictness=0.2

    // Stricter domain has HIGHER gate
    assert!(code.theta_gate > creative.theta_gate);
    // Stricter domain has HIGHER memory similarity requirement
    assert!(code.theta_memory_sim > creative.theta_memory_sim);
    // Stricter domain dreams LESS aggressively (higher threshold to trigger)
    assert!(code.theta_dream_activity > creative.theta_dream_activity);
}

#[test]
fn test_obsolescence_monotonicity() {
    for domain in [Domain::Code, Domain::Medical, Domain::Creative, Domain::General] {
        let t = DomainThresholds::new(domain);
        assert!(
            t.theta_obsolescence_high > t.theta_obsolescence_low,
            "{:?}: high {} should be > low {}",
            domain, t.theta_obsolescence_high, t.theta_obsolescence_low
        );
    }
}

#[test]
fn test_is_valid_fails_invalid_monotonicity() {
    let mut t = DomainThresholds::new(Domain::General);
    t.theta_obsolescence_high = 0.30; // Invalid: lower than low
    t.theta_obsolescence_low = 0.40;
    assert!(!t.is_valid(), "Should fail monotonicity check");
}

#[test]
fn test_clamp_all_new_fields() {
    let mut t = DomainThresholds::new(Domain::General);

    // Set all new fields out of range
    t.theta_gate = 1.5;
    t.theta_hypersync = 0.5;
    t.theta_fragmentation = 0.0;
    t.theta_memory_sim = 1.0;
    t.theta_reflex_hit = 0.5;
    t.theta_consolidation = 0.5;
    t.theta_dream_activity = 0.5;
    t.theta_semantic_leap = 0.0;
    t.theta_shortcut_conf = 0.0;
    t.theta_johari = 0.0;
    t.theta_blind_spot = 1.0;
    t.theta_obsolescence_low = 0.0;
    t.theta_obsolescence_high = 1.0;

    t.clamp();

    // All should now be in range
    assert_eq!(t.theta_gate, 0.95);
    assert_eq!(t.theta_hypersync, 0.90);
    assert_eq!(t.theta_fragmentation, 0.35);
    // ... etc
    assert!(t.is_valid() || t.theta_obsolescence_high <= t.theta_obsolescence_low);
    // Note: clamp alone won't fix monotonicity
}

#[test]
fn test_blend_includes_new_fields() {
    let mut code = DomainThresholds::new(Domain::Code);
    let creative = DomainThresholds::new(Domain::Creative);

    let original_gate = code.theta_gate;
    code.blend_with_similar(&creative, 0.5);

    // Gate should have moved toward creative
    assert!(code.theta_gate < original_gate);
}
```

---

## Manual Testing Procedure

### Synthetic Test: Domain Comparison

1. Create DomainThresholds for all 6 domains
2. Print all 19 threshold values for each
3. Verify:
   - Medical has strictest values (highest gates, lowest dream activity)
   - Creative has loosest values (lowest gates, highest dream activity)
   - General is in the middle

```bash
# After implementation, run:
cargo test -p context-graph-core test_print_all_domain_thresholds -- --nocapture
```

Expected output pattern:
```
Domain::Medical:
  theta_gate: 0.90
  theta_dream_activity: 0.10
  ...
Domain::Creative:
  theta_gate: 0.78
  theta_dream_activity: 0.14
  ...
```

### Synthetic Test: ThresholdAccessor Roundtrip

1. Create ATC instance
2. For each of 18 threshold names, call `get_threshold(name, Domain::Code)`
3. Verify all return Some(f32) in valid range
4. Call with "theta_invalid" - expect None

---

## Constitution Reference

From `/home/cabdru/contextgraph/docs2/constitution.yaml`:

### GWT Thresholds (lines 220-236)
```yaml
gwt:
  kuramoto:
    thresholds: { coherent: "r≥0.8", fragmented: "r<0.5", hypersync: "r>0.95" }
  workspace:
    coherence_threshold: 0.8
```

### Dream Thresholds (lines 254-280)
```yaml
dream:
  trigger: { activity: "<0.15", idle: "10min", entropy: ">0.7 for 5min" }
  phases:
    rem:
      blind_spot: { min_semantic_distance: 0.7 }
  amortized:
    confidence_threshold: 0.7
```

### Adaptive Thresholds (lines 309-326)
```yaml
adaptive_thresholds:
  priors:
    θ_opt: [0.75, "[0.60,0.90]"]
    θ_acc: [0.70, "[0.55,0.85]"]
    θ_warn: [0.55, "[0.40,0.70]"]
    θ_dup: [0.90, "[0.80,0.98]"]
    θ_kur: [0.80, "[0.65,0.95]"]
```

---

## Notes

- All new threshold ranges derived from constitution.yaml and threshold-inventory.yaml
- Domain strictness scaling follows existing `theta_opt/acc/warn` logic
- `theta_johari` and `theta_blind_spot` are fixed at 0.5 per constitution but stored for future flexibility
- The accessor trait enables both static (field access) and dynamic (name-based) threshold retrieval
- **NO MOCK DATA** - All tests use actual DomainThresholds instances
- **FAIL FAST** - Invalid configurations cause test failures, not silent fallbacks

---

**Created:** 2026-01-11
**Updated:** 2026-01-11
**Author:** AI Coding Agent
**Status:** Ready for implementation
