# TASK-ATC-P2-002: Extend DomainThresholds Struct with New Threshold Fields

**Version:** 1.0
**Status:** Ready
**Layer:** Foundation
**Sequence:** 2
**Implements:** REQ-ATC-006, REQ-ATC-007
**Depends On:** TASK-ATC-P2-001
**Estimated Complexity:** Medium
**Priority:** P2

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
  - TASK-ATC-P2-001
estimated_complexity: medium
```

---

## Context

The current `DomainThresholds` struct in `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` contains only 6 threshold fields:
- `theta_opt`
- `theta_acc`
- `theta_warn`
- `theta_dup`
- `theta_edge`
- `confidence_bias`

Based on the inventory from TASK-ATC-P2-001, we need to extend this struct to include all critical thresholds that require domain-specific adaptation. This task also creates the `ThresholdAccessor` trait for unified threshold retrieval.

---

## Input Context Files

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/specs/tasks/threshold-inventory.yaml` | Threshold inventory from TASK-001 |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | Current DomainThresholds struct |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/mod.rs` | ATC orchestrator |
| `/home/cabdru/contextgraph/docs2/constitution.yaml` | Threshold ranges and priors |

---

## Prerequisites

- [x] TASK-ATC-P2-001 completed (threshold inventory)
- [x] ATC module exists and is functional

---

## Scope

### In Scope

1. Extend `DomainThresholds` struct with new fields for all critical thresholds
2. Update `DomainThresholds::new()` with domain-specific initialization
3. Update `DomainThresholds::is_valid()` with new range checks
4. Update `DomainThresholds::clamp()` with new fields
5. Update `DomainThresholds::blend_with_similar()` for transfer learning
6. Create `ThresholdAccessor` trait for name-based access
7. Implement `ThresholdAccessor` for `AdaptiveThresholdCalibration`
8. Add comprehensive unit tests

### Out of Scope

- Actual migration of calling code (subsequent tasks)
- MCP tool updates
- Integration tests with other modules

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/atc/domain.rs

/// Extended threshold fields for domain-specific calibration
#[derive(Debug, Clone)]
pub struct DomainThresholds {
    pub domain: Domain,

    // === Existing fields (unchanged semantics) ===
    pub theta_opt: f32,           // Optimal alignment [0.60, 0.90]
    pub theta_acc: f32,           // Acceptable alignment [0.55, 0.85]
    pub theta_warn: f32,          // Warning alignment [0.40, 0.70]
    pub theta_dup: f32,           // Duplicate detection [0.80, 0.98]
    pub theta_edge: f32,          // Edge creation [0.50, 0.85]
    pub confidence_bias: f32,     // Domain confidence adjustment

    // === NEW: GWT thresholds ===
    pub theta_gate: f32,          // GW broadcast gate [0.65, 0.95]
    pub theta_hypersync: f32,     // Hypersync detection [0.90, 0.99]
    pub theta_fragmentation: f32, // Fragmentation warning [0.35, 0.65]

    // === NEW: Layer thresholds ===
    pub theta_memory_sim: f32,    // Memory similarity [0.35, 0.75]
    pub theta_reflex_hit: f32,    // Reflex cache hit [0.70, 0.95]
    pub theta_consolidation: f32, // Consolidation trigger [0.05, 0.30]

    // === NEW: Dream thresholds ===
    pub theta_dream_activity: f32,  // Dream trigger [0.05, 0.30]
    pub theta_semantic_leap: f32,   // REM exploration [0.50, 0.90]
    pub theta_shortcut_conf: f32,   // Shortcut confidence [0.50, 0.85]

    // === NEW: Classification thresholds ===
    pub theta_johari: f32,          // Johari boundary [0.35, 0.65]
    pub theta_blind_spot: f32,      // Blind spot detection [0.35, 0.65]

    // === NEW: Autonomous thresholds ===
    pub theta_obsolescence_low: f32,  // Low relevance [0.20, 0.50]
    pub theta_obsolescence_mid: f32,  // Medium confidence [0.45, 0.75]
    pub theta_obsolescence_high: f32, // High confidence [0.65, 0.90]
}
```

```rust
// File: crates/context-graph-core/src/atc/accessor.rs (NEW FILE)

use super::{AdaptiveThresholdCalibration, Domain, DomainThresholds};

/// Unified threshold access by name
pub trait ThresholdAccessor {
    /// Get threshold value by name and domain
    /// Returns domain-specific value, or General domain default if unavailable
    fn get_threshold(&self, name: &str, domain: Domain) -> f32;

    /// Get threshold with fallback to General domain
    fn get_threshold_or_default(&self, name: &str, domain: Domain) -> f32;

    /// Observe threshold usage for EWMA drift tracking
    fn observe_threshold_usage(&mut self, name: &str, value: f32, outcome: bool);

    /// List all available threshold names
    fn list_threshold_names() -> &'static [&'static str];
}

impl ThresholdAccessor for AdaptiveThresholdCalibration {
    fn get_threshold(&self, name: &str, domain: Domain) -> f32 {
        let thresholds = self.get_domain_thresholds(domain)
            .or_else(|| self.get_domain_thresholds(Domain::General))
            .expect("General domain must always exist");

        match name {
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
            "theta_obsolescence_mid" => thresholds.theta_obsolescence_mid,
            "theta_obsolescence_high" => thresholds.theta_obsolescence_high,
            _ => {
                tracing::warn!("Unknown threshold name: {}", name);
                0.5 // Safe default
            }
        }
    }

    fn get_threshold_or_default(&self, name: &str, domain: Domain) -> f32 {
        self.get_threshold(name, domain)
    }

    fn observe_threshold_usage(&mut self, name: &str, value: f32, _outcome: bool) {
        self.observe_threshold(name, value);
    }

    fn list_threshold_names() -> &'static [&'static str] {
        &[
            "theta_opt", "theta_acc", "theta_warn", "theta_dup", "theta_edge",
            "theta_gate", "theta_hypersync", "theta_fragmentation",
            "theta_memory_sim", "theta_reflex_hit", "theta_consolidation",
            "theta_dream_activity", "theta_semantic_leap", "theta_shortcut_conf",
            "theta_johari", "theta_blind_spot",
            "theta_obsolescence_low", "theta_obsolescence_mid", "theta_obsolescence_high",
        ]
    }
}
```

### Constraints

- MUST maintain backward compatibility with existing code using `theta_opt`, `theta_acc`, etc.
- MUST initialize all new fields with domain-aware defaults
- MUST enforce monotonicity where applicable (e.g., obsolescence_high > mid > low)
- MUST respect constitution-defined ranges
- MUST NOT break any existing tests

### Verification

```bash
# Compile check
cargo build --package context-graph-core

# Run existing ATC tests
cargo test --package context-graph-core atc::

# Run new tests
cargo test --package context-graph-core atc::domain::tests::test_extended_thresholds
cargo test --package context-graph-core atc::accessor::tests::
```

---

## Pseudo Code

```
FUNCTION DomainThresholds::new(domain: Domain) -> DomainThresholds:
    strictness = domain.strictness()  // 0.0 (loose) to 1.0 (strict)

    // Base thresholds (existing logic)
    theta_opt = 0.75 + (strictness * 0.10)  // [0.75, 0.85]
    theta_acc = 0.70 + (strictness * 0.08)  // [0.70, 0.78]
    theta_warn = 0.55 + (strictness * 0.05) // [0.55, 0.60]

    // GWT thresholds: stricter domains have higher gates
    theta_gate = 0.70 + (strictness * 0.15)          // [0.70, 0.85]
    theta_hypersync = 0.93 + (strictness * 0.04)     // [0.93, 0.97]
    theta_fragmentation = 0.50 - (strictness * 0.10) // [0.40, 0.50]

    // Layer thresholds
    theta_memory_sim = 0.50 + (strictness * 0.15)    // [0.50, 0.65]
    theta_reflex_hit = 0.85 + (strictness * 0.05)    // [0.85, 0.90]
    theta_consolidation = 0.10 + (strictness * 0.05) // [0.10, 0.15]

    // Dream thresholds: creative domains dream more aggressively
    theta_dream_activity = 0.15 - (strictness * 0.05)  // [0.10, 0.15]
    theta_semantic_leap = 0.70 - (strictness * 0.10)   // [0.60, 0.70]
    theta_shortcut_conf = 0.70 + (strictness * 0.10)   // [0.70, 0.80]

    // Classification thresholds
    theta_johari = 0.50                    // Fixed per constitution
    theta_blind_spot = 0.50                // Fixed per constitution

    // Autonomous thresholds
    theta_obsolescence_low = 0.30 + (strictness * 0.10)  // [0.30, 0.40]
    theta_obsolescence_mid = 0.60 + (strictness * 0.10)  // [0.60, 0.70]
    theta_obsolescence_high = 0.80 + (strictness * 0.05) // [0.80, 0.85]

    RETURN DomainThresholds { ... }


FUNCTION DomainThresholds::is_valid() -> bool:
    // Existing checks
    IF NOT (theta_opt > theta_acc > theta_warn): RETURN false
    IF theta_opt NOT IN [0.60, 0.90]: RETURN false
    // ... existing range checks ...

    // NEW checks
    IF theta_gate NOT IN [0.65, 0.95]: RETURN false
    IF theta_hypersync NOT IN [0.90, 0.99]: RETURN false
    IF theta_fragmentation NOT IN [0.35, 0.65]: RETURN false

    IF theta_memory_sim NOT IN [0.35, 0.75]: RETURN false
    IF theta_reflex_hit NOT IN [0.70, 0.95]: RETURN false
    IF theta_consolidation NOT IN [0.05, 0.30]: RETURN false

    IF theta_dream_activity NOT IN [0.05, 0.30]: RETURN false
    IF theta_semantic_leap NOT IN [0.50, 0.90]: RETURN false
    IF theta_shortcut_conf NOT IN [0.50, 0.85]: RETURN false

    IF theta_johari NOT IN [0.35, 0.65]: RETURN false
    IF theta_blind_spot NOT IN [0.35, 0.65]: RETURN false

    // Monotonicity check for obsolescence
    IF NOT (theta_obsolescence_high > theta_obsolescence_mid > theta_obsolescence_low):
        RETURN false
    IF theta_obsolescence_low NOT IN [0.20, 0.50]: RETURN false
    IF theta_obsolescence_mid NOT IN [0.45, 0.75]: RETURN false
    IF theta_obsolescence_high NOT IN [0.65, 0.90]: RETURN false

    RETURN true
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/accessor.rs` | ThresholdAccessor trait and impl |

---

## Files to Modify

| Path | Changes |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | Extend DomainThresholds struct, update methods |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/mod.rs` | Add `pub mod accessor;` and re-exports |

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| New fields exist in DomainThresholds | Compilation success |
| Domain initialization uses strictness | Unit test assertions |
| is_valid() checks new fields | Unit tests with invalid values |
| clamp() respects new ranges | Unit tests |
| blend_with_similar() includes new fields | Unit tests |
| ThresholdAccessor returns correct values | Unit tests per threshold |
| Unknown threshold names return default | Unit test |
| All domains initialize successfully | Unit test iteration |

---

## Test Commands

```bash
# Unit tests
cargo test --package context-graph-core atc::domain::tests
cargo test --package context-graph-core atc::accessor::tests

# Full ATC module tests
cargo test --package context-graph-core atc::

# Ensure no regression
cargo test --package context-graph-core
```

---

## Test Cases

### TC-ATC-002-001: Extended Fields Exist
```rust
#[test]
fn test_extended_fields_exist() {
    let thresholds = DomainThresholds::new(Domain::Code);
    assert!(thresholds.theta_gate > 0.0);
    assert!(thresholds.theta_hypersync > 0.0);
    assert!(thresholds.theta_fragmentation > 0.0);
    // ... all new fields
}
```

### TC-ATC-002-002: Domain Strictness Affects Values
```rust
#[test]
fn test_domain_strictness_affects_thresholds() {
    let code = DomainThresholds::new(Domain::Code);
    let creative = DomainThresholds::new(Domain::Creative);

    assert!(code.theta_gate > creative.theta_gate);
    assert!(code.theta_memory_sim > creative.theta_memory_sim);
    // Creative dreams more aggressively
    assert!(code.theta_dream_activity > creative.theta_dream_activity);
}
```

### TC-ATC-002-003: Monotonicity Enforcement
```rust
#[test]
fn test_obsolescence_monotonicity() {
    let thresholds = DomainThresholds::new(Domain::General);
    assert!(thresholds.theta_obsolescence_high > thresholds.theta_obsolescence_mid);
    assert!(thresholds.theta_obsolescence_mid > thresholds.theta_obsolescence_low);
}
```

### TC-ATC-002-004: ThresholdAccessor Returns Values
```rust
#[test]
fn test_threshold_accessor() {
    let atc = AdaptiveThresholdCalibration::new();
    let gate = atc.get_threshold("theta_gate", Domain::Code);
    assert!(gate >= 0.65 && gate <= 0.95);
}
```

### TC-ATC-002-005: Unknown Threshold Returns Default
```rust
#[test]
fn test_unknown_threshold_returns_default() {
    let atc = AdaptiveThresholdCalibration::new();
    let unknown = atc.get_threshold("theta_nonexistent", Domain::Code);
    assert_eq!(unknown, 0.5);
}
```

---

## Notes

- All new threshold ranges are derived from constitution.yaml and Gap Analysis
- Domain strictness scaling is consistent with existing `theta_opt/acc/warn` logic
- Johari and blind spot thresholds are fixed per constitution (0.5) but stored in struct for future flexibility
- The accessor trait enables both static and dynamic threshold retrieval patterns

---

**Created:** 2026-01-11
**Author:** Specification Agent
