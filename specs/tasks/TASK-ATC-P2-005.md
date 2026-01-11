# TASK-ATC-P2-005: Migrate Dream Layer Thresholds to ATC

**Version:** 1.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 5
**Implements:** REQ-ATC-003
**Depends On:** TASK-ATC-P2-002
**Estimated Complexity:** Low
**Priority:** P2

---

## Metadata

```yaml
id: TASK-ATC-P2-005
title: Migrate Dream Layer Thresholds to ATC
status: ready
layer: logic
sequence: 5
implements:
  - REQ-ATC-003
depends_on:
  - TASK-ATC-P2-002
estimated_complexity: low
```

---

## Context

The dream layer in `/home/cabdru/contextgraph/crates/context-graph-core/src/dream/mod.rs` implements NREM/REM phases for memory consolidation and blind spot discovery. It contains three behavioral thresholds:

1. `ACTIVITY_THRESHOLD = 0.15` - Triggers dream when activity drops below this level
2. `MIN_SEMANTIC_LEAP = 0.7` - Minimum semantic distance for REM exploration
3. `SHORTCUT_CONFIDENCE_THRESHOLD = 0.7` - Confidence required for shortcut creation

Domain adaptation rationale:
- **Creative:** Lower activity threshold (dream sooner), lower semantic leap (more exploration)
- **Medical/Code:** Higher thresholds (dream less frequently, more conservative shortcuts)
- **Research:** Moderate thresholds favoring novelty discovery

---

## Input Context Files

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/dream/mod.rs` | Dream layer with hardcoded thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | Extended DomainThresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/accessor.rs` | ThresholdAccessor trait |
| `/home/cabdru/contextgraph/docs2/constitution.yaml` | Dream layer spec (lines 391-396) |

---

## Prerequisites

- [x] TASK-ATC-P2-002 completed (extended DomainThresholds with dream fields)

---

## Scope

### In Scope

1. Remove/deprecate hardcoded dream thresholds
2. Create `DreamThresholds` struct for grouped access
3. Update DreamController to accept threshold parameters
4. Add domain-specific dreaming behavior
5. Unit tests for threshold-aware dreaming

### Out of Scope

- NREM/REM phase logic implementation (separate task per Gap Analysis)
- GPU monitoring thresholds (hardware-specific, intentionally static)
- Dream duration constants (not behavioral thresholds)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/dream/thresholds.rs (NEW FILE)

use crate::atc::{AdaptiveThresholdCalibration, Domain, ThresholdAccessor};

/// Dream layer thresholds for NREM/REM behavior
#[derive(Debug, Clone, Copy)]
pub struct DreamThresholds {
    /// Activity level below which dreaming triggers
    pub activity: f32,

    /// Minimum semantic leap for REM exploration
    pub semantic_leap: f32,

    /// Confidence required for shortcut creation
    pub shortcut_confidence: f32,
}

impl DreamThresholds {
    /// Create from ATC for a specific domain
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> Self {
        Self {
            activity: atc.get_threshold("theta_dream_activity", domain),
            semantic_leap: atc.get_threshold("theta_semantic_leap", domain),
            shortcut_confidence: atc.get_threshold("theta_shortcut_conf", domain),
        }
    }

    /// Create with General domain defaults (backward compat)
    pub fn default_general() -> Self {
        Self {
            activity: 0.15,
            semantic_leap: 0.70,
            shortcut_confidence: 0.70,
        }
    }

    /// Validate thresholds are within acceptable ranges
    pub fn is_valid(&self) -> bool {
        (0.05..=0.30).contains(&self.activity)
            && (0.50..=0.90).contains(&self.semantic_leap)
            && (0.50..=0.85).contains(&self.shortcut_confidence)
    }
}

impl Default for DreamThresholds {
    fn default() -> Self {
        Self::default_general()
    }
}
```

```rust
// File: crates/context-graph-core/src/dream/mod.rs

// DEPRECATE existing constants:
#[deprecated(since = "0.5.0", note = "Use DreamThresholds.activity instead")]
pub const ACTIVITY_THRESHOLD: f32 = 0.15;

#[deprecated(since = "0.5.0", note = "Use DreamThresholds.semantic_leap instead")]
pub const MIN_SEMANTIC_LEAP: f32 = 0.7;

#[deprecated(since = "0.5.0", note = "Use DreamThresholds.shortcut_confidence instead")]
pub const SHORTCUT_CONFIDENCE_THRESHOLD: f32 = 0.7;

pub mod thresholds;
pub use thresholds::DreamThresholds;

impl DreamController {
    /// Check if dreaming should be triggered
    pub fn should_trigger_dream(&self, activity: f32, thresholds: &DreamThresholds) -> bool {
        activity < thresholds.activity
    }

    /// Check if semantic leap is sufficient for REM exploration
    pub fn is_valid_semantic_leap(&self, distance: f32, thresholds: &DreamThresholds) -> bool {
        distance >= thresholds.semantic_leap
    }

    /// Check if shortcut can be created
    pub fn can_create_shortcut(&self, confidence: f32, thresholds: &DreamThresholds) -> bool {
        confidence >= thresholds.shortcut_confidence
    }
}
```

### Constraints

- MUST NOT change behavior for General domain defaults
- MUST allow creative domains to dream more aggressively
- MUST maintain GPU monitoring thresholds as static (hardware-specific)
- MUST NOT affect NREM/REM duration constants

### Verification

```bash
# Compile check
cargo build --package context-graph-core

# Run dream tests
cargo test --package context-graph-core dream::

# Ensure no regression
cargo test --package context-graph-core
```

---

## Pseudo Code

```
FUNCTION migrate_dream_thresholds():
    // 1. Create DreamThresholds struct
    CREATE FILE dream/thresholds.rs

    // 2. Update dream/mod.rs
    ADD `pub mod thresholds;`
    ADD deprecation to old constants
    ADD threshold-aware methods

    // 3. Domain-specific initialization
    FUNCTION DreamThresholds::from_atc(atc, domain):
        strictness = domain.strictness()

        // Creative domains dream more aggressively
        activity = 0.15 - (strictness * 0.05)         // [0.10, 0.15]
        semantic_leap = 0.70 - (strictness * 0.10)    // [0.60, 0.70]
        shortcut_conf = 0.70 + (strictness * 0.10)    // [0.70, 0.80]

        RETURN DreamThresholds { activity, semantic_leap, shortcut_conf }
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/dream/thresholds.rs` | DreamThresholds struct |

---

## Files to Modify

| Path | Changes |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/dream/mod.rs` | Deprecate constants, add threshold module |

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| DreamThresholds struct exists | Compilation |
| Old constants deprecated | Deprecation warnings |
| from_atc() returns domain values | Unit test |
| default_general() matches old values | Unit test |
| Creative domain dreams more | Unit test |
| Shortcut creation respects threshold | Unit test |

---

## Test Commands

```bash
# Unit tests
cargo test --package context-graph-core dream::thresholds::tests
cargo test --package context-graph-core dream::

# Full module tests
cargo test --package context-graph-core
```

---

## Test Cases

### TC-ATC-005-001: DreamThresholds Default Matches Old Constants
```rust
#[test]
fn test_dream_thresholds_default_matches_old() {
    let thresholds = DreamThresholds::default_general();
    assert_eq!(thresholds.activity, 0.15);
    assert_eq!(thresholds.semantic_leap, 0.70);
    assert_eq!(thresholds.shortcut_confidence, 0.70);
}
```

### TC-ATC-005-002: Creative Domain Dreams More Aggressively
```rust
#[test]
fn test_creative_dreams_more() {
    let atc = AdaptiveThresholdCalibration::new();
    let creative = DreamThresholds::from_atc(&atc, Domain::Creative);
    let code = DreamThresholds::from_atc(&atc, Domain::Code);

    // Creative triggers dream at higher activity (lower threshold)
    assert!(creative.activity < code.activity);

    // Creative accepts smaller semantic leaps
    assert!(creative.semantic_leap < code.semantic_leap);
}
```

### TC-ATC-005-003: Should Trigger Dream
```rust
#[test]
fn test_should_trigger_dream() {
    let controller = DreamController::new();

    let low_threshold = DreamThresholds { activity: 0.20, ..Default::default() };
    let high_threshold = DreamThresholds { activity: 0.10, ..Default::default() };

    let activity = 0.15;

    // Should trigger with low threshold (0.15 < 0.20)
    assert!(controller.should_trigger_dream(activity, &low_threshold));

    // Should NOT trigger with high threshold (0.15 > 0.10)
    assert!(!controller.should_trigger_dream(activity, &high_threshold));
}
```

### TC-ATC-005-004: Semantic Leap Validation
```rust
#[test]
fn test_semantic_leap_validation() {
    let controller = DreamController::new();
    let thresholds = DreamThresholds::default();

    // Distance below threshold should fail
    assert!(!controller.is_valid_semantic_leap(0.50, &thresholds));

    // Distance at/above threshold should pass
    assert!(controller.is_valid_semantic_leap(0.70, &thresholds));
    assert!(controller.is_valid_semantic_leap(0.80, &thresholds));
}
```

### TC-ATC-005-005: Shortcut Creation
```rust
#[test]
fn test_shortcut_creation() {
    let controller = DreamController::new();
    let thresholds = DreamThresholds::default();

    // Low confidence should fail
    assert!(!controller.can_create_shortcut(0.50, &thresholds));

    // High confidence should pass
    assert!(controller.can_create_shortcut(0.75, &thresholds));
}
```

---

## Migration Pattern

### Before (Hardcoded)
```rust
pub const ACTIVITY_THRESHOLD: f32 = 0.15;

fn should_dream(&self, activity: f32) -> bool {
    activity < ACTIVITY_THRESHOLD
}
```

### After (ATC-Managed)
```rust
fn should_trigger_dream(&self, activity: f32, thresholds: &DreamThresholds) -> bool {
    activity < thresholds.activity
}

// At call site:
let thresholds = DreamThresholds::from_atc(&atc, domain);
controller.should_trigger_dream(activity, &thresholds)
```

---

## Notes

- Dream layer thresholds inversely correlate with domain strictness
- Creative domains should dream more frequently (lower activity threshold)
- Creative domains accept smaller semantic leaps (more exploration)
- Shortcut confidence increases with strictness (more conservative)
- GPU usage threshold (MAX_GPU_USAGE = 0.30) remains static as it's hardware-constrained

---

**Created:** 2026-01-11
**Author:** Specification Agent
