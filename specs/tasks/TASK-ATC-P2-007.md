# TASK-ATC-P2-007: Migrate Autonomous Services Thresholds to ATC

**Version:** 1.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 7
**Implements:** REQ-ATC-001
**Depends On:** TASK-ATC-P2-002
**Estimated Complexity:** Low
**Priority:** P2

---

## Metadata

```yaml
id: TASK-ATC-P2-007
title: Migrate Autonomous Services Thresholds to ATC
status: ready
layer: logic
sequence: 7
implements:
  - REQ-ATC-001
depends_on:
  - TASK-ATC-P2-002
estimated_complexity: low
```

---

## Context

The autonomous services in NORTH-009 through NORTH-020 contain hardcoded thresholds for:

### Obsolescence Detector (`autonomous/services/obsolescence_detector.rs`)
- `DEFAULT_RELEVANCE_THRESHOLD = 0.3` - Low relevance detection
- `HIGH_CONFIDENCE_THRESHOLD = 0.8` - High confidence obsolescence
- `MEDIUM_CONFIDENCE_THRESHOLD = 0.6` - Medium confidence obsolescence

### Drift Detector (`autonomous/services/drift_detector.rs`)
- `SLOPE_THRESHOLD = 0.005` - Drift slope detection

### Threshold Learner (`autonomous/services/threshold_learner.rs`)
- `DEFAULT_ALPHA = 0.2` - EWMA smoothing factor (internal, may remain static)

These thresholds control autonomous memory curation and should adapt to domain context:
- **Medical/Code:** More conservative obsolescence (higher thresholds)
- **Creative:** More aggressive curation (lower thresholds)
- **Research:** Balanced, favoring knowledge retention

---

## Input Context Files

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/services/obsolescence_detector.rs` | Obsolescence thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/services/drift_detector.rs` | Drift slope threshold |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/services/threshold_learner.rs` | EWMA alpha (evaluate) |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | Extended DomainThresholds |

---

## Prerequisites

- [x] TASK-ATC-P2-002 completed (extended DomainThresholds with autonomous fields)

---

## Scope

### In Scope

1. Migrate obsolescence detector thresholds (3 constants)
2. Migrate drift detector slope threshold
3. Create `AutonomousThresholds` struct for grouped access
4. Update services to accept threshold parameters
5. Unit tests for domain-specific behavior

### Out of Scope

- Internal EWMA alpha (stays in Level 1 ATC)
- Calibration level thresholds (part of ATC core)
- Scheduler timing constants (not thresholds)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/autonomous/thresholds.rs (NEW FILE)

use crate::atc::{AdaptiveThresholdCalibration, Domain, ThresholdAccessor};

/// Thresholds for autonomous curation services
#[derive(Debug, Clone, Copy)]
pub struct AutonomousThresholds {
    // Obsolescence detection
    pub obsolescence_low: f32,   // Low relevance threshold
    pub obsolescence_mid: f32,   // Medium confidence
    pub obsolescence_high: f32,  // High confidence

    // Drift detection
    pub drift_slope: f32,        // Slope threshold for drift
}

impl AutonomousThresholds {
    /// Create from ATC for a specific domain
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> Self {
        Self {
            obsolescence_low: atc.get_threshold("theta_obsolescence_low", domain),
            obsolescence_mid: atc.get_threshold("theta_obsolescence_mid", domain),
            obsolescence_high: atc.get_threshold("theta_obsolescence_high", domain),
            drift_slope: 0.005, // Fixed for now, could be domain-adaptive later
        }
    }

    /// Create with General domain defaults (backward compat)
    pub fn default_general() -> Self {
        Self {
            obsolescence_low: 0.30,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope: 0.005,
        }
    }

    /// Validate thresholds are within acceptable ranges and monotonic
    pub fn is_valid(&self) -> bool {
        // Monotonicity check
        if !(self.obsolescence_high > self.obsolescence_mid
            && self.obsolescence_mid > self.obsolescence_low)
        {
            return false;
        }

        // Range checks
        (0.20..=0.50).contains(&self.obsolescence_low)
            && (0.45..=0.75).contains(&self.obsolescence_mid)
            && (0.65..=0.90).contains(&self.obsolescence_high)
            && (0.001..=0.01).contains(&self.drift_slope)
    }
}

impl Default for AutonomousThresholds {
    fn default() -> Self {
        Self::default_general()
    }
}
```

```rust
// File: crates/context-graph-core/src/autonomous/services/obsolescence_detector.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.obsolescence_low instead")]
const DEFAULT_RELEVANCE_THRESHOLD: f32 = 0.3;

#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.obsolescence_high instead")]
const HIGH_CONFIDENCE_THRESHOLD: f32 = 0.8;

#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.obsolescence_mid instead")]
const MEDIUM_CONFIDENCE_THRESHOLD: f32 = 0.6;

use super::super::thresholds::AutonomousThresholds;

impl ObsolescenceDetector {
    /// Check if memory is obsolete with domain-specific thresholds
    pub fn check_obsolescence_with_thresholds(
        &self,
        relevance: f32,
        thresholds: &AutonomousThresholds,
    ) -> ObsolescenceLevel {
        if relevance < thresholds.obsolescence_low {
            ObsolescenceLevel::High
        } else if relevance < thresholds.obsolescence_mid {
            ObsolescenceLevel::Medium
        } else if relevance < thresholds.obsolescence_high {
            ObsolescenceLevel::Low
        } else {
            ObsolescenceLevel::None
        }
    }
}
```

```rust
// File: crates/context-graph-core/src/autonomous/services/drift_detector.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.drift_slope instead")]
const SLOPE_THRESHOLD: f32 = 0.005;

use super::super::thresholds::AutonomousThresholds;

impl DriftDetector {
    /// Check if drift slope exceeds threshold
    pub fn is_significant_drift_with_threshold(
        &self,
        slope: f32,
        thresholds: &AutonomousThresholds,
    ) -> bool {
        slope.abs() > thresholds.drift_slope
    }
}
```

### Constraints

- MUST maintain obsolescence threshold monotonicity (high > mid > low)
- MUST NOT change behavior for General domain
- MUST ensure conservative behavior for Medical/Code domains
- MAY keep drift_slope static initially

### Verification

```bash
# Compile check
cargo build --package context-graph-core

# Run autonomous service tests
cargo test --package context-graph-core autonomous::services::

# Ensure no regression
cargo test --package context-graph-core
```

---

## Pseudo Code

```
FUNCTION migrate_autonomous_thresholds():
    // 1. Create AutonomousThresholds struct
    CREATE FILE autonomous/thresholds.rs

    // 2. Update autonomous/mod.rs
    ADD `pub mod thresholds;`

    // 3. Update obsolescence_detector.rs
    ADD deprecation to old constants
    ADD check_obsolescence_with_thresholds() method

    // 4. Update drift_detector.rs
    ADD deprecation to SLOPE_THRESHOLD
    ADD is_significant_drift_with_threshold() method

    // 5. Update call sites
    FOR each call to obsolescence/drift checks:
        thresholds = AutonomousThresholds::from_atc(atc, domain)
        CALL new method with thresholds
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/thresholds.rs` | AutonomousThresholds struct |

---

## Files to Modify

| Path | Changes |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/mod.rs` | Add thresholds module |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/services/obsolescence_detector.rs` | Deprecate constants, add method |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/services/drift_detector.rs` | Deprecate constant, add method |

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| AutonomousThresholds struct exists | Compilation |
| Old constants deprecated | Deprecation warnings |
| Monotonicity enforced | Unit test |
| Domain strictness affects values | Unit test |
| Obsolescence detection uses thresholds | Unit test |
| Drift detection uses thresholds | Unit test |

---

## Test Commands

```bash
# Unit tests
cargo test --package context-graph-core autonomous::thresholds::tests
cargo test --package context-graph-core autonomous::services::obsolescence_detector::tests
cargo test --package context-graph-core autonomous::services::drift_detector::tests

# Full autonomous tests
cargo test --package context-graph-core autonomous::
```

---

## Test Cases

### TC-ATC-007-001: AutonomousThresholds Default Matches Old Constants
```rust
#[test]
fn test_autonomous_thresholds_default_matches_old() {
    let thresholds = AutonomousThresholds::default_general();
    assert_eq!(thresholds.obsolescence_low, 0.30);
    assert_eq!(thresholds.obsolescence_mid, 0.60);
    assert_eq!(thresholds.obsolescence_high, 0.80);
    assert_eq!(thresholds.drift_slope, 0.005);
}
```

### TC-ATC-007-002: Monotonicity Enforcement
```rust
#[test]
fn test_monotonicity() {
    let thresholds = AutonomousThresholds::default();
    assert!(thresholds.obsolescence_high > thresholds.obsolescence_mid);
    assert!(thresholds.obsolescence_mid > thresholds.obsolescence_low);
    assert!(thresholds.is_valid());
}
```

### TC-ATC-007-003: Invalid Monotonicity Rejected
```rust
#[test]
fn test_invalid_monotonicity_rejected() {
    let invalid = AutonomousThresholds {
        obsolescence_low: 0.5,  // Higher than mid!
        obsolescence_mid: 0.4,
        obsolescence_high: 0.8,
        drift_slope: 0.005,
    };
    assert!(!invalid.is_valid());
}
```

### TC-ATC-007-004: Domain Strictness Affects Thresholds
```rust
#[test]
fn test_domain_strictness() {
    let atc = AdaptiveThresholdCalibration::new();
    let medical = AutonomousThresholds::from_atc(&atc, Domain::Medical);
    let creative = AutonomousThresholds::from_atc(&atc, Domain::Creative);

    // Medical should be more conservative (higher thresholds)
    assert!(medical.obsolescence_low > creative.obsolescence_low);
    assert!(medical.obsolescence_high > creative.obsolescence_high);
}
```

### TC-ATC-007-005: Obsolescence Detection
```rust
#[test]
fn test_obsolescence_detection() {
    let detector = ObsolescenceDetector::new();
    let thresholds = AutonomousThresholds::default();

    // Very low relevance -> High obsolescence
    assert_eq!(
        detector.check_obsolescence_with_thresholds(0.2, &thresholds),
        ObsolescenceLevel::High
    );

    // Medium relevance -> Medium obsolescence
    assert_eq!(
        detector.check_obsolescence_with_thresholds(0.5, &thresholds),
        ObsolescenceLevel::Medium
    );

    // High relevance -> No obsolescence
    assert_eq!(
        detector.check_obsolescence_with_thresholds(0.9, &thresholds),
        ObsolescenceLevel::None
    );
}
```

### TC-ATC-007-006: Drift Detection
```rust
#[test]
fn test_drift_detection() {
    let detector = DriftDetector::new();
    let thresholds = AutonomousThresholds::default();

    // Small slope -> not significant
    assert!(!detector.is_significant_drift_with_threshold(0.003, &thresholds));

    // Large slope -> significant
    assert!(detector.is_significant_drift_with_threshold(0.01, &thresholds));
}
```

---

## Migration Pattern

### Before (Hardcoded)
```rust
const DEFAULT_RELEVANCE_THRESHOLD: f32 = 0.3;
const HIGH_CONFIDENCE_THRESHOLD: f32 = 0.8;

fn check_obsolescence(&self, relevance: f32) -> ObsolescenceLevel {
    if relevance < DEFAULT_RELEVANCE_THRESHOLD {
        ObsolescenceLevel::High
    } else if relevance < HIGH_CONFIDENCE_THRESHOLD {
        // ...
    }
}
```

### After (ATC-Managed)
```rust
fn check_obsolescence_with_thresholds(
    &self,
    relevance: f32,
    thresholds: &AutonomousThresholds,
) -> ObsolescenceLevel {
    if relevance < thresholds.obsolescence_low {
        ObsolescenceLevel::High
    } else if relevance < thresholds.obsolescence_high {
        // ...
    }
}

// At call site:
let thresholds = AutonomousThresholds::from_atc(&atc, domain);
detector.check_obsolescence_with_thresholds(relevance, &thresholds)
```

---

## Notes

- Obsolescence thresholds must maintain strict ordering (high > mid > low)
- Medical/Code domains should be more conservative (less aggressive pruning)
- Creative domains can prune more aggressively
- Drift slope threshold may become domain-adaptive in future
- The threshold_learner's DEFAULT_ALPHA stays internal to Level 1 ATC

---

**Created:** 2026-01-11
**Author:** Specification Agent
