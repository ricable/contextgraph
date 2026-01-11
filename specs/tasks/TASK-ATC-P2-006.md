# TASK-ATC-P2-006: Migrate Johari Classification Thresholds to ATC

**Version:** 1.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 6
**Implements:** REQ-ATC-002
**Depends On:** TASK-ATC-P2-002
**Estimated Complexity:** Low
**Priority:** P2

---

## Metadata

```yaml
id: TASK-ATC-P2-006
title: Migrate Johari Classification Thresholds to ATC
status: ready
layer: logic
sequence: 6
implements:
  - REQ-ATC-002
depends_on:
  - TASK-ATC-P2-002
estimated_complexity: low
```

---

## Context

The Johari Window classification system assigns memories to one of four quadrants based on entropy (ΔS) and coherence (ΔC) values:

- **Open (Known Self):** ΔS < θ_joh AND ΔC > θ_joh
- **Blind (Unknown Self):** ΔS > θ_joh AND ΔC < θ_joh
- **Hidden (Hidden Self):** ΔS < θ_joh AND ΔC < θ_joh
- **Unknown (Unknown Unknown):** ΔS > θ_joh AND ΔC > θ_joh

Currently, the threshold is hardcoded in two locations:
1. `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint/johari/core.rs` - `ENTROPY_THRESHOLD = 0.5`, `COHERENCE_THRESHOLD = 0.5`
2. `/home/cabdru/contextgraph/crates/context-graph-utl/src/johari/classifier.rs` - `DEFAULT_THRESHOLD = 0.5`

The constitution (line 824) specifies: `θ_joh: [0.50, "[0.35,0.65]", per-embedder]`

This indicates the threshold should be adaptive per-embedder and possibly per-domain.

---

## Input Context Files

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint/johari/core.rs` | Core Johari constants |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/johari/classifier.rs` | UTL Johari classifier |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | Extended DomainThresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/config/constants.rs` | BLIND_SPOT_THRESHOLD |
| `/home/cabdru/contextgraph/docs2/constitution.yaml` | Johari spec (lines 259-263) |

---

## Prerequisites

- [x] TASK-ATC-P2-002 completed (extended DomainThresholds with Johari fields)

---

## Scope

### In Scope

1. Remove/deprecate hardcoded Johari thresholds in both locations
2. Create `JohariThresholds` struct (may have per-embedder variants)
3. Update classifier to accept threshold parameters
4. Update config/constants.rs `BLIND_SPOT_THRESHOLD`
5. Ensure classification consistency across all usages
6. Unit tests for threshold-aware classification

### Out of Scope

- Per-embedder threshold adaptation (future enhancement)
- Johari visualization changes
- Cross-space Johari insights (already implemented)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/types/fingerprint/johari/thresholds.rs (NEW FILE)

use crate::atc::{AdaptiveThresholdCalibration, Domain, ThresholdAccessor};

/// Johari Window classification thresholds
#[derive(Debug, Clone, Copy)]
pub struct JohariThresholds {
    /// Boundary for ΔS (entropy) classification
    pub entropy: f32,

    /// Boundary for ΔC (coherence) classification
    pub coherence: f32,

    /// Blind spot detection threshold
    pub blind_spot: f32,
}

impl JohariThresholds {
    /// Create from ATC for a specific domain
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> Self {
        Self {
            entropy: atc.get_threshold("theta_johari", domain),
            coherence: atc.get_threshold("theta_johari", domain), // Same threshold for both
            blind_spot: atc.get_threshold("theta_blind_spot", domain),
        }
    }

    /// Create with General domain defaults (backward compat)
    pub fn default_general() -> Self {
        Self {
            entropy: 0.50,
            coherence: 0.50,
            blind_spot: 0.50,
        }
    }

    /// Validate thresholds are within acceptable ranges
    pub fn is_valid(&self) -> bool {
        (0.35..=0.65).contains(&self.entropy)
            && (0.35..=0.65).contains(&self.coherence)
            && (0.35..=0.65).contains(&self.blind_spot)
    }
}

impl Default for JohariThresholds {
    fn default() -> Self {
        Self::default_general()
    }
}
```

```rust
// File: crates/context-graph-core/src/types/fingerprint/johari/core.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use JohariThresholds.entropy instead")]
pub const ENTROPY_THRESHOLD: f32 = 0.5;

#[deprecated(since = "0.5.0", note = "Use JohariThresholds.coherence instead")]
pub const COHERENCE_THRESHOLD: f32 = 0.5;

use super::thresholds::JohariThresholds;

impl JohariQuadrant {
    /// Classify based on ΔS and ΔC with configurable thresholds
    pub fn classify_with_thresholds(
        delta_s: f32,
        delta_c: f32,
        thresholds: &JohariThresholds,
    ) -> Self {
        let high_entropy = delta_s > thresholds.entropy;
        let high_coherence = delta_c > thresholds.coherence;

        match (high_entropy, high_coherence) {
            (false, true) => JohariQuadrant::Open,    // Known self
            (true, false) => JohariQuadrant::Blind,   // Blind spot
            (false, false) => JohariQuadrant::Hidden, // Hidden self
            (true, true) => JohariQuadrant::Unknown,  // Frontier
        }
    }

    // Keep old method with deprecation
    #[deprecated(since = "0.5.0", note = "Use classify_with_thresholds instead")]
    pub fn classify(delta_s: f32, delta_c: f32) -> Self {
        Self::classify_with_thresholds(delta_s, delta_c, &JohariThresholds::default())
    }
}
```

```rust
// File: crates/context-graph-utl/src/johari/classifier.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use JohariThresholds from context-graph-core")]
const DEFAULT_THRESHOLD: f32 = 0.5;

// Update to use shared JohariThresholds
use context_graph_core::types::fingerprint::johari::JohariThresholds;
```

```rust
// File: crates/context-graph-core/src/config/constants.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use JohariThresholds.blind_spot instead")]
pub const BLIND_SPOT_THRESHOLD: f32 = 0.5;
```

### Constraints

- MUST maintain symmetric entropy/coherence thresholds by default
- MUST ensure classification consistency between core and UTL crates
- MUST NOT change quadrant boundaries for General domain
- SHOULD support future per-embedder threshold adaptation

### Verification

```bash
# Compile check
cargo build --package context-graph-core
cargo build --package context-graph-utl

# Run Johari tests
cargo test --package context-graph-core johari::
cargo test --package context-graph-utl johari::

# Ensure no regression
cargo test --all
```

---

## Pseudo Code

```
FUNCTION migrate_johari_thresholds():
    // 1. Create JohariThresholds struct
    CREATE FILE types/fingerprint/johari/thresholds.rs

    // 2. Update johari/core.rs
    ADD deprecation to old constants
    ADD classify_with_thresholds() method

    // 3. Update UTL classifier
    REMOVE local DEFAULT_THRESHOLD
    IMPORT JohariThresholds from core

    // 4. Update config/constants.rs
    ADD deprecation to BLIND_SPOT_THRESHOLD

    // 5. Update all call sites
    FOR each call to JohariQuadrant::classify():
        thresholds = JohariThresholds::from_atc(atc, domain)
            OR JohariThresholds::default()
        CALL classify_with_thresholds(delta_s, delta_c, thresholds)
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint/johari/thresholds.rs` | JohariThresholds struct |

---

## Files to Modify

| Path | Changes |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint/johari/mod.rs` | Add thresholds module |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint/johari/core.rs` | Deprecate constants, add method |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/johari/classifier.rs` | Use shared thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/config/constants.rs` | Deprecate BLIND_SPOT_THRESHOLD |

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| JohariThresholds struct exists | Compilation |
| Old constants deprecated | Deprecation warnings |
| Classification uses thresholds | Unit test |
| Quadrant boundaries correct | Unit test |
| Blind spot detection works | Unit test |
| Both crates use same thresholds | Integration test |

---

## Test Commands

```bash
# Unit tests
cargo test --package context-graph-core johari::thresholds::tests
cargo test --package context-graph-core johari::core::tests
cargo test --package context-graph-utl johari::

# Full tests
cargo test --all
```

---

## Test Cases

### TC-ATC-006-001: JohariThresholds Default Matches Old Constants
```rust
#[test]
fn test_johari_thresholds_default_matches_old() {
    let thresholds = JohariThresholds::default_general();
    assert_eq!(thresholds.entropy, 0.50);
    assert_eq!(thresholds.coherence, 0.50);
    assert_eq!(thresholds.blind_spot, 0.50);
}
```

### TC-ATC-006-002: Quadrant Classification with Thresholds
```rust
#[test]
fn test_quadrant_classification_with_thresholds() {
    let thresholds = JohariThresholds::default();

    // Open: low entropy, high coherence
    assert_eq!(
        JohariQuadrant::classify_with_thresholds(0.3, 0.7, &thresholds),
        JohariQuadrant::Open
    );

    // Blind: high entropy, low coherence
    assert_eq!(
        JohariQuadrant::classify_with_thresholds(0.7, 0.3, &thresholds),
        JohariQuadrant::Blind
    );

    // Hidden: low entropy, low coherence
    assert_eq!(
        JohariQuadrant::classify_with_thresholds(0.3, 0.3, &thresholds),
        JohariQuadrant::Hidden
    );

    // Unknown: high entropy, high coherence
    assert_eq!(
        JohariQuadrant::classify_with_thresholds(0.7, 0.7, &thresholds),
        JohariQuadrant::Unknown
    );
}
```

### TC-ATC-006-003: Threshold Affects Classification
```rust
#[test]
fn test_threshold_affects_classification() {
    let low = JohariThresholds { entropy: 0.4, coherence: 0.4, blind_spot: 0.4 };
    let high = JohariThresholds { entropy: 0.6, coherence: 0.6, blind_spot: 0.6 };

    // At 0.5/0.5: different quadrant depending on threshold
    let delta_s = 0.5;
    let delta_c = 0.5;

    // With low threshold (0.4): both are "high" -> Unknown
    assert_eq!(
        JohariQuadrant::classify_with_thresholds(delta_s, delta_c, &low),
        JohariQuadrant::Unknown
    );

    // With high threshold (0.6): both are "low" -> Hidden
    assert_eq!(
        JohariQuadrant::classify_with_thresholds(delta_s, delta_c, &high),
        JohariQuadrant::Hidden
    );
}
```

### TC-ATC-006-004: Blind Spot Detection
```rust
#[test]
fn test_blind_spot_detection() {
    let thresholds = JohariThresholds::default();

    // Blind quadrant indicates blind spot
    let quadrant = JohariQuadrant::classify_with_thresholds(0.7, 0.3, &thresholds);
    assert_eq!(quadrant, JohariQuadrant::Blind);

    // Verify it's detected as blind spot
    assert!(quadrant.is_blind_spot());
}
```

---

## Migration Pattern

### Before (Hardcoded)
```rust
pub const ENTROPY_THRESHOLD: f32 = 0.5;
pub const COHERENCE_THRESHOLD: f32 = 0.5;

fn classify(delta_s: f32, delta_c: f32) -> JohariQuadrant {
    let high_entropy = delta_s > ENTROPY_THRESHOLD;
    let high_coherence = delta_c > COHERENCE_THRESHOLD;
    // ...
}
```

### After (ATC-Managed)
```rust
fn classify_with_thresholds(
    delta_s: f32,
    delta_c: f32,
    thresholds: &JohariThresholds,
) -> JohariQuadrant {
    let high_entropy = delta_s > thresholds.entropy;
    let high_coherence = delta_c > thresholds.coherence;
    // ...
}

// At call site:
let thresholds = JohariThresholds::from_atc(&atc, domain);
let quadrant = JohariQuadrant::classify_with_thresholds(delta_s, delta_c, &thresholds);
```

---

## Notes

- Johari thresholds are symmetric by default (entropy = coherence = 0.5)
- Constitution allows per-embedder thresholds (future enhancement)
- Blind spot detection uses same threshold as classification
- This task unifies thresholds across two crates (core and utl)
- The threshold range [0.35, 0.65] prevents extreme quadrant imbalances

---

**Created:** 2026-01-11
**Author:** Specification Agent
