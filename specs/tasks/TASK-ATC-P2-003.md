# TASK-ATC-P2-003: Migrate GWT Layer Thresholds to ATC

**Version:** 1.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 3
**Implements:** REQ-ATC-001
**Depends On:** TASK-ATC-P2-002
**Estimated Complexity:** Medium
**Priority:** P2

---

## Metadata

```yaml
id: TASK-ATC-P2-003
title: Migrate GWT Layer Thresholds to ATC
status: ready
layer: logic
sequence: 3
implements:
  - REQ-ATC-001
depends_on:
  - TASK-ATC-P2-002
estimated_complexity: medium
```

---

## Context

The Global Workspace Theory (GWT) coherence layer in `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/coherence.rs` contains three critical hardcoded thresholds:

1. `GW_THRESHOLD = 0.7` - Gate for workspace broadcast
2. `HYPERSYNC_THRESHOLD = 0.95` - Pathological hypersynchronization detection
3. `FRAGMENTATION_THRESHOLD = 0.5` - Workspace fragmentation warning

These thresholds directly affect consciousness emergence and should vary by domain:
- **Medical/Legal:** Higher gates for more rigorous broadcast criteria
- **Creative:** Lower gates to encourage exploration
- **Code:** Standard gates with strict hypersync monitoring

---

## Input Context Files

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/coherence.rs` | Current hardcoded thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | Extended DomainThresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/accessor.rs` | ThresholdAccessor trait |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/workspace.rs` | GWT workspace using thresholds |

---

## Prerequisites

- [x] TASK-ATC-P2-002 completed (extended DomainThresholds)
- [x] ThresholdAccessor trait available

---

## Scope

### In Scope

1. Remove/deprecate hardcoded `GW_THRESHOLD`, `HYPERSYNC_THRESHOLD`, `FRAGMENTATION_THRESHOLD`
2. Update coherence layer to accept ATC reference or threshold values
3. Create helper function to retrieve GWT thresholds by domain
4. Update all call sites to pass domain context
5. Maintain backward compatibility via General domain defaults
6. Add unit tests for domain-specific GWT behavior

### Out of Scope

- Other layer thresholds (TASK-ATC-P2-004)
- Dream layer thresholds (TASK-ATC-P2-005)
- MCP tool updates

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/layers/coherence.rs

// REMOVE or DEPRECATE these constants:
// pub const GW_THRESHOLD: f32 = 0.7;           // REMOVED
// pub const HYPERSYNC_THRESHOLD: f32 = 0.95;   // REMOVED
// pub const FRAGMENTATION_THRESHOLD: f32 = 0.5; // REMOVED

// ADD deprecation warnings if keeping for transition:
#[deprecated(since = "0.5.0", note = "Use ATC get_threshold('theta_gate', domain) instead")]
pub const GW_THRESHOLD: f32 = 0.7;

// NEW: GWT threshold retrieval
use crate::atc::{AdaptiveThresholdCalibration, Domain, ThresholdAccessor};

/// Get GWT thresholds for a specific domain
pub struct GwtThresholds {
    pub gate: f32,
    pub hypersync: f32,
    pub fragmentation: f32,
}

impl GwtThresholds {
    /// Create from ATC for a specific domain
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> Self {
        Self {
            gate: atc.get_threshold("theta_gate", domain),
            hypersync: atc.get_threshold("theta_hypersync", domain),
            fragmentation: atc.get_threshold("theta_fragmentation", domain),
        }
    }

    /// Create with General domain defaults (backward compat)
    pub fn default_general() -> Self {
        Self {
            gate: 0.70,
            hypersync: 0.95,
            fragmentation: 0.50,
        }
    }
}
```

```rust
// File: crates/context-graph-core/src/gwt/workspace.rs (example update)

use crate::layers::coherence::GwtThresholds;

impl GlobalWorkspace {
    /// Check if memory should enter workspace
    /// Uses domain-specific thresholds from ATC
    pub fn should_broadcast(&self, coherence: f32, thresholds: &GwtThresholds) -> bool {
        coherence >= thresholds.gate
    }

    /// Check for pathological hypersynchronization
    pub fn is_hypersync(&self, order_param: f32, thresholds: &GwtThresholds) -> bool {
        order_param > thresholds.hypersync
    }

    /// Check for workspace fragmentation
    pub fn is_fragmented(&self, order_param: f32, thresholds: &GwtThresholds) -> bool {
        order_param < thresholds.fragmentation
    }
}
```

### Constraints

- MUST NOT change observable behavior for General domain
- MUST NOT require ATC initialization for all code paths (allow fallback)
- MUST log threshold usage for EWMA tracking
- MUST maintain existing public API signatures where possible

### Verification

```bash
# Compile check
cargo build --package context-graph-core

# Run GWT tests
cargo test --package context-graph-core gwt::

# Run coherence layer tests
cargo test --package context-graph-core layers::coherence::

# Ensure no regression
cargo test --package context-graph-core
```

---

## Pseudo Code

```
FUNCTION migrate_gwt_thresholds():
    // 1. Create GwtThresholds struct in coherence.rs
    STRUCT GwtThresholds:
        gate: f32
        hypersync: f32
        fragmentation: f32

    // 2. Add factory methods
    IMPL GwtThresholds:
        FUNCTION from_atc(atc: &ATC, domain: Domain) -> Self:
            RETURN GwtThresholds {
                gate: atc.get_threshold("theta_gate", domain),
                hypersync: atc.get_threshold("theta_hypersync", domain),
                fragmentation: atc.get_threshold("theta_fragmentation", domain),
            }

        FUNCTION default_general() -> Self:
            RETURN GwtThresholds {
                gate: 0.70,
                hypersync: 0.95,
                fragmentation: 0.50,
            }

    // 3. Update workspace methods to accept thresholds
    // Find all usages of GW_THRESHOLD, HYPERSYNC_THRESHOLD, FRAGMENTATION_THRESHOLD
    // Replace with thresholds.gate, thresholds.hypersync, thresholds.fragmentation

    // 4. Update call sites
    FOR each call site using hardcoded constants:
        IF atc available:
            thresholds = GwtThresholds::from_atc(atc, domain)
        ELSE:
            thresholds = GwtThresholds::default_general()
        CALL method with thresholds

    // 5. Deprecate old constants
    ADD #[deprecated] to old constants
```

---

## Files to Modify

| Path | Changes |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/coherence.rs` | Deprecate constants, add GwtThresholds struct |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/workspace.rs` | Update methods to accept thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/mod.rs` | Update public interface if needed |

---

## Files to Create

None - modifications only.

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| Old constants deprecated | Deprecation warnings on use |
| GwtThresholds struct exists | Compilation |
| from_atc() returns domain values | Unit test |
| default_general() matches old values | Unit test |
| Workspace methods accept thresholds | Compilation |
| General domain behavior unchanged | Regression test |
| Code domain has stricter thresholds | Unit test |
| Creative domain has looser thresholds | Unit test |

---

## Test Commands

```bash
# Unit tests
cargo test --package context-graph-core layers::coherence::tests
cargo test --package context-graph-core gwt::workspace::tests

# Full GWT tests
cargo test --package context-graph-core gwt::

# Full module tests
cargo test --package context-graph-core
```

---

## Test Cases

### TC-ATC-003-001: GwtThresholds Default Matches Old Constants
```rust
#[test]
fn test_gwt_thresholds_default_matches_old() {
    let thresholds = GwtThresholds::default_general();
    assert_eq!(thresholds.gate, 0.70);
    assert_eq!(thresholds.hypersync, 0.95);
    assert_eq!(thresholds.fragmentation, 0.50);
}
```

### TC-ATC-003-002: Domain-Specific Thresholds Differ
```rust
#[test]
fn test_gwt_domain_thresholds_differ() {
    let atc = AdaptiveThresholdCalibration::new();
    let code = GwtThresholds::from_atc(&atc, Domain::Code);
    let creative = GwtThresholds::from_atc(&atc, Domain::Creative);

    // Code should have stricter gate
    assert!(code.gate > creative.gate);
}
```

### TC-ATC-003-003: Should Broadcast Uses Threshold
```rust
#[test]
fn test_should_broadcast_uses_threshold() {
    let workspace = GlobalWorkspace::new();
    let strict = GwtThresholds { gate: 0.85, hypersync: 0.95, fragmentation: 0.5 };
    let loose = GwtThresholds { gate: 0.65, hypersync: 0.95, fragmentation: 0.5 };

    let coherence = 0.75;
    assert!(!workspace.should_broadcast(coherence, &strict));
    assert!(workspace.should_broadcast(coherence, &loose));
}
```

### TC-ATC-003-004: Hypersync Detection
```rust
#[test]
fn test_hypersync_detection() {
    let workspace = GlobalWorkspace::new();
    let thresholds = GwtThresholds::default_general();

    assert!(!workspace.is_hypersync(0.90, &thresholds));
    assert!(workspace.is_hypersync(0.96, &thresholds));
}
```

---

## Migration Pattern

### Before (Hardcoded)
```rust
pub const GW_THRESHOLD: f32 = 0.7;

fn should_enter_workspace(coherence: f32) -> bool {
    coherence >= GW_THRESHOLD
}
```

### After (ATC-Managed)
```rust
fn should_enter_workspace(coherence: f32, thresholds: &GwtThresholds) -> bool {
    coherence >= thresholds.gate
}

// At call site:
let thresholds = if let Some(atc) = self.atc.as_ref() {
    GwtThresholds::from_atc(atc, self.current_domain())
} else {
    GwtThresholds::default_general()
};
self.should_enter_workspace(coherence, &thresholds)
```

---

## Notes

- GWT thresholds are critical for consciousness behavior
- Domain adaptation enables context-aware consciousness
- Medical/Legal domains should require higher coherence for broadcast
- Creative domains can broadcast with lower coherence to encourage exploration
- Hypersync threshold is relatively stable across domains (pathological state)

---

**Created:** 2026-01-11
**Author:** Specification Agent
