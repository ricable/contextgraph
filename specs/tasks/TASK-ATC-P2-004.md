# TASK-ATC-P2-004: Migrate Bio-Nervous Layer Thresholds to ATC

**Version:** 1.0
**Status:** Ready
**Layer:** Logic
**Sequence:** 4
**Implements:** REQ-ATC-001, REQ-ATC-005
**Depends On:** TASK-ATC-P2-002
**Estimated Complexity:** Medium
**Priority:** P2

---

## Metadata

```yaml
id: TASK-ATC-P2-004
title: Migrate Bio-Nervous Layer Thresholds to ATC
status: ready
layer: logic
sequence: 4
implements:
  - REQ-ATC-001
  - REQ-ATC-005
depends_on:
  - TASK-ATC-P2-002
estimated_complexity: medium
```

---

## Context

The 5-layer bio-nervous system contains hardcoded thresholds in three layer modules:

### Memory Layer (`layers/memory.rs`)
- `MIN_MEMORY_SIMILARITY = 0.5` - Minimum similarity for memory retrieval

### Reflex Layer (`layers/reflex.rs`)
- `MIN_HIT_SIMILARITY = 0.85` - Minimum similarity for cache hit

### Learning Layer (`layers/learning.rs`)
- `DEFAULT_CONSOLIDATION_THRESHOLD = 0.1` - Threshold for memory consolidation trigger

These thresholds affect the fundamental memory operations and should adapt to domain context:
- **Medical/Code:** Higher similarity thresholds for precision
- **Creative:** Lower thresholds for exploratory retrieval
- **Research:** Balanced thresholds favoring novelty discovery

---

## Input Context Files

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/memory.rs` | Memory layer with MIN_MEMORY_SIMILARITY |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/reflex.rs` | Reflex layer with MIN_HIT_SIMILARITY |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/learning.rs` | Learning layer with consolidation threshold |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/domain.rs` | Extended DomainThresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/accessor.rs` | ThresholdAccessor trait |

---

## Prerequisites

- [x] TASK-ATC-P2-002 completed (extended DomainThresholds with layer fields)

---

## Scope

### In Scope

1. Remove/deprecate hardcoded constants in all three layer modules
2. Create `LayerThresholds` struct for grouped layer threshold access
3. Update layer implementations to accept threshold parameters
4. Update call sites to provide domain context
5. Add comprehensive unit tests for domain-specific behavior

### Out of Scope

- Coherence layer (handled in TASK-ATC-P2-003)
- Sensing layer (L1 - no behavioral thresholds)
- Dream layer thresholds (TASK-ATC-P2-005)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/layers/thresholds.rs (NEW FILE)

use crate::atc::{AdaptiveThresholdCalibration, Domain, ThresholdAccessor};

/// Aggregated thresholds for bio-nervous layers
#[derive(Debug, Clone, Copy)]
pub struct LayerThresholds {
    // Memory layer (L3)
    pub memory_similarity: f32,

    // Reflex layer (L2)
    pub reflex_hit: f32,

    // Learning layer (L4)
    pub consolidation: f32,
}

impl LayerThresholds {
    /// Create from ATC for a specific domain
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> Self {
        Self {
            memory_similarity: atc.get_threshold("theta_memory_sim", domain),
            reflex_hit: atc.get_threshold("theta_reflex_hit", domain),
            consolidation: atc.get_threshold("theta_consolidation", domain),
        }
    }

    /// Create with General domain defaults (backward compat)
    pub fn default_general() -> Self {
        Self {
            memory_similarity: 0.50,
            reflex_hit: 0.85,
            consolidation: 0.10,
        }
    }

    /// Validate thresholds are within acceptable ranges
    pub fn is_valid(&self) -> bool {
        (0.35..=0.75).contains(&self.memory_similarity)
            && (0.70..=0.95).contains(&self.reflex_hit)
            && (0.05..=0.30).contains(&self.consolidation)
    }
}

impl Default for LayerThresholds {
    fn default() -> Self {
        Self::default_general()
    }
}
```

```rust
// File: crates/context-graph-core/src/layers/memory.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use LayerThresholds.memory_similarity instead")]
pub const MIN_MEMORY_SIMILARITY: f32 = 0.5;

use super::thresholds::LayerThresholds;

impl MemoryLayer {
    /// Search memory with domain-specific similarity threshold
    pub fn search_with_threshold(
        &self,
        query: &[f32],
        k: usize,
        thresholds: &LayerThresholds,
    ) -> Vec<MemoryMatch> {
        self.search_internal(query, k)
            .into_iter()
            .filter(|m| m.similarity >= thresholds.memory_similarity)
            .collect()
    }

    // Keep old method with deprecation for backward compat
    #[deprecated(since = "0.5.0", note = "Use search_with_threshold instead")]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<MemoryMatch> {
        self.search_with_threshold(query, k, &LayerThresholds::default())
    }
}
```

```rust
// File: crates/context-graph-core/src/layers/reflex.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use LayerThresholds.reflex_hit instead")]
pub const MIN_HIT_SIMILARITY: f32 = 0.85;

use super::thresholds::LayerThresholds;

impl ReflexLayer {
    /// Check cache with domain-specific hit threshold
    pub fn lookup_with_threshold(
        &self,
        key: &[f32],
        thresholds: &LayerThresholds,
    ) -> Option<CacheHit> {
        let result = self.hopfield.recall(key)?;
        if result.confidence >= thresholds.reflex_hit {
            Some(result)
        } else {
            None
        }
    }
}
```

```rust
// File: crates/context-graph-core/src/layers/learning.rs

// DEPRECATE:
#[deprecated(since = "0.5.0", note = "Use LayerThresholds.consolidation instead")]
pub const DEFAULT_CONSOLIDATION_THRESHOLD: f32 = 0.1;

use super::thresholds::LayerThresholds;

impl LearningLayer {
    /// Check if consolidation should trigger
    pub fn should_consolidate(&self, utl_score: f32, thresholds: &LayerThresholds) -> bool {
        utl_score < thresholds.consolidation
    }
}
```

### Constraints

- MUST NOT change behavior for General domain defaults
- MUST deprecate old constants with clear migration path
- MUST maintain layer independence (no cross-layer dependencies on thresholds)
- MUST NOT require ATC for basic functionality

### Verification

```bash
# Compile check
cargo build --package context-graph-core

# Run layer tests
cargo test --package context-graph-core layers::

# Ensure no regression
cargo test --package context-graph-core
```

---

## Pseudo Code

```
FUNCTION migrate_layer_thresholds():
    // 1. Create LayerThresholds struct in new file layers/thresholds.rs
    CREATE FILE layers/thresholds.rs

    // 2. Update layers/mod.rs to include new module
    ADD `pub mod thresholds;`
    ADD `pub use thresholds::LayerThresholds;`

    // 3. For each layer:
    FOR layer IN [memory, reflex, learning]:
        // Deprecate old constant
        ADD #[deprecated] to OLD_CONSTANT

        // Add new methods accepting LayerThresholds
        ADD *_with_threshold() variants

        // Keep old methods with deprecation
        ADD #[deprecated] to old method signatures

    // 4. Update call sites
    FOR each call site:
        IF atc available:
            thresholds = LayerThresholds::from_atc(atc, domain)
        ELSE:
            thresholds = LayerThresholds::default()
        CALL new method with thresholds
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/thresholds.rs` | LayerThresholds struct |

---

## Files to Modify

| Path | Changes |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/mod.rs` | Add thresholds module |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/memory.rs` | Deprecate constant, add threshold param |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/reflex.rs` | Deprecate constant, add threshold param |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/learning.rs` | Deprecate constant, add threshold param |

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| LayerThresholds struct exists | Compilation |
| Old constants deprecated | Deprecation warnings |
| from_atc() returns domain values | Unit test |
| default_general() matches old values | Unit test |
| Memory search respects threshold | Unit test |
| Reflex lookup respects threshold | Unit test |
| Consolidation trigger respects threshold | Unit test |
| Domain strictness affects values | Unit test |

---

## Test Commands

```bash
# Unit tests
cargo test --package context-graph-core layers::thresholds::tests
cargo test --package context-graph-core layers::memory::tests
cargo test --package context-graph-core layers::reflex::tests
cargo test --package context-graph-core layers::learning::tests

# Full layer tests
cargo test --package context-graph-core layers::
```

---

## Test Cases

### TC-ATC-004-001: LayerThresholds Default Matches Old Constants
```rust
#[test]
fn test_layer_thresholds_default_matches_old() {
    let thresholds = LayerThresholds::default_general();
    assert_eq!(thresholds.memory_similarity, 0.50);
    assert_eq!(thresholds.reflex_hit, 0.85);
    assert_eq!(thresholds.consolidation, 0.10);
}
```

### TC-ATC-004-002: Domain Strictness Affects Layer Thresholds
```rust
#[test]
fn test_layer_domain_strictness() {
    let atc = AdaptiveThresholdCalibration::new();
    let medical = LayerThresholds::from_atc(&atc, Domain::Medical);
    let creative = LayerThresholds::from_atc(&atc, Domain::Creative);

    // Medical should be stricter
    assert!(medical.memory_similarity > creative.memory_similarity);
    assert!(medical.reflex_hit > creative.reflex_hit);
}
```

### TC-ATC-004-003: Memory Search Respects Threshold
```rust
#[test]
fn test_memory_search_respects_threshold() {
    let memory = MemoryLayer::new();
    // ... setup with memories at similarity 0.6 ...

    let strict = LayerThresholds { memory_similarity: 0.7, ..Default::default() };
    let loose = LayerThresholds { memory_similarity: 0.5, ..Default::default() };

    let strict_results = memory.search_with_threshold(&query, 10, &strict);
    let loose_results = memory.search_with_threshold(&query, 10, &loose);

    assert!(strict_results.len() < loose_results.len());
}
```

### TC-ATC-004-004: Reflex Cache Hit Threshold
```rust
#[test]
fn test_reflex_hit_threshold() {
    let reflex = ReflexLayer::new();
    // ... setup with cached item at confidence 0.80 ...

    let strict = LayerThresholds { reflex_hit: 0.85, ..Default::default() };
    let loose = LayerThresholds { reflex_hit: 0.75, ..Default::default() };

    assert!(reflex.lookup_with_threshold(&key, &strict).is_none());
    assert!(reflex.lookup_with_threshold(&key, &loose).is_some());
}
```

### TC-ATC-004-005: Consolidation Trigger
```rust
#[test]
fn test_consolidation_trigger() {
    let learning = LearningLayer::new();
    let thresholds = LayerThresholds::default();

    // Score below threshold should trigger consolidation
    assert!(learning.should_consolidate(0.05, &thresholds));
    // Score above threshold should not
    assert!(!learning.should_consolidate(0.15, &thresholds));
}
```

---

## Migration Pattern

### Before (Hardcoded)
```rust
pub const MIN_MEMORY_SIMILARITY: f32 = 0.5;

fn search(&self, query: &[f32]) -> Vec<Match> {
    results.filter(|m| m.similarity >= MIN_MEMORY_SIMILARITY).collect()
}
```

### After (ATC-Managed)
```rust
fn search_with_threshold(&self, query: &[f32], thresholds: &LayerThresholds) -> Vec<Match> {
    results.filter(|m| m.similarity >= thresholds.memory_similarity).collect()
}

// At call site:
let thresholds = LayerThresholds::from_atc(&atc, domain);
layer.search_with_threshold(&query, &thresholds)
```

---

## Notes

- Each layer operates independently with its own threshold subset
- LayerThresholds groups related thresholds for convenient passing
- Domain adaptation is especially important for memory retrieval precision
- Reflex layer has the highest threshold (0.85) as it's for confident instant recall
- Consolidation threshold is inverse (lower UTL score triggers consolidation)

---

**Created:** 2026-01-11
**Author:** Specification Agent
