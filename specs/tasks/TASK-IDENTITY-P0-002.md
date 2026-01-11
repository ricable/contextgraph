# Task Specification: Purpose Vector History Interface

**Task ID:** TASK-IDENTITY-P0-002
**Version:** 1.0.0
**Status:** Ready
**Layer:** Foundation
**Sequence:** 2
**Estimated Complexity:** Low

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-010 |
| Depends On | None (foundation task) |
| Blocks | TASK-IDENTITY-P0-003 |
| Priority | P1 - High |

---

## Context

Identity continuity requires comparing consecutive purpose vectors (PV_t vs PV_{t-1}). This task creates a formal interface for managing purpose vector history within the `IdentityContinuityMonitor`.

The existing `SelfEgoNode.identity_trajectory` stores `PurposeSnapshot` objects, but we need a dedicated history manager that:
1. Provides O(1) access to current and previous PV
2. Handles the "first vector" edge case
3. Manages memory efficiently (1000 snapshot limit)

---

## Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Existing PurposeSnapshot, identity_trajectory |
| `specs/functional/SPEC-IDENTITY-001.md` | Section 5.1 for data model |

---

## Prerequisites

- [x] Rust workspace compiles successfully
- [x] PurposeSnapshot type exists in ego_node.rs

---

## Scope

### In Scope

1. Create `PurposeVectorHistory` struct with current/previous tracking
2. Implement push/get methods with edge case handling
3. Add trait `PurposeVectorHistoryProvider` for abstraction
4. Add comprehensive unit tests

### Out of Scope

- IC computation (TASK-IDENTITY-P0-003)
- Persistence to RocksDB (future task)
- Integration with SelfEgoNode (use composition)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs

/// Manages purpose vector history for identity continuity calculation
///
/// Provides O(1) access to current and previous purpose vectors,
/// handling the edge case of first vector (no previous).
///
/// # Memory Management
/// Limited to MAX_HISTORY_SIZE (1000) entries with FIFO eviction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeVectorHistory {
    /// Ring buffer of purpose vectors
    history: VecDeque<PurposeSnapshot>,
    /// Maximum history size (1000 per constitution)
    max_size: usize,
}

/// Trait for purpose vector history operations
pub trait PurposeVectorHistoryProvider {
    /// Push a new purpose vector with context
    ///
    /// # Arguments
    /// * `pv` - The 13D purpose vector
    /// * `context` - Description of what triggered this snapshot
    ///
    /// # Returns
    /// The previous purpose vector, if any existed
    fn push(&mut self, pv: [f32; 13], context: impl Into<String>) -> Option<[f32; 13]>;

    /// Get the current (most recent) purpose vector
    fn current(&self) -> Option<&[f32; 13]>;

    /// Get the previous purpose vector
    fn previous(&self) -> Option<&[f32; 13]>;

    /// Get both current and previous for IC calculation
    /// Returns (current, previous) or None if no history
    fn current_and_previous(&self) -> Option<(&[f32; 13], Option<&[f32; 13]>)>;

    /// Get the number of snapshots in history
    fn len(&self) -> usize;

    /// Check if history is empty
    fn is_empty(&self) -> bool;

    /// Check if this is the first vector (no previous)
    fn is_first_vector(&self) -> bool;
}

impl PurposeVectorHistory {
    /// Create new history with default max size (1000)
    pub fn new() -> Self;

    /// Create with custom max size
    pub fn with_max_size(max_size: usize) -> Self;
}

impl Default for PurposeVectorHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PurposeVectorHistoryProvider for PurposeVectorHistory {
    // All trait methods implemented
}
```

### Constants

```rust
/// Maximum purpose vector history size per constitution
pub const MAX_PV_HISTORY_SIZE: usize = 1000;
```

### Constraints

1. History MUST use `VecDeque` for efficient front removal
2. `push()` MUST evict oldest entry when at capacity
3. `current()` MUST return `None` for empty history
4. `previous()` MUST return `None` for single-entry history
5. `is_first_vector()` MUST return `true` when len() == 1
6. All operations MUST be O(1) or O(log n)
7. NO panics from index operations

### Verification Commands

```bash
# Build check
cargo build -p context-graph-core

# Run tests
cargo test -p context-graph-core purpose_vector_history

# Clippy
cargo clippy -p context-graph-core -- -D warnings
```

---

## Pseudo Code

```rust
use std::collections::VecDeque;

pub const MAX_PV_HISTORY_SIZE: usize = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeVectorHistory {
    history: VecDeque<PurposeSnapshot>,
    max_size: usize,
}

impl PurposeVectorHistory {
    pub fn new() -> Self {
        Self::with_max_size(MAX_PV_HISTORY_SIZE)
    }

    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_size.min(1024)),
            max_size,
        }
    }
}

impl PurposeVectorHistoryProvider for PurposeVectorHistory {
    fn push(&mut self, pv: [f32; 13], context: impl Into<String>) -> Option<[f32; 13]> {
        // Get previous before pushing
        let previous = self.current().copied();

        // Evict oldest if at capacity
        if self.history.len() >= self.max_size {
            self.history.pop_front();
        }

        // Add new snapshot
        self.history.push_back(PurposeSnapshot {
            vector: pv,
            timestamp: Utc::now(),
            context: context.into(),
        });

        previous
    }

    fn current(&self) -> Option<&[f32; 13]> {
        self.history.back().map(|s| &s.vector)
    }

    fn previous(&self) -> Option<&[f32; 13]> {
        if self.history.len() < 2 {
            return None;
        }
        self.history.get(self.history.len() - 2).map(|s| &s.vector)
    }

    fn current_and_previous(&self) -> Option<(&[f32; 13], Option<&[f32; 13]>)> {
        self.current().map(|curr| (curr, self.previous()))
    }

    fn len(&self) -> usize {
        self.history.len()
    }

    fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    fn is_first_vector(&self) -> bool {
        self.history.len() == 1
    }
}
```

---

## Files to Create

None - all additions go to existing file.

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `PurposeVectorHistory`, `PurposeVectorHistoryProvider`, constant |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| PurposeVectorHistory compiles | `cargo build` |
| Trait is implemented correctly | Unit tests for each method |
| FIFO eviction at capacity | Unit test pushing > max_size |
| is_first_vector correct | Unit tests for 0, 1, 2 entries |
| current_and_previous handles edge cases | Unit tests |
| Serialization works | Round-trip test |

---

## Test Cases

```rust
#[cfg(test)]
mod purpose_vector_history_tests {
    use super::*;

    fn pv(val: f32) -> [f32; 13] {
        [val; 13]
    }

    #[test]
    fn test_new_is_empty() {
        let history = PurposeVectorHistory::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert!(history.current().is_none());
        assert!(history.previous().is_none());
    }

    #[test]
    fn test_push_first_returns_none() {
        let mut history = PurposeVectorHistory::new();
        let prev = history.push(pv(0.5), "First");
        assert!(prev.is_none());
    }

    #[test]
    fn test_push_second_returns_first() {
        let mut history = PurposeVectorHistory::new();
        history.push(pv(0.5), "First");
        let prev = history.push(pv(0.7), "Second");
        assert!(prev.is_some());
        assert_eq!(prev.unwrap(), pv(0.5));
    }

    #[test]
    fn test_is_first_vector() {
        let mut history = PurposeVectorHistory::new();
        assert!(!history.is_first_vector()); // Empty is not first

        history.push(pv(0.5), "First");
        assert!(history.is_first_vector());

        history.push(pv(0.6), "Second");
        assert!(!history.is_first_vector());
    }

    #[test]
    fn test_current_and_previous() {
        let mut history = PurposeVectorHistory::new();

        // Empty
        assert!(history.current_and_previous().is_none());

        // One entry
        history.push(pv(0.5), "First");
        let (curr, prev) = history.current_and_previous().unwrap();
        assert_eq!(*curr, pv(0.5));
        assert!(prev.is_none());

        // Two entries
        history.push(pv(0.7), "Second");
        let (curr, prev) = history.current_and_previous().unwrap();
        assert_eq!(*curr, pv(0.7));
        assert_eq!(*prev.unwrap(), pv(0.5));
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut history = PurposeVectorHistory::with_max_size(3);

        history.push(pv(0.1), "1");
        history.push(pv(0.2), "2");
        history.push(pv(0.3), "3");
        assert_eq!(history.len(), 3);

        // Should evict oldest (0.1)
        history.push(pv(0.4), "4");
        assert_eq!(history.len(), 3);

        // Verify oldest was evicted
        let (curr, prev) = history.current_and_previous().unwrap();
        assert_eq!(*curr, pv(0.4));
        assert_eq!(*prev.unwrap(), pv(0.3));
    }

    #[test]
    fn test_serialization() {
        let mut history = PurposeVectorHistory::new();
        history.push(pv(0.5), "Test");
        history.push(pv(0.7), "Test2");

        let serialized = bincode::serialize(&history).unwrap();
        let deserialized: PurposeVectorHistory = bincode::deserialize(&serialized).unwrap();

        assert_eq!(history.len(), deserialized.len());
        assert_eq!(history.current(), deserialized.current());
    }
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
