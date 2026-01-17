# Task: TASK-P5-004 - PriorityRanker

## Metadata
- **Phase**: 5 (Injection Pipeline)
- **Sequence**: 39
- **Layer**: logic
- **Estimated LOC**: 200
- **Status**: COMPLETE

## Dependencies
| Task ID | Artifact | Status |
|---------|----------|--------|
| TASK-P5-001 | `InjectionCandidate` type | COMPLETE |
| TASK-P5-002 | `TokenBudget` type | COMPLETE |
| TASK-P5-003 | `InjectionResult` type | COMPLETE |
| TASK-P5-003b | `TemporalEnrichmentProvider` | COMPLETE |

## Produces
| Artifact | Type | File Path |
|----------|------|-----------|
| `RecencyFactor` | struct | `crates/context-graph-core/src/injection/priority.rs` |
| `DiversityBonus` | struct | `crates/context-graph-core/src/injection/priority.rs` |
| `PriorityRanker` | struct | `crates/context-graph-core/src/injection/priority.rs` |

---

## Context

### Background
Priority ranking determines which memories get selected for context injection. The formula:

```
priority = relevance_score * recency_factor * diversity_bonus
```

This rewards memories that are not just similar but also recent and matched across multiple embedding spaces (using `weighted_agreement`, NOT raw space count).

### Business Value
Ensures context injection includes the most valuable memories by combining similarity with temporal relevance and multi-space confirmation.

### Technical Context
- **RecencyFactor**: Time decay multiplier (0.8..=1.3) based on memory age
- **DiversityBonus**: Multi-space agreement multiplier (0.8..=1.5) based on `weighted_agreement`
- **PriorityRanker**: Orchestrates computation and sorting

---

## Current Codebase State (Verified 2026-01-17)

### Existing Infrastructure

**File**: `crates/context-graph-core/src/injection/candidate.rs`
- `InjectionCandidate` struct with fields:
  - `memory_id: Uuid`
  - `content: String`
  - `relevance_score: f32` (validated 0.0..=1.0)
  - `recency_factor: f32` (default 1.0, range 0.8..=1.3)
  - `diversity_bonus: f32` (default 1.0, range 0.8..=1.5)
  - `weighted_agreement: f32` (validated 0.0..=8.5)
  - `matching_spaces: Vec<Embedder>`
  - `priority: f32` (computed as relevance * recency * diversity)
  - `token_count: u32`
  - `category: InjectionCategory`
  - `created_at: DateTime<Utc>`
- `set_priority_factors(&mut self, recency: f32, diversity: f32)` already exists
- `InjectionCandidate` implements `Ord` for sorting by category then priority

**File**: `crates/context-graph-core/src/injection/mod.rs`
- Exports: `InjectionCandidate`, `InjectionCategory`, `TokenBudget`, `InjectionResult`
- Constants exported: `MAX_DIVERSITY_BONUS=1.5`, `MIN_DIVERSITY_BONUS=0.8`, `MAX_RECENCY_FACTOR=1.3`, `MIN_RECENCY_FACTOR=0.8`

**File**: `crates/context-graph-core/src/embeddings/config.rs` (line 326)
- `pub fn is_temporal(embedder: Embedder) -> bool` exists

**File**: `crates/context-graph-core/src/teleological/embedder.rs`
- `Embedder` enum with 13 variants (E1-E13)
- `Embedder::all()` returns iterator over all embedders

### Constitution Thresholds (constitution.yaml lines 596-607)

**Recency Factors**:
```yaml
"<1h": 1.3   # Within 1 hour
"<1d": 1.2   # Within 1 day
"<7d": 1.1   # Within 1 week
"<30d": 1.0  # Within 1 month
">90d": 0.8  # Older than 90 days
```

**Diversity Bonus** (based on `weighted_agreement`):
```yaml
"weighted_agreement >= 5.0": 1.5   # Strong topic signal
"weighted_agreement in [2.5, 5.0)": 1.2  # Topic threshold met
"weighted_agreement in [1.0, 2.5)": 1.0  # Related
"weighted_agreement < 1.0": 0.8   # Weak
```

---

## Scope

### Includes
1. `RecencyFactor` struct with time-based computation
2. `DiversityBonus` struct with weighted_agreement-based computation
3. `PriorityRanker::compute_priority()` method
4. `PriorityRanker::rank_candidates()` method
5. Unit tests for all computations
6. Manual verification with synthetic data

### Excludes
- Budget selection logic (TASK-P5-005)
- Pipeline orchestration (TASK-P5-007)

---

## Implementation Specification

### File to Create: `crates/context-graph-core/src/injection/priority.rs`

```rust
//! PriorityRanker with RecencyFactor and DiversityBonus.
//!
//! Computes priority scores for injection candidates using the formula:
//! `priority = relevance_score * recency_factor * diversity_bonus`
//!
//! # Constitution Compliance
//! - ARCH-09: Topic threshold = weighted_agreement >= 2.5
//! - AP-60: Temporal embedders NEVER count toward topics
//! - AP-10: No NaN/Infinity in similarity scores
//! - AP-12: No magic numbers - use named constants

use chrono::{DateTime, Duration, Utc};

use super::candidate::InjectionCandidate;

// =============================================================================
// RecencyFactor
// =============================================================================

/// Time-based relevance multiplier.
///
/// Recent memories get boosted, old ones get penalized.
/// From constitution.yaml `injection.recency_factors`.
pub struct RecencyFactor;

impl RecencyFactor {
    /// Within 1 hour: strong boost (1.3)
    pub const HOUR_1: f32 = 1.3;
    /// Within 1 day: moderate boost (1.2)
    pub const DAY_1: f32 = 1.2;
    /// Within 1 week: slight boost (1.1)
    pub const WEEK_1: f32 = 1.1;
    /// Within 1 month: neutral (1.0)
    pub const MONTH_1: f32 = 1.0;
    /// Older than 90 days: penalty (0.8)
    pub const OLDER: f32 = 0.8;

    /// Compute recency factor based on memory age.
    ///
    /// Uses `Utc::now()` as reference time.
    #[inline]
    pub fn compute(created_at: DateTime<Utc>) -> f32 {
        Self::compute_relative(created_at, Utc::now())
    }

    /// Compute recency factor with explicit reference time (for testing).
    ///
    /// # Arguments
    /// * `created_at` - When the memory was created
    /// * `now` - Reference time (usually current time)
    ///
    /// # Returns
    /// Recency factor in range [0.8, 1.3]
    pub fn compute_relative(created_at: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
        let age = now.signed_duration_since(created_at);

        if age < Duration::hours(1) {
            Self::HOUR_1
        } else if age < Duration::days(1) {
            Self::DAY_1
        } else if age < Duration::days(7) {
            Self::WEEK_1
        } else if age < Duration::days(30) {
            Self::MONTH_1
        } else {
            Self::OLDER
        }
    }
}

// =============================================================================
// DiversityBonus
// =============================================================================

/// Multi-space match multiplier based on weighted_agreement.
///
/// Uses weighted agreement from topic clustering (NOT raw space count).
/// Temporal embedders (E2-E4) have weight 0.0, so they don't contribute.
///
/// From constitution.yaml `injection.diversity_bonus`.
pub struct DiversityBonus;

impl DiversityBonus {
    /// Strong topic signal: weighted_agreement >= 5.0 -> 1.5x
    pub const STRONG_TOPIC: f32 = 1.5;
    /// Topic threshold met: weighted_agreement in [2.5, 5.0) -> 1.2x
    pub const TOPIC_THRESHOLD: f32 = 1.2;
    /// Related but below threshold: weighted_agreement in [1.0, 2.5) -> 1.0x
    pub const RELATED: f32 = 1.0;
    /// Weak signal: weighted_agreement < 1.0 -> 0.8x
    pub const WEAK: f32 = 0.8;

    /// Compute diversity bonus based on weighted_agreement.
    ///
    /// # Arguments
    /// * `weighted_agreement` - Cross-space agreement score (0.0..=8.5)
    ///
    /// # Returns
    /// Diversity bonus in range [0.8, 1.5]
    #[inline]
    pub fn compute(weighted_agreement: f32) -> f32 {
        if weighted_agreement >= 5.0 {
            Self::STRONG_TOPIC
        } else if weighted_agreement >= 2.5 {
            Self::TOPIC_THRESHOLD
        } else if weighted_agreement >= 1.0 {
            Self::RELATED
        } else {
            Self::WEAK
        }
    }
}

// =============================================================================
// PriorityRanker
// =============================================================================

/// Computes priority scores and ranks injection candidates.
///
/// # Priority Formula
/// ```text
/// priority = relevance_score * recency_factor * diversity_bonus
/// ```
///
/// # Sorting Order
/// 1. By category (lower priority number first: DivergenceAlert=1, HighRelevanceCluster=2, etc.)
/// 2. Within category, by priority score descending (higher score first)
pub struct PriorityRanker;

impl PriorityRanker {
    /// Compute priority for a single candidate.
    ///
    /// Sets `recency_factor`, `diversity_bonus`, and final `priority`
    /// using the candidate's `created_at` and `weighted_agreement` fields.
    #[inline]
    pub fn compute_priority(candidate: &mut InjectionCandidate) {
        let recency = RecencyFactor::compute(candidate.created_at);
        let diversity = DiversityBonus::compute(candidate.weighted_agreement);
        candidate.set_priority_factors(recency, diversity);
    }

    /// Compute priority with explicit reference time (for deterministic testing).
    #[inline]
    pub fn compute_priority_at(candidate: &mut InjectionCandidate, now: DateTime<Utc>) {
        let recency = RecencyFactor::compute_relative(candidate.created_at, now);
        let diversity = DiversityBonus::compute(candidate.weighted_agreement);
        candidate.set_priority_factors(recency, diversity);
    }

    /// Rank all candidates by category then priority.
    ///
    /// 1. Computes priority for each candidate
    /// 2. Sorts: category ascending, priority descending within category
    ///
    /// Modifies candidates in place.
    pub fn rank_candidates(candidates: &mut [InjectionCandidate]) {
        for candidate in candidates.iter_mut() {
            Self::compute_priority(candidate);
        }
        candidates.sort();
    }

    /// Rank with explicit reference time (for deterministic testing).
    pub fn rank_candidates_at(candidates: &mut [InjectionCandidate], now: DateTime<Utc>) {
        for candidate in candidates.iter_mut() {
            Self::compute_priority_at(candidate, now);
        }
        candidates.sort();
    }
}
```

### File to Modify: `crates/context-graph-core/src/injection/mod.rs`

Add after line 16 (`pub mod temporal_enrichment;`):
```rust
pub mod priority;
```

Add to exports (after line 28):
```rust
pub use priority::{DiversityBonus, PriorityRanker, RecencyFactor};
```

---

## Definition of Done

### DOD-1: RecencyFactor Computation
- [x] `RecencyFactor::compute()` returns correct factors for all time ranges
- **Verification**: Unit tests for <1h, <1d, <7d, <30d, >=30d boundaries

### DOD-2: DiversityBonus Computation
- [x] `DiversityBonus::compute()` returns correct bonuses for weighted_agreement thresholds
- **Verification**: Unit tests for <1.0, [1.0,2.5), [2.5,5.0), >=5.0 boundaries

### DOD-3: Priority Formula
- [x] `compute_priority()` correctly computes: priority = relevance * recency * diversity
- **Verification**: Unit test with known values verifies formula

### DOD-4: Ranking Order
- [x] `rank_candidates()` sorts by category then priority descending
- **Verification**: Unit test with mixed categories verifies order

---

## Constraints

| Type | Constraint |
|------|------------|
| Formula | `priority = relevance_score * recency_factor * diversity_bonus` |
| Range | RecencyFactor output: 0.8..=1.3 |
| Range | DiversityBonus output: 0.8..=1.5 |
| Threshold | weighted_agreement >= 5.0 -> 1.5x (strong topic) |
| Threshold | weighted_agreement >= 2.5 -> 1.2x (topic threshold met) |
| Threshold | weighted_agreement >= 1.0 -> 1.0x (related) |
| Threshold | weighted_agreement < 1.0 -> 0.8x (weak) |
| Sort | Primary by category.priority() ascending, secondary by priority descending |

---

## Unit Tests (Required)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::candidate::InjectionCategory;
    use crate::teleological::Embedder;
    use uuid::Uuid;

    fn make_candidate_at(
        relevance: f32,
        created_at: DateTime<Utc>,
        weighted_agreement: f32,
        category: InjectionCategory,
    ) -> InjectionCandidate {
        InjectionCandidate::new(
            Uuid::new_v4(),
            "test content".to_string(),
            relevance,
            weighted_agreement,
            vec![Embedder::Semantic],
            category,
            created_at,
        )
    }

    // =========================================================================
    // RecencyFactor Tests
    // =========================================================================

    #[test]
    fn test_recency_factor_within_hour() {
        let now = Utc::now();
        let created = now - Duration::minutes(30);
        let factor = RecencyFactor::compute_relative(created, now);
        assert!((factor - 1.3).abs() < 0.001, "Within 1h should be 1.3, got {}", factor);
        println!("[PASS] RecencyFactor <1h = 1.3");
    }

    #[test]
    fn test_recency_factor_within_day() {
        let now = Utc::now();
        let created = now - Duration::hours(6);
        let factor = RecencyFactor::compute_relative(created, now);
        assert!((factor - 1.2).abs() < 0.001, "Within 1d should be 1.2, got {}", factor);
        println!("[PASS] RecencyFactor <1d = 1.2");
    }

    #[test]
    fn test_recency_factor_within_week() {
        let now = Utc::now();
        let created = now - Duration::days(3);
        let factor = RecencyFactor::compute_relative(created, now);
        assert!((factor - 1.1).abs() < 0.001, "Within 7d should be 1.1, got {}", factor);
        println!("[PASS] RecencyFactor <7d = 1.1");
    }

    #[test]
    fn test_recency_factor_within_month() {
        let now = Utc::now();
        let created = now - Duration::days(15);
        let factor = RecencyFactor::compute_relative(created, now);
        assert!((factor - 1.0).abs() < 0.001, "Within 30d should be 1.0, got {}", factor);
        println!("[PASS] RecencyFactor <30d = 1.0");
    }

    #[test]
    fn test_recency_factor_older() {
        let now = Utc::now();
        let created = now - Duration::days(100);
        let factor = RecencyFactor::compute_relative(created, now);
        assert!((factor - 0.8).abs() < 0.001, "Older should be 0.8, got {}", factor);
        println!("[PASS] RecencyFactor >=30d = 0.8");
    }

    #[test]
    fn test_recency_factor_boundary_exactly_1_hour() {
        let now = Utc::now();
        let created = now - Duration::hours(1);
        let factor = RecencyFactor::compute_relative(created, now);
        // Exactly 1 hour should be in <1d bucket (1.2)
        assert!((factor - 1.2).abs() < 0.001, "Exactly 1h should be 1.2, got {}", factor);
        println!("[PASS] RecencyFactor boundary at 1h = 1.2");
    }

    // =========================================================================
    // DiversityBonus Tests
    // =========================================================================

    #[test]
    fn test_diversity_bonus_weak() {
        assert!((DiversityBonus::compute(0.0) - 0.8).abs() < 0.001);
        assert!((DiversityBonus::compute(0.5) - 0.8).abs() < 0.001);
        assert!((DiversityBonus::compute(0.99) - 0.8).abs() < 0.001);
        println!("[PASS] DiversityBonus <1.0 = 0.8 (WEAK)");
    }

    #[test]
    fn test_diversity_bonus_related() {
        assert!((DiversityBonus::compute(1.0) - 1.0).abs() < 0.001);
        assert!((DiversityBonus::compute(2.0) - 1.0).abs() < 0.001);
        assert!((DiversityBonus::compute(2.49) - 1.0).abs() < 0.001);
        println!("[PASS] DiversityBonus [1.0, 2.5) = 1.0 (RELATED)");
    }

    #[test]
    fn test_diversity_bonus_topic_threshold() {
        assert!((DiversityBonus::compute(2.5) - 1.2).abs() < 0.001);
        assert!((DiversityBonus::compute(3.5) - 1.2).abs() < 0.001);
        assert!((DiversityBonus::compute(4.99) - 1.2).abs() < 0.001);
        println!("[PASS] DiversityBonus [2.5, 5.0) = 1.2 (TOPIC_THRESHOLD)");
    }

    #[test]
    fn test_diversity_bonus_strong_topic() {
        assert!((DiversityBonus::compute(5.0) - 1.5).abs() < 0.001);
        assert!((DiversityBonus::compute(7.0) - 1.5).abs() < 0.001);
        assert!((DiversityBonus::compute(8.5) - 1.5).abs() < 0.001);
        println!("[PASS] DiversityBonus >=5.0 = 1.5 (STRONG_TOPIC)");
    }

    // =========================================================================
    // PriorityRanker Tests
    // =========================================================================

    #[test]
    fn test_priority_computation_formula() {
        let now = Utc::now();
        let created = now - Duration::hours(2); // <1d -> recency=1.2

        let mut candidate = make_candidate_at(
            0.82,  // relevance
            created,
            3.5,   // weighted_agreement -> diversity=1.2 (topic threshold)
            InjectionCategory::HighRelevanceCluster,
        );

        PriorityRanker::compute_priority_at(&mut candidate, now);

        // Verify factors
        assert!((candidate.recency_factor - 1.2).abs() < 0.001, "recency should be 1.2");
        assert!((candidate.diversity_bonus - 1.2).abs() < 0.001, "diversity should be 1.2");

        // priority = 0.82 * 1.2 * 1.2 = 1.1808
        let expected = 0.82 * 1.2 * 1.2;
        assert!((candidate.priority - expected).abs() < 0.01,
            "priority should be {}, got {}", expected, candidate.priority);
        println!("[PASS] Priority formula: 0.82 * 1.2 * 1.2 = {:.4}", candidate.priority);
    }

    #[test]
    fn test_ranking_by_category_first() {
        let now = Utc::now();

        let mut candidates = vec![
            make_candidate_at(0.9, now, 5.5, InjectionCategory::SingleSpaceMatch),
            make_candidate_at(0.5, now, 0.5, InjectionCategory::DivergenceAlert),
            make_candidate_at(0.7, now, 3.0, InjectionCategory::HighRelevanceCluster),
        ];

        PriorityRanker::rank_candidates_at(&mut candidates, now);

        assert_eq!(candidates[0].category, InjectionCategory::DivergenceAlert);
        assert_eq!(candidates[1].category, InjectionCategory::HighRelevanceCluster);
        assert_eq!(candidates[2].category, InjectionCategory::SingleSpaceMatch);
        println!("[PASS] Ranking sorts by category first");
    }

    #[test]
    fn test_ranking_by_priority_within_category() {
        let now = Utc::now();

        let mut candidates = vec![
            make_candidate_at(0.7, now, 3.0, InjectionCategory::HighRelevanceCluster),
            make_candidate_at(0.9, now, 3.0, InjectionCategory::HighRelevanceCluster),
            make_candidate_at(0.8, now, 3.0, InjectionCategory::HighRelevanceCluster),
        ];

        PriorityRanker::rank_candidates_at(&mut candidates, now);

        // Same category, sorted by priority descending
        assert!(candidates[0].priority > candidates[1].priority);
        assert!(candidates[1].priority > candidates[2].priority);
        println!("[PASS] Within category, sort by priority descending");
    }

    #[test]
    fn test_spec_example_memory_a_vs_b() {
        // From constitution spec example:
        // Memory A: relevance=0.82, 2h ago (recency=1.2), weighted_agreement=3.5 (diversity=1.2)
        //           priority = 0.82 * 1.2 * 1.2 = 1.1808
        // Memory B: relevance=0.90, 3d ago (recency=1.1), weighted_agreement=1.5 (diversity=1.0)
        //           priority = 0.90 * 1.1 * 1.0 = 0.99
        // Memory A should rank higher despite lower relevance

        let now = Utc::now();

        let mut memory_a = make_candidate_at(
            0.82,
            now - Duration::hours(2),
            3.5,
            InjectionCategory::HighRelevanceCluster,
        );

        let mut memory_b = make_candidate_at(
            0.90,
            now - Duration::days(3),
            1.5,
            InjectionCategory::HighRelevanceCluster,
        );

        PriorityRanker::compute_priority_at(&mut memory_a, now);
        PriorityRanker::compute_priority_at(&mut memory_b, now);

        // Memory A should have higher priority
        assert!(memory_a.priority > memory_b.priority,
            "Memory A ({}) should rank higher than Memory B ({})",
            memory_a.priority, memory_b.priority);

        // Verify exact values
        assert!((memory_a.priority - 1.1808).abs() < 0.01);
        assert!((memory_b.priority - 0.99).abs() < 0.01);

        println!("[PASS] Spec example: Memory A (1.18) > Memory B (0.99)");
    }
}
```

---

## Full State Verification Protocol

After implementing the logic, you MUST verify the results through the following protocol:

### 1. Source of Truth Identification
The source of truth is the `InjectionCandidate` struct's fields:
- `recency_factor: f32` - Set by `set_priority_factors()`
- `diversity_bonus: f32` - Set by `set_priority_factors()`
- `priority: f32` - Computed as `relevance_score * recency_factor * diversity_bonus`

### 2. Execute & Inspect Protocol

After running `cargo test injection::priority`, manually verify:

```bash
# Run tests with output
cargo test injection::priority::tests --package context-graph-core -- --nocapture

# Verify compilation
cargo build --package context-graph-core
```

### 3. Edge Case Audit (Manual Verification Required)

You MUST manually test these 3 edge cases and print before/after state:

**Edge Case 1: Boundary at exactly 1 hour**
```rust
// Before: created_at = now - 1 hour exactly
// Expected: Should fall into <1d bucket (1.2), not <1h (1.3)
// Verify: Print the age duration and resulting factor
```

**Edge Case 2: Boundary at weighted_agreement = 2.5 exactly**
```rust
// Before: weighted_agreement = 2.5
// Expected: Should be TOPIC_THRESHOLD (1.2), not RELATED (1.0)
// Verify: Print weighted_agreement and resulting diversity bonus
```

**Edge Case 3: Zero relevance score**
```rust
// Before: relevance_score = 0.0, recency = 1.3, diversity = 1.5
// Expected: priority = 0.0 * 1.3 * 1.5 = 0.0
// Verify: Print all three factors and final priority
```

### 4. Evidence of Success

After running tests, provide:
1. Full test output showing all tests passed
2. The specific priority values computed in the spec example test
3. Confirmation that `cargo build` completes without errors

---

## Test Commands

```bash
# Build
cargo build --package context-graph-core

# Run all priority tests
cargo test injection::priority --package context-graph-core -- --nocapture

# Run specific test
cargo test injection::priority::tests::test_spec_example_memory_a_vs_b --package context-graph-core -- --nocapture

# Verify no regressions in existing injection tests
cargo test injection::candidate --package context-graph-core
```

---

## Validation Criteria

| Type | Criterion |
|------|-----------|
| Compilation | `cargo build --package context-graph-core` compiles without errors |
| Tests | All tests in `injection::priority::tests` pass |
| Spec | Spec example (Memory A vs B) ranks correctly |
| Boundaries | Edge case tests at boundaries pass |
| Integration | `cargo test injection::candidate` still passes |

---

## Anti-Patterns to Avoid

| ID | Anti-Pattern | Correct Approach |
|----|--------------|------------------|
| AP-10 | NaN/Infinity in scores | Already validated in `InjectionCandidate::set_priority_factors()` |
| AP-12 | Magic numbers | Use named constants (HOUR_1, STRONG_TOPIC, etc.) |
| AP-14 | `.unwrap()` in library code | Not needed - no fallible ops in pure computation |
| AP-60 | Temporal embedders in topic detection | DiversityBonus uses `weighted_agreement` which already excludes temporal |

---

## Notes for Implementing Agent

1. **InjectionCandidate already has `set_priority_factors()`** - You just need to call it with the computed values

2. **Sorting is already implemented** - `InjectionCandidate` implements `Ord`, sorting by category then priority

3. **chrono::Duration** - Use `Duration::hours()`, `Duration::days()`, etc. directly

4. **Validation is handled** - `set_priority_factors()` validates ranges and panics on invalid values

5. **Do NOT use mock data in tests** - Use real `InjectionCandidate::new()` with synthetic values

6. **Test determinism** - Always use `compute_relative()` and `rank_candidates_at()` with explicit `now` for reproducible tests
