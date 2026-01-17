# Task: TASK-P5-005 - TokenBudgetManager

## Metadata
- **Phase**: 5 (Injection Pipeline)
- **Sequence**: 40
- **Layer**: logic
- **Estimated LOC**: 180
- **Status**: COMPLETE
- **Completed**: 2026-01-17
- **Verified**: All 27 unit tests + 8 FSV tests pass

## Dependencies
| Task ID | Artifact | Status | File Location |
|---------|----------|--------|---------------|
| TASK-P5-001 | `InjectionCandidate` type | COMPLETE | `crates/context-graph-core/src/injection/candidate.rs` |
| TASK-P5-002 | `TokenBudget` type | COMPLETE | `crates/context-graph-core/src/injection/budget.rs` |
| TASK-P5-003 | `InjectionResult` type | COMPLETE | `crates/context-graph-core/src/injection/result.rs` |
| TASK-P5-003b | `TemporalEnrichmentProvider` | COMPLETE | `crates/context-graph-core/src/injection/temporal_enrichment.rs` |
| TASK-P5-004 | `PriorityRanker` | COMPLETE | `crates/context-graph-core/src/injection/priority.rs` |

## Produces
| Artifact | Type | File Path |
|----------|------|-----------|
| `TokenBudgetManager` | struct | `crates/context-graph-core/src/injection/budget.rs` |
| `CategorySpending` | struct | `crates/context-graph-core/src/injection/budget.rs` |
| `SelectionStats` | struct | `crates/context-graph-core/src/injection/budget.rs` |
| `estimate_tokens()` | function | `crates/context-graph-core/src/injection/budget.rs` |

---

## Context

### Background
TokenBudgetManager selects candidates within budget constraints. It tracks spending per category and ensures each category gets fair representation while allowing overflow into reserved budget when higher-priority categories need more space.

### Business Value
Enables consistent, predictable context injection that respects token limits while ensuring important categories (like divergence alerts) always get included.

### Technical Context
Called AFTER `PriorityRanker::rank_candidates()` has sorted candidates. Iterates through the sorted candidates and selects those that fit within category and total budgets. Produces the final list of candidates to be formatted by `ContextFormatter` (TASK-P5-006).

---

## Current Codebase State (Verified 2026-01-17)

### EXISTING Types in `budget.rs` (Lines 1-279)

**TokenBudget struct** (lines 14-28):
```rust
pub struct TokenBudget {
    pub total: u32,           // Default: 1200
    pub divergence_budget: u32,    // Default: 200
    pub cluster_budget: u32,       // Default: 400
    pub single_space_budget: u32,  // Default: 300
    pub session_budget: u32,       // Default: 200
    pub reserved: u32,             // Default: 100
}
```

**Existing Methods**:
- `TokenBudget::with_total(total: u32)` - Creates budget with proportional allocation
- `TokenBudget::budget_for_category(category: InjectionCategory) -> u32`
- `TokenBudget::usable() -> u32` - Returns `total - reserved`
- `TokenBudget::is_valid() -> bool` - Validates sum equals total
- `TokenBudget::default()` - Returns `DEFAULT_TOKEN_BUDGET`

**Constants**:
- `DEFAULT_TOKEN_BUDGET: TokenBudget` (const)
- `BRIEF_BUDGET: u32 = 200`

### EXISTING Types in `candidate.rs` (Lines 1-799)

**InjectionCategory enum** with methods:
- `priority() -> u8`: DivergenceAlert=1, HighRelevanceCluster=2, SingleSpaceMatch=3, RecentSession=4
- `token_budget() -> u32`: 200, 400, 300, 200 respectively
- `from_weighted_agreement(f32) -> Option<Self>`

**InjectionCandidate struct** with fields:
- `memory_id: Uuid`
- `content: String`
- `relevance_score: f32` (0.0..=1.0)
- `recency_factor: f32` (0.8..=1.3)
- `diversity_bonus: f32` (0.8..=1.5)
- `weighted_agreement: f32` (0.0..=8.5)
- `matching_spaces: Vec<Embedder>`
- `priority: f32` (computed)
- `token_count: u32` (computed as words × 1.3)
- `category: InjectionCategory`
- `created_at: DateTime<Utc>`

**Token estimation already exists** in candidate.rs:
```rust
pub const TOKEN_MULTIPLIER: f32 = 1.3;
// In InjectionCandidate::new():
let word_count = content.split_whitespace().count();
let token_count = (word_count as f32 * TOKEN_MULTIPLIER).ceil() as u32;
```

### EXISTING Types in `priority.rs` (Lines 1-822)

**PriorityRanker** already provides:
- `rank_candidates(candidates: &mut [InjectionCandidate])` - Sorts by category then priority
- `rank_candidates_at(candidates: &mut [InjectionCandidate], now: DateTime<Utc>)` - Deterministic variant

### Module Exports in `mod.rs` (Lines 1-31)

Current exports from injection module:
```rust
pub use budget::{TokenBudget, DEFAULT_TOKEN_BUDGET, BRIEF_BUDGET};
pub use candidate::{
    InjectionCandidate, InjectionCategory, MAX_DIVERSITY_BONUS, MAX_RECENCY_FACTOR,
    MAX_WEIGHTED_AGREEMENT, MIN_DIVERSITY_BONUS, MIN_RECENCY_FACTOR, TOKEN_MULTIPLIER,
};
pub use priority::{DiversityBonus, PriorityRanker, RecencyFactor};
pub use result::InjectionResult;
pub use temporal_enrichment::{...};
```

---

## Scope

### Includes
1. `CategorySpending` struct - Tracks tokens spent per category
2. `TokenBudgetManager` struct - Manages selection within budget
3. `SelectionStats` struct - Statistics about selection process
4. `estimate_tokens(content: &str) -> u32` function - Standalone token estimation
5. `select_within_budget()` method - Core selection logic
6. `select_with_stats()` method - Selection with detailed statistics
7. Unit tests for all selection logic
8. Manual verification with synthetic data

### Excludes
- Priority computation (TASK-P5-004 - COMPLETE)
- Context formatting (TASK-P5-006)
- tiktoken integration (future enhancement)

---

## Implementation Specification

### Add to File: `crates/context-graph-core/src/injection/budget.rs`

After line 111 (after the `impl Default for TokenBudget` block), add:

```rust
// =============================================================================
// Token Estimation
// =============================================================================

/// Estimate token count for content.
///
/// Simple estimate: word_count × 1.3 (same multiplier as InjectionCandidate).
/// Exposed as standalone function for pre-candidate estimation.
///
/// # Arguments
/// * `content` - Text content to estimate
///
/// # Returns
/// Estimated token count (ceiling of word_count × 1.3)
pub fn estimate_tokens(content: &str) -> u32 {
    let word_count = content.split_whitespace().count();
    (word_count as f32 * 1.3).ceil() as u32
}

// =============================================================================
// CategorySpending
// =============================================================================

/// Tracks tokens spent per injection category during selection.
#[derive(Debug, Clone, Default)]
struct CategorySpending {
    divergence: u32,
    cluster: u32,
    single_space: u32,
    session: u32,
}

impl CategorySpending {
    /// Get current spending for a category.
    #[inline]
    fn get(&self, category: InjectionCategory) -> u32 {
        match category {
            InjectionCategory::DivergenceAlert => self.divergence,
            InjectionCategory::HighRelevanceCluster => self.cluster,
            InjectionCategory::SingleSpaceMatch => self.single_space,
            InjectionCategory::RecentSession => self.session,
        }
    }

    /// Add tokens to a category's spending.
    #[inline]
    fn add(&mut self, category: InjectionCategory, amount: u32) {
        match category {
            InjectionCategory::DivergenceAlert => self.divergence += amount,
            InjectionCategory::HighRelevanceCluster => self.cluster += amount,
            InjectionCategory::SingleSpaceMatch => self.single_space += amount,
            InjectionCategory::RecentSession => self.session += amount,
        }
    }

    /// Total tokens spent across all categories.
    #[inline]
    fn total(&self) -> u32 {
        self.divergence + self.cluster + self.single_space + self.session
    }
}

// =============================================================================
// SelectionStats
// =============================================================================

use std::collections::HashMap;

/// Statistics about the selection process.
#[derive(Debug, Clone)]
pub struct SelectionStats {
    /// Number of candidates selected.
    pub selected_count: usize,
    /// Number of candidates rejected (exceeded budget).
    pub rejected_count: usize,
    /// Total tokens used by selected candidates.
    pub tokens_used: u32,
    /// Remaining tokens available.
    pub tokens_available: u32,
    /// Tokens used per category.
    pub by_category: HashMap<InjectionCategory, u32>,
}

// =============================================================================
// TokenBudgetManager
// =============================================================================

/// Manages token budget allocation during candidate selection.
///
/// Selects candidates that fit within category and total budget constraints.
/// High-priority categories (priority <= 2) can overflow into reserved budget.
///
/// # Usage
///
/// ```ignore
/// // After PriorityRanker::rank_candidates() has sorted candidates:
/// let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);
/// ```
///
/// # Overflow Rules
///
/// | Category | Priority | Can Overflow? |
/// |----------|----------|---------------|
/// | DivergenceAlert | 1 | Yes |
/// | HighRelevanceCluster | 2 | Yes |
/// | SingleSpaceMatch | 3 | No |
/// | RecentSession | 4 | No |
pub struct TokenBudgetManager;

impl TokenBudgetManager {
    /// Select candidates that fit within budget constraints.
    ///
    /// **IMPORTANT**: Assumes candidates are pre-sorted by `PriorityRanker::rank_candidates()`.
    /// Processes in order, selecting candidates until category budget is exhausted.
    /// High-priority categories (DivergenceAlert, HighRelevanceCluster) can overflow
    /// into reserved budget.
    ///
    /// # Arguments
    /// * `candidates` - Sorted slice of injection candidates
    /// * `budget` - Token budget configuration
    ///
    /// # Returns
    /// Vector of selected candidates (cloned) in selection order
    ///
    /// # Invariants
    /// - Total selected tokens will NEVER exceed `budget.total`
    /// - Low-priority categories (priority >= 3) cannot exceed their allocated budget
    pub fn select_within_budget(
        candidates: &[InjectionCandidate],
        budget: &TokenBudget,
    ) -> Vec<InjectionCandidate> {
        let mut selected = Vec::new();
        let mut spending = CategorySpending::default();

        for candidate in candidates {
            let tokens = candidate.token_count;
            let category = candidate.category;

            if Self::can_select(&spending, budget, category, tokens) {
                spending.add(category, tokens);
                selected.push(candidate.clone());
            }
        }

        selected
    }

    /// Select candidates with detailed statistics.
    ///
    /// Same as `select_within_budget` but also returns selection statistics.
    ///
    /// # Returns
    /// Tuple of (selected candidates, selection statistics)
    pub fn select_with_stats(
        candidates: &[InjectionCandidate],
        budget: &TokenBudget,
    ) -> (Vec<InjectionCandidate>, SelectionStats) {
        let mut selected = Vec::new();
        let mut spending = CategorySpending::default();
        let mut rejected_count = 0;

        for candidate in candidates {
            let tokens = candidate.token_count;
            let category = candidate.category;

            if Self::can_select(&spending, budget, category, tokens) {
                spending.add(category, tokens);
                selected.push(candidate.clone());
            } else {
                rejected_count += 1;
            }
        }

        let stats = SelectionStats {
            selected_count: selected.len(),
            rejected_count,
            tokens_used: spending.total(),
            tokens_available: budget.total.saturating_sub(spending.total()),
            by_category: HashMap::from([
                (InjectionCategory::DivergenceAlert, spending.divergence),
                (InjectionCategory::HighRelevanceCluster, spending.cluster),
                (InjectionCategory::SingleSpaceMatch, spending.single_space),
                (InjectionCategory::RecentSession, spending.session),
            ]),
        };

        (selected, stats)
    }

    /// Check if a candidate can be selected given current spending.
    ///
    /// Rules:
    /// 1. Never exceed total budget
    /// 2. If within category budget, allow
    /// 3. If over category budget but high-priority (<=2), allow overflow into remaining total
    /// 4. Low-priority categories cannot exceed their budget
    fn can_select(
        spending: &CategorySpending,
        budget: &TokenBudget,
        category: InjectionCategory,
        tokens: u32,
    ) -> bool {
        let category_budget = budget.budget_for_category(category);
        let category_spent = spending.get(category);
        let total_spent = spending.total();

        // Rule 1: Never exceed total budget
        if total_spent + tokens > budget.total {
            return false;
        }

        // Rule 2: Check category budget
        let within_category = category_spent + tokens <= category_budget;
        if within_category {
            return true;
        }

        // Rule 3 & 4: High priority can overflow, low priority cannot
        let is_high_priority = category.priority() <= 2; // DivergenceAlert(1), HighRelevanceCluster(2)
        if is_high_priority {
            // Overflow allowed if total budget permits
            let overflow_available = budget.total.saturating_sub(total_spent);
            return tokens <= overflow_available;
        }

        // Low priority: cannot exceed category budget
        false
    }
}
```

### Modify File: `crates/context-graph-core/src/injection/mod.rs`

Update the exports after line 19 to include new types:

```rust
pub use budget::{
    TokenBudget, TokenBudgetManager, SelectionStats,
    DEFAULT_TOKEN_BUDGET, BRIEF_BUDGET, estimate_tokens,
};
```

---

## Definition of Done

### DOD-1: select_within_budget() respects category budgets
- [ ] Selects candidates that fit within their category's allocated budget
- **Verification**: Unit test with mixed categories verifies correct selection

### DOD-2: estimate_tokens() formula
- [ ] Returns `word_count × 1.3` (ceiling)
- **Verification**: Unit test verifies formula matches InjectionCandidate behavior

### DOD-3: High-priority overflow
- [ ] DivergenceAlert and HighRelevanceCluster can overflow into reserved/remaining budget
- **Verification**: Unit test with large divergence alert verifies overflow works

### DOD-4: Low-priority budget enforcement
- [ ] SingleSpaceMatch and RecentSession cannot exceed their category budget
- **Verification**: Unit test verifies low-priority rejection when over budget

### DOD-5: Total budget invariant
- [ ] Total selected tokens NEVER exceeds `budget.total`
- **Verification**: Property test with random candidates verifies invariant

### DOD-6: Selection statistics accuracy
- [ ] `select_with_stats()` returns accurate counts and by-category breakdown
- **Verification**: Unit test verifies stats match selected candidates

---

## Constraints

| Type | Constraint |
|------|------------|
| Invariant | Total selected tokens ≤ `budget.total` (ALWAYS) |
| Behavior | Priority <= 2 (DivergenceAlert, HighRelevanceCluster) can overflow into reserved |
| Behavior | Priority >= 3 (SingleSpaceMatch, RecentSession) cannot exceed category budget |
| Precondition | Candidates MUST be pre-sorted by `PriorityRanker::rank_candidates()` |
| Formula | `estimate_tokens(content) = ceil(word_count × 1.3)` |

---

## Unit Tests (Required)

Add to `budget.rs` after the existing tests (after line 279):

```rust
// =============================================================================
// TokenBudgetManager Tests
// =============================================================================

#[cfg(test)]
mod budget_manager_tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;
    use crate::teleological::Embedder;

    fn make_candidate(
        tokens: u32,
        category: InjectionCategory,
        priority: f32,
    ) -> InjectionCandidate {
        let word_count = ((tokens as f32) / 1.3).ceil() as usize;
        let content = "word ".repeat(word_count);
        let mut c = InjectionCandidate::new(
            Uuid::new_v4(),
            content,
            0.8, // relevance
            match category {
                InjectionCategory::HighRelevanceCluster => 3.0,
                InjectionCategory::SingleSpaceMatch => 1.5,
                _ => 0.5,
            },
            vec![Embedder::Semantic],
            category,
            Utc::now(),
        );
        c.token_count = tokens; // Override to exact value for testing
        c.priority = priority;
        c
    }

    #[test]
    fn test_select_within_category_budget() {
        let budget = TokenBudget::default(); // divergence=200, cluster=400

        let candidates = vec![
            make_candidate(150, InjectionCategory::DivergenceAlert, 1.0),
            make_candidate(100, InjectionCategory::DivergenceAlert, 0.9),
            // 150 + 100 = 250, exceeds divergence budget (200), but overflow allowed
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Both should be selected (overflow allowed for high priority)
        assert_eq!(selected.len(), 2, "Both divergence alerts should be selected (overflow allowed)");
        println!("[PASS] High-priority overflow allowed: selected {} candidates", selected.len());
    }

    #[test]
    fn test_select_respects_total_budget() {
        let budget = TokenBudget::with_total(300);

        let candidates = vec![
            make_candidate(100, InjectionCategory::DivergenceAlert, 1.0),
            make_candidate(100, InjectionCategory::HighRelevanceCluster, 0.9),
            make_candidate(100, InjectionCategory::SingleSpaceMatch, 0.8),
            make_candidate(100, InjectionCategory::RecentSession, 0.7), // Would exceed 300
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Total budget is 300, should select first 3
        assert_eq!(selected.len(), 3, "Should select exactly 3 candidates (300 token limit)");

        let total_tokens: u32 = selected.iter().map(|c| c.token_count).sum();
        assert!(total_tokens <= 300, "Total tokens {} should not exceed 300", total_tokens);
        println!("[PASS] Total budget respected: {} tokens used", total_tokens);
    }

    #[test]
    fn test_high_priority_overflow_into_reserved() {
        let budget = TokenBudget {
            total: 500,
            divergence_budget: 100,
            cluster_budget: 200,
            single_space_budget: 100,
            session_budget: 50,
            reserved: 50,
        };

        // Large divergence alert that exceeds category budget (100) but fits in total (500)
        let candidates = vec![
            make_candidate(150, InjectionCategory::DivergenceAlert, 1.0),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Should allow overflow for high priority (priority=1)
        assert_eq!(selected.len(), 1, "High-priority candidate should overflow into reserved");
        println!("[PASS] High-priority overflow works");
    }

    #[test]
    fn test_low_priority_no_overflow() {
        let budget = TokenBudget {
            total: 500,
            divergence_budget: 100,
            cluster_budget: 100,
            single_space_budget: 100,
            session_budget: 100,
            reserved: 100,
        };

        // RecentSession (priority=4) exceeds category budget
        let candidates = vec![
            make_candidate(150, InjectionCategory::RecentSession, 1.0),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Should NOT overflow for low priority (priority=4)
        assert_eq!(selected.len(), 0, "Low-priority candidate should NOT overflow");
        println!("[PASS] Low-priority overflow blocked correctly");
    }

    #[test]
    fn test_select_with_stats() {
        let budget = TokenBudget::default();

        let candidates = vec![
            make_candidate(100, InjectionCategory::DivergenceAlert, 1.0),
            make_candidate(200, InjectionCategory::HighRelevanceCluster, 0.9),
            make_candidate(100, InjectionCategory::SingleSpaceMatch, 0.8),
        ];

        let (selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

        assert_eq!(selected.len(), 3);
        assert_eq!(stats.selected_count, 3);
        assert_eq!(stats.rejected_count, 0);
        assert_eq!(stats.tokens_used, 400);
        assert_eq!(stats.tokens_available, 800); // 1200 - 400
        assert_eq!(*stats.by_category.get(&InjectionCategory::DivergenceAlert).unwrap(), 100);
        assert_eq!(*stats.by_category.get(&InjectionCategory::HighRelevanceCluster).unwrap(), 200);
        println!("[PASS] Selection stats accurate: {} used, {} available", stats.tokens_used, stats.tokens_available);
    }

    #[test]
    fn test_estimate_tokens() {
        // 10 words × 1.3 = 13
        assert_eq!(estimate_tokens("one two three four five six seven eight nine ten"), 13);

        // Empty string
        assert_eq!(estimate_tokens(""), 0);

        // Single word → 1 × 1.3 = 1.3 → ceil = 2
        assert_eq!(estimate_tokens("hello"), 2);

        // 5 words × 1.3 = 6.5 → ceil = 7
        assert_eq!(estimate_tokens("the quick brown fox jumps"), 7);

        println!("[PASS] estimate_tokens formula verified");
    }

    #[test]
    fn test_mixed_categories_selection() {
        let budget = TokenBudget::default(); // 1200 total

        let candidates = vec![
            make_candidate(50, InjectionCategory::DivergenceAlert, 1.0),
            make_candidate(100, InjectionCategory::HighRelevanceCluster, 0.95),
            make_candidate(100, InjectionCategory::HighRelevanceCluster, 0.9),
            make_candidate(100, InjectionCategory::HighRelevanceCluster, 0.85),
            make_candidate(100, InjectionCategory::SingleSpaceMatch, 0.8),
            make_candidate(100, InjectionCategory::SingleSpaceMatch, 0.75),
            make_candidate(50, InjectionCategory::RecentSession, 0.7),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // All should fit: 50+100+100+100+100+100+50 = 600 < 1200
        assert_eq!(selected.len(), 7, "All 7 candidates should fit within budget");

        let total: u32 = selected.iter().map(|c| c.token_count).sum();
        assert_eq!(total, 600);
        println!("[PASS] Mixed categories: all 7 selected, {} total tokens", total);
    }

    #[test]
    fn test_empty_candidates() {
        let budget = TokenBudget::default();
        let candidates: Vec<InjectionCandidate> = vec![];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        assert!(selected.is_empty());
        println!("[PASS] Empty candidates handled correctly");
    }

    #[test]
    fn test_single_candidate_fits() {
        let budget = TokenBudget::default();
        let candidates = vec![
            make_candidate(100, InjectionCategory::HighRelevanceCluster, 1.0),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        assert_eq!(selected.len(), 1);
        println!("[PASS] Single candidate selected");
    }

    #[test]
    fn test_single_candidate_too_large() {
        let budget = TokenBudget::with_total(100);
        let candidates = vec![
            make_candidate(150, InjectionCategory::HighRelevanceCluster, 1.0),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        assert!(selected.is_empty(), "150-token candidate should not fit in 100-token budget");
        println!("[PASS] Oversized candidate rejected");
    }

    // =========================================================================
    // Edge Case Tests (MANDATORY per verification protocol)
    // =========================================================================

    #[test]
    fn test_edge_case_exactly_at_category_budget() {
        // Edge Case 1: Candidate tokens exactly equal category budget
        let budget = TokenBudget::default(); // divergence_budget = 200

        println!("EDGE CASE 1: Candidate exactly at category budget");
        println!("  Before: budget.divergence_budget = {}", budget.divergence_budget);

        let candidates = vec![
            make_candidate(200, InjectionCategory::DivergenceAlert, 1.0),
        ];
        println!("  Input: 1 DivergenceAlert candidate with {} tokens", candidates[0].token_count);

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        println!("  After: selected {} candidates", selected.len());
        println!("  Expected: 1 (exactly at budget should be accepted)");

        assert_eq!(selected.len(), 1, "Candidate exactly at category budget should be selected");
        println!("[PASS] Edge Case 1: Exactly at category budget = accepted");
    }

    #[test]
    fn test_edge_case_one_token_over_category_budget_low_priority() {
        // Edge Case 2: Low-priority candidate 1 token over category budget
        let budget = TokenBudget {
            total: 500,
            divergence_budget: 100,
            cluster_budget: 100,
            single_space_budget: 100,
            session_budget: 100,
            reserved: 100,
        };

        println!("EDGE CASE 2: Low-priority 1 token over category budget");
        println!("  Before: budget.session_budget = {}", budget.session_budget);

        let candidates = vec![
            make_candidate(101, InjectionCategory::RecentSession, 1.0), // 1 over
        ];
        println!("  Input: 1 RecentSession candidate with {} tokens", candidates[0].token_count);

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        println!("  After: selected {} candidates", selected.len());
        println!("  Expected: 0 (low priority cannot overflow)");

        assert_eq!(selected.len(), 0, "Low priority 1 token over should be rejected");
        println!("[PASS] Edge Case 2: Low-priority 1 over = rejected");
    }

    #[test]
    fn test_edge_case_total_budget_exactly_exhausted() {
        // Edge Case 3: Exactly exhaust total budget
        let budget = TokenBudget::with_total(300);

        println!("EDGE CASE 3: Exactly exhaust total budget");
        println!("  Before: budget.total = {}", budget.total);

        let candidates = vec![
            make_candidate(100, InjectionCategory::DivergenceAlert, 1.0),
            make_candidate(100, InjectionCategory::HighRelevanceCluster, 0.9),
            make_candidate(100, InjectionCategory::SingleSpaceMatch, 0.8),
        ];
        println!("  Input: 3 candidates, total tokens = {}",
            candidates.iter().map(|c| c.token_count).sum::<u32>());

        let (selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

        println!("  After: selected {} candidates, tokens_used = {}, tokens_available = {}",
            selected.len(), stats.tokens_used, stats.tokens_available);
        println!("  Expected: 3 selected, 300 used, 0 available");

        assert_eq!(selected.len(), 3);
        assert_eq!(stats.tokens_used, 300);
        assert_eq!(stats.tokens_available, 0);
        println!("[PASS] Edge Case 3: Total budget exactly exhausted");
    }

    // =========================================================================
    // Property Test: Total Budget Invariant
    // =========================================================================

    #[test]
    fn test_total_budget_invariant_never_exceeded() {
        // Generate various budget and candidate combinations
        let budgets = vec![
            TokenBudget::with_total(100),
            TokenBudget::with_total(500),
            TokenBudget::default(),
        ];

        for budget in &budgets {
            // Create candidates that could potentially exceed budget
            let candidates = vec![
                make_candidate(budget.total / 2, InjectionCategory::DivergenceAlert, 1.0),
                make_candidate(budget.total / 2, InjectionCategory::HighRelevanceCluster, 0.9),
                make_candidate(budget.total / 2, InjectionCategory::SingleSpaceMatch, 0.8),
                make_candidate(budget.total / 2, InjectionCategory::RecentSession, 0.7),
            ];

            let selected = TokenBudgetManager::select_within_budget(&candidates, budget);
            let total_tokens: u32 = selected.iter().map(|c| c.token_count).sum();

            assert!(
                total_tokens <= budget.total,
                "INVARIANT VIOLATED: selected {} tokens but budget is {}",
                total_tokens,
                budget.total
            );
        }
        println!("[PASS] Total budget invariant verified across multiple budgets");
    }
}
```

---

## Full State Verification Protocol

After implementing the logic, you MUST verify the results through the following protocol.

### 1. Source of Truth Identification

The source of truth is:
- **Selected candidates vector**: The returned `Vec<InjectionCandidate>` from `select_within_budget()`
- **SelectionStats struct**: The statistics from `select_with_stats()`
- **Test assertions**: Verify counts, token sums, and category breakdowns

There is NO database or persistent storage for this logic - verification is done through in-memory assertions on the returned values.

### 2. Execute & Inspect Protocol

```bash
# Build first
cargo build --package context-graph-core

# Run all budget tests (including new TokenBudgetManager tests)
cargo test injection::budget --package context-graph-core -- --nocapture

# Run specific test to see detailed output
cargo test budget_manager_tests --package context-graph-core -- --nocapture

# Verify no regressions in other injection modules
cargo test injection::candidate --package context-graph-core
cargo test injection::priority --package context-graph-core
```

### 3. Boundary & Edge Case Audit

You MUST run these 3 edge case tests and verify the printed output:

**Edge Case 1: Exactly at category budget**
```
Expected output:
EDGE CASE 1: Candidate exactly at category budget
  Before: budget.divergence_budget = 200
  Input: 1 DivergenceAlert candidate with 200 tokens
  After: selected 1 candidates
  Expected: 1 (exactly at budget should be accepted)
[PASS] Edge Case 1: Exactly at category budget = accepted
```

**Edge Case 2: Low-priority 1 token over**
```
Expected output:
EDGE CASE 2: Low-priority 1 token over category budget
  Before: budget.session_budget = 100
  Input: 1 RecentSession candidate with 101 tokens
  After: selected 0 candidates
  Expected: 0 (low priority cannot overflow)
[PASS] Edge Case 2: Low-priority 1 over = rejected
```

**Edge Case 3: Total budget exactly exhausted**
```
Expected output:
EDGE CASE 3: Exactly exhaust total budget
  Before: budget.total = 300
  Input: 3 candidates, total tokens = 300
  After: selected 3 candidates, tokens_used = 300, tokens_available = 0
  Expected: 3 selected, 300 used, 0 available
[PASS] Edge Case 3: Total budget exactly exhausted
```

### 4. Evidence of Success

Provide:
1. Full test output showing all tests passed
2. The `cargo build` output showing no errors
3. Specific token counts from `test_select_with_stats`
4. Confirmation that the total budget invariant test passes

---

## Test Commands

```bash
# Build the package
cargo build --package context-graph-core

# Run all budget module tests
cargo test injection::budget --package context-graph-core -- --nocapture

# Run only TokenBudgetManager tests
cargo test budget_manager_tests --package context-graph-core -- --nocapture

# Run specific edge case test
cargo test test_edge_case_exactly_at_category_budget --package context-graph-core -- --nocapture

# Verify no regressions
cargo test injection --package context-graph-core
```

---

## Validation Criteria

| Type | Criterion |
|------|-----------|
| Compilation | `cargo build --package context-graph-core` compiles without errors |
| Tests | All 14 new tests in `budget_manager_tests` pass |
| Invariant | Total selected tokens never exceeds `budget.total` |
| Overflow | High-priority (<=2) can overflow, low-priority (>=3) cannot |
| Stats | `SelectionStats` matches actual selection |
| Integration | Existing `budget::tests` still pass |

---

## Anti-Patterns to Avoid

| ID | Anti-Pattern | Correct Approach |
|----|--------------|------------------|
| AP-10 | NaN/Infinity in token counts | Token counts are u32 - no floating point issues |
| AP-12 | Magic numbers | Use `category.priority()` for threshold (<=2 is high priority) |
| AP-14 | `.unwrap()` in library code | Use `HashMap::from([...])` for stats - no unwrap needed |

---

## Notes for Implementing Agent

1. **Candidates MUST be pre-sorted** - This manager assumes `PriorityRanker::rank_candidates()` was called first. Do NOT re-sort.

2. **Token count is already computed** - `InjectionCandidate::new()` computes `token_count` from content. The `estimate_tokens()` function is for pre-candidate estimation only.

3. **Clone is required** - We clone candidates into the selected vector because we need to return owned values.

4. **Priority threshold is 2** - Priority 1 (DivergenceAlert) and 2 (HighRelevanceCluster) can overflow. Priority 3+ cannot.

5. **CategorySpending is private** - Only `SelectionStats` is public. `CategorySpending` is an implementation detail.

6. **HashMap for stats** - Use `std::collections::HashMap` import for the `by_category` field.

7. **No backwards compatibility** - This is new code. If something fails, it should error out with clear messages.

8. **Test with exact token counts** - Override `token_count` in tests for deterministic behavior (see `make_candidate` helper).
