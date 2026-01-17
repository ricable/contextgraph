# Task: TASK-P5-002 - TokenBudget Type

## Task Overview
**Status**: COMPLETE ✅
**Phase**: 5 (Injection Pipeline)
**Sequence**: 37
**Layer**: foundation
**Estimated LOC**: 120 (including tests)
**Last Audit Date**: 2026-01-17
**Completion Date**: 2026-01-17

## CRITICAL: Pre-Implementation Checklist

Before writing ANY code:
1. Read this ENTIRE document
2. Verify dependency TASK-P5-001 is COMPLETE (it is - see Evidence section)
3. Run `cargo build --package context-graph-core` to confirm clean baseline
4. Understand that TokenBudget is a SIMPLE TYPE - no complex logic

---

## Dependency Verification (CONFIRMED COMPLETE)

### TASK-P5-001 Status: COMPLETE ✅
| Artifact | Location | Verified |
|----------|----------|----------|
| `InjectionCategory` enum | `crates/context-graph-core/src/injection/candidate.rs:33-51` | ✅ |
| `InjectionCategory::token_budget()` | `crates/context-graph-core/src/injection/candidate.rs:71-78` | ✅ |
| `InjectionCategory::all()` | `crates/context-graph-core/src/injection/candidate.rs:82-89` | ✅ |
| Re-exports in lib.rs | `crates/context-graph-core/src/lib.rs:106` | ✅ |

**Evidence of TASK-P5-001 Completion:**
```bash
# Run this to verify:
cargo test injection::candidate --package context-graph-core
# Expected: 24 passed, 0 failed
```

---

## Context

### What TokenBudget Does
TokenBudget is a **data structure** that holds token allocation limits for context injection.
It answers: "How many tokens can each injection category use?"

### Constitution Reference
From `constitution.yaml` injection.priorities:
```yaml
P1: { type: "Divergence Alerts", budget: "~200 tokens" }
P2: { type: "High-Relevance Topics", budget: "~400 tokens" }
P3: { type: "Related Memories", budget: "~300 tokens" }
P4: { type: "Recent Context", budget: "~200 tokens" }
# Total: 1100 tokens + ~100 reserved = 1200 total
```

### Why This Matters
- Prevents context overflow by enforcing hard limits
- Ensures balanced representation across categories
- Used by TokenBudgetManager (TASK-P5-005) to track spending

---

## Exact File Paths

### File to CREATE:
```
crates/context-graph-core/src/injection/budget.rs
```

### File to MODIFY:
```
crates/context-graph-core/src/injection/mod.rs
```
Add after line 11:
```rust
pub mod budget;
```
Add to pub use statement (line 13-16):
```rust
pub use budget::{TokenBudget, DEFAULT_TOKEN_BUDGET, BRIEF_BUDGET};
```

### File to MODIFY:
```
crates/context-graph-core/src/lib.rs
```
Update the injection re-exports (line 106) to include:
```rust
pub use injection::{
    InjectionCandidate, InjectionCategory, TokenBudget,
    DEFAULT_TOKEN_BUDGET, BRIEF_BUDGET,
};
```

---

## Implementation Specification

### File: `crates/context-graph-core/src/injection/budget.rs`

```rust
//! Token budget allocation for context injection.
//!
//! Provides the [`TokenBudget`] struct for tracking token allocations
//! across injection categories.
//!
//! # Constitution Compliance
//! - Total budget: 1200 tokens (per injection.priorities)
//! - Category allocations sum to total

use super::candidate::InjectionCategory;

/// Token budget allocation for context injection.
/// Each category has a dedicated budget pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBudget {
    /// Total token limit for all injected context
    pub total: u32,
    /// Budget for divergence alerts (highest priority)
    pub divergence_budget: u32,
    /// Budget for high-relevance cluster matches (weighted_agreement >= 2.5)
    pub cluster_budget: u32,
    /// Budget for single-space matches (weighted_agreement in [1.0, 2.5))
    pub single_space_budget: u32,
    /// Budget for last session summary
    pub session_budget: u32,
    /// Reserved for formatting overhead
    pub reserved: u32,
}

/// Default budget allocation for SessionStart/UserPromptSubmit hooks.
/// Total: 1200 tokens per constitution.yaml.
pub const DEFAULT_TOKEN_BUDGET: TokenBudget = TokenBudget {
    total: 1200,
    divergence_budget: 200,
    cluster_budget: 400,
    single_space_budget: 300,
    session_budget: 200,
    reserved: 100,
};

/// Brief budget for PreToolUse hook.
/// Only 200 tokens for quick context per constitution.yaml.
pub const BRIEF_BUDGET: u32 = 200;

impl TokenBudget {
    /// Create custom budget with specified total.
    /// Allocates proportionally based on default ratios.
    ///
    /// # Arguments
    /// * `total` - Total token budget (must be >= 100)
    ///
    /// # Panics
    /// Panics if `total < 100` (insufficient for meaningful allocation)
    pub fn with_total(total: u32) -> Self {
        assert!(total >= 100, "total must be at least 100, got {}", total);

        // Ratios from DEFAULT_TOKEN_BUDGET:
        // divergence: 200/1200 = 1/6
        // cluster: 400/1200 = 1/3
        // single_space: 300/1200 = 1/4
        // session: 200/1200 = 1/6
        // reserved: remainder
        let divergence_budget = total / 6;
        let cluster_budget = total / 3;
        let single_space_budget = total / 4;
        let session_budget = total / 6;
        let reserved = total - divergence_budget - cluster_budget - single_space_budget - session_budget;

        Self {
            total,
            divergence_budget,
            cluster_budget,
            single_space_budget,
            session_budget,
            reserved,
        }
    }

    /// Get budget allocation for a specific category.
    #[inline]
    pub fn budget_for_category(&self, category: InjectionCategory) -> u32 {
        match category {
            InjectionCategory::DivergenceAlert => self.divergence_budget,
            InjectionCategory::HighRelevanceCluster => self.cluster_budget,
            InjectionCategory::SingleSpaceMatch => self.single_space_budget,
            InjectionCategory::RecentSession => self.session_budget,
        }
    }

    /// Usable budget (total minus reserved).
    #[inline]
    pub fn usable(&self) -> u32 {
        self.total.saturating_sub(self.reserved)
    }

    /// Validate that budgets sum correctly.
    pub fn is_valid(&self) -> bool {
        let sum = self.divergence_budget
            + self.cluster_budget
            + self.single_space_budget
            + self.session_budget
            + self.reserved;
        sum == self.total
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        DEFAULT_TOKEN_BUDGET
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_budget_values() {
        let budget = TokenBudget::default();

        assert_eq!(budget.total, 1200, "Total should be 1200 per constitution");
        assert_eq!(budget.divergence_budget, 200, "Divergence budget should be 200");
        assert_eq!(budget.cluster_budget, 400, "Cluster budget should be 400");
        assert_eq!(budget.single_space_budget, 300, "Single space budget should be 300");
        assert_eq!(budget.session_budget, 200, "Session budget should be 200");
        assert_eq!(budget.reserved, 100, "Reserved should be 100");
        println!("[PASS] Default budget values match constitution");
    }

    #[test]
    fn test_default_budget_sums_correctly() {
        let budget = TokenBudget::default();

        let sum = budget.divergence_budget
            + budget.cluster_budget
            + budget.single_space_budget
            + budget.session_budget
            + budget.reserved;

        assert_eq!(sum, budget.total, "Budgets must sum to total");
        assert!(budget.is_valid(), "is_valid() must return true");
        println!("[PASS] Budget sums correctly: {} = {}", sum, budget.total);
    }

    #[test]
    fn test_budget_for_category() {
        let budget = TokenBudget::default();

        assert_eq!(
            budget.budget_for_category(InjectionCategory::DivergenceAlert),
            200,
            "DivergenceAlert budget"
        );
        assert_eq!(
            budget.budget_for_category(InjectionCategory::HighRelevanceCluster),
            400,
            "HighRelevanceCluster budget"
        );
        assert_eq!(
            budget.budget_for_category(InjectionCategory::SingleSpaceMatch),
            300,
            "SingleSpaceMatch budget"
        );
        assert_eq!(
            budget.budget_for_category(InjectionCategory::RecentSession),
            200,
            "RecentSession budget"
        );
        println!("[PASS] budget_for_category returns correct values");
    }

    #[test]
    fn test_with_total_proportional() {
        let budget = TokenBudget::with_total(600);

        assert_eq!(budget.total, 600);
        assert!(budget.is_valid(), "Custom budget must be valid");

        // Verify proportions roughly maintained
        assert!(budget.cluster_budget >= budget.divergence_budget,
            "Cluster should be >= divergence");
        assert!(budget.cluster_budget >= budget.single_space_budget,
            "Cluster should be >= single_space");

        println!("[PASS] with_total(600) creates valid budget: {:?}", budget);
    }

    #[test]
    fn test_with_total_minimum() {
        let budget = TokenBudget::with_total(100);
        assert_eq!(budget.total, 100);
        assert!(budget.is_valid());
        println!("[PASS] Minimum total (100) accepted");
    }

    #[test]
    #[should_panic(expected = "total must be at least 100")]
    fn test_with_total_below_minimum() {
        TokenBudget::with_total(50);
    }

    #[test]
    fn test_usable_budget() {
        let budget = TokenBudget::default();
        // 1200 - 100 reserved = 1100 usable
        assert_eq!(budget.usable(), 1100, "Usable should be total - reserved");
        println!("[PASS] usable() returns {} (1200 - 100)", budget.usable());
    }

    #[test]
    fn test_brief_budget_constant() {
        assert_eq!(BRIEF_BUDGET, 200, "BRIEF_BUDGET should be 200 for PreToolUse");
        println!("[PASS] BRIEF_BUDGET = 200");
    }

    #[test]
    fn test_default_equals_constant() {
        let budget = TokenBudget::default();
        assert_eq!(budget, DEFAULT_TOKEN_BUDGET, "Default should equal constant");
        println!("[PASS] TokenBudget::default() == DEFAULT_TOKEN_BUDGET");
    }
}
```

---

## Full State Verification Protocol (FSV)

### Source of Truth
- **Primary**: `crates/context-graph-core/src/injection/budget.rs`
- **Module export**: `crates/context-graph-core/src/injection/mod.rs`
- **Crate export**: `crates/context-graph-core/src/lib.rs`

### Execute & Inspect

After implementation, run these commands in sequence:

#### Step 1: Build Verification
```bash
cargo build --package context-graph-core 2>&1 | tee /tmp/build.log
# MUST show: no errors from injection/budget.rs
# Existing warnings in coherence/constants.rs are acceptable
```

#### Step 2: Test Verification
```bash
cargo test injection::budget::tests --package context-graph-core -- --nocapture 2>&1 | tee /tmp/test.log
# MUST show: "test result: ok. 10 passed; 0 failed"
```

#### Step 3: Export Verification
```bash
# Verify file exists
ls -la crates/context-graph-core/src/injection/budget.rs

# Verify module export
grep "pub mod budget" crates/context-graph-core/src/injection/mod.rs

# Verify pub use
grep "TokenBudget" crates/context-graph-core/src/injection/mod.rs
grep "DEFAULT_TOKEN_BUDGET" crates/context-graph-core/src/injection/mod.rs

# Verify lib.rs export
grep "TokenBudget" crates/context-graph-core/src/lib.rs
```

#### Step 4: Integration Test (manual)
```bash
# Create a test file to verify exports work:
cat > /tmp/budget_test.rs << 'EOF'
use context_graph_core::{TokenBudget, DEFAULT_TOKEN_BUDGET, BRIEF_BUDGET, InjectionCategory};

fn main() {
    let budget = TokenBudget::default();
    println!("Total: {}", budget.total);
    println!("Usable: {}", budget.usable());
    println!("Valid: {}", budget.is_valid());

    for cat in InjectionCategory::all() {
        println!("{}: {}", cat, budget.budget_for_category(cat));
    }

    println!("BRIEF_BUDGET: {}", BRIEF_BUDGET);
    println!("DEFAULT == default(): {}", DEFAULT_TOKEN_BUDGET == budget);
}
EOF
```

---

## Edge Case Testing (MANDATORY)

### Edge Case 1: Empty Budget Allocation
**Input**: TokenBudget with all zero budgets except total
**Expected**: is_valid() returns false
**How to test**:
```rust
#[test]
fn test_invalid_budget_detection() {
    let invalid = TokenBudget {
        total: 1200,
        divergence_budget: 0,
        cluster_budget: 0,
        single_space_budget: 0,
        session_budget: 0,
        reserved: 0,
    };
    assert!(!invalid.is_valid(), "Zero allocations should be invalid");
}
```

### Edge Case 2: Minimum Total (100)
**Input**: with_total(100)
**Expected**: Creates valid budget with positive allocations
**Verification**:
```rust
let b = TokenBudget::with_total(100);
assert!(b.divergence_budget > 0);
assert!(b.is_valid());
```

### Edge Case 3: Large Total (1_000_000)
**Input**: with_total(1_000_000)
**Expected**: Creates valid budget, no overflow
**Verification**:
```rust
let b = TokenBudget::with_total(1_000_000);
assert!(b.is_valid());
assert!(b.total == 1_000_000);
```

---

## Anti-Patterns to AVOID

| Rule | Violation | Correct |
|------|-----------|---------|
| AP-14 | Using `.unwrap()` | Use `assert!()` for invariants |
| AP-12 | Magic numbers | Use `DEFAULT_TOKEN_BUDGET` constant |
| New files | Creating extra files | Only create `budget.rs` |

---

## NO Backwards Compatibility

- This is a NEW type in a NEW file
- If tests fail, the implementation is WRONG - fix it
- NO mock data - use real type construction
- NO workarounds - if something breaks, find root cause

---

## Evidence of Success Log Template

After completion, produce this log:

```
================================================================================
TASK-P5-002 VERIFICATION LOG - [timestamp]
================================================================================
BUILD STATUS: [PASS/FAIL]
  - cargo build: [errors] errors, [warnings] warnings

TEST STATUS: [PASS/FAIL]
  - Tests run: 10
  - Tests passed: [count]
  - Tests failed: [count]

FILES CREATED:
  - crates/context-graph-core/src/injection/budget.rs ([✓/✗] exists, [bytes] bytes)

FILES MODIFIED:
  - crates/context-graph-core/src/injection/mod.rs (pub mod budget: [✓/✗])
  - crates/context-graph-core/src/lib.rs (TokenBudget export: [✓/✗])

EXPORTS VERIFIED:
  - TokenBudget from lib.rs: [✓/✗]
  - DEFAULT_TOKEN_BUDGET from lib.rs: [✓/✗]
  - BRIEF_BUDGET from lib.rs: [✓/✗]

EDGE CASES VERIFIED:
  - Minimum total (100): [✓/✗]
  - Default sums correctly: [✓/✗]
  - budget_for_category all variants: [✓/✗]
  - usable() calculation: [✓/✗]

CONSTITUTION COMPLIANCE:
  - Total = 1200: [✓/✗]
  - Divergence = 200: [✓/✗]
  - Cluster = 400: [✓/✗]
  - SingleSpace = 300: [✓/✗]
  - Session = 200: [✓/✗]
  - Reserved = 100: [✓/✗]
================================================================================
```

---

## Acceptance Criteria

- [ ] `crates/context-graph-core/src/injection/budget.rs` exists
- [ ] `pub mod budget;` in injection/mod.rs
- [ ] `pub use budget::*` exports TokenBudget, DEFAULT_TOKEN_BUDGET, BRIEF_BUDGET
- [ ] lib.rs re-exports all three items
- [ ] `cargo build --package context-graph-core` passes (0 new errors/warnings from budget.rs)
- [ ] All 10 unit tests pass
- [ ] FSV protocol executed with evidence log produced
- [ ] Edge cases manually verified
