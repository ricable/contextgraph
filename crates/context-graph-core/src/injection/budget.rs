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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        // reserved: remainder (absorbs integer division rounding)
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

    // =========================================================================
    // Edge Case Tests (MANDATORY per task spec)
    // =========================================================================

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
        println!("[PASS] Invalid budget (zero allocations) detected correctly");
    }

    #[test]
    fn test_zero_reserved_is_valid() {
        let budget = TokenBudget {
            total: 1100,
            divergence_budget: 200,
            cluster_budget: 400,
            single_space_budget: 300,
            session_budget: 200,
            reserved: 0,
        };
        assert!(budget.is_valid(), "Zero reserved should be valid if sums match");
        assert_eq!(budget.usable(), 1100, "Usable should equal total when reserved is 0");
        println!("[PASS] Zero reserved budget is valid");
    }

    #[test]
    fn test_large_total() {
        let budget = TokenBudget::with_total(1_000_000);
        assert!(budget.is_valid());
        assert_eq!(budget.total, 1_000_000);
        println!("[PASS] Large total (1_000_000) accepted and valid: {:?}", budget);
    }

    #[test]
    fn test_minimum_total_has_positive_allocations() {
        let b = TokenBudget::with_total(100);
        assert!(b.divergence_budget > 0, "divergence_budget should be > 0");
        assert!(b.cluster_budget > 0, "cluster_budget should be > 0");
        assert!(b.is_valid());
        println!(
            "[PASS] Minimum total has positive allocations: div={}, cluster={}, single={}, session={}, reserved={}",
            b.divergence_budget, b.cluster_budget, b.single_space_budget, b.session_budget, b.reserved
        );
    }
}
