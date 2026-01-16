# Task: TASK-P5-002 - TokenBudget Type

```xml
<task_spec id="TASK-P5-002" version="1.0">
<metadata>
  <title>TokenBudget Type</title>
  <phase>5</phase>
  <sequence>37</sequence>
  <layer>foundation</layer>
  <estimated_loc>80</estimated_loc>
  <dependencies>
    <dependency task="TASK-P5-001">InjectionCategory for budget categories</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">TokenBudget</artifact>
    <artifact type="const">DEFAULT_TOKEN_BUDGET</artifact>
    <artifact type="const">BRIEF_BUDGET</artifact>
  </produces>
</metadata>

<context>
  <background>
    Token budgets control how much context can be injected into each hook.
    Different categories of memories get different budget allocations to ensure
    balanced representation across divergence alerts, high-relevance clusters,
    single-space matches, and session summaries.
  </background>
  <business_value>
    Prevents context overflow while ensuring important categories always get
    representation in the injected context.
  </business_value>
  <technical_context>
    Used by TokenBudgetManager to track spending during candidate selection.
    Category budgets are soft limits that can overflow into the reserve.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-core/src/injection/candidate.rs with InjectionCategory</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>TokenBudget struct with category budgets</item>
    <item>DEFAULT_TOKEN_BUDGET constant (total: 1200)</item>
    <item>BRIEF_BUDGET constant (200 tokens)</item>
    <item>TokenBudget::default() implementation</item>
    <item>TokenBudget::budget_for_category() method</item>
    <item>TokenBudget::with_total() builder method</item>
    <item>Unit tests for budget allocation</item>
  </includes>
  <excludes>
    <item>Budget management/tracking logic (TASK-P5-005)</item>
    <item>Token estimation logic (TASK-P5-005)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>TokenBudget struct with all budget fields</description>
    <verification>cargo build --package context-graph-core</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>DEFAULT_TOKEN_BUDGET with correct allocations</description>
    <verification>Test verifies total=1200, divergence=200, cluster=400, etc.</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>budget_for_category returns correct budget per category</description>
    <verification>Unit test for each InjectionCategory variant</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Budget allocations sum to total</description>
    <verification>Test verifies all category budgets + reserved = total</verification>
  </criterion>

  <signatures>
    <signature name="TokenBudget">
      <code>
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBudget {
    pub total: u32,
    pub divergence_budget: u32,
    pub cluster_budget: u32,
    pub single_space_budget: u32,
    pub session_budget: u32,
    pub reserved: u32,
}
      </code>
    </signature>
    <signature name="DEFAULT_TOKEN_BUDGET">
      <code>
pub const DEFAULT_TOKEN_BUDGET: TokenBudget = TokenBudget {
    total: 1200,
    divergence_budget: 200,
    cluster_budget: 400,
    single_space_budget: 300,
    session_budget: 200,
    reserved: 100,
};
      </code>
    </signature>
    <signature name="BRIEF_BUDGET">
      <code>
pub const BRIEF_BUDGET: u32 = 200;
      </code>
    </signature>
    <signature name="budget_for_category">
      <code>
impl TokenBudget {
    pub fn budget_for_category(&amp;self, category: InjectionCategory) -> u32
}
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="invariant">divergence + cluster + single_space + session + reserved = total</constraint>
    <constraint type="validation">All budget values must be &gt; 0</constraint>
    <constraint type="spec">DEFAULT_TOKEN_BUDGET.total = 1200 per TECH-PHASE5</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-core/src/injection/budget.rs

use super::candidate::InjectionCategory;

/// Token budget allocation for context injection.
/// Each category has a dedicated budget pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBudget {
    /// Total token limit for all injected context
    pub total: u32,
    /// Budget for divergence alerts (highest priority)
    pub divergence_budget: u32,
    /// Budget for high-relevance cluster matches (3+ spaces)
    pub cluster_budget: u32,
    /// Budget for single-space matches (1-2 spaces)
    pub single_space_budget: u32,
    /// Budget for last session summary
    pub session_budget: u32,
    /// Reserved for formatting overhead
    pub reserved: u32,
}

/// Default budget allocation for SessionStart hook.
/// Total: 1200 tokens per TECH-PHASE5 spec.
pub const DEFAULT_TOKEN_BUDGET: TokenBudget = TokenBudget {
    total: 1200,
    divergence_budget: 200,
    cluster_budget: 400,
    single_space_budget: 300,
    session_budget: 200,
    reserved: 100,
};

/// Brief budget for PreToolUse hook.
/// Only 200 tokens for quick context.
pub const BRIEF_BUDGET: u32 = 200;

impl TokenBudget {
    /// Create custom budget with specified total.
    /// Allocates proportionally based on default ratios.
    pub fn with_total(total: u32) -> Self {
        assert!(total >= 100, "total must be at least 100, got {}", total);

        // Default ratios: divergence=1/6, cluster=1/3, single=1/4, session=1/6, reserved=1/12
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
    pub fn budget_for_category(&self, category: InjectionCategory) -> u32 {
        match category {
            InjectionCategory::DivergenceAlert => self.divergence_budget,
            InjectionCategory::HighRelevanceCluster => self.cluster_budget,
            InjectionCategory::SingleSpaceMatch => self.single_space_budget,
            InjectionCategory::RecentSession => self.session_budget,
        }
    }

    /// Usable budget (total minus reserved).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_budget_values() {
        let budget = TokenBudget::default();

        assert_eq!(budget.total, 1200);
        assert_eq!(budget.divergence_budget, 200);
        assert_eq!(budget.cluster_budget, 400);
        assert_eq!(budget.single_space_budget, 300);
        assert_eq!(budget.session_budget, 200);
        assert_eq!(budget.reserved, 100);
    }

    #[test]
    fn test_default_budget_sums_correctly() {
        let budget = TokenBudget::default();

        let sum = budget.divergence_budget
            + budget.cluster_budget
            + budget.single_space_budget
            + budget.session_budget
            + budget.reserved;

        assert_eq!(sum, budget.total);
        assert!(budget.is_valid());
    }

    #[test]
    fn test_budget_for_category() {
        let budget = TokenBudget::default();

        assert_eq!(
            budget.budget_for_category(InjectionCategory::DivergenceAlert),
            200
        );
        assert_eq!(
            budget.budget_for_category(InjectionCategory::HighRelevanceCluster),
            400
        );
        assert_eq!(
            budget.budget_for_category(InjectionCategory::SingleSpaceMatch),
            300
        );
        assert_eq!(
            budget.budget_for_category(InjectionCategory::RecentSession),
            200
        );
    }

    #[test]
    fn test_with_total_proportional() {
        let budget = TokenBudget::with_total(600);

        assert_eq!(budget.total, 600);
        assert!(budget.is_valid());
        // Proportions should be roughly maintained
        assert!(budget.cluster_budget >= budget.divergence_budget);
    }

    #[test]
    fn test_usable_budget() {
        let budget = TokenBudget::default();

        // 1200 - 100 reserved = 1100 usable
        assert_eq!(budget.usable(), 1100);
    }

    #[test]
    fn test_brief_budget_constant() {
        assert_eq!(BRIEF_BUDGET, 200);
    }

    #[test]
    #[should_panic(expected = "total must be at least 100")]
    fn test_with_total_minimum() {
        TokenBudget::with_total(50);
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/injection/budget.rs">
    TokenBudget struct and constants
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/injection/mod.rs">
    Add pub mod budget; pub use budget::*;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-core compiles without errors</criterion>
  <criterion type="test">cargo test injection::budget::tests -- all 7 tests pass</criterion>
  <criterion type="constraint">DEFAULT_TOKEN_BUDGET.is_valid() returns true</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-core</command>
  <command>cargo test injection::budget::tests --package context-graph-core</command>
</test_commands>
</task_spec>
```
