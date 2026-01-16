# Task: TASK-P5-005 - TokenBudgetManager

```xml
<task_spec id="TASK-P5-005" version="1.0">
<metadata>
  <title>TokenBudgetManager</title>
  <phase>5</phase>
  <sequence>40</sequence>
  <layer>logic</layer>
  <estimated_loc>180</estimated_loc>
  <dependencies>
    <dependency task="TASK-P5-001">InjectionCandidate type</dependency>
    <dependency task="TASK-P5-002">TokenBudget type</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">TokenBudgetManager</artifact>
  </produces>
</metadata>

<context>
  <background>
    TokenBudgetManager selects candidates within budget constraints. It tracks
    spending per category and ensures each category gets fair representation
    while allowing overflow into reserved budget when higher-priority categories
    need more space.
  </background>
  <business_value>
    Enables consistent, predictable context injection that respects token limits
    while ensuring important categories (like divergence alerts) always get included.
  </business_value>
  <technical_context>
    Called after PriorityRanker has sorted candidates. Iterates through sorted
    candidates and selects those that fit within category and total budgets.
    Produces the final list of candidates to be formatted.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-core/src/injection/candidate.rs with InjectionCandidate</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/injection/budget.rs with TokenBudget</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>TokenBudgetManager struct</item>
    <item>select_within_budget() method</item>
    <item>estimate_tokens() function</item>
    <item>Category budget tracking</item>
    <item>Overflow handling to reserved budget</item>
    <item>Unit tests for selection logic</item>
  </includes>
  <excludes>
    <item>Priority computation (TASK-P5-004)</item>
    <item>Formatting (TASK-P5-006)</item>
    <item>tiktoken integration (future enhancement)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>select_within_budget() selects candidates respecting category budgets</description>
    <verification>Unit test with mixed categories verifies correct selection</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>estimate_tokens() returns word_count × 1.3</description>
    <verification>Unit test verifies formula</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>Overflow from high-priority categories into reserved budget works</description>
    <verification>Unit test with large divergence alert verifies overflow</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Total budget never exceeded</description>
    <verification>Property test with random candidates verifies invariant</verification>
  </criterion>

  <signatures>
    <signature name="TokenBudgetManager">
      <code>
pub struct TokenBudgetManager {
    budget: TokenBudget,
    spent: CategorySpending,
}
      </code>
    </signature>
    <signature name="select_within_budget">
      <code>
impl TokenBudgetManager {
    pub fn select_within_budget(
        candidates: &amp;[InjectionCandidate],
        budget: &amp;TokenBudget,
    ) -> Vec&lt;InjectionCandidate&gt;
}
      </code>
    </signature>
    <signature name="estimate_tokens">
      <code>
pub fn estimate_tokens(content: &amp;str) -> u32
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="invariant">Total selected tokens ≤ budget.total</constraint>
    <constraint type="behavior">Higher priority categories can overflow into reserved</constraint>
    <constraint type="behavior">Lower priority categories cannot exceed their budget</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-core/src/injection/budget.rs (add to existing file)

use std::collections::HashMap;
use super::candidate::{InjectionCandidate, InjectionCategory};

/// Tracks spending per category during selection.
#[derive(Debug, Clone, Default)]
struct CategorySpending {
    divergence: u32,
    cluster: u32,
    single_space: u32,
    session: u32,
}

impl CategorySpending {
    fn get(&self, category: InjectionCategory) -> u32 {
        match category {
            InjectionCategory::DivergenceAlert => self.divergence,
            InjectionCategory::HighRelevanceCluster => self.cluster,
            InjectionCategory::SingleSpaceMatch => self.single_space,
            InjectionCategory::RecentSession => self.session,
        }
    }

    fn add(&mut self, category: InjectionCategory, amount: u32) {
        match category {
            InjectionCategory::DivergenceAlert => self.divergence += amount,
            InjectionCategory::HighRelevanceCluster => self.cluster += amount,
            InjectionCategory::SingleSpaceMatch => self.single_space += amount,
            InjectionCategory::RecentSession => self.session += amount,
        }
    }

    fn total(&self) -> u32 {
        self.divergence + self.cluster + self.single_space + self.session
    }
}

/// Manages token budget allocation during candidate selection.
pub struct TokenBudgetManager;

impl TokenBudgetManager {
    /// Select candidates that fit within budget constraints.
    ///
    /// Assumes candidates are pre-sorted by category then priority.
    /// Processes in order, selecting candidates until category budget
    /// is exhausted. High-priority categories can overflow into reserved.
    pub fn select_within_budget(
        candidates: &[InjectionCandidate],
        budget: &TokenBudget,
    ) -> Vec<InjectionCandidate> {
        let mut selected = Vec::new();
        let mut spending = CategorySpending::default();

        for candidate in candidates {
            let tokens = candidate.token_count;
            let category = candidate.category;

            // Check if we can fit this candidate
            if Self::can_select(&spending, budget, category, tokens) {
                spending.add(category, tokens);
                selected.push(candidate.clone());
            }
        }

        selected
    }

    /// Check if a candidate can be selected given current spending.
    fn can_select(
        spending: &CategorySpending,
        budget: &TokenBudget,
        category: InjectionCategory,
        tokens: u32,
    ) -> bool {
        let category_budget = budget.budget_for_category(category);
        let category_spent = spending.get(category);
        let total_spent = spending.total();

        // Check total budget
        if total_spent + tokens > budget.total {
            return false;
        }

        // Check category budget
        let within_category = category_spent + tokens <= category_budget;

        if within_category {
            return true;
        }

        // High priority categories can overflow into reserved
        let can_overflow = category.priority() <= 2; // DivergenceAlert, HighRelevanceCluster
        if can_overflow {
            let overflow_available = budget.total - total_spent;
            return tokens <= overflow_available;
        }

        false
    }

    /// Get selection with detailed stats.
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
            tokens_available: budget.total - spending.total(),
            by_category: HashMap::from([
                (InjectionCategory::DivergenceAlert, spending.divergence),
                (InjectionCategory::HighRelevanceCluster, spending.cluster),
                (InjectionCategory::SingleSpaceMatch, spending.single_space),
                (InjectionCategory::RecentSession, spending.session),
            ]),
        };

        (selected, stats)
    }
}

/// Statistics about selection process.
#[derive(Debug, Clone)]
pub struct SelectionStats {
    pub selected_count: usize,
    pub rejected_count: usize,
    pub tokens_used: u32,
    pub tokens_available: u32,
    pub by_category: HashMap<InjectionCategory, u32>,
}

/// Estimate token count for content.
/// Simple estimate: word_count × 1.3
pub fn estimate_tokens(content: &str) -> u32 {
    let word_count = content.split_whitespace().count();
    (word_count as f32 * 1.3).ceil() as u32
}

#[cfg(test)]
mod budget_manager_tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_candidate(
        tokens: u32,
        category: InjectionCategory,
        priority: f32,
    ) -> InjectionCandidate {
        let content = "word ".repeat(tokens as usize); // Approximate
        let mut c = InjectionCandidate::new(
            Uuid::new_v4(),
            content,
            0.8,
            vec![],
            category,
            Utc::now(),
        );
        c.token_count = tokens; // Override estimate
        c.priority = priority;
        c
    }

    #[test]
    fn test_select_within_category_budget() {
        let budget = TokenBudget::default(); // divergence=200, cluster=400, etc.

        let candidates = vec![
            make_candidate(150, InjectionCategory::DivergenceAlert, 1.0),
            make_candidate(100, InjectionCategory::DivergenceAlert, 0.9),
            // Second divergence alert exceeds 200 budget
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Should select first divergence alert (150 tokens)
        // Second would put us at 250, but overflow into reserved is allowed
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_respects_total_budget() {
        let budget = TokenBudget::with_total(300);

        let candidates = vec![
            make_candidate(100, InjectionCategory::DivergenceAlert, 1.0),
            make_candidate(100, InjectionCategory::HighRelevanceCluster, 0.9),
            make_candidate(100, InjectionCategory::SingleSpaceMatch, 0.8),
            make_candidate(100, InjectionCategory::RecentSession, 0.7),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Total budget is 300, should select first 3
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_high_priority_overflow() {
        let budget = TokenBudget {
            total: 500,
            divergence_budget: 100,
            cluster_budget: 200,
            single_space_budget: 100,
            session_budget: 50,
            reserved: 50,
        };

        // Large divergence alert that overflows category budget
        let candidates = vec![
            make_candidate(150, InjectionCategory::DivergenceAlert, 1.0),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Should allow overflow for high priority category
        assert_eq!(selected.len(), 1);
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

        let candidates = vec![
            make_candidate(150, InjectionCategory::RecentSession, 1.0),
        ];

        let selected = TokenBudgetManager::select_within_budget(&candidates, &budget);

        // Session is low priority (4), should not overflow
        assert_eq!(selected.len(), 0);
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
        assert_eq!(stats.tokens_available, 800);
    }

    #[test]
    fn test_estimate_tokens() {
        // 10 words × 1.3 = 13
        assert_eq!(estimate_tokens("one two three four five six seven eight nine ten"), 13);

        // Empty string
        assert_eq!(estimate_tokens(""), 0);

        // Single word
        assert_eq!(estimate_tokens("hello"), 2); // 1 × 1.3 = 1.3 → ceil = 2
    }

    #[test]
    fn test_mixed_categories_selection() {
        let budget = TokenBudget::default();

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
        assert_eq!(selected.len(), 7);
    }
}
```
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-core/src/injection/budget.rs">
    Add TokenBudgetManager, CategorySpending, estimate_tokens, SelectionStats
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-core compiles without errors</criterion>
  <criterion type="test">cargo test injection::budget_manager_tests -- all 7 tests pass</criterion>
  <criterion type="invariant">No selection ever exceeds total budget</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-core</command>
  <command>cargo test injection::budget --package context-graph-core</command>
</test_commands>
</task_spec>
```
