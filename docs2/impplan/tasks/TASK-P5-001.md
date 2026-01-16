# Task: TASK-P5-001 - InjectionCandidate and InjectionCategory Types

```xml
<task_spec id="TASK-P5-001" version="1.0">
<metadata>
  <title>InjectionCandidate and InjectionCategory Types</title>
  <phase>5</phase>
  <sequence>36</sequence>
  <layer>foundation</layer>
  <estimated_loc>120</estimated_loc>
  <dependencies>
    <dependency task="TASK-P1-001">MemoryId type</dependency>
    <dependency task="TASK-P2-001">Embedder enum</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">InjectionCandidate</artifact>
    <artifact type="enum">InjectionCategory</artifact>
  </produces>
</metadata>

<context>
  <background>
    The injection pipeline needs to track candidate memories for context injection
    with their computed scores, token estimates, and categorization. Each candidate
    carries all information needed for priority ranking and budget selection.
  </background>
  <business_value>
    Enables priority-based memory selection with category-specific budgets and
    multi-factor relevance scoring for intelligent context injection.
  </business_value>
  <technical_context>
    InjectionCandidate is the central data structure flowing through the entire
    injection pipeline. InjectionCategory determines budget allocation and sort
    order during selection.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-core/src/memory/types.rs with MemoryId</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/embedding/embedder.rs with Embedder enum</prerequisite>
  <prerequisite type="directory">crates/context-graph-core/src/injection/</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>InjectionCandidate struct with all fields</item>
    <item>InjectionCategory enum with priority variants</item>
    <item>InjectionCandidate::new() constructor</item>
    <item>InjectionCategory::priority() method</item>
    <item>Ord/PartialOrd implementations for sorting</item>
    <item>Unit tests for type construction and ordering</item>
  </includes>
  <excludes>
    <item>Priority computation logic (TASK-P5-004)</item>
    <item>Token estimation logic (TASK-P5-005)</item>
    <item>Formatting logic (TASK-P5-006)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>InjectionCandidate struct with all fields defined</description>
    <verification>cargo build --package context-graph-core</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>InjectionCategory enum with 4 variants and priority method</description>
    <verification>Unit test verifies priority ordering</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>InjectionCandidate sorts by category then priority</description>
    <verification>cargo test injection::candidate::tests</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>All constraints validated in constructor</description>
    <verification>Test invalid inputs panic or error</verification>
  </criterion>

  <signatures>
    <signature name="InjectionCandidate">
      <code>
pub struct InjectionCandidate {
    pub memory_id: Uuid,
    pub content: String,
    pub relevance_score: f32,         // 0.0..=1.0
    pub recency_factor: f32,          // 0.0..=2.0
    pub diversity_bonus: f32,         // 1.0..=1.5
    pub weighted_agreement: f32,      // 0.0..=8.5 (for topic matching)
    pub matching_spaces: Vec&lt;Embedder&gt;,
    pub priority: f32,                // computed
    pub token_count: u32,             // estimated
    pub category: InjectionCategory,
    pub created_at: DateTime&lt;Utc&gt;,
}
      </code>
    </signature>
    <signature name="InjectionCategory">
      <code>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionCategory {
    DivergenceAlert,      // priority 1 (highest)
    HighRelevanceCluster, // priority 2 (3+ spaces match)
    SingleSpaceMatch,     // priority 3 (1-2 spaces match)
    RecentSession,        // priority 4 (last session summary)
}
      </code>
    </signature>
    <signature name="new">
      <code>
impl InjectionCandidate {
    pub fn new(
        memory_id: Uuid,
        content: String,
        relevance_score: f32,
        matching_spaces: Vec&lt;Embedder&gt;,
        category: InjectionCategory,
        created_at: DateTime&lt;Utc&gt;,
    ) -> Self
}
      </code>
    </signature>
    <signature name="priority">
      <code>
impl InjectionCategory {
    pub fn priority(&amp;self) -> u8
}
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="validation">relevance_score must be in 0.0..=1.0</constraint>
    <constraint type="validation">recency_factor initialized to 1.0, computed later</constraint>
    <constraint type="validation">diversity_bonus initialized to 1.0, computed later</constraint>
    <constraint type="validation">weighted_agreement in 0.0..=8.5 (used for diversity_bonus calculation)</constraint>
    <constraint type="validation">priority initialized to 0.0, computed later</constraint>
    <constraint type="invariant">DivergenceAlert.priority() &lt; HighRelevanceCluster.priority()</constraint>
    <constraint type="invariant">Lower priority number = higher rank</constraint>
    <constraint type="invariant">HighRelevanceCluster requires weighted_agreement >= 2.5</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-core/src/injection/candidate.rs

use chrono::{DateTime, Utc};
use uuid::Uuid;
use crate::embedding::Embedder;

/// Category of injection candidate determining budget and sort order.
/// Lower priority number = higher rank (processed first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionCategory {
    /// Divergence alerts have highest priority (budget: 200 tokens)
    DivergenceAlert,
    /// High relevance cluster matches (3+ spaces, budget: 400 tokens)
    HighRelevanceCluster,
    /// Single space matches (1-2 spaces, budget: 300 tokens)
    SingleSpaceMatch,
    /// Recent session summary (budget: 200 tokens)
    RecentSession,
}

impl InjectionCategory {
    /// Returns priority rank (1 = highest, 4 = lowest).
    pub fn priority(&self) -> u8 {
        match self {
            InjectionCategory::DivergenceAlert => 1,
            InjectionCategory::HighRelevanceCluster => 2,
            InjectionCategory::SingleSpaceMatch => 3,
            InjectionCategory::RecentSession => 4,
        }
    }
}

impl Ord for InjectionCategory {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower priority number = higher rank
        self.priority().cmp(&other.priority())
    }
}

impl PartialOrd for InjectionCategory {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A candidate memory for context injection with computed priority scores.
#[derive(Debug, Clone)]
pub struct InjectionCandidate {
    /// Memory identifier
    pub memory_id: Uuid,
    /// Memory content text
    pub content: String,
    /// Base relevance score from similarity search (0.0..=1.0)
    pub relevance_score: f32,
    /// Time-based multiplier (0.0..=2.0), computed by PriorityRanker
    pub recency_factor: f32,
    /// Multi-space match multiplier (1.0..=1.5), computed by PriorityRanker
    pub diversity_bonus: f32,
    /// Weighted agreement score (0.0..=8.5) from topic clustering
    /// Uses embedder category weights: semantic=1.0, temporal=0.0, relational=0.5, structural=0.5
    pub weighted_agreement: f32,
    /// Which embedding spaces matched threshold
    pub matching_spaces: Vec<Embedder>,
    /// Final priority = relevance × recency × diversity
    pub priority: f32,
    /// Estimated token count for budget tracking
    pub token_count: u32,
    /// Category determines budget pool and sort order
    pub category: InjectionCategory,
    /// When memory was created
    pub created_at: DateTime<Utc>,
}

impl InjectionCandidate {
    /// Create new candidate with initial scores.
    /// recency_factor, diversity_bonus, priority are computed later.
    pub fn new(
        memory_id: Uuid,
        content: String,
        relevance_score: f32,
        weighted_agreement: f32,
        matching_spaces: Vec<Embedder>,
        category: InjectionCategory,
        created_at: DateTime<Utc>,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&relevance_score),
            "relevance_score must be 0.0..=1.0, got {}",
            relevance_score
        );
        assert!(
            (0.0..=8.5).contains(&weighted_agreement),
            "weighted_agreement must be 0.0..=8.5, got {}",
            weighted_agreement
        );

        // Estimate tokens: words × 1.3
        let word_count = content.split_whitespace().count();
        let token_count = (word_count as f32 * 1.3).ceil() as u32;

        Self {
            memory_id,
            content,
            relevance_score,
            recency_factor: 1.0,       // default, computed later
            diversity_bonus: 1.0,      // default, computed later
            weighted_agreement,
            matching_spaces,
            priority: 0.0,             // computed later
            token_count,
            category,
            created_at,
        }
    }

    /// Set computed priority factors.
    pub fn set_priority_factors(&mut self, recency: f32, diversity: f32) {
        assert!(
            (0.0..=2.0).contains(&recency),
            "recency_factor must be 0.0..=2.0, got {}",
            recency
        );
        assert!(
            (1.0..=1.5).contains(&diversity),
            "diversity_bonus must be 1.0..=1.5, got {}",
            diversity
        );

        self.recency_factor = recency;
        self.diversity_bonus = diversity;
        self.priority = self.relevance_score * recency * diversity;
    }
}

impl Ord for InjectionCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First by category (lower priority number first)
        match self.category.cmp(&other.category) {
            std::cmp::Ordering::Equal => {
                // Within category, by priority descending (higher priority first)
                other.priority
                    .partial_cmp(&self.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            ordering => ordering,
        }
    }
}

impl PartialOrd for InjectionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for InjectionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.memory_id == other.memory_id
    }
}

impl Eq for InjectionCandidate {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(
        relevance: f32,
        category: InjectionCategory,
        priority: f32,
    ) -> InjectionCandidate {
        let mut c = InjectionCandidate::new(
            Uuid::new_v4(),
            "test content".to_string(),
            relevance,
            vec![],
            category,
            Utc::now(),
        );
        c.priority = priority;
        c
    }

    #[test]
    fn test_category_priority_order() {
        assert!(InjectionCategory::DivergenceAlert.priority() <
                InjectionCategory::HighRelevanceCluster.priority());
        assert!(InjectionCategory::HighRelevanceCluster.priority() <
                InjectionCategory::SingleSpaceMatch.priority());
        assert!(InjectionCategory::SingleSpaceMatch.priority() <
                InjectionCategory::RecentSession.priority());
    }

    #[test]
    fn test_candidate_sorting_by_category() {
        let mut candidates = vec![
            make_candidate(0.9, InjectionCategory::SingleSpaceMatch, 0.9),
            make_candidate(0.8, InjectionCategory::DivergenceAlert, 0.8),
            make_candidate(0.7, InjectionCategory::RecentSession, 0.7),
            make_candidate(0.85, InjectionCategory::HighRelevanceCluster, 0.85),
        ];

        candidates.sort();

        assert_eq!(candidates[0].category, InjectionCategory::DivergenceAlert);
        assert_eq!(candidates[1].category, InjectionCategory::HighRelevanceCluster);
        assert_eq!(candidates[2].category, InjectionCategory::SingleSpaceMatch);
        assert_eq!(candidates[3].category, InjectionCategory::RecentSession);
    }

    #[test]
    fn test_candidate_sorting_within_category() {
        let mut candidates = vec![
            make_candidate(0.7, InjectionCategory::HighRelevanceCluster, 0.7),
            make_candidate(0.9, InjectionCategory::HighRelevanceCluster, 0.9),
            make_candidate(0.8, InjectionCategory::HighRelevanceCluster, 0.8),
        ];

        candidates.sort();

        // Within same category, sorted by priority descending
        assert!((candidates[0].priority - 0.9).abs() < 0.001);
        assert!((candidates[1].priority - 0.8).abs() < 0.001);
        assert!((candidates[2].priority - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_token_estimation() {
        let c = InjectionCandidate::new(
            Uuid::new_v4(),
            "one two three four five".to_string(), // 5 words
            0.5,
            vec![],
            InjectionCategory::SingleSpaceMatch,
            Utc::now(),
        );

        // 5 words × 1.3 = 6.5 → ceil = 7
        assert_eq!(c.token_count, 7);
    }

    #[test]
    #[should_panic(expected = "relevance_score must be 0.0..=1.0")]
    fn test_invalid_relevance_score() {
        InjectionCandidate::new(
            Uuid::new_v4(),
            "test".to_string(),
            1.5, // Invalid
            vec![],
            InjectionCategory::SingleSpaceMatch,
            Utc::now(),
        );
    }

    #[test]
    fn test_set_priority_factors() {
        let mut c = InjectionCandidate::new(
            Uuid::new_v4(),
            "test".to_string(),
            0.8,
            vec![],
            InjectionCategory::SingleSpaceMatch,
            Utc::now(),
        );

        c.set_priority_factors(1.2, 1.3);

        assert!((c.recency_factor - 1.2).abs() < 0.001);
        assert!((c.diversity_bonus - 1.3).abs() < 0.001);
        // priority = 0.8 × 1.2 × 1.3 = 1.248
        assert!((c.priority - 1.248).abs() < 0.001);
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/injection/candidate.rs">
    InjectionCandidate struct and InjectionCategory enum
  </file>
  <file path="crates/context-graph-core/src/injection/mod.rs">
    Module exports (create if not exists)
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/lib.rs">
    Add pub mod injection
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-core compiles without errors</criterion>
  <criterion type="test">cargo test injection::candidate::tests -- all 6 tests pass</criterion>
  <criterion type="constraint">InjectionCategory variants maintain priority ordering invariant</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-core</command>
  <command>cargo test injection::candidate::tests --package context-graph-core</command>
</test_commands>
</task_spec>
```
