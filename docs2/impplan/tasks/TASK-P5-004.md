# Task: TASK-P5-004 - PriorityRanker

```xml
<task_spec id="TASK-P5-004" version="1.0">
<metadata>
  <title>PriorityRanker with RecencyFactor and DiversityBonus</title>
  <phase>5</phase>
  <sequence>39</sequence>
  <layer>logic</layer>
  <estimated_loc>150</estimated_loc>
  <dependencies>
    <dependency task="TASK-P5-001">InjectionCandidate type</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">PriorityRanker</artifact>
    <artifact type="struct">RecencyFactor</artifact>
    <artifact type="struct">DiversityBonus</artifact>
  </produces>
</metadata>

<context>
  <background>
    Priority ranking determines which memories get selected for context injection.
    The formula priority = relevance × recency × diversity rewards memories that
    are not just similar but also recent and matched across multiple embedding spaces.
  </background>
  <business_value>
    Ensures context injection includes the most valuable memories by combining
    similarity with temporal relevance and multi-space confirmation.
  </business_value>
  <technical_context>
    RecencyFactor applies time decay to boost recent memories. DiversityBonus
    rewards memories that matched in multiple embedding spaces (more confident
    matches). PriorityRanker orchestrates computation and sorting.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-core/src/injection/candidate.rs with InjectionCandidate</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>RecencyFactor with time-based computation</item>
    <item>DiversityBonus with space-count computation</item>
    <item>PriorityRanker::compute_priority() method</item>
    <item>PriorityRanker::rank_candidates() method</item>
    <item>Unit tests for all computations</item>
  </includes>
  <excludes>
    <item>Budget selection logic (TASK-P5-005)</item>
    <item>Pipeline orchestration (TASK-P5-007)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>RecencyFactor::compute() returns correct factors for all time ranges</description>
    <verification>Unit tests for 1h, 1d, 7d, 30d, older thresholds</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>DiversityBonus::compute() returns correct bonuses for space counts</description>
    <verification>Unit tests for 1, 2, 3, 4, 5+ spaces</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>compute_priority() correctly computes priority = relevance × recency × diversity</description>
    <verification>Unit test with known values verifies formula</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>rank_candidates() sorts by category then priority descending</description>
    <verification>Unit test with mixed categories verifies order</verification>
  </criterion>

  <signatures>
    <signature name="RecencyFactor">
      <code>
pub struct RecencyFactor;

impl RecencyFactor {
    pub const HOUR_1: f32 = 1.3;
    pub const DAY_1: f32 = 1.2;
    pub const WEEK_1: f32 = 1.1;
    pub const MONTH_1: f32 = 1.0;
    pub const OLDER: f32 = 0.8;

    pub fn compute(created_at: DateTime&lt;Utc&gt;) -> f32;
}
      </code>
    </signature>
    <signature name="DiversityBonus">
      <code>
/// Multi-space match multiplier based on weighted_agreement.
/// Uses weighted agreement from topic clustering (not raw space count).
pub struct DiversityBonus;

impl DiversityBonus {
    /// Strong topic signal: weighted_agreement >= 5.0
    pub const STRONG_TOPIC: f32 = 1.5;
    /// Topic threshold met: weighted_agreement >= 2.5
    pub const TOPIC_THRESHOLD: f32 = 1.2;
    /// Related but below topic threshold: weighted_agreement < 2.5
    pub const RELATED: f32 = 1.0;
    /// Weak signal: weighted_agreement < 1.0
    pub const WEAK: f32 = 0.8;

    /// Compute diversity bonus based on weighted_agreement (not raw space count).
    /// weighted_agreement uses embedder category weights:
    ///   semantic=1.0, temporal=0.0, relational=0.5, structural=0.5
    pub fn compute(weighted_agreement: f32) -> f32;
}
      </code>
    </signature>
    <signature name="PriorityRanker">
      <code>
pub struct PriorityRanker;

impl PriorityRanker {
    pub fn compute_priority(candidate: &amp;mut InjectionCandidate);
    pub fn rank_candidates(candidates: &amp;mut [InjectionCandidate]);
}
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="formula">priority = relevance_score × recency_factor × diversity_bonus</constraint>
    <constraint type="range">RecencyFactor output: 0.8..=1.3</constraint>
    <constraint type="range">DiversityBonus output: 0.8..=1.5 (based on weighted_agreement)</constraint>
    <constraint type="threshold">weighted_agreement >= 5.0 → 1.5x (strong topic)</constraint>
    <constraint type="threshold">weighted_agreement >= 2.5 → 1.2x (topic threshold met)</constraint>
    <constraint type="threshold">weighted_agreement >= 1.0 → 1.0x (related)</constraint>
    <constraint type="threshold">weighted_agreement < 1.0 → 0.8x (weak)</constraint>
    <constraint type="sort">Primary sort by category.priority() ascending, secondary by priority descending</constraint>
    <constraint type="category">HighRelevanceCluster requires weighted_agreement >= 2.5</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-core/src/injection/priority.rs

use chrono::{DateTime, Duration, Utc};
use super::candidate::InjectionCandidate;

/// Time-based relevance multiplier.
/// Recent memories get boosted, old ones get penalized.
pub struct RecencyFactor;

impl RecencyFactor {
    /// Within 1 hour: strong boost
    pub const HOUR_1: f32 = 1.3;
    /// Within 1 day: moderate boost
    pub const DAY_1: f32 = 1.2;
    /// Within 1 week: slight boost
    pub const WEEK_1: f32 = 1.1;
    /// Within 1 month: neutral
    pub const MONTH_1: f32 = 1.0;
    /// Older than 1 month: penalty
    pub const OLDER: f32 = 0.8;

    /// Compute recency factor based on memory age.
    pub fn compute(created_at: DateTime<Utc>) -> f32 {
        let now = Utc::now();
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

    /// Compute with explicit reference time (for testing).
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

/// Multi-space match multiplier based on weighted_agreement.
/// Uses weighted agreement from topic clustering, NOT raw space count.
/// Temporal embedders (E2-E4) have weight 0.0 so they don't contribute.
pub struct DiversityBonus;

impl DiversityBonus {
    /// Strong topic signal: weighted_agreement >= 5.0
    pub const STRONG_TOPIC: f32 = 1.5;
    /// Topic threshold met: weighted_agreement >= 2.5
    pub const TOPIC_THRESHOLD: f32 = 1.2;
    /// Related but below topic threshold: weighted_agreement >= 1.0
    pub const RELATED: f32 = 1.0;
    /// Weak signal: weighted_agreement < 1.0
    pub const WEAK: f32 = 0.8;

    /// Compute diversity bonus based on weighted_agreement.
    /// weighted_agreement uses embedder category weights:
    ///   semantic (E1,E5,E6,E7,E10,E12,E13) = 1.0
    ///   temporal (E2,E3,E4) = 0.0 (excluded)
    ///   relational (E8,E11) = 0.5
    ///   structural (E9) = 0.5
    /// Max possible = 8.5
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

/// Computes priority scores and ranks injection candidates.
pub struct PriorityRanker;

impl PriorityRanker {
    /// Compute priority for a single candidate.
    /// Sets recency_factor, diversity_bonus, and final priority.
    /// Uses weighted_agreement (not raw space count) for diversity calculation.
    pub fn compute_priority(candidate: &mut InjectionCandidate) {
        let recency = RecencyFactor::compute(candidate.created_at);
        let diversity = DiversityBonus::compute(candidate.weighted_agreement);

        candidate.set_priority_factors(recency, diversity);
    }

    /// Compute priority for a single candidate with explicit reference time.
    pub fn compute_priority_at(candidate: &mut InjectionCandidate, now: DateTime<Utc>) {
        let recency = RecencyFactor::compute_relative(candidate.created_at, now);
        let diversity = DiversityBonus::compute(candidate.weighted_agreement);

        candidate.set_priority_factors(recency, diversity);
    }

    /// Rank all candidates by category then priority.
    /// Modifies candidates in place.
    pub fn rank_candidates(candidates: &mut [InjectionCandidate]) {
        // First compute priority for each
        for candidate in candidates.iter_mut() {
            Self::compute_priority(candidate);
        }

        // Sort: category ascending, priority descending within category
        candidates.sort();
    }

    /// Rank with explicit reference time (for testing).
    pub fn rank_candidates_at(candidates: &mut [InjectionCandidate], now: DateTime<Utc>) {
        for candidate in candidates.iter_mut() {
            Self::compute_priority_at(candidate, now);
        }
        candidates.sort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::candidate::InjectionCategory;
    use uuid::Uuid;

    fn make_candidate_at(
        relevance: f32,
        created_at: DateTime<Utc>,
        weighted_agreement: f32,
        category: InjectionCategory,
    ) -> InjectionCandidate {
        use crate::embedding::Embedder;

        // Create dummy matching spaces (not used for diversity calculation anymore)
        let matching_spaces: Vec<Embedder> = vec![Embedder::E1Semantic];

        InjectionCandidate::new(
            Uuid::new_v4(),
            "test content".to_string(),
            relevance,
            weighted_agreement,
            matching_spaces,
            category,
            created_at,
        )
    }

    #[test]
    fn test_recency_factor_hour() {
        let now = Utc::now();
        let created_30min_ago = now - Duration::minutes(30);

        let factor = RecencyFactor::compute_relative(created_30min_ago, now);
        assert!((factor - 1.3).abs() < 0.001);
    }

    #[test]
    fn test_recency_factor_day() {
        let now = Utc::now();
        let created_6hours_ago = now - Duration::hours(6);

        let factor = RecencyFactor::compute_relative(created_6hours_ago, now);
        assert!((factor - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_recency_factor_week() {
        let now = Utc::now();
        let created_3days_ago = now - Duration::days(3);

        let factor = RecencyFactor::compute_relative(created_3days_ago, now);
        assert!((factor - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_recency_factor_month() {
        let now = Utc::now();
        let created_15days_ago = now - Duration::days(15);

        let factor = RecencyFactor::compute_relative(created_15days_ago, now);
        assert!((factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_recency_factor_older() {
        let now = Utc::now();
        let created_60days_ago = now - Duration::days(60);

        let factor = RecencyFactor::compute_relative(created_60days_ago, now);
        assert!((factor - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_diversity_bonus_weak() {
        // weighted_agreement < 1.0 -> WEAK (0.8)
        assert!((DiversityBonus::compute(0.5) - 0.8).abs() < 0.001);
        assert!((DiversityBonus::compute(0.9) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_diversity_bonus_related() {
        // weighted_agreement >= 1.0 and < 2.5 -> RELATED (1.0)
        assert!((DiversityBonus::compute(1.0) - 1.0).abs() < 0.001);
        assert!((DiversityBonus::compute(2.4) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_diversity_bonus_topic_threshold() {
        // weighted_agreement >= 2.5 and < 5.0 -> TOPIC_THRESHOLD (1.2)
        assert!((DiversityBonus::compute(2.5) - 1.2).abs() < 0.001);
        assert!((DiversityBonus::compute(4.9) - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_diversity_bonus_strong_topic() {
        // weighted_agreement >= 5.0 -> STRONG_TOPIC (1.5)
        assert!((DiversityBonus::compute(5.0) - 1.5).abs() < 0.001);
        assert!((DiversityBonus::compute(8.5) - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_priority_computation() {
        let now = Utc::now();
        let created_2hours_ago = now - Duration::hours(2);

        let mut candidate = make_candidate_at(
            0.82,  // relevance
            created_2hours_ago,
            3.5,   // weighted_agreement 3.5 → diversity 1.2 (topic threshold)
            InjectionCategory::HighRelevanceCluster,
        );

        PriorityRanker::compute_priority_at(&mut candidate, now);

        // recency = 1.2 (within 1 day)
        // diversity = 1.2 (weighted_agreement >= 2.5)
        // priority = 0.82 × 1.2 × 1.2 = 1.1808
        assert!((candidate.recency_factor - 1.2).abs() < 0.001);
        assert!((candidate.diversity_bonus - 1.2).abs() < 0.001);
        assert!((candidate.priority - 1.1808).abs() < 0.01);
    }

    #[test]
    fn test_ranking_category_first() {
        let now = Utc::now();

        let mut candidates = vec![
            make_candidate_at(0.9, now, 5.5, InjectionCategory::SingleSpaceMatch),
            make_candidate_at(0.5, now, 0.5, InjectionCategory::DivergenceAlert),
            make_candidate_at(0.7, now, 3.0, InjectionCategory::HighRelevanceCluster),
        ];

        PriorityRanker::rank_candidates_at(&mut candidates, now);

        // Should be sorted by category priority
        assert_eq!(candidates[0].category, InjectionCategory::DivergenceAlert);
        assert_eq!(candidates[1].category, InjectionCategory::HighRelevanceCluster);
        assert_eq!(candidates[2].category, InjectionCategory::SingleSpaceMatch);
    }

    #[test]
    fn test_ranking_priority_within_category() {
        let now = Utc::now();

        let mut candidates = vec![
            make_candidate_at(0.7, now, 3.0, InjectionCategory::HighRelevanceCluster),
            make_candidate_at(0.9, now, 3.0, InjectionCategory::HighRelevanceCluster),
            make_candidate_at(0.8, now, 3.0, InjectionCategory::HighRelevanceCluster),
        ];

        PriorityRanker::rank_candidates_at(&mut candidates, now);

        // Same category, should be sorted by priority descending
        assert!(candidates[0].relevance_score > candidates[1].relevance_score);
        assert!(candidates[1].relevance_score > candidates[2].relevance_score);
    }

    #[test]
    fn test_priority_spec_example() {
        // Example from impplan spec (updated for weighted_agreement):
        // Memory A: relevance=0.82, 2h ago (recency=1.2), weighted_agreement=3.5 (diversity=1.2)
        //           priority = 0.82 × 1.2 × 1.2 = 1.18
        // Memory B: relevance=0.90, 3d ago (recency=1.1), weighted_agreement=1.5 (diversity=1.0)
        //           priority = 0.90 × 1.1 × 1.0 = 0.99
        // Memory A should rank higher

        let now = Utc::now();

        let mut memory_a = make_candidate_at(
            0.82,
            now - Duration::hours(2),
            3.5,  // weighted_agreement >= 2.5 -> diversity 1.2
            InjectionCategory::HighRelevanceCluster,
        );

        let mut memory_b = make_candidate_at(
            0.90,
            now - Duration::days(3),
            1.5,  // weighted_agreement >= 1.0 but < 2.5 -> diversity 1.0
            InjectionCategory::HighRelevanceCluster,
        );

        PriorityRanker::compute_priority_at(&mut memory_a, now);
        PriorityRanker::compute_priority_at(&mut memory_b, now);

        // Memory A should have higher priority despite lower relevance
        assert!(memory_a.priority > memory_b.priority);
        assert!((memory_a.priority - 1.18).abs() < 0.01);
        assert!((memory_b.priority - 0.99).abs() < 0.01);
    }

    #[test]
    fn test_high_relevance_cluster_requires_weighted_agreement() {
        // HighRelevanceCluster should have weighted_agreement >= 2.5
        let now = Utc::now();

        let mut candidate = make_candidate_at(
            0.9,
            now,
            2.5,  // Exactly at topic threshold
            InjectionCategory::HighRelevanceCluster,
        );

        PriorityRanker::compute_priority_at(&mut candidate, now);

        // Should get topic threshold diversity bonus
        assert!((candidate.diversity_bonus - 1.2).abs() < 0.001);
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/injection/priority.rs">
    RecencyFactor, DiversityBonus, PriorityRanker
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/injection/mod.rs">
    Add pub mod priority; pub use priority::*;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-core compiles without errors</criterion>
  <criterion type="test">cargo test injection::priority::tests -- all 13 tests pass</criterion>
  <criterion type="spec">Spec example (Memory A vs B) ranks correctly</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-core</command>
  <command>cargo test injection::priority::tests --package context-graph-core</command>
</test_commands>
</task_spec>
```
