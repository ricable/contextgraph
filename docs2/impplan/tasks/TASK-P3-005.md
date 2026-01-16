# TASK-P3-005: MultiSpaceSimilarity

```xml
<task_spec id="TASK-P3-005" version="1.0">
<metadata>
  <title>MultiSpaceSimilarity Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>24</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-01</requirement_ref>
    <requirement_ref>REQ-P3-02</requirement_ref>
    <requirement_ref>REQ-P3-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P3-001</task_ref>
    <task_ref>TASK-P3-003</task_ref>
    <task_ref>TASK-P3-004</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the MultiSpaceSimilarity component that computes similarity scores
across all 13 embedding spaces, determines relevance using the ANY() logic,
and calculates weighted relevance scores.

This is the core comparison engine used by both retrieval and divergence detection.

CATEGORY-AWARE WEIGHTING: The weighted similarity calculation uses EmbedderCategory
topic_weights to determine each space's contribution:
- Semantic spaces (E1, E5-E7, E10, E12-E13): weight 1.0
- Temporal spaces (E2-E4): weight 0.0 (EXCLUDED from relevance scoring)
- Relational spaces (E8, E11): weight 0.5
- Structural space (E9): weight 0.5

Temporal spaces are explicitly excluded from the weighted relevance score calculation
as they capture time-based features (recency, periodicity) rather than topic relevance.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md#component_contracts</file>
  <file purpose="distance">crates/context-graph-core/src/retrieval/distance.rs</file>
  <file purpose="config">crates/context-graph-core/src/retrieval/config.rs</file>
  <file purpose="category">crates/context-graph-core/src/embedding/category.rs</file>
  <file purpose="embedder_config">crates/context-graph-core/src/embedding/config.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P3-001 complete (PerSpaceScores exists)</check>
  <check>TASK-P3-003 complete (thresholds exist)</check>
  <check>TASK-P3-004 complete (distance calculator exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement compute_similarity for full TeleologicalArray comparison
    - Implement is_relevant with ANY() logic
    - Implement matching_spaces to find matching embedders
    - Implement compute_relevance_score with category-weighted aggregation
    - Implement weighted_similarity using EmbedderCategory::topic_weight()
    - Create MultiSpaceSimilarity service struct
    - Add SPACE_WEIGHTS constant derived from EmbedderCategory
    - Exclude temporal spaces (E2-E4) from weighted relevance calculations
  </in_scope>
  <out_of_scope>
    - Memory storage/retrieval (TASK-P3-007)
    - Divergence detection (TASK-P3-006)
    - Index building for large databases
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/multi_space.rs">
      /// Space weights derived from EmbedderCategory::topic_weight()
      /// Temporal spaces (E2-E4) have weight 0.0 and are excluded from relevance scoring
      pub static SPACE_WEIGHTS: [f32; 13];

      pub struct MultiSpaceSimilarity {
          thresholds: SimilarityThresholds,
          weights: SpaceWeights,
      }

      impl MultiSpaceSimilarity {
          pub fn new(thresholds: SimilarityThresholds, weights: SpaceWeights) -> Self;
          pub fn with_defaults() -> Self;
          pub fn compute_similarity(&amp;self, query: &amp;TeleologicalArray, memory: &amp;TeleologicalArray) -> PerSpaceScores;
          pub fn is_relevant(&amp;self, scores: &amp;PerSpaceScores) -> bool;
          pub fn matching_spaces(&amp;self, scores: &amp;PerSpaceScores) -> Vec&lt;Embedder&gt;;
          pub fn compute_relevance_score(&amp;self, scores: &amp;PerSpaceScores) -> f32;
          pub fn compute_weighted_similarity(&amp;self, scores: &amp;PerSpaceScores) -> f32;
          pub fn compute_full_result(&amp;self, memory_id: Uuid, query: &amp;TeleologicalArray, memory: &amp;TeleologicalArray) -> SimilarityResult;
      }
    </signature>
  </signatures>

  <constraints>
    - is_relevant returns true if ANY space above high threshold
    - relevance_score = Σ(category_weight_i × max(0, score_i - threshold_i))
    - relevance_score normalized to 0.0..1.0 range
    - All 13 spaces computed in compute_similarity
    - Temporal spaces (E2-E4) excluded from weighted calculations (topic_weight = 0.0)
    - SPACE_WEIGHTS derived from get_topic_weight() for each embedder
    - compute_weighted_similarity uses category weights, not uniform weights
  </constraints>

  <verification>
    - Memory with one matching space returns is_relevant = true
    - Memory with no matching spaces returns is_relevant = false
    - Relevance score higher for more matching spaces
    - Category weights correctly applied to scores
    - Temporal spaces (E2-E4) contribute 0.0 to weighted_similarity
    - Semantic spaces (E1, E5-E7, E10, E12-E13) contribute full weight (1.0)
    - Relational/Structural spaces contribute half weight (0.5)
    - SPACE_WEIGHTS[1..4] all equal 0.0 (temporal)
    - SPACE_WEIGHTS[0] equals 1.0 (E1 Semantic)
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/retrieval/multi_space.rs

use uuid::Uuid;
use crate::embedding::{Embedder, TeleologicalArray};
use crate::embedding::config::get_topic_weight;
use super::similarity::{PerSpaceScores, SimilarityResult};
use super::config::{
    PerSpaceThresholds, SpaceWeights, SimilarityThresholds,
    high_thresholds, low_thresholds, default_weights,
};
use super::distance::compute_similarity_for_space;

/// Space weights derived from EmbedderCategory::topic_weight()
/// Index matches Embedder::index() (0=E1, 1=E2, etc.)
/// Temporal spaces (E2-E4, indices 1-3) have weight 0.0
pub static SPACE_WEIGHTS: [f32; 13] = [
    1.0,  // E1 Semantic
    0.0,  // E2 TempRecent (excluded)
    0.0,  // E3 TempPeriodic (excluded)
    0.0,  // E4 TempPosition (excluded)
    1.0,  // E5 Causal
    1.0,  // E6 Sparse
    1.0,  // E7 Code
    0.5,  // E8 Emotional (Relational)
    0.5,  // E9 HDC (Structural)
    1.0,  // E10 Multimodal
    0.5,  // E11 Entity (Relational)
    1.0,  // E12 LateInteract
    1.0,  // E13 SPLADE
];

/// Multi-space similarity computation service
pub struct MultiSpaceSimilarity {
    thresholds: SimilarityThresholds,
    weights: SpaceWeights,
}

impl MultiSpaceSimilarity {
    /// Create with custom thresholds and weights
    pub fn new(thresholds: SimilarityThresholds, weights: SpaceWeights) -> Self {
        Self { thresholds, weights }
    }

    /// Create with default configuration from spec
    pub fn with_defaults() -> Self {
        Self {
            thresholds: SimilarityThresholds::default(),
            weights: default_weights().normalized(),
        }
    }

    /// Compute similarity scores across all 13 embedding spaces
    pub fn compute_similarity(
        &amp;self,
        query: &amp;TeleologicalArray,
        memory: &amp;TeleologicalArray,
    ) -> PerSpaceScores {
        let mut scores = PerSpaceScores::new();

        for embedder in Embedder::all() {
            let sim = compute_similarity_for_space(embedder, query, memory);
            scores.set_score(embedder, sim);
        }

        scores
    }

    /// Check if memory is relevant (ANY space above high threshold)
    pub fn is_relevant(&amp;self, scores: &amp;PerSpaceScores) -> bool {
        for embedder in Embedder::all() {
            let score = scores.get_score(embedder);
            let threshold = self.thresholds.high.get_threshold(embedder);
            if score > threshold {
                return true;
            }
        }
        false
    }

    /// Get list of embedders where score exceeds threshold
    pub fn matching_spaces(&amp;self, scores: &amp;PerSpaceScores) -> Vec&lt;Embedder&gt; {
        let mut matches = Vec::new();

        for embedder in Embedder::all() {
            let score = scores.get_score(embedder);
            let threshold = self.thresholds.high.get_threshold(embedder);
            if score > threshold {
                matches.push(embedder);
            }
        }

        matches
    }

    /// Compute weighted relevance score using category weights
    /// Formula: Σ(category_weight_i × max(0, score_i - threshold_i)) / max_possible
    /// NOTE: Temporal spaces (E2-E4) have category_weight 0.0 and are excluded
    pub fn compute_relevance_score(&amp;self, scores: &amp;PerSpaceScores) -> f32 {
        let mut weighted_sum = 0.0;
        let mut max_possible = 0.0;

        for embedder in Embedder::all() {
            let score = scores.get_score(embedder);
            let threshold = self.thresholds.high.get_threshold(embedder);
            // Use category-derived weight instead of uniform weight
            let category_weight = get_topic_weight(embedder);

            // Skip temporal spaces (weight = 0.0)
            if category_weight == 0.0 {
                continue;
            }

            // Score above threshold contributes positively
            let contribution = (score - threshold).max(0.0);
            weighted_sum += category_weight * contribution;

            // Maximum possible is if score was 1.0
            max_possible += category_weight * (1.0 - threshold).max(0.0);
        }

        // Normalize to 0.0..1.0
        if max_possible > 0.0 {
            (weighted_sum / max_possible).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Compute weighted similarity using category weights (excludes temporal)
    /// This is a simpler version that just sums weighted scores without threshold subtraction
    pub fn compute_weighted_similarity(&amp;self, scores: &amp;PerSpaceScores) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for embedder in Embedder::all() {
            let category_weight = get_topic_weight(embedder);

            // Skip temporal spaces (weight = 0.0)
            if category_weight == 0.0 {
                continue;
            }

            let score = scores.get_score(embedder);
            weighted_sum += category_weight * score;
            total_weight += category_weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Compute complete SimilarityResult for a memory
    pub fn compute_full_result(
        &amp;self,
        memory_id: Uuid,
        query: &amp;TeleologicalArray,
        memory: &amp;TeleologicalArray,
    ) -> SimilarityResult {
        let scores = self.compute_similarity(query, memory);
        let matching = self.matching_spaces(&amp;scores);
        let relevance = self.compute_relevance_score(&amp;scores);

        SimilarityResult::with_relevance(memory_id, scores, relevance, matching)
    }

    /// Get reference to thresholds
    pub fn thresholds(&amp;self) -> &amp;SimilarityThresholds {
        &amp;self.thresholds
    }

    /// Get reference to weights
    pub fn weights(&amp;self) -> &amp;SpaceWeights {
        &amp;self.weights
    }

    /// Check if score is below low threshold (for divergence detection)
    pub fn is_below_low_threshold(&amp;self, embedder: Embedder, score: f32) -> bool {
        score < self.thresholds.low.get_threshold(embedder)
    }
}

/// Batch comparison for multiple memories
pub fn compute_similarities_batch(
    similarity: &amp;MultiSpaceSimilarity,
    query: &amp;TeleologicalArray,
    memories: &amp;[(Uuid, TeleologicalArray)],
) -> Vec&lt;SimilarityResult&gt; {
    memories
        .iter()
        .map(|(id, memory)| similarity.compute_full_result(*id, query, memory))
        .collect()
}

/// Filter to relevant results only
pub fn filter_relevant(
    similarity: &amp;MultiSpaceSimilarity,
    results: Vec&lt;SimilarityResult&gt;,
) -> Vec&lt;SimilarityResult&gt; {
    results
        .into_iter()
        .filter(|r| similarity.is_relevant(&amp;r.per_space_scores))
        .collect()
}

/// Sort results by relevance score (highest first)
pub fn sort_by_relevance(mut results: Vec&lt;SimilarityResult&gt;) -> Vec&lt;SimilarityResult&gt; {
    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&amp;a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_array_with_scores(semantic: f32, code: f32) -> TeleologicalArray {
        use crate::embedding::vector::{DenseVector, SparseVector, BinaryVector};

        // Create array where we can control similarity
        let mut array = TeleologicalArray::new();

        // Set semantic embedding to predictable values
        let mut semantic_data = vec![0.0; 1024];
        semantic_data[0] = semantic;
        array.e1_semantic = DenseVector::new(semantic_data);

        // Set code embedding to predictable values
        let mut code_data = vec![0.0; 1536];
        code_data[0] = code;
        array.e7_code = DenseVector::new(code_data);

        array
    }

    #[test]
    fn test_is_relevant_one_match() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        let mut scores = PerSpaceScores::new();
        scores.set_score(Embedder::E1Semantic, 0.80); // Above 0.75 threshold
        scores.set_score(Embedder::E7Code, 0.50);     // Below 0.80 threshold

        assert!(similarity.is_relevant(&amp;scores));
    }

    #[test]
    fn test_is_relevant_no_match() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        let mut scores = PerSpaceScores::new();
        scores.set_score(Embedder::E1Semantic, 0.70); // Below 0.75 threshold
        scores.set_score(Embedder::E7Code, 0.50);     // Below 0.80 threshold

        assert!(!similarity.is_relevant(&amp;scores));
    }

    #[test]
    fn test_matching_spaces() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        let mut scores = PerSpaceScores::new();
        scores.set_score(Embedder::E1Semantic, 0.80); // Match
        scores.set_score(Embedder::E7Code, 0.85);     // Match
        scores.set_score(Embedder::E6Sparse, 0.30);   // No match

        let matches = similarity.matching_spaces(&amp;scores);
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&amp;Embedder::E1Semantic));
        assert!(matches.contains(&amp;Embedder::E7Code));
    }

    #[test]
    fn test_relevance_score_higher_with_more_matches() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        let mut scores_one = PerSpaceScores::new();
        scores_one.set_score(Embedder::E1Semantic, 0.80);

        let mut scores_two = PerSpaceScores::new();
        scores_two.set_score(Embedder::E1Semantic, 0.80);
        scores_two.set_score(Embedder::E7Code, 0.85);

        let rel_one = similarity.compute_relevance_score(&amp;scores_one);
        let rel_two = similarity.compute_relevance_score(&amp;scores_two);

        assert!(rel_two > rel_one);
    }

    #[test]
    fn test_relevance_score_normalized() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        // Maximum possible scores
        let mut scores = PerSpaceScores::new();
        for embedder in Embedder::all() {
            scores.set_score(embedder, 1.0);
        }

        let rel = similarity.compute_relevance_score(&amp;scores);
        assert!(rel >= 0.0 && rel <= 1.0);
    }

    #[test]
    fn test_below_low_threshold() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        // E1 low threshold is 0.30
        assert!(similarity.is_below_low_threshold(Embedder::E1Semantic, 0.25));
        assert!(!similarity.is_below_low_threshold(Embedder::E1Semantic, 0.35));
    }

    #[test]
    fn test_compute_full_result() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let memory_id = Uuid::new_v4();
        let query = TeleologicalArray::new();
        let memory = TeleologicalArray::new();

        let result = similarity.compute_full_result(memory_id, &amp;query, &amp;memory);

        assert_eq!(result.memory_id, memory_id);
        assert_eq!(result.space_count as usize, result.matching_spaces.len());
    }

    #[test]
    fn test_space_weights_values() {
        // Verify SPACE_WEIGHTS matches expected category weights
        assert_eq!(SPACE_WEIGHTS[0], 1.0);  // E1 Semantic
        assert_eq!(SPACE_WEIGHTS[1], 0.0);  // E2 Temporal (excluded)
        assert_eq!(SPACE_WEIGHTS[2], 0.0);  // E3 Temporal (excluded)
        assert_eq!(SPACE_WEIGHTS[3], 0.0);  // E4 Temporal (excluded)
        assert_eq!(SPACE_WEIGHTS[4], 1.0);  // E5 Semantic
        assert_eq!(SPACE_WEIGHTS[5], 1.0);  // E6 Semantic
        assert_eq!(SPACE_WEIGHTS[6], 1.0);  // E7 Semantic
        assert_eq!(SPACE_WEIGHTS[7], 0.5);  // E8 Relational
        assert_eq!(SPACE_WEIGHTS[8], 0.5);  // E9 Structural
        assert_eq!(SPACE_WEIGHTS[9], 1.0);  // E10 Semantic
        assert_eq!(SPACE_WEIGHTS[10], 0.5); // E11 Relational
        assert_eq!(SPACE_WEIGHTS[11], 1.0); // E12 Semantic
        assert_eq!(SPACE_WEIGHTS[12], 1.0); // E13 Semantic
    }

    #[test]
    fn test_temporal_excluded_from_weighted() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        // Scores with high temporal but low semantic
        let mut scores_temporal = PerSpaceScores::new();
        scores_temporal.set_score(Embedder::E2TempRecent, 0.95);
        scores_temporal.set_score(Embedder::E3TempPeriodic, 0.95);
        scores_temporal.set_score(Embedder::E4TempPosition, 0.95);

        // Temporal scores should not contribute to weighted similarity
        let weighted = similarity.compute_weighted_similarity(&amp;scores_temporal);
        // All other scores are 0.0, temporal excluded, so result should be low/zero
        assert!(weighted < 0.1);
    }

    #[test]
    fn test_semantic_contributes_full_weight() {
        let similarity = MultiSpaceSimilarity::with_defaults();

        let mut scores = PerSpaceScores::new();
        scores.set_score(Embedder::E1Semantic, 0.90);
        scores.set_score(Embedder::E7Code, 0.90);

        let weighted = similarity.compute_weighted_similarity(&amp;scores);
        // Semantic spaces contribute full weight
        assert!(weighted > 0.0);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/multi_space.rs">MultiSpaceSimilarity implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Add pub mod multi_space and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>compute_similarity returns scores for all 13 spaces</criterion>
  <criterion>is_relevant returns true if ANY space above threshold</criterion>
  <criterion>matching_spaces returns correct embedder list</criterion>
  <criterion>relevance_score higher with more matching spaces</criterion>
  <criterion>relevance_score normalized to 0.0..1.0</criterion>
  <criterion>Category weights correctly applied to scores</criterion>
  <criterion>SPACE_WEIGHTS constant has correct values for all 13 embedders</criterion>
  <criterion>Temporal spaces (E2-E4) excluded from weighted calculations</criterion>
  <criterion>compute_weighted_similarity returns correct category-weighted result</criterion>
  <criterion>High temporal scores do not boost relevance (weight=0.0)</criterion>
</validation_criteria>

<test_commands>
  <command description="Run multi_space tests">cargo test --package context-graph-core multi_space</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create multi_space.rs in retrieval directory
- [ ] Add SPACE_WEIGHTS constant with category-derived weights
- [ ] Implement MultiSpaceSimilarity struct
- [ ] Implement compute_similarity method
- [ ] Implement is_relevant with ANY() logic
- [ ] Implement matching_spaces method
- [ ] Implement compute_relevance_score with category weights
- [ ] Implement compute_weighted_similarity method
- [ ] Ensure temporal spaces (E2-E4) excluded from weighted calculations
- [ ] Implement compute_full_result method
- [ ] Add batch comparison helpers
- [ ] Write unit tests for SPACE_WEIGHTS values
- [ ] Write unit tests for temporal exclusion
- [ ] Write unit tests for category weight application
- [ ] Run tests to verify
- [ ] Proceed to TASK-P3-006
