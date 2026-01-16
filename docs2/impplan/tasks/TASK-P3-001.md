# TASK-P3-001: PerSpaceScores and SimilarityResult

```xml
<task_spec id="TASK-P3-001" version="1.0">
<metadata>
  <title>PerSpaceScores and SimilarityResult Types</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>20</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-01</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the core data types for similarity scoring across all 13 embedding spaces.
PerSpaceScores holds individual similarity scores for each embedder, while
SimilarityResult wraps this with metadata about the matching memory.

These types are the foundation for the entire retrieval and divergence detection system.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md#data_models</file>
  <file purpose="embedder_enum">crates/context-graph-core/src/embedding/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>Embedder enum exists (from Phase 2)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create PerSpaceScores struct with 13 score fields
    - Create SimilarityResult struct
    - Implement Default trait for PerSpaceScores
    - Add iterator support for PerSpaceScores
    - Add helper methods (get_score, set_score by Embedder)
    - Implement Clone, Debug, Serialize, Deserialize
  </in_scope>
  <out_of_scope>
    - Similarity computation (TASK-P3-005)
    - Threshold comparison (TASK-P3-005)
    - Distance calculation (TASK-P3-004)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/similarity.rs">
      #[derive(Debug, Clone, Default, Serialize, Deserialize)]
      pub struct PerSpaceScores {
          pub e1_semantic: f32,
          pub e2_temp_recent: f32,
          pub e3_temp_periodic: f32,
          pub e4_temp_position: f32,
          pub e5_causal: f32,
          pub e6_sparse: f32,
          pub e7_code: f32,
          pub e8_emotional: f32,
          pub e9_hdc: f32,
          pub e10_multimodal: f32,
          pub e11_entity: f32,
          pub e12_late_interact: f32,
          pub e13_splade: f32,
      }

      impl PerSpaceScores {
          pub fn new() -> Self;
          pub fn get_score(&amp;self, embedder: Embedder) -> f32;
          pub fn set_score(&amp;mut self, embedder: Embedder, score: f32);
          pub fn iter(&amp;self) -> impl Iterator&lt;Item = (Embedder, f32)&gt;;
          pub fn max_score(&amp;self) -> f32;
          pub fn mean_score(&amp;self) -> f32;
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct SimilarityResult {
          pub memory_id: Uuid,
          pub per_space_scores: PerSpaceScores,
          pub relevance_score: f32,
          pub matching_spaces: Vec&lt;Embedder&gt;,
          pub space_count: u8,
      }

      impl SimilarityResult {
          pub fn new(memory_id: Uuid, scores: PerSpaceScores) -> Self;
      }
    </signature>
  </signatures>

  <constraints>
    - All scores must be in 0.0..=1.0 range
    - Default PerSpaceScores initializes all to 0.0
    - SimilarityResult.space_count = matching_spaces.len()
    - Both structs must be serializable
  </constraints>

  <verification>
    - PerSpaceScores can get/set by Embedder
    - Iterator visits all 13 scores
    - Default values are all 0.0
    - Serialization roundtrip works
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/retrieval/similarity.rs

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::embedding::Embedder;

/// Similarity scores for all 13 embedding spaces
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PerSpaceScores {
    pub e1_semantic: f32,
    pub e2_temp_recent: f32,
    pub e3_temp_periodic: f32,
    pub e4_temp_position: f32,
    pub e5_causal: f32,
    pub e6_sparse: f32,
    pub e7_code: f32,
    pub e8_emotional: f32,
    pub e9_hdc: f32,
    pub e10_multimodal: f32,
    pub e11_entity: f32,
    pub e12_late_interact: f32,
    pub e13_splade: f32,
}

impl PerSpaceScores {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get score for a specific embedder
    pub fn get_score(&amp;self, embedder: Embedder) -> f32 {
        match embedder {
            Embedder::E1Semantic => self.e1_semantic,
            Embedder::E2TempRecent => self.e2_temp_recent,
            Embedder::E3TempPeriodic => self.e3_temp_periodic,
            Embedder::E4TempPosition => self.e4_temp_position,
            Embedder::E5Causal => self.e5_causal,
            Embedder::E6Sparse => self.e6_sparse,
            Embedder::E7Code => self.e7_code,
            Embedder::E8Emotional => self.e8_emotional,
            Embedder::E9HDC => self.e9_hdc,
            Embedder::E10Multimodal => self.e10_multimodal,
            Embedder::E11Entity => self.e11_entity,
            Embedder::E12LateInteract => self.e12_late_interact,
            Embedder::E13SPLADE => self.e13_splade,
        }
    }

    /// Set score for a specific embedder
    pub fn set_score(&amp;mut self, embedder: Embedder, score: f32) {
        let score = score.clamp(0.0, 1.0);
        match embedder {
            Embedder::E1Semantic => self.e1_semantic = score,
            Embedder::E2TempRecent => self.e2_temp_recent = score,
            Embedder::E3TempPeriodic => self.e3_temp_periodic = score,
            Embedder::E4TempPosition => self.e4_temp_position = score,
            Embedder::E5Causal => self.e5_causal = score,
            Embedder::E6Sparse => self.e6_sparse = score,
            Embedder::E7Code => self.e7_code = score,
            Embedder::E8Emotional => self.e8_emotional = score,
            Embedder::E9HDC => self.e9_hdc = score,
            Embedder::E10Multimodal => self.e10_multimodal = score,
            Embedder::E11Entity => self.e11_entity = score,
            Embedder::E12LateInteract => self.e12_late_interact = score,
            Embedder::E13SPLADE => self.e13_splade = score,
        }
    }

    /// Iterate over all scores with their embedder
    pub fn iter(&amp;self) -> impl Iterator&lt;Item = (Embedder, f32)&gt; + '_ {
        Embedder::all().into_iter().map(|e| (e, self.get_score(e)))
    }

    /// Get the maximum score across all spaces
    pub fn max_score(&amp;self) -> f32 {
        self.iter().map(|(_, s)| s).fold(0.0, f32::max)
    }

    /// Get the mean score across all spaces
    pub fn mean_score(&amp;self) -> f32 {
        let sum: f32 = self.iter().map(|(_, s)| s).sum();
        sum / 13.0
    }

    /// Convert to array for compact operations
    pub fn to_array(&amp;self) -> [f32; 13] {
        [
            self.e1_semantic,
            self.e2_temp_recent,
            self.e3_temp_periodic,
            self.e4_temp_position,
            self.e5_causal,
            self.e6_sparse,
            self.e7_code,
            self.e8_emotional,
            self.e9_hdc,
            self.e10_multimodal,
            self.e11_entity,
            self.e12_late_interact,
            self.e13_splade,
        ]
    }

    /// Create from array
    pub fn from_array(arr: [f32; 13]) -> Self {
        Self {
            e1_semantic: arr[0],
            e2_temp_recent: arr[1],
            e3_temp_periodic: arr[2],
            e4_temp_position: arr[3],
            e5_causal: arr[4],
            e6_sparse: arr[5],
            e7_code: arr[6],
            e8_emotional: arr[7],
            e9_hdc: arr[8],
            e10_multimodal: arr[9],
            e11_entity: arr[10],
            e12_late_interact: arr[11],
            e13_splade: arr[12],
        }
    }
}

/// Result of similarity search for a single memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// ID of the matching memory
    pub memory_id: Uuid,
    /// Similarity scores in each embedding space
    pub per_space_scores: PerSpaceScores,
    /// Weighted aggregate relevance score (0.0..1.0)
    pub relevance_score: f32,
    /// Embedders where score exceeded threshold
    pub matching_spaces: Vec&lt;Embedder&gt;,
    /// Number of matching spaces (= matching_spaces.len())
    pub space_count: u8,
}

impl SimilarityResult {
    /// Create a new SimilarityResult with just scores (other fields set later)
    pub fn new(memory_id: Uuid, scores: PerSpaceScores) -> Self {
        Self {
            memory_id,
            per_space_scores: scores,
            relevance_score: 0.0,
            matching_spaces: Vec::new(),
            space_count: 0,
        }
    }

    /// Create with full computed fields
    pub fn with_relevance(
        memory_id: Uuid,
        scores: PerSpaceScores,
        relevance_score: f32,
        matching_spaces: Vec&lt;Embedder&gt;,
    ) -> Self {
        let space_count = matching_spaces.len() as u8;
        Self {
            memory_id,
            per_space_scores: scores,
            relevance_score,
            matching_spaces,
            space_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_space_scores_default() {
        let scores = PerSpaceScores::new();
        assert_eq!(scores.e1_semantic, 0.0);
        assert_eq!(scores.e7_code, 0.0);
        assert_eq!(scores.max_score(), 0.0);
    }

    #[test]
    fn test_get_set_score() {
        let mut scores = PerSpaceScores::new();
        scores.set_score(Embedder::E1Semantic, 0.85);
        scores.set_score(Embedder::E7Code, 0.92);

        assert_eq!(scores.get_score(Embedder::E1Semantic), 0.85);
        assert_eq!(scores.get_score(Embedder::E7Code), 0.92);
        assert_eq!(scores.max_score(), 0.92);
    }

    #[test]
    fn test_score_clamping() {
        let mut scores = PerSpaceScores::new();
        scores.set_score(Embedder::E1Semantic, 1.5);  // Should clamp to 1.0
        scores.set_score(Embedder::E2TempRecent, -0.5);  // Should clamp to 0.0

        assert_eq!(scores.get_score(Embedder::E1Semantic), 1.0);
        assert_eq!(scores.get_score(Embedder::E2TempRecent), 0.0);
    }

    #[test]
    fn test_iterator() {
        let scores = PerSpaceScores::new();
        let count = scores.iter().count();
        assert_eq!(count, 13);
    }

    #[test]
    fn test_array_conversion() {
        let mut scores = PerSpaceScores::new();
        scores.set_score(Embedder::E1Semantic, 0.5);
        scores.set_score(Embedder::E7Code, 0.7);

        let arr = scores.to_array();
        assert_eq!(arr[0], 0.5);
        assert_eq!(arr[6], 0.7);

        let recovered = PerSpaceScores::from_array(arr);
        assert_eq!(recovered.e1_semantic, 0.5);
        assert_eq!(recovered.e7_code, 0.7);
    }

    #[test]
    fn test_similarity_result() {
        let id = Uuid::new_v4();
        let scores = PerSpaceScores::new();
        let result = SimilarityResult::with_relevance(
            id,
            scores,
            0.75,
            vec![Embedder::E1Semantic, Embedder::E7Code],
        );

        assert_eq!(result.memory_id, id);
        assert_eq!(result.relevance_score, 0.75);
        assert_eq!(result.space_count, 2);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/similarity.rs">PerSpaceScores and SimilarityResult</file>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Module definition</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/lib.rs">Add pub mod retrieval</file>
</files_to_modify>

<validation_criteria>
  <criterion>PerSpaceScores has all 13 fields</criterion>
  <criterion>get_score/set_score work for all Embedder variants</criterion>
  <criterion>Score clamping enforces 0.0..1.0 range</criterion>
  <criterion>Iterator visits all 13 spaces</criterion>
  <criterion>SimilarityResult holds memory_id and scores</criterion>
  <criterion>Serialization roundtrip works</criterion>
</validation_criteria>

<test_commands>
  <command description="Run similarity type tests">cargo test --package context-graph-core similarity</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create retrieval directory in context-graph-core/src
- [ ] Create mod.rs for retrieval module
- [ ] Create similarity.rs with PerSpaceScores struct
- [ ] Implement get_score/set_score by Embedder
- [ ] Implement iterator and helper methods
- [ ] Create SimilarityResult struct
- [ ] Update lib.rs to export retrieval module
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P3-002
