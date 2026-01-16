# TASK-P3-003: Threshold Configurations

```xml
<task_spec id="TASK-P3-003" version="1.0">
<metadata>
  <title>Threshold Configuration Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>22</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-02</requirement_ref>
    <requirement_ref>REQ-P3-03</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the threshold configuration types for similarity and divergence detection.
Includes PerSpaceThresholds for both high (relevance) and low (divergence) thresholds,
SpaceWeights for weighted aggregation, and SimilarityThresholds container.

All default values come from the technical specification.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md#static_configuration</file>
</input_context_files>

<prerequisites>
  <check>Embedder enum exists (from Phase 2)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create PerSpaceThresholds struct
    - Create SpaceWeights struct with normalization
    - Create SimilarityThresholds container
    - Define static HIGH_THRESHOLDS, LOW_THRESHOLDS, SPACE_WEIGHTS
    - Implement get/set by Embedder
    - Implement Clone, Debug, Serialize, Deserialize
  </in_scope>
  <out_of_scope>
    - Threshold comparison logic (TASK-P3-005)
    - Runtime threshold adjustment
    - Threshold persistence
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/config.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct PerSpaceThresholds {
          pub e1_semantic: f32,
          pub e2_temp_recent: f32,
          // ... all 13 fields
      }

      impl PerSpaceThresholds {
          pub fn get_threshold(&amp;self, embedder: Embedder) -> f32;
          pub fn set_threshold(&amp;mut self, embedder: Embedder, threshold: f32);
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct SpaceWeights {
          weights: [f32; 13],
      }

      impl SpaceWeights {
          pub fn new(weights: [f32; 13]) -> Self;
          pub fn get_weight(&amp;self, embedder: Embedder) -> f32;
          pub fn normalize(&amp;mut self);
          pub fn sum(&amp;self) -> f32;
      }

      #[derive(Debug, Clone)]
      pub struct SimilarityThresholds {
          pub high: PerSpaceThresholds,
          pub low: PerSpaceThresholds,
      }

      pub static HIGH_THRESHOLDS: PerSpaceThresholds;
      pub static LOW_THRESHOLDS: PerSpaceThresholds;
      pub static SPACE_WEIGHTS: SpaceWeights;
    </signature>
  </signatures>

  <constraints>
    - All thresholds must be in 0.0..=1.0 range
    - High thresholds > low thresholds for each space
    - Weights should be positive
    - Default values match technical specification exactly
  </constraints>

  <verification>
    - Static configs have correct values from spec
    - get/set work for all embedders
    - Weight normalization produces sum ~13.0
    - Serialization roundtrip works
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/retrieval/config.rs

use serde::{Serialize, Deserialize};
use crate::embedding::Embedder;

/// Thresholds for each embedding space
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerSpaceThresholds {
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

impl PerSpaceThresholds {
    pub fn get_threshold(&amp;self, embedder: Embedder) -> f32 {
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

    pub fn set_threshold(&amp;mut self, embedder: Embedder, threshold: f32) {
        let threshold = threshold.clamp(0.0, 1.0);
        match embedder {
            Embedder::E1Semantic => self.e1_semantic = threshold,
            Embedder::E2TempRecent => self.e2_temp_recent = threshold,
            Embedder::E3TempPeriodic => self.e3_temp_periodic = threshold,
            Embedder::E4TempPosition => self.e4_temp_position = threshold,
            Embedder::E5Causal => self.e5_causal = threshold,
            Embedder::E6Sparse => self.e6_sparse = threshold,
            Embedder::E7Code => self.e7_code = threshold,
            Embedder::E8Emotional => self.e8_emotional = threshold,
            Embedder::E9HDC => self.e9_hdc = threshold,
            Embedder::E10Multimodal => self.e10_multimodal = threshold,
            Embedder::E11Entity => self.e11_entity = threshold,
            Embedder::E12LateInteract => self.e12_late_interact = threshold,
            Embedder::E13SPLADE => self.e13_splade = threshold,
        }
    }

    pub fn to_array(&amp;self) -> [f32; 13] {
        [
            self.e1_semantic, self.e2_temp_recent, self.e3_temp_periodic,
            self.e4_temp_position, self.e5_causal, self.e6_sparse,
            self.e7_code, self.e8_emotional, self.e9_hdc,
            self.e10_multimodal, self.e11_entity, self.e12_late_interact,
            self.e13_splade,
        ]
    }
}

/// High thresholds for relevance detection
pub fn high_thresholds() -> PerSpaceThresholds {
    PerSpaceThresholds {
        e1_semantic: 0.75,
        e2_temp_recent: 0.70,
        e3_temp_periodic: 0.70,
        e4_temp_position: 0.70,
        e5_causal: 0.70,
        e6_sparse: 0.60,
        e7_code: 0.80,
        e8_emotional: 0.70,
        e9_hdc: 0.70,
        e10_multimodal: 0.70,
        e11_entity: 0.70,
        e12_late_interact: 0.70,
        e13_splade: 0.60,
    }
}

/// Low thresholds for divergence detection
pub fn low_thresholds() -> PerSpaceThresholds {
    PerSpaceThresholds {
        e1_semantic: 0.30,
        e2_temp_recent: 0.30,
        e3_temp_periodic: 0.30,
        e4_temp_position: 0.30,
        e5_causal: 0.25,
        e6_sparse: 0.20,
        e7_code: 0.35,
        e8_emotional: 0.30,
        e9_hdc: 0.30,
        e10_multimodal: 0.30,
        e11_entity: 0.30,
        e12_late_interact: 0.30,
        e13_splade: 0.20,
    }
}

/// Weights for combining space scores
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpaceWeights {
    weights: [f32; 13],
}

impl SpaceWeights {
    pub fn new(weights: [f32; 13]) -> Self {
        Self { weights }
    }

    pub fn get_weight(&amp;self, embedder: Embedder) -> f32 {
        self.weights[embedder.index()]
    }

    pub fn set_weight(&amp;mut self, embedder: Embedder, weight: f32) {
        self.weights[embedder.index()] = weight.max(0.0);
    }

    pub fn sum(&amp;self) -> f32 {
        self.weights.iter().sum()
    }

    /// Normalize weights so they sum to 13.0 (one per space)
    pub fn normalize(&amp;mut self) {
        let sum = self.sum();
        if sum > 0.0 {
            let factor = 13.0 / sum;
            for w in &amp;mut self.weights {
                *w *= factor;
            }
        }
    }

    pub fn normalized(&amp;self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    pub fn as_slice(&amp;self) -> &amp;[f32; 13] {
        &amp;self.weights
    }
}

/// Default space weights from spec
pub fn default_weights() -> SpaceWeights {
    SpaceWeights::new([
        1.0,  // E1 Semantic
        0.6,  // E2 Temporal Recent
        0.6,  // E3 Temporal Periodic
        0.6,  // E4 Temporal Position
        0.9,  // E5 Causal
        0.7,  // E6 Sparse
        0.85, // E7 Code
        0.6,  // E8 Emotional
        0.6,  // E9 HDC
        0.6,  // E10 Multimodal
        0.6,  // E11 Entity
        0.6,  // E12 Late Interact
        0.7,  // E13 SPLADE
    ])
}

/// Container for both high and low thresholds
#[derive(Debug, Clone)]
pub struct SimilarityThresholds {
    pub high: PerSpaceThresholds,
    pub low: PerSpaceThresholds,
}

impl Default for SimilarityThresholds {
    fn default() -> Self {
        Self {
            high: high_thresholds(),
            low: low_thresholds(),
        }
    }
}

impl SimilarityThresholds {
    pub fn new(high: PerSpaceThresholds, low: PerSpaceThresholds) -> Self {
        Self { high, low }
    }
}

/// Lookback duration for divergence detection
pub const RECENT_LOOKBACK_SECS: u64 = 2 * 60 * 60; // 2 hours

/// Maximum recent memories for divergence check
pub const MAX_RECENT_MEMORIES: usize = 50;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_thresholds_values() {
        let thresholds = high_thresholds();
        assert_eq!(thresholds.e1_semantic, 0.75);
        assert_eq!(thresholds.e7_code, 0.80);
        assert_eq!(thresholds.e6_sparse, 0.60);
    }

    #[test]
    fn test_low_thresholds_values() {
        let thresholds = low_thresholds();
        assert_eq!(thresholds.e1_semantic, 0.30);
        assert_eq!(thresholds.e5_causal, 0.25);
        assert_eq!(thresholds.e6_sparse, 0.20);
    }

    #[test]
    fn test_high_greater_than_low() {
        let high = high_thresholds();
        let low = low_thresholds();

        for embedder in Embedder::all() {
            let h = high.get_threshold(embedder);
            let l = low.get_threshold(embedder);
            assert!(h > l, "{:?}: high {} should be > low {}", embedder, h, l);
        }
    }

    #[test]
    fn test_weights_sum() {
        let weights = default_weights();
        let sum = weights.sum();
        assert!(sum > 8.0 && sum < 10.0);
    }

    #[test]
    fn test_weights_normalization() {
        let mut weights = default_weights();
        weights.normalize();
        let sum = weights.sum();
        assert!((sum - 13.0).abs() < 0.001);
    }

    #[test]
    fn test_get_set_threshold() {
        let mut thresholds = high_thresholds();
        thresholds.set_threshold(Embedder::E1Semantic, 0.85);
        assert_eq!(thresholds.get_threshold(Embedder::E1Semantic), 0.85);
    }

    #[test]
    fn test_threshold_clamping() {
        let mut thresholds = high_thresholds();
        thresholds.set_threshold(Embedder::E1Semantic, 1.5);
        assert_eq!(thresholds.get_threshold(Embedder::E1Semantic), 1.0);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/config.rs">Threshold and weight configurations</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Add pub mod config and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>HIGH_THRESHOLDS matches spec values exactly</criterion>
  <criterion>LOW_THRESHOLDS matches spec values exactly</criterion>
  <criterion>All high thresholds > corresponding low thresholds</criterion>
  <criterion>SPACE_WEIGHTS matches spec values</criterion>
  <criterion>Weight normalization produces sum = 13.0</criterion>
  <criterion>get/set work for all embedders</criterion>
</validation_criteria>

<test_commands>
  <command description="Run config tests">cargo test --package context-graph-core config</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create config.rs in retrieval directory
- [ ] Implement PerSpaceThresholds struct
- [ ] Implement SpaceWeights struct with normalization
- [ ] Define high_thresholds() with spec values
- [ ] Define low_thresholds() with spec values
- [ ] Define default_weights() with spec values
- [ ] Implement SimilarityThresholds container
- [ ] Write unit tests verifying spec values
- [ ] Run tests to verify
- [ ] Proceed to TASK-P3-004
