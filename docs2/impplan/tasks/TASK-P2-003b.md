# TASK-P2-003b: EmbedderCategory Enum

```xml
<task_spec id="TASK-P2-003b" version="1.0">
<metadata>
  <title>EmbedderCategory Enum Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>15</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-02</requirement_ref>
    <requirement_ref>REQ-P2-05</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the EmbedderCategory enum that classifies each of the 13 embedders into
one of four semantic categories: Semantic, Temporal, Relational, or Structural.

Each category has an associated topic_weight that determines how much the embedder
contributes to weighted similarity calculations:
- Semantic: 1.0 (full contribution to topic relevance)
- Temporal: 0.0 (excluded from topic relevance, used for recency)
- Relational: 0.5 (partial contribution)
- Structural: 0.5 (partial contribution)

This enum must be created BEFORE EmbedderConfig (TASK-P2-003) as it is a field
in the EmbedderConfig struct.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE2-EMBEDDING-13SPACE.md#embedder_categories</file>
</input_context_files>

<prerequisites>
  <!-- No dependencies - foundation type -->
</prerequisites>

<scope>
  <in_scope>
    - Create EmbedderCategory enum with four variants
    - Implement topic_weight() method returning f32
    - Implement is_semantic() helper method
    - Implement is_temporal() helper method
    - Implement is_relational() helper method
    - Implement is_structural() helper method
    - Derive necessary traits (Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)
  </in_scope>
  <out_of_scope>
    - Embedder to category mapping (handled in TASK-P2-003)
    - Usage in similarity calculations (handled in Phase 3)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embedding/category.rs">
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
      pub enum EmbedderCategory {
          /// Semantic embedders capture meaning and content
          /// E1, E5, E6, E7, E10, E12, E13
          Semantic,
          /// Temporal embedders capture time-based features
          /// E2, E3, E4
          Temporal,
          /// Relational embedders capture relationships and emotions
          /// E8, E11
          Relational,
          /// Structural embedders capture form and patterns
          /// E9
          Structural,
      }

      impl EmbedderCategory {
          pub fn topic_weight(&amp;self) -> f32;
          pub fn is_semantic(&amp;self) -> bool;
          pub fn is_temporal(&amp;self) -> bool;
          pub fn is_relational(&amp;self) -> bool;
          pub fn is_structural(&amp;self) -> bool;
          pub fn all() -> [EmbedderCategory; 4];
      }
    </signature>
  </signatures>

  <constraints>
    - topic_weight values are compile-time constants
    - Semantic returns 1.0
    - Temporal returns 0.0
    - Relational returns 0.5
    - Structural returns 0.5
    - All methods are const where possible
  </constraints>

  <verification>
    - EmbedderCategory::Semantic.topic_weight() == 1.0
    - EmbedderCategory::Temporal.topic_weight() == 0.0
    - EmbedderCategory::Relational.topic_weight() == 0.5
    - EmbedderCategory::Structural.topic_weight() == 0.5
    - is_semantic() returns true only for Semantic variant
    - is_temporal() returns true only for Temporal variant
    - all() returns all four variants
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/embedding/category.rs

use serde::{Serialize, Deserialize};

/// Category classification for embedders, determining their role in similarity calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedderCategory {
    /// Semantic embedders capture meaning and content relevance.
    /// These contribute fully (weight 1.0) to topic similarity.
    /// Includes: E1 (Semantic), E5 (Causal), E6 (Sparse), E7 (Code),
    ///           E10 (Multimodal), E12 (LateInteract), E13 (SPLADE)
    Semantic,

    /// Temporal embedders capture time-based features like recency and periodicity.
    /// These are excluded (weight 0.0) from topic similarity but used for recency boosting.
    /// Includes: E2 (TempRecent), E3 (TempPeriodic), E4 (TempPosition)
    Temporal,

    /// Relational embedders capture relationships, emotions, and entity connections.
    /// These contribute partially (weight 0.5) to topic similarity.
    /// Includes: E8 (Emotional), E11 (Entity)
    Relational,

    /// Structural embedders capture form, patterns, and compositional structure.
    /// These contribute partially (weight 0.5) to topic similarity.
    /// Includes: E9 (HDC)
    Structural,
}

impl EmbedderCategory {
    /// Returns the topic weight for this category.
    /// Used in weighted similarity calculations.
    ///
    /// - Semantic: 1.0 (full contribution to topic relevance)
    /// - Temporal: 0.0 (excluded from topic relevance)
    /// - Relational: 0.5 (partial contribution)
    /// - Structural: 0.5 (partial contribution)
    pub const fn topic_weight(&amp;self) -> f32 {
        match self {
            EmbedderCategory::Semantic => 1.0,
            EmbedderCategory::Temporal => 0.0,
            EmbedderCategory::Relational => 0.5,
            EmbedderCategory::Structural => 0.5,
        }
    }

    /// Returns true if this is a Semantic category embedder
    pub const fn is_semantic(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Semantic)
    }

    /// Returns true if this is a Temporal category embedder
    pub const fn is_temporal(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Temporal)
    }

    /// Returns true if this is a Relational category embedder
    pub const fn is_relational(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Relational)
    }

    /// Returns true if this is a Structural category embedder
    pub const fn is_structural(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Structural)
    }

    /// Returns all category variants
    pub const fn all() -> [EmbedderCategory; 4] {
        [
            EmbedderCategory::Semantic,
            EmbedderCategory::Temporal,
            EmbedderCategory::Relational,
            EmbedderCategory::Structural,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_weights() {
        assert_eq!(EmbedderCategory::Semantic.topic_weight(), 1.0);
        assert_eq!(EmbedderCategory::Temporal.topic_weight(), 0.0);
        assert_eq!(EmbedderCategory::Relational.topic_weight(), 0.5);
        assert_eq!(EmbedderCategory::Structural.topic_weight(), 0.5);
    }

    #[test]
    fn test_is_semantic() {
        assert!(EmbedderCategory::Semantic.is_semantic());
        assert!(!EmbedderCategory::Temporal.is_semantic());
        assert!(!EmbedderCategory::Relational.is_semantic());
        assert!(!EmbedderCategory::Structural.is_semantic());
    }

    #[test]
    fn test_is_temporal() {
        assert!(!EmbedderCategory::Semantic.is_temporal());
        assert!(EmbedderCategory::Temporal.is_temporal());
        assert!(!EmbedderCategory::Relational.is_temporal());
        assert!(!EmbedderCategory::Structural.is_temporal());
    }

    #[test]
    fn test_is_relational() {
        assert!(!EmbedderCategory::Semantic.is_relational());
        assert!(!EmbedderCategory::Temporal.is_relational());
        assert!(EmbedderCategory::Relational.is_relational());
        assert!(!EmbedderCategory::Structural.is_relational());
    }

    #[test]
    fn test_is_structural() {
        assert!(!EmbedderCategory::Semantic.is_structural());
        assert!(!EmbedderCategory::Temporal.is_structural());
        assert!(!EmbedderCategory::Relational.is_structural());
        assert!(EmbedderCategory::Structural.is_structural());
    }

    #[test]
    fn test_all_categories() {
        let all = EmbedderCategory::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&amp;EmbedderCategory::Semantic));
        assert!(all.contains(&amp;EmbedderCategory::Temporal));
        assert!(all.contains(&amp;EmbedderCategory::Relational));
        assert!(all.contains(&amp;EmbedderCategory::Structural));
    }

    #[test]
    fn test_topic_weight_sum() {
        // Verify the weights make sense for normalization
        let total: f32 = EmbedderCategory::all()
            .iter()
            .map(|c| c.topic_weight())
            .sum();
        // 1.0 + 0.0 + 0.5 + 0.5 = 2.0
        assert_eq!(total, 2.0);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/embedding/category.rs">EmbedderCategory enum</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/embedding/mod.rs">Add pub mod category and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>EmbedderCategory enum has exactly 4 variants</criterion>
  <criterion>topic_weight() returns correct values for all variants</criterion>
  <criterion>is_semantic() returns true only for Semantic</criterion>
  <criterion>is_temporal() returns true only for Temporal</criterion>
  <criterion>is_relational() returns true only for Relational</criterion>
  <criterion>is_structural() returns true only for Structural</criterion>
  <criterion>all() returns all 4 variants</criterion>
  <criterion>Enum derives Serialize and Deserialize</criterion>
</validation_criteria>

<test_commands>
  <command description="Run category tests">cargo test --package context-graph-core category</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create category.rs in embedding directory
- [ ] Implement EmbedderCategory enum with 4 variants
- [ ] Add documentation comments for each variant listing which embedders belong
- [ ] Implement topic_weight() const method
- [ ] Implement is_semantic() const method
- [ ] Implement is_temporal() const method
- [ ] Implement is_relational() const method
- [ ] Implement is_structural() const method
- [ ] Implement all() const method
- [ ] Write unit tests for all topic_weight values
- [ ] Write unit tests for all is_* methods
- [ ] Update embedding/mod.rs to export category module
- [ ] Run tests to verify
- [ ] Proceed to TASK-P2-003
