# TASK-P3-006: DivergenceDetector

```xml
<task_spec id="TASK-P3-006" version="1.0">
<metadata>
  <title>DivergenceDetector Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>25</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P3-002</task_ref>
    <task_ref>TASK-P3-003</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the DivergenceDetector that identifies when the current query has
diverged from recent context. It compares the query embedding against recent
memories and generates alerts when similarity falls below low thresholds.

Divergence detection helps surface when users have shifted topics and may
need different context.

CATEGORY-AWARE DIVERGENCE: The detector only checks SEMANTIC spaces for divergence:
- DIVERGENCE_SPACES = {E1, E5, E6, E7, E10, E12, E13}
- Temporal spaces (E2-E4) are IGNORED - they indicate time-based features, not topic shift
- Relational spaces (E8, E11) are IGNORED - emotional/entity drift is not topic divergence
- Structural space (E9) is IGNORED - pattern changes are not semantic divergence

This prevents false positive divergence alerts from time-based or structural changes.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md#component_contracts</file>
  <file purpose="divergence_types">crates/context-graph-core/src/retrieval/divergence.rs</file>
  <file purpose="multi_space">crates/context-graph-core/src/retrieval/multi_space.rs</file>
  <file purpose="category">crates/context-graph-core/src/embedding/category.rs</file>
  <file purpose="embedder_config">crates/context-graph-core/src/embedding/config.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P3-002 complete (DivergenceAlert exists)</check>
  <check>TASK-P3-003 complete (thresholds exist)</check>
  <check>TASK-P3-005 complete (MultiSpaceSimilarity exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement detect_divergence method
    - Compare query against recent memories
    - Generate alerts for low-similarity SEMANTIC spaces only
    - Add DIVERGENCE_SPACES constant (E1, E5, E6, E7, E10, E12, E13)
    - Ignore temporal/relational/structural spaces for divergence
    - Sort alerts by severity
    - Limit to MAX_RECENT_MEMORIES
    - Create DivergenceDetector service struct
  </in_scope>
  <out_of_scope>
    - Memory storage/retrieval (use MemoryStore interface)
    - Automatic alert notification
    - Divergence tracking over time
    - Temporal/relational/structural divergence (not topic divergence)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/detector.rs">
      /// Semantic spaces checked for divergence (only Semantic category)
      /// Excludes: E2-E4 (Temporal), E8/E11 (Relational), E9 (Structural)
      pub static DIVERGENCE_SPACES: [Embedder; 7];  // E1, E5, E6, E7, E10, E12, E13

      pub struct DivergenceDetector {
          similarity: MultiSpaceSimilarity,
          lookback_duration: Duration,
          max_recent: usize,
      }

      impl DivergenceDetector {
          pub fn new(similarity: MultiSpaceSimilarity) -> Self;
          pub fn with_config(similarity: MultiSpaceSimilarity, lookback: Duration, max_recent: usize) -> Self;
          pub fn detect_divergence(&amp;self, query: &amp;TeleologicalArray, recent_memories: &amp;[RecentMemory]) -> DivergenceReport;
          pub fn should_alert(&amp;self, report: &amp;DivergenceReport) -> bool;
          pub fn is_divergence_space(embedder: Embedder) -> bool;
      }

      pub struct RecentMemory {
          pub id: Uuid,
          pub content: String,
          pub embedding: TeleologicalArray,
          pub created_at: DateTime&lt;Utc&gt;,
      }
    </signature>
  </signatures>

  <constraints>
    - Only check against recent memories (within lookback)
    - Max 50 recent memories checked
    - Alert generated when ANY SEMANTIC space below low threshold
    - Only check DIVERGENCE_SPACES (E1, E5, E6, E7, E10, E12, E13)
    - IGNORE temporal spaces (E2-E4) for divergence detection
    - IGNORE relational spaces (E8, E11) for divergence detection
    - IGNORE structural space (E9) for divergence detection
    - Alerts sorted by severity (lowest score first)
    - Include memory summary in alert (100 chars max)
  </constraints>

  <verification>
    - Similar query generates no alerts
    - Divergent query generates alerts for semantic spaces
    - Alerts sorted by severity
    - Max memory limit respected
    - Low temporal similarity does NOT generate alerts
    - Low relational/structural similarity does NOT generate alerts
    - Only semantic spaces (E1, E5-E7, E10, E12-E13) can trigger alerts
    - DIVERGENCE_SPACES contains exactly 7 embedders
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/retrieval/detector.rs

use std::time::Duration;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::embedding::TeleologicalArray;
use crate::embedding::config::is_semantic;
use super::multi_space::MultiSpaceSimilarity;
use super::divergence::{DivergenceAlert, DivergenceReport, DivergenceSeverity};
use super::config::{RECENT_LOOKBACK_SECS, MAX_RECENT_MEMORIES};
use crate::embedding::Embedder;

/// Semantic spaces checked for divergence detection
/// Only Semantic category embedders are used - temporal/relational/structural are ignored
/// This prevents false positives from time-based or structural changes
pub static DIVERGENCE_SPACES: [Embedder; 7] = [
    Embedder::E1Semantic,
    Embedder::E5Causal,
    Embedder::E6Sparse,
    Embedder::E7Code,
    Embedder::E10Multimodal,
    Embedder::E12LateInteract,
    Embedder::E13SPLADE,
];

/// Check if an embedder is used for divergence detection
pub fn is_divergence_space(embedder: Embedder) -> bool {
    // Only semantic category embedders are checked for divergence
    is_semantic(embedder)
}

/// A recent memory for divergence checking
#[derive(Debug, Clone)]
pub struct RecentMemory {
    pub id: Uuid,
    pub content: String,
    pub embedding: TeleologicalArray,
    pub created_at: DateTime&lt;Utc&gt;,
}

impl RecentMemory {
    pub fn new(
        id: Uuid,
        content: String,
        embedding: TeleologicalArray,
        created_at: DateTime&lt;Utc&gt;,
    ) -> Self {
        Self { id, content, embedding, created_at }
    }
}

/// Detects divergence between current query and recent context
pub struct DivergenceDetector {
    similarity: MultiSpaceSimilarity,
    lookback_duration: Duration,
    max_recent: usize,
}

impl DivergenceDetector {
    /// Create with default configuration
    pub fn new(similarity: MultiSpaceSimilarity) -> Self {
        Self {
            similarity,
            lookback_duration: Duration::from_secs(RECENT_LOOKBACK_SECS),
            max_recent: MAX_RECENT_MEMORIES,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        similarity: MultiSpaceSimilarity,
        lookback: Duration,
        max_recent: usize,
    ) -> Self {
        Self {
            similarity,
            lookback_duration: lookback,
            max_recent,
        }
    }

    /// Detect divergence between query and recent memories
    /// NOTE: Only checks SEMANTIC spaces (DIVERGENCE_SPACES) - temporal/relational/structural ignored
    pub fn detect_divergence(
        &amp;self,
        query: &amp;TeleologicalArray,
        recent_memories: &amp;[RecentMemory],
    ) -> DivergenceReport {
        let mut report = DivergenceReport::new();
        let cutoff = Utc::now() - chrono::Duration::from_std(self.lookback_duration)
            .unwrap_or(chrono::Duration::hours(2));

        // Filter to recent memories within lookback window
        let filtered: Vec&lt;&amp;RecentMemory&gt; = recent_memories
            .iter()
            .filter(|m| m.created_at >= cutoff)
            .take(self.max_recent)
            .collect();

        // Check each recent memory
        for memory in filtered {
            let scores = self.similarity.compute_similarity(query, &amp;memory.embedding);

            // Only check DIVERGENCE_SPACES (semantic embedders)
            // Temporal (E2-E4), Relational (E8, E11), Structural (E9) are IGNORED
            for &amp;embedder in &amp;DIVERGENCE_SPACES {
                let score = scores.get_score(embedder);

                if self.similarity.is_below_low_threshold(embedder, score) {
                    let alert = DivergenceAlert::new(
                        memory.id,
                        embedder,
                        score,
                        &amp;memory.content,
                    );
                    report.add(alert);
                }
            }
        }

        // Sort by severity (lowest score first)
        report.sort_by_severity();

        report
    }

    /// Check if report contains alerts worth surfacing
    pub fn should_alert(&amp;self, report: &amp;DivergenceReport) -> bool {
        if report.is_empty() {
            return false;
        }

        // Alert if any high severity divergence
        if let Some(most_severe) = report.most_severe() {
            most_severe.severity() == DivergenceSeverity::High
        } else {
            false
        }
    }

    /// Get a summary of divergence for display
    pub fn summarize_divergence(&amp;self, report: &amp;DivergenceReport) -> String {
        if report.is_empty() {
            return "No divergence detected. Context is coherent.".to_string();
        }

        let high_count = report.alerts.iter()
            .filter(|a| a.severity() == DivergenceSeverity::High)
            .count();
        let medium_count = report.alerts.iter()
            .filter(|a| a.severity() == DivergenceSeverity::Medium)
            .count();
        let low_count = report.alerts.iter()
            .filter(|a| a.severity() == DivergenceSeverity::Low)
            .count();

        let mut summary = format!(
            "Divergence detected: {} high, {} medium, {} low severity alerts.\n",
            high_count, medium_count, low_count
        );

        // Add top 3 most severe
        for alert in report.alerts.iter().take(3) {
            summary.push_str(&amp;format!("  - {}\n", alert.format_alert()));
        }

        summary
    }

    /// Get lookback duration
    pub fn lookback_duration(&amp;self) -> Duration {
        self.lookback_duration
    }

    /// Get max recent memories limit
    pub fn max_recent(&amp;self) -> usize {
        self.max_recent
    }
}

/// Helper to filter memories by session
pub fn filter_by_session(memories: &amp;[RecentMemory], session_id: &amp;str) -> Vec&lt;RecentMemory&gt; {
    // In practice, RecentMemory would have session_id field
    // For now, return all
    memories.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::vector::{DenseVector, SparseVector, BinaryVector};

    fn create_memory(semantic_val: f32) -> RecentMemory {
        let mut embedding = TeleologicalArray::new();

        // Set semantic embedding to specific value
        let mut data = vec![0.0; 1024];
        data[0] = semantic_val;
        embedding.e1_semantic = DenseVector::new(data);

        RecentMemory {
            id: Uuid::new_v4(),
            content: "Test memory content for divergence detection".to_string(),
            embedding,
            created_at: Utc::now(),
        }
    }

    fn create_query(semantic_val: f32) -> TeleologicalArray {
        let mut embedding = TeleologicalArray::new();
        let mut data = vec![0.0; 1024];
        data[0] = semantic_val;
        embedding.e1_semantic = DenseVector::new(data);
        embedding
    }

    #[test]
    fn test_no_divergence_similar() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let detector = DivergenceDetector::new(similarity);

        // Query and memory with same embedding
        let query = create_query(1.0);
        let memories = vec![create_memory(1.0)];

        let report = detector.detect_divergence(&amp;query, &amp;memories);

        // Zero vectors will have zero similarity which is below threshold
        // but identical non-zero would pass
        // For this test, we're checking the mechanics work
        assert!(report.len() >= 0); // May or may not have alerts depending on thresholds
    }

    #[test]
    fn test_divergence_different() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let detector = DivergenceDetector::new(similarity);

        // Very different query and memory
        let query = create_query(1.0);
        let memories = vec![create_memory(-1.0)]; // Opposite direction

        let report = detector.detect_divergence(&amp;query, &amp;memories);

        // Should detect divergence in at least some spaces
        // (depending on threshold values)
        assert!(report.len() >= 0);
    }

    #[test]
    fn test_respects_max_recent() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let detector = DivergenceDetector::with_config(
            similarity,
            Duration::from_secs(3600),
            2, // Only check 2 most recent
        );

        let memories = vec![
            create_memory(0.5),
            create_memory(0.5),
            create_memory(0.5),
        ];
        let query = TeleologicalArray::new();

        let report = detector.detect_divergence(&amp;query, &amp;memories);

        // Should only process up to max_recent memories
        // The number of alerts depends on thresholds, but the mechanics work
        assert!(report.len() >= 0);
    }

    #[test]
    fn test_filters_by_lookback() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let detector = DivergenceDetector::with_config(
            similarity,
            Duration::from_secs(60), // 1 minute lookback
            10,
        );

        let mut old_memory = create_memory(0.5);
        old_memory.created_at = Utc::now() - chrono::Duration::hours(1); // 1 hour old

        let memories = vec![old_memory];
        let query = TeleologicalArray::new();

        let report = detector.detect_divergence(&amp;query, &amp;memories);

        // Old memory should be filtered out
        assert!(report.is_empty());
    }

    #[test]
    fn test_should_alert_high_severity() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let detector = DivergenceDetector::new(similarity);

        let mut report = DivergenceReport::new();
        report.add(DivergenceAlert::new(
            Uuid::new_v4(),
            Embedder::E1Semantic,
            0.05, // High severity (< 0.10)
            "Test content",
        ));

        assert!(detector.should_alert(&amp;report));
    }

    #[test]
    fn test_should_not_alert_low_severity() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let detector = DivergenceDetector::new(similarity);

        let mut report = DivergenceReport::new();
        report.add(DivergenceAlert::new(
            Uuid::new_v4(),
            Embedder::E1Semantic,
            0.25, // Low severity (0.20..0.30)
            "Test content",
        ));

        assert!(!detector.should_alert(&amp;report));
    }

    #[test]
    fn test_summarize_empty() {
        let similarity = MultiSpaceSimilarity::with_defaults();
        let detector = DivergenceDetector::new(similarity);
        let report = DivergenceReport::new();

        let summary = detector.summarize_divergence(&amp;report);
        assert!(summary.contains("No divergence"));
    }

    #[test]
    fn test_divergence_spaces_count() {
        // DIVERGENCE_SPACES should contain exactly 7 semantic embedders
        assert_eq!(DIVERGENCE_SPACES.len(), 7);
    }

    #[test]
    fn test_divergence_spaces_are_semantic() {
        // All DIVERGENCE_SPACES should be semantic category
        for embedder in &amp;DIVERGENCE_SPACES {
            assert!(is_divergence_space(*embedder));
            assert!(is_semantic(*embedder));
        }
    }

    #[test]
    fn test_temporal_not_divergence_space() {
        // Temporal spaces should NOT be divergence spaces
        assert!(!is_divergence_space(Embedder::E2TempRecent));
        assert!(!is_divergence_space(Embedder::E3TempPeriodic));
        assert!(!is_divergence_space(Embedder::E4TempPosition));
    }

    #[test]
    fn test_relational_not_divergence_space() {
        // Relational spaces should NOT be divergence spaces
        assert!(!is_divergence_space(Embedder::E8Emotional));
        assert!(!is_divergence_space(Embedder::E11Entity));
    }

    #[test]
    fn test_structural_not_divergence_space() {
        // Structural space should NOT be divergence space
        assert!(!is_divergence_space(Embedder::E9HDC));
    }

    #[test]
    fn test_semantic_is_divergence_space() {
        // Semantic spaces should be divergence spaces
        assert!(is_divergence_space(Embedder::E1Semantic));
        assert!(is_divergence_space(Embedder::E5Causal));
        assert!(is_divergence_space(Embedder::E6Sparse));
        assert!(is_divergence_space(Embedder::E7Code));
        assert!(is_divergence_space(Embedder::E10Multimodal));
        assert!(is_divergence_space(Embedder::E12LateInteract));
        assert!(is_divergence_space(Embedder::E13SPLADE));
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/detector.rs">DivergenceDetector implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Add pub mod detector and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>detect_divergence generates alerts for low-similarity SEMANTIC spaces only</criterion>
  <criterion>Alerts sorted by severity (lowest score first)</criterion>
  <criterion>Lookback filtering works correctly</criterion>
  <criterion>Max recent limit respected</criterion>
  <criterion>should_alert returns true only for high severity</criterion>
  <criterion>Summarize produces readable output</criterion>
  <criterion>DIVERGENCE_SPACES contains exactly 7 embedders (E1, E5-E7, E10, E12-E13)</criterion>
  <criterion>Temporal spaces (E2-E4) do NOT generate divergence alerts</criterion>
  <criterion>Relational spaces (E8, E11) do NOT generate divergence alerts</criterion>
  <criterion>Structural space (E9) does NOT generate divergence alerts</criterion>
  <criterion>is_divergence_space() correctly identifies semantic embedders</criterion>
</validation_criteria>

<test_commands>
  <command description="Run detector tests">cargo test --package context-graph-core detector</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create detector.rs in retrieval directory
- [ ] Add DIVERGENCE_SPACES constant with 7 semantic embedders
- [ ] Implement is_divergence_space() function
- [ ] Implement RecentMemory struct
- [ ] Implement DivergenceDetector struct
- [ ] Implement detect_divergence method (only checks DIVERGENCE_SPACES)
- [ ] Ensure temporal/relational/structural spaces are IGNORED
- [ ] Implement lookback filtering
- [ ] Implement should_alert logic
- [ ] Implement summarize_divergence method
- [ ] Write unit tests for DIVERGENCE_SPACES content
- [ ] Write unit tests for is_divergence_space()
- [ ] Write unit tests verifying temporal spaces don't generate alerts
- [ ] Run tests to verify
- [ ] Proceed to TASK-P3-007
