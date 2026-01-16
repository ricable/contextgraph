# TASK-P3-002: DivergenceAlert Type

```xml
<task_spec id="TASK-P3-002" version="1.0">
<metadata>
  <title>DivergenceAlert Type Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>21</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-04</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the DivergenceAlert type that represents a detected divergence between
the current query and a recent memory. Divergence alerts help surface when the
user's current work has shifted away from recent context.

Alerts include the memory ID, which embedding space detected the divergence,
the similarity score, and a summary of the memory content.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md#data_models</file>
</input_context_files>

<prerequisites>
  <check>Embedder enum exists (from Phase 2)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create DivergenceAlert struct
    - Implement formatting method for display
    - Add severity level based on score
    - Implement Clone, Debug, Serialize, Deserialize
  </in_scope>
  <out_of_scope>
    - Divergence detection logic (TASK-P3-006)
    - Threshold configuration (TASK-P3-003)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/divergence.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct DivergenceAlert {
          pub memory_id: Uuid,
          pub space: Embedder,
          pub similarity_score: f32,
          pub memory_summary: String,
          pub detected_at: DateTime&lt;Utc&gt;,
      }

      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
      pub enum DivergenceSeverity {
          Low,    // score 0.20..0.30
          Medium, // score 0.10..0.20
          High,   // score 0.00..0.10
      }

      impl DivergenceAlert {
          pub fn new(memory_id: Uuid, space: Embedder, similarity_score: f32, memory_summary: String) -> Self;
          pub fn severity(&amp;self) -> DivergenceSeverity;
          pub fn format_alert(&amp;self) -> String;
      }
    </signature>
  </signatures>

  <constraints>
    - memory_summary limited to first 100 characters
    - detected_at auto-set to Utc::now()
    - similarity_score indicates low similarity (below threshold)
    - Format matches spec: "⚠️ DIVERGENCE in {space}: ..."
  </constraints>

  <verification>
    - DivergenceAlert can be created and serialized
    - Severity correctly categorizes scores
    - Format string matches specification
    - Summary truncation works
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/retrieval/divergence.rs

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::embedding::Embedder;

/// Severity level of a divergence alert
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceSeverity {
    /// Score 0.20..0.30 - slight divergence
    Low,
    /// Score 0.10..0.20 - moderate divergence
    Medium,
    /// Score 0.00..0.10 - significant divergence
    High,
}

impl DivergenceSeverity {
    pub fn from_score(score: f32) -> Self {
        if score < 0.10 {
            DivergenceSeverity::High
        } else if score < 0.20 {
            DivergenceSeverity::Medium
        } else {
            DivergenceSeverity::Low
        }
    }

    pub fn as_str(&amp;self) -> &amp;'static str {
        match self {
            DivergenceSeverity::Low => "LOW",
            DivergenceSeverity::Medium => "MEDIUM",
            DivergenceSeverity::High => "HIGH",
        }
    }
}

/// Alert indicating divergence between current query and recent memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceAlert {
    /// ID of the recent memory that diverges
    pub memory_id: Uuid,
    /// Embedding space where divergence detected
    pub space: Embedder,
    /// Similarity score (low = more divergent)
    pub similarity_score: f32,
    /// First 100 chars of memory content
    pub memory_summary: String,
    /// When the alert was generated
    pub detected_at: DateTime&lt;Utc&gt;,
}

impl DivergenceAlert {
    /// Maximum characters in memory summary
    const MAX_SUMMARY_LEN: usize = 100;

    /// Create a new divergence alert
    pub fn new(
        memory_id: Uuid,
        space: Embedder,
        similarity_score: f32,
        memory_content: &amp;str,
    ) -> Self {
        let memory_summary = truncate_summary(memory_content, Self::MAX_SUMMARY_LEN);

        Self {
            memory_id,
            space,
            similarity_score,
            memory_summary,
            detected_at: Utc::now(),
        }
    }

    /// Get the severity level of this divergence
    pub fn severity(&amp;self) -> DivergenceSeverity {
        DivergenceSeverity::from_score(self.similarity_score)
    }

    /// Format the alert for display
    pub fn format_alert(&amp;self) -> String {
        format!(
            "⚠️ DIVERGENCE in {:?}: Recent work on \"{}\" (similarity: {:.2})",
            self.space,
            self.memory_summary,
            self.similarity_score
        )
    }

    /// Format with severity indicator
    pub fn format_with_severity(&amp;self) -> String {
        let severity = self.severity();
        format!(
            "[{}] {} DIVERGENCE in {:?}: \"{}\" (similarity: {:.2})",
            severity.as_str(),
            match severity {
                DivergenceSeverity::High => "🔴",
                DivergenceSeverity::Medium => "🟡",
                DivergenceSeverity::Low => "🟢",
            },
            self.space,
            self.memory_summary,
            self.similarity_score
        )
    }
}

/// Truncate content to max_len, adding ellipsis if needed
fn truncate_summary(content: &amp;str, max_len: usize) -> String {
    let content = content.trim();
    if content.len() <= max_len {
        content.to_string()
    } else {
        // Find word boundary near max_len - 3 (for "...")
        let target = max_len.saturating_sub(3);
        let truncated = &amp;content[..target];

        // Try to break at last space
        if let Some(last_space) = truncated.rfind(' ') {
            if last_space > target / 2 {
                return format!("{}...", &amp;content[..last_space]);
            }
        }

        format!("{}...", truncated)
    }
}

/// Collection of divergence alerts for a query
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DivergenceReport {
    pub alerts: Vec&lt;DivergenceAlert&gt;,
}

impl DivergenceReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&amp;mut self, alert: DivergenceAlert) {
        self.alerts.push(alert);
    }

    pub fn is_empty(&amp;self) -> bool {
        self.alerts.is_empty()
    }

    pub fn len(&amp;self) -> usize {
        self.alerts.len()
    }

    /// Get the most severe alert
    pub fn most_severe(&amp;self) -> Option&lt;&amp;DivergenceAlert&gt; {
        self.alerts.iter().min_by(|a, b| {
            a.similarity_score.partial_cmp(&amp;b.similarity_score).unwrap()
        })
    }

    /// Sort alerts by severity (lowest score first)
    pub fn sort_by_severity(&amp;mut self) {
        self.alerts.sort_by(|a, b| {
            a.similarity_score.partial_cmp(&amp;b.similarity_score).unwrap()
        });
    }

    /// Format all alerts for display
    pub fn format_all(&amp;self) -> String {
        if self.alerts.is_empty() {
            return String::from("No divergence detected.");
        }

        self.alerts
            .iter()
            .map(|a| a.format_alert())
            .collect::&lt;Vec&lt;_&gt;&gt;()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divergence_alert_creation() {
        let alert = DivergenceAlert::new(
            Uuid::new_v4(),
            Embedder::E1Semantic,
            0.15,
            "This is some content from a recent memory",
        );

        assert_eq!(alert.space, Embedder::E1Semantic);
        assert_eq!(alert.similarity_score, 0.15);
        assert!(!alert.memory_summary.is_empty());
    }

    #[test]
    fn test_severity_levels() {
        assert_eq!(DivergenceSeverity::from_score(0.05), DivergenceSeverity::High);
        assert_eq!(DivergenceSeverity::from_score(0.15), DivergenceSeverity::Medium);
        assert_eq!(DivergenceSeverity::from_score(0.25), DivergenceSeverity::Low);
    }

    #[test]
    fn test_summary_truncation() {
        let long_content = "a".repeat(200);
        let summary = truncate_summary(&amp;long_content, 100);
        assert!(summary.len() <= 100);
        assert!(summary.ends_with("..."));
    }

    #[test]
    fn test_summary_no_truncation() {
        let short_content = "Short content";
        let summary = truncate_summary(short_content, 100);
        assert_eq!(summary, short_content);
    }

    #[test]
    fn test_format_alert() {
        let alert = DivergenceAlert::new(
            Uuid::new_v4(),
            Embedder::E7Code,
            0.12,
            "Implementing new feature",
        );

        let formatted = alert.format_alert();
        assert!(formatted.contains("DIVERGENCE"));
        assert!(formatted.contains("E7Code"));
        assert!(formatted.contains("0.12"));
    }

    #[test]
    fn test_divergence_report() {
        let mut report = DivergenceReport::new();
        assert!(report.is_empty());

        report.add(DivergenceAlert::new(
            Uuid::new_v4(),
            Embedder::E1Semantic,
            0.20,
            "First memory",
        ));
        report.add(DivergenceAlert::new(
            Uuid::new_v4(),
            Embedder::E7Code,
            0.05,
            "Second memory",
        ));

        assert_eq!(report.len(), 2);

        let most_severe = report.most_severe().unwrap();
        assert_eq!(most_severe.similarity_score, 0.05);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/divergence.rs">DivergenceAlert and related types</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Add pub mod divergence and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>DivergenceAlert created with correct fields</criterion>
  <criterion>Summary truncated to 100 chars</criterion>
  <criterion>Severity correctly categorizes scores</criterion>
  <criterion>Format string matches specification</criterion>
  <criterion>DivergenceReport collects and sorts alerts</criterion>
</validation_criteria>

<test_commands>
  <command description="Run divergence type tests">cargo test --package context-graph-core divergence</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create divergence.rs in retrieval directory
- [ ] Implement DivergenceSeverity enum
- [ ] Implement DivergenceAlert struct
- [ ] Implement format_alert method
- [ ] Implement summary truncation
- [ ] Implement DivergenceReport collection
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P3-003
