# Task: TASK-P5-006 - ContextFormatter

```xml
<task_spec id="TASK-P5-006" version="1.0">
<metadata>
  <title>ContextFormatter</title>
  <phase>5</phase>
  <sequence>41</sequence>
  <layer>logic</layer>
  <estimated_loc>200</estimated_loc>
  <dependencies>
    <dependency task="TASK-P5-001">InjectionCandidate type</dependency>
    <dependency task="TASK-P3-003">DivergenceAlert type</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">ContextFormatter</artifact>
  </produces>
</metadata>

<context>
  <background>
    ContextFormatter transforms selected candidates and alerts into markdown-formatted
    text suitable for injection into Claude Code hooks. The format must be human-readable
    and provide useful context without being overwhelming.
  </background>
  <business_value>
    Produces the actual context string that gets injected, making stored memories
    actionable during coding sessions.
  </business_value>
  <technical_context>
    Creates markdown with sections for each category. Includes timestamps,
    relevance indicators, and memory summaries. Output goes directly into
    system prompt injection.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-core/src/injection/candidate.rs with InjectionCandidate</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/similarity/divergence.rs with DivergenceAlert</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>ContextFormatter struct</item>
    <item>format_full_context() for SessionStart</item>
    <item>format_brief_context() for PreToolUse</item>
    <item>format_divergence_alert() helper</item>
    <item>summarize_memory() helper</item>
    <item>Unit tests for formatting</item>
  </includes>
  <excludes>
    <item>Candidate selection (TASK-P5-005)</item>
    <item>Pipeline orchestration (TASK-P5-007)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>format_full_context() produces markdown with correct sections</description>
    <verification>Unit test verifies section headers and structure</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>format_brief_context() produces concise output under 200 tokens</description>
    <verification>Unit test verifies length constraint</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>summarize_memory() truncates at word boundary</description>
    <verification>Unit test verifies truncation behavior</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Divergence alerts formatted prominently</description>
    <verification>Unit test verifies alert formatting</verification>
  </criterion>

  <signatures>
    <signature name="ContextFormatter">
      <code>
pub struct ContextFormatter;
      </code>
    </signature>
    <signature name="format_full_context">
      <code>
impl ContextFormatter {
    pub fn format_full_context(
        candidates: &amp;[InjectionCandidate],
        alerts: &amp;[DivergenceAlert],
    ) -> String
}
      </code>
    </signature>
    <signature name="format_brief_context">
      <code>
impl ContextFormatter {
    pub fn format_brief_context(candidates: &amp;[InjectionCandidate]) -> String
}
      </code>
    </signature>
    <signature name="summarize_memory">
      <code>
impl ContextFormatter {
    pub fn summarize_memory(content: &amp;str, max_words: usize) -> String
}
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="output">Full context uses markdown headers (##, ###)</constraint>
    <constraint type="output">Brief context is single paragraph, max 200 tokens</constraint>
    <constraint type="config">SUMMARY_MAX_WORDS = 50</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-core/src/injection/formatter.rs

use chrono::{DateTime, Utc};
use super::candidate::{InjectionCandidate, InjectionCategory};
use crate::similarity::DivergenceAlert;

/// Maximum words for memory summaries.
pub const SUMMARY_MAX_WORDS: usize = 50;

/// Maximum tokens for brief context.
pub const BRIEF_MAX_TOKENS: usize = 200;

/// Formats selected candidates into context strings.
pub struct ContextFormatter;

impl ContextFormatter {
    /// Format full context for SessionStart hook.
    /// Produces markdown with sections per category.
    pub fn format_full_context(
        candidates: &[InjectionCandidate],
        alerts: &[DivergenceAlert],
    ) -> String {
        if candidates.is_empty() && alerts.is_empty() {
            return String::new();
        }

        let mut output = String::from("## Relevant Context\n\n");

        // Group candidates by category
        let mut cluster_matches: Vec<_> = candidates
            .iter()
            .filter(|c| c.category == InjectionCategory::HighRelevanceCluster)
            .collect();

        let mut single_matches: Vec<_> = candidates
            .iter()
            .filter(|c| c.category == InjectionCategory::SingleSpaceMatch)
            .collect();

        let session_summaries: Vec<_> = candidates
            .iter()
            .filter(|c| c.category == InjectionCategory::RecentSession)
            .collect();

        // Recent Related Work (high-relevance clusters)
        if !cluster_matches.is_empty() {
            output.push_str("### Recent Related Work\n");
            for candidate in cluster_matches {
                let time_ago = Self::format_time_ago(candidate.created_at);
                let summary = Self::summarize_memory(&candidate.content, SUMMARY_MAX_WORDS);
                output.push_str(&format!("- **{}**: {}\n", time_ago, summary));
            }
            output.push('\n');
        }

        // Potentially Related (single-space matches)
        if !single_matches.is_empty() {
            output.push_str("### Potentially Related\n");
            for candidate in single_matches {
                let time_ago = Self::format_time_ago(candidate.created_at);
                let summary = Self::summarize_memory(&candidate.content, SUMMARY_MAX_WORDS);
                output.push_str(&format!("- {} ({})\n", summary, time_ago));
            }
            output.push('\n');
        }

        // Activity Shift Detected (divergence alerts)
        if !alerts.is_empty() {
            output.push_str("### Note: Activity Shift Detected\n");
            for alert in alerts {
                output.push_str(&Self::format_divergence_alert(alert));
                output.push('\n');
            }
            output.push('\n');
        }

        // Last Session Summary
        if !session_summaries.is_empty() {
            output.push_str("### Previous Session\n");
            for candidate in session_summaries {
                let summary = Self::summarize_memory(&candidate.content, SUMMARY_MAX_WORDS * 2);
                output.push_str(&summary);
                output.push('\n');
            }
        }

        output.trim_end().to_string()
    }

    /// Format brief context for PreToolUse hook.
    /// Produces compact single-paragraph output.
    pub fn format_brief_context(candidates: &[InjectionCandidate]) -> String {
        if candidates.is_empty() {
            return String::new();
        }

        let mut summaries: Vec<String> = Vec::new();
        let mut token_estimate = 10; // "Related: " prefix

        for candidate in candidates.iter().take(5) {
            let summary = Self::summarize_memory(&candidate.content, 20);
            let summary_tokens = summary.split_whitespace().count() * 13 / 10;

            if token_estimate + summary_tokens > BRIEF_MAX_TOKENS {
                break;
            }

            summaries.push(summary);
            token_estimate += summary_tokens + 2; // +2 for ", "
        }

        if summaries.is_empty() {
            return String::new();
        }

        format!("Related: {}", summaries.join(", "))
    }

    /// Format a divergence alert message.
    pub fn format_divergence_alert(alert: &DivergenceAlert) -> String {
        format!(
            "Your current query has low similarity to recent work in {}. Recent context: \"{}\" (similarity: {:.2})",
            alert.space,
            Self::summarize_memory(&alert.recent_context, 20),
            alert.similarity
        )
    }

    /// Summarize memory content to max_words.
    /// Truncates at sentence boundary if possible, adds "..." if truncated.
    pub fn summarize_memory(content: &str, max_words: usize) -> String {
        let words: Vec<&str> = content.split_whitespace().collect();

        if words.len() <= max_words {
            return content.to_string();
        }

        // Take first max_words words
        let truncated: String = words[..max_words].join(" ");

        // Try to truncate at sentence boundary
        if let Some(period_idx) = truncated.rfind(". ") {
            return truncated[..=period_idx].to_string();
        }
        if let Some(period_idx) = truncated.rfind('.') {
            if period_idx > truncated.len() / 2 {
                return truncated[..=period_idx].to_string();
            }
        }

        // No good sentence boundary, truncate with ellipsis
        format!("{}...", truncated)
    }

    /// Format time ago in human-readable form.
    fn format_time_ago(created_at: DateTime<Utc>) -> String {
        let now = Utc::now();
        let duration = now.signed_duration_since(created_at);

        if duration.num_minutes() < 60 {
            format!("{} minutes ago", duration.num_minutes())
        } else if duration.num_hours() < 24 {
            let hours = duration.num_hours();
            if hours == 1 {
                "1 hour ago".to_string()
            } else {
                format!("{} hours ago", hours)
            }
        } else if duration.num_days() < 7 {
            let days = duration.num_days();
            if days == 1 {
                "Yesterday".to_string()
            } else {
                format!("{} days ago", days)
            }
        } else if duration.num_weeks() < 4 {
            let weeks = duration.num_weeks();
            if weeks == 1 {
                "1 week ago".to_string()
            } else {
                format!("{} weeks ago", weeks)
            }
        } else {
            format!("{} days ago", duration.num_days())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use chrono::Duration;

    fn make_candidate(
        content: &str,
        category: InjectionCategory,
        hours_ago: i64,
    ) -> InjectionCandidate {
        InjectionCandidate::new(
            Uuid::new_v4(),
            content.to_string(),
            0.8,
            vec![],
            category,
            Utc::now() - Duration::hours(hours_ago),
        )
    }

    #[test]
    fn test_format_full_context_structure() {
        let candidates = vec![
            make_candidate(
                "Implemented HDBSCAN clustering",
                InjectionCategory::HighRelevanceCluster,
                2,
            ),
            make_candidate(
                "Rust async patterns",
                InjectionCategory::SingleSpaceMatch,
                72,
            ),
        ];

        let output = ContextFormatter::format_full_context(&candidates, &[]);

        assert!(output.contains("## Relevant Context"));
        assert!(output.contains("### Recent Related Work"));
        assert!(output.contains("### Potentially Related"));
        assert!(output.contains("HDBSCAN"));
        assert!(output.contains("Rust async"));
    }

    #[test]
    fn test_format_full_context_empty() {
        let output = ContextFormatter::format_full_context(&[], &[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_format_brief_context() {
        let candidates = vec![
            make_candidate(
                "HDBSCAN clustering implementation",
                InjectionCategory::HighRelevanceCluster,
                2,
            ),
            make_candidate(
                "BIRCH tree incremental clustering",
                InjectionCategory::HighRelevanceCluster,
                5,
            ),
        ];

        let output = ContextFormatter::format_brief_context(&candidates);

        assert!(output.starts_with("Related:"));
        assert!(output.contains("HDBSCAN"));
        assert!(output.contains("BIRCH"));
    }

    #[test]
    fn test_format_brief_context_empty() {
        let output = ContextFormatter::format_brief_context(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_summarize_memory_short() {
        let content = "This is a short memory.";
        let summary = ContextFormatter::summarize_memory(content, 50);
        assert_eq!(summary, content);
    }

    #[test]
    fn test_summarize_memory_truncate() {
        let content = "This is a longer memory that needs to be truncated because it has too many words for the maximum limit we set.";
        let summary = ContextFormatter::summarize_memory(content, 10);

        assert!(summary.len() < content.len());
        assert!(summary.ends_with("...") || summary.ends_with('.'));
    }

    #[test]
    fn test_summarize_memory_sentence_boundary() {
        let content = "First sentence here. Second sentence that goes on for a while with many words.";
        let summary = ContextFormatter::summarize_memory(content, 8);

        // Should truncate at first sentence
        assert!(summary.contains("First sentence"));
    }

    #[test]
    fn test_format_time_ago_hours() {
        let created = Utc::now() - Duration::hours(3);
        let formatted = ContextFormatter::format_time_ago(created);
        assert!(formatted.contains("3 hours ago"));
    }

    #[test]
    fn test_format_time_ago_yesterday() {
        let created = Utc::now() - Duration::hours(30);
        let formatted = ContextFormatter::format_time_ago(created);
        assert!(formatted.contains("Yesterday"));
    }

    #[test]
    fn test_format_time_ago_days() {
        let created = Utc::now() - Duration::days(4);
        let formatted = ContextFormatter::format_time_ago(created);
        assert!(formatted.contains("4 days ago"));
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/injection/formatter.rs">
    ContextFormatter struct with all formatting methods
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/injection/mod.rs">
    Add pub mod formatter; pub use formatter::*;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-core compiles without errors</criterion>
  <criterion type="test">cargo test injection::formatter::tests -- all 10 tests pass</criterion>
  <criterion type="output">Full context includes proper markdown structure</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-core</command>
  <command>cargo test injection::formatter::tests --package context-graph-core</command>
</test_commands>
</task_spec>
```
