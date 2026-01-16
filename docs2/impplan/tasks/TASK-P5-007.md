# Task: TASK-P5-007 - InjectionPipeline

```xml
<task_spec id="TASK-P5-007" version="1.0">
<metadata>
  <title>InjectionPipeline</title>
  <phase>5</phase>
  <sequence>42</sequence>
  <layer>logic</layer>
  <estimated_loc>250</estimated_loc>
  <dependencies>
    <dependency task="TASK-P5-001">InjectionCandidate type</dependency>
    <dependency task="TASK-P5-002">TokenBudget type</dependency>
    <dependency task="TASK-P5-003">InjectionResult type</dependency>
    <dependency task="TASK-P5-004">PriorityRanker</dependency>
    <dependency task="TASK-P5-005">TokenBudgetManager</dependency>
    <dependency task="TASK-P5-006">ContextFormatter</dependency>
    <dependency task="TASK-P3-005">DivergenceDetector</dependency>
    <dependency task="TASK-P3-006">SimilarityRetriever</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">InjectionPipeline</artifact>
    <artifact type="enum">InjectionError</artifact>
  </produces>
</metadata>

<context>
  <background>
    InjectionPipeline is the main orchestration component that ties together all
    injection subsystems. It coordinates divergence detection, memory retrieval,
    priority ranking, budget selection, and formatting to produce the final
    context injection result.
  </background>
  <business_value>
    Single entry point for context injection that handles all complexity internally,
    providing a simple interface for the CLI hooks to use.
  </business_value>
  <technical_context>
    Called from SessionStart hook with full budget and from PreToolUse hook with
    brief budget. Async to support database queries. Returns InjectionResult or
    InjectionError.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">All TASK-P5-001 through TASK-P5-006 types and logic</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/similarity/detector.rs with DivergenceDetector</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/similarity/retriever.rs with SimilarityRetriever</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>InjectionPipeline struct</item>
    <item>generate_context() for SessionStart</item>
    <item>generate_brief_context() for PreToolUse</item>
    <item>InjectionError enum</item>
    <item>Integration with SimilarityRetriever and DivergenceDetector</item>
    <item>Unit and integration tests</item>
  </includes>
  <excludes>
    <item>CLI hook integration (TASK-P6-*)</item>
    <item>Session context provider (separate task if needed)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>generate_context() orchestrates full pipeline correctly</description>
    <verification>Integration test with mock retriever verifies flow</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>generate_brief_context() produces compact output</description>
    <verification>Unit test verifies brief format and budget</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>InjectionError covers all failure modes</description>
    <verification>Error variants match spec</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Empty result returned when no relevant context</description>
    <verification>Test with empty retrieval returns InjectionResult::empty()</verification>
  </criterion>

  <signatures>
    <signature name="InjectionPipeline">
      <code>
pub struct InjectionPipeline {
    retriever: Arc&lt;SimilarityRetriever&gt;,
    detector: Arc&lt;DivergenceDetector&gt;,
}
      </code>
    </signature>
    <signature name="generate_context">
      <code>
impl InjectionPipeline {
    pub async fn generate_context(
        &amp;self,
        query: &amp;TeleologicalArray,
        session_id: &amp;str,
        budget: &amp;TokenBudget,
    ) -> Result&lt;InjectionResult, InjectionError&gt;
}
      </code>
    </signature>
    <signature name="generate_brief_context">
      <code>
impl InjectionPipeline {
    pub async fn generate_brief_context(
        &amp;self,
        query: &amp;TeleologicalArray,
        budget: u32,
    ) -> Result&lt;String, InjectionError&gt;
}
      </code>
    </signature>
    <signature name="InjectionError">
      <code>
#[derive(Debug, Error)]
pub enum InjectionError {
    #[error("Retrieval error: {source}")]
    RetrievalError { source: RetrievalError },

    #[error("Formatting error: {message}")]
    FormattingError { message: String },

    #[error("Budget exceeded: requested {requested}, available {available}")]
    BudgetExceeded { requested: u32, available: u32 },

    #[error("No context found")]
    NoContext,
}
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="behavior">Empty result is normal, not an error (NoContext only for explicit failures)</constraint>
    <constraint type="performance">generate_context() completes in &lt;100ms</constraint>
    <constraint type="performance">generate_brief_context() completes in &lt;50ms</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-core/src/injection/pipeline.rs

use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

use super::{
    budget::{TokenBudget, TokenBudgetManager, estimate_tokens, BRIEF_BUDGET},
    candidate::{InjectionCandidate, InjectionCategory},
    formatter::ContextFormatter,
    priority::PriorityRanker,
    result::InjectionResult,
};
use crate::embedding::TeleologicalArray;
use crate::similarity::{
    DivergenceDetector, DivergenceAlert,
    SimilarityRetriever, RetrievalError, SimilarityMatch,
};

/// Errors that can occur during context injection.
#[derive(Debug, Error)]
pub enum InjectionError {
    #[error("Retrieval error: {source}")]
    RetrievalError {
        #[from]
        source: RetrievalError,
    },

    #[error("Formatting error: {message}")]
    FormattingError { message: String },

    #[error("Budget exceeded: requested {requested}, available {available}")]
    BudgetExceeded { requested: u32, available: u32 },

    #[error("No context available")]
    NoContext,
}

/// Main orchestration component for context injection.
/// Coordinates retrieval, ranking, selection, and formatting.
pub struct InjectionPipeline {
    retriever: Arc<SimilarityRetriever>,
    detector: Arc<DivergenceDetector>,
}

impl InjectionPipeline {
    /// Create new pipeline with retriever and detector.
    pub fn new(
        retriever: Arc<SimilarityRetriever>,
        detector: Arc<DivergenceDetector>,
    ) -> Self {
        Self { retriever, detector }
    }

    /// Generate full context for SessionStart hook.
    ///
    /// Pipeline steps:
    /// 1. Detect divergence against recent memories
    /// 2. Retrieve similar memories
    /// 3. Build InjectionCandidates for each
    /// 4. Compute priority = relevance × recency × diversity
    /// 5. Sort by priority descending
    /// 6. Select candidates within budget
    /// 7. Format selected content
    /// 8. Return InjectionResult
    pub async fn generate_context(
        &self,
        query: &TeleologicalArray,
        session_id: &str,
        budget: &TokenBudget,
    ) -> Result<InjectionResult, InjectionError> {
        // Step 1: Detect divergence
        let alerts = self.detector.detect(query, session_id).await?;

        // Step 2: Retrieve similar memories
        let matches = self.retriever.retrieve_similar(query, 20).await?;

        if matches.is_empty() && alerts.is_empty() {
            return Ok(InjectionResult::empty());
        }

        // Step 3: Build candidates
        let mut candidates = self.build_candidates(&matches, &alerts);

        // Step 4 & 5: Compute priority and sort
        PriorityRanker::rank_candidates(&mut candidates);

        // Step 6: Select within budget
        let selected = TokenBudgetManager::select_within_budget(&candidates, budget);

        if selected.is_empty() && alerts.is_empty() {
            return Ok(InjectionResult::empty());
        }

        // Step 7: Format
        let formatted_context = ContextFormatter::format_full_context(&selected, &alerts);

        // Step 8: Build result
        let included_memories: Vec<Uuid> = selected.iter().map(|c| c.memory_id).collect();
        let tokens_used = estimate_tokens(&formatted_context);
        let categories_included: Vec<InjectionCategory> = selected
            .iter()
            .map(|c| c.category)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Ok(InjectionResult::new(
            formatted_context,
            included_memories,
            alerts,
            tokens_used,
            categories_included,
        ))
    }

    /// Generate brief context for PreToolUse hook.
    /// Simplified version focusing on top-3 similar memories.
    pub async fn generate_brief_context(
        &self,
        query: &TeleologicalArray,
        budget: u32,
    ) -> Result<String, InjectionError> {
        let effective_budget = budget.min(BRIEF_BUDGET);

        // Retrieve fewer memories for brief context
        let matches = self.retriever.retrieve_similar(query, 5).await?;

        if matches.is_empty() {
            return Ok(String::new());
        }

        // Build candidates (skip divergence for brief)
        let candidates: Vec<InjectionCandidate> = matches
            .iter()
            .map(|m| InjectionCandidate::new(
                m.memory_id,
                m.content.clone(),
                m.similarity,
                m.matching_spaces.clone(),
                Self::categorize_match(m),
                m.created_at,
            ))
            .collect();

        // Take top candidates within budget
        let mut tokens = 0u32;
        let mut selected = Vec::new();
        for candidate in candidates.iter() {
            if tokens + candidate.token_count <= effective_budget {
                tokens += candidate.token_count;
                selected.push(candidate.clone());
            }
            if selected.len() >= 3 {
                break;
            }
        }

        Ok(ContextFormatter::format_brief_context(&selected))
    }

    /// Build injection candidates from similarity matches.
    fn build_candidates(
        &self,
        matches: &[SimilarityMatch],
        alerts: &[DivergenceAlert],
    ) -> Vec<InjectionCandidate> {
        let mut candidates = Vec::new();

        // Add divergence alert candidates
        for alert in alerts {
            candidates.push(InjectionCandidate::new(
                alert.memory_id,
                alert.recent_context.clone(),
                1.0 - alert.similarity, // Inverse: low similarity = high relevance for alert
                vec![alert.space.clone()],
                InjectionCategory::DivergenceAlert,
                alert.detected_at,
            ));
        }

        // Add similarity match candidates
        for m in matches {
            candidates.push(InjectionCandidate::new(
                m.memory_id,
                m.content.clone(),
                m.similarity,
                m.matching_spaces.clone(),
                Self::categorize_match(m),
                m.created_at,
            ));
        }

        candidates
    }

    /// Categorize a similarity match based on matching space count.
    fn categorize_match(m: &SimilarityMatch) -> InjectionCategory {
        if m.matching_spaces.len() >= 3 {
            InjectionCategory::HighRelevanceCluster
        } else {
            InjectionCategory::SingleSpaceMatch
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // Note: Full integration tests require mock implementations
    // of SimilarityRetriever and DivergenceDetector

    #[test]
    fn test_categorize_match_high_relevance() {
        use crate::embedding::Embedder;

        let m = SimilarityMatch {
            memory_id: Uuid::new_v4(),
            content: "test".to_string(),
            similarity: 0.8,
            matching_spaces: vec![Embedder::E1Semantic, Embedder::E5Code, Embedder::E7Temporal],
            created_at: Utc::now(),
        };

        assert_eq!(
            InjectionPipeline::categorize_match(&m),
            InjectionCategory::HighRelevanceCluster
        );
    }

    #[test]
    fn test_categorize_match_single_space() {
        use crate::embedding::Embedder;

        let m = SimilarityMatch {
            memory_id: Uuid::new_v4(),
            content: "test".to_string(),
            similarity: 0.8,
            matching_spaces: vec![Embedder::E1Semantic],
            created_at: Utc::now(),
        };

        assert_eq!(
            InjectionPipeline::categorize_match(&m),
            InjectionCategory::SingleSpaceMatch
        );
    }

    #[test]
    fn test_injection_error_display() {
        let err = InjectionError::BudgetExceeded {
            requested: 1500,
            available: 1200,
        };

        assert!(err.to_string().contains("1500"));
        assert!(err.to_string().contains("1200"));
    }

    #[test]
    fn test_injection_error_no_context() {
        let err = InjectionError::NoContext;
        assert!(err.to_string().contains("No context"));
    }
}

// crates/context-graph-core/src/injection/error.rs

use thiserror::Error;
use crate::similarity::RetrievalError;

/// Re-export InjectionError from pipeline module
pub use super::pipeline::InjectionError;
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/injection/pipeline.rs">
    InjectionPipeline struct and InjectionError enum
  </file>
  <file path="crates/context-graph-core/src/injection/error.rs">
    Re-exports and any additional error utilities
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/injection/mod.rs">
    Add pub mod pipeline; pub mod error; pub use pipeline::*;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-core compiles without errors</criterion>
  <criterion type="test">cargo test injection::pipeline::tests -- all tests pass</criterion>
  <criterion type="integration">Pipeline integrates with retriever and detector</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-core</command>
  <command>cargo test injection::pipeline --package context-graph-core</command>
  <command>cargo test injection --package context-graph-core</command>
</test_commands>
</task_spec>
```
