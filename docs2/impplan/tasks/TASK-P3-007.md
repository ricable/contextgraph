# TASK-P3-007: SimilarityRetriever

```xml
<task_spec id="TASK-P3-007" version="1.0">
<metadata>
  <title>SimilarityRetriever Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>26</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P3-005</task_ref>
    <task_ref>TASK-P3-006</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the SimilarityRetriever that retrieves memories similar to a query
from the MemoryStore. Combines MultiSpaceSimilarity for scoring with storage
access for memory retrieval.

This is the primary interface for memory retrieval in the injection pipeline.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md#component_contracts</file>
  <file purpose="multi_space">crates/context-graph-core/src/retrieval/multi_space.rs</file>
  <file purpose="memory_store">crates/context-graph-core/src/memory/store.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P3-005 complete (MultiSpaceSimilarity exists)</check>
  <check>TASK-P3-006 complete (DivergenceDetector exists)</check>
  <check>TASK-P1-005 complete (MemoryStore exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SimilarityRetriever struct
    - Implement retrieve_similar method
    - Implement retrieve_by_space method
    - Implement retrieve_with_divergence check
    - Filter by relevance threshold
    - Sort by relevance score
    - Limit results
    - Create RetrievalError enum
  </in_scope>
  <out_of_scope>
    - Index building (linear scan for now)
    - Caching layer
    - Approximate nearest neighbor
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/retriever.rs">
      pub struct SimilarityRetriever {
          store: Arc&lt;MemoryStore&gt;,
          similarity: MultiSpaceSimilarity,
          divergence_detector: DivergenceDetector,
      }

      impl SimilarityRetriever {
          pub fn new(store: Arc&lt;MemoryStore&gt;, similarity: MultiSpaceSimilarity) -> Self;
          pub async fn retrieve_similar(&amp;self, query: &amp;TeleologicalArray, limit: usize) -> Result&lt;Vec&lt;SimilarityResult&gt;, RetrievalError&gt;;
          pub async fn retrieve_by_space(&amp;self, query: &amp;TeleologicalArray, space: Embedder, limit: usize) -> Result&lt;Vec&lt;SimilarityResult&gt;, RetrievalError&gt;;
          pub async fn retrieve_with_divergence(&amp;self, query: &amp;TeleologicalArray, session_id: &amp;str, limit: usize) -> Result&lt;RetrievalWithDivergence, RetrievalError&gt;;
      }

      pub struct RetrievalWithDivergence {
          pub results: Vec&lt;SimilarityResult&gt;,
          pub divergence_report: DivergenceReport,
      }
    </signature>
    <signature file="crates/context-graph-core/src/retrieval/error.rs">
      #[derive(Debug, Error)]
      pub enum RetrievalError {
          #[error("Storage error: {0}")]
          StorageError(#[from] StorageError),
          #[error("Invalid query: {message}")]
          InvalidQuery { message: String },
          #[error("No memories in database")]
          NoMemories,
      }
    </signature>
  </signatures>

  <constraints>
    - Linear scan acceptable for &lt;1000 memories
    - Only return memories where ANY space above threshold
    - Sort by relevance_score descending
    - Divergence check uses recent memories only
  </constraints>

  <verification>
    - retrieve_similar returns sorted results
    - retrieve_by_space filters to single space
    - Divergence report included when requested
    - Empty database returns NoMemories error
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/retrieval/error.rs

use thiserror::Error;
use crate::memory::store::StorageError;

#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),
    #[error("Invalid query: {message}")]
    InvalidQuery { message: String },
    #[error("No memories in database")]
    NoMemories,
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
}

---
File: crates/context-graph-core/src/retrieval/retriever.rs

use std::sync::Arc;
use uuid::Uuid;
use chrono::Utc;

use crate::memory::{Memory, MemoryStore};
use crate::embedding::{Embedder, TeleologicalArray};
use super::similarity::SimilarityResult;
use super::multi_space::{MultiSpaceSimilarity, sort_by_relevance, filter_relevant};
use super::divergence::DivergenceReport;
use super::detector::{DivergenceDetector, RecentMemory};
use super::distance::compute_similarity_for_space;
use super::error::RetrievalError;
use super::config::{RECENT_LOOKBACK_SECS, MAX_RECENT_MEMORIES};

/// Combined retrieval result with divergence information
#[derive(Debug)]
pub struct RetrievalWithDivergence {
    pub results: Vec&lt;SimilarityResult&gt;,
    pub divergence_report: DivergenceReport,
}

/// Retrieves similar memories from storage
pub struct SimilarityRetriever {
    store: Arc&lt;MemoryStore&gt;,
    similarity: MultiSpaceSimilarity,
    divergence_detector: DivergenceDetector,
}

impl SimilarityRetriever {
    pub fn new(store: Arc&lt;MemoryStore&gt;, similarity: MultiSpaceSimilarity) -> Self {
        let divergence_detector = DivergenceDetector::new(similarity.clone());
        Self {
            store,
            similarity,
            divergence_detector,
        }
    }

    pub fn with_divergence_detector(
        store: Arc&lt;MemoryStore&gt;,
        similarity: MultiSpaceSimilarity,
        divergence_detector: DivergenceDetector,
    ) -> Self {
        Self {
            store,
            similarity,
            divergence_detector,
        }
    }

    /// Retrieve similar memories, sorted by relevance
    pub async fn retrieve_similar(
        &amp;self,
        query: &amp;TeleologicalArray,
        limit: usize,
    ) -> Result&lt;Vec&lt;SimilarityResult&gt;, RetrievalError&gt; {
        // Get all memories from store
        let memories = self.get_all_memories().await?;

        if memories.is_empty() {
            return Err(RetrievalError::NoMemories);
        }

        // Compute similarity for each memory
        let mut results: Vec&lt;SimilarityResult&gt; = memories
            .into_iter()
            .map(|memory| {
                self.similarity.compute_full_result(
                    memory.id,
                    query,
                    &amp;memory.teleological_array,
                )
            })
            .collect();

        // Filter to relevant only
        results = filter_relevant(&amp;self.similarity, results);

        // Sort by relevance (highest first)
        results = sort_by_relevance(results);

        // Take top limit
        results.truncate(limit);

        Ok(results)
    }

    /// Retrieve memories sorted by similarity in a specific space
    pub async fn retrieve_by_space(
        &amp;self,
        query: &amp;TeleologicalArray,
        space: Embedder,
        limit: usize,
    ) -> Result&lt;Vec&lt;SimilarityResult&gt;, RetrievalError&gt; {
        let memories = self.get_all_memories().await?;

        if memories.is_empty() {
            return Err(RetrievalError::NoMemories);
        }

        // Compute similarity and include space-specific score
        let mut results: Vec&lt;(f32, SimilarityResult)&gt; = memories
            .into_iter()
            .map(|memory| {
                let space_score = compute_similarity_for_space(
                    space,
                    query,
                    &amp;memory.teleological_array,
                );
                let result = self.similarity.compute_full_result(
                    memory.id,
                    query,
                    &amp;memory.teleological_array,
                );
                (space_score, result)
            })
            .collect();

        // Sort by space-specific score (highest first)
        results.sort_by(|a, b| {
            b.0.partial_cmp(&amp;a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top limit
        let results: Vec&lt;SimilarityResult&gt; = results
            .into_iter()
            .take(limit)
            .map(|(_, r)| r)
            .collect();

        Ok(results)
    }

    /// Retrieve similar memories and check for divergence
    pub async fn retrieve_with_divergence(
        &amp;self,
        query: &amp;TeleologicalArray,
        session_id: &amp;str,
        limit: usize,
    ) -> Result&lt;RetrievalWithDivergence, RetrievalError&gt; {
        // Get recent memories for divergence check
        let recent = self.get_recent_memories(session_id).await?;

        // Check for divergence
        let divergence_report = self.divergence_detector.detect_divergence(
            query,
            &amp;recent,
        );

        // Get similar memories
        let results = self.retrieve_similar(query, limit).await.unwrap_or_default();

        Ok(RetrievalWithDivergence {
            results,
            divergence_report,
        })
    }

    /// Get all memories from store (for linear scan)
    async fn get_all_memories(&amp;self) -> Result&lt;Vec&lt;Memory&gt;, RetrievalError&gt; {
        // Note: This is a simplified implementation
        // In production, use pagination or streaming
        let count = self.store.count().await?;

        if count == 0 {
            return Ok(Vec::new());
        }

        // Get memories by iterating (MemoryStore would need this method)
        // For now, assume we have access to all memories
        // This would need enhancement in MemoryStore

        // Placeholder: would need MemoryStore.get_all() or similar
        Ok(Vec::new())
    }

    /// Get recent memories for divergence check
    async fn get_recent_memories(
        &amp;self,
        session_id: &amp;str,
    ) -> Result&lt;Vec&lt;RecentMemory&gt;, RetrievalError&gt; {
        let memories = self.store.get_by_session(session_id).await?;

        let cutoff = Utc::now() - chrono::Duration::seconds(RECENT_LOOKBACK_SECS as i64);

        let recent: Vec&lt;RecentMemory&gt; = memories
            .into_iter()
            .filter(|m| m.created_at >= cutoff)
            .take(MAX_RECENT_MEMORIES)
            .map(|m| RecentMemory::new(
                m.id,
                m.content.clone(),
                m.teleological_array.clone(),
                m.created_at,
            ))
            .collect();

        Ok(recent)
    }
}

impl Clone for MultiSpaceSimilarity {
    fn clone(&amp;self) -> Self {
        // Implementation needed if not derived
        MultiSpaceSimilarity::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_retriever() -> SimilarityRetriever {
        let dir = tempdir().unwrap();
        let store = Arc::new(MemoryStore::new(dir.path()).unwrap());
        let similarity = MultiSpaceSimilarity::with_defaults();
        SimilarityRetriever::new(store, similarity)
    }

    #[tokio::test]
    async fn test_retrieve_empty_store() {
        let retriever = create_test_retriever().await;
        let query = TeleologicalArray::new();

        let result = retriever.retrieve_similar(&amp;query, 10).await;

        // Should return NoMemories error for empty store
        assert!(matches!(result, Err(RetrievalError::NoMemories)));
    }

    #[tokio::test]
    async fn test_retrieve_with_divergence_empty() {
        let retriever = create_test_retriever().await;
        let query = TeleologicalArray::new();

        let result = retriever.retrieve_with_divergence(&amp;query, "test-session", 10).await;

        assert!(result.is_ok());
        let rwd = result.unwrap();
        assert!(rwd.divergence_report.is_empty());
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/retriever.rs">SimilarityRetriever implementation</file>
  <file path="crates/context-graph-core/src/retrieval/error.rs">RetrievalError enum</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Add pub mod retriever, error and re-exports</file>
  <file path="crates/context-graph-core/src/memory/store.rs">May need get_all method</file>
</files_to_modify>

<validation_criteria>
  <criterion>retrieve_similar returns sorted results</criterion>
  <criterion>Only relevant memories returned (ANY space above threshold)</criterion>
  <criterion>retrieve_by_space sorts by single space score</criterion>
  <criterion>Divergence report included in combined result</criterion>
  <criterion>Empty database returns NoMemories error</criterion>
  <criterion>Results limited to specified count</criterion>
</validation_criteria>

<test_commands>
  <command description="Run retriever tests">cargo test --package context-graph-core retriever</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>

<notes>
  <note category="performance">
    Current implementation uses linear scan.
    For &gt;1000 memories, implement per-space HNSW indexes.
  </note>
  <note category="store_integration">
    MemoryStore may need get_all() or iterator method.
    Current implementation uses placeholder.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Create error.rs with RetrievalError enum
- [ ] Create retriever.rs with SimilarityRetriever struct
- [ ] Implement retrieve_similar method
- [ ] Implement retrieve_by_space method
- [ ] Implement retrieve_with_divergence method
- [ ] Add RetrievalWithDivergence result struct
- [ ] Integrate with MemoryStore
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Phase 3 complete!
