# TASK-P2-005: MultiArrayProvider

```xml
<task_spec id="TASK-P2-005" version="1.0">
<metadata>
  <title>MultiArrayProvider Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>18</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-01</requirement_ref>
    <requirement_ref>REQ-P2-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P2-001</task_ref>
    <task_ref>TASK-P2-002</task_ref>
    <task_ref>TASK-P2-003</task_ref>
    <task_ref>TASK-P2-004</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

<context>
Implements the MultiArrayProvider that orchestrates all 13 embedders to produce
a complete TeleologicalArray. Embeddings are computed in parallel using tokio,
with individual timeouts and fail-fast error handling.

This is the central component for embedding generation - all memory capture
flows through this provider.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE2-EMBEDDING-13SPACE.md#component_contracts</file>
  <file purpose="teleological">crates/context-graph-core/src/embedding/teleological.rs</file>
  <file purpose="validator">crates/context-graph-core/src/embedding/validator.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P2-001 complete (TeleologicalArray exists)</check>
  <check>TASK-P2-002 complete (vector types exist)</check>
  <check>TASK-P2-003 complete (EmbedderConfig registry exists)</check>
  <check>TASK-P2-004 complete (DimensionValidator exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create MultiArrayProvider struct
    - Define EmbedderImpl trait for individual embedders
    - Implement embed_all() with parallel execution
    - Implement embed_single() for individual embedders
    - Add timeouts per embedder and overall
    - Validate dimensions before returning
    - Create EmbedderError enum
    - Implement MockEmbedder for testing
  </in_scope>
  <out_of_scope>
    - Actual model implementations (stub/mock for now)
    - GPU/CUDA integration
    - Model caching and optimization
    - Quantization (TASK-P2-006)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embedding/provider.rs">
      #[async_trait]
      pub trait EmbedderImpl: Send + Sync {
          fn embedder_type(&amp;self) -> Embedder;
          async fn embed(&amp;self, content: &amp;str) -> Result&lt;EmbeddingResult, EmbedderError&gt;;
      }

      pub enum EmbeddingResult {
          Dense(DenseVector),
          Sparse(SparseVector),
          Binary(BinaryVector),
          LateInteract(Vec&lt;DenseVector&gt;),
      }

      pub struct MultiArrayProvider {
          embedders: [Arc&lt;dyn EmbedderImpl&gt;; 13],
          timeout_per_embedder: Duration,
          timeout_total: Duration,
      }

      impl MultiArrayProvider {
          pub fn new(embedders: [Arc&lt;dyn EmbedderImpl&gt;; 13]) -> Self;
          pub fn with_mocks() -> Self;
          pub async fn embed_all(&amp;self, content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt;;
          pub async fn embed_single(&amp;self, content: &amp;str, embedder: Embedder) -> Result&lt;EmbeddingResult, EmbedderError&gt;;
      }
    </signature>
    <signature file="crates/context-graph-core/src/embedding/error.rs">
      #[derive(Debug, Error)]
      pub enum EmbedderError {
          #[error("Model failed for {embedder:?}: {source}")]
          ModelFailed { embedder: Embedder, source: Box&lt;dyn std::error::Error + Send + Sync&gt; },
          #[error("Dimension mismatch for {embedder:?}: expected {expected}, got {actual}")]
          DimensionMismatch { embedder: Embedder, expected: usize, actual: usize },
          #[error("Timeout for {embedder:?} after {duration:?}")]
          Timeout { embedder: Embedder, duration: Duration },
          #[error("Out of memory for {embedder:?}")]
          OutOfMemory { embedder: Embedder },
          #[error("Validation failed: {0}")]
          ValidationFailed(#[from] ValidationError),
      }
    </signature>
  </signatures>

  <constraints>
    - All 13 embeddings computed in parallel via tokio::join!
    - Timeout per embedder: 500ms
    - Total timeout: 1000ms
    - ANY failure = overall failure (fail fast)
    - Dimensions validated before returning
    - Empty content produces zero vectors (not an error)
  </constraints>

  <verification>
    - embed_all returns TeleologicalArray with all 13 embeddings
    - Timeout error returned if embedder exceeds limit
    - Dimension validation catches mismatches
    - Parallel execution faster than sequential
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/embedding/error.rs

use thiserror::Error;
use std::time::Duration;
use super::Embedder;
use super::validator::ValidationError;

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("Model failed for {embedder:?}: {message}")]
    ModelFailed {
        embedder: Embedder,
        message: String,
    },
    #[error("Dimension mismatch for {embedder:?}: expected {expected}, got {actual}")]
    DimensionMismatch {
        embedder: Embedder,
        expected: usize,
        actual: usize,
    },
    #[error("Timeout for {embedder:?} after {duration:?}")]
    Timeout {
        embedder: Embedder,
        duration: Duration,
    },
    #[error("Out of memory for {embedder:?}")]
    OutOfMemory {
        embedder: Embedder,
    },
    #[error("Validation failed: {0}")]
    ValidationFailed(#[from] ValidationError),
}

---
File: crates/context-graph-core/src/embedding/provider.rs

use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use tokio::time::timeout;

use super::{Embedder, TeleologicalArray};
use super::vector::{DenseVector, SparseVector, BinaryVector};
use super::config::get_dimension;
use super::error::EmbedderError;
use super::validator::validate_teleological_array;

/// Result type for individual embedder outputs
pub enum EmbeddingResult {
    Dense(DenseVector),
    Sparse(SparseVector),
    Binary(BinaryVector),
    LateInteract(Vec&lt;DenseVector&gt;),
}

/// Trait for individual embedder implementations
#[async_trait]
pub trait EmbedderImpl: Send + Sync {
    fn embedder_type(&amp;self) -&gt; Embedder;
    async fn embed(&amp;self, content: &amp;str) -&gt; Result&lt;EmbeddingResult, EmbedderError&gt;;
}

/// Multi-array provider that orchestrates all 13 embedders
pub struct MultiArrayProvider {
    embedders: [Arc&lt;dyn EmbedderImpl&gt;; 13],
    timeout_per_embedder: Duration,
    timeout_total: Duration,
}

impl MultiArrayProvider {
    pub fn new(embedders: [Arc&lt;dyn EmbedderImpl&gt;; 13]) -> Self {
        Self {
            embedders,
            timeout_per_embedder: Duration::from_millis(500),
            timeout_total: Duration::from_millis(1000),
        }
    }

    /// Create provider with mock embedders for testing
    pub fn with_mocks() -> Self {
        let embedders: [Arc&lt;dyn EmbedderImpl&gt;; 13] = [
            Arc::new(MockEmbedder::new(Embedder::E1Semantic)),
            Arc::new(MockEmbedder::new(Embedder::E2TempRecent)),
            Arc::new(MockEmbedder::new(Embedder::E3TempPeriodic)),
            Arc::new(MockEmbedder::new(Embedder::E4TempPosition)),
            Arc::new(MockEmbedder::new(Embedder::E5Causal)),
            Arc::new(MockEmbedder::new(Embedder::E6Sparse)),
            Arc::new(MockEmbedder::new(Embedder::E7Code)),
            Arc::new(MockEmbedder::new(Embedder::E8Emotional)),
            Arc::new(MockEmbedder::new(Embedder::E9HDC)),
            Arc::new(MockEmbedder::new(Embedder::E10Multimodal)),
            Arc::new(MockEmbedder::new(Embedder::E11Entity)),
            Arc::new(MockEmbedder::new(Embedder::E12LateInteract)),
            Arc::new(MockEmbedder::new(Embedder::E13SPLADE)),
        ];
        Self::new(embedders)
    }

    /// Embed content using all 13 embedders in parallel
    pub async fn embed_all(&amp;self, content: &amp;str) -&gt; Result&lt;TeleologicalArray, EmbedderError&gt; {
        // Execute all embedders in parallel with overall timeout
        let result = timeout(self.timeout_total, self.embed_all_inner(content)).await;

        match result {
            Ok(inner_result) =&gt; inner_result,
            Err(_) =&gt; Err(EmbedderError::Timeout {
                embedder: Embedder::E1Semantic, // First to timeout
                duration: self.timeout_total,
            }),
        }
    }

    async fn embed_all_inner(&amp;self, content: &amp;str) -&gt; Result&lt;TeleologicalArray, EmbedderError&gt; {
        // Run all 13 embedders in parallel
        let (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13) = tokio::join!(
            self.embed_with_timeout(&amp;self.embedders[0], content),
            self.embed_with_timeout(&amp;self.embedders[1], content),
            self.embed_with_timeout(&amp;self.embedders[2], content),
            self.embed_with_timeout(&amp;self.embedders[3], content),
            self.embed_with_timeout(&amp;self.embedders[4], content),
            self.embed_with_timeout(&amp;self.embedders[5], content),
            self.embed_with_timeout(&amp;self.embedders[6], content),
            self.embed_with_timeout(&amp;self.embedders[7], content),
            self.embed_with_timeout(&amp;self.embedders[8], content),
            self.embed_with_timeout(&amp;self.embedders[9], content),
            self.embed_with_timeout(&amp;self.embedders[10], content),
            self.embed_with_timeout(&amp;self.embedders[11], content),
            self.embed_with_timeout(&amp;self.embedders[12], content),
        );

        // Extract results (fail fast on any error)
        let array = TeleologicalArray {
            e1_semantic: extract_dense(r1?)?,
            e2_temp_recent: extract_dense(r2?)?,
            e3_temp_periodic: extract_dense(r3?)?,
            e4_temp_position: extract_dense(r4?)?,
            e5_causal: extract_dense(r5?)?,
            e6_sparse: extract_sparse(r6?)?,
            e7_code: extract_dense(r7?)?,
            e8_emotional: extract_dense(r8?)?,
            e9_hdc: extract_binary(r9?)?,
            e10_multimodal: extract_dense(r10?)?,
            e11_entity: extract_dense(r11?)?,
            e12_late_interact: extract_late_interact(r12?)?,
            e13_splade: extract_sparse(r13?)?,
        };

        // Validate dimensions
        validate_teleological_array(&amp;array)?;

        Ok(array)
    }

    async fn embed_with_timeout(
        &amp;self,
        embedder: &amp;Arc&lt;dyn EmbedderImpl&gt;,
        content: &amp;str,
    ) -&gt; Result&lt;EmbeddingResult, EmbedderError&gt; {
        let embedder_type = embedder.embedder_type();

        match timeout(self.timeout_per_embedder, embedder.embed(content)).await {
            Ok(result) =&gt; result,
            Err(_) =&gt; Err(EmbedderError::Timeout {
                embedder: embedder_type,
                duration: self.timeout_per_embedder,
            }),
        }
    }

    /// Embed using a single embedder
    pub async fn embed_single(
        &amp;self,
        content: &amp;str,
        embedder: Embedder,
    ) -&gt; Result&lt;EmbeddingResult, EmbedderError&gt; {
        let idx = embedder.index();
        self.embed_with_timeout(&amp;self.embedders[idx], content).await
    }
}

// Helper functions for extracting typed results
fn extract_dense(result: EmbeddingResult) -&gt; Result&lt;DenseVector, EmbedderError&gt; {
    match result {
        EmbeddingResult::Dense(v) =&gt; Ok(v),
        _ =&gt; Err(EmbedderError::ModelFailed {
            embedder: Embedder::E1Semantic,
            message: "Expected dense vector".into(),
        }),
    }
}

fn extract_sparse(result: EmbeddingResult) -&gt; Result&lt;SparseVector, EmbedderError&gt; {
    match result {
        EmbeddingResult::Sparse(v) =&gt; Ok(v),
        _ =&gt; Err(EmbedderError::ModelFailed {
            embedder: Embedder::E6Sparse,
            message: "Expected sparse vector".into(),
        }),
    }
}

fn extract_binary(result: EmbeddingResult) -&gt; Result&lt;BinaryVector, EmbedderError&gt; {
    match result {
        EmbeddingResult::Binary(v) =&gt; Ok(v),
        _ =&gt; Err(EmbedderError::ModelFailed {
            embedder: Embedder::E9HDC,
            message: "Expected binary vector".into(),
        }),
    }
}

fn extract_late_interact(result: EmbeddingResult) -&gt; Result&lt;Vec&lt;DenseVector&gt;, EmbedderError&gt; {
    match result {
        EmbeddingResult::LateInteract(v) =&gt; Ok(v),
        _ =&gt; Err(EmbedderError::ModelFailed {
            embedder: Embedder::E12LateInteract,
            message: "Expected late interaction vectors".into(),
        }),
    }
}

// =============================================================================
// Mock Embedder for Testing
// =============================================================================

pub struct MockEmbedder {
    embedder: Embedder,
}

impl MockEmbedder {
    pub fn new(embedder: Embedder) -&gt; Self {
        Self { embedder }
    }
}

#[async_trait]
impl EmbedderImpl for MockEmbedder {
    fn embedder_type(&amp;self) -&gt; Embedder {
        self.embedder
    }

    async fn embed(&amp;self, content: &amp;str) -&gt; Result&lt;EmbeddingResult, EmbedderError&gt; {
        use super::config::{get_config, is_sparse};

        let config = get_config(self.embedder);

        // Generate deterministic pseudo-random values based on content hash
        let hash = content.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        let seed = (hash as f32) / (u32::MAX as f32);

        match self.embedder {
            Embedder::E6Sparse | Embedder::E13SPLADE =&gt; {
                // Sparse vector: generate random indices
                let nnz = (content.len() % 100).max(5);
                let indices: Vec&lt;u32&gt; = (0..nnz).map(|i| (i * 300) as u32).collect();
                let values: Vec&lt;f32&gt; = indices.iter().map(|i| (*i as f32 * seed) % 1.0).collect();
                Ok(EmbeddingResult::Sparse(
                    SparseVector::new(indices, values, config.dimension as u32)
                        .unwrap_or_else(|_| SparseVector::empty(config.dimension as u32))
                ))
            }
            Embedder::E9HDC =&gt; {
                // Binary vector
                let mut vec = BinaryVector::zeros(config.dimension);
                for i in 0..config.dimension {
                    vec.set_bit(i, ((hash + i as u32) % 2) == 0);
                }
                Ok(EmbeddingResult::Binary(vec))
            }
            Embedder::E12LateInteract =&gt; {
                // Late interaction: variable number of tokens
                let num_tokens = (content.split_whitespace().count()).min(512).max(1);
                let vecs: Vec&lt;DenseVector&gt; = (0..num_tokens)
                    .map(|i| {
                        let data: Vec&lt;f32&gt; = (0..config.dimension)
                            .map(|j| ((hash + i as u32 + j as u32) as f32 / u32::MAX as f32) - 0.5)
                            .collect();
                        DenseVector::new(data)
                    })
                    .collect();
                Ok(EmbeddingResult::LateInteract(vecs))
            }
            _ =&gt; {
                // Dense vector
                let data: Vec&lt;f32&gt; = (0..config.dimension)
                    .map(|i| ((hash + i as u32) as f32 / u32::MAX as f32) - 0.5)
                    .collect();
                Ok(EmbeddingResult::Dense(DenseVector::new(data)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embed_all_with_mocks() {
        let provider = MultiArrayProvider::with_mocks();
        let result = provider.embed_all("Test content for embedding.").await;
        assert!(result.is_ok());

        let array = result.unwrap();
        assert_eq!(array.e1_semantic.len(), 1024);
        assert_eq!(array.e7_code.len(), 1536);
        assert_eq!(array.e9_hdc.bit_len(), 1024);
    }

    #[tokio::test]
    async fn test_embed_single() {
        let provider = MultiArrayProvider::with_mocks();
        let result = provider.embed_single("Test", Embedder::E1Semantic).await;
        assert!(result.is_ok());

        if let EmbeddingResult::Dense(vec) = result.unwrap() {
            assert_eq!(vec.len(), 1024);
        } else {
            panic!("Expected dense vector");
        }
    }

    #[tokio::test]
    async fn test_empty_content() {
        let provider = MultiArrayProvider::with_mocks();
        let result = provider.embed_all("").await;
        // Empty content is valid - produces zero-ish vectors
        assert!(result.is_ok());
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/embedding/provider.rs">MultiArrayProvider implementation</file>
  <file path="crates/context-graph-core/src/embedding/error.rs">EmbedderError enum</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/embedding/mod.rs">Add pub mod provider, error and re-exports</file>
  <file path="crates/context-graph-core/Cargo.toml">Add async-trait, tokio dependencies</file>
</files_to_modify>

<validation_criteria>
  <criterion>embed_all produces valid TeleologicalArray</criterion>
  <criterion>All 13 embedders run in parallel</criterion>
  <criterion>Timeout errors propagate correctly</criterion>
  <criterion>Dimension validation catches mismatches</criterion>
  <criterion>MockEmbedder produces correct dimensions for all types</criterion>
  <criterion>Empty content handled without error</criterion>
</validation_criteria>

<test_commands>
  <command description="Run provider tests">cargo test --package context-graph-core provider</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create error.rs with EmbedderError enum
- [ ] Create provider.rs with EmbedderImpl trait
- [ ] Implement EmbeddingResult enum
- [ ] Implement MultiArrayProvider struct
- [ ] Implement embed_all with parallel execution
- [ ] Implement embed_single for individual embedders
- [ ] Add timeout handling per embedder and total
- [ ] Implement MockEmbedder for testing
- [ ] Write integration tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P2-006
