//! Document Embedder - Integrates OCR output with 13-embedder pipeline
//!
//! This module provides document embedding capabilities that integrate
//! with the existing 13-embedder pipeline (TeleologicalArray).
//!
//! # Usage
//!
//! ```rust
//! use context_graph_embeddings::{ModelId, ProductionMultiArrayProvider, DocumentEmbedder};
//!
//! // Create embedder provider
//! let provider = ProductionMultiArrayProvider::new();
//! provider.initialize().await?;
//!
//! // Create document embedder
//! let doc_embedder = DocumentEmbedder::new(provider);
//!
//! // Embed document chunks
//! let chunks = vec![...]; // DocumentChunkInput from OCR
//! let embeddings = doc_embedder.embed_chunks(&chunks).await?;
//! ```

use context_graph_core::traits::MultiArrayEmbeddingProvider;
use context_graph_core::error::CoreError;
use crate::error::EmbeddingResult;
use crate::types::{MultiArrayEmbedding, ModelEmbedding, ModelId};
use crate::ProductionMultiArrayProvider;

/// Document embedder - wraps ProductionMultiArrayProvider for document chunks
pub struct DocumentEmbedder {
    provider: ProductionMultiArrayProvider,
}

impl DocumentEmbedder {
    /// Create a new DocumentEmbedder
    pub fn new(provider: ProductionMultiArrayProvider) -> Self {
        Self { provider }
    }

    /// Embed document chunks through all 13 embedders
    ///
    /// Takes OCR-generated chunks and produces embeddings for each
    /// through the full TeleologicalArray pipeline.
    pub async fn embed_chunks(
        &self,
        chunks: &[DocumentChunkInput],
    ) -> EmbeddingResult<Vec<DocumentEmbeddings>> {
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        // Get text content from chunks
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

        // Generate all 13 embeddings for each chunk
        // Note: embed_batch_all returns Vec<MultiArrayEmbeddingOutput> from core trait
        // Pass empty metadata slice - uses defaults
        let outputs = self.provider.embed_batch_all(&texts, &[]).await
            .map_err(|e: CoreError| crate::error::EmbeddingError::InternalError { message: e.to_string() })?;

        // Convert MultiArrayEmbeddingOutput to MultiArrayEmbedding
        let results: Vec<DocumentEmbeddings> = chunks
            .iter()
            .zip(outputs.into_iter())
            .map(|(chunk, output)| {
                // Extract fingerprint from output and convert to MultiArrayEmbedding
                let embeddings = convert_fingerprint_to_multi_array(&output.fingerprint);

                DocumentEmbeddings {
                    chunk_id: chunk.id.clone(),
                    document_id: chunk.document_id.clone(),
                    embeddings,
                    char_offset: chunk.char_offset,
                    char_end: chunk.char_end,
                }
            })
            .collect();

        Ok(results)
    }

    /// Embed a single document through all 13 embedders
    pub async fn embed_document(&self, document: &str) -> EmbeddingResult<MultiArrayEmbedding> {
        let output = self.provider.embed_all(document).await
            .map_err(|e: CoreError| crate::error::EmbeddingError::InternalError { message: e.to_string() })?;
        Ok(convert_fingerprint_to_multi_array(&output.fingerprint))
    }

    /// Check if the embedder is ready
    pub fn is_ready(&self) -> bool {
        self.provider.is_ready()
    }

    /// Get health status for all embedders
    pub fn health_status(&self) -> [bool; 13] {
        self.provider.health_status()
    }

    /// Get model IDs for all 13 embedders
    pub fn model_ids(&self) -> [&str; 13] {
        self.provider.model_ids()
    }
}

/// Input for document chunk embedding
#[derive(Debug, Clone)]
pub struct DocumentChunkInput {
    /// Chunk unique ID
    pub id: String,
    /// Parent document ID
    pub document_id: String,
    /// Chunk content
    pub content: String,
    /// Character offset in original document
    pub char_offset: usize,
    /// Character end position
    pub char_end: usize,
}

impl DocumentChunkInput {
    /// Create a new DocumentChunkInput
    pub fn new(
        id: String,
        document_id: String,
        content: String,
        char_offset: usize,
        char_end: usize,
    ) -> Self {
        Self {
            id,
            document_id,
            content,
            char_offset,
            char_end,
        }
    }
}

/// Output from document embedding
#[derive(Debug, Clone)]
pub struct DocumentEmbeddings {
    /// Source chunk ID
    pub chunk_id: String,
    /// Parent document ID
    pub document_id: String,
    /// All 13 embeddings for this chunk
    pub embeddings: MultiArrayEmbedding,
    /// Character offset
    pub char_offset: usize,
    /// Character end position
    pub char_end: usize,
}

impl DocumentEmbeddings {
    /// Get embedding by model ID
    pub fn get_embedding(&self, model_id: crate::types::ModelId) -> Option<&crate::types::ModelEmbedding> {
        self.embeddings.get(model_id)
    }

    /// Get the semantic embedding (E1)
    pub fn semantic(&self) -> Option<&crate::types::ModelEmbedding> {
        self.embeddings.get(crate::types::ModelId::Semantic)
    }

    /// Get the code embedding (E2)
    pub fn code(&self) -> Option<&crate::types::ModelEmbedding> {
        self.embeddings.get(crate::types::ModelId::Code)
    }

    /// Get the entity embedding (E11)
    pub fn entity(&self) -> Option<&crate::types::ModelEmbedding> {
        self.embeddings.get(crate::types::ModelId::Entity)
    }
}

// =============================================================================
// Type Conversion Helpers
// =============================================================================

/// Convert core's SemanticFingerprint to embeddings crate's MultiArrayEmbedding.
///
/// This bridges the gap between the core API (which uses SemanticFingerprint
/// inside MultiArrayEmbeddingOutput) and the embeddings crate's internal
/// MultiArrayEmbedding storage type.
fn convert_fingerprint_to_multi_array(
    fingerprint: &context_graph_core::types::fingerprint::SemanticFingerprint,
) -> MultiArrayEmbedding {
    let mut multi = MultiArrayEmbedding::new();

    // Map core Embedder enum to embeddings crate ModelId
    // E1: Semantic
    if !fingerprint.e1_semantic.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::Semantic,
            vector: fingerprint.e1_semantic.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E2: Temporal-Recent
    if !fingerprint.e2_temporal_recent.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::TemporalRecent,
            vector: fingerprint.e2_temporal_recent.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E3: Temporal-Periodic
    if !fingerprint.e3_temporal_periodic.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::TemporalPeriodic,
            vector: fingerprint.e3_temporal_periodic.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E4: Temporal-Positional
    if !fingerprint.e4_temporal_positional.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::TemporalPositional,
            vector: fingerprint.e4_temporal_positional.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E5: Causal (dual vectors - use cause vector as primary)
    if !fingerprint.e5_causal_as_cause.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::Causal,
            vector: fingerprint.e5_causal_as_cause.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E7: Code
    if !fingerprint.e7_code.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::Code,
            vector: fingerprint.e7_code.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E8: Graph (dual vectors - use source vector as primary)
    if !fingerprint.e8_graph_as_source.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::Graph,
            vector: fingerprint.e8_graph_as_source.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E9: HDC
    if !fingerprint.e9_hdc.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::Hdc,
            vector: fingerprint.e9_hdc.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E10: Contextual (dual vectors - use paraphrase as primary)
    if !fingerprint.e10_multimodal_paraphrase.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::Contextual,
            vector: fingerprint.e10_multimodal_paraphrase.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // E11: Entity
    if !fingerprint.e11_entity.is_empty() {
        multi.set(ModelEmbedding {
            model_id: ModelId::Entity,
            vector: fingerprint.e11_entity.clone(),
            latency_us: 0,
            attention_weights: None,
            is_projected: false,
        });
    }

    // Note: E6 (Sparse), E12 (LateInteraction), E13 (SPLADE) are not stored
    // in MultiArrayEmbedding as they use different representations (sparse/token-level).
    // These would need specialized handling if required.

    multi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_chunk_input_creation() {
        let input = DocumentChunkInput::new(
            "chunk-1".to_string(),
            "doc-123".to_string(),
            "Hello world".to_string(),
            0,
            11,
        );

        assert_eq!(input.id, "chunk-1");
        assert_eq!(input.document_id, "doc-123");
        assert_eq!(input.content, "Hello world");
    }
}
