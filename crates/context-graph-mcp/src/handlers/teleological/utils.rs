//! Utility functions for Teleological handlers.
//!
//! Contains embedding extraction, alignment computation, and error types.
//!
//! # Constitution Compliance Note - AP-03 VIOLATION REMOVED
//!
//! The function `project_to_embedding_dim()` was REMOVED because it violated:
//! Constitution AP-03: "No dimension projection to fake compatibility"
//!
//! The 13 embedders produce vectors of different native dimensions:
//!
//! - E1 Semantic: 1024D
//! - E2-E4 Temporal: 512D each
//! - E5 Causal: 768D
//! - E6 Sparse: ~30K vocab
//! - E7 Code: 1536D
//! - E8 Graph: 384D
//! - E9 HDC: 1024D
//! - E10 Multimodal: 768D
//! - E11 Entity: 384D
//! - E12 Late Interaction: 128D/token
//! - E13 SPLADE: ~30K vocab (sparse)
//!
//! Per constitution, embeddings MUST be kept in native dimensions.
//! Use FusionEngine::fuse_from_alignments() which works with alignment scores
//! (one scalar per embedder) instead of requiring uniform-dimension embeddings.

use context_graph_core::teleological::types::NUM_EMBEDDERS;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Error types for embedding extraction.
#[derive(Debug)]
pub enum ExtractError {
    /// Empty token-level embeddings (violates AP-04)
    EmptyTokenLevel(usize),
    /// Missing embedder (violates ARCH-05)
    MissingEmbedder(usize),
}

impl ExtractError {
    pub fn to_error_string(&self) -> String {
        match self {
            Self::EmptyTokenLevel(i) => format!(
                "FAIL FAST [AP-04]: Embedder E{} returned empty TokenLevel. \
                 All 13 embedders must produce valid embeddings.",
                i + 1
            ),
            Self::MissingEmbedder(i) => format!(
                "FAIL FAST [ARCH-05]: Embedder E{} returned None. \
                 Constitution requires ALL 13 embedders to be present.",
                i + 1
            ),
        }
    }
}

// ============================================================================
// ALIGNMENT COMPUTATION
// ============================================================================

/// Compute alignment scores from embedding L2 norms (normalized to [0,1]).
pub fn compute_alignments_from_embeddings(embeddings: &[Vec<f32>]) -> [f32; NUM_EMBEDDERS] {
    let mut alignments = [0.0f32; NUM_EMBEDDERS];

    for (i, emb) in embeddings.iter().enumerate().take(NUM_EMBEDDERS) {
        // Compute L2 norm as a proxy for "information content"
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        alignments[i] = norm.min(1.0); // Clamp to [0, 1]
    }

    // Normalize to sum to 1.0
    let sum: f32 = alignments.iter().sum();
    if sum > 0.0 {
        for a in &mut alignments {
            *a /= sum;
        }
    } else {
        // Uniform distribution if all zero
        for a in &mut alignments {
            *a = 1.0 / NUM_EMBEDDERS as f32;
        }
    }

    alignments
}

// ============================================================================
// EMBEDDING EXTRACTION
// ============================================================================

/// Extract embeddings from fingerprint in native dimensions.
///
/// CONSTITUTION COMPLIANT:
/// - Per AP-03: "No dimension projection to fake compatibility"
/// - Per ARCH-05: "Missing embedders are a fatal error"
pub fn extract_embeddings_from_fingerprint(
    fingerprint: &context_graph_core::types::SemanticFingerprint,
) -> Result<Vec<Vec<f32>>, ExtractError> {
    let mut embeddings = Vec::with_capacity(NUM_EMBEDDERS);

    for i in 0..NUM_EMBEDDERS {
        let embedding = match fingerprint.get_embedding(i) {
            Some(context_graph_core::types::EmbeddingSlice::Dense(slice)) => slice.to_vec(),
            Some(context_graph_core::types::EmbeddingSlice::Sparse(sparse)) => {
                sparse.values.clone()
            }
            Some(context_graph_core::types::EmbeddingSlice::TokenLevel(tokens)) => {
                if tokens.is_empty() {
                    return Err(ExtractError::EmptyTokenLevel(i));
                }
                // Average token embeddings (keep native token dimension)
                let dim = tokens[0].len();
                let mut avg = vec![0.0f32; dim];
                for token in tokens {
                    for (j, &v) in token.iter().enumerate() {
                        avg[j] += v;
                    }
                }
                let n = tokens.len() as f32;
                for v in &mut avg {
                    *v /= n;
                }
                avg
            }
            None => {
                return Err(ExtractError::MissingEmbedder(i));
            }
        };
        embeddings.push(embedding);
    }

    Ok(embeddings)
}
