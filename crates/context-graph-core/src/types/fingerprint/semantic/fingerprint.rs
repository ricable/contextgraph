//! SemanticFingerprint struct and implementation.

use serde::{Deserialize, Serialize};

use super::constants::{
    E10_DIM, E11_DIM, E12_TOKEN_DIM, E13_SPLADE_VOCAB, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM,
    E6_SPARSE_VOCAB, E7_DIM, E8_DIM, E9_DIM,
};
use super::slice::EmbeddingSlice;
use crate::types::fingerprint::SparseVector;

/// SemanticFingerprint: Stores all 13 embeddings without fusion.
///
/// # Philosophy
///
/// **NO FUSION.** Each embedding space preserved independently for:
/// - Per-space HNSW search
/// - Per-space Johari classification
/// - Per-space teleological alignment
/// - 100% information preservation
///
/// # Storage
///
/// Typical storage: ~46KB (vs 6KB fused = 67% info loss)
///
/// # Design Note
///
/// Uses `Vec<f32>` instead of fixed-size arrays to:
/// 1. Enable serde serialization for large embeddings (E9 has 10000 dims)
/// 2. Avoid stack overflow with large arrays
/// 3. Maintain flexibility for future dimension changes
///
/// Dimension validation is performed via `validate()` and construction methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFingerprint {
    /// E1: Semantic (e5-large-v2) - 1024D dense embedding.
    pub e1_semantic: Vec<f32>,

    /// E2: Temporal-Recent (exponential decay) - 512D dense embedding.
    pub e2_temporal_recent: Vec<f32>,

    /// E3: Temporal-Periodic (Fourier) - 512D dense embedding.
    pub e3_temporal_periodic: Vec<f32>,

    /// E4: Temporal-Positional (sinusoidal PE) - 512D dense embedding.
    pub e4_temporal_positional: Vec<f32>,

    /// E5: Causal (Longformer SCM) - 768D dense embedding.
    pub e5_causal: Vec<f32>,

    /// E6: Sparse Lexical (SPLADE) - sparse vector with ~1500 active of 30522 vocab.
    pub e6_sparse: SparseVector,

    /// E7: Code (CodeT5p) - 256D dense embedding.
    pub e7_code: Vec<f32>,

    /// E8: Graph (MiniLM for structure) - 384D dense embedding.
    pub e8_graph: Vec<f32>,

    /// E9: HDC (10K-bit hyperdimensional) - 10000D dense embedding.
    pub e9_hdc: Vec<f32>,

    /// E10: Multimodal (CLIP) - 768D dense embedding.
    pub e10_multimodal: Vec<f32>,

    /// E11: Entity (MiniLM for facts) - 384D dense embedding.
    pub e11_entity: Vec<f32>,

    /// E12: Late-Interaction (ColBERT) - 128D per token, variable token count.
    pub e12_late_interaction: Vec<Vec<f32>>,

    /// E13: SPLADE v3 sparse embedding for lexical-semantic hybrid search.
    pub e13_splade: SparseVector,
}

impl SemanticFingerprint {
    /// Create a zeroed fingerprint (all embeddings initialized to 0.0).
    pub fn zeroed() -> Self {
        Self {
            e1_semantic: vec![0.0; E1_DIM],
            e2_temporal_recent: vec![0.0; E2_DIM],
            e3_temporal_periodic: vec![0.0; E3_DIM],
            e4_temporal_positional: vec![0.0; E4_DIM],
            e5_causal: vec![0.0; E5_DIM],
            e6_sparse: SparseVector::empty(),
            e7_code: vec![0.0; E7_DIM],
            e8_graph: vec![0.0; E8_DIM],
            e9_hdc: vec![0.0; E9_DIM],
            e10_multimodal: vec![0.0; E10_DIM],
            e11_entity: vec![0.0; E11_DIM],
            e12_late_interaction: Vec::new(),
            e13_splade: SparseVector::empty(),
        }
    }

    /// Get embedding by index (0-12).
    pub fn get_embedding(&self, idx: usize) -> Option<EmbeddingSlice<'_>> {
        match idx {
            0 => Some(EmbeddingSlice::Dense(&self.e1_semantic)),
            1 => Some(EmbeddingSlice::Dense(&self.e2_temporal_recent)),
            2 => Some(EmbeddingSlice::Dense(&self.e3_temporal_periodic)),
            3 => Some(EmbeddingSlice::Dense(&self.e4_temporal_positional)),
            4 => Some(EmbeddingSlice::Dense(&self.e5_causal)),
            5 => Some(EmbeddingSlice::Sparse(&self.e6_sparse)),
            6 => Some(EmbeddingSlice::Dense(&self.e7_code)),
            7 => Some(EmbeddingSlice::Dense(&self.e8_graph)),
            8 => Some(EmbeddingSlice::Dense(&self.e9_hdc)),
            9 => Some(EmbeddingSlice::Dense(&self.e10_multimodal)),
            10 => Some(EmbeddingSlice::Dense(&self.e11_entity)),
            11 => Some(EmbeddingSlice::TokenLevel(&self.e12_late_interaction)),
            12 => Some(EmbeddingSlice::Sparse(&self.e13_splade)),
            _ => None,
        }
    }

    /// Compute total storage size in bytes (heap allocations only).
    pub fn storage_size(&self) -> usize {
        let dense_size = (self.e1_semantic.len()
            + self.e2_temporal_recent.len()
            + self.e3_temporal_periodic.len()
            + self.e4_temporal_positional.len()
            + self.e5_causal.len()
            + self.e7_code.len()
            + self.e8_graph.len()
            + self.e9_hdc.len()
            + self.e10_multimodal.len()
            + self.e11_entity.len())
            * std::mem::size_of::<f32>();

        let e6_sparse_size = self.e6_sparse.memory_size();
        let token_size: usize = self
            .e12_late_interaction
            .iter()
            .map(|t| t.len() * std::mem::size_of::<f32>())
            .sum();
        let e13_sparse_size = self.e13_splade.memory_size();

        dense_size + e6_sparse_size + token_size + e13_sparse_size
    }

    /// Get the number of tokens in E12 late-interaction embedding.
    #[inline]
    pub fn token_count(&self) -> usize {
        self.e12_late_interaction.len()
    }

    /// Get the number of non-zero entries in E13 SPLADE embedding.
    #[inline]
    pub fn e13_splade_nnz(&self) -> usize {
        self.e13_splade.nnz()
    }

    /// Get embedding name by index.
    pub fn embedding_name(idx: usize) -> Option<&'static str> {
        match idx {
            0 => Some("E1_Semantic"),
            1 => Some("E2_Temporal_Recent"),
            2 => Some("E3_Temporal_Periodic"),
            3 => Some("E4_Temporal_Positional"),
            4 => Some("E5_Causal"),
            5 => Some("E6_Sparse_Lexical"),
            6 => Some("E7_Code"),
            7 => Some("E8_Graph"),
            8 => Some("E9_HDC"),
            9 => Some("E10_Multimodal"),
            10 => Some("E11_Entity"),
            11 => Some("E12_Late_Interaction"),
            12 => Some("E13_SPLADE"),
            _ => None,
        }
    }

    /// Get embedding dimension by index.
    pub fn embedding_dim(idx: usize) -> Option<usize> {
        match idx {
            0 => Some(E1_DIM),
            1 => Some(E2_DIM),
            2 => Some(E3_DIM),
            3 => Some(E4_DIM),
            4 => Some(E5_DIM),
            5 => Some(E6_SPARSE_VOCAB),
            6 => Some(E7_DIM),
            7 => Some(E8_DIM),
            8 => Some(E9_DIM),
            9 => Some(E10_DIM),
            10 => Some(E11_DIM),
            11 => Some(E12_TOKEN_DIM),
            12 => Some(E13_SPLADE_VOCAB),
            _ => None,
        }
    }
}

impl Default for SemanticFingerprint {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl PartialEq for SemanticFingerprint {
    fn eq(&self, other: &Self) -> bool {
        self.e1_semantic == other.e1_semantic
            && self.e2_temporal_recent == other.e2_temporal_recent
            && self.e3_temporal_periodic == other.e3_temporal_periodic
            && self.e4_temporal_positional == other.e4_temporal_positional
            && self.e5_causal == other.e5_causal
            && self.e6_sparse == other.e6_sparse
            && self.e7_code == other.e7_code
            && self.e8_graph == other.e8_graph
            && self.e9_hdc == other.e9_hdc
            && self.e10_multimodal == other.e10_multimodal
            && self.e11_entity == other.e11_entity
            && self.e12_late_interaction == other.e12_late_interaction
            && self.e13_splade == other.e13_splade
    }
}
