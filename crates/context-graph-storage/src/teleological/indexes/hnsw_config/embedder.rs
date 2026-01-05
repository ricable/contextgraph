//! Embedder index enum matching constitution.yaml embedder list.

use serde::{Deserialize, Serialize};

use super::constants::*;
use super::distance::DistanceMetric;

/// Embedder index enum matching constitution.yaml embedder list.
///
/// 15 variants total:
/// - E1-E13: Core embedders (13)
/// - E1Matryoshka128: E1 truncated to 128D for Stage 2 fast filtering
/// - PurposeVector: 13D teleological alignment vector
///
/// # Non-HNSW Embedders
/// - E6Sparse: Inverted index (legacy sparse slot)
/// - E12LateInteraction: ColBERT MaxSim (token-level)
/// - E13Splade: Inverted index with BM25 (Stage 1 recall)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbedderIndex {
    /// E1: 1024D semantic (e5-large-v2, Matryoshka-capable)
    E1Semantic,
    /// E1 truncated to 128D for Stage 2 fast filtering
    E1Matryoshka128,
    /// E2: 512D temporal recent (exponential decay)
    E2TemporalRecent,
    /// E3: 512D temporal periodic (Fourier)
    E3TemporalPeriodic,
    /// E4: 512D temporal positional (sinusoidal PE)
    E4TemporalPositional,
    /// E5: 768D causal (Longformer SCM, asymmetric similarity)
    E5Causal,
    /// E6: ~30K sparse (inverted index, NOT HNSW)
    E6Sparse,
    /// E7: 256D code (CodeT5p)
    E7Code,
    /// E8: 384D graph (MiniLM)
    E8Graph,
    /// E9: 10000D HDC (holographic)
    E9HDC,
    /// E10: 768D multimodal (CLIP)
    E10Multimodal,
    /// E11: 384D entity (MiniLM)
    E11Entity,
    /// E12: 128D per-token ColBERT (MaxSim, NOT HNSW)
    E12LateInteraction,
    /// E13: ~30K SPLADE sparse (inverted index, NOT HNSW)
    E13Splade,
    /// 13D teleological purpose vector
    PurposeVector,
}

impl EmbedderIndex {
    /// Map 0-12 index to embedder. Panics on out of bounds.
    ///
    /// # Panics
    ///
    /// Panics with "INDEX ERROR" if `idx >= 13`.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_storage::teleological::indexes::EmbedderIndex;
    ///
    /// let e1 = EmbedderIndex::from_index(0);
    /// assert_eq!(e1, EmbedderIndex::E1Semantic);
    /// ```
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::E1Semantic,
            1 => Self::E2TemporalRecent,
            2 => Self::E3TemporalPeriodic,
            3 => Self::E4TemporalPositional,
            4 => Self::E5Causal,
            5 => Self::E6Sparse,
            6 => Self::E7Code,
            7 => Self::E8Graph,
            8 => Self::E9HDC,
            9 => Self::E10Multimodal,
            10 => Self::E11Entity,
            11 => Self::E12LateInteraction,
            12 => Self::E13Splade,
            _ => panic!("INDEX ERROR: embedder index {} out of bounds (max 12)", idx),
        }
    }

    /// Get 0-12 index from embedder. Returns None for E1Matryoshka128, PurposeVector.
    ///
    /// These special embedders are not part of the core 13-embedder array.
    pub fn to_index(&self) -> Option<usize> {
        match self {
            Self::E1Semantic => Some(0),
            Self::E2TemporalRecent => Some(1),
            Self::E3TemporalPeriodic => Some(2),
            Self::E4TemporalPositional => Some(3),
            Self::E5Causal => Some(4),
            Self::E6Sparse => Some(5),
            Self::E7Code => Some(6),
            Self::E8Graph => Some(7),
            Self::E9HDC => Some(8),
            Self::E10Multimodal => Some(9),
            Self::E11Entity => Some(10),
            Self::E12LateInteraction => Some(11),
            Self::E13Splade => Some(12),
            Self::E1Matryoshka128 | Self::PurposeVector => None,
        }
    }

    /// Check if this embedder uses HNSW indexing.
    ///
    /// Returns false for:
    /// - E6Sparse (inverted index)
    /// - E12LateInteraction (MaxSim token-level)
    /// - E13Splade (inverted index with BM25)
    #[inline]
    pub fn uses_hnsw(&self) -> bool {
        !matches!(
            self,
            Self::E6Sparse | Self::E12LateInteraction | Self::E13Splade
        )
    }

    /// Check if this embedder uses inverted indexing.
    ///
    /// Returns true for E6Sparse and E13Splade only.
    #[inline]
    pub fn uses_inverted_index(&self) -> bool {
        matches!(self, Self::E6Sparse | Self::E13Splade)
    }

    /// Get all HNSW-capable embedder indexes.
    ///
    /// Returns 12 entries (excludes E6, E12, E13):
    /// - 10 dense embedders (E1-E5, E7-E11)
    /// - E1Matryoshka128 (Stage 2 fast filter)
    /// - PurposeVector (Stage 5 teleological)
    pub fn all_hnsw() -> Vec<Self> {
        vec![
            Self::E1Semantic,
            Self::E1Matryoshka128,
            Self::E2TemporalRecent,
            Self::E3TemporalPeriodic,
            Self::E4TemporalPositional,
            Self::E5Causal,
            Self::E7Code,
            Self::E8Graph,
            Self::E9HDC,
            Self::E10Multimodal,
            Self::E11Entity,
            Self::PurposeVector,
        ]
    }

    /// Get the embedding dimension for this embedder.
    ///
    /// Returns None for E6Sparse, E12LateInteraction, E13Splade (non-dense).
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Self::E1Semantic => Some(E1_DIM),
            Self::E1Matryoshka128 => Some(E1_MATRYOSHKA_DIM),
            Self::E2TemporalRecent => Some(E2_DIM),
            Self::E3TemporalPeriodic => Some(E3_DIM),
            Self::E4TemporalPositional => Some(E4_DIM),
            Self::E5Causal => Some(E5_DIM),
            Self::E6Sparse => None, // Inverted index
            Self::E7Code => Some(E7_DIM),
            Self::E8Graph => Some(E8_DIM),
            Self::E9HDC => Some(E9_DIM),
            Self::E10Multimodal => Some(E10_DIM),
            Self::E11Entity => Some(E11_DIM),
            Self::E12LateInteraction => None, // Token-level
            Self::E13Splade => None,          // Inverted index
            Self::PurposeVector => Some(PURPOSE_VECTOR_DIM),
        }
    }

    /// Get the recommended distance metric for this embedder.
    pub fn recommended_metric(&self) -> Option<DistanceMetric> {
        match self {
            Self::E5Causal => Some(DistanceMetric::AsymmetricCosine),
            Self::E6Sparse | Self::E13Splade => None, // Inverted index
            Self::E12LateInteraction => Some(DistanceMetric::MaxSim),
            _ => Some(DistanceMetric::Cosine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_index_all_valid() {
        for i in 0..13 {
            let idx = EmbedderIndex::from_index(i);
            assert_eq!(
                idx.to_index(),
                Some(i),
                "Index {} should map back to Some({})",
                i,
                i
            );
        }
    }

    #[test]
    #[should_panic(expected = "INDEX ERROR")]
    fn test_panic_on_index_13() {
        let _ = EmbedderIndex::from_index(13);
    }

    #[test]
    fn test_embedder_uses_hnsw() {
        assert!(EmbedderIndex::E1Semantic.uses_hnsw());
        assert!(EmbedderIndex::E1Matryoshka128.uses_hnsw());
        assert!(EmbedderIndex::PurposeVector.uses_hnsw());

        assert!(!EmbedderIndex::E6Sparse.uses_hnsw());
        assert!(!EmbedderIndex::E12LateInteraction.uses_hnsw());
        assert!(!EmbedderIndex::E13Splade.uses_hnsw());
    }

    #[test]
    fn test_embedder_uses_inverted_index() {
        assert!(EmbedderIndex::E6Sparse.uses_inverted_index());
        assert!(EmbedderIndex::E13Splade.uses_inverted_index());

        assert!(!EmbedderIndex::E1Semantic.uses_inverted_index());
        assert!(!EmbedderIndex::E12LateInteraction.uses_inverted_index());
    }

    #[test]
    fn test_all_hnsw_count_is_12() {
        let hnsw_embedders = EmbedderIndex::all_hnsw();
        assert_eq!(hnsw_embedders.len(), 12);
    }

    #[test]
    fn test_to_index_special_embedders() {
        assert_eq!(EmbedderIndex::E1Matryoshka128.to_index(), None);
        assert_eq!(EmbedderIndex::PurposeVector.to_index(), None);
    }

    #[test]
    fn test_max_index_is_12() {
        let idx = EmbedderIndex::from_index(12);
        assert_eq!(idx, EmbedderIndex::E13Splade);
    }
}
