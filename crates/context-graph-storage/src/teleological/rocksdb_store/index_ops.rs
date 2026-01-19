//! Index operations for RocksDbTeleologicalStore.
//!
//! Contains methods for adding/removing fingerprints to/from HNSW indexes
//! and computing embedder scores.

use tracing::debug;
use uuid::Uuid;

use context_graph_core::types::fingerprint::{SemanticFingerprint, TeleologicalFingerprint};

use crate::teleological::indexes::{EmbedderIndex, EmbedderIndexOps, IndexError};
use crate::teleological::search::compute_maxsim_direct;

use super::helpers::compute_cosine_similarity;
use super::store::RocksDbTeleologicalStore;

impl RocksDbTeleologicalStore {
    /// Add fingerprint to per-embedder HNSW indexes for O(log n) search.
    ///
    /// Inserts vectors into all HNSW-capable embedder indexes.
    /// E6, E12, E13 are skipped (they use different index types).
    ///
    /// # FAIL FAST
    ///
    /// - DimensionMismatch: panic with detailed error
    /// - InvalidVector (NaN/Inf): panic with location
    pub(crate) fn add_to_indexes(&self, fp: &TeleologicalFingerprint) -> Result<(), IndexError> {
        let id = fp.id;

        // Add to all HNSW-capable dense embedder indexes
        for embedder in EmbedderIndex::all_hnsw() {
            if let Some(index) = self.index_registry.get(embedder) {
                let vector = Self::get_embedder_vector(&fp.semantic, embedder);
                index.insert(id, vector)?;
            }
        }

        debug!(
            "Added fingerprint {} to {} indexes",
            id,
            self.index_registry.len()
        );
        Ok(())
    }

    /// Extract vector for specific embedder from SemanticFingerprint.
    ///
    /// Returns the appropriate vector slice for the given embedder index.
    ///
    /// # FAIL FAST
    ///
    /// Panics for embedders that don't use HNSW:
    /// - E6Sparse: Use inverted index
    /// - E12LateInteraction: Use MaxSim
    /// - E13Splade: Use inverted index
    pub(crate) fn get_embedder_vector(
        semantic: &SemanticFingerprint,
        embedder: EmbedderIndex,
    ) -> &[f32] {
        match embedder {
            EmbedderIndex::E1Semantic => &semantic.e1_semantic,
            EmbedderIndex::E1Matryoshka128 => {
                // Truncate E1 to 128D - return first 128 elements
                &semantic.e1_semantic[..128.min(semantic.e1_semantic.len())]
            }
            EmbedderIndex::E2TemporalRecent => &semantic.e2_temporal_recent,
            EmbedderIndex::E3TemporalPeriodic => &semantic.e3_temporal_periodic,
            EmbedderIndex::E4TemporalPositional => &semantic.e4_temporal_positional,
            EmbedderIndex::E5Causal => &semantic.e5_causal,
            EmbedderIndex::E6Sparse => {
                panic!("FAIL FAST: E6 is sparse - use inverted index, not HNSW")
            }
            EmbedderIndex::E7Code => &semantic.e7_code,
            EmbedderIndex::E8Graph => &semantic.e8_graph,
            EmbedderIndex::E9HDC => &semantic.e9_hdc,
            EmbedderIndex::E10Multimodal => &semantic.e10_multimodal,
            EmbedderIndex::E11Entity => &semantic.e11_entity,
            EmbedderIndex::E12LateInteraction => {
                panic!("FAIL FAST: E12 is late-interaction - use MaxSim, not HNSW")
            }
            EmbedderIndex::E13Splade => {
                panic!("FAIL FAST: E13 is sparse - use inverted index, not HNSW")
            }
        }
    }

    /// Remove fingerprint from all per-embedder indexes.
    ///
    /// Removes the ID from all 11 HNSW indexes.
    pub(crate) fn remove_from_indexes(&self, id: Uuid) -> Result<(), IndexError> {
        for (_embedder, index) in self.index_registry.iter() {
            // Remove returns bool (found or not), we ignore it
            let _ = index.remove(id)?;
        }
        debug!("Removed fingerprint {} from all indexes", id);
        Ok(())
    }

    /// Compute similarity scores for all 13 embedders.
    ///
    /// Uses cosine similarity for dense embedders (E1-E5, E7-E11).
    /// Uses SparseVector::cosine_similarity for sparse embedders (E6, E13).
    /// Uses MaxSim for late-interaction embedder (E12).
    ///
    /// # ARCH-02 Compliance
    /// All comparisons are apples-to-apples: E1<->E1, E6<->E6, etc.
    pub(crate) fn compute_embedder_scores(
        &self,
        query: &SemanticFingerprint,
        stored: &SemanticFingerprint,
    ) -> [f32; 13] {
        [
            // E1-E5: Dense embedders - cosine similarity
            compute_cosine_similarity(&query.e1_semantic, &stored.e1_semantic),
            compute_cosine_similarity(&query.e2_temporal_recent, &stored.e2_temporal_recent),
            compute_cosine_similarity(&query.e3_temporal_periodic, &stored.e3_temporal_periodic),
            compute_cosine_similarity(
                &query.e4_temporal_positional,
                &stored.e4_temporal_positional,
            ),
            compute_cosine_similarity(&query.e5_causal, &stored.e5_causal),
            // E6: Sparse embedder - sparse cosine similarity
            query.e6_sparse.cosine_similarity(&stored.e6_sparse),
            // E7-E11: Dense embedders - cosine similarity
            compute_cosine_similarity(&query.e7_code, &stored.e7_code),
            compute_cosine_similarity(&query.e8_graph, &stored.e8_graph),
            compute_cosine_similarity(&query.e9_hdc, &stored.e9_hdc),
            compute_cosine_similarity(&query.e10_multimodal, &stored.e10_multimodal),
            compute_cosine_similarity(&query.e11_entity, &stored.e11_entity),
            // E12: Late-interaction - MaxSim over token embeddings
            compute_maxsim_direct(&query.e12_late_interaction, &stored.e12_late_interaction),
            // E13: SPLADE sparse - sparse cosine similarity
            query.e13_splade.cosine_similarity(&stored.e13_splade),
        ]
    }
}
