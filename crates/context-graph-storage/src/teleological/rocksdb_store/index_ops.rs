//! Index operations for RocksDbTeleologicalStore.
//!
//! Contains methods for adding/removing fingerprints to/from HNSW indexes
//! and computing embedder scores.
//!
//! # ARCH-16 Compliance
//!
//! E7 Code embedder supports query-type-aware similarity computation:
//! - Code2Code: Sharpens similarity for structural matches
//! - Text2Code: Standard semantic similarity
//! - NonCode: Reduced E7 weight

use tracing::debug;
use uuid::Uuid;

use context_graph_core::causal::asymmetric::CausalDirection;
use context_graph_core::code::{CodeQueryType, compute_e7_similarity_with_query_type};
use context_graph_core::retrieval::distance::compute_similarity_for_space_with_direction;
use context_graph_core::teleological::Embedder;
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
    /// # ARCH-15, AP-77: E5 Asymmetric Indexes
    ///
    /// - E5CausalCause: Returns e5_causal_as_cause vector (for effect-seeking queries)
    /// - E5CausalEffect: Returns e5_causal_as_effect vector (for cause-seeking queries)
    /// - E5Causal: Returns active vector (legacy, for backward compatibility)
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
            // E5 legacy - uses active vector (whichever is populated)
            EmbedderIndex::E5Causal => semantic.e5_active_vector(),
            // E5 asymmetric indexes (ARCH-15, AP-77)
            // Cause index stores cause vectors - queried when seeking effects
            EmbedderIndex::E5CausalCause => semantic.get_e5_as_cause(),
            // Effect index stores effect vectors - queried when seeking causes
            EmbedderIndex::E5CausalEffect => semantic.get_e5_as_effect(),
            EmbedderIndex::E6Sparse => {
                panic!("FAIL FAST: E6 is sparse - use inverted index, not HNSW")
            }
            EmbedderIndex::E7Code => &semantic.e7_code,
            EmbedderIndex::E8Graph => semantic.e8_active_vector(),
            EmbedderIndex::E9HDC => &semantic.e9_hdc,
            // E10 legacy - uses active vector (whichever is populated)
            EmbedderIndex::E10Multimodal => semantic.e10_active_vector(),
            // E10 asymmetric indexes (ARCH-15, AP-77)
            // Intent index stores intent vectors - queried when seeking contexts
            EmbedderIndex::E10MultimodalIntent => semantic.get_e10_as_intent(),
            // Context index stores context vectors - queried when seeking intents
            EmbedderIndex::E10MultimodalContext => semantic.get_e10_as_context(),
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
    /// Removes the ID from all 13 HNSW indexes (including E5CausalCause and E5CausalEffect).
    pub(crate) fn remove_from_indexes(&self, id: Uuid) -> Result<(), IndexError> {
        for (_embedder, index) in self.index_registry.iter() {
            // Remove returns bool (found or not), we ignore it
            let _ = index.remove(id)?;
        }
        debug!("Removed fingerprint {} from all indexes", id);
        Ok(())
    }

    /// Compute similarity scores for all 13 embedders with E7 query-type awareness.
    ///
    /// Per ARCH-16: E7 Code embedder MUST detect query type and use appropriate
    /// similarity computation.
    ///
    /// # Arguments
    ///
    /// * `query` - Query semantic fingerprint
    /// * `stored` - Stored document semantic fingerprint
    /// * `code_query_type` - Optional code query type for E7 adjustment
    ///
    /// # Query Type Adjustments (E7 only)
    ///
    /// * `Code2Code`: Sharpens similarity curve (boosts high, penalizes low)
    /// * `Text2Code`: Standard semantic similarity (no adjustment)
    /// * `NonCode`: Reduces E7 similarity weight by 50%
    /// * `None`: No adjustment (backward compatible)
    ///
    /// # ARCH-02 Compliance
    /// All comparisons are apples-to-apples: E1<->E1, E6<->E6, etc.
    pub(crate) fn compute_embedder_scores_with_code_query_type(
        &self,
        query: &SemanticFingerprint,
        stored: &SemanticFingerprint,
        code_query_type: Option<CodeQueryType>,
    ) -> [f32; 13] {
        // E7 Code similarity with query-type awareness
        let e7_score = match code_query_type {
            Some(query_type) => {
                compute_e7_similarity_with_query_type(&query.e7_code, &stored.e7_code, query_type)
            }
            None => {
                // No query type - use standard cosine similarity
                compute_cosine_similarity(&query.e7_code, &stored.e7_code)
            }
        };

        [
            // E1-E5: Dense embedders - cosine similarity
            compute_cosine_similarity(&query.e1_semantic, &stored.e1_semantic),
            compute_cosine_similarity(&query.e2_temporal_recent, &stored.e2_temporal_recent),
            compute_cosine_similarity(&query.e3_temporal_periodic, &stored.e3_temporal_periodic),
            compute_cosine_similarity(
                &query.e4_temporal_positional,
                &stored.e4_temporal_positional,
            ),
            compute_cosine_similarity(query.e5_active_vector(), stored.e5_active_vector()),
            // E6: Sparse embedder - sparse cosine similarity
            query.e6_sparse.cosine_similarity(&stored.e6_sparse),
            // E7: Code embedder with query-type awareness (ARCH-16)
            e7_score,
            // E8-E11: Dense embedders - cosine similarity
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

    /// Compute similarity scores for all 13 embedders with direction-aware E5.
    ///
    /// Per ARCH-15 and AP-77: When causal direction is known, E5 uses asymmetric
    /// similarity computation with direction modifiers (cause→effect 1.2x, effect→cause 0.8x).
    ///
    /// # Arguments
    ///
    /// * `query` - Query semantic fingerprint
    /// * `stored` - Stored document semantic fingerprint
    /// * `code_query_type` - Optional code query type for E7 adjustment
    /// * `causal_direction` - Detected causal direction of the query
    ///
    /// # Direction-Aware E5 (ARCH-15, AP-77)
    ///
    /// * `Cause`: Query seeks causes (use asymmetric cause vs effect comparison)
    /// * `Effect`: Query seeks effects (use asymmetric effect vs cause comparison)
    /// * `Unknown`: Use symmetric E5 similarity (backward compatible)
    ///
    /// # ARCH-02 Compliance
    /// All comparisons are apples-to-apples: E1<->E1, E6<->E6, etc.
    pub(crate) fn compute_embedder_scores_with_direction(
        &self,
        query: &SemanticFingerprint,
        stored: &SemanticFingerprint,
        code_query_type: Option<CodeQueryType>,
        causal_direction: CausalDirection,
    ) -> [f32; 13] {
        // E7 Code similarity with query-type awareness
        let e7_score = match code_query_type {
            Some(query_type) => {
                compute_e7_similarity_with_query_type(&query.e7_code, &stored.e7_code, query_type)
            }
            None => {
                compute_cosine_similarity(&query.e7_code, &stored.e7_code)
            }
        };

        // E5 Causal with direction-aware asymmetric similarity (ARCH-15, AP-77)
        let e5_score = compute_similarity_for_space_with_direction(
            Embedder::Causal,
            query,
            stored,
            causal_direction,
        );

        [
            // E1-E4: Dense embedders - cosine similarity
            compute_cosine_similarity(&query.e1_semantic, &stored.e1_semantic),
            compute_cosine_similarity(&query.e2_temporal_recent, &stored.e2_temporal_recent),
            compute_cosine_similarity(&query.e3_temporal_periodic, &stored.e3_temporal_periodic),
            compute_cosine_similarity(
                &query.e4_temporal_positional,
                &stored.e4_temporal_positional,
            ),
            // E5: Causal with direction-aware asymmetric similarity (ARCH-15, AP-77)
            e5_score,
            // E6: Sparse embedder - sparse cosine similarity
            query.e6_sparse.cosine_similarity(&stored.e6_sparse),
            // E7: Code embedder with query-type awareness (ARCH-16)
            e7_score,
            // E8-E11: Dense embedders - cosine similarity
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
