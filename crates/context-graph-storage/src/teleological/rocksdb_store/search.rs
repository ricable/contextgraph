//! Search operations for TeleologicalMemoryStore trait.
//!
//! Contains semantic, purpose, text, and sparse search implementations.
//!
//! # Multi-Space Search Strategies (TASK-MULTISPACE)
//!
//! Three search strategies are supported:
//!
//! - **E1Only**: Original E1-only HNSW search (backward compatible)
//! - **MultiSpace**: Weighted fusion of semantic embedders
//! - **Pipeline**: Full 3-stage retrieval
//!
//! ## Key Research Insights
//!
//! Temporal embedders (E2-E4) measure TIME proximity, not TOPIC similarity.
//! They are excluded from similarity scoring and applied as post-retrieval boosts.
//!
//! References:
//! - [Cascading Retrieval](https://www.pinecone.io/blog/cascading-retrieval/)
//! - [Fusion Analysis](https://dl.acm.org/doi/10.1145/3596512)

use std::collections::{HashMap, HashSet};

use tracing::{debug, error, info, warn};
use uuid::Uuid;

use context_graph_core::error::{CoreError, CoreResult};
use context_graph_core::traits::{SearchStrategy, TeleologicalSearchOptions, TeleologicalSearchResult};
use context_graph_core::types::fingerprint::{SemanticFingerprint, SparseVector};

use crate::teleological::column_families::CF_E13_SPLADE_INVERTED;
use crate::teleological::indexes::{EmbedderIndex, EmbedderIndexOps};
use crate::teleological::schema::e13_splade_inverted_key;
use crate::teleological::serialization::{
    deserialize_memory_id_list, deserialize_teleological_fingerprint,
};

use super::store::RocksDbTeleologicalStore;
use super::types::TeleologicalStoreError;

// =============================================================================
// SEMANTIC EMBEDDER INDICES (for multi-space scoring)
// Per constitution: Temporal (E2-E4) have weight 0.0 in semantic search
// =============================================================================

/// Semantic embedder indices that contribute to similarity scoring.
/// E2-E4 (temporal) are EXCLUDED per AP-71 and research findings.
#[allow(dead_code)]
const SEMANTIC_EMBEDDER_INDICES: [usize; 7] = [
    0,  // E1 - Semantic
    4,  // E5 - Causal
    5,  // E6 - Sparse
    6,  // E7 - Code
    9,  // E10 - Multimodal
    10, // E11 - Entity
    11, // E12 - Late Interaction
];

/// Default weights for semantic search profile.
/// Sum of non-zero weights = 1.0
/// E2-E4 (temporal) = 0.0 per research
const DEFAULT_SEMANTIC_WEIGHTS: [f32; 13] = [
    0.35, // E1 - Semantic (primary)
    0.0,  // E2 - Temporal Recent (metadata only)
    0.0,  // E3 - Temporal Periodic (metadata only)
    0.0,  // E4 - Temporal Positional (metadata only)
    0.15, // E5 - Causal
    0.05, // E6 - Sparse (keyword backup)
    0.20, // E7 - Code
    0.05, // E8 - Graph (relational)
    0.0,  // E9 - HDC (noise-robust backup, not used in fusion)
    0.15, // E10 - Multimodal
    0.05, // E11 - Entity (relational)
    0.0,  // E12 - Late Interaction (used in Stage 3 rerank only)
    0.0,  // E13 - SPLADE (used in Stage 1 recall only)
];

// =============================================================================
// PIPELINE CONFIGURATION
// Per AP-76: Stage 1 (sparse recall) → Stage 2 (dense scoring) → Stage 3 (rerank)
// =============================================================================

/// Stage 1 recall multiplier (how many candidates to retrieve)
const STAGE1_RECALL_MULTIPLIER: usize = 10;

/// Stage 2 scoring keeps this many candidates for optional re-ranking
const STAGE2_CANDIDATE_MULTIPLIER: usize = 3;

// =============================================================================
// WEIGHTED FUSION FUNCTIONS
// =============================================================================

/// Compute weighted fusion of embedder scores.
///
/// Only uses embedders with non-zero weights (semantic embedders).
/// Temporal embedders (E2-E4) have weight 0.0 per AP-71.
///
/// # Arguments
///
/// * `scores` - All 13 embedder similarity scores
/// * `weights` - Weight profile (must have 13 elements)
///
/// # Returns
///
/// Weighted average of scores (0.0-1.0)
fn compute_semantic_fusion(scores: &[f32; 13], weights: &[f32; 13]) -> f32 {
    let mut weighted_sum = 0.0f32;
    let mut weight_total = 0.0f32;

    for (&score, &weight) in scores.iter().zip(weights.iter()) {
        // Skip zero-weight embedders (temporal)
        if weight > 0.0 {
            weighted_sum += score * weight;
            weight_total += weight;
        }
    }

    if weight_total > 0.0 {
        weighted_sum / weight_total
    } else {
        // Fallback to E1 if all weights are zero
        scores[0]
    }
}

impl RocksDbTeleologicalStore {
    /// Search by semantic fingerprint (internal async wrapper).
    ///
    /// Supports three strategies:
    /// - `E1Only`: Original E1-only HNSW search (backward compatible)
    /// - `MultiSpace`: Weighted fusion of semantic embedders
    /// - `Pipeline`: Full 3-stage retrieval
    pub(crate) async fn search_semantic_async(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        debug!(
            "Searching semantic with strategy={:?}, top_k={}, min_similarity={}",
            options.strategy, options.top_k, options.min_similarity
        );

        // Branch based on search strategy
        let mut results = match options.strategy {
            SearchStrategy::E1Only => {
                self.search_e1_only(query, &options).await?
            }
            SearchStrategy::MultiSpace => {
                self.search_multi_space(query, &options).await?
            }
            SearchStrategy::Pipeline => {
                // Full 3-stage pipeline per AP-76:
                // Stage 1 (Recall): E13 SPLADE + E1 HNSW
                // Stage 2 (Scoring): Weighted fusion of semantic embedders
                // Stage 3 (Re-rank): Optional E12 ColBERT MaxSim
                self.search_pipeline(query, &options).await?
            }
        };

        // Apply recency boost if configured (POST-retrieval per ARCH-14)
        if options.recency_boost > 0.0 {
            self.apply_recency_boost(&mut results, query, options.recency_boost);
        }

        debug!("Semantic search returned {} results", results.len());
        Ok(results)
    }

    /// Original E1-only search (backward compatible).
    ///
    /// Uses only E1 Semantic HNSW index for ranking.
    /// Fastest option, suitable for simple queries.
    async fn search_e1_only(
        &self,
        query: &SemanticFingerprint,
        options: &TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        // Search E1 Semantic as primary
        let entry_embedder = EmbedderIndex::E1Semantic;
        let entry_index = self.index_registry.get(entry_embedder).ok_or_else(|| {
            CoreError::IndexError(format!("Index {:?} not found", entry_embedder))
        })?;

        // Search E1 semantic space with 2x top_k to allow filtering
        let k = (options.top_k * 2).max(20);
        let candidates = entry_index
            .search(&query.e1_semantic, k, None)
            .map_err(|e| {
                error!("E1 search failed: {}", e);
                CoreError::IndexError(e.to_string())
            })?;

        // Fetch full fingerprints for candidates
        let mut results = Vec::with_capacity(candidates.len());

        for (id, distance) in candidates {
            // Convert distance to similarity (HNSW returns distance, not similarity)
            let similarity = 1.0 - distance.min(1.0);

            // Skip soft-deleted
            if !options.include_deleted && self.is_soft_deleted(&id) {
                continue;
            }

            // Apply min_similarity filter
            if similarity < options.min_similarity {
                continue;
            }

            // Fetch full fingerprint from RocksDB
            if let Some(data) = self.get_fingerprint_raw(id)? {
                let fp = deserialize_teleological_fingerprint(&data);

                // Compute all 13 embedder scores using helper
                let embedder_scores = self.compute_embedder_scores(query, &fp.semantic);

                results.push(TeleologicalSearchResult::new(fp, similarity, embedder_scores));
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        results.truncate(options.top_k);

        Ok(results)
    }

    /// Multi-space weighted fusion search.
    ///
    /// Uses weighted combination of semantic embedders (E1, E5, E7, E10).
    /// Temporal embedders (E2-E4) have weight 0.0 per AP-71.
    async fn search_multi_space(
        &self,
        query: &SemanticFingerprint,
        options: &TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        // Step 1: Get candidates from E1 HNSW (primary dense retrieval)
        let entry_embedder = EmbedderIndex::E1Semantic;
        let entry_index = self.index_registry.get(entry_embedder).ok_or_else(|| {
            CoreError::IndexError(format!("Index {:?} not found", entry_embedder))
        })?;

        // Use larger candidate pool for multi-space scoring
        let k = (options.top_k * 3).max(50);
        let candidates = entry_index
            .search(&query.e1_semantic, k, None)
            .map_err(|e| {
                error!("E1 search failed: {}", e);
                CoreError::IndexError(e.to_string())
            })?;

        // Step 2: Get candidate IDs from E5 Causal (if available)
        let mut candidate_ids: HashSet<Uuid> = candidates.iter().map(|(id, _)| *id).collect();

        if let Some(e5_index) = self.index_registry.get(EmbedderIndex::E5Causal) {
            if let Ok(e5_candidates) = e5_index.search(&query.e5_causal, k, None) {
                candidate_ids.extend(e5_candidates.iter().map(|(id, _)| *id));
            }
        }

        // Step 3: Get candidate IDs from E7 Code (if available)
        if let Some(e7_index) = self.index_registry.get(EmbedderIndex::E7Code) {
            if let Ok(e7_candidates) = e7_index.search(&query.e7_code, k, None) {
                candidate_ids.extend(e7_candidates.iter().map(|(id, _)| *id));
            }
        }

        debug!(
            "Multi-space search: {} unique candidates from E1+E5+E7",
            candidate_ids.len()
        );

        // Step 4: Compute weighted fusion scores for all candidates
        let weights = self.resolve_weights(options);
        let mut results = Vec::with_capacity(candidate_ids.len());

        for id in candidate_ids {
            // Skip soft-deleted
            if !options.include_deleted && self.is_soft_deleted(&id) {
                continue;
            }

            // Fetch full fingerprint from RocksDB
            if let Some(data) = self.get_fingerprint_raw(id)? {
                let fp = deserialize_teleological_fingerprint(&data);

                // Compute all 13 embedder scores
                let embedder_scores = self.compute_embedder_scores(query, &fp.semantic);

                // Compute weighted fusion (semantic embedders only)
                let fusion_score = compute_semantic_fusion(&embedder_scores, &weights);

                // Apply min_similarity filter
                if fusion_score < options.min_similarity {
                    continue;
                }

                results.push(TeleologicalSearchResult::new(fp, fusion_score, embedder_scores));
            }
        }

        // Sort by fusion score descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        results.truncate(options.top_k);

        debug!("Multi-space search returned {} results", results.len());
        Ok(results)
    }

    /// Full 3-stage pipeline search.
    ///
    /// Per AP-76: Stage 1 (sparse recall) → Stage 2 (dense scoring) → Stage 3 (optional rerank)
    ///
    /// # Stages
    ///
    /// 1. **Recall** (E13 SPLADE + E1 HNSW): Broad candidate generation
    /// 2. **Scoring** (Weighted fusion): Multi-space semantic scoring
    /// 3. **Re-rank** (E12 ColBERT MaxSim): Optional precise re-ranking
    ///
    /// # References
    ///
    /// - [Cascading Retrieval](https://www.pinecone.io/blog/cascading-retrieval/) - 48% improvement
    /// - [ColBERT Late Interaction](https://weaviate.io/blog/late-interaction-overview)
    async fn search_pipeline(
        &self,
        query: &SemanticFingerprint,
        options: &TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        let recall_k = options.top_k * STAGE1_RECALL_MULTIPLIER;
        let stage2_k = options.top_k * STAGE2_CANDIDATE_MULTIPLIER;

        info!(
            "Pipeline search: Stage1 recall_k={}, Stage2 candidates={}, final_k={}",
            recall_k, stage2_k, options.top_k
        );

        // =========================================================================
        // STAGE 1: FAST RECALL (E13 SPLADE inverted + E1 HNSW)
        // Per AP-75: E13 SPLADE for Stage 1 recall, NOT final ranking
        // =========================================================================
        let mut candidate_ids: HashSet<Uuid> = HashSet::new();

        // E13 SPLADE sparse recall (inverted index)
        if !query.e13_splade.is_empty() {
            match self.search_sparse_async(&query.e13_splade, recall_k).await {
                Ok(sparse_results) => {
                    let sparse_count = sparse_results.len();
                    candidate_ids.extend(sparse_results.into_iter().map(|(id, _)| id));
                    debug!("Stage 1: E13 SPLADE returned {} candidates", sparse_count);
                }
                Err(e) => {
                    warn!("Stage 1: E13 SPLADE search failed: {}, continuing with E1 only", e);
                }
            }
        } else {
            debug!("Stage 1: E13 SPLADE empty, skipping sparse recall");
        }

        // E1 Semantic HNSW recall (primary dense)
        let entry_embedder = EmbedderIndex::E1Semantic;
        if let Some(entry_index) = self.index_registry.get(entry_embedder) {
            match entry_index.search(&query.e1_semantic, recall_k, None) {
                Ok(e1_candidates) => {
                    let e1_count = e1_candidates.len();
                    candidate_ids.extend(e1_candidates.into_iter().map(|(id, _)| id));
                    debug!("Stage 1: E1 HNSW returned {} candidates", e1_count);
                }
                Err(e) => {
                    error!("Stage 1: E1 HNSW search failed: {}", e);
                    return Err(CoreError::IndexError(e.to_string()));
                }
            }
        } else {
            return Err(CoreError::IndexError("E1 Semantic index not found".into()));
        }

        // Also add candidates from E5 Causal for better recall on reasoning queries
        if let Some(e5_index) = self.index_registry.get(EmbedderIndex::E5Causal) {
            if let Ok(e5_candidates) = e5_index.search(&query.e5_causal, recall_k / 2, None) {
                let e5_count = e5_candidates.len();
                candidate_ids.extend(e5_candidates.into_iter().map(|(id, _)| id));
                debug!("Stage 1: E5 Causal returned {} additional candidates", e5_count);
            }
        }

        // Also add candidates from E7 Code for better recall on code queries
        if let Some(e7_index) = self.index_registry.get(EmbedderIndex::E7Code) {
            if let Ok(e7_candidates) = e7_index.search(&query.e7_code, recall_k / 2, None) {
                let e7_count = e7_candidates.len();
                candidate_ids.extend(e7_candidates.into_iter().map(|(id, _)| id));
                debug!("Stage 1: E7 Code returned {} additional candidates", e7_count);
            }
        }

        info!(
            "Stage 1 complete: {} unique candidates from E13+E1+E5+E7",
            candidate_ids.len()
        );

        if candidate_ids.is_empty() {
            debug!("Stage 1: No candidates found, returning empty results");
            return Ok(Vec::new());
        }

        // =========================================================================
        // STAGE 2: MULTI-SPACE SCORING (Weighted fusion of semantic embedders)
        // Per AP-73: Temporal (E2-E4) MUST NOT be used in similarity scoring
        // =========================================================================
        let weights = self.resolve_weights(options);
        let mut scored_candidates: Vec<(Uuid, f32, [f32; 13], SemanticFingerprint)> =
            Vec::with_capacity(candidate_ids.len());

        for id in candidate_ids {
            // Skip soft-deleted
            if !options.include_deleted && self.is_soft_deleted(&id) {
                continue;
            }

            // Fetch full fingerprint from RocksDB
            if let Some(data) = self.get_fingerprint_raw(id)? {
                let fp = deserialize_teleological_fingerprint(&data);

                // Compute all 13 embedder scores (includes E6 sparse similarity)
                let embedder_scores = self.compute_embedder_scores(query, &fp.semantic);

                // Compute weighted fusion (semantic embedders only, temporal=0)
                let fusion_score = compute_semantic_fusion(&embedder_scores, &weights);

                // Apply min_similarity filter at Stage 2
                if fusion_score >= options.min_similarity {
                    scored_candidates.push((id, fusion_score, embedder_scores, fp.semantic.clone()));
                }
            }
        }

        // Sort by fusion score descending
        scored_candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top candidates for Stage 3 (or final results if no rerank)
        scored_candidates.truncate(stage2_k);

        info!(
            "Stage 2 complete: {} candidates after weighted fusion scoring",
            scored_candidates.len()
        );

        if scored_candidates.is_empty() {
            debug!("Stage 2: No candidates passed min_similarity filter");
            return Ok(Vec::new());
        }

        // =========================================================================
        // STAGE 3: OPTIONAL E12 COLBERT RE-RANKING (MaxSim)
        // Per AP-74: E12 ColBERT MUST only be used for re-ranking, NOT initial retrieval
        // =========================================================================
        let final_candidates = if options.enable_rerank && !query.e12_late_interaction.is_empty() {
            info!("Stage 3: Applying E12 ColBERT MaxSim re-ranking");
            self.rerank_with_colbert(query, scored_candidates, options.top_k)
        } else {
            if options.enable_rerank {
                debug!("Stage 3: Re-ranking requested but E12 query tokens empty, skipping");
            }
            // No re-ranking, just truncate to top_k
            scored_candidates.truncate(options.top_k);
            scored_candidates
        };

        info!(
            "Stage 3 complete: {} final results",
            final_candidates.len()
        );

        // =========================================================================
        // BUILD FINAL RESULTS
        // =========================================================================
        let mut results = Vec::with_capacity(final_candidates.len());

        for (id, score, embedder_scores, _semantic) in final_candidates {
            // Re-fetch the full TeleologicalFingerprint for the result
            if let Some(data) = self.get_fingerprint_raw(id)? {
                let fp = deserialize_teleological_fingerprint(&data);
                results.push(TeleologicalSearchResult::new(fp, score, embedder_scores));
            }
        }

        debug!("Pipeline search returned {} results", results.len());
        Ok(results)
    }

    /// Re-rank candidates using E12 ColBERT MaxSim scoring.
    ///
    /// Per AP-74: ColBERT is for re-ranking only.
    /// Uses token-level MaxSim for precise similarity scoring.
    fn rerank_with_colbert(
        &self,
        query: &SemanticFingerprint,
        mut candidates: Vec<(Uuid, f32, [f32; 13], SemanticFingerprint)>,
        top_k: usize,
    ) -> Vec<(Uuid, f32, [f32; 13], SemanticFingerprint)> {
        use crate::teleological::search::compute_maxsim_direct;

        // Compute MaxSim scores for all candidates
        for (_, score, embedder_scores, semantic) in candidates.iter_mut() {
            if !semantic.e12_late_interaction.is_empty() {
                // Compute E12 ColBERT MaxSim
                let maxsim_score = compute_maxsim_direct(
                    &query.e12_late_interaction,
                    &semantic.e12_late_interaction,
                );

                // Update E12 score in embedder_scores
                embedder_scores[11] = maxsim_score;

                // Blend Stage 2 score with MaxSim (60% fusion, 40% MaxSim)
                // This preserves semantic ranking while adding precision
                let blended = *score * 0.6 + maxsim_score * 0.4;
                *score = blended;

                debug!(
                    "E12 rerank: fusion={:.4}, maxsim={:.4}, blended={:.4}",
                    *score * 0.6 / 0.6, maxsim_score, blended
                );
            }
        }

        // Re-sort by blended score
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        candidates.truncate(top_k);
        candidates
    }

    /// Resolve weight profile to weight array.
    ///
    /// Returns DEFAULT_SEMANTIC_WEIGHTS if profile not found.
    fn resolve_weights(&self, options: &TeleologicalSearchOptions) -> [f32; 13] {
        // For now, use default weights. In the future, integrate with weights.rs profiles.
        // TODO: Look up weight profile from options.weight_profile
        if let Some(ref _profile) = options.weight_profile {
            // Weight profiles are defined in MCP crate - use defaults here
            // The MCP handler can resolve profiles before calling this
            debug!("Weight profile specified but using defaults in storage layer");
        }
        DEFAULT_SEMANTIC_WEIGHTS
    }

    /// Apply recency boost POST-retrieval.
    ///
    /// Per ARCH-14: Temporal is a POST-retrieval boost, not similarity.
    /// Formula: final = semantic * (1.0 - boost) + temporal * boost
    fn apply_recency_boost(
        &self,
        results: &mut [TeleologicalSearchResult],
        _query: &SemanticFingerprint,
        boost_factor: f32,
    ) {
        debug!("Applying recency boost factor: {}", boost_factor);

        for result in results.iter_mut() {
            // E2 temporal recent score (index 1)
            let temporal_score = result.embedder_scores[1];

            // Blend semantic and temporal scores
            let original = result.similarity;
            result.similarity = original * (1.0 - boost_factor) + temporal_score * boost_factor;

            debug!(
                "Recency boost: {} -> {} (temporal: {})",
                original, result.similarity, temporal_score
            );
        }

        // Re-sort after boosting
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Search by text (internal async wrapper).
    pub(crate) async fn search_text_async(
        &self,
        _text: &str,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        // Text search requires embedding generation, which is not available in storage layer
        // Return empty results with a warning
        warn!("search_text called but embedding generation not available in storage layer");
        warn!("Use embedding service to generate query embeddings, then call search_semantic");
        Ok(Vec::with_capacity(options.top_k))
    }

    /// Search by sparse vector (internal async wrapper).
    pub(crate) async fn search_sparse_async(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>> {
        debug!(
            "Searching sparse with {} active terms, top_k={}",
            sparse_query.nnz(),
            top_k
        );

        let cf = self.get_cf(CF_E13_SPLADE_INVERTED)?;

        // Accumulate scores per document
        let mut doc_scores: HashMap<Uuid, f32> = HashMap::new();

        for (i, &term_id) in sparse_query.indices.iter().enumerate() {
            let term_key = e13_splade_inverted_key(term_id);
            let query_weight = sparse_query.values[i];

            if let Some(data) = self.db.get_cf(cf, term_key).map_err(|e| {
                TeleologicalStoreError::rocksdb_op("get", CF_E13_SPLADE_INVERTED, None, e)
            })? {
                let doc_ids = deserialize_memory_id_list(&data);

                for doc_id in doc_ids {
                    // Skip soft-deleted
                    if self.is_soft_deleted(&doc_id) {
                        continue;
                    }

                    // Simple term frequency scoring
                    // TODO: Implement BM25 or other scoring
                    *doc_scores.entry(doc_id).or_insert(0.0) += query_weight;
                }
            }
        }

        // Sort by score descending
        let mut results: Vec<(Uuid, f32)> = doc_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to top_k
        results.truncate(top_k);

        debug!("Sparse search returned {} results", results.len());
        Ok(results)
    }
}
