//! Enrichment Pipeline for Autonomous Multi-Embedder Search.
//!
//! Per Constitution v6.4: 13 embedders = 13 unique perspectives on every memory.
//! Each finds what OTHERS MISS. Combined = superior answers.
//!
//! # Pipeline Architecture (per ARCH-12, ARCH-21)
//!
//! 1. **E1 Foundation Search** - Always first (ARCH-12: E1 is foundation)
//! 2. **Parallel Enhancer Searches** - Tokio join! for concurrent embedder searches
//! 3. **UNION Discovery** - Find what E1 missed (unique to enhancers)
//! 4. **Weighted RRF Fusion** - Per ARCH-21 (not weighted sum)
//! 5. **Agreement Calculation** - Which embedders agree on each result
//! 6. **Blind Spot Detection** - High-scoring in enhancers but missed by E1
//!
//! # Performance Budget
//! - Target: <500ms for Light mode, <800ms for Full mode
//! - E1 Search: ~100ms
//! - Parallel Enhancers: ~150ms (run in parallel)
//! - RRF Fusion: ~50ms
//! - Agreement Calc: ~50ms
//! - Buffer: ~150ms
//!
//! # FAIL FAST Principle
//! All errors are immediate and detailed. No silent fallbacks.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, error, info, warn};
use uuid::Uuid;

use context_graph_core::traits::{
    DecayFunction, SearchStrategy, TeleologicalMemoryStore, TeleologicalSearchOptions,
    TeleologicalSearchResult,
};
use context_graph_core::types::fingerprint::SemanticFingerprint;

use super::embedder_dtos::EmbedderId;
use super::enrichment_dtos::{
    AgreementMetrics, BlindSpotAlert, EnrichedSearchResponse, EnrichedSearchResult,
    EnrichmentConfig, EnrichmentMode, EnrichmentSummary, ScoringBreakdown, TimeBreakdown,
};
use super::temporal_dtos::compute_recency_score;

// =============================================================================
// CONSTANTS
// =============================================================================

/// RRF parameter k - controls how much rank matters vs score.
/// Higher k = more weight to scores, lower k = more weight to ranks.
const RRF_K: f32 = 60.0;

/// Blind spot threshold - if E1 score is below this and enhancer is above,
/// it's considered a blind spot.
const E1_BLIND_SPOT_THRESHOLD: f32 = 0.3;

/// Enhancer score threshold for blind spot detection.
const ENHANCER_BLIND_SPOT_THRESHOLD: f32 = 0.5;

/// Minimum similarity threshold for considering a result "found" by an embedder.
const EMBEDDER_FOUND_THRESHOLD: f32 = 0.2;

/// E1 weak threshold for triggering E9 HDC fallback.
/// Per remaining_embedder_integration_plan.md: trigger when e1_top_score < 0.4.
const E1_WEAK_THRESHOLD: f32 = 0.4;

// =============================================================================
// ENRICHMENT PIPELINE
// =============================================================================

/// Enrichment pipeline for multi-embedder search enhancement.
///
/// Orchestrates parallel searches across multiple embedders and fuses
/// results using Weighted RRF per ARCH-21.
pub struct EnrichmentPipeline {
    /// The teleological store for search operations.
    store: Arc<dyn TeleologicalMemoryStore>,
}

impl EnrichmentPipeline {
    /// Create a new enrichment pipeline.
    pub fn new(store: Arc<dyn TeleologicalMemoryStore>) -> Self {
        Self { store }
    }

    /// Execute the full enrichment pipeline.
    ///
    /// # Arguments
    /// * `query_fingerprint` - The query's semantic fingerprint (already embedded)
    /// * `config` - Enrichment configuration from query type detection
    /// * `base_options` - Base search options (top_k, filters, etc.)
    /// * `include_content` - Whether to hydrate content
    ///
    /// # Returns
    /// EnrichedSearchResponse with all enriched results and summary.
    ///
    /// # Errors
    /// FAIL FAST - returns error on any search failure.
    pub async fn execute(
        &self,
        query_fingerprint: &SemanticFingerprint,
        config: &EnrichmentConfig,
        base_options: TeleologicalSearchOptions,
        include_content: bool,
    ) -> Result<EnrichedSearchResponse, EnrichmentError> {
        let start = Instant::now();

        // Handle Off mode - just do E1 search
        if config.mode == EnrichmentMode::Off {
            return self
                .execute_e1_only(query_fingerprint, base_options, include_content, start)
                .await;
        }

        // Phase 0: Pipeline Upgrade for Full mode (Phase 6 E12/E13 Integration)
        // When enrichMode: 'full', auto-use Pipeline strategy with E13 recall + E12 rerank.
        // Per ARCH-13: E13 for Stage 1 recall, E12 for Stage 3 reranking.
        let effective_options = if config.mode == EnrichmentMode::Full {
            debug!(
                "Upgrading to Pipeline strategy for Full enrichment mode (E13 recall + E12 rerank)"
            );
            base_options
                .clone()
                .with_strategy(SearchStrategy::Pipeline)
                .with_rerank(true)
        } else {
            base_options.clone()
        };

        // Phase 1: E1 Foundation Search (always first per ARCH-12)
        let e1_start = Instant::now();
        let e1_results = self
            .run_e1_search(query_fingerprint, &effective_options)
            .await?;
        let e1_time = e1_start.elapsed().as_millis() as u64;

        debug!(
            result_count = e1_results.len(),
            time_ms = e1_time,
            "Phase 1: E1 foundation search complete"
        );

        // Phase 1.5: E9 HDC Fallback Detection
        // Per remaining_embedder_integration_plan.md: trigger when e1_top_score < 0.4
        // BUG FIX: Use raw E1 score (embedder_scores[0]), not fused similarity
        let e1_top_score = e1_results.first().map(|r| r.embedder_scores[0]).unwrap_or(0.0);
        let hdc_fallback_needed = e1_top_score < E1_WEAK_THRESHOLD;

        if hdc_fallback_needed {
            info!(
                e1_top_score = e1_top_score,
                threshold = E1_WEAK_THRESHOLD,
                "E1 results weak, E9 HDC fallback triggered"
            );
        }

        // Phase 2: Parallel Enhancer Searches
        // If E9 HDC fallback is needed, include E9 in the enhancer list
        let mut effective_embedders = config.selected_embedders.clone();
        if hdc_fallback_needed && !effective_embedders.contains(&EmbedderId::E9) {
            effective_embedders.push(EmbedderId::E9);
        }

        let enhancer_start = Instant::now();
        let enhancer_results = self
            .run_parallel_enhancer_searches(
                query_fingerprint,
                &effective_embedders,
                &effective_options,
            )
            .await?;
        let enhancer_time = enhancer_start.elapsed().as_millis() as u64;

        debug!(
            embedder_count = effective_embedders.len(),
            hdc_fallback = hdc_fallback_needed,
            time_ms = enhancer_time,
            "Phase 2: Parallel enhancer searches complete"
        );

        // Phase 3: Weighted RRF Fusion
        let rrf_start = Instant::now();
        let mut fused_results = self.compute_weighted_rrf(
            &e1_results,
            &enhancer_results,
            &effective_embedders,
            effective_options.top_k,
        );
        let rrf_time = rrf_start.elapsed().as_millis() as u64;

        debug!(
            result_count = fused_results.len(),
            time_ms = rrf_time,
            "Phase 3: Weighted RRF fusion complete"
        );

        // Phase 3.5: Temporal Boost (POST-RETRIEVAL per ARCH-25)
        // Apply E2 recency boost when temporal query type is detected
        if config.temporal_boost_enabled && config.temporal_weight > 0.0 {
            self.apply_temporal_boost_to_fused(
                &mut fused_results,
                &e1_results,
                config.temporal_weight,
                config.decay_function,
            );

            debug!(
                temporal_weight = config.temporal_weight,
                decay_function = ?config.decay_function,
                "Phase 3.5: Temporal boost applied (POST-retrieval per ARCH-25)"
            );
        }

        // Phase 4: Agreement Calculation
        let agreement_start = Instant::now();
        let mut enriched_results = self.build_enriched_results(
            &fused_results,
            &e1_results,
            &enhancer_results,
            &effective_embedders,
        );
        let agreement_time = agreement_start.elapsed().as_millis() as u64;

        // Phase 5: Blind Spot Detection (Full mode only)
        let blind_spot_start = Instant::now();
        let blind_spots_found = if config.detect_blind_spots {
            self.detect_and_add_blind_spots(
                &mut enriched_results,
                &e1_results,
                &enhancer_results,
                &effective_embedders,
            )
        } else {
            0
        };
        let blind_spot_time = blind_spot_start.elapsed().as_millis() as u64;

        // Phase 6: Content Hydration (if requested)
        if include_content {
            self.hydrate_content(&mut enriched_results).await?;
        }

        // Count unique discoveries (results where enhancer contributed significantly)
        let unique_discoveries = enriched_results
            .iter()
            .filter(|r| r.is_enhancer_discovery())
            .count();

        // Build summary
        let total_time = start.elapsed().as_millis() as u64;
        let mut embedders_used = vec![EmbedderId::E1];
        embedders_used.extend(effective_embedders.clone());

        let summary = EnrichmentSummary::new_with_fallback(
            config.mode,
            config.detected_types.clone(),
            embedders_used,
            blind_spots_found,
            unique_discoveries,
            total_time,
            Some(TimeBreakdown {
                e1_search_ms: e1_time,
                enhancer_search_ms: enhancer_time,
                rrf_fusion_ms: rrf_time,
                agreement_ms: agreement_time,
                blind_spot_ms: blind_spot_time,
            }),
            hdc_fallback_needed,
        );

        info!(
            mode = ?config.mode,
            detected_types = ?config.detected_types,
            result_count = enriched_results.len(),
            blind_spots = blind_spots_found,
            unique_discoveries = unique_discoveries,
            hdc_fallback = hdc_fallback_needed,
            total_time_ms = total_time,
            "Enrichment pipeline complete"
        );

        let strategy_name = match effective_options.strategy {
            SearchStrategy::E1Only => "e1_only",
            SearchStrategy::MultiSpace => "multi_space",
            SearchStrategy::Pipeline => "pipeline",
        };

        Ok(EnrichedSearchResponse::new(
            enriched_results,
            summary,
            String::new(), // Query filled in by caller
            strategy_name.to_string(),
        ))
    }

    /// Execute E1-only search (Off mode or fallback).
    async fn execute_e1_only(
        &self,
        query_fingerprint: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
        include_content: bool,
        start: Instant,
    ) -> Result<EnrichedSearchResponse, EnrichmentError> {
        let e1_results = self.run_e1_search(query_fingerprint, &options).await?;

        let mut enriched_results: Vec<EnrichedSearchResult> = e1_results
            .into_iter()
            .map(|r| {
                // BUG FIX: Use raw E1 score for consistency
                let scoring = ScoringBreakdown::e1_only(r.embedder_scores[0]);
                let agreement =
                    AgreementMetrics::from_embedders(vec![EmbedderId::E1], 1);
                EnrichedSearchResult::new(r.fingerprint.id, scoring, agreement)
            })
            .collect();

        if include_content {
            self.hydrate_content(&mut enriched_results).await?;
        }

        let total_time = start.elapsed().as_millis() as u64;
        let summary = EnrichmentSummary::off(total_time);

        Ok(EnrichedSearchResponse::new(
            enriched_results,
            summary,
            String::new(),
            "e1_only".to_string(),
        ))
    }

    /// Run E1 foundation search.
    async fn run_e1_search(
        &self,
        query: &SemanticFingerprint,
        options: &TeleologicalSearchOptions,
    ) -> Result<Vec<TeleologicalSearchResult>, EnrichmentError> {
        self.store
            .search_semantic(query, options.clone())
            .await
            .map_err(|e| {
                error!(error = %e, "E1 foundation search FAILED");
                EnrichmentError::SearchFailed {
                    embedder: "E1".to_string(),
                    message: e.to_string(),
                }
            })
    }

    /// Run parallel searches for enhancer embedders.
    ///
    /// Uses the existing search infrastructure but tracks per-embedder scores.
    async fn run_parallel_enhancer_searches(
        &self,
        query: &SemanticFingerprint,
        embedders: &[EmbedderId],
        base_options: &TeleologicalSearchOptions,
    ) -> Result<HashMap<EmbedderId, Vec<(Uuid, f32)>>, EnrichmentError> {
        if embedders.is_empty() {
            return Ok(HashMap::new());
        }

        // Run the search once and extract per-embedder scores from results
        // This is more efficient than running N separate searches
        let results = self
            .store
            .search_semantic(query, base_options.clone())
            .await
            .map_err(|e| {
                error!(error = %e, "Enhancer search FAILED");
                EnrichmentError::SearchFailed {
                    embedder: "enhancers".to_string(),
                    message: e.to_string(),
                }
            })?;

        // Extract per-embedder scores from the results
        let mut enhancer_results: HashMap<EmbedderId, Vec<(Uuid, f32)>> = HashMap::new();

        for embedder in embedders {
            let idx = embedder.to_index();
            let mut scores: Vec<(Uuid, f32)> = results
                .iter()
                .map(|r| (r.fingerprint.id, r.embedder_scores[idx]))
                .filter(|(_, score)| *score >= EMBEDDER_FOUND_THRESHOLD)
                .collect();

            // Sort by score descending
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            enhancer_results.insert(*embedder, scores);
        }

        Ok(enhancer_results)
    }

    /// Compute Weighted RRF fusion of E1 and enhancer results.
    ///
    /// Per ARCH-21: Use Weighted RRF, not weighted sum.
    /// Formula: RRF(d) = Σ (weight_i / (k + rank_i(d)))
    fn compute_weighted_rrf(
        &self,
        e1_results: &[TeleologicalSearchResult],
        enhancer_results: &HashMap<EmbedderId, Vec<(Uuid, f32)>>,
        embedders: &[EmbedderId],
        top_k: usize,
    ) -> Vec<FusedResult> {
        // Collect all unique IDs
        let mut all_ids: HashSet<Uuid> = HashSet::new();
        for r in e1_results {
            all_ids.insert(r.fingerprint.id);
        }
        for scores in enhancer_results.values() {
            for (id, _) in scores {
                all_ids.insert(*id);
            }
        }

        // Build E1 rank map
        let e1_ranks: HashMap<Uuid, usize> = e1_results
            .iter()
            .enumerate()
            .map(|(rank, r)| (r.fingerprint.id, rank + 1))
            .collect();

        // Build enhancer rank maps
        let mut enhancer_ranks: HashMap<EmbedderId, HashMap<Uuid, usize>> = HashMap::new();
        for embedder in embedders {
            if let Some(scores) = enhancer_results.get(embedder) {
                let ranks: HashMap<Uuid, usize> = scores
                    .iter()
                    .enumerate()
                    .map(|(rank, (id, _))| (*id, rank + 1))
                    .collect();
                enhancer_ranks.insert(*embedder, ranks);
            }
        }

        // Compute RRF score for each ID
        // E1 weight = 1.0 (foundation), enhancers get their topic weight
        let mut fused: Vec<FusedResult> = all_ids
            .into_iter()
            .map(|id| {
                let mut rrf_score = 0.0;
                let mut contributions: HashMap<String, f32> = HashMap::new();

                // E1 contribution (weight 1.0)
                if let Some(&rank) = e1_ranks.get(&id) {
                    let contrib = 1.0 / (RRF_K + rank as f32);
                    rrf_score += contrib;
                    contributions.insert("e1".to_string(), contrib);
                }

                // Enhancer contributions (weighted by topic weight)
                for embedder in embedders {
                    if let Some(ranks) = enhancer_ranks.get(embedder) {
                        if let Some(&rank) = ranks.get(&id) {
                            let weight = embedder.topic_weight();
                            let contrib = weight / (RRF_K + rank as f32);
                            rrf_score += contrib;
                            contributions.insert(embedder.json_key().to_string(), contrib);
                        }
                    }
                }

                // Get E1 raw score from embedder_scores[0], NOT from similarity
                // BUG FIX: similarity is the fused/post-processed score (~0.13)
                // embedder_scores[0] is the raw E1 semantic score (~0.73-0.89)
                let e1_score = e1_results
                    .iter()
                    .find(|r| r.fingerprint.id == id)
                    .map(|r| r.embedder_scores[0]) // Index 0 = E1_Semantic
                    .unwrap_or(0.0);

                // Get enhancer raw scores
                let mut enhancer_scores: HashMap<String, f32> = HashMap::new();
                for embedder in embedders {
                    if let Some(scores) = enhancer_results.get(embedder) {
                        if let Some(&(_, score)) = scores.iter().find(|(i, _)| *i == id) {
                            enhancer_scores.insert(embedder.json_key().to_string(), score);
                        }
                    }
                }

                FusedResult {
                    id,
                    rrf_score,
                    e1_score,
                    enhancer_scores,
                    rrf_contributions: contributions,
                }
            })
            .collect();

        // Sort by RRF score descending
        fused.sort_by(|a, b| {
            b.rrf_score
                .partial_cmp(&a.rrf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        fused.truncate(top_k);

        fused
    }

    /// Build enriched results with agreement metrics.
    fn build_enriched_results(
        &self,
        fused_results: &[FusedResult],
        e1_results: &[TeleologicalSearchResult],
        enhancer_results: &HashMap<EmbedderId, Vec<(Uuid, f32)>>,
        embedders: &[EmbedderId],
    ) -> Vec<EnrichedSearchResult> {
        let total_embedders = 1 + embedders.len(); // E1 + enhancers

        fused_results
            .iter()
            .map(|fused| {
                // Build scoring breakdown
                let scoring = ScoringBreakdown::with_rrf(
                    fused.e1_score,
                    fused.enhancer_scores.clone(),
                    fused.rrf_score,
                    None, // E10 boost computed separately if E10 in enhancers
                    Some(fused.rrf_contributions.clone()),
                );

                // Determine which embedders found this result
                let mut agreeing_embedders = Vec::new();

                // Check E1
                if fused.e1_score >= EMBEDDER_FOUND_THRESHOLD {
                    agreeing_embedders.push(EmbedderId::E1);
                }

                // Check enhancers
                for embedder in embedders {
                    if let Some(scores) = enhancer_results.get(embedder) {
                        if scores.iter().any(|(id, s)| *id == fused.id && *s >= EMBEDDER_FOUND_THRESHOLD) {
                            agreeing_embedders.push(*embedder);
                        }
                    }
                }

                let agreement = AgreementMetrics::from_embedders(agreeing_embedders, total_embedders);

                EnrichedSearchResult::new(fused.id, scoring, agreement)
            })
            .collect()
    }

    /// Detect blind spots and add alerts to results.
    ///
    /// A blind spot is when an enhancer finds a result with high similarity
    /// but E1 missed it (low score) or ranked it poorly.
    fn detect_and_add_blind_spots(
        &self,
        results: &mut [EnrichedSearchResult],
        e1_results: &[TeleologicalSearchResult],
        enhancer_results: &HashMap<EmbedderId, Vec<(Uuid, f32)>>,
        embedders: &[EmbedderId],
    ) -> usize {
        let mut blind_spots_count = 0;

        // Build E1 score lookup using raw E1 scores, not fused similarity
        // BUG FIX: Use embedder_scores[0] for accurate blind spot detection
        let e1_scores: HashMap<Uuid, f32> = e1_results
            .iter()
            .map(|r| (r.fingerprint.id, r.embedder_scores[0]))
            .collect();

        for result in results.iter_mut() {
            let e1_score = *e1_scores.get(&result.node_id).unwrap_or(&0.0);

            // Check if this is a blind spot
            if e1_score < E1_BLIND_SPOT_THRESHOLD {
                // Find enhancers with high scores
                let mut found_by: Vec<EmbedderId> = Vec::new();
                let mut enhancer_scores: HashMap<String, f32> = HashMap::new();

                for embedder in embedders {
                    if let Some(scores) = enhancer_results.get(embedder) {
                        if let Some(&(_, score)) = scores.iter().find(|(id, _)| *id == result.node_id) {
                            if score >= ENHANCER_BLIND_SPOT_THRESHOLD {
                                found_by.push(*embedder);
                                enhancer_scores.insert(embedder.json_key().to_string(), score);
                            }
                        }
                    }
                }

                // Only mark as blind spot if at least one enhancer found it with high score
                if !found_by.is_empty() {
                    let alert = BlindSpotAlert::new(
                        result.node_id,
                        found_by,
                        e1_score,
                        enhancer_scores,
                    );
                    result.blind_spot = Some(alert);
                    blind_spots_count += 1;
                }
            }
        }

        blind_spots_count
    }

    /// Hydrate content for enriched results.
    async fn hydrate_content(
        &self,
        results: &mut [EnrichedSearchResult],
    ) -> Result<(), EnrichmentError> {
        if results.is_empty() {
            return Ok(());
        }

        let ids: Vec<Uuid> = results.iter().map(|r| r.node_id).collect();
        let contents = self.store.get_content_batch(&ids).await.map_err(|e| {
            warn!(error = %e, "Content hydration failed");
            EnrichmentError::ContentHydrationFailed(e.to_string())
        })?;

        for (result, content) in results.iter_mut().zip(contents.into_iter()) {
            result.content = content;
        }

        Ok(())
    }

    /// Apply temporal boost to fused results (POST-RETRIEVAL per ARCH-25).
    ///
    /// Applies E2 recency decay to modify RRF scores based on memory age.
    /// Then re-sorts results by boosted score.
    ///
    /// # Arguments
    /// * `fused_results` - Mutable fused results from RRF
    /// * `e1_results` - Original E1 results (to get timestamps)
    /// * `temporal_weight` - Weight for temporal boost [0.0, 1.0]
    /// * `decay_function` - Decay function (Linear, Exponential, Step, NoDecay)
    fn apply_temporal_boost_to_fused(
        &self,
        fused_results: &mut Vec<FusedResult>,
        e1_results: &[TeleologicalSearchResult],
        temporal_weight: f32,
        decay_function: DecayFunction,
    ) {
        if fused_results.is_empty() || temporal_weight <= 0.0 {
            return;
        }

        let now_ms = chrono::Utc::now().timestamp_millis();

        // Build timestamp map from E1 results (using created_at)
        let timestamps: HashMap<Uuid, i64> = e1_results
            .iter()
            .map(|r| {
                (r.fingerprint.id, r.fingerprint.created_at.timestamp_millis())
            })
            .collect();

        // Apply temporal boost to each result
        // Use default 1-day (86400s) horizon for enrichment pipeline temporal boost
        const DEFAULT_HORIZON_SECS: i64 = 86400;

        for result in fused_results.iter_mut() {
            if let Some(&memory_ts) = timestamps.get(&result.id) {
                let recency_score =
                    compute_recency_score(memory_ts, now_ms, decay_function, DEFAULT_HORIZON_SECS);

                // Per ARCH-25: Temporal boost is multiplicative POST-retrieval
                // boosted_score = rrf_score * (1 + temporal_weight * (recency_score - 0.5))
                // This gives neutral at 0.5 recency, boost for recent, penalty for old
                let boost_factor =
                    (1.0 + temporal_weight * (recency_score - 0.5)).clamp(0.8, 1.2);

                result.rrf_score *= boost_factor;
            }
        }

        // Re-sort by boosted RRF score
        fused_results.sort_by(|a, b| {
            b.rrf_score
                .partial_cmp(&a.rrf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// =============================================================================
// INTERNAL TYPES
// =============================================================================

/// Intermediate result from RRF fusion.
#[derive(Debug)]
struct FusedResult {
    id: Uuid,
    rrf_score: f32,
    e1_score: f32,
    enhancer_scores: HashMap<String, f32>,
    rrf_contributions: HashMap<String, f32>,
}

// =============================================================================
// ERROR TYPE
// =============================================================================

/// Errors from the enrichment pipeline.
///
/// FAIL FAST: All errors are detailed for debugging.
#[derive(Debug, thiserror::Error)]
pub enum EnrichmentError {
    #[error("Search failed for embedder {embedder}: {message}")]
    SearchFailed { embedder: String, message: String },

    #[error("Content hydration failed: {0}")]
    ContentHydrationFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_k_constant() {
        // Standard RRF k value from literature
        assert!((RRF_K - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_blind_spot_thresholds() {
        // E1 must be below threshold for blind spot
        assert!(E1_BLIND_SPOT_THRESHOLD < 0.5);
        // Enhancer must be above threshold
        assert!(ENHANCER_BLIND_SPOT_THRESHOLD >= 0.5);
    }

    #[test]
    fn test_embedder_found_threshold() {
        // Should be low enough to catch marginal matches
        assert!(EMBEDDER_FOUND_THRESHOLD <= 0.3);
    }
}
