//! Default teleological retrieval pipeline implementation.
//!
//! This module provides `DefaultTeleologicalPipeline`, the standard
//! implementation of the 5-stage teleological retrieval process.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::{debug, error, instrument, warn};
use uuid::Uuid;

use crate::alignment::{AlignmentConfig, GoalAlignmentCalculator};
use crate::error::{CoreError, CoreResult};
use crate::johari::JohariTransitionManager;
use crate::purpose::GoalHierarchy;
use crate::traits::TeleologicalMemoryStore;
use crate::types::fingerprint::{TeleologicalFingerprint, NUM_EMBEDDERS};

use super::super::teleological_query::TeleologicalQuery;
use super::super::teleological_result::{PipelineBreakdown, ScoredMemory, TeleologicalRetrievalResult};
use super::super::{
    MultiEmbeddingQueryExecutor, MultiEmbeddingResult, PipelineStageTiming,
};
use super::traits::{PipelineHealth, TeleologicalRetrievalPipeline};

/// Default implementation of the teleological retrieval pipeline.
///
/// Integrates L001-L007 components to provide the full 5-stage
/// retrieval process with teleological filtering.
///
/// # Thread Safety
///
/// All internal components are wrapped in `Arc` for shared access
/// across async tasks.
pub struct DefaultTeleologicalPipeline<E, A, J, S>
where
    E: MultiEmbeddingQueryExecutor,
    A: GoalAlignmentCalculator,
    J: JohariTransitionManager,
    S: TeleologicalMemoryStore,
{
    /// Multi-embedding executor (Stages 1-3, 5).
    pub(crate) executor: Arc<E>,

    /// Goal alignment calculator (Stage 4).
    pub(crate) alignment_calculator: Arc<A>,

    /// Johari manager for quadrant classification (Stage 4).
    #[allow(dead_code)]
    pub(crate) johari_manager: Arc<J>,

    /// Teleological memory store for fetching fingerprints (Stage 4).
    pub(crate) store: Arc<S>,

    /// Goal hierarchy for alignment computation.
    pub(crate) goal_hierarchy: Arc<GoalHierarchy>,

    /// Index size tracking for health checks.
    pub(crate) index_size: std::sync::atomic::AtomicUsize,

    /// Last successful query time.
    pub(crate) last_query_time: std::sync::RwLock<Option<Duration>>,
}

impl<E, A, J, S> DefaultTeleologicalPipeline<E, A, J, S>
where
    E: MultiEmbeddingQueryExecutor,
    A: GoalAlignmentCalculator,
    J: JohariTransitionManager,
    S: TeleologicalMemoryStore,
{
    /// Create a new pipeline with all required components.
    ///
    /// # Arguments
    /// * `executor` - Multi-embedding query executor
    /// * `alignment_calculator` - Goal alignment calculator
    /// * `johari_manager` - Johari transition manager
    /// * `store` - Teleological memory store for fingerprint retrieval (Stage 4)
    /// * `goal_hierarchy` - Goal hierarchy for alignment
    pub fn new(
        executor: Arc<E>,
        alignment_calculator: Arc<A>,
        johari_manager: Arc<J>,
        store: Arc<S>,
        goal_hierarchy: GoalHierarchy,
    ) -> Self {
        Self {
            executor,
            alignment_calculator,
            johari_manager,
            store,
            goal_hierarchy: Arc::new(goal_hierarchy),
            index_size: std::sync::atomic::AtomicUsize::new(0),
            last_query_time: std::sync::RwLock::new(None),
        }
    }

    /// Update the goal hierarchy.
    pub fn with_goal_hierarchy(mut self, hierarchy: GoalHierarchy) -> Self {
        self.goal_hierarchy = Arc::new(hierarchy);
        self
    }

    /// Build pipeline breakdown from multi-embedding result.
    pub(crate) fn build_breakdown(&self, me_result: &MultiEmbeddingResult) -> PipelineBreakdown {
        let mut breakdown = PipelineBreakdown::new();

        if let Some(ref space_breakdown) = me_result.space_breakdown {
            // Stage 1: First space (SPLADE-like)
            if let Some(s1) = space_breakdown.first() {
                breakdown.stage1_candidates = s1.ranked_ids();
            }

            // Stage 2: Second space (Matryoshka-like)
            if let Some(s2) = space_breakdown.get(1) {
                breakdown.stage2_candidates = s2.ranked_ids();
            }

            // Stage 3: All aggregated results
            breakdown.stage3_candidates = me_result.results.iter().map(|r| r.memory_id).collect();
        }

        // Stage 4 and 5 would be populated during actual filtering
        breakdown.stage4_candidates = me_result.results.iter().map(|r| r.memory_id).collect();
        breakdown.stage5_candidates = me_result
            .results
            .iter()
            .take(20) // Late interaction limit
            .map(|r| r.memory_id)
            .collect();

        breakdown
    }
}

#[async_trait]
impl<E, A, J, S> TeleologicalRetrievalPipeline for DefaultTeleologicalPipeline<E, A, J, S>
where
    E: MultiEmbeddingQueryExecutor + Send + Sync,
    A: GoalAlignmentCalculator + Send + Sync,
    J: JohariTransitionManager + Send + Sync,
    S: TeleologicalMemoryStore + Send + Sync,
{
    #[instrument(skip(self, query), fields(query_text = %query.text))]
    async fn execute(&self, query: &TeleologicalQuery) -> CoreResult<TeleologicalRetrievalResult> {
        let pipeline_start = Instant::now();

        // FAIL FAST: Validate query
        query.validate()?;

        let config = query.effective_config();

        debug!(
            text = %query.text,
            has_goals = query.has_goals(),
            has_johari_filter = query.has_johari_filter(),
            include_breakdown = query.include_breakdown,
            "Starting teleological pipeline"
        );

        // Build multi-embedding query for Stages 1-3, 5
        let me_query = super::super::MultiEmbeddingQuery {
            query_text: query.text.clone(),
            active_spaces: super::super::EmbeddingSpaceMask::ALL,
            space_weights: None, // Equal weighting
            per_space_limit: config.full_search_limit,
            final_limit: config.late_interaction_limit, // Final output limit
            min_similarity: 0.0,
            pipeline_config: Some(super::super::PipelineStageConfig {
                splade_candidates: config.splade_candidates,
                matryoshka_128d_limit: config.matryoshka_128d_limit,
                full_search_limit: config.full_search_limit,
                teleological_limit: config.teleological_limit,
                late_interaction_limit: config.late_interaction_limit,
                rrf_k: config.rrf_k,
                min_alignment_threshold: config.min_alignment_threshold,
            }),
            include_space_breakdown: query.include_breakdown,
            aggregation: super::super::AggregationStrategy::RRF { k: config.rrf_k },
        };

        // Execute Stages 1-3, 5 via multi-embedding executor
        let me_result = self.executor.execute(me_query).await.map_err(|e| {
            error!(error = %e, "Multi-embedding execution failed");
            CoreError::IndexError(format!("Pipeline Stages 1-3,5 failed: {}", e))
        })?;

        // Stage 4: Teleological filtering
        let stage4_start = Instant::now();

        // Fetch actual fingerprints from store for Stage 4 teleological computation
        let memory_ids: Vec<Uuid> = me_result.results.iter().map(|r| r.memory_id).collect();
        let fingerprints = self.store.retrieve_batch(&memory_ids).await.map_err(|e| {
            error!(error = %e, "Failed to fetch fingerprints for Stage 4");
            CoreError::StorageError(format!("Stage 4 fingerprint fetch failed: {}", e))
        })?;

        // Pair fingerprints with their aggregated matches
        let mut candidates: Vec<(&TeleologicalFingerprint, &super::super::AggregatedMatch)> = Vec::new();
        for (i, maybe_fp) in fingerprints.iter().enumerate() {
            if let Some(fp) = maybe_fp {
                candidates.push((fp, &me_result.results[i]));
            } else {
                warn!(
                    memory_id = %memory_ids[i],
                    "Fingerprint not found in store - skipping candidate"
                );
            }
        }

        // Apply proper Stage 4 filtering with real fingerprints
        let (stage4_results, filtered_count, avg_filtered_alignment) =
            self.apply_stage4_filtering(&candidates, query).await?;

        let stage4_time = stage4_start.elapsed();

        debug!(
            candidates_in = candidates.len(),
            results_out = stage4_results.len(),
            filtered = filtered_count,
            avg_filtered_alignment = avg_filtered_alignment,
            "Stage 4 teleological filtering complete"
        );

        // Build timing from executor result + Stage 4
        let timing = if let Some(ref me_timing) = me_result.stage_timings {
            PipelineStageTiming::new(
                me_timing.stage1_splade,
                me_timing.stage2_matryoshka,
                me_timing.stage3_full_hnsw,
                stage4_time,
                me_timing.stage5_late_interaction,
                [
                    me_timing.candidates_per_stage[0],
                    me_timing.candidates_per_stage[1],
                    me_timing.candidates_per_stage[2],
                    stage4_results.len(),
                    me_timing.candidates_per_stage[4],
                ],
            )
        } else {
            PipelineStageTiming::new(
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                stage4_time,
                Duration::ZERO,
                [0, 0, 0, stage4_results.len(), 0],
            )
        };

        let total_time = pipeline_start.elapsed();

        // Update last query time
        if let Ok(mut guard) = self.last_query_time.write() {
            *guard = Some(total_time);
        }

        // Build result
        let mut result = TeleologicalRetrievalResult::new(
            stage4_results,
            timing,
            total_time,
            me_result.spaces_searched,
            me_result.spaces_failed,
        );

        // Add breakdown if requested
        if query.include_breakdown {
            let breakdown = self.build_breakdown(&me_result);
            result = result.with_breakdown(breakdown);
        }

        debug!(
            result_count = result.len(),
            total_time_ms = total_time.as_millis(),
            within_target = result.within_latency_target(),
            "Teleological pipeline complete"
        );

        Ok(result)
    }

    async fn filter_by_alignment(
        &self,
        candidates: &[&TeleologicalFingerprint],
        query: &TeleologicalQuery,
    ) -> CoreResult<Vec<ScoredMemory>> {
        let config = query.effective_config();
        let min_alignment = config.min_alignment_threshold;

        let alignment_config = AlignmentConfig::with_hierarchy((*self.goal_hierarchy).clone())
            .with_min_alignment(min_alignment);

        let mut results = Vec::with_capacity(candidates.len());

        for fingerprint in candidates {
            let alignment_result = self
                .alignment_calculator
                .compute_alignment(fingerprint, &alignment_config)
                .await;

            let (goal_alignment, is_misaligned) = match alignment_result {
                Ok(r) => (r.score.composite_score, r.flags.needs_intervention()),
                Err(_) => (0.0, true),
            };

            if goal_alignment < min_alignment {
                continue;
            }

            let johari_quadrant = self.compute_dominant_quadrant(fingerprint);

            if let Some(ref allowed) = query.johari_filter {
                if !allowed.contains(&johari_quadrant) {
                    continue;
                }
            }

            let scored = ScoredMemory::new(
                fingerprint.id,
                goal_alignment, // Use alignment as score for direct filtering
                fingerprint.alignment_score,
                fingerprint.alignment_score,
                goal_alignment,
                johari_quadrant,
                NUM_EMBEDDERS, // Assume all spaces
            )
            .with_misalignment(is_misaligned);

            results.push(scored);
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let limit = config.teleological_limit;
        if results.len() > limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn health_check(&self) -> CoreResult<PipelineHealth> {
        let spaces = self.executor.available_spaces();
        // TASK-P0-005: Renamed from has_north_star per ARCH-03
        let has_strategic_goal = self.goal_hierarchy.has_top_level_goals();
        let last_time = self.last_query_time.read().ok().and_then(|g| *g);
        let index_size = self.index_size.load(std::sync::atomic::Ordering::Relaxed);

        Ok(PipelineHealth {
            is_healthy: spaces.len() == 13 && has_strategic_goal,
            spaces_available: spaces.len(),
            has_goal_hierarchy: has_strategic_goal,
            index_size,
            last_query_time: last_time,
        })
    }
}
