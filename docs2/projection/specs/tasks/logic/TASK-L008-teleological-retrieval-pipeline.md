# TASK-L008: Teleological Retrieval Pipeline

```yaml
metadata:
  id: "TASK-L008"
  title: "Teleological Retrieval Pipeline"
  layer: "logic"
  priority: "P0"
  estimated_hours: 12
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-L001"  # Multi-Embedding Query Executor
    - "TASK-L002"  # Purpose Vector Computation
    - "TASK-L003"  # Goal Alignment Calculator
    - "TASK-L004"  # Johari Transition Manager
    - "TASK-L005"  # Per-Space HNSW Index Builder
    - "TASK-L006"  # Purpose Pattern Index
    - "TASK-L007"  # Cross-Space Similarity Engine
  spec_refs:
    - "projectionplan1.md:retrieval-pipeline"
    - "projectionplan2.md:teleological-retrieval"
```

## Problem Statement

Implement the complete teleological retrieval pipeline that orchestrates multi-stage retrieval across semantic, purpose, and goal dimensions, producing purpose-aligned, explainable results.

## Context

The Teleological Retrieval Pipeline is the capstone of the Logic Layer, integrating all preceding components into a unified retrieval flow:

1. **Fast Pre-filter**: HNSW search across active embedding spaces
2. **Multi-Embedding Rerank**: Cross-space similarity aggregation
3. **Teleological Alignment**: Purpose and goal scoring
4. **Late Interaction Rerank**: Fine-grained token matching
5. **Misalignment Check**: Flag off-purpose results

This enables retrieval that balances content relevance with goal alignment, producing not just similar memories but *purposefully relevant* memories.

## Technical Specification

### Data Structures

```rust
/// Configuration for the teleological retrieval pipeline
#[derive(Clone, Debug)]
pub struct TeleologicalRetrievalConfig {
    /// Stage 1: Pre-filter configuration
    pub prefilter: PrefilterConfig,

    /// Stage 2: Multi-embedding rerank configuration
    pub rerank: RerankConfig,

    /// Stage 3: Teleological alignment configuration
    pub alignment: AlignmentStageConfig,

    /// Stage 4: Late interaction configuration
    pub late_interaction: LateInteractionConfig,

    /// Stage 5: Misalignment check configuration
    pub misalignment: MisalignmentConfig,

    /// Final result limit
    pub final_limit: usize,

    /// Whether to include full breakdown
    pub include_breakdown: bool,

    /// Timeout for entire pipeline
    pub timeout: Duration,
}

#[derive(Clone, Debug)]
pub struct PrefilterConfig {
    /// Which embedding spaces to search
    pub active_spaces: EmbeddingSpaceMask,

    /// Results per space
    pub per_space_k: usize,

    /// Minimum similarity threshold
    pub min_similarity: f32,

    /// HNSW ef_search override
    pub ef_search: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct RerankConfig {
    /// Weighting strategy for aggregation
    pub weighting_strategy: WeightingStrategy,

    /// How many to pass to next stage
    pub pass_through_k: usize,

    /// Whether to use purpose weighting
    pub use_purpose_weighting: bool,
}

#[derive(Clone, Debug)]
pub struct AlignmentStageConfig {
    /// Weight for purpose alignment in final score
    pub purpose_weight: f32,

    /// Weight for goal alignment in final score
    pub goal_weight: f32,

    /// Active goals for alignment
    pub active_goals: Vec<GoalId>,

    /// How many to pass to next stage
    pub pass_through_k: usize,
}

#[derive(Clone, Debug)]
pub struct LateInteractionConfig {
    /// Whether to enable late interaction
    pub enabled: bool,

    /// Weight in final score
    pub weight: f32,

    /// Token similarity threshold
    pub min_token_similarity: f32,
}

#[derive(Clone, Debug)]
pub struct MisalignmentConfig {
    /// Whether to flag misaligned results
    pub enabled: bool,

    /// Threshold below which to flag
    pub alignment_threshold: f32,

    /// Whether to filter out misaligned (vs just flag)
    pub filter_misaligned: bool,
}

/// Query for teleological retrieval
#[derive(Clone, Debug)]
pub struct TeleologicalQuery {
    /// Query text
    pub text: String,

    /// Pre-computed query embeddings (optional)
    pub embeddings: Option<SemanticFingerprint>,

    /// Query's purpose (optional, computed if not provided)
    pub purpose: Option<PurposeVector>,

    /// Target goals (optional)
    pub target_goals: Option<Vec<GoalId>>,

    /// Johari filter (only return specific quadrants)
    pub johari_filter: Option<Vec<JohariQuadrant>>,

    /// Configuration overrides
    pub config: Option<TeleologicalRetrievalConfig>,
}

/// Result from teleological retrieval
#[derive(Clone, Debug)]
pub struct TeleologicalRetrievalResult {
    /// Final ranked results
    pub results: Vec<ScoredMemory>,

    /// Query metadata
    pub query_metadata: QueryMetadata,

    /// Pipeline execution stats
    pub pipeline_stats: PipelineStats,

    /// Detailed breakdown (if requested)
    pub breakdown: Option<PipelineBreakdown>,
}

/// A scored memory result
#[derive(Clone, Debug)]
pub struct ScoredMemory {
    /// Memory identifier
    pub memory_id: MemoryId,

    /// Final composite score
    pub score: f32,

    /// Content similarity score
    pub content_similarity: f32,

    /// Purpose alignment score
    pub purpose_alignment: f32,

    /// Goal alignment score
    pub goal_alignment: f32,

    /// Johari quadrant
    pub johari_quadrant: JohariQuadrant,

    /// Misalignment flag
    pub is_misaligned: bool,

    /// Explanation for ranking
    pub explanation: Option<String>,
}

/// Metadata about the query
#[derive(Clone, Debug)]
pub struct QueryMetadata {
    pub computed_embeddings: bool,
    pub computed_purpose: bool,
    pub active_spaces: Vec<usize>,
    pub active_goals: Vec<GoalId>,
}

/// Statistics about pipeline execution
#[derive(Clone, Debug)]
pub struct PipelineStats {
    pub total_time_ms: f32,
    pub prefilter_time_ms: f32,
    pub rerank_time_ms: f32,
    pub alignment_time_ms: f32,
    pub late_interaction_time_ms: f32,
    pub misalignment_time_ms: f32,
    pub candidates_per_stage: [usize; 5],
}

/// Detailed breakdown per stage
#[derive(Clone, Debug)]
pub struct PipelineBreakdown {
    pub prefilter_results: Vec<(MemoryId, f32)>,
    pub rerank_results: Vec<(MemoryId, CrossSpaceSimilarity)>,
    pub alignment_results: Vec<(MemoryId, GoalAlignmentScore)>,
    pub late_interaction_results: Option<Vec<(MemoryId, f32)>>,
    pub filtered_misaligned: Vec<MemoryId>,
}
```

### Core Trait

```rust
/// Teleological retrieval pipeline
#[async_trait]
pub trait TeleologicalRetrievalPipeline: Send + Sync {
    /// Execute full retrieval pipeline
    async fn retrieve(
        &self,
        query: TeleologicalQuery,
    ) -> Result<TeleologicalRetrievalResult, RetrievalError>;

    /// Execute with streaming results
    async fn retrieve_streaming(
        &self,
        query: TeleologicalQuery,
    ) -> Result<impl Stream<Item = ScoredMemory>, RetrievalError>;

    /// Get recommended configuration for query
    fn recommend_config(
        &self,
        query: &TeleologicalQuery,
    ) -> TeleologicalRetrievalConfig;

    /// Warm up pipeline components
    async fn warm_up(&self) -> Result<(), RetrievalError>;

    /// Get pipeline health status
    fn health_status(&self) -> PipelineHealth;
}

/// Health status of the pipeline
#[derive(Clone, Debug)]
pub struct PipelineHealth {
    pub is_ready: bool,
    pub indexes_loaded: Vec<usize>,
    pub missing_components: Vec<String>,
    pub last_query_time: Option<Timestamp>,
}
```

### Implementation

```rust
pub struct DefaultTeleologicalPipeline {
    query_executor: Arc<dyn MultiEmbeddingQueryExecutor>,
    purpose_computer: Arc<dyn PurposeVectorComputer>,
    alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
    johari_manager: Arc<dyn JohariTransitionManager>,
    index_manager: Arc<dyn MultiSpaceIndexManager>,
    purpose_index: Arc<dyn PurposePatternIndex>,
    similarity_engine: Arc<dyn CrossSpaceSimilarityEngine>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    memory_store: Arc<dyn MemoryStore>,
    default_config: TeleologicalRetrievalConfig,
}

#[async_trait]
impl TeleologicalRetrievalPipeline for DefaultTeleologicalPipeline {
    async fn retrieve(
        &self,
        query: TeleologicalQuery,
    ) -> Result<TeleologicalRetrievalResult, RetrievalError> {
        let start = Instant::now();
        let config = query.config.as_ref().unwrap_or(&self.default_config);

        // Prepare query embeddings
        let query_embeddings = match query.embeddings {
            Some(ref emb) => emb.clone(),
            None => self.embedding_provider.embed(&query.text).await?,
        };

        // Prepare query purpose
        let query_purpose = match query.purpose {
            Some(ref p) => p.clone(),
            None => self.purpose_computer.compute_purpose(
                &query_embeddings,
                &PurposeComputeConfig::default(),
            ).await?,
        };

        // Stage 1: Pre-filter
        let prefilter_start = Instant::now();
        let prefilter_results = self.execute_prefilter(
            &query_embeddings,
            &config.prefilter,
        ).await?;
        let prefilter_time = prefilter_start.elapsed();

        // Stage 2: Multi-embedding Rerank
        let rerank_start = Instant::now();
        let rerank_results = self.execute_rerank(
            &query_embeddings,
            &query_purpose,
            prefilter_results,
            &config.rerank,
        ).await?;
        let rerank_time = rerank_start.elapsed();

        // Stage 3: Teleological Alignment
        let alignment_start = Instant::now();
        let alignment_results = self.execute_alignment(
            &query_purpose,
            &query.target_goals.as_ref().unwrap_or(&config.alignment.active_goals),
            rerank_results,
            &config.alignment,
        ).await?;
        let alignment_time = alignment_start.elapsed();

        // Stage 4: Late Interaction (optional)
        let late_start = Instant::now();
        let late_results = if config.late_interaction.enabled {
            Some(self.execute_late_interaction(
                &query_embeddings,
                alignment_results.clone(),
                &config.late_interaction,
            ).await?)
        } else {
            None
        };
        let late_time = late_start.elapsed();

        // Stage 5: Misalignment Check
        let misalign_start = Instant::now();
        let final_results = self.execute_misalignment_check(
            late_results.as_ref().unwrap_or(&alignment_results),
            &config.misalignment,
        ).await?;
        let misalign_time = misalign_start.elapsed();

        // Build final results
        let results = self.build_scored_memories(
            final_results,
            &query_purpose,
            query.johari_filter.as_ref(),
            config.final_limit,
        ).await?;

        Ok(TeleologicalRetrievalResult {
            results,
            query_metadata: QueryMetadata {
                computed_embeddings: query.embeddings.is_none(),
                computed_purpose: query.purpose.is_none(),
                active_spaces: config.prefilter.active_spaces.to_list(),
                active_goals: config.alignment.active_goals.clone(),
            },
            pipeline_stats: PipelineStats {
                total_time_ms: start.elapsed().as_secs_f32() * 1000.0,
                prefilter_time_ms: prefilter_time.as_secs_f32() * 1000.0,
                rerank_time_ms: rerank_time.as_secs_f32() * 1000.0,
                alignment_time_ms: alignment_time.as_secs_f32() * 1000.0,
                late_interaction_time_ms: late_time.as_secs_f32() * 1000.0,
                misalignment_time_ms: misalign_time.as_secs_f32() * 1000.0,
                candidates_per_stage: [
                    prefilter_results.len(),
                    rerank_results.len(),
                    alignment_results.len(),
                    late_results.as_ref().map(|r| r.len()).unwrap_or(0),
                    final_results.len(),
                ],
            },
            breakdown: if config.include_breakdown {
                Some(PipelineBreakdown { /* ... */ })
            } else {
                None
            },
        })
    }
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-L001 complete (Multi-Embedding Query Executor)
- [ ] TASK-L002 complete (Purpose Vector Computation)
- [ ] TASK-L003 complete (Goal Alignment Calculator)
- [ ] TASK-L004 complete (Johari Transition Manager)
- [ ] TASK-L005 complete (Per-Space HNSW Index Builder)
- [ ] TASK-L006 complete (Purpose Pattern Index)
- [ ] TASK-L007 complete (Cross-Space Similarity Engine)

### Scope

#### In Scope

- 5-stage retrieval pipeline
- Configuration for each stage
- Pipeline orchestration
- Result aggregation and scoring
- Streaming results
- Pipeline health monitoring

#### Out of Scope

- Individual component implementations (prior tasks)
- UI/API layer (Surface Layer)
- Learning from feedback (future enhancement)

### Constraints

- End-to-end latency < 100ms for typical queries
- Support up to 100K candidates in pre-filter
- Memory efficient (no full materialization until needed)
- Thread-safe and parallelizable

## Pseudo Code

```
FUNCTION retrieve(query):
    config = query.config OR default_config
    start = now()

    // ===== PREPARE QUERY =====
    IF query.embeddings IS NULL:
        query_embeddings = embedding_provider.embed(query.text)
    ELSE:
        query_embeddings = query.embeddings

    IF query.purpose IS NULL:
        query_purpose = purpose_computer.compute(query_embeddings)
    ELSE:
        query_purpose = query.purpose

    // ===== STAGE 1: PRE-FILTER =====
    // Fast HNSW search across active embedding spaces
    prefilter_candidates = []

    FOR space_idx IN config.prefilter.active_spaces:
        query_vector = query_embeddings.embeddings[space_idx]
        IF query_vector IS NOT NULL:
            space_results = index_manager.search(
                space_idx,
                query_vector,
                config.prefilter.per_space_k,
                config.prefilter.ef_search
            )
            prefilter_candidates.extend(space_results)

    // Deduplicate by memory_id, keeping highest similarity
    prefilter_candidates = deduplicate_by_id(prefilter_candidates)

    // ===== STAGE 2: MULTI-EMBEDDING RERANK =====
    // Compute cross-space similarity for all candidates
    rerank_results = []

    FOR (memory_id, _) IN prefilter_candidates:
        memory_fingerprint = memory_store.get_fingerprint(memory_id)

        similarity = similarity_engine.compute_similarity(
            query_embeddings,
            memory_fingerprint.semantic_fingerprint,
            CrossSpaceConfig {
                weighting_strategy: config.rerank.weighting_strategy,
                use_purpose_weighting: config.rerank.use_purpose_weighting,
                ...
            }
        )

        rerank_results.push((memory_id, similarity))

    // Sort by cross-space similarity
    rerank_results.sort_by(|(_, s1), (_, s2)| s2.score.cmp(s1.score))
    rerank_results.truncate(config.rerank.pass_through_k)

    // ===== STAGE 3: TELEOLOGICAL ALIGNMENT =====
    // Score by purpose and goal alignment
    alignment_results = []

    FOR (memory_id, cross_sim) IN rerank_results:
        memory_fingerprint = memory_store.get_fingerprint(memory_id)

        // Purpose alignment
        purpose_sim = cosine_similarity(
            query_purpose.alignment,
            memory_fingerprint.purpose_vector.alignment
        )

        // Goal alignment
        goal_score = alignment_calculator.calculate(
            memory_fingerprint,
            config.alignment.active_goals
        )

        // Combined score
        combined = cross_sim.score * (1.0 - config.alignment.purpose_weight - config.alignment.goal_weight)
                 + purpose_sim * config.alignment.purpose_weight
                 + goal_score.composite_score * config.alignment.goal_weight

        alignment_results.push((memory_id, combined, purpose_sim, goal_score))

    // Sort by combined score
    alignment_results.sort_by_key(|r| -r.1)
    alignment_results.truncate(config.alignment.pass_through_k)

    // ===== STAGE 4: LATE INTERACTION (Optional) =====
    IF config.late_interaction.enabled:
        late_results = []
        query_tokens = query_embeddings.embeddings[9]  // E10: late interaction

        FOR (memory_id, combined, purpose_sim, goal_score) IN alignment_results:
            memory_fingerprint = memory_store.get_fingerprint(memory_id)
            memory_tokens = memory_fingerprint.semantic_fingerprint.embeddings[9]

            IF query_tokens IS NOT NULL AND memory_tokens IS NOT NULL:
                // MaxSim computation
                max_sim_score = compute_max_sim(query_tokens, memory_tokens)

                // Blend with previous score
                final_score = combined * (1.0 - config.late_interaction.weight)
                            + max_sim_score * config.late_interaction.weight
            ELSE:
                final_score = combined

            late_results.push((memory_id, final_score, purpose_sim, goal_score))

        working_results = late_results
    ELSE:
        working_results = alignment_results

    // ===== STAGE 5: MISALIGNMENT CHECK =====
    final_results = []
    filtered_misaligned = []

    FOR (memory_id, score, purpose_sim, goal_score) IN working_results:
        is_misaligned = goal_score.composite_score < config.misalignment.alignment_threshold
                     OR goal_score.misalignment_flags.any_set()

        IF is_misaligned AND config.misalignment.filter_misaligned:
            filtered_misaligned.push(memory_id)
            CONTINUE

        // Get Johari quadrant
        johari = johari_manager.get_johari(memory_id)
        dominant_quadrant = get_dominant_quadrant(johari)

        // Apply Johari filter if specified
        IF query.johari_filter IS NOT NULL:
            IF dominant_quadrant NOT IN query.johari_filter:
                CONTINUE

        final_results.push(ScoredMemory {
            memory_id,
            score,
            content_similarity: ...,
            purpose_alignment: purpose_sim,
            goal_alignment: goal_score.composite_score,
            johari_quadrant: dominant_quadrant,
            is_misaligned,
            explanation: generate_explanation(...)
        })

    // Final sort and limit
    final_results.sort_by_key(|m| -m.score)
    final_results.truncate(config.final_limit)

    RETURN TeleologicalRetrievalResult {
        results: final_results,
        query_metadata: ...,
        pipeline_stats: ...,
        breakdown: IF config.include_breakdown THEN Some(...) ELSE None
    }
```

## Definition of Done

### Implementation Checklist

- [ ] `TeleologicalRetrievalConfig` with all stage configs
- [ ] `TeleologicalQuery` with all query options
- [ ] `TeleologicalRetrievalResult` with stats and breakdown
- [ ] `ScoredMemory` with all scoring components
- [ ] `TeleologicalRetrievalPipeline` trait
- [ ] Default implementation with 5 stages
- [ ] Pre-filter stage (multi-space HNSW)
- [ ] Rerank stage (cross-space similarity)
- [ ] Alignment stage (purpose + goal)
- [ ] Late interaction stage (MaxSim)
- [ ] Misalignment check stage
- [ ] Pipeline health monitoring
- [ ] Streaming results support

### Testing Requirements

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_full_pipeline() {
        let pipeline = create_test_pipeline().await;
        populate_test_data(&pipeline, 1000).await;

        let query = TeleologicalQuery {
            text: "How to implement authentication?".into(),
            embeddings: None,
            purpose: None,
            target_goals: None,
            johari_filter: None,
            config: None,
        };

        let result = pipeline.retrieve(query).await.unwrap();

        assert!(!result.results.is_empty());
        assert!(result.results.len() <= DEFAULT_FINAL_LIMIT);
        assert!(result.pipeline_stats.total_time_ms < 100.0);
    }

    #[tokio::test]
    async fn test_purpose_weighted_retrieval() {
        let pipeline = create_test_pipeline().await;
        populate_test_data(&pipeline, 1000).await;

        let query = TeleologicalQuery {
            text: "authentication".into(),
            purpose: Some(create_security_focused_purpose()),
            ..Default::default()
        };

        let result = pipeline.retrieve(query).await.unwrap();

        // Results should favor security-aligned memories
        let avg_purpose_alignment: f32 = result.results.iter()
            .map(|r| r.purpose_alignment)
            .sum::<f32>() / result.results.len() as f32;

        assert!(avg_purpose_alignment > 0.5);
    }

    #[tokio::test]
    async fn test_goal_filtering() {
        let pipeline = create_test_pipeline().await;
        populate_test_data(&pipeline, 1000).await;

        let query = TeleologicalQuery {
            text: "implementation".into(),
            target_goals: Some(vec![GoalId("security".into())]),
            ..Default::default()
        };

        let config = TeleologicalRetrievalConfig {
            misalignment: MisalignmentConfig {
                enabled: true,
                alignment_threshold: 0.3,
                filter_misaligned: true,
            },
            ..Default::default()
        };

        let result = pipeline.retrieve(TeleologicalQuery {
            config: Some(config),
            ..query
        }).await.unwrap();

        // No misaligned results should be present
        for r in &result.results {
            assert!(!r.is_misaligned);
            assert!(r.goal_alignment >= 0.3);
        }
    }

    #[tokio::test]
    async fn test_johari_filter() {
        let pipeline = create_test_pipeline().await;
        populate_test_data(&pipeline, 1000).await;

        let query = TeleologicalQuery {
            text: "test query".into(),
            johari_filter: Some(vec![JohariQuadrant::Open]),
            ..Default::default()
        };

        let result = pipeline.retrieve(query).await.unwrap();

        for r in &result.results {
            assert_eq!(r.johari_quadrant, JohariQuadrant::Open);
        }
    }

    #[tokio::test]
    async fn test_pipeline_stages_timing() {
        let pipeline = create_test_pipeline().await;
        populate_test_data(&pipeline, 10000).await;

        let query = TeleologicalQuery {
            text: "complex query with many matches".into(),
            ..Default::default()
        };

        let config = TeleologicalRetrievalConfig {
            include_breakdown: true,
            ..Default::default()
        };

        let result = pipeline.retrieve(TeleologicalQuery {
            config: Some(config),
            ..query
        }).await.unwrap();

        // Verify all stages executed
        assert!(result.pipeline_stats.prefilter_time_ms > 0.0);
        assert!(result.pipeline_stats.rerank_time_ms > 0.0);
        assert!(result.pipeline_stats.alignment_time_ms > 0.0);

        // Verify breakdown present
        assert!(result.breakdown.is_some());
    }

    #[tokio::test]
    async fn test_late_interaction_toggle() {
        let pipeline = create_test_pipeline().await;
        populate_test_data(&pipeline, 100).await;

        let query = TeleologicalQuery {
            text: "test".into(),
            ..Default::default()
        };

        let config_with = TeleologicalRetrievalConfig {
            late_interaction: LateInteractionConfig {
                enabled: true,
                weight: 0.3,
                min_token_similarity: 0.5,
            },
            ..Default::default()
        };

        let config_without = TeleologicalRetrievalConfig {
            late_interaction: LateInteractionConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let result_with = pipeline.retrieve(TeleologicalQuery {
            config: Some(config_with),
            ..query.clone()
        }).await.unwrap();

        let result_without = pipeline.retrieve(TeleologicalQuery {
            config: Some(config_without),
            ..query
        }).await.unwrap();

        // Late interaction should affect timing
        assert!(result_with.pipeline_stats.late_interaction_time_ms > 0.0);
        assert_eq!(result_without.pipeline_stats.late_interaction_time_ms, 0.0);
    }
}
```

### Verification Commands

```bash
# Run unit tests
cargo test -p context-graph-core teleological_retrieval

# Run integration tests
cargo test -p context-graph-core --features integration retrieval_pipeline

# Benchmark full pipeline
cargo bench -p context-graph-core -- teleological_pipeline

# Load test
cargo test -p context-graph-core retrieval_load_test -- --ignored --nocapture
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-core/src/retrieval/pipeline.rs` | TeleologicalRetrievalPipeline trait and impl |
| `crates/context-graph-core/src/retrieval/config.rs` | All configuration structs |
| `crates/context-graph-core/src/retrieval/query.rs` | TeleologicalQuery |
| `crates/context-graph-core/src/retrieval/result.rs` | Result types |
| `crates/context-graph-core/src/retrieval/stages/mod.rs` | Stage implementations |
| `crates/context-graph-core/src/retrieval/stages/prefilter.rs` | Pre-filter stage |
| `crates/context-graph-core/src/retrieval/stages/rerank.rs` | Rerank stage |
| `crates/context-graph-core/src/retrieval/stages/alignment.rs` | Alignment stage |
| `crates/context-graph-core/src/retrieval/stages/late_interaction.rs` | Late interaction stage |
| `crates/context-graph-core/src/retrieval/stages/misalignment.rs` | Misalignment check stage |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-core/src/retrieval/mod.rs` | Add pipeline exports |
| `crates/context-graph-core/src/lib.rs` | Re-export pipeline |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| 5-stage pipeline | projectionplan1.md:retrieval | Complete |
| Purpose alignment | projectionplan1.md:purpose | Complete |
| Goal alignment | projectionplan2.md:goals | Complete |
| Late interaction | projectionplan2.md:colbert | Complete |
| Misalignment check | projectionplan2.md:drift | Complete |
| Johari filtering | projectionplan1.md:johari | Complete |

---

*Task created: 2026-01-04*
*Layer: Logic*
*Priority: P0 - Capstone integration*
