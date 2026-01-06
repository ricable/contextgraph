# TASK-L008: Teleological Retrieval Pipeline

```yaml
metadata:
  id: "TASK-L008"
  title: "Teleological Retrieval Pipeline"
  layer: "logic"
  priority: "P0"
  estimated_hours: 3  # Reduced - core done, integration/testing remains
  created: "2026-01-04"
  updated: "2026-01-05"
  status: "in_progress"  # Core implementation exists, needs integration
  dependencies:
    - "TASK-L001"  # Multi-Embedding Query Executor [COMPLETED]
    - "TASK-L002"  # Purpose Vector Computation [COMPLETED]
    - "TASK-L003"  # Goal Alignment Calculator [COMPLETED]
    - "TASK-L004"  # Johari Transition Manager [COMPLETED]
    - "TASK-L005"  # Per-Space HNSW Index Builder [COMPLETED]
    - "TASK-L006"  # Purpose Pattern Index [COMPLETED]
    - "TASK-L007"  # Cross-Space Similarity Engine [COMPLETED]
  spec_refs:
    - "constitution.yaml:retrieval-pipeline"
    - "constitution.yaml:performance-targets"
```

---

## CURRENT STATE (2026-01-05)

### FILES ALREADY CREATED

These files exist and contain functional implementations:

| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `crates/context-graph-core/src/retrieval/pipeline.rs` | **EXISTS** | 854 | `TeleologicalRetrievalPipeline` trait + `DefaultTeleologicalPipeline` |
| `crates/context-graph-core/src/retrieval/teleological_query.rs` | **EXISTS** | 467 | `TeleologicalQuery` with full validation |
| `crates/context-graph-core/src/retrieval/teleological_result.rs` | **EXISTS** | 300+ | `TeleologicalRetrievalResult`, `ScoredMemory`, `PipelineBreakdown` |
| `crates/context-graph-core/src/retrieval/mod.rs` | **MODIFIED** | - | Exports added for new modules |
| `crates/context-graph-core/src/retrieval/query.rs` | **MODIFIED** | - | `PipelineStageConfig::validate()` added |
| `crates/context-graph-core/src/retrieval/result.rs` | **MODIFIED** | - | Serialization added to `PipelineStageTiming` |

### WHAT'S DONE

1. **`TeleologicalRetrievalPipeline` trait** (pipeline.rs:67-104)
   - `execute(&self, query: &TeleologicalQuery) -> CoreResult<TeleologicalRetrievalResult>`
   - `filter_by_alignment(&self, candidates, query) -> CoreResult<Vec<ScoredMemory>>`
   - `health_check(&self) -> CoreResult<PipelineHealth>`

2. **`DefaultTeleologicalPipeline<E, A, J>`** (pipeline.rs:125-638)
   - Generic over executor, alignment calculator, johari manager
   - Stage 4 filtering implemented (`apply_stage4_filtering`)
   - Dominant quadrant computation
   - Builder pattern with `with_goal_hierarchy()`

3. **`TeleologicalQuery`** (teleological_query.rs:46-290)
   - All fields: text, embeddings, purpose, target_goals, johari_filter, pipeline_config, include_breakdown
   - `validate()` with FAIL FAST errors
   - Builder methods: `from_text()`, `from_embeddings()`, `with_*()` chain

4. **`TeleologicalRetrievalResult`** (teleological_result.rs)
   - `results: Vec<ScoredMemory>`
   - `timing: PipelineStageTiming`
   - `breakdown: Option<PipelineBreakdown>`
   - `within_latency_target()`, `misaligned_count()`

5. **`ScoredMemory`** (teleological_result.rs)
   - All fields: memory_id, score, content_similarity, purpose_alignment, goal_alignment, johari_quadrant, is_misaligned, space_count
   - `alignment_threshold() -> AlignmentLevel`

6. **Unit tests in each file** - All pass

### WHAT REMAINS (THE ACTUAL TASK)

The implementation has a **placeholder** in `DefaultTeleologicalPipeline::execute()`:

```rust
// pipeline.rs:564-606 - PLACEHOLDER STAGE 4
async fn stage4_placeholder_filtering(
    &self,
    me_result: &MultiEmbeddingResult,
    query: &TeleologicalQuery,
    config: &super::PipelineStageConfig,
) -> Vec<ScoredMemory> {
    // Creates ScoredMemory with ESTIMATED alignments:
    // - purpose_alignment = agg.purpose_alignment.unwrap_or(agg.aggregate_score)
    // - goal_alignment = content_sim * 0.9  // ← PLACEHOLDER ESTIMATION
    // - johari_quadrant = JohariQuadrant::Open  // ← DEFAULT, NOT COMPUTED
}
```

**The task is to replace this placeholder with real integration:**

1. **Fetch actual TeleologicalFingerprints** from store for each candidate
2. **Compute real goal alignment** using `GoalAlignmentCalculator`
3. **Get real Johari quadrant** using `JohariTransitionManager`
4. **Wire up to actual data store** (not stub)

---

## Source of Truth

**constitution.yaml** is the canonical source for all constants.

```yaml
# From constitution.yaml - MUST match these exactly
retrieval_pipeline:
  total_latency_ms: 60    # HARD LIMIT @ 1M memories
  stages:
    stage1_splade_ms: 5
    stage2_matryoshka_ms: 10
    stage3_full_hnsw_ms: 20
    stage4_teleological_ms: 10
    stage5_late_interaction_ms: 15
  rrf_k: 60
  min_alignment_threshold: 0.55

alignment_thresholds:
  optimal: 0.75
  acceptable_min: 0.70
  warning_min: 0.55
  critical: 0.55
```

---

## REMAINING IMPLEMENTATION

### 1. Replace Placeholder Stage 4 (pipeline.rs:564-606)

The `stage4_placeholder_filtering` must be replaced with:

```rust
async fn apply_real_stage4_filtering(
    &self,
    candidates: &[AggregatedMatch],
    query: &TeleologicalQuery,
    config: &PipelineStageConfig,
) -> CoreResult<Vec<ScoredMemory>> {
    let mut results = Vec::with_capacity(candidates.len());

    for agg in candidates {
        // 1. FETCH REAL FINGERPRINT from store
        let fingerprint = self.store
            .get_by_id(agg.memory_id)
            .await?
            .ok_or_else(|| CoreError::NotFound {
                entity: "TeleologicalFingerprint".to_string(),
                id: agg.memory_id.to_string(),
            })?;

        // 2. COMPUTE REAL GOAL ALIGNMENT using L003
        let alignment_result = self.alignment_calculator
            .compute_alignment(&fingerprint, &alignment_config)
            .await?;

        // 3. GET REAL JOHARI QUADRANT from fingerprint
        let johari_quadrant = fingerprint.johari.dominant_quadrant_overall();

        // 4. Filter by threshold - FAIL FAST, no silent drops
        if alignment_result.score.composite_score < config.min_alignment_threshold {
            tracing::debug!(
                target: "pipeline",
                memory_id = %agg.memory_id,
                alignment = alignment_result.score.composite_score,
                threshold = config.min_alignment_threshold,
                "Filtered by alignment threshold"
            );
            continue;
        }

        // 5. Apply Johari filter if specified
        if let Some(ref filter) = query.johari_filter {
            if !filter.contains(&johari_quadrant) {
                continue;
            }
        }

        results.push(ScoredMemory::new(
            agg.memory_id,
            agg.aggregate_score,
            self.compute_avg_similarity(agg),
            fingerprint.theta_to_north_star,
            alignment_result.score.composite_score,
            johari_quadrant,
            agg.space_count,
        ));
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    if results.len() > config.teleological_limit {
        results.truncate(config.teleological_limit);
    }

    Ok(results)
}
```

### 2. Update `DefaultTeleologicalPipeline` to Accept Store

Current:
```rust
pub struct DefaultTeleologicalPipeline<E, A, J>
```

Change to:
```rust
pub struct DefaultTeleologicalPipeline<E, A, J, S>
where
    E: MultiEmbeddingQueryExecutor,
    A: GoalAlignmentCalculator,
    J: JohariTransitionManager,
    S: TeleologicalMemoryStore,  // NEW
{
    // ... existing fields
    store: Arc<S>,  // ADD THIS
}
```

### 3. Wire Up in `execute()`

Replace line ~416:
```rust
// OLD:
let stage4_results = self.stage4_placeholder_filtering(&me_result, query, &config).await;

// NEW:
let (stage4_results, filtered_count, filtered_avg) = self
    .apply_real_stage4_filtering(&me_result.results, query, &config)
    .await?;
```

---

## Existing Code References (EXACT LOCATIONS)

### Types You Need
```rust
// crates/context-graph-core/src/traits/storage.rs
pub trait TeleologicalMemoryStore: Send + Sync {
    async fn get_by_id(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>>;
    // ...
}

// crates/context-graph-core/src/alignment/calculator.rs
pub trait GoalAlignmentCalculator: Send + Sync {
    async fn compute_alignment(
        &self,
        fingerprint: &TeleologicalFingerprint,
        config: &AlignmentConfig,
    ) -> CoreResult<AlignmentResult>;
}

// crates/context-graph-core/src/types/fingerprint.rs
impl TeleologicalFingerprint {
    pub theta_to_north_star: f32;  // Purpose alignment
    pub johari: JohariClassification;  // Per-space Johari
}

// crates/context-graph-core/src/types/johari.rs
impl JohariClassification {
    pub fn dominant_quadrant_overall(&self) -> JohariQuadrant;
    pub fn dominant_quadrant(&self, space_index: usize) -> JohariQuadrant;
}
```

### Stubs Available for Testing
```rust
// crates/context-graph-core/src/stubs/mod.rs
pub struct InMemoryTeleologicalStore;  // For tests
pub struct StubMultiArrayProvider;     // For tests
```

---

## FAIL FAST Requirements

**NO BACKWARDS COMPATIBILITY.** If something fails:

1. **Return error immediately** - No fallbacks
2. **Log with full context** - memory_id, stage, threshold, actual value
3. **Propagate all errors** - No swallowing
4. **No mock data in tests** - Use `InMemoryTeleologicalStore` with real fingerprints

```rust
// WRONG - Silent fallback
let alignment = alignment_result.unwrap_or(0.5);

// CORRECT - Fail fast
let alignment = alignment_result.map_err(|e| {
    tracing::error!(
        target: "pipeline",
        memory_id = %id,
        error = %e,
        "Goal alignment computation failed"
    );
    e
})?;
```

---

## Testing Requirements

### Run Existing Tests First
```bash
# These should PASS with current implementation
cargo test -p context-graph-core --lib pipeline -- --nocapture
cargo test -p context-graph-core --lib teleological_query -- --nocapture
cargo test -p context-graph-core --lib teleological_result -- --nocapture
```

### After Your Changes - Integration Tests
```bash
# Full pipeline with real store
cargo test -p context-graph-core test_execute_with_real_store -- --nocapture

# Verify alignment filtering works
cargo test -p context-graph-core test_real_alignment_filtering -- --nocapture
```

### New Test to Add (integration)
```rust
#[tokio::test]
async fn test_execute_with_real_store_integration() {
    // 1. Create store with real fingerprints
    let store = InMemoryTeleologicalStore::new();

    // 2. Populate with diverse alignment data
    let high_align_fp = TeleologicalFingerprint::builder()
        .theta_to_north_star(0.85)
        .build();
    let low_align_fp = TeleologicalFingerprint::builder()
        .theta_to_north_star(0.40)
        .build();

    store.insert(high_align_fp.id, high_align_fp.clone()).await?;
    store.insert(low_align_fp.id, low_align_fp.clone()).await?;

    // 3. Create pipeline with real store
    let pipeline = DefaultTeleologicalPipeline::new(
        executor,
        alignment_calc,
        johari_manager,
        Arc::new(store),  // REAL STORE
        hierarchy,
    );

    // 4. Query with alignment filter
    let query = TeleologicalQuery::from_text("test")
        .with_min_alignment(0.55);

    let result = pipeline.execute(&query).await?;

    // 5. VERIFY: High alignment included, low alignment filtered
    assert!(result.results.iter().any(|r| r.memory_id == high_align_fp.id));
    assert!(!result.results.iter().any(|r| r.memory_id == low_align_fp.id));

    // 6. EVIDENCE: Print actual state
    println!("BEFORE: 2 memories (alignment 0.85, 0.40)");
    println!("AFTER: {} results", result.results.len());
    for r in &result.results {
        println!("  - {} alignment={:.2}", r.memory_id, r.goal_alignment);
    }
    println!("[VERIFIED] Only high-alignment memory returned");
}
```

---

## Full State Verification (REQUIRED)

After implementation, you MUST verify:

### 1. Source of Truth Check
```bash
# Verify constants match constitution.yaml
grep -n "min_alignment_threshold" crates/context-graph-core/src/retrieval/query.rs
# Should show: default: 0.55

grep -n "rrf_k" crates/context-graph-core/src/retrieval/query.rs
# Should show: default: 60.0
```

### 2. Execute & Inspect
```bash
# Run with tracing to see actual flow
RUST_LOG=pipeline=debug cargo test -p context-graph-core test_execute_basic_query -- --nocapture 2>&1 | grep -E "(BEFORE|AFTER|VERIFIED)"
```

### 3. Edge Case Audit (3 cases)

**Case 1: Empty Index**
```rust
#[tokio::test]
async fn test_edge_empty_index() {
    let store = InMemoryTeleologicalStore::new();  // EMPTY
    let pipeline = create_pipeline_with_store(store);

    println!("BEFORE: store.count() = 0");

    let query = TeleologicalQuery::from_text("anything");
    let result = pipeline.execute(&query).await?;

    println!("AFTER: results.len() = {}", result.results.len());
    assert!(result.results.is_empty());
    println!("[VERIFIED] Empty index returns empty results");
}
```

**Case 2: All Below Threshold**
```rust
#[tokio::test]
async fn test_edge_all_filtered() {
    let store = populate_low_alignment_only(0.30);  // All at 0.30
    let pipeline = create_pipeline_with_store(store);

    println!("BEFORE: 100 memories all at alignment=0.30");

    let query = TeleologicalQuery::from_text("test")
        .with_min_alignment(0.55);  // Threshold above all
    let result = pipeline.execute(&query).await?;

    println!("AFTER: results.len() = {}", result.results.len());
    assert!(result.results.is_empty());
    println!("[VERIFIED] All filtered when below threshold");
}
```

**Case 3: Exactly At Threshold**
```rust
#[tokio::test]
async fn test_edge_at_threshold() {
    let store = InMemoryTeleologicalStore::new();
    let fp = TeleologicalFingerprint::builder()
        .theta_to_north_star(0.55)  // EXACTLY at threshold
        .build();
    store.insert(fp.id, fp.clone()).await?;

    println!("BEFORE: 1 memory at alignment=0.55");

    let query = TeleologicalQuery::from_text("test")
        .with_min_alignment(0.55);
    let result = pipeline.execute(&query).await?;

    println!("AFTER: results.len() = {}", result.results.len());
    assert_eq!(result.results.len(), 1, "Boundary case: >= should include");
    println!("[VERIFIED] Exactly-at-threshold is included");
}
```

### 4. Evidence of Success
```
After all tests pass, output should show:
- [VERIFIED] markers for each test
- Actual data counts (not mocked)
- Timing measurements
- No tracing::error! in normal operation
```

---

## Definition of Done

- [ ] `stage4_placeholder_filtering` replaced with real implementation
- [ ] `DefaultTeleologicalPipeline` accepts `TeleologicalMemoryStore`
- [ ] Real `GoalAlignmentCalculator::compute_alignment()` called
- [ ] Real `JohariClassification::dominant_quadrant_overall()` used
- [ ] All existing tests still pass
- [ ] New integration test with real store passes
- [ ] 3 edge case tests pass with printed evidence
- [ ] `cargo test -p context-graph-core teleological` all green
- [ ] No `unwrap()` in implementation code
- [ ] All errors have `tracing::error!` with context

---

## Verification Commands

```bash
# 1. Check existing tests pass
cargo test -p context-graph-core --lib pipeline
cargo test -p context-graph-core --lib teleological

# 2. Run with detailed output
RUST_LOG=pipeline=debug cargo test -p context-graph-core teleological -- --nocapture

# 3. Verify no compilation errors
cargo check -p context-graph-core

# 4. Check for unwrap() in new code
grep -n "unwrap()" crates/context-graph-core/src/retrieval/pipeline.rs
# Should return nothing (or only in tests)
```

---

*Task updated: 2026-01-05*
*Layer: Logic*
*Priority: P0 - Capstone integration*
*Status: IN_PROGRESS - Core done, store integration remaining*
