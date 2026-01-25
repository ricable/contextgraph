# Remaining Embedder Integration Plan

**Version**: 1.0.0
**Date**: 2026-01-24
**Status**: Draft
**Target Embedders**: E2, E3, E9, E12, E13

---

## Executive Summary

Five embedders have implementations but lack full integration into the Context Graph system:

| Embedder | Name | Status | Priority |
|----------|------|--------|----------|
| **E12** | V_precision (ColBERT) | MaxSim implemented, Stage 5 not wired | **Critical** |
| **E13** | V_keyword_precision (SPLADE) | Inverted index exists, Stage 1 not wired | **Critical** |
| **E2** | V_freshness | Model exists, no post-retrieval boost | Medium |
| **E3** | V_periodicity | Model exists, no post-retrieval boost | Medium |
| **E9** | V_robustness (HDC) | Model exists, no MCP tools or benchmarks | Low |

**Critical Path**: E12 and E13 are required for the 5-stage retrieval pipeline per constitution:
```
S1: E13 SPLADE → 10K candidates
S2: E1 Matryoshka → 1K candidates
S3: RRF fusion → 100 candidates
S4: Topic alignment → 50 candidates
S5: E12 MaxSim → 10 final results
```

---

## 1. How Integrated Embedders Work (Reference Patterns)

### 1.1 E10 Intent (Multiplicative Boost Pattern)

E10 demonstrates the **enhancer pattern** - it doesn't replace E1 but boosts it:

```rust
// From intent_tools.rs - ARCH-17 compliant
let e1_score = search_result.similarity;
let e10_alignment = compute_intent_alignment(query, memory);

// Adaptive boost based on E1 strength (ARCH-29)
let boost = match e1_score {
    s if s > 0.8 => 0.05,  // Strong E1 = light refinement
    s if s > 0.4 => 0.10,  // Medium E1 = balanced
    _ => 0.15,             // Weak E1 = broaden search
};

// Multiplicative boost (ARCH-28), clamped [0.8, 1.2] (ARCH-33)
let multiplier = (1.0 + boost * (e10_alignment - 0.5)).clamp(0.8, 1.2);
let final_score = e1_score * multiplier;
```

**Key Insight**: E10 enhances E1, never overrides. When E1=0, result=0 (AP-84).

### 1.2 E5 Causal (Asymmetric Similarity Pattern)

E5 demonstrates **directional similarity** - cause→effect is different from effect→cause:

```rust
// From causal_tools.rs - asymmetric per AP-77
let base_similarity = cosine_similarity(query_e5, memory_e5);

// Direction modifiers (ARCH-18)
let modifier = match detected_direction {
    CausalDirection::CauseToEffect => 1.2,  // "What did X cause?"
    CausalDirection::EffectToCause => 0.8,  // "What caused X?"
    CausalDirection::Unknown => 1.0,
};

let causal_score = base_similarity * modifier;
```

### 1.3 E11 Entity (TransE Knowledge Graph Pattern)

E11 uses **knowledge graph embeddings** for entity relationships:

```rust
// From entity_tools.rs - TransE scoring
// score = -||h + r - t||  where h=head, r=relation, t=tail
let head_embedding = kepler.embed_entity(head);
let tail_embedding = kepler.embed_entity(tail);
let relation_vector = infer_relation_vector(head, tail);

let transe_score = -(head_embedding + relation_vector - tail_embedding).norm();
// Valid: > -2.0, Uncertain: -2.0 to -5.0, Invalid: < -5.0
```

### 1.4 E7 Code (Separate Pipeline Pattern)

E7 demonstrates **separate storage** for specialized content:

```rust
// Code is stored separately from teleological memories (ARCH-CODE-01)
// E7 is PRIMARY for code, not part of 13-embedder array (ARCH-CODE-02)

// AST-aware chunking with tree-sitter (ARCH-CODE-03)
let code_entities = treesitter.parse_file(file)?;
for entity in code_entities {
    let e7_embedding = qodo_model.embed_code(&entity.source)?;
    code_store.store_entity(entity, e7_embedding)?;
}

// Code search uses E7-first with optional E1 blend
let e7_results = code_store.search_e7(query_embedding, top_k)?;
let blended = blend_with_e1(e7_results, e1_results, blend_weight);
```

### 1.5 Enrichment Pipeline (RRF Fusion Pattern)

The enrichment pipeline shows how **multiple embedders combine**:

```rust
// From enrichment_pipeline.rs - Weighted RRF (ARCH-21)
// Step 1: E1 Foundation (ARCH-12)
let e1_results = search_e1(query_fingerprint).await?;

// Step 2: Parallel Enhancers
let (e5_results, e7_results, e11_results) = tokio::join!(
    search_e5_if_causal(query),
    search_e7_if_code(query),
    search_e11_if_entity(query),
);

// Step 3: Weighted RRF Fusion (not weighted sum - AP-79)
const RRF_K: f32 = 60.0;
for (rank, result) in e1_results.iter().enumerate() {
    let e1_rrf = 1.0 / (RRF_K + rank as f32);
    rrf_scores[result.id] += e1_weight * e1_rrf;
}
// Repeat for each enhancer...
```

---

## 2. E12 ColBERT Integration (Stage 5 Reranking)

### 2.1 Current State

- **Model**: `crates/context-graph-embeddings/src/models/pretrained/late_interaction/`
  - GPU attention, encoder, and scoring implemented
  - 128D per-token embeddings
- **MaxSim**: `crates/context-graph-storage/src/teleological/search/maxsim.rs`
  - SIMD-optimized (AVX2) MaxSim computation
  - Target: Score 50 candidates in <15ms
- **Pipeline Stage**: `stages.rs` has Stage 4 MaxSim stub
- **Token Storage**: Schema exists in `token_storage.rs`

### 2.2 Missing Integration

1. **MCP Tool**: No `rerank_tools.rs` for direct ColBERT reranking
2. **Pipeline Wiring**: Stage 5 not called in `search_graph`
3. **Token Generation**: Embeddings generated but not stored on ingest
4. **Benchmark**: No E12 benchmark suite

### 2.3 Integration Tasks

#### Task E12-1: Token Embedding Storage on Ingest
```rust
// In store_memory flow, after TeleologicalArray generation:
if config.enable_e12_tokens {
    let tokens = colbert_model.embed_tokens(&content)?;
    token_storage.store(memory_id, tokens)?;
}
```

**Files to modify**:
- `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`
- `crates/context-graph-storage/src/teleological/search/token_storage.rs`

#### Task E12-2: Wire Stage 5 in Pipeline Search
```rust
// In search_graph with strategy=Pipeline:
let stage4_results = rrf_fusion(stage3_results)?;

// Stage 5: E12 MaxSim Rerank (per constitution)
if config.enable_maxsim_rerank {
    let query_tokens = colbert_model.embed_tokens(&query)?;
    let final_results = maxsim_rerank(
        stage4_results.take(50),  // Rerank top 50
        query_tokens,
        token_storage,
        top_k,  // Return top 10
    )?;
    return Ok(final_results);
}
```

**Files to modify**:
- `crates/context-graph-storage/src/teleological/search/pipeline/execution.rs`
- `crates/context-graph-mcp/src/handlers/tools/search_tools.rs`

#### Task E12-3: MCP Rerank Tool
```rust
// New tool: rerank_with_precision
pub async fn handle_rerank_with_precision(
    params: RerankParams,
    store: &TeleologicalStore,
) -> Result<RerankResponse> {
    // 1. Get candidate IDs from previous search
    // 2. Load token embeddings for candidates
    // 3. Compute MaxSim scores
    // 4. Return reranked results with token-level matches
}
```

**New files**:
- `crates/context-graph-mcp/src/handlers/tools/rerank_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/rerank_dtos.rs`

#### Task E12-4: Benchmark Suite
```rust
// Benchmark: E12 precision vs E1-only
// Metrics: MRR@10, P@1, NDCG@10
// Test cases: Exact phrase matches, multi-word queries
```

**New file**: `crates/context-graph-benchmark/src/bin/e12_precision_bench.rs`

### 2.4 Architectural Constraints

- **AP-74**: E12 is reranking ONLY, never initial retrieval
- **ARCH-21**: MaxSim scores integrate via RRF in multi-stage pipeline
- **Performance**: <15ms for 50 candidates

---

## 3. E13 SPLADE Integration (Stage 1 Recall)

### 3.1 Current State

- **Model**: `crates/context-graph-embeddings/src/models/pretrained/sparse/`
  - SPLADE model with MLM head
  - ~30K sparse vocabulary dimensions
- **Inverted Index**: Exists in `rocksdb_store/inverted_index.rs`
- **Stage 1**: Implemented in `stages.rs` (stage_splade_filter)

### 3.2 Missing Integration

1. **Pipeline Wiring**: Stage 1 not called by default in `search_graph`
2. **Sparse Embedding on Ingest**: SPLADE vectors not generated for memories
3. **MCP Tool**: No `expand_query` or direct SPLADE search tool
4. **Benchmark**: No E13 recall benchmark

### 3.3 Integration Tasks

#### Task E13-1: SPLADE Embedding on Ingest
```rust
// In TeleologicalArray generation:
// E13 is part of the 13-embedder array but uses sparse format

// Generate SPLADE sparse vector
let splade_vector = splade_model.embed(&content)?;  // ~30K sparse
let sparse_terms: Vec<(usize, f32)> = splade_vector
    .iter()
    .enumerate()
    .filter(|(_, &v)| v > 0.0)
    .collect();

// Store in inverted index for fast recall
inverted_index.add(memory_id, &sparse_terms)?;
```

**Files to modify**:
- `crates/context-graph-core/src/embedder.rs` (TeleologicalArray generation)
- `crates/context-graph-storage/src/teleological/rocksdb_store/inverted_index.rs`

#### Task E13-2: Wire Stage 1 in Pipeline Search
```rust
// In search_graph with strategy=Pipeline:
// Stage 1: E13 SPLADE Recall (cast wide net)
let query_sparse = splade_model.embed(&query)?;
let stage1_candidates = splade_index.search(&query_sparse, 10_000)?;

// Stage 2: E1 Matryoshka ANN on stage1 candidates
let stage2_results = matryoshka_search_filtered(
    query_fingerprint.e1,
    stage1_candidates.ids(),
    1_000,
)?;
```

**Files to modify**:
- `crates/context-graph-storage/src/teleological/search/pipeline/execution.rs`

#### Task E13-3: MCP Query Expansion Tool
```rust
// New tool: expand_query_terms
// Shows what terms SPLADE expands (fast→quick, db→database)
pub async fn handle_expand_query_terms(
    params: ExpandParams,
    splade_model: &SparseModel,
) -> Result<ExpandResponse> {
    let sparse = splade_model.embed(&params.query)?;
    let top_terms: Vec<(String, f32)> = sparse
        .top_k(params.top_k.unwrap_or(20))
        .map(|(idx, score)| (vocab.decode(idx), score))
        .collect();
    Ok(ExpandResponse { query: params.query, expanded_terms: top_terms })
}
```

**New files**:
- `crates/context-graph-mcp/src/handlers/tools/expansion_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/expansion_dtos.rs`

#### Task E13-4: Benchmark Suite
```rust
// Benchmark: E13 recall@1000 on synonym/expansion queries
// Test cases: Abbreviations (k8s→kubernetes), synonyms (fast→quick)
```

**New file**: `crates/context-graph-benchmark/src/bin/e13_recall_bench.rs`

### 3.4 Architectural Constraints

- **AP-75**: E13 is Stage 1 recall ONLY, never final ranking
- **Constitution S1**: Target 10K candidates from SPLADE
- **Performance**: <50ms for 10K candidate retrieval

---

## 4. E2/E3 Temporal Integration (Post-Retrieval Boost)

### 4.1 Current State

- **E2 (V_freshness)**: `models/custom/temporal_recent/`
  - 512D exponential decay across 4 time scales (hour, day, week, month)
- **E3 (V_periodicity)**: `models/custom/temporal_periodic/`
  - 512D Fourier basis functions for cyclical patterns

### 4.2 Missing Integration

Per **ARCH-25** and **AP-73**, temporal embeddings are POST-RETRIEVAL ONLY:
- Never used in similarity fusion
- Never contribute to topic detection (topic_weight = 0.0)
- Applied as **badges** after retrieval

1. **No Post-Retrieval Hook**: Enrichment pipeline doesn't apply temporal boosts
2. **No Temporal Badges**: Results don't include recency/periodicity metadata
3. **No MCP Exposure**: No tools to query by temporal patterns

### 4.3 Integration Tasks

#### Task E2E3-1: Post-Retrieval Temporal Boost
```rust
// In enrichment_pipeline.rs, AFTER RRF fusion:
if config.apply_temporal_boost {
    for result in &mut results {
        // E2: Recency boost (constitution §6.2)
        let age_hours = (now - result.created_at).as_secs_f64() / 3600.0;
        let recency_boost = match age_hours {
            h if h < 1.0 => 1.3,    // <1h
            h if h < 24.0 => 1.2,   // <1d
            h if h < 168.0 => 1.1,  // <7d
            h if h < 720.0 => 1.0,  // <30d
            _ => 0.8,               // >30d
        };

        // E3: Periodicity alignment (time-of-day similarity)
        let query_periodic = e3_model.embed_timestamp(now)?;
        let memory_periodic = result.fingerprint.e3;
        let periodic_alignment = cosine_similarity(&query_periodic, &memory_periodic);

        // Apply as multiplicative badge (not fusion)
        result.temporal_boost = recency_boost;
        result.periodic_alignment = periodic_alignment;
        result.boosted_score = result.rrf_score * recency_boost;
    }
}
```

**Files to modify**:
- `crates/context-graph-mcp/src/handlers/tools/enrichment_pipeline.rs`
- `crates/context-graph-mcp/src/handlers/tools/enrichment_dtos.rs`

#### Task E2E3-2: Temporal Badges in Response
```rust
// EnrichedSearchResult includes temporal context:
pub struct TemporalBadge {
    pub age_human: String,           // "2 hours ago", "yesterday"
    pub recency_boost: f32,          // 1.0-1.3
    pub time_of_day_match: f32,      // 0.0-1.0 (morning matches morning)
    pub day_of_week_match: f32,      // 0.0-1.0 (weekday matches weekday)
}
```

#### Task E2E3-3: MCP Temporal Query Tool
```rust
// New tool: search_by_recency
// Explicitly prioritize recent memories (without using in retrieval fusion)
pub async fn handle_search_by_recency(
    params: RecencyParams,
) -> Result<RecencySearchResponse> {
    // 1. Normal E1 search
    // 2. Heavy temporal boost (2.0x for <1h)
    // 3. Return with temporal badges
}
```

### 4.4 Architectural Constraints

- **ARCH-25**: Temporal is POST-RETRIEVAL ONLY
- **AP-73**: NEVER in similarity fusion
- **AP-60**: NEVER count toward topics (topic_weight = 0.0)

---

## 5. E9 HDC Integration (Noise-Robust Fallback)

### 5.1 Current State

- **Model**: `models/custom/hdc/`
  - 10K-bit binary hypervectors
  - Projected to 1024D float for fusion
  - Operations: bind (XOR), bundle (majority), permute (shift)

### 5.2 Missing Integration

1. **No MCP Tool**: No direct HDC search exposure
2. **No Benchmark**: No noise-robustness validation
3. **Not in Enrichment Pipeline**: E9 not used as backup when E1 fails

### 5.3 Integration Tasks

#### Task E9-1: HDC as Noise-Robust Fallback
```rust
// In enrichment_pipeline.rs, when E1 results are weak:
if e1_top_score < 0.4 && config.enable_hdc_fallback {
    // E1 struggling - try noise-robust E9
    let e9_results = search_e9(query_fingerprint.e9, top_k * 2)?;

    // Blend E9 into results via RRF
    for (rank, result) in e9_results.iter().enumerate() {
        let e9_rrf = 1.0 / (RRF_K + rank as f32);
        rrf_scores[result.id] += e9_weight * e9_rrf;
    }

    // Flag as HDC-assisted
    summary.hdc_fallback_used = true;
}
```

#### Task E9-2: MCP Typo-Tolerant Search Tool
```rust
// New tool: search_typo_tolerant
// Uses E9 HDC for queries with potential typos/noise
pub async fn handle_search_typo_tolerant(
    params: TypoTolerantParams,
) -> Result<TypoTolerantResponse> {
    // 1. Generate E9 hypervector from noisy query
    // 2. Search using Hamming distance
    // 3. Return results with noise-tolerance metadata
}
```

#### Task E9-3: Benchmark Suite
```rust
// Benchmark: E9 vs E1 on noisy queries
// Test cases: Typos, OCR errors, phonetic variations
// Metrics: Recall@10 degradation with increasing noise
```

**New file**: `crates/context-graph-benchmark/src/bin/e9_robustness_bench.rs`

### 5.4 Architectural Constraints

- **topic_weight = 0.5**: E9 is STRUCTURAL, contributes to topics
- **ARCH-21**: E9 integrates via Weighted RRF, not weighted sum
- **Use Case**: Backup when E1 fails due to noise

---

## 6. Implementation Priority & Timeline

### Phase 1: Critical Pipeline (E12 + E13)

These complete the 5-stage retrieval pipeline per constitution.

| Task | Effort | Dependencies |
|------|--------|--------------|
| E13-1: SPLADE on ingest | 2d | None |
| E13-2: Wire Stage 1 | 2d | E13-1 |
| E12-1: Token storage | 2d | None |
| E12-2: Wire Stage 5 | 2d | E12-1 |
| E12-3: Rerank MCP tool | 1d | E12-2 |
| E13-3: Expand MCP tool | 1d | E13-2 |
| E12-4 + E13-4: Benchmarks | 2d | E12-2, E13-2 |

**Total Phase 1**: ~12 days

### Phase 2: Temporal Enhancement (E2 + E3)

| Task | Effort | Dependencies |
|------|--------|--------------|
| E2E3-1: Post-retrieval boost | 2d | None |
| E2E3-2: Temporal badges | 1d | E2E3-1 |
| E2E3-3: Recency MCP tool | 1d | E2E3-1 |

**Total Phase 2**: ~4 days

### Phase 3: Noise Robustness (E9)

| Task | Effort | Dependencies |
|------|--------|--------------|
| E9-1: HDC fallback | 2d | None |
| E9-2: Typo-tolerant tool | 1d | E9-1 |
| E9-3: Benchmark | 1d | E9-1 |

**Total Phase 3**: ~4 days

---

## 7. Success Metrics

### E12 ColBERT
- MRR@10 improvement: +15% vs E1-only on exact phrase queries
- P@1 for multi-word queries: >0.80
- Latency: <15ms for 50 candidates

### E13 SPLADE
- Recall@1000: >0.95 for synonym/abbreviation queries
- Expansion coverage: 3-5 related terms per query term
- Latency: <50ms for 10K candidates

### E2/E3 Temporal
- Recency boost accuracy: Recent memories ranked higher when relevant
- Time-of-day alignment: Morning queries prefer morning memories

### E9 HDC
- Robustness: <10% recall degradation with 10% character noise
- Fallback trigger rate: <5% of queries (E1 usually sufficient)

---

## 8. Testing Strategy

### Unit Tests
- Each embedder's MCP tool handler
- Stage integration in pipeline
- Inverted index operations (E13)
- Token storage operations (E12)

### Integration Tests
- Full pipeline: E13 → E1 → RRF → E12
- Temporal boost application
- HDC fallback trigger conditions

### Benchmark Validation
- Compare against E1-only baseline
- Measure latency budgets
- GPU memory profiling

---

## References

- Constitution: `/home/cabdru/contextgraph/docs2/constitution.yaml`
- PRD: `/home/cabdru/contextgraph/docs2/contextprd.md`
- Existing patterns:
  - `enrichment_pipeline.rs` (RRF fusion)
  - `intent_tools.rs` (multiplicative boost)
  - `causal_tools.rs` (asymmetric similarity)
  - `entity_tools.rs` (TransE scoring)
