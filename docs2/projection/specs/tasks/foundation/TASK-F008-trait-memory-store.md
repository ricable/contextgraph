# Task: TASK-F008 - Modify MemoryStore Trait for Multi-Array Queries

## Metadata
- **ID**: TASK-F008
- **Layer**: Foundation
- **Priority**: P0 (Critical Path)
- **Estimated Effort**: L (Large)
- **Dependencies**: TASK-F001, TASK-F002, TASK-F003, TASK-F006
- **Traces To**: TS-302, FR-301, FR-302, FR-303, FR-401

## Description

Modify the `MemoryStore` trait to support multi-embedding queries with configurable weights. The new trait operates on `TeleologicalFingerprint` instead of `MemoryNode` with single embedding.

Key changes:
1. **Store**: Accepts `TeleologicalFingerprint` with 12 embeddings
2. **Search**: Weighted similarity across all 12 spaces
3. **Single-space search**: Query specific embedder index
4. **Purpose search**: Find by 12D purpose vector similarity
5. **Goal alignment**: Find memories aligned with goals
6. **Johari filter**: Filter by quadrant per embedder

## Acceptance Criteria

- [ ] `SimilarityWeights` struct with 12 weights summing to 1.0
- [ ] `QueryType` enum for automatic weight selection
- [ ] `MultiEmbeddingSearchOptions` with comprehensive filters
- [ ] `MultiEmbeddingSearchResult` with per-embedder breakdown
- [ ] `TeleologicalMemoryStore` trait with multi-embedding operations
- [ ] `search_multi()` for weighted multi-space search
- [ ] `search_single_space()` for targeted single-embedder search
- [ ] `search_by_purpose()` for teleological search
- [ ] `find_aligned_to_goal()` for goal-based retrieval
- [ ] `find_by_johari()` for awareness-based filtering
- [ ] Unit tests for weight validation and search options

## Implementation Steps

1. Read existing `crates/context-graph-core/src/traits/memory_store.rs`
2. Create `crates/context-graph-core/src/traits/teleological_store.rs`:
   - Import fingerprint types
   - Implement SimilarityWeights with QueryType presets
   - Implement MultiEmbeddingSearchOptions
   - Implement MultiEmbeddingSearchResult
   - Implement TeleologicalMemoryStore trait
3. Update `crates/context-graph-core/src/traits/mod.rs`:
   - Export new module
4. Update stub in `crates/context-graph-core/src/stubs/memory_stub.rs`:
   - Implement TeleologicalMemoryStore for testing

## Files Affected

### Files to Create
- `crates/context-graph-core/src/traits/teleological_store.rs` - New store trait

### Files to Modify
- `crates/context-graph-core/src/traits/mod.rs` - Export new module
- `crates/context-graph-core/src/stubs/memory_stub.rs` - Implement new trait

## Code Signature (Definition of Done)

```rust
// teleological_store.rs
use crate::error::CoreResult;
use crate::types::fingerprint::{TeleologicalFingerprint, PurposeVector, JohariFingerprint, EvolutionTrigger};
use crate::types::johari::JohariQuadrant;
use async_trait::async_trait;
use uuid::Uuid;

/// Query weights for multi-embedding similarity
#[derive(Debug, Clone)]
pub struct SimilarityWeights {
    /// Weights for each embedder [0.0, 1.0], must sum to 1.0
    pub weights: [f32; 12],
    /// Query type for automatic weight selection
    pub query_type: QueryType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    SemanticSearch,      // Heavy E1, E7
    CausalReasoning,     // Heavy E5
    CodeSearch,          // Heavy E4, E7
    TemporalNavigation,  // Heavy E2-E4
    FactChecking,        // Heavy E11
    Balanced,            // Equal weights
}

impl SimilarityWeights {
    /// Create weights for a specific query type
    pub fn for_query_type(qt: QueryType) -> Self;

    /// Validate weights sum to ~1.0
    pub fn validate(&self) -> bool;
}

/// Multi-embedding search options
#[derive(Debug, Clone)]
pub struct MultiEmbeddingSearchOptions {
    /// Maximum results to return
    pub top_k: usize,
    /// Minimum aggregate similarity threshold
    pub min_similarity: f32,
    /// Similarity weights per embedder
    pub weights: SimilarityWeights,
    /// Filter by Johari quadrant (for specific embedder)
    pub johari_filter: Option<(usize, JohariQuadrant)>,
    /// Minimum alignment to North Star
    pub min_alignment: Option<f32>,
    /// Specific embedder index for single-space search (0-11)
    pub single_space: Option<usize>,
}

impl Default for MultiEmbeddingSearchOptions {
    fn default() -> Self;
}

/// Search result with multi-embedding similarity breakdown
#[derive(Debug, Clone)]
pub struct MultiEmbeddingSearchResult {
    /// The matched fingerprint
    pub fingerprint: TeleologicalFingerprint,
    /// Aggregate similarity score
    pub similarity: f32,
    /// Per-embedder similarity scores
    pub per_embedder_similarity: [f32; 12],
    /// Top contributing embedders (indices)
    pub top_contributors: Vec<(usize, f32)>,
    /// Alignment to North Star
    pub alignment_score: f32,
}

/// Teleological Memory Store trait
///
/// REPLACES legacy MemoryStore that accepted Vec<f32> embeddings.
/// Operates on TeleologicalFingerprint with multi-embedding search.
#[async_trait]
pub trait TeleologicalMemoryStore: Send + Sync {
    /// Store a teleological fingerprint
    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid>;

    /// Retrieve fingerprint by ID
    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>>;

    /// Multi-embedding semantic search with weighted similarity
    async fn search_multi(
        &self,
        query: &TeleologicalFingerprint,
        options: MultiEmbeddingSearchOptions,
    ) -> CoreResult<Vec<MultiEmbeddingSearchResult>>;

    /// Single-space search (uses only one embedder's index)
    async fn search_single_space(
        &self,
        query_embedding: &[f32],
        embedder_index: usize,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    /// Search by purpose vector similarity
    async fn search_by_purpose(
        &self,
        purpose: &PurposeVector,
        top_k: usize,
        min_similarity: f32,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    /// Find memories aligned with a goal
    async fn find_aligned_to_goal(
        &self,
        goal_id: Uuid,
        min_alignment: f32,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    /// Find memories by Johari quadrant for specific embedder
    async fn find_by_johari(
        &self,
        embedder_index: usize,
        quadrant: JohariQuadrant,
        top_k: usize,
    ) -> CoreResult<Vec<Uuid>>;

    /// Update fingerprint alignment and Johari classification
    async fn update_alignment(
        &self,
        id: Uuid,
        purpose_vector: PurposeVector,
        johari: JohariFingerprint,
    ) -> CoreResult<bool>;

    /// Record purpose evolution snapshot
    async fn record_evolution(
        &self,
        id: Uuid,
        trigger: EvolutionTrigger,
    ) -> CoreResult<()>;

    /// Delete fingerprint
    async fn delete(&self, id: Uuid) -> CoreResult<bool>;

    /// Get total count
    async fn count(&self) -> CoreResult<usize>;

    /// Compact storage and rebuild indexes
    async fn compact(&self) -> CoreResult<()>;
}
```

## Weight Profiles by Query Type

| Query Type | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 |
|------------|----|----|----|----|----|----|----|----|----|----|-----|-----|
| SemanticSearch | 0.30 | 0.05 | 0.05 | 0.05 | 0.10 | 0.05 | 0.20 | 0.05 | 0.05 | 0.05 | 0.03 | 0.02 |
| CausalReasoning | 0.15 | 0.03 | 0.03 | 0.03 | 0.45 | 0.03 | 0.10 | 0.03 | 0.03 | 0.05 | 0.05 | 0.02 |
| CodeSearch | 0.15 | 0.02 | 0.02 | 0.35 | 0.05 | 0.03 | 0.25 | 0.02 | 0.02 | 0.03 | 0.03 | 0.03 |
| TemporalNavigation | 0.15 | 0.20 | 0.20 | 0.20 | 0.05 | 0.02 | 0.05 | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 |
| FactChecking | 0.10 | 0.02 | 0.02 | 0.02 | 0.20 | 0.05 | 0.05 | 0.02 | 0.02 | 0.05 | 0.43 | 0.02 |
| Balanced | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.083 | 0.087 |

## Testing Requirements

### Unit Tests
- `test_similarity_weights_validate_valid` - Sums to 1.0
- `test_similarity_weights_validate_invalid` - Detects wrong sum
- `test_weights_for_query_type` - Each type has correct profile
- `test_search_options_default` - Correct defaults
- `test_search_result_top_contributors` - Identifies highest weighted

### Integration Tests
- Test store/retrieve round-trip with full fingerprint
- Test search_multi with various query types
- Test search_single_space for each embedder 0-11
- Test search_by_purpose with similar/different purposes
- Test find_by_johari returns correct quadrant matches

## Verification

```bash
# Compile check
cargo check -p context-graph-core

# Run unit tests
cargo test -p context-graph-core teleological_store

# Verify trait exports
cargo doc -p context-graph-core --open
```

## Constraints

- Weights MUST sum to 1.0 (within 0.01 tolerance)
- QueryType::Balanced distributes weight equally
- search_multi computes: S = sum(w_i * cos(A_i, B_i))
- search_single_space uses ONLY the specified index
- find_by_johari requires JohariFingerprint quadrant match
- All methods async for non-blocking I/O

## Notes

This trait is the primary interface for memory operations in the multi-array architecture. The implementation (TASK-L002 or similar in Logic Layer) will:
1. Use TeleologicalSchema from TASK-F004 for storage
2. Use HNSW configs from TASK-F005 for indexing
3. Compute weighted similarity per TECH-SPEC-001 Section 4.1

The key insight is that query type determines which embedding spaces matter most:
- "What does this code do?" -> CodeSearch (heavy E4, E7)
- "Why did this happen?" -> CausalReasoning (heavy E5)
- "What happened recently?" -> TemporalNavigation (heavy E2-E4)

Reference implementation in TECH-SPEC-001 Section 3.2 (TS-302).
