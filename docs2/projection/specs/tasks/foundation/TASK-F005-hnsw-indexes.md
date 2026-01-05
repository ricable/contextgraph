# Task: TASK-F005 - Implement Per-Embedder HNSW Index Configuration

## Metadata
- **ID**: TASK-F005
- **Layer**: Foundation
- **Priority**: P1 (High)
- **Estimated Effort**: M (Medium)
- **Dependencies**: TASK-F001
- **Traces To**: TS-202, FR-302

## Description

Implement the HNSW (Hierarchical Navigable Small World) index configuration for 12 per-embedder indexes plus one Purpose Vector index. Each embedding space requires a separate index with dimension-appropriate configuration.

Index types:
- **E1-E5, E7-E11**: Standard HNSW with cosine similarity
- **E6 (Sparse)**: Inverted index (NOT HNSW)
- **E12 (Late-Interaction)**: ColBERT MaxSim (NOT HNSW)
- **Purpose Vector**: 12D HNSW for teleological search

This task defines the configuration; actual index instantiation happens in Logic Layer.

## Acceptance Criteria

- [ ] `EmbedderIndex` enum for all 12 embedders + PurposeVector
- [ ] `HnswConfig` struct with M, ef_construction, ef_search, dimension
- [ ] `DistanceMetric` enum (Cosine, DotProduct, Euclidean, AsymmetricCosine)
- [ ] `get_hnsw_config(index)` returns appropriate config or None
- [ ] `all_hnsw_configs()` returns map of all HNSW-able indexes
- [ ] `recommended_metric(embedder_idx)` for query planning
- [ ] Documentation of which indexes use HNSW vs alternatives
- [ ] Unit tests for configuration correctness

## Implementation Steps

1. Create `crates/context-graph-storage/src/teleological/indexes/mod.rs`:
   - Define module structure
2. Create `crates/context-graph-storage/src/teleological/indexes/hnsw_config.rs`:
   - Implement `DistanceMetric` enum
   - Implement `EmbedderIndex` enum
   - Implement `HnswConfig` struct
   - Implement `get_hnsw_config()` function
   - Implement `all_hnsw_configs()` function
3. Create `crates/context-graph-storage/src/teleological/indexes/metrics.rs`:
   - Implement `recommended_metric()` function
   - Document metric selection rationale
4. Update `crates/context-graph-storage/src/teleological/mod.rs` to export indexes

## Files Affected

### Files to Create
- `crates/context-graph-storage/src/teleological/indexes/mod.rs` - Module definition
- `crates/context-graph-storage/src/teleological/indexes/hnsw_config.rs` - HNSW configuration
- `crates/context-graph-storage/src/teleological/indexes/metrics.rs` - Distance metrics

### Files to Modify
- `crates/context-graph-storage/src/teleological/mod.rs` - Export indexes module

## Code Signature (Definition of Done)

```rust
// hnsw_config.rs
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Number of connections per node (M parameter)
    pub m: usize,
    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search
    pub ef_search: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Embedding dimension
    pub dimension: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
    AsymmetricCosine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedderIndex {
    E1TextGeneral,      // 1024D HNSW
    E2TextSmall,        // 512D HNSW
    E3Multilingual,     // 512D HNSW
    E4Code,             // 512D HNSW
    E5QueryDoc,         // 768D x 2 HNSW (asymmetric)
    E6Sparse,           // Inverted index (NOT HNSW)
    E7OpenaiAda,        // 1536D HNSW
    E8Minilm,           // 384D HNSW
    E9Simhash,          // 1024D HNSW
    E10Instructor,      // 768D HNSW
    E11Fast,            // 384D HNSW
    E12TokenLevel,      // ColBERT MaxSim (NOT HNSW)
    PurposeVector,      // 12D HNSW
}

/// Get HNSW config for index type. Returns None for non-HNSW indexes.
pub fn get_hnsw_config(index: EmbedderIndex) -> Option<HnswConfig>;

/// Get all indexes that use HNSW
pub fn all_hnsw_configs() -> HashMap<EmbedderIndex, HnswConfig>;

// metrics.rs
/// Get recommended distance metric for embedder
pub fn recommended_metric(embedder_index: usize) -> DistanceMetric;

/// Compute distance using specified metric
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32;

/// Convert distance to similarity [0.0, 1.0]
pub fn distance_to_similarity(distance: f32, metric: DistanceMetric) -> f32;
```

## HNSW Configuration Table

| Index | Dimension | M | ef_construction | ef_search | Metric |
|-------|-----------|---|-----------------|-----------|--------|
| E1TextGeneral | 1024 | 16 | 200 | 100 | Cosine |
| E2TextSmall | 512 | 16 | 200 | 100 | Cosine |
| E3Multilingual | 512 | 16 | 200 | 100 | Cosine |
| E4Code | 512 | 16 | 200 | 100 | Cosine |
| E5QueryDoc | 768 | 16 | 200 | 100 | AsymmetricCosine |
| E6Sparse | N/A | - | - | - | Jaccard (inverted) |
| E7OpenaiAda | 1536 | 16 | 200 | 100 | Cosine |
| E8Minilm | 384 | 16 | 200 | 100 | Cosine |
| E9Simhash | 1024 | 16 | 200 | 100 | Cosine |
| E10Instructor | 768 | 16 | 200 | 100 | Cosine |
| E11Fast | 384 | 16 | 200 | 100 | Cosine |
| E12TokenLevel | 128/token | - | - | - | MaxSim |
| PurposeVector | 12 | 16 | 200 | 100 | Cosine |

## Testing Requirements

### Unit Tests
- `test_get_hnsw_config_e1` - Returns 1024D config
- `test_get_hnsw_config_e6_none` - Returns None (sparse)
- `test_get_hnsw_config_e12_none` - Returns None (late-interaction)
- `test_get_hnsw_config_purpose` - Returns 12D config
- `test_all_hnsw_configs_count` - Returns 11 configs (excludes E6, E12)
- `test_recommended_metric_dense` - Cosine for E1-E5, E7-E11
- `test_recommended_metric_sparse` - Jaccard for E6
- `test_recommended_metric_maxsim` - MaxSim for E12
- `test_compute_distance_cosine` - Correct cosine distance
- `test_distance_to_similarity` - Correct conversion

## Verification

```bash
# Compile check
cargo check -p context-graph-storage

# Run unit tests
cargo test -p context-graph-storage indexes
```

## Constraints

- HNSW parameters from constitution.yaml:
  - M = 16 (connections per node)
  - ef_construction = 200 (build quality)
  - ef_search = 100 (search quality)
- E6 uses inverted index for sparse vectors
- E12 uses ColBERT-style MaxSim aggregation
- Purpose vector is only 12D (very fast search)

## Notes

This task defines index configuration only. The actual HNSW index instantiation and management happens in Logic Layer (TASK-L004 or similar).

For E5 (asymmetric), we need TWO indexes:
1. Query index - for indexing query embeddings
2. Document index - for indexing document embeddings

Reference implementation in TECH-SPEC-001 Section 2.2 (TS-202).
