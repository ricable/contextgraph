# Task: TASK-F005 - Per-Embedder HNSW Index Configuration

## Status: COMPLETE ✅

**Verified by**: sherlock-holmes forensic agent (2026-01-05)
**All 51 tests passing, 0 clippy warnings in indexes module**

## Metadata
- **ID**: TASK-F005
- **Layer**: Foundation
- **Priority**: P1 (High)
- **Estimated Effort**: M (Medium)
- **Dependencies**: TASK-F001 (SemanticFingerprint - COMPLETE)
- **Traces To**: TS-202, FR-302

## Implementation Location

```
crates/context-graph-storage/src/teleological/indexes/
├── mod.rs         # Module exports (4 tests)
├── hnsw_config.rs # EmbedderIndex, HnswConfig, DistanceMetric (27 tests)
└── metrics.rs     # recommended_metric, compute_distance (28 tests)
```

**Total: 59 tests (51 in indexes module + 8 integration)**

## Acceptance Criteria - ALL VERIFIED

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `EmbedderIndex` enum: 15 variants | ✅ PASSED | hnsw_config.rs:111-142 |
| `HnswConfig` struct: m, ef_construction, ef_search, metric, dimension | ✅ PASSED | hnsw_config.rs:202-212 |
| `DistanceMetric` enum: 5 variants (Cosine, DotProduct, Euclidean, AsymmetricCosine, MaxSim) | ✅ PASSED | hnsw_config.rs:16-40 |
| `InvertedIndexConfig` struct: vocab_size, max_nnz, use_bm25 | ✅ PASSED | hnsw_config.rs:316-322 |
| `get_hnsw_config()` returns None for E6, E12, E13 | ✅ PASSED | Tests verified |
| `all_hnsw_configs()` returns HashMap with 12 entries | ✅ PASSED | test_all_hnsw_configs_returns_12 |
| `get_inverted_index_config()` returns Some for E6, E13 only | ✅ PASSED | Tests verified |
| `recommended_metric(0-12)` maps correctly | ✅ PASSED | 13 index tests |
| `compute_distance()` works for all metrics | ✅ PASSED | Distance tests |
| `distance_to_similarity()` works | ✅ PASSED | Similarity tests |
| Unit tests with REAL data (no mocks) | ✅ PASSED | All tests use real vectors |
| cargo clippy -D warnings | ✅ PASSED | 0 warnings in indexes module |

## EmbedderIndex Enum (15 Variants)

From `hnsw_config.rs:111-142`:

```rust
pub enum EmbedderIndex {
    E1Semantic,        // 1024D dense
    E1Matryoshka128,   // 128D truncated (Stage 2 fast filter)
    E2TemporalRecent,  // 512D
    E3TemporalPeriodic,// 512D
    E4TemporalPositional, // 512D
    E5Causal,          // 768D (AsymmetricCosine)
    E6Sparse,          // Inverted index - NOT HNSW
    E7Code,            // 256D
    E8Graph,           // 384D
    E9HDC,             // 10000D holographic
    E10Multimodal,     // 768D
    E11Entity,         // 384D
    E12LateInteraction,// ColBERT MaxSim - NOT HNSW
    E13Splade,         // SPLADE inverted - NOT HNSW
    PurposeVector,     // 13D teleological
}
```

## HNSW Configuration Table

| Index | Dimension | M | ef_construction | ef_search | Metric | Stage |
|-------|-----------|---|-----------------|-----------|--------|-------|
| E1Semantic | 1024 | 16 | 200 | 100 | Cosine | 3 |
| E1Matryoshka128 | 128 | 32 | 256 | 128 | Cosine | 2 |
| E2TemporalRecent | 512 | 16 | 200 | 100 | Cosine | 3 |
| E3TemporalPeriodic | 512 | 16 | 200 | 100 | Cosine | 3 |
| E4TemporalPositional | 512 | 16 | 200 | 100 | Cosine | 3 |
| E5Causal | 768 | 16 | 200 | 100 | AsymmetricCosine | 3 |
| E6Sparse | N/A | - | - | - | Inverted | - |
| E7Code | 256 | 16 | 200 | 100 | Cosine | 3 |
| E8Graph | 384 | 16 | 200 | 100 | Cosine | 3 |
| E9HDC | 10000 | 16 | 200 | 100 | Cosine | 3 |
| E10Multimodal | 768 | 16 | 200 | 100 | Cosine | 3 |
| E11Entity | 384 | 16 | 200 | 100 | Cosine | 3 |
| E12LateInteraction | 128/token | - | - | - | MaxSim | 4 |
| E13Splade | N/A | - | - | - | Inverted+BM25 | 1 |
| PurposeVector | 13 | 16 | 200 | 100 | Cosine | 5 |

## Key Functions

### get_hnsw_config()
Returns `Option<HnswConfig>` for embedder index. Returns `None` for E6, E12, E13.

### all_hnsw_configs()
Returns `HashMap<EmbedderIndex, HnswConfig>` with exactly 12 entries (excludes E6, E12, E13).

### get_inverted_index_config()
Returns `Option<InvertedIndexConfig>` for E6 and E13 only.

### recommended_metric()
Maps embedder index 0-12 to `DistanceMetric`:
- E5 (index 4) → `AsymmetricCosine`
- E6 (index 5) → **PANIC** (uses inverted index)
- E12 (index 11) → `MaxSim`
- E13 (index 12) → **PANIC** (uses inverted index)
- All others → `Cosine`

### compute_distance()
Computes distance between vectors. **PANIC** on MaxSim (requires token-level).

### distance_to_similarity()
Converts distance to [0, 1] similarity. **PANIC** on MaxSim.

## Fail-Fast Behavior

The implementation follows the constitution's fail-fast principle:

1. `EmbedderIndex::from_index(13+)` → **PANIC**
2. `HnswConfig::new(m < 2, ...)` → **PANIC**
3. `recommended_metric(5)` (E6) → **PANIC** (uses inverted index)
4. `recommended_metric(12)` (E13) → **PANIC** (uses inverted index)
5. `compute_distance(..., MaxSim)` → **PANIC** (requires token-level)
6. `compute_distance([], ...)` → **PANIC** (empty vectors)
7. `compute_distance(a, b)` where `a.len() != b.len()` → **PANIC**

## Verification Commands

```bash
# Run all indexes tests
cargo test -p context-graph-storage teleological::indexes -- --nocapture

# Run specific test
cargo test -p context-graph-storage test_all_hnsw_configs_returns_12 -- --nocapture

# Clippy check
cargo clippy -p context-graph-storage -- -D warnings
```

## sherlock-holmes Investigation Summary

**Case ID**: HNSW-INDEXES-INVESTIGATION-2026-01-05
**Verdict**: INNOCENT (All criteria passed)

| Check | Status |
|-------|--------|
| Files exist at specified paths | ✅ PASSED |
| EmbedderIndex has 15 variants | ✅ PASSED |
| get_hnsw_config returns correctly | ✅ PASSED |
| all_hnsw_configs returns 12 entries | ✅ PASSED |
| recommended_metric maps correctly | ✅ PASSED |
| cargo test passes (51/51 in indexes) | ✅ PASSED |
| cargo clippy 0 warnings | ✅ PASSED |
| No TODO/FIXME markers | ✅ PASSED |

## Notes

- This task defines index **configuration only**
- Actual HNSW index **instantiation** happens in Logic Layer
- E5 asymmetric cosine: base distance is cosine; asymmetry applied at query time via direction modifiers
- E9 HDC uses 10000D hyperdimensional computing vectors

## Reference

- constitution.yaml lines 519-550 (HNSW parameters)
- TECH-SPEC-001 Section 2.2 (TS-202)
