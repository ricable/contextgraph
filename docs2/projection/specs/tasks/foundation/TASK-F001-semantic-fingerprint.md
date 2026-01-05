# Task: TASK-F001 - Implement SemanticFingerprint Struct

## Metadata
- **ID**: TASK-F001
- **Layer**: Foundation
- **Priority**: P0 (Critical Path)
- **Estimated Effort**: M (Medium)
- **Dependencies**: None (Foundation start)
- **Traces To**: TS-101, FR-101, FR-102, FR-103, FR-104

## Description

Implement the `SemanticFingerprint` struct that stores all 12 embedding vectors without fusion, preserving 100% of semantic information. This is THE core data structure replacing the legacy `Vec<f32>` single-vector representation.

The SemanticFingerprint stores embeddings from 12 different embedding models with varying dimensions:
- E1: 1024D (text-embedding-3-large)
- E2: 512D (text-embedding-3-small)
- E3: 512D (multilingual-e5-base)
- E4: 512D (codet5p-110m code)
- E5: 768D x 2 (asymmetric query/document)
- E6: ~1500 active of 30K sparse (splade-v3)
- E7: 1536D (text-embedding-ada-002)
- E8: 384D (all-MiniLM-L6)
- E9: 1024D (SimHash from 10K-bit HDC)
- E10: 768D (instructor-xl)
- E11: 384D (e5-small-v2)
- E12: 128D per token (ColBERT late-interaction)

**Total storage: ~46KB per fingerprint (vs 6KB legacy fused)**

## Acceptance Criteria

- [ ] `SemanticFingerprint` struct defined with exactly 12 embedding fields
- [ ] `SparseVector30K` struct for E6 sparse embeddings with validation
- [ ] Correct dimensions for each embedder (fixed-size arrays where possible)
- [ ] `storage_size()` method returns accurate byte count
- [ ] `zeroed()` constructor for initialization
- [ ] `get_embedding(idx)` accessor for index-based retrieval
- [ ] All types derive `Debug`, `Clone`, `Serialize`, `Deserialize`
- [ ] NO fusion logic, NO gating, NO single-vector output
- [ ] Unit tests with REAL dimension data (no mocks)

## Implementation Steps

1. Create module directory: `crates/context-graph-core/src/types/fingerprint/`
2. Create `mod.rs` with module exports
3. Implement `SparseVector30K` struct in `sparse.rs`:
   - Fields: `indices: Vec<u16>`, `values: Vec<f32>`
   - Const: `VOCAB_SIZE = 30_000`, `SPARSITY = 0.05`
   - Method: `new(indices, values) -> Result<Self, &'static str>`
   - Method: `similarity(&self, other: &Self) -> f32`
4. Implement `SemanticFingerprint` struct in `semantic.rs`:
   - E1-E4, E7-E11: Fixed-size `[f32; N]` arrays
   - E5: Tuple of two `[f32; 768]` arrays (query, doc)
   - E6: `SparseVector30K`
   - E12: `Vec<[f32; 128]>` for variable token count
5. Add helper constants: `TOTAL_DENSE_DIMS`, `EXPECTED_SIZE_BYTES`
6. Update `crates/context-graph-core/src/types/mod.rs` to export fingerprint module

## Files Affected

### Files to Create
- `crates/context-graph-core/src/types/fingerprint/mod.rs` - Module definition and exports
- `crates/context-graph-core/src/types/fingerprint/sparse.rs` - SparseVector30K implementation
- `crates/context-graph-core/src/types/fingerprint/semantic.rs` - SemanticFingerprint struct

### Files to Modify
- `crates/context-graph-core/src/types/mod.rs` - Add `pub mod fingerprint;` export

## Code Signature (Definition of Done)

```rust
// sparse.rs
pub const VOCAB_SIZE: usize = 30_000;
pub const SPARSITY: f32 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseVector30K {
    pub indices: Vec<u16>,
    pub values: Vec<f32>,
}

impl SparseVector30K {
    pub fn new(indices: Vec<u16>, values: Vec<f32>) -> Result<Self, &'static str>;
    pub fn similarity(&self, other: &Self) -> f32;
}

// semantic.rs
pub const TOTAL_DENSE_DIMS: usize = 1024 + 512 + 512 + 512 + 768 + 1536 + 384 + 1024 + 768 + 384;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFingerprint {
    pub e1_text_general: [f32; 1024],
    pub e2_text_small: [f32; 512],
    pub e3_multilingual: [f32; 512],
    pub e4_code: [f32; 512],
    pub e5_query_doc: ([f32; 768], [f32; 768]),
    pub e6_sparse: SparseVector30K,
    pub e7_openai_ada: [f32; 1536],
    pub e8_minilm: [f32; 384],
    pub e9_simhash: [f32; 1024],
    pub e10_instructor: [f32; 768],
    pub e11_fast: [f32; 384],
    pub e12_token_level: Vec<[f32; 128]>,
}

impl SemanticFingerprint {
    pub fn storage_size(&self) -> usize;
    pub fn zeroed() -> Self;
    pub fn get_embedding(&self, idx: usize) -> Option<&[f32]>;
}
```

## Testing Requirements

### Unit Tests (in `semantic.rs` or separate test file)
- `test_semantic_fingerprint_zeroed` - All arrays initialized to zero
- `test_semantic_fingerprint_storage_size` - Returns ~46KB for typical fingerprint
- `test_semantic_fingerprint_get_embedding` - Correct slice for each index 0-11
- `test_sparse_vector_new_valid` - Accepts valid index/value pairs
- `test_sparse_vector_new_invalid_length` - Rejects mismatched lengths
- `test_sparse_vector_new_invalid_index` - Rejects out-of-bounds indices
- `test_sparse_vector_similarity` - Jaccard + weighted overlap computation
- `test_semantic_fingerprint_serialization` - Round-trip with bincode

### Test Fixtures
- Create `tests/fixtures/fingerprints/` directory
- Add `sample_semantic_fingerprint.json` with realistic dimensions

## Verification

```bash
# Compile check
cargo check -p context-graph-core

# Run unit tests
cargo test -p context-graph-core fingerprint

# Type check
cargo clippy -p context-graph-core -- -D warnings
```

## Constraints

- NO `unwrap()` in production code - use `expect()` with descriptive message
- NO inline mock data - use test fixtures
- Fixed-size arrays where dimension is constant (better memory layout)
- `Vec<[f32; 128]>` for E12 due to variable token count
- All f32 values for GPU compatibility

## Notes

This task establishes the foundational data structure. All subsequent tasks (TASK-F002 through TASK-F008) depend on this being complete and correct.

Reference implementation in TECH-SPEC-001 Section 1.1 (TS-101).
