# TASK-EMB-001: Fix Dimension Constants

<task_spec id="TASK-EMB-001" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Fix Dimension Constants |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 1 |
| **Implements** | REQ-EMB-001, REQ-EMB-002 |
| **Depends On** | None |
| **Estimated Complexity** | low |
| **Parallel Group** | A (start immediately) |

---

## Context

The Constitution v4.0.0 specifies:
- `E6_Sparse: { dim: "~30K 5%active" }` for sparse vocabulary
- `E1_Semantic: { dim: 1024 }` with `matryoshka: true, truncatable: [512, 256, 128]`

Current codebase has inconsistent dimension constants (768 vs 1536) causing `DimensionMismatch` errors at runtime. This foundation task establishes the canonical constants that ALL other tasks depend on.

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Constitution reference | `docs2/constitution.yaml` |
| Current constants | `crates/context-graph-core/src/config/mod.rs` |
| Sparse model | `crates/context-graph-embeddings/src/models/pretrained/sparse/` |

---

## Prerequisites

- [ ] Read `docs2/constitution.yaml` section `embeddings.models`
- [ ] Identify all files with dimension constants

---

## Scope

### In Scope

- Define canonical dimension constants in a single location
- Create `config/constants.rs` module
- Update `SPARSE_PROJECTED_DIMENSION` to 1536
- Define `SPARSE_VOCAB_SIZE` as 30522 (BERT vocab)
- Ensure consistency across all models

### Out of Scope

- Actual projection implementation (TASK-EMB-011, TASK-EMB-012)
- Storage dimension handling (TASK-EMB-005)
- Model loading (TASK-EMB-013)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/config/constants.rs

//! Canonical dimension constants per Constitution v4.0.0.
//!
//! # Critical
//! These constants define the ONLY valid dimensions for the embedding pipeline.
//! Changing these requires updating: storage, search, quantization, and all models.

/// Sparse vocabulary size (BERT WordPiece vocab)
pub const SPARSE_VOCAB_SIZE: usize = 30522;

/// Sparse projection output dimension
/// MUST match E1_Semantic.dim from Constitution
pub const SPARSE_PROJECTED_DIMENSION: usize = 1536;

/// Semantic embedding dimension (E1)
/// Constitution: E1_Semantic: { dim: 1024, matryoshka: true }
pub const SEMANTIC_EMBEDDING_DIM: usize = 1024;

/// Matryoshka truncation dimensions
pub const MATRYOSHKA_DIMS: [usize; 4] = [1024, 512, 256, 128];

/// Fast ANN search dimension (Stage 2 pipeline)
pub const MATRYOSHKA_FAST_DIM: usize = 128;

/// Temporal embedding dimension (E2, E3, E4)
pub const TEMPORAL_EMBEDDING_DIM: usize = 512;

/// Causal embedding dimension (E5)
pub const CAUSAL_EMBEDDING_DIM: usize = 768;

/// Code embedding dimension (E7)
pub const CODE_EMBEDDING_DIM: usize = 1536;

/// Graph/MiniLM embedding dimension (E8)
pub const GRAPH_EMBEDDING_DIM: usize = 384;

/// HDC binary dimension (E9)
pub const HDC_EMBEDDING_DIM: usize = 1024;  // 10K-bit compressed to 1024

/// Multimodal embedding dimension (E10)
pub const MULTIMODAL_EMBEDDING_DIM: usize = 768;

/// Entity/MiniLM embedding dimension (E11)
pub const ENTITY_EMBEDDING_DIM: usize = 384;

/// Late interaction per-token dimension (E12)
pub const LATE_INTERACTION_DIM: usize = 128;

/// Total embedder count
pub const EMBEDDER_COUNT: usize = 13;

/// Purpose vector dimension (13D teleological signature)
pub const PURPOSE_VECTOR_DIM: usize = 13;
```

### Constraints

- All constants MUST match Constitution v4.0.0 `embeddings.models` section
- No magic numbers in codebase - reference these constants
- Breaking change: requires updating all dependent modules

### Verification

- `grep -r "768" crates/` returns 0 results for dimension usage
- `grep -r "1536" crates/` uses `SPARSE_PROJECTED_DIMENSION` constant
- All tests pass with new constants

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-core/src/config/constants.rs` | Canonical dimension constants |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-core/src/config/mod.rs` | Add `pub mod constants;` |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | Use `SPARSE_PROJECTED_DIMENSION` |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Use constants |

---

## Validation Criteria

- [ ] `constants.rs` file created with all 14 constants
- [ ] Constants match Constitution v4.0.0 exactly
- [ ] `mod.rs` exports `constants` module
- [ ] No hardcoded dimension values remain in sparse module
- [ ] `cargo check` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-core
cargo check -p context-graph-embeddings
grep -r "= 768" crates/ --include "*.rs" | grep -v "test" | wc -l  # Should be 0
```

---

## Pseudo Code

```
constants.rs:
  Define all 14 dimension constants
  Add doc comments referencing Constitution section
  Export via pub

mod.rs:
  Add `pub mod constants;`

sparse/types.rs:
  Replace hardcoded 768 with SPARSE_PROJECTED_DIMENSION
  Import from config::constants
```

</task_spec>
