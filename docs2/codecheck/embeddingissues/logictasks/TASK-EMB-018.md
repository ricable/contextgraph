# TASK-EMB-008: Update SparseVector Struct

<task_spec id="TASK-EMB-008" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Update SparseVector Struct |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 8 |
| **Implements** | REQ-EMB-001, REQ-EMB-002 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | low |
| **Parallel Group** | B |

---

## Context

Current `SparseVector` implementation uses broken hash modulo projection:
```rust
// BROKEN - destroys semantic information
idx % SPARSE_PROJECTED_DIMENSION
```

This task updates SparseVector to:
1. Use correct dimension constants
2. Add CSR conversion for GPU SpMM
3. Remove hash-based projection
4. Prepare for learned projection integration (TASK-EMB-012)

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current implementation | `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` |
| Dimension constants | `crates/context-graph-core/src/config/constants.rs` (TASK-EMB-001) |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants exist)
- [ ] Read current `sparse/types.rs` implementation
- [ ] Understand CSR format for sparse matrices

---

## Scope

### In Scope

- Update SparseVector to use SPARSE_VOCAB_SIZE constant
- Add `to_csr()` method for GPU sparse matrix ops
- Add validation methods
- Remove/deprecate hash-based methods
- Add documentation clarifying projection happens elsewhere

### Out of Scope

- Actual projection logic (TASK-EMB-012)
- Dense output generation (TASK-EMB-012)
- GPU tensor creation (TASK-EMB-012)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs

use crate::config::constants::{SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION};

/// Sparse vector representation for SPLADE-style embeddings.
///
/// # Dimension
/// - Vocabulary dimension: 30522 (BERT WordPiece)
/// - Active dimensions: ~5% (~1500 non-zero entries)
///
/// # Constitution Compliance
/// - Uses learned projection (TASK-EMB-012), NOT hash modulo
/// - Hash-based projection (`idx % dim`) is FORBIDDEN (AP-007)
///
/// # Usage
/// 1. Create sparse vector from tokenizer output
/// 2. Pass to ProjectionMatrix::project() for dense output
/// 3. Dense output is what gets stored/searched
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Active dimension indices (sorted, unique)
    /// Range: [0, SPARSE_VOCAB_SIZE)
    indices: Vec<u32>,

    /// Values at active indices
    /// Typically positive (SPLADE uses ReLU)
    values: Vec<f32>,
}

impl SparseVector {
    /// Create a new sparse vector from indices and values.
    ///
    /// # Arguments
    /// * `indices` - Active dimension indices in [0, SPARSE_VOCAB_SIZE)
    /// * `values` - Corresponding values
    ///
    /// # Panics
    /// - If indices.len() != values.len()
    /// - If any index >= SPARSE_VOCAB_SIZE
    /// - If indices are not sorted
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        assert_eq!(indices.len(), values.len(),
            "Indices and values must have same length");

        // Validate indices
        for (i, &idx) in indices.iter().enumerate() {
            assert!((idx as usize) < SPARSE_VOCAB_SIZE,
                "Index {} out of range: {} >= {}", i, idx, SPARSE_VOCAB_SIZE);
            if i > 0 {
                assert!(idx > indices[i-1],
                    "Indices must be sorted: {} followed by {}", indices[i-1], idx);
            }
        }

        Self { indices, values }
    }

    /// Create empty sparse vector.
    pub fn empty() -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Convert to CSR (Compressed Sparse Row) format for GPU SpMM.
    ///
    /// Returns (row_ptr, col_indices, values) for a 1×SPARSE_VOCAB_SIZE matrix.
    ///
    /// # Returns
    /// - `row_ptr`: [0, nnz] for single-row matrix
    /// - `col_indices`: Column indices (same as self.indices cast to i32)
    /// - `values`: Copy of self.values
    pub fn to_csr(&self) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let nnz = self.indices.len() as i32;
        let row_ptr = vec![0i32, nnz];
        let col_indices: Vec<i32> = self.indices.iter().map(|&i| i as i32).collect();
        let values = self.values.clone();

        (row_ptr, col_indices, values)
    }

    /// Convert to dense representation (for CPU operations).
    ///
    /// # Warning
    /// This creates a full SPARSE_VOCAB_SIZE (30522) element vector.
    /// Use `to_csr()` for GPU operations instead.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; SPARSE_VOCAB_SIZE];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            dense[idx as usize] = val;
        }
        dense
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Sparsity ratio (nnz / vocab_size).
    pub fn sparsity(&self) -> f32 {
        self.indices.len() as f32 / SPARSE_VOCAB_SIZE as f32
    }

    /// Check if sparsity is within expected range (~5% active).
    pub fn has_expected_sparsity(&self) -> bool {
        let sparsity = self.sparsity();
        sparsity > 0.01 && sparsity < 0.15  // 1% to 15% non-zero
    }

    /// Get L1 norm (sum of absolute values).
    pub fn l1_norm(&self) -> f32 {
        self.values.iter().map(|v| v.abs()).sum()
    }

    /// Get L2 norm.
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Normalize to unit L2 norm (in-place).
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 1e-8 {
            for v in &mut self.values {
                *v /= norm;
            }
        }
    }

    /// Get indices slice (read-only).
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get values slice (read-only).
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Vocabulary dimension.
    pub fn dim(&self) -> usize {
        SPARSE_VOCAB_SIZE
    }

    /// Projected output dimension (for reference).
    pub fn projected_dim(&self) -> usize {
        SPARSE_PROJECTED_DIMENSION
    }

    /// Dot product with another sparse vector.
    pub fn dot(&self, other: &Self) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        result
    }

    /// Cosine similarity with another sparse vector.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let norm_self = self.l2_norm();
        let norm_other = other.l2_norm();

        if norm_self > 1e-8 && norm_other > 1e-8 {
            dot / (norm_self * norm_other)
        } else {
            0.0
        }
    }
}

// ============================================================
// DEPRECATED: Hash-based projection methods
// These are kept for reference but MUST NOT be used in production.
// ============================================================

#[deprecated(since = "4.0.0", note = "Use ProjectionMatrix::project() instead. Hash modulo destroys semantic information.")]
impl SparseVector {
    /// DEPRECATED: Hash-based projection.
    ///
    /// # Constitution Violation
    /// This method uses `idx % projected_dim` which DESTROYS semantic information.
    /// It is FORBIDDEN by Constitution AP-007.
    ///
    /// Use `ProjectionMatrix::project()` from TASK-EMB-012 instead.
    #[deprecated]
    pub fn hash_project_deprecated(&self) -> Vec<f32> {
        panic!("[EMB-E099] HASH_PROJECTION_FORBIDDEN: This method is deprecated. \
                Use ProjectionMatrix::project() instead. Constitution AP-007 violation.");
    }
}
```

### Constraints

- MUST use dimension constants from TASK-EMB-001
- MUST NOT contain hash-based projection logic (except deprecated/panic version)
- `to_csr()` format MUST be compatible with cuSPARSE
- Indices MUST be sorted for efficient sparse ops

### Verification

- `new()` validates index bounds
- `to_csr()` returns correct format
- Deprecated methods panic with Constitution reference

---

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs` | Complete rewrite per spec |

---

## Validation Criteria

- [ ] `SparseVector::new()` validates indices are in [0, SPARSE_VOCAB_SIZE)
- [ ] `to_csr()` returns (row_ptr, col_indices, values) tuple
- [ ] `to_dense()` creates SPARSE_VOCAB_SIZE element vector
- [ ] `nnz()`, `sparsity()`, `l1_norm()`, `l2_norm()` methods work
- [ ] `dot()` and `cosine_similarity()` handle sparse-sparse ops
- [ ] Hash projection methods are deprecated and panic
- [ ] `cargo check -p context-graph-embeddings` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings sparse::types:: -- --nocapture
```

---

## Pseudo Code

```
types.rs:
  Import dimension constants (SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION)

  Define SparseVector struct:
    indices: Vec<u32> (sorted, unique)
    values: Vec<f32>

  Impl SparseVector:
    new() - validate bounds, sorted indices
    empty() - create empty vector
    to_csr() - (row_ptr, col_indices, values) for GPU
    to_dense() - full SPARSE_VOCAB_SIZE vector
    nnz() - non-zero count
    sparsity() - nnz / vocab_size
    l1_norm(), l2_norm(), normalize()
    dot(), cosine_similarity() - sparse-sparse ops
    indices(), values() - accessors
    dim(), projected_dim() - dimension info

  Deprecated impl:
    hash_project_deprecated() - PANIC with Constitution reference
```

</task_spec>
