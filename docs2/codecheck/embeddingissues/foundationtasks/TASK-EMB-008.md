# TASK-EMB-008: Update SparseVector Struct - Add to_csr() and Remove to_dense_projected()

<task_spec id="TASK-EMB-008" version="2.0" updated="2026-01-06">

## CRITICAL CONTEXT FOR AI AGENT

**This task is BLOCKED until you verify TASK-EMB-001 is complete.** Before starting, you MUST:
1. Read `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`
2. Verify `SPARSE_PROJECTED_DIMENSION = 1536` (NOT 768)
3. If it's still 768, stop and complete TASK-EMB-001 first

---

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-008 |
| **Title** | Update SparseVector Struct - Add to_csr(), Remove to_dense_projected() |
| **Status** | complete |
| **Layer** | foundation |
| **Sequence** | 8 |
| **Implements** | REQ-EMB-001, REQ-EMB-002 |
| **Depends On** | TASK-EMB-001 (dimension constant fix) |
| **Blocks** | TASK-EMB-012 (ProjectionMatrix logic layer integration) |
| **Estimated Complexity** | low |
| **Constitution Ref** | `E6_Sparse: { dim: "~30K 5%active" }`, `AP-007: No stub data in prod` |

---

## Problem Statement

The `SparseVector::to_dense_projected()` method in `types.rs` uses a **BROKEN hash-based projection**:

```rust
// CURRENT BROKEN CODE - LINE 71-91 in types.rs
pub fn to_dense_projected(&self, projected_dim: usize) -> Vec<f32> {
    let mut dense = vec![0.0f32; projected_dim];
    for (&idx, &weight) in self.indices.iter().zip(&self.weights) {
        let dense_idx = idx % projected_dim;  // <-- HASH COLLISION BUG
        dense[dense_idx] += weight;
    }
    // L2 normalize
    // ...
}
```

### Why This Is Catastrophic

1. **Hash Collisions Destroy Semantics**: Unrelated tokens map to the same dimension (`idx % projected_dim`)
2. **Violates Constitution AP-007**: "No stub data in prod" - hash projection is effectively stub/mock behavior
3. **No Learned Representation**: This is NOT a learned projection - it's random noise masquerading as structure
4. **Currently USED**: `model.rs:291` calls this method: `sparse.to_dense_projected(SPARSE_PROJECTED_DIMENSION)`

### The Fix

1. **REMOVE** `to_dense_projected()` entirely (or mark deprecated with panic)
2. **ADD** `to_csr()` method for cuBLAS sparse matrix operations
3. Callers must migrate to `ProjectionMatrix::project()` (TASK-EMB-012)

---

## Current Codebase State (Verified 2026-01-06)

### File: `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`

**Current contents (relevant sections):**
- Lines 1-45: Module docs, imports, constants
- Lines 46-97: `SparseVector` struct and impl including the broken `to_dense_projected()` method
- `SPARSE_PROJECTED_DIMENSION = 1536` (TASK-EMB-001 sets this)
- `SPARSE_VOCAB_SIZE = 30522`

### File: `crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs`

**Line 291:** Uses `to_dense_projected()`:
```rust
let vector = sparse.to_dense_projected(SPARSE_PROJECTED_DIMENSION);
```

### File: `crates/context-graph-embeddings/src/models/pretrained/sparse/tests.rs`

**Line 222:** Test uses `to_dense_projected()`:
```rust
let dense = sparse.to_dense_projected(SPARSE_PROJECTED_DIMENSION);
```

### File: `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs`

**ALREADY EXISTS (372 lines)** - Contains:
- `ProjectionMatrix` struct (data structure only, no `load()` or `project()` logic yet)
- `ProjectionError` enum with all 5 variants (EMB-E001, E004, E005, E006, E008)
- Constants: `PROJECTION_WEIGHT_FILE`, `PROJECTION_TENSOR_NAME`
- Compile-time assertions for dimensions

### Exports in `mod.rs`

Line 61 exports projection types:
```rust
pub use projection::{ProjectionError, ProjectionMatrix, PROJECTION_TENSOR_NAME, PROJECTION_WEIGHT_FILE};
```

---

## Scope

### In Scope (This Task)
1. Remove `to_dense_projected()` method from `SparseVector` (delete or make it panic)
2. Add `to_csr()` method to `SparseVector` for CSR format conversion
3. Add `nnz()` method for non-zero count
4. Update documentation to reference `ProjectionMatrix::project()`
5. Update test file to remove/modify tests using `to_dense_projected()`
6. **Temporarily** update `model.rs` to panic if `embed()` is called (since projection isn't ready)

### Out of Scope
- `ProjectionMatrix::load()` implementation (TASK-EMB-012 - Logic Layer)
- `ProjectionMatrix::project()` implementation (TASK-EMB-012 - Logic Layer)
- CUDA kernel integration (Logic Layer)
- Weight file downloading/training

---

## Definition of Done

### Exact Changes Required

#### 1. Modify `types.rs` - SparseVector Implementation

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`

**REPLACE lines 46-97 (entire SparseVector struct and impl) with:**

```rust
/// Sparse vector output with term indices and weights.
///
/// # Constitution Alignment
/// - Dimension: SPARSE_VOCAB_SIZE (30522)
/// - Expected sparsity: ~95% zeros (~5% active)
/// - Output after projection: 1536D dense (via ProjectionMatrix)
///
/// # BREAKING CHANGE v4.0.0
/// `to_dense_projected()` has been REMOVED. The hash-based projection
/// (`idx % projected_dim`) destroyed semantic information and violated
/// Constitution AP-007 (no stub data in prod).
///
/// Use `ProjectionMatrix::project()` instead for learned sparse-to-dense
/// conversion that preserves semantic relationships.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector {
    /// Token indices with non-zero weights (sorted ascending).
    pub indices: Vec<usize>,
    /// Corresponding weights for each index.
    pub weights: Vec<f32>,
    /// Total number of dimensions (vocabulary size = 30522).
    pub dimension: usize,
}

impl SparseVector {
    /// Create a new sparse vector.
    ///
    /// # Arguments
    /// * `indices` - Token indices with non-zero weights (should be sorted ascending)
    /// * `weights` - Corresponding weights for each index
    ///
    /// # Invariants
    /// - `indices.len() == weights.len()`
    /// - All indices < SPARSE_VOCAB_SIZE (30522)
    /// - Indices should be sorted ascending (for efficient CSR conversion)
    ///
    /// # Panics
    /// Debug builds will panic if `indices.len() != weights.len()`
    pub fn new(indices: Vec<usize>, weights: Vec<f32>) -> Self {
        debug_assert_eq!(
            indices.len(),
            weights.len(),
            "indices and weights must have same length"
        );
        Self {
            indices,
            weights,
            dimension: SPARSE_VOCAB_SIZE,
        }
    }

    /// Convert to CSR (Compressed Sparse Row) format for cuBLAS.
    ///
    /// CSR format is required for efficient sparse matrix-vector multiplication
    /// with `ProjectionMatrix` using cuBLAS `csrmm2` or similar operations.
    ///
    /// # Returns
    /// `(row_ptr, col_indices, values)` tuple for CSR representation:
    /// - `row_ptr`: Row pointers [0, nnz] for single-row sparse matrix
    /// - `col_indices`: Column indices (the token indices as i32)
    /// - `values`: Non-zero values (the weights)
    ///
    /// # Implementation Note
    /// For a single vector (1 row), CSR format is:
    /// - `row_ptr = [0, nnz]` where nnz = number of non-zero elements
    /// - `col_indices = indices` converted to i32
    /// - `values = weights`
    ///
    /// # Example
    /// ```rust,ignore
    /// let sparse = SparseVector::new(vec![10, 100, 500], vec![0.5, 0.3, 0.8]);
    /// let (row_ptr, col_idx, vals) = sparse.to_csr();
    /// assert_eq!(row_ptr, vec![0, 3]);
    /// assert_eq!(col_idx, vec![10, 100, 500]);
    /// assert_eq!(vals, vec![0.5, 0.3, 0.8]);
    /// ```
    pub fn to_csr(&self) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
        let nnz = self.indices.len() as i32;
        let row_ptr = vec![0i32, nnz];
        let col_indices: Vec<i32> = self.indices.iter().map(|&i| i as i32).collect();
        let values = self.weights.clone();
        (row_ptr, col_indices, values)
    }

    /// Get number of non-zero elements.
    ///
    /// # Returns
    /// Count of active (non-zero weight) indices in this sparse vector.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get sparsity as ratio of zeros (0.0 to 1.0).
    ///
    /// # Returns
    /// Sparsity ratio: `1.0 - (nnz / dimension)`
    ///
    /// # Example
    /// A vector with 150 non-zero elements out of 30522:
    /// `sparsity = 1.0 - 150/30522 = 0.9951` (~99.5% sparse)
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.indices.len() as f32 / self.dimension as f32)
    }

    // =========================================================================
    // REMOVED: to_dense_projected()
    // =========================================================================
    // The hash-based projection (`idx % projected_dim`) has been DELETED.
    // It violated Constitution AP-007 by using stub/mock behavior in production.
    //
    // Hash collision example:
    //   Token "machine" (idx 3057) and "learning" (idx 4593) would collide
    //   if 3057 % 1536 == 4593 % 1536 (which destroys semantic meaning)
    //
    // Migration path:
    //   OLD: let dense = sparse.to_dense_projected(1536);
    //   NEW: let dense = projection_matrix.project(&sparse)?;
    //
    // See TASK-EMB-012 for ProjectionMatrix implementation.
    // =========================================================================
}

impl Default for SparseVector {
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            weights: Vec::new(),
            dimension: SPARSE_VOCAB_SIZE,
        }
    }
}
```

#### 2. Modify `model.rs` - Temporary Panic Until TASK-EMB-012

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs`

**REPLACE lines 276-295 (the embed function) with:**

```rust
    /// Embed input to dense 1536D vector (for multi-array storage compatibility).
    /// Per Constitution E6_Sparse: "~30K 5%active" projects to 1536D.
    ///
    /// # TEMPORARY: PANICS until TASK-EMB-012 is complete
    ///
    /// The hash-based `to_dense_projected()` has been removed per Constitution AP-007.
    /// This method will panic until `ProjectionMatrix` integration is complete.
    ///
    /// # Panics
    /// Always panics with clear message indicating migration needed.
    pub async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: self.model_id(),
            });
        }

        self.validate_input(input)?;

        // CRITICAL: to_dense_projected() has been removed (Constitution AP-007)
        // ProjectionMatrix integration is required (TASK-EMB-012)
        panic!(
            "[EMB-MIGRATION] SparseModel::embed() is temporarily unavailable.\n\
             The hash-based projection was removed (violated AP-007).\n\
             Waiting for ProjectionMatrix integration (TASK-EMB-012).\n\
             For sparse output, use embed_sparse() directly."
        );
    }
```

#### 3. Modify `tests.rs` - Update Tests

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/tests.rs`

**REPLACE lines 217-243 (sparse vector tests) with:**

```rust
    // ==================== Sparse Vector Tests ====================

    #[test]
    fn test_sparse_vector_new() {
        let indices = vec![10, 100, 500];
        let weights = vec![0.5, 0.3, 0.8];
        let sparse = SparseVector::new(indices.clone(), weights.clone());

        assert_eq!(sparse.indices, indices);
        assert_eq!(sparse.weights, weights);
        assert_eq!(sparse.dimension, SPARSE_VOCAB_SIZE);
        assert_eq!(sparse.dimension, 30522);
    }

    #[test]
    fn test_sparse_vector_to_csr() {
        let sparse = SparseVector::new(vec![10, 100, 500], vec![0.5, 0.3, 0.8]);
        let (row_ptr, col_indices, values) = sparse.to_csr();

        // CSR format verification
        assert_eq!(row_ptr, vec![0, 3], "row_ptr must be [0, nnz]");
        assert_eq!(col_indices, vec![10, 100, 500], "col_indices must match indices as i32");
        assert_eq!(values, vec![0.5, 0.3, 0.8], "values must match weights");
    }

    #[test]
    fn test_sparse_vector_to_csr_empty() {
        // EDGE CASE 1: Empty sparse vector (boundary - empty input)
        let sparse = SparseVector::new(vec![], vec![]);
        let (row_ptr, col_indices, values) = sparse.to_csr();

        assert_eq!(row_ptr, vec![0, 0], "Empty vector: row_ptr = [0, 0]");
        assert!(col_indices.is_empty(), "Empty vector: no col_indices");
        assert!(values.is_empty(), "Empty vector: no values");
        assert_eq!(sparse.nnz(), 0, "Empty vector: nnz = 0");
    }

    #[test]
    fn test_sparse_vector_to_csr_single_element() {
        // EDGE CASE 2: Single element (boundary - minimum non-empty)
        let sparse = SparseVector::new(vec![12345], vec![0.99]);
        let (row_ptr, col_indices, values) = sparse.to_csr();

        assert_eq!(row_ptr, vec![0, 1], "Single element: row_ptr = [0, 1]");
        assert_eq!(col_indices, vec![12345], "Single element: col_indices");
        assert_eq!(values, vec![0.99], "Single element: values");
        assert_eq!(sparse.nnz(), 1, "Single element: nnz = 1");
    }

    #[test]
    fn test_sparse_vector_to_csr_max_index() {
        // EDGE CASE 3: Maximum valid index (boundary - max vocab index)
        let max_idx = SPARSE_VOCAB_SIZE - 1; // 30521
        let sparse = SparseVector::new(vec![0, max_idx], vec![0.1, 0.2]);
        let (row_ptr, col_indices, values) = sparse.to_csr();

        assert_eq!(row_ptr, vec![0, 2]);
        assert_eq!(col_indices, vec![0, 30521]);
        assert_eq!(values, vec![0.1, 0.2]);
    }

    #[test]
    fn test_sparse_vector_nnz() {
        let sparse = SparseVector::new(vec![0, 100, 500, 1000, 2000], vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        assert_eq!(sparse.nnz(), 5);
    }

    #[test]
    fn test_sparse_vector_sparsity() {
        let sparse = SparseVector::new(vec![0, 100, 500], vec![1.0, 0.5, 0.8]);

        // 3 non-zero out of 30522 = ~99.99% sparse
        let sparsity = sparse.sparsity();
        assert!(sparsity > 0.99, "Sparsity should be >99%, got {}", sparsity);

        // More precise check: 1.0 - 3/30522 = 0.999901...
        let expected = 1.0 - (3.0 / 30522.0);
        assert!((sparsity - expected).abs() < 0.0001, "Sparsity mismatch: {} vs {}", sparsity, expected);
    }

    #[test]
    fn test_sparse_vector_default() {
        let sparse = SparseVector::default();
        assert!(sparse.indices.is_empty());
        assert!(sparse.weights.is_empty());
        assert_eq!(sparse.dimension, SPARSE_VOCAB_SIZE);
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.sparsity(), 1.0);
    }

    #[test]
    fn test_sparse_vector_equality() {
        let sparse1 = SparseVector::new(vec![10, 20], vec![0.5, 0.6]);
        let sparse2 = SparseVector::new(vec![10, 20], vec![0.5, 0.6]);
        let sparse3 = SparseVector::new(vec![10, 20], vec![0.5, 0.7]);

        assert_eq!(sparse1, sparse2, "Same content should be equal");
        assert_ne!(sparse1, sparse3, "Different weights should not be equal");
    }
```

**ALSO MODIFY test `test_embed_returns_1536d_vector` (around line 140):**

**REPLACE with:**

```rust
    #[tokio::test]
    #[should_panic(expected = "EMB-MIGRATION")]
    async fn test_embed_panics_until_projection_ready() {
        let model = create_and_load_model().await;
        let input = ModelInput::text("Test sparse embedding").expect("Input");

        // This SHOULD panic until ProjectionMatrix integration is complete
        let _ = model.embed(&input).await;
    }
```

**ALSO MODIFY or REMOVE tests that call `embed()` expecting success:**
- `test_embed_returns_l2_normalized_vector` -> Remove or convert to #[should_panic]
- `test_embed_deterministic` -> Remove or convert to #[should_panic]
- `test_embed_different_inputs_differ` -> Remove or convert to #[should_panic]
- `test_embed_model_id_is_sparse` -> Remove or convert to #[should_panic]

---

## Verification Commands

Execute ALL commands in order. ALL must pass:

```bash
cd /home/cabdru/contextgraph

# 1. Verify TASK-EMB-001 prerequisite (MUST show 1536)
grep -n "SPARSE_PROJECTED_DIMENSION.*=" crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs
# Expected: pub const SPARSE_PROJECTED_DIMENSION: usize = 1536;

# 2. Compile check (no errors)
cargo check -p context-graph-embeddings 2>&1 | head -20

# 3. Verify to_dense_projected is REMOVED
grep -n "to_dense_projected" crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs
# Expected: Only comment references, no function definition

# 4. Verify to_csr EXISTS
grep -n "pub fn to_csr" crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs
# Expected: pub fn to_csr(&self) -> (Vec<i32>, Vec<i32>, Vec<f32>)

# 5. Run sparse module tests
cargo test -p context-graph-embeddings sparse::tests -- --nocapture 2>&1 | tail -30

# 6. Verify model.rs has panic (temporarily)
grep -n "TASK-EMB-012\|EMB-MIGRATION" crates/context-graph-embeddings/src/models/pretrained/sparse/model.rs
# Expected: Reference to TASK-EMB-012 and EMB-MIGRATION panic
```

---

## Full State Verification Protocol

After completing changes, you MUST execute this verification:

### Source of Truth Locations

| Data | Location | How to Verify |
|------|----------|---------------|
| SparseVector struct | `sparse/types.rs` | Read the file, verify struct definition |
| to_csr method | `sparse/types.rs` | `grep "pub fn to_csr"` |
| to_dense_projected removed | `sparse/types.rs` | `grep -c "fn to_dense_projected"` should return 0 |
| Tests pass | Test output | `cargo test sparse::tests` |

### Execute & Inspect Protocol

1. **Read Operation - Verify struct definition:**
```bash
cargo doc -p context-graph-embeddings --no-deps --document-private-items 2>/dev/null
grep -A30 "impl SparseVector" crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs
```

2. **Test Execution - Run tests and capture output:**
```bash
cargo test -p context-graph-embeddings sparse::tests -- --nocapture 2>&1 | tee /tmp/sparse_test_output.txt
cat /tmp/sparse_test_output.txt | grep -E "test|ok|FAILED|passed"
```

### Boundary & Edge Case Audit

You MUST manually simulate these 3 edge cases and print before/after state:

**Edge Case 1: Empty Input**
```rust
// Print BEFORE state:
println!("EDGE CASE 1: Empty sparse vector");
let sparse = SparseVector::new(vec![], vec![]);
println!("  indices.len() = {}", sparse.indices.len());
println!("  weights.len() = {}", sparse.weights.len());
println!("  nnz() = {}", sparse.nnz());

// Execute operation
let (row_ptr, col_indices, values) = sparse.to_csr();

// Print AFTER state:
println!("  row_ptr = {:?}", row_ptr);
println!("  col_indices = {:?}", col_indices);
println!("  values = {:?}", values);
println!("  PASS: row_ptr=[0,0], empty arrays for empty input");
```

**Edge Case 2: Maximum Index**
```rust
// Print BEFORE state:
println!("EDGE CASE 2: Maximum vocab index (30521)");
let sparse = SparseVector::new(vec![30521], vec![1.0]);
println!("  indices = {:?}", sparse.indices);

// Execute operation
let (row_ptr, col_indices, values) = sparse.to_csr();

// Print AFTER state:
println!("  col_indices = {:?}", col_indices);
println!("  PASS: Max index 30521 converted to i32 correctly");
```

**Edge Case 3: Many Elements (Typical SPLADE output)**
```rust
// Print BEFORE state:
println!("EDGE CASE 3: Typical SPLADE output (~150 active terms)");
let indices: Vec<usize> = (0..150).map(|i| i * 200).collect();
let weights: Vec<f32> = (0..150).map(|i| 0.01 * (i as f32)).collect();
let sparse = SparseVector::new(indices, weights);
println!("  nnz() = {}", sparse.nnz());
println!("  sparsity() = {:.4}", sparse.sparsity());

// Execute operation
let (row_ptr, col_indices, values) = sparse.to_csr();

// Print AFTER state:
println!("  row_ptr[1] = {} (must equal nnz)", row_ptr[1]);
println!("  col_indices.len() = {}", col_indices.len());
println!("  values.len() = {}", values.len());
println!("  PASS: 150 elements converted correctly");
```

### Evidence of Success

Provide a log showing:
```
========================================
TASK-EMB-008 VERIFICATION LOG
========================================

1. PREREQUISITE CHECK:
   SPARSE_PROJECTED_DIMENSION = 1536 (CONFIRMED)

2. METHOD REMOVAL CHECK:
   to_dense_projected function definition: NOT FOUND (REMOVED)
   to_dense_projected in comments only: YES

3. METHOD ADDITION CHECK:
   to_csr method: FOUND at line XX
   nnz method: FOUND at line XX

4. COMPILE CHECK:
   cargo check: SUCCESS (no errors)

5. TEST RESULTS:
   test sparse::tests::test_sparse_vector_new ... ok
   test sparse::tests::test_sparse_vector_to_csr ... ok
   test sparse::tests::test_sparse_vector_to_csr_empty ... ok
   test sparse::tests::test_sparse_vector_to_csr_single_element ... ok
   test sparse::tests::test_sparse_vector_to_csr_max_index ... ok
   test sparse::tests::test_sparse_vector_nnz ... ok
   test sparse::tests::test_sparse_vector_sparsity ... ok
   test sparse::tests::test_sparse_vector_default ... ok
   test sparse::tests::test_sparse_vector_equality ... ok

6. EDGE CASE AUDIT:
   Empty input: PASS (row_ptr=[0,0])
   Max index: PASS (30521 as i32)
   150 elements: PASS (typical SPLADE)

7. MODEL PANIC CHECK:
   embed() method: PANICS with EMB-MIGRATION message

STATUS: COMPLETE
========================================
```

---

## Files to Modify

| File | Line Numbers | Change Type | Description |
|------|--------------|-------------|-------------|
| `sparse/types.rs` | 46-97 | REPLACE | New SparseVector impl with to_csr(), remove to_dense_projected() |
| `sparse/model.rs` | 276-295 | REPLACE | Temporary panic in embed() until TASK-EMB-012 |
| `sparse/tests.rs` | 140-165, 217-243 | REPLACE | Update tests for new API |

---

## What NOT to Change

| File | Item | Reason |
|------|------|--------|
| `sparse/types.rs` | `SPARSE_VOCAB_SIZE` | Correct at 30522 |
| `sparse/types.rs` | `SPARSE_PROJECTED_DIMENSION` | Should be 1536 (TASK-EMB-001) |
| `sparse/types.rs` | `SPARSE_HIDDEN_SIZE` | Correct at 768 (BERT hidden) |
| `sparse/projection.rs` | All | Already has ProjectionMatrix struct |
| `sparse/mod.rs` | Exports | Already correct |

---

## Dependencies

### This Task Requires (Prerequisites)
- **TASK-EMB-001**: `SPARSE_PROJECTED_DIMENSION` must be 1536

### This Task Enables (Unblocks)
- **TASK-EMB-012**: ProjectionMatrix logic implementation needs to_csr() for cuBLAS

---

## Traceability

| Requirement | Constitution Section | Code Location |
|-------------|---------------------|---------------|
| E6 Sparse Dimension | `embeddings.models.E6_Sparse.dim` | `SPARSE_VOCAB_SIZE = 30522` |
| No Stub Data | `forbidden.AP-007` | Remove hash projection |
| CSR for cuBLAS | `stack.gpu` | `to_csr()` method |

---

## Notes for Implementing Agent

1. **DO NOT** create fallback behavior. If something doesn't work, it should error.
2. **DO NOT** use mock data in tests. Use real values and verify correctness.
3. The `embed()` panic is INTENTIONAL. It prevents using broken code in production.
4. After TASK-EMB-012 completes, the panic will be replaced with `ProjectionMatrix::project()`.
5. Tests MUST verify actual CSR output, not mock expectations.

---

## Anti-Patterns to Avoid

- **AP-007 VIOLATION**: DO NOT add any form of fallback for missing projection
- **Hidden Failures**: DO NOT silently return empty or zero vectors
- **Mock Tests**: DO NOT use mocked data that passes regardless of implementation
- **Backwards Compatibility**: DO NOT keep to_dense_projected() "for compatibility"

</task_spec>
