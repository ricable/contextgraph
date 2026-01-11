# TASK-UTL-P1-003: Implement JaccardCodeEntropy for E7 (Code/ContentHash)

**Priority:** P1
**Status:** pending
**Spec Reference:** SPEC-UTL-003
**Estimated Effort:** 2-3 hours
**Implements:** REQ-UTL-003-01, REQ-UTL-003-02

---

## Summary

Create a specialized entropy calculator for E7 (Code) embeddings using Jaccard distance on active feature sets. Code embeddings from Qodo-Embed (1536D) represent AST-like structural features where many dimensions are near-zero. Jaccard similarity of "active" dimensions (above threshold) is more semantically meaningful than Euclidean/cosine distance.

---

## Input Context Files

Read these files before implementation:

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` | `EmbedderEntropy` trait definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | Factory routing (to be updated) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/jaccard_active.rs` | Reference: Jaccard implementation for E13 |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/teleological/embedder.rs` | `Embedder::Code` definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/config.rs` | `SurpriseConfig` for thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/error.rs` | `UtlError`, `UtlResult` types |

---

## Definition of Done

### 1. Create New File

**Path:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/jaccard_code.rs`

**Exact Signatures Required:**

```rust
/// E7 (Code) entropy using Jaccard similarity of active feature dimensions.
///
/// Code embeddings represent AST/structural features where most dimensions are
/// near-zero. This calculator measures surprise by the overlap of active
/// (above-threshold) dimensions.
///
/// # Algorithm
///
/// 1. Extract active dimensions (|value| > threshold) from current embedding
/// 2. For each history item, compute Jaccard similarity of active dimensions
/// 3. Average the top-k Jaccard similarities
/// 4. ΔS = 1 - avg_jaccard, clamped to [0, 1]
#[derive(Debug, Clone)]
pub struct JaccardCodeEntropy {
    /// Activation threshold for considering a dimension "active" in code features.
    /// Default: 0.01 (code features often have small but meaningful activations)
    activation_threshold: f32,
    /// Smoothing factor for edge cases (empty sets).
    smoothing: f32,
}

impl JaccardCodeEntropy {
    /// Create a new Jaccard code entropy calculator.
    pub fn new() -> Self;

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self;

    /// Set the activation threshold.
    /// Code features use 0.01 default (lower than SPLADE's 0.0) because
    /// code embeddings have denser meaningful activations.
    pub fn with_threshold(self, threshold: f32) -> Self;

    /// Set the smoothing factor.
    pub fn with_smoothing(self, smoothing: f32) -> Self;

    /// Extract active dimension indices from embedding.
    fn get_active_dims(&self, embedding: &[f32]) -> HashSet<usize>;

    /// Compute Jaccard similarity between two sets.
    fn jaccard_similarity(&self, a: &HashSet<usize>, b: &HashSet<usize>) -> f32;
}

impl Default for JaccardCodeEntropy {
    fn default() -> Self;
}

impl EmbedderEntropy for JaccardCodeEntropy {
    fn compute_delta_s(
        &self,
        current: &[f32],
        history: &[Vec<f32>],
        k: usize,
    ) -> UtlResult<f32>;

    fn embedder_type(&self) -> Embedder;

    fn reset(&mut self);
}
```

### 2. Update Module Exports

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs`

Add:
```rust
mod jaccard_code;
pub use jaccard_code::JaccardCodeEntropy;
```

### 3. Update Factory Routing

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs`

Change the `Embedder::Code` arm from:
```rust
Embedder::Code => Box::new(DefaultKnnEntropy::from_config(embedder, config))
```

To:
```rust
Embedder::Code => Box::new(JaccardCodeEntropy::from_config(config))
```

### 4. Required Tests

Implement in `jaccard_code.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_jaccard_code_empty_history_returns_one` | Empty history = max surprise |
| `test_jaccard_code_identical_returns_near_zero` | Same embedding = low ΔS |
| `test_jaccard_code_disjoint_returns_near_one` | No overlap = high ΔS |
| `test_jaccard_code_partial_overlap` | 50% overlap = moderate ΔS |
| `test_jaccard_code_empty_input_error` | Empty current = EmptyInput error |
| `test_jaccard_code_embedder_type` | Returns `Embedder::Code` |
| `test_jaccard_code_valid_range` | All outputs in [0, 1] |
| `test_jaccard_code_no_nan_infinity` | No NaN/Infinity outputs |
| `test_jaccard_code_threshold_affects_active_dims` | Threshold changes active set |
| `test_jaccard_code_from_config` | Config values applied |

---

## Validation Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Compiles | `cargo build -p context-graph-utl` | Success |
| Tests pass | `cargo test -p context-graph-utl jaccard_code` | All tests pass |
| No warnings | `cargo clippy -p context-graph-utl -- -D warnings` | No warnings |
| Factory routes correctly | Run factory test | E7 -> JaccardCodeEntropy |

---

## Implementation Notes

1. **Threshold Selection**: Use 0.01 default for code (vs 0.0 for SPLADE) because code embeddings have denser feature activations. Code AST features are less sparse than keyword expansions.

2. **Code vs SPLADE Jaccard**: Although both use Jaccard, they differ in:
   - Threshold (0.01 vs 0.0)
   - Interpretation (structural features vs lexical terms)
   - Configuration source (`code_activation_threshold` vs `splade_activation_threshold`)

3. **Config Fields**: May need to add `code_activation_threshold: f32` to `SurpriseConfig` if not present. Default to 0.01.

4. **Reference Implementation**: Use `jaccard_active.rs` as a template but adjust for code-specific semantics.

---

## Rollback Plan

If implementation causes issues:
1. Revert factory routing to `DefaultKnnEntropy` for `Embedder::Code`
2. Remove `jaccard_code.rs` from module exports
3. Delete `jaccard_code.rs` file

---

## Related Tasks

- **TASK-UTL-P1-004**: CrossModalEntropy for E10
- **TASK-UTL-P1-005**: ExponentialDecayEntropy for E11
- **TASK-UTL-P1-006**: MaxSimTokenEntropy for E12
