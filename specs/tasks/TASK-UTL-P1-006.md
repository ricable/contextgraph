# TASK-UTL-P1-006: Implement MaxSimTokenEntropy for E12 (LateInteraction/EmotionalValence)

**Priority:** P1
**Status:** pending
**Spec Reference:** SPEC-UTL-003
**Estimated Effort:** 3-4 hours
**Implements:** REQ-UTL-003-07, REQ-UTL-003-08

---

## Summary

Create a specialized entropy calculator for E12 (LateInteraction) embeddings using MaxSim aggregation. ColBERT-style late interaction embeddings consist of variable-length sequences of per-token vectors (128D each). The MaxSim metric finds, for each query token, its maximum similarity with any document token, then aggregates these scores. This captures token-level semantic precision that simple vector averaging loses.

---

## Input Context Files

Read these files before implementation:

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` | `EmbedderEntropy` trait definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | Factory routing (to be updated) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/default_knn.rs` | Reference: KNN baseline |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/teleological/embedder.rs` | `Embedder::LateInteraction` definition (index 11) |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint.rs` | `E12_TOKEN_DIM = 128` constant |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/config.rs` | `SurpriseConfig` for parameters |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/error.rs` | `UtlError`, `UtlResult` types |

---

## Definition of Done

### 1. Create New File

**Path:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/maxsim_token.rs`

**Exact Signatures Required:**

```rust
/// E12 (LateInteraction) entropy using ColBERT-style MaxSim aggregation.
///
/// Late interaction embeddings are variable-length sequences of per-token
/// vectors (128D each). MaxSim finds, for each query token, its maximum
/// cosine similarity with any document token, then averages these scores.
///
/// # Algorithm
///
/// 1. Reshape current embedding into tokens (chunks of 128D)
/// 2. For each history embedding:
///    a. Reshape into tokens
///    b. For each current token, find max similarity to any history token
///    c. Average the per-token max similarities = MaxSim score
/// 3. ΔS = 1 - avg(top-k MaxSim scores), clamped to [0, 1]
///
/// # Token Representation
///
/// Embeddings are stored as flattened Vec<f32> with length = num_tokens * 128.
/// - 128 elements = 1 token
/// - 256 elements = 2 tokens
/// - etc.
///
/// Empty or non-divisible lengths are handled gracefully.
#[derive(Debug, Clone)]
pub struct MaxSimTokenEntropy {
    /// Per-token embedding dimension. Default: 128 (ColBERT standard)
    token_dim: usize,
    /// Minimum token count to consider valid. Default: 1
    min_tokens: usize,
    /// Running mean for score normalization.
    running_mean: f32,
    /// Running variance for score normalization.
    running_variance: f32,
    /// Sample count for statistics.
    sample_count: usize,
    /// EMA alpha.
    ema_alpha: f32,
}

impl MaxSimTokenEntropy {
    /// Create a new MaxSim token entropy calculator.
    pub fn new() -> Self;

    /// Create with a specific token dimension.
    ///
    /// # Arguments
    /// * `token_dim` - Per-token dimension (typically 128)
    pub fn with_token_dim(token_dim: usize) -> Self;

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self;

    /// Set the minimum token count for valid embeddings.
    pub fn with_min_tokens(self, min_tokens: usize) -> Self;

    /// Reshape a flat embedding into token vectors.
    ///
    /// # Arguments
    /// * `embedding` - Flat f32 slice of length num_tokens * token_dim
    ///
    /// # Returns
    /// Vector of token slices, each of length token_dim.
    /// Returns empty vec if embedding is not evenly divisible.
    fn tokenize<'a>(&self, embedding: &'a [f32]) -> Vec<&'a [f32]>;

    /// Compute cosine similarity between two token vectors.
    fn token_similarity(&self, a: &[f32], b: &[f32]) -> f32;

    /// Compute MaxSim score between two tokenized embeddings.
    ///
    /// For each token in `query`, finds the max similarity to any token in `doc`.
    /// Returns the average of these max similarities.
    ///
    /// # Arguments
    /// * `query_tokens` - Tokenized query embedding
    /// * `doc_tokens` - Tokenized document embedding
    ///
    /// # Returns
    /// MaxSim score in [0, 1]. Returns 0.0 if either is empty.
    fn compute_maxsim(&self, query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32;
}

impl Default for MaxSimTokenEntropy {
    fn default() -> Self;
}

impl EmbedderEntropy for MaxSimTokenEntropy {
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
mod maxsim_token;
pub use maxsim_token::MaxSimTokenEntropy;
```

### 3. Update Factory Routing

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs`

Change the `Embedder::LateInteraction` arm from:
```rust
Embedder::LateInteraction => Box::new(DefaultKnnEntropy::from_config(embedder, config))
```

To:
```rust
Embedder::LateInteraction => Box::new(MaxSimTokenEntropy::from_config(config))
```

### 4. Required Tests

Implement in `maxsim_token.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_maxsim_empty_history_returns_one` | Empty history = max surprise |
| `test_maxsim_identical_returns_near_zero` | Same tokens = ΔS ≈ 0 |
| `test_maxsim_orthogonal_returns_near_one` | Orthogonal tokens = high ΔS |
| `test_maxsim_partial_overlap` | Some matching tokens = moderate ΔS |
| `test_maxsim_empty_input_error` | Empty current = EmptyInput error |
| `test_maxsim_embedder_type` | Returns `Embedder::LateInteraction` |
| `test_maxsim_valid_range` | All outputs in [0, 1] |
| `test_maxsim_no_nan_infinity` | No NaN/Infinity outputs |
| `test_maxsim_variable_length_query` | Different query lengths work |
| `test_maxsim_variable_length_doc` | Different doc lengths work |
| `test_maxsim_single_token` | Degenerates to cosine similarity |
| `test_maxsim_tokenize_valid` | 256 elements -> 2 tokens |
| `test_maxsim_tokenize_invalid` | 257 elements -> empty (not divisible) |
| `test_maxsim_from_config` | Config values applied |
| `test_maxsim_reset` | State clears properly |

---

## Validation Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Compiles | `cargo build -p context-graph-utl` | Success |
| Tests pass | `cargo test -p context-graph-utl maxsim_token` | All tests pass |
| No warnings | `cargo clippy -p context-graph-utl -- -D warnings` | No warnings |
| Factory routes correctly | Run factory test | E12 -> MaxSimTokenEntropy |

---

## Implementation Notes

1. **Token Dimension**: E12 uses 128D per token (ColBERT standard). This is defined in `E12_TOKEN_DIM` constant.

2. **Variable Length Handling**: Different documents have different token counts. The current embedding might have 5 tokens while a history item has 10. MaxSim handles this naturally by only requiring max-matching per query token.

3. **MaxSim Formula**:
   ```
   MaxSim(Q, D) = (1/|Q|) * Σ_{q ∈ Q} max_{d ∈ D} cos(q, d)
   ```
   For each query token q, find the document token d that maximizes similarity.

4. **Computational Complexity**: O(|Q| * |D|) per comparison (all pairs of tokens). For typical token counts (5-50), this is manageable.

5. **Edge Cases**:
   - Empty tokens (length not divisible by 128): Skip gracefully
   - Single token: Degenerates to simple cosine similarity
   - Zero-length history items: Skip in averaging

6. **Config Fields**: May need to add:
   - `late_interaction_token_dim: usize` (default 128)
   - `late_interaction_min_tokens: usize` (default 1)

---

## Mathematical Background

ColBERT MaxSim is designed for passage retrieval with token-level granularity:

```
MaxSim(Q, D) = Σ_{i=1}^{|Q|} max_{j=1}^{|D|} E_q[i] · E_d[j]
```

Properties:
- Asymmetric: MaxSim(Q, D) ≠ MaxSim(D, Q)
- Token-level: Captures fine-grained semantic matching
- Soft matching: Synonyms/paraphrases can match different tokens

For entropy:
```
ΔS = 1 - avg(top-k MaxSim scores)
```

High MaxSim = similar content = low surprise
Low MaxSim = different content = high surprise

---

## Example

Current embedding (2 tokens):
```
[0.5, 0.5, ...(128)..., 0.7, 0.3, ...(128)...]
```

History embedding (3 tokens):
```
[0.5, 0.5, ..., 0.1, 0.9, ..., 0.8, 0.2, ...]
```

MaxSim calculation:
1. Token 0 of current vs all 3 history tokens -> max similarity
2. Token 1 of current vs all 3 history tokens -> max similarity
3. Average = MaxSim score

---

## Rollback Plan

If implementation causes issues:
1. Revert factory routing to `DefaultKnnEntropy` for `Embedder::LateInteraction`
2. Remove `maxsim_token.rs` from module exports
3. Delete `maxsim_token.rs` file

---

## Related Tasks

- **TASK-UTL-P1-003**: JaccardCodeEntropy for E7
- **TASK-UTL-P1-004**: CrossModalEntropy for E10
- **TASK-UTL-P1-005**: ExponentialDecayEntropy for E11
