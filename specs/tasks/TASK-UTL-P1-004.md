# TASK-UTL-P1-004: Implement CrossModalEntropy for E10 (Multimodal/SemanticContext)

**Priority:** P1
**Status:** pending
**Spec Reference:** SPEC-UTL-003
**Estimated Effort:** 3-4 hours
**Implements:** REQ-UTL-003-03, REQ-UTL-003-04

---

## Summary

Create a specialized entropy calculator for E10 (Multimodal) embeddings that uses cross-modal distance metrics. Multimodal embeddings from CLIP (768D) represent semantic concepts that may span text, code, and visual modalities. The entropy calculation should account for domain-specific semantic coherence with modality-aware weighting.

---

## Input Context Files

Read these files before implementation:

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` | `EmbedderEntropy` trait definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | Factory routing (to be updated) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/gmm_mahalanobis.rs` | Reference: GMM-based entropy for E1 |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/default_knn.rs` | Reference: KNN baseline |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/teleological/embedder.rs` | `Embedder::Multimodal` definition (index 9) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/config.rs` | `SurpriseConfig` for parameters |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/error.rs` | `UtlError`, `UtlResult` types |

---

## Definition of Done

### 1. Create New File

**Path:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/cross_modal.rs`

**Exact Signatures Required:**

```rust
/// E10 (Multimodal) entropy using cross-modal distance metrics.
///
/// Multimodal embeddings from CLIP represent concepts that span text, code,
/// and visual modalities. This calculator computes surprise using a weighted
/// combination of intra-modal coherence and cross-modal semantic alignment.
///
/// # Algorithm
///
/// 1. Compute cosine distances to all history embeddings
/// 2. Apply domain coherence weighting based on embedding characteristics:
///    - Intra-modal weight: embeddings with similar activation patterns
///    - Cross-modal weight: embeddings with different activation patterns
/// 3. Compute weighted KNN distance with modal awareness
/// 4. Normalize via sigmoid, clamp to [0, 1]
///
/// # Modality Detection Heuristic
///
/// CLIP embeddings have different activation patterns for different modalities:
/// - Text: Higher activation in lower dimensions (linguistic features)
/// - Visual: Higher activation in higher dimensions (perceptual features)
/// - This is detected via first-half vs second-half energy ratio
#[derive(Debug, Clone)]
pub struct CrossModalEntropy {
    /// Weight for intra-modal (same modality) comparisons. Default: 0.7
    intra_modal_weight: f32,
    /// Weight for cross-modal (different modality) comparisons. Default: 0.3
    cross_modal_weight: f32,
    /// Running mean for distance normalization.
    running_mean: f32,
    /// Running variance for distance normalization.
    running_variance: f32,
    /// Number of samples seen.
    sample_count: usize,
    /// EMA alpha for updating statistics.
    ema_alpha: f32,
}

impl CrossModalEntropy {
    /// Create a new cross-modal entropy calculator.
    pub fn new() -> Self;

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self;

    /// Set the intra-modal weight (0.0 to 1.0).
    /// Higher values give more weight to same-modality comparisons.
    pub fn with_intra_weight(self, weight: f32) -> Self;

    /// Set the cross-modal weight (0.0 to 1.0).
    /// Higher values give more weight to cross-modality comparisons.
    pub fn with_cross_weight(self, weight: f32) -> Self;

    /// Detect modality indicator from embedding.
    /// Returns a value in [0, 1] where:
    /// - 0.0 = strongly text-like (energy in lower dimensions)
    /// - 1.0 = strongly visual-like (energy in higher dimensions)
    /// - 0.5 = balanced/hybrid
    fn detect_modality_indicator(&self, embedding: &[f32]) -> f32;

    /// Compute modality-weighted distance between two embeddings.
    fn compute_modal_distance(
        &self,
        current: &[f32],
        other: &[f32],
        current_modality: f32,
        other_modality: f32,
    ) -> f32;

    /// Sigmoid normalization function.
    fn sigmoid(x: f32) -> f32;
}

impl Default for CrossModalEntropy {
    fn default() -> Self;
}

impl EmbedderEntropy for CrossModalEntropy {
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
mod cross_modal;
pub use cross_modal::CrossModalEntropy;
```

### 3. Update Factory Routing

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs`

Change the `Embedder::Multimodal` arm from:
```rust
Embedder::Multimodal => Box::new(DefaultKnnEntropy::from_config(embedder, config))
```

To:
```rust
Embedder::Multimodal => Box::new(CrossModalEntropy::from_config(config))
```

### 4. Required Tests

Implement in `cross_modal.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_cross_modal_empty_history_returns_one` | Empty history = max surprise |
| `test_cross_modal_identical_returns_low` | Same embedding = low ΔS |
| `test_cross_modal_different_modality_adjusted` | Cross-modal distance weighted |
| `test_cross_modal_same_modality_baseline` | Intra-modal uses baseline distance |
| `test_cross_modal_empty_input_error` | Empty current = EmptyInput error |
| `test_cross_modal_embedder_type` | Returns `Embedder::Multimodal` |
| `test_cross_modal_valid_range` | All outputs in [0, 1] |
| `test_cross_modal_no_nan_infinity` | No NaN/Infinity outputs |
| `test_cross_modal_modality_detection` | Text-like vs visual-like detection |
| `test_cross_modal_from_config` | Config values applied |
| `test_cross_modal_weight_balance` | Intra + cross weights sum correctly |
| `test_cross_modal_reset` | State clears properly |

---

## Validation Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Compiles | `cargo build -p context-graph-utl` | Success |
| Tests pass | `cargo test -p context-graph-utl cross_modal` | All tests pass |
| No warnings | `cargo clippy -p context-graph-utl -- -D warnings` | No warnings |
| Factory routes correctly | Run factory test | E10 -> CrossModalEntropy |

---

## Implementation Notes

1. **Modality Detection Heuristic**: CLIP embeddings have predictable activation patterns:
   - Text inputs activate lower-indexed dimensions more strongly
   - Visual inputs activate higher-indexed dimensions more strongly
   - Compute energy ratio: `sum(dims[0:384]) / sum(dims[384:768])`

2. **Cross-Modal Weighting**: When current and history embeddings are from different modalities (detected via indicator), apply `cross_modal_weight`. When same modality, apply `intra_modal_weight`.

3. **Default Weights**: `intra_modal_weight = 0.7`, `cross_modal_weight = 0.3` gives preference to same-modality similarity while still accounting for cross-modal semantic alignment.

4. **Config Fields**: May need to add:
   - `multimodal_intra_weight: f32` (default 0.7)
   - `multimodal_cross_weight: f32` (default 0.3)

5. **Fallback Behavior**: If modality detection is unreliable (indicator near 0.5), use standard KNN-like calculation.

---

## Rollback Plan

If implementation causes issues:
1. Revert factory routing to `DefaultKnnEntropy` for `Embedder::Multimodal`
2. Remove `cross_modal.rs` from module exports
3. Delete `cross_modal.rs` file

---

## Related Tasks

- **TASK-UTL-P1-003**: JaccardCodeEntropy for E7
- **TASK-UTL-P1-005**: ExponentialDecayEntropy for E11
- **TASK-UTL-P1-006**: MaxSimTokenEntropy for E12
