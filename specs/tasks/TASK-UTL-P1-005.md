# TASK-UTL-P1-005: Implement ExponentialDecayEntropy for E11 (Entity/TemporalDecay)

**Priority:** P1
**Status:** pending
**Spec Reference:** SPEC-UTL-003
**Estimated Effort:** 2-3 hours
**Implements:** REQ-UTL-003-05, REQ-UTL-003-06

---

## Summary

Create a specialized entropy calculator for E11 (Entity) embeddings using exponential decay functions. Entity embeddings from MiniLM (384D) represent named entities, concepts, and relationships. Temporal decay ensures that older entity references contribute less to current surprise calculations, modeling the natural forgetting curve of entity relevance.

---

## Input Context Files

Read these files before implementation:

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/mod.rs` | `EmbedderEntropy` trait definition |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs` | Factory routing (to be updated) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/default_knn.rs` | Reference: KNN baseline |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/teleological/embedder.rs` | `Embedder::Entity` definition (index 10) |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/config.rs` | `SurpriseConfig` for decay parameters |
| `/home/cabdru/contextgraph/crates/context-graph-utl/src/error.rs` | `UtlError`, `UtlResult` types |

---

## Definition of Done

### 1. Create New File

**Path:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/exponential_decay.rs`

**Exact Signatures Required:**

```rust
/// E11 (Entity) entropy using exponential temporal decay.
///
/// Entity embeddings represent named entities and concepts whose relevance
/// decays over time. This calculator weights history embeddings by their
/// recency, so that recent entity mentions have higher influence on surprise.
///
/// # Algorithm
///
/// 1. Compute cosine distances to all history embeddings
/// 2. Weight each distance by exponential decay: w_i = exp(-λ * age_i)
///    where age_i is the index position (0 = most recent)
/// 3. Compute weighted mean of k-nearest distances
/// 4. Normalize via sigmoid, clamp to [0, 1]
///
/// # Decay Function
///
/// The decay constant λ controls how quickly relevance fades:
/// - λ = 0.1: Half-life ≈ 7 positions (slow decay, long memory)
/// - λ = 0.3: Half-life ≈ 2.3 positions (moderate decay)
/// - λ = 0.5: Half-life ≈ 1.4 positions (fast decay, short memory)
///
/// Half-life = ln(2) / λ ≈ 0.693 / λ
#[derive(Debug, Clone)]
pub struct ExponentialDecayEntropy {
    /// Decay constant λ. Default: 0.1 (slow decay, ~7-position half-life)
    lambda: f32,
    /// Minimum weight floor to prevent zero weights. Default: 0.01
    min_weight: f32,
    /// Running mean for distance normalization.
    running_mean: f32,
    /// Running variance for distance normalization.
    running_variance: f32,
    /// Number of samples seen.
    sample_count: usize,
    /// EMA alpha for updating statistics.
    ema_alpha: f32,
}

impl ExponentialDecayEntropy {
    /// Create a new exponential decay entropy calculator.
    pub fn new() -> Self;

    /// Create with a specific decay constant.
    ///
    /// # Arguments
    /// * `lambda` - Decay constant (0.01 to 1.0). Higher = faster decay.
    pub fn with_lambda(lambda: f32) -> Self;

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self;

    /// Set the decay constant λ.
    ///
    /// # Arguments
    /// * `lambda` - Decay constant (clamped to 0.01 to 1.0)
    pub fn set_lambda(&mut self, lambda: f32);

    /// Set the minimum weight floor.
    pub fn with_min_weight(self, min_weight: f32) -> Self;

    /// Compute decay weight for a given age (position in history).
    /// Returns exp(-λ * age), floored at min_weight.
    fn compute_decay_weight(&self, age: usize) -> f32;

    /// Compute half-life in history positions.
    /// Returns ln(2) / λ.
    pub fn half_life(&self) -> f32;

    /// Sigmoid normalization function.
    fn sigmoid(x: f32) -> f32;
}

impl Default for ExponentialDecayEntropy {
    fn default() -> Self;
}

impl EmbedderEntropy for ExponentialDecayEntropy {
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
mod exponential_decay;
pub use exponential_decay::ExponentialDecayEntropy;
```

### 3. Update Factory Routing

**File:** `/home/cabdru/contextgraph/crates/context-graph-utl/src/surprise/embedder_entropy/factory.rs`

Change the `Embedder::Entity` arm from:
```rust
Embedder::Entity => Box::new(DefaultKnnEntropy::from_config(embedder, config))
```

To:
```rust
Embedder::Entity => Box::new(ExponentialDecayEntropy::from_config(config))
```

### 4. Required Tests

Implement in `exponential_decay.rs` tests module:

| Test Name | Description |
|-----------|-------------|
| `test_decay_empty_history_returns_one` | Empty history = max surprise |
| `test_decay_identical_returns_low` | Same embedding = low ΔS |
| `test_decay_recent_weighted_higher` | Position 0 has weight ≈ 1.0 |
| `test_decay_old_weighted_lower` | Position 10 has weight << 1.0 |
| `test_decay_empty_input_error` | Empty current = EmptyInput error |
| `test_decay_embedder_type` | Returns `Embedder::Entity` |
| `test_decay_valid_range` | All outputs in [0, 1] |
| `test_decay_no_nan_infinity` | No NaN/Infinity outputs |
| `test_decay_lambda_affects_halflife` | λ = 0.1 vs λ = 0.5 differ |
| `test_decay_from_config` | Config values applied |
| `test_decay_weight_function` | exp(-λ * age) computed correctly |
| `test_decay_min_weight_floor` | Weights never below floor |
| `test_decay_reset` | State clears properly |
| `test_decay_half_life_calculation` | half_life() = ln(2) / λ |

---

## Validation Criteria

| Check | Command | Expected |
|-------|---------|----------|
| Compiles | `cargo build -p context-graph-utl` | Success |
| Tests pass | `cargo test -p context-graph-utl exponential_decay` | All tests pass |
| No warnings | `cargo clippy -p context-graph-utl -- -D warnings` | No warnings |
| Factory routes correctly | Run factory test | E11 -> ExponentialDecayEntropy |

---

## Implementation Notes

1. **History Ordering**: History is expected to be ordered most-recent-first (index 0 = newest). This aligns with typical usage patterns.

2. **Decay Constant Selection**: Default λ = 0.1 gives ~7-position half-life, meaning an entity from 7 positions ago has half the influence of the most recent one. This is appropriate for session-length entity tracking.

3. **Weighted KNN**: Instead of simple k-nearest averaging, compute weighted average:
   ```
   weighted_avg = Σ(w_i * d_i) / Σ(w_i)
   ```
   where w_i = exp(-λ * i) and d_i is the cosine distance.

4. **Config Fields**: May need to add:
   - `entity_decay_lambda: f32` (default 0.1)
   - `entity_min_weight: f32` (default 0.01)

5. **Minimum Weight Floor**: Prevents very old items from having zero influence, which could cause division issues and ignores potentially relevant ancient entities.

---

## Mathematical Background

The exponential decay function models the Ebbinghaus forgetting curve:

```
w(t) = exp(-λ * t)
```

Properties:
- w(0) = 1.0 (most recent has full weight)
- w(ln(2)/λ) = 0.5 (half-life point)
- w(∞) → 0 (old items fade)

For λ = 0.1:
- w(1) ≈ 0.905
- w(5) ≈ 0.607
- w(7) ≈ 0.497 (half-life)
- w(10) ≈ 0.368
- w(20) ≈ 0.135

---

## Rollback Plan

If implementation causes issues:
1. Revert factory routing to `DefaultKnnEntropy` for `Embedder::Entity`
2. Remove `exponential_decay.rs` from module exports
3. Delete `exponential_decay.rs` file

---

## Related Tasks

- **TASK-UTL-P1-003**: JaccardCodeEntropy for E7
- **TASK-UTL-P1-004**: CrossModalEntropy for E10
- **TASK-UTL-P1-006**: MaxSimTokenEntropy for E12
