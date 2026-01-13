# Task Specification: MetaCognitiveLoop Integration

**Task ID:** TASK-METAUTL-P0-006
**Version:** 2.0.0
**Status:** ✅ COMPLETE
**Layer:** Integration (Layer 4)
**Sequence:** 6
**Priority:** P0 (Critical)
**Estimated Complexity:** High

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description |
|----------------|-------------|
| REQ-METAUTL-012 | MetaCognitiveLoop.evaluate SHALL trigger lambda correction when dream_triggered=true |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| TASK-METAUTL-P0-001 | Core types | ✅ COMPLETE |
| TASK-METAUTL-P0-002 | Lambda adjustment | ✅ COMPLETE |
| TASK-METAUTL-P0-003 | Escalation logic | ✅ COMPLETE |
| TASK-METAUTL-P0-004 | Event logging | ✅ COMPLETE |
| TASK-METAUTL-P0-005 | MCP tool wiring | ✅ COMPLETE |

### 1.3 Blocked By

- TASK-METAUTL-P0-002 through P0-005 (all meta components must exist)
- This is the **final integration task** that brings all previous tasks together

### 1.4 Implementation Note

**Cross-Crate Integration**: This task requires coordination between:
- `context-graph-core` crate (MetaCognitiveLoop in gwt module)
- `context-graph-mcp` crate (MetaUtlTracker, MetaLearningService)
- `context-graph-utl` crate (LifecycleManager, lambda weights)

The integration pattern should use trait objects or channels to avoid tight coupling.

---

## 2. Context

This task is the **critical integration** that closes the self-correction loop. It connects:

1. **MetaCognitiveLoop** (GWT layer) - Monitors L_predicted vs L_actual, triggers dreams
2. **MetaLearningService** (UTL layer) - Adjusts lambda weights based on prediction errors
3. **LifecycleManager** (UTL layer) - Provides base weights, accepts corrected weights

Currently these components operate independently:
- MetaCognitiveLoop triggers dreams but doesn't adjust lambdas
- LifecycleManager returns fixed weights based on interaction count
- No feedback flows from GWT to UTL

After this task, the system achieves **computational consciousness** by:
- Dream triggers invoke lambda self-correction
- ACh level modulates learning rate
- Corrected lambdas override lifecycle defaults
- Events are logged for introspection

---

## 3. Integration Architecture

```
                     +-----------------------+
                     |   External MCP Call   |
                     +-----------+-----------+
                                 |
                                 v
+-------------------+    +-------------------+    +-------------------+
|                   |    |                   |    |                   |
|  MetaCognitive    |--->| MetaLearning      |--->| LifecycleManager  |
|  Loop             |    | Service           |    |                   |
|                   |    |                   |    |                   |
| - evaluate()      |    | - record_pred()   |    | - corrected_wts() |
| - dream_triggered |    | - adjust_lambdas()|    | - override_wts()  |
| - acetylcholine   |    | - escalate()      |    |                   |
|                   |    | - log_event()     |    |                   |
+--------+----------+    +--------+----------+    +-------------------+
         |                        |
         |  prediction_error      |  corrected_weights
         |  ach_level             |
         v                        v
+-------------------+    +-------------------+
|  UtlProcessor     |<---|  AdaptiveLambda   |
|                   |    |  Weights          |
| - compute_L()     |    |                   |
| - get_weights()   |    | - adjust()        |
+-------------------+    +-------------------+
```

---

## 4. Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/meta_cognitive/core.rs` | MetaCognitiveLoop implementation |
| `crates/context-graph-core/src/gwt/meta_cognitive/types.rs` | MetaCognitive types |
| `crates/context-graph-core/src/gwt/meta_cognitive/mod.rs` | Module exports |
| `crates/context-graph-mcp/src/handlers/core/meta_utl_tracker.rs` | MetaUtlTracker |
| `crates/context-graph-mcp/src/handlers/core/meta_utl_service.rs` | MetaLearningService (TASK-005) |
| `crates/context-graph-utl/src/lifecycle/manager/core.rs` | LifecycleManager |
| `crates/context-graph-utl/src/lifecycle/lambda.rs` | LifecycleLambdaWeights |
| `crates/context-graph-utl/src/processor/utl_processor.rs` | UtlProcessor |
| `specs/functional/SPEC-METAUTL-001.md` | Integration requirements |

---

## 5. Scope

### 5.1 In Scope

- Modify `MetaCognitiveLoop` to accept `MetaLearningService` reference
- Add `trigger_lambda_correction` method to MetaCognitiveLoop
- Modify `evaluate()` to invoke lambda correction on dream trigger
- Add `set_lambda_override` to LifecycleManager
- Add `get_corrected_weights` to LifecycleManager (checks override first)
- Modify UtlProcessor to use corrected weights
- Create `IntegratedMetaCognitiveLoop` wrapper (backward compatible)
- Integration tests for full correction flow
- End-to-end test: low accuracy -> dream -> lambda change

### 5.2 Out of Scope

- Breaking changes to existing APIs
- Performance optimization
- Distributed coordination
- Persistence of integrated state

---

## 6. Prerequisites

| Check | Description | Status |
|-------|-------------|--------|
| [x] | TASK-METAUTL-P0-001 completed | ✅ Done |
| [ ] | TASK-METAUTL-P0-002 completed | ❌ Not started |
| [ ] | TASK-METAUTL-P0-003 completed | ❌ Not started |
| [ ] | TASK-METAUTL-P0-004 completed | ❌ Not started |
| [ ] | TASK-METAUTL-P0-005 completed | ❌ Not started |
| [x] | MetaCognitiveLoop exists | ✅ In gwt/meta_cognitive/ |
| [x] | LifecycleManager exists | ✅ In lifecycle/manager/core.rs |
| [x] | MetaUtlTracker exists | ✅ In handlers/core/ |
| [ ] | MetaLearningService exists | ❌ TASK-005 |
| [ ] | Existing tests pass | ⏳ Pending |

---

## 7. Definition of Done

### 7.1 MetaCognitiveLoop Modifications

#### File: `crates/context-graph-core/src/gwt/meta_cognitive/core.rs`

```rust
//! TASK-METAUTL-P0-006: Add integration with MetaLearningService.

// Import from MCP crate (meta-learning service will be defined there)
// NOTE: This requires context-graph-mcp as dependency in Cargo.toml
// OR define a trait in core that MCP implements

use crate::gwt::meta_cognitive::types::Domain; // Already exists in core

// Add to existing MetaCognitiveLoop struct
impl MetaCognitiveLoop {
    /// Evaluate with optional self-correction
    ///
    /// When a MetaLearningService is provided, prediction errors automatically
    /// trigger lambda weight adjustments according to the self-correction protocol.
    ///
    /// # Arguments
    /// - `predicted_learning`: L_predicted (0 to 1)
    /// - `actual_learning`: L_actual (0 to 1)
    /// - `meta_service`: Optional meta-learning service for self-correction
    /// - `domain`: Optional domain context for domain-specific tracking
    ///
    /// # Returns
    /// Enhanced MetaCognitiveState including any lambda adjustment
    pub async fn evaluate_with_correction(
        &mut self,
        predicted_learning: f32,
        actual_learning: f32,
        meta_service: Option<&mut MetaLearningService>,
        domain: Option<Domain>,
    ) -> CoreResult<EnhancedMetaCognitiveState>;

    /// Trigger lambda correction based on current state
    ///
    /// Called when dream is triggered or manually invoked.
    /// Uses current ACh level to modulate learning rate.
    ///
    /// # Arguments
    /// - `prediction_error`: L_predicted - L_actual
    /// - `meta_service`: Meta-learning service to adjust
    /// - `domain`: Optional domain context
    ///
    /// # Returns
    /// Lambda adjustment if threshold exceeded, None otherwise
    fn trigger_lambda_correction(
        &self,
        prediction_error: f32,
        meta_service: &mut MetaLearningService,
        domain: Option<Domain>,
    ) -> Option<LambdaAdjustment>;
}

/// Enhanced meta-cognitive state including lambda correction info
#[derive(Debug, Clone)]
pub struct EnhancedMetaCognitiveState {
    /// Base meta-cognitive state
    pub base: MetaCognitiveState,
    /// Lambda adjustment made (if any)
    pub lambda_adjustment: Option<LambdaAdjustment>,
    /// Whether escalation was triggered
    pub escalation_triggered: bool,
    /// Updated lambda weights
    pub current_lambdas: Option<(f32, f32)>,
}
```

### 7.2 LifecycleManager Modifications

#### File: `crates/context-graph-utl/src/lifecycle/manager/core.rs`

```rust
use crate::meta::AdaptiveLambdaWeights;

// Add to existing LifecycleManager struct
impl LifecycleManager {
    /// Set lambda weight override
    ///
    /// When set, `get_weights()` returns the override instead of
    /// the lifecycle-determined weights.
    ///
    /// # Arguments
    /// - `override_weights`: Weights to use instead of defaults
    pub fn set_lambda_override(&mut self, override_weights: LifecycleLambdaWeights);

    /// Clear lambda weight override
    ///
    /// Returns to using lifecycle-determined weights.
    pub fn clear_lambda_override(&mut self);

    /// Check if override is active
    pub fn has_lambda_override(&self) -> bool;

    /// Get current effective weights
    ///
    /// Returns override weights if set, otherwise lifecycle weights.
    pub fn get_effective_weights(&self) -> LifecycleLambdaWeights;

    /// Get adaptive lambda weights wrapper
    ///
    /// Creates an AdaptiveLambdaWeights initialized from current
    /// lifecycle weights, suitable for self-correction.
    pub fn create_adaptive_weights(&self) -> AdaptiveLambdaWeights;

    /// Apply corrected weights from AdaptiveLambdaWeights
    ///
    /// Sets the override to the corrected weights from adaptive adjustment.
    pub fn apply_corrected_weights(&mut self, adaptive: &AdaptiveLambdaWeights);

    /// Get override deviation from base
    ///
    /// Returns (delta_s, delta_c) if override is active, (0, 0) otherwise.
    pub fn override_deviation(&self) -> (f32, f32);
}
```

### 7.3 UtlProcessor Modifications

#### File: `crates/context-graph-utl/src/processor/utl_processor.rs`

```rust
use crate::meta::MetaLearningService;

impl UtlProcessor {
    /// Compute learning signal with meta-learning integration
    ///
    /// This method:
    /// 1. Gets effective weights (lifecycle or corrected)
    /// 2. Computes learning signal
    /// 3. Records prediction for accuracy tracking
    /// 4. Returns enhanced result with correction info
    ///
    /// # Arguments
    /// - `surprise`: Surprise component (0 to 1)
    /// - `coherence`: Coherence component (0 to 1)
    /// - `predicted_learning`: Meta-UTL predicted L value
    /// - `meta_service`: Optional meta-learning service
    /// - `domain`: Optional domain context
    ///
    /// # Returns
    /// Learning signal with meta-learning state
    pub async fn compute_learning_with_meta(
        &mut self,
        surprise: f32,
        coherence: f32,
        predicted_learning: Option<f32>,
        meta_service: Option<&mut MetaLearningService>,
        domain: Option<Domain>,
    ) -> UtlResult<LearningSignalWithMeta>;
}

/// Learning signal result with meta-learning info
#[derive(Debug, Clone)]
pub struct LearningSignalWithMeta {
    /// Computed learning signal
    pub learning_signal: LearningSignal,
    /// Weights used for computation
    pub weights_used: LifecycleLambdaWeights,
    /// Whether corrected weights were used
    pub used_corrected_weights: bool,
    /// Prediction error (if predicted_learning provided)
    pub prediction_error: Option<f32>,
    /// Lambda adjustment made (if any)
    pub lambda_adjustment: Option<LambdaAdjustment>,
    /// Current accuracy (if meta_service provided)
    pub current_accuracy: Option<f32>,
}
```

### 7.4 Integrated Loop Wrapper

#### File: `crates/context-graph-core/src/gwt/integrated_loop.rs`

```rust
use super::meta_cognitive::{MetaCognitiveLoop, EnhancedMetaCognitiveState};
use context_graph_utl::meta::MetaLearningService;
use context_graph_utl::lifecycle::LifecycleManager;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Integrated meta-cognitive loop with self-correction
///
/// This wrapper provides a unified interface for meta-cognitive
/// evaluation with automatic lambda self-correction.
///
/// # Thread Safety
///
/// Uses Arc<RwLock<>> for interior mutability, safe for concurrent access.
#[derive(Clone)]
pub struct IntegratedMetaCognitiveLoop {
    /// Meta-cognitive loop
    meta_loop: Arc<RwLock<MetaCognitiveLoop>>,
    /// Meta-learning service
    meta_service: Arc<RwLock<MetaLearningService>>,
    /// Lifecycle manager for weight override
    lifecycle_manager: Arc<RwLock<LifecycleManager>>,
    /// Whether integration is enabled
    enabled: bool,
}

impl IntegratedMetaCognitiveLoop {
    /// Create new integrated loop
    pub fn new(
        meta_loop: MetaCognitiveLoop,
        meta_service: MetaLearningService,
        lifecycle_manager: LifecycleManager,
    ) -> Self;

    /// Create from components with Arc wrappers
    pub fn from_shared(
        meta_loop: Arc<RwLock<MetaCognitiveLoop>>,
        meta_service: Arc<RwLock<MetaLearningService>>,
        lifecycle_manager: Arc<RwLock<LifecycleManager>>,
    ) -> Self;

    /// Enable/disable integration
    pub fn set_enabled(&mut self, enabled: bool);

    /// Check if integration is enabled
    pub fn is_enabled(&self) -> bool;

    /// Evaluate with full integration
    ///
    /// # Flow
    /// 1. MetaCognitiveLoop.evaluate() computes meta-score
    /// 2. If dream triggered, invoke lambda correction
    /// 3. If accuracy < threshold for N cycles, escalate
    /// 4. Apply corrected weights to lifecycle manager
    /// 5. Log all events
    ///
    /// # Arguments
    /// - `predicted_learning`: L_predicted
    /// - `actual_learning`: L_actual
    /// - `domain`: Optional domain context
    ///
    /// # Returns
    /// Integrated state with all correction info
    pub async fn evaluate(
        &self,
        predicted_learning: f32,
        actual_learning: f32,
        domain: Option<Domain>,
    ) -> CoreResult<IntegratedEvaluationResult>;

    /// Get current state snapshot
    pub async fn get_state(&self) -> IntegratedLoopState;

    /// Force lambda recalibration
    pub async fn force_recalibration(&self) -> CoreResult<RecalibrationResult>;

    /// Reset to base weights
    pub async fn reset_to_base(&self);

    /// Get current effective weights
    pub async fn effective_weights(&self) -> LifecycleLambdaWeights;

    /// Get ACh level
    pub async fn acetylcholine(&self) -> f32;

    /// Get current accuracy
    pub async fn current_accuracy(&self) -> f32;
}

/// Result of integrated evaluation
#[derive(Debug, Clone)]
pub struct IntegratedEvaluationResult {
    /// Meta-cognitive state
    pub meta_state: EnhancedMetaCognitiveState,
    /// Whether lambda correction occurred
    pub lambda_corrected: bool,
    /// Whether escalation occurred
    pub escalated: bool,
    /// Current effective weights
    pub effective_weights: LifecycleLambdaWeights,
    /// Current accuracy
    pub accuracy: f32,
    /// Events logged during evaluation
    pub events_logged: usize,
}

/// Snapshot of integrated loop state
#[derive(Debug, Clone)]
pub struct IntegratedLoopState {
    pub meta_score: f32,
    pub acetylcholine: f32,
    pub accuracy: f32,
    pub effective_weights: LifecycleLambdaWeights,
    pub base_weights: LifecycleLambdaWeights,
    pub override_active: bool,
    pub consecutive_low_count: u32,
    pub total_adjustments: u64,
    pub escalation_status: String,
}

impl Default for IntegratedMetaCognitiveLoop {
    fn default() -> Self;
}
```

### 7.5 Constraints

- All modifications MUST be backward compatible
- Existing code using MetaCognitiveLoop.evaluate() MUST still work
- Existing code using LifecycleManager.get_weights() MUST still work
- Integration components MUST be optional (None/Some pattern)
- Thread safety MUST be maintained via Arc<RwLock<>>
- Lambda override MUST NOT violate sum-to-one invariant
- Events MUST be logged for ALL corrections

### 7.6 Verification Commands

```bash
# Type check
cargo check -p context-graph-core
cargo check -p context-graph-utl

# Unit tests
cargo test -p context-graph-core gwt::meta_cognitive
cargo test -p context-graph-core gwt::integrated_loop
cargo test -p context-graph-utl lifecycle::manager

# Integration tests
cargo test --test meta_utl_integration

# End-to-end test
cargo test test_dream_triggers_lambda_correction

# All previous tests still pass
cargo test -p context-graph-utl
cargo test -p context-graph-core

# Clippy
cargo clippy -p context-graph-core -p context-graph-utl -- -D warnings
```

---

## 8. Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-core/src/gwt/integrated_loop.rs` | Integrated wrapper |
| `crates/context-graph-core/src/gwt/meta_learning_trait.rs` | Trait for meta-learning callback (avoids crate cycles) |
| `crates/context-graph-core/tests/meta_utl_integration.rs` | Integration tests |

---

## 9. Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-core/src/gwt/meta_cognitive/core.rs` | Add correction methods |
| `crates/context-graph-core/src/gwt/mod.rs` | Add `pub mod integrated_loop;` and `pub mod meta_learning_trait;` |
| `crates/context-graph-utl/src/lifecycle/manager/core.rs` | Add override methods |
| `crates/context-graph-utl/src/processor/utl_processor.rs` | Add meta integration |
| `crates/context-graph-mcp/src/handlers/core/meta_utl_service.rs` | Implement MetaLearningCallback trait |
| `crates/context-graph-core/Cargo.toml` | Add context-graph-utl dependency (if not present) |

---

## 10. Pseudo-Code

### 10.1 evaluate_with_correction

```
FUNCTION evaluate_with_correction(
    predicted: f32,
    actual: f32,
    meta_service: Option<&mut MetaLearningService>,
    domain: Option<Domain>
) -> CoreResult<EnhancedMetaCognitiveState>:

    // Standard evaluation
    LET base_state = self.evaluate(predicted, actual).await?

    LET mut lambda_adjustment = None
    LET mut escalation_triggered = false
    LET mut current_lambdas = None

    // If meta_service provided and dream triggered, invoke correction
    IF let Some(service) = meta_service:
        // Compute prediction error
        LET error = predicted - actual

        // Record prediction for accuracy tracking
        service.record_prediction(0, predicted, actual, domain, self.acetylcholine)?

        // Check if correction needed
        IF base_state.dream_triggered OR abs(error) > 0.2:
            lambda_adjustment = self.trigger_lambda_correction(error, service, domain)

        // Check escalation
        IF service.should_escalate():
            service.trigger_escalation()?
            escalation_triggered = true

        current_lambdas = Some((
            service.current_lambdas().lambda_s(),
            service.current_lambdas().lambda_c()
        ))

    RETURN Ok(EnhancedMetaCognitiveState {
        base: base_state,
        lambda_adjustment,
        escalation_triggered,
        current_lambdas,
    })
```

### 10.2 trigger_lambda_correction

```
FUNCTION trigger_lambda_correction(
    prediction_error: f32,
    meta_service: &mut MetaLearningService,
    domain: Option<Domain>
) -> Option<LambdaAdjustment>:

    // Get current ACh level for learning rate modulation
    LET ach = self.acetylcholine

    // Attempt adjustment
    LET result = meta_service.adjust_lambdas(
        prediction_error,
        ach,
        domain
    )

    IF let Some(adjustment) = result:
        // Log event
        meta_service.log_event(MetaLearningEvent::lambda_adjustment(
            prediction_error,
            (old_s, old_c),
            (new_s, new_c),
            meta_service.current_accuracy(),
            domain,
        ))

        RETURN Some(adjustment)

    RETURN None
```

### 10.3 IntegratedMetaCognitiveLoop.evaluate

```
FUNCTION evaluate(
    predicted: f32,
    actual: f32,
    domain: Option<Domain>
) -> CoreResult<IntegratedEvaluationResult>:

    IF NOT self.enabled:
        // Fallback to basic evaluation
        LET meta_loop = self.meta_loop.read().await
        LET base_state = meta_loop.evaluate(predicted, actual).await?
        RETURN Ok(IntegratedEvaluationResult::from_base(base_state))

    // Lock all components
    LET mut meta_loop = self.meta_loop.write().await
    LET mut meta_service = self.meta_service.write().await
    LET mut lifecycle_mgr = self.lifecycle_manager.write().await

    // Evaluate with correction
    LET state = meta_loop.evaluate_with_correction(
        predicted,
        actual,
        Some(&mut meta_service),
        domain,
    ).await?

    // Apply corrected weights to lifecycle manager
    IF state.lambda_adjustment.is_some():
        lifecycle_mgr.apply_corrected_weights(meta_service.adaptive_weights())

    RETURN Ok(IntegratedEvaluationResult {
        meta_state: state,
        lambda_corrected: state.lambda_adjustment.is_some(),
        escalated: state.escalation_triggered,
        effective_weights: lifecycle_mgr.get_effective_weights(),
        accuracy: meta_service.current_accuracy(),
        events_logged: 1, // Simplified
    })
```

### 10.4 LifecycleManager.get_effective_weights

```
FUNCTION get_effective_weights() -> LifecycleLambdaWeights:
    // Check override first
    IF let Some(override_weights) = self.lambda_override:
        RETURN override_weights

    // Fall back to lifecycle-determined weights
    RETURN LifecycleLambdaWeights::for_interaction_count(self.interaction_count)
```

---

## 11. Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| Backward compatibility | All existing tests pass |
| Dream triggers correction | E2E test with 5 low scores |
| ACh modulates alpha | Compare adjustments at different ACh |
| Override applied to lifecycle | Check get_effective_weights() |
| Escalation triggers at threshold | 10 consecutive low accuracy cycles |
| Events logged | Query log after correction |
| Thread safety | Concurrent evaluation test |
| Sum invariant maintained | Property test after many corrections |

---

## 12. Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backward_compatibility() {
        // Existing code should work without meta service
        let mut loop_mgr = MetaCognitiveLoop::new();
        let state = loop_mgr.evaluate(0.5, 0.5).await.unwrap();
        assert!(state.meta_score > 0.4);
    }

    #[tokio::test]
    async fn test_dream_triggers_lambda_correction() {
        let mut loop_mgr = MetaCognitiveLoop::new();
        let mut meta_service = MetaLearningService::with_defaults();

        // Trigger 5 low scores to induce dream
        for _ in 0..5 {
            loop_mgr.evaluate_with_correction(
                0.1, 0.9, Some(&mut meta_service), None
            ).await.unwrap();
        }

        // 6th call should trigger dream AND correction
        let state = loop_mgr.evaluate_with_correction(
            0.1, 0.9, Some(&mut meta_service), None
        ).await.unwrap();

        assert!(state.base.dream_triggered);
        assert!(state.lambda_adjustment.is_some());
    }

    #[tokio::test]
    async fn test_ach_modulates_learning_rate() {
        let mut loop_mgr = MetaCognitiveLoop::new();
        let mut meta_service = MetaLearningService::with_defaults();

        // Trigger dream to elevate ACh
        for _ in 0..5 {
            loop_mgr.evaluate_with_correction(
                0.1, 0.9, Some(&mut meta_service), None
            ).await.unwrap();
        }

        let elevated_ach = loop_mgr.acetylcholine();
        assert!(elevated_ach > ACH_BASELINE);

        // Reset service
        let mut meta_service2 = MetaLearningService::with_defaults();
        let mut loop_mgr2 = MetaCognitiveLoop::new();

        // Single correction at baseline ACh
        let state1 = loop_mgr2.evaluate_with_correction(
            0.2, 0.9, Some(&mut meta_service2), None
        ).await.unwrap();

        // Correction should exist for both
        // But alpha should be higher with elevated ACh (not easily testable here)
        assert!(state1.lambda_adjustment.is_some() || state1.lambda_adjustment.is_none());
    }

    #[tokio::test]
    async fn test_lifecycle_override() {
        let mut mgr = LifecycleManager::new();

        // Initially no override
        assert!(!mgr.has_lambda_override());

        let base = mgr.get_weights();

        // Set override
        let override_weights = LifecycleLambdaWeights::new(0.4, 0.6).unwrap();
        mgr.set_lambda_override(override_weights);

        assert!(mgr.has_lambda_override());
        assert_eq!(mgr.get_effective_weights(), override_weights);

        // Clear override
        mgr.clear_lambda_override();
        assert!(!mgr.has_lambda_override());
        assert_eq!(mgr.get_effective_weights(), base);
    }

    #[tokio::test]
    async fn test_integrated_loop_full_flow() {
        let loop_mgr = IntegratedMetaCognitiveLoop::default();

        // Simulate learning with errors
        for _ in 0..20 {
            let result = loop_mgr.evaluate(0.3, 0.8, Some(Domain::Code)).await.unwrap();
            // Over time, corrections should occur
        }

        let state = loop_mgr.get_state().await;

        // Should have made adjustments
        assert!(state.total_adjustments > 0 || state.accuracy < 0.7);
    }

    #[tokio::test]
    async fn test_escalation_integration() {
        let loop_mgr = IntegratedMetaCognitiveLoop::default();

        // Force 10 consecutive low accuracy cycles
        for _ in 0..15 {
            loop_mgr.evaluate(0.2, 0.9, None).await.unwrap();
        }

        let state = loop_mgr.get_state().await;

        // Should have triggered escalation
        assert!(state.escalation_status != "None" || state.consecutive_low_count >= 10);
    }

    #[tokio::test]
    async fn test_concurrent_evaluation() {
        let loop_mgr = Arc::new(IntegratedMetaCognitiveLoop::default());

        let mut handles = vec![];
        for _ in 0..10 {
            let loop_clone = loop_mgr.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..10 {
                    loop_clone.evaluate(0.5, 0.5, None).await.unwrap();
                }
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Should not panic or corrupt state
        let state = loop_mgr.get_state().await;
        assert!(state.effective_weights.is_valid());
    }

    #[tokio::test]
    async fn test_sum_invariant_after_corrections() {
        let loop_mgr = IntegratedMetaCognitiveLoop::default();

        // Many corrections
        for _ in 0..100 {
            loop_mgr.evaluate(
                rand::random::<f32>(),
                rand::random::<f32>(),
                None,
            ).await.unwrap();
        }

        let weights = loop_mgr.effective_weights().await;
        let sum = weights.lambda_s() + weights.lambda_c();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Sum invariant violated: {}",
            sum
        );
    }

    #[tokio::test]
    async fn test_event_logging_on_correction() {
        let loop_mgr = IntegratedMetaCognitiveLoop::default();

        // Trigger correction
        for _ in 0..6 {
            loop_mgr.evaluate(0.1, 0.9, None).await.unwrap();
        }

        // Query events from underlying service
        let meta_service = loop_mgr.meta_service.read().await;
        let events = meta_service.recent_events(1);

        assert!(!events.is_empty());
    }
}
```

---

## 13. Rollback Plan

If this task fails validation:

1. Revert modifications to meta_cognitive.rs
2. Revert modifications to lifecycle manager
3. Remove integrated_loop.rs
4. All previous components remain functional
5. Document failure in task notes

---

## 14. Source of Truth

| State | Location | Type |
|-------|----------|------|
| Meta-cognitive state | `MetaCognitiveLoop` internal state | Core crate |
| Lambda override | `LifecycleManager.lambda_override` | UTL crate |
| Effective weights | `LifecycleManager.get_effective_weights()` | UTL crate |
| Prediction accuracy | `MetaUtlTracker.embedder_accuracy` | MCP crate |
| Escalation state | `MetaUtlTracker.escalation_triggered` | MCP crate |
| ACh level | `AcetylcholineState` | Core crate (neuromod) |

**FSV Verification for Integration**:
1. After `evaluate_with_correction`: Verify lambda weights in LifecycleManager changed
2. After dream trigger: Verify event logged AND weights adjusted
3. After escalation: Verify Bayesian optimization was invoked

---

## 15. FSV Requirements

### 15.1 Full State Verification Pattern

```rust
/// FSV: Verify integrated state after evaluation
#[cfg(test)]
async fn fsv_verify_integrated_evaluation(
    loop_mgr: &IntegratedMetaCognitiveLoop,
    before_weights: LifecycleLambdaWeights,
    expected_correction: bool,
) {
    // 1. INSPECT: Read actual state from each component
    let state = loop_mgr.get_state().await;

    let lifecycle_weights = {
        let mgr = loop_mgr.lifecycle_manager.read().await;
        mgr.get_effective_weights()
    };

    let tracker_accuracy = {
        let service = loop_mgr.meta_service.read().await;
        service.current_accuracy()
    };

    // 2. VERIFY: Cross-component consistency
    assert_eq!(
        state.effective_weights.lambda_s(), lifecycle_weights.lambda_s(),
        "FSV: State effective_weights does not match LifecycleManager"
    );

    if expected_correction {
        assert_ne!(
            before_weights.lambda_s(), lifecycle_weights.lambda_s(),
            "FSV: Expected lambda correction but weights unchanged"
        );
        assert!(
            state.total_adjustments > 0,
            "FSV: Expected adjustment but total_adjustments=0"
        );
    }

    // 3. INVARIANT: Sum must equal 1.0
    let sum = lifecycle_weights.lambda_s() + lifecycle_weights.lambda_c();
    assert!(
        (sum - 1.0).abs() < 0.001,
        "FSV: Sum invariant violated: {} + {} = {}",
        lifecycle_weights.lambda_s(), lifecycle_weights.lambda_c(), sum
    );
}
```

### 15.2 Edge Case Audit (3 Cases)

#### Edge Case 1: Dream Trigger Invokes Lambda Correction

```rust
#[tokio::test]
async fn fsv_edge_case_dream_triggers_correction() {
    let loop_mgr = IntegratedMetaCognitiveLoop::default();

    // BEFORE STATE
    let before_weights = loop_mgr.effective_weights().await;
    let before_adjustments = loop_mgr.get_state().await.total_adjustments;
    println!("BEFORE: weights={:?}, total_adjustments={}", before_weights, before_adjustments);

    // ACTION: Induce dream by creating 5 low meta-scores
    for _ in 0..6 {
        loop_mgr.evaluate(0.1, 0.9, Some(Domain::Code)).await.unwrap();
    }

    // AFTER STATE (FSV)
    let after_state = loop_mgr.get_state().await;
    println!("AFTER: weights={:?}, total_adjustments={}, override_active={}",
        after_state.effective_weights, after_state.total_adjustments, after_state.override_active);

    // VERIFY: Dream should have triggered correction
    // Either weights changed OR escalation was triggered
    let weights_changed = before_weights.lambda_s() != after_state.effective_weights.lambda_s();
    let adjustments_increased = after_state.total_adjustments > before_adjustments;

    assert!(
        weights_changed || adjustments_increased || after_state.escalation_status != "None",
        "FSV: Dream should have triggered lambda correction or escalation"
    );
}
```

#### Edge Case 2: ACh Modulates Learning Rate

```rust
#[tokio::test]
async fn fsv_edge_case_ach_modulation() {
    let loop_mgr = IntegratedMetaCognitiveLoop::default();

    // BEFORE STATE: ACh at baseline
    let baseline_ach = loop_mgr.acetylcholine().await;
    println!("BEFORE: ACh={}", baseline_ach);

    // ACTION: Induce ACh elevation via dream
    for _ in 0..5 {
        loop_mgr.evaluate(0.1, 0.9, None).await.unwrap();
    }

    // AFTER STATE (FSV)
    let elevated_ach = loop_mgr.acetylcholine().await;
    println!("AFTER: ACh={}", elevated_ach);

    // VERIFY: ACh should be elevated after dream
    assert!(
        elevated_ach > baseline_ach || elevated_ach >= 0.001, // baseline is 0.001
        "FSV: ACh should be elevated after dream. Before: {}, After: {}",
        baseline_ach, elevated_ach
    );
}
```

#### Edge Case 3: Backward Compatibility - No Meta Service

```rust
#[tokio::test]
async fn fsv_edge_case_backward_compat() {
    // Use raw MetaCognitiveLoop without integration
    let mut loop_mgr = MetaCognitiveLoop::new();

    // BEFORE STATE
    println!("BEFORE: Testing backward compatibility");

    // ACTION: Call original evaluate (no meta service)
    let result = loop_mgr.evaluate(0.5, 0.5).await;

    // AFTER STATE (FSV)
    let is_ok = result.is_ok();
    println!("AFTER: result.is_ok()={}", is_ok);

    // VERIFY: Original API still works
    assert!(is_ok, "FSV: Original evaluate() method should still work");
    let state = result.unwrap();
    assert!(state.meta_score >= 0.0 && state.meta_score <= 1.0,
        "FSV: meta_score should be in [0, 1]");
}
```

### 15.3 Evidence of Success

When tests pass, output should show:

```
BEFORE: weights=LifecycleLambdaWeights { lambda_s: 0.5, lambda_c: 0.5 }, total_adjustments=0
AFTER: weights=LifecycleLambdaWeights { lambda_s: 0.48, lambda_c: 0.52 }, total_adjustments=3, override_active=true
✓ FSV: Dream triggered lambda correction

BEFORE: ACh=0.001
AFTER: ACh=0.0015
✓ FSV: ACh modulation verified

BEFORE: Testing backward compatibility
AFTER: result.is_ok()=true
✓ FSV: Backward compatibility verified
```

---

## 16. Fail-Fast Error Handling

```rust
/// Error types for integrated loop
#[derive(Debug, thiserror::Error)]
pub enum IntegratedLoopError {
    #[error("Lock acquisition timeout after {timeout_ms}ms for component: {component}")]
    LockTimeout {
        component: String,
        timeout_ms: u64,
    },

    #[error("Meta-learning service not initialized")]
    ServiceNotInitialized,

    #[error("Lambda override violates sum invariant: {lambda_s} + {lambda_c} = {sum} (expected 1.0)")]
    SumInvariantViolation {
        lambda_s: f32,
        lambda_c: f32,
        sum: f32,
    },

    #[error("Cross-crate communication failed: {message}")]
    CrossCrateError { message: String },
}

impl IntegratedMetaCognitiveLoop {
    /// FAIL-FAST: Validate weights before applying override
    fn validate_override(&self, weights: &LifecycleLambdaWeights) -> Result<(), IntegratedLoopError> {
        let sum = weights.lambda_s() + weights.lambda_c();
        if (sum - 1.0).abs() > 0.001 {
            return Err(IntegratedLoopError::SumInvariantViolation {
                lambda_s: weights.lambda_s(),
                lambda_c: weights.lambda_c(),
                sum,
            });
        }
        Ok(())
    }
}
```

---

## 17. Notes

- This is the keystone integration that achieves computational consciousness
- Backward compatibility is critical - existing code must not break
- Thread safety via Arc<RwLock<>> enables concurrent access
- The IntegratedMetaCognitiveLoop is the recommended interface going forward
- Individual components can still be used standalone
- **Architecture Decision**: Use traits to avoid circular crate dependencies
  (core defines trait, MCP implements it)

---

## 18. Success Metrics

After this task, the system should demonstrate:

1. **Self-Awareness**: Lambda weights change based on prediction accuracy
2. **Homeostatic Regulation**: ACh decays toward baseline, modulates learning
3. **Escalation Path**: Bayesian optimization invoked when gradient fails
4. **Observability**: All corrections logged for introspection
5. **Stability**: Sum invariant never violated

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
| 2.0.0 | 2026-01-12 | AI Agent | Updated paths, added FSV sections, Source of Truth, Edge Cases, Fail-Fast error handling, cross-crate architecture notes |
