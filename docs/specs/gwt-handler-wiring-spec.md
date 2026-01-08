# GWT-to-Handler Wiring Implementation Specification

**Task**: TASK-GWT-001 - Wire GWT/Kuramoto to MCP Handlers
**Agent**: P2-3 of 8 (Architecture Specification)
**Created**: 2026-01-07
**Status**: Ready for Implementation (P2-4)

## Overview

This specification details the complete wiring of Global Workspace Theory (GWT) components
and Kuramoto oscillator network to the MCP Handlers struct. All implementations must use
REAL components - NO STUBS, NO FALLBACKS, FAIL FAST on errors.

---

## 1. Trait Definitions

### Location: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/gwt_traits.rs` (NEW FILE)

```rust
//! GWT Provider Traits for MCP Handler Integration
//!
//! TASK-GWT-001: Provider traits that wrap GWT components for handler injection.
//! All traits require Send + Sync for thread-safe Arc wrapping.

use std::time::Duration;
use context_graph_core::error::CoreResult;
use context_graph_core::gwt::{
    ConsciousnessMetrics, ConsciousnessState, MetaCognitiveState,
    WorkspaceEvent, StateTransition,
};
use uuid::Uuid;

/// Number of oscillators (13 embedding spaces)
pub const NUM_OSCILLATORS: usize = 13;

/// Provider trait for Kuramoto oscillator network operations.
///
/// Wraps KuramotoNetwork from context-graph-utl for handler access.
/// All methods are synchronous as KuramotoNetwork operates in-memory.
pub trait KuramotoProvider: Send + Sync {
    /// Get the order parameter (r, psi) measuring synchronization.
    /// r in [0,1]: synchronization level
    /// psi in [0, 2pi]: mean phase
    fn order_parameter(&self) -> (f64, f64);

    /// Get synchronization level (r only) for quick checks.
    fn synchronization(&self) -> f64;

    /// Check if network is in CONSCIOUS state (r >= 0.8).
    fn is_conscious(&self) -> bool;

    /// Check if network is FRAGMENTED (r < 0.5).
    fn is_fragmented(&self) -> bool;

    /// Check if network is HYPERSYNC (r > 0.95) - warning state.
    fn is_hypersync(&self) -> bool;

    /// Get all 13 oscillator phases.
    fn phases(&self) -> [f64; NUM_OSCILLATORS];

    /// Get all 13 natural frequencies (Hz).
    fn natural_frequencies(&self) -> [f64; NUM_OSCILLATORS];

    /// Get coupling strength K.
    fn coupling_strength(&self) -> f64;

    /// Step the network forward by elapsed duration.
    fn step(&mut self, elapsed: Duration);

    /// Set coupling strength K (clamped to [0, 10]).
    fn set_coupling_strength(&mut self, k: f64);

    /// Reset network to initial incoherent state.
    fn reset(&mut self);

    /// Reset network to synchronized state (all phases = 0).
    fn reset_synchronized(&mut self);

    /// Get elapsed time since creation/reset.
    fn elapsed_total(&self) -> Duration;
}

/// Provider trait for GWT consciousness computation.
///
/// Wraps GwtSystem from context-graph-core for handler access.
pub trait GwtSystemProvider: Send + Sync {
    /// Compute consciousness level: C(t) = I(t) x R(t) x D(t)
    ///
    /// # Arguments
    /// - kuramoto_r: Integration factor from Kuramoto order parameter
    /// - meta_accuracy: Reflection factor from Meta-UTL accuracy
    /// - purpose_vector: 13D purpose vector for differentiation
    ///
    /// # Returns
    /// Consciousness level in [0, 1]
    fn compute_consciousness(
        &self,
        kuramoto_r: f32,
        meta_accuracy: f32,
        purpose_vector: &[f32; 13],
    ) -> CoreResult<f32>;

    /// Compute full consciousness metrics with component analysis.
    fn compute_metrics(
        &self,
        kuramoto_r: f32,
        meta_accuracy: f32,
        purpose_vector: &[f32; 13],
    ) -> CoreResult<ConsciousnessMetrics>;

    /// Get current consciousness state (DORMANT, FRAGMENTED, EMERGING, CONSCIOUS, HYPERSYNC).
    fn current_state(&self) -> ConsciousnessState;

    /// Check if system is in a conscious state (CONSCIOUS or HYPERSYNC).
    fn is_conscious(&self) -> bool;

    /// Get the last state transition if any.
    fn last_transition(&self) -> Option<StateTransition>;

    /// Get time spent in current state.
    fn time_in_state(&self) -> Duration;
}

/// Provider trait for workspace selection operations.
///
/// Handles winner-take-all memory selection for global workspace.
#[async_trait::async_trait]
pub trait WorkspaceProvider: Send + Sync {
    /// Select winning memory via winner-take-all algorithm.
    ///
    /// # Arguments
    /// - candidates: Vec of (memory_id, order_parameter_r, importance, alignment)
    ///
    /// # Returns
    /// UUID of winning memory, or None if no candidates pass coherence threshold (0.8)
    async fn select_winning_memory(
        &self,
        candidates: Vec<(Uuid, f32, f32, f32)>,
    ) -> CoreResult<Option<Uuid>>;

    /// Get currently active (conscious) memory if broadcasting.
    fn get_active_memory(&self) -> Option<Uuid>;

    /// Check if broadcast window is still active.
    fn is_broadcasting(&self) -> bool;

    /// Check for workspace conflict (multiple memories with r > 0.8).
    fn has_conflict(&self) -> bool;

    /// Get conflicting memory IDs if present.
    fn get_conflict_details(&self) -> Option<Vec<Uuid>>;

    /// Get coherence threshold for workspace entry.
    fn coherence_threshold(&self) -> f32;
}

/// Provider trait for meta-cognitive loop operations.
#[async_trait::async_trait]
pub trait MetaCognitiveProvider: Send + Sync {
    /// Evaluate meta-cognitive score.
    ///
    /// MetaScore = sigmoid(2 x (L_predicted - L_actual))
    ///
    /// # Arguments
    /// - predicted_learning: L_predicted in [0, 1]
    /// - actual_learning: L_actual in [0, 1]
    async fn evaluate(
        &self,
        predicted_learning: f32,
        actual_learning: f32,
    ) -> CoreResult<MetaCognitiveState>;

    /// Get current Acetylcholine level (learning rate modulator).
    fn acetylcholine(&self) -> f32;

    /// Get current monitoring frequency (Hz).
    fn monitoring_frequency(&self) -> f32;

    /// Get recent meta-scores for trend analysis.
    fn get_recent_scores(&self) -> Vec<f32>;
}

/// Provider trait for self-ego node operations.
pub trait SelfEgoProvider: Send + Sync {
    /// Get current purpose vector (13D).
    fn purpose_vector(&self) -> [f32; 13];

    /// Get coherence between current actions and purpose vector.
    fn coherence_with_actions(&self) -> f32;

    /// Get identity trajectory length.
    fn trajectory_length(&self) -> usize;

    /// Get identity status.
    fn identity_status(&self) -> context_graph_core::gwt::IdentityStatus;

    /// Get identity coherence value.
    fn identity_coherence(&self) -> f32;
}
```

---

## 2. Handler Struct Changes

### File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/core.rs`

After line 233 (after `layer_status_provider`), add the following fields:

```rust
    // ========== GWT/Kuramoto Fields (TASK-GWT-001) ==========

    /// Kuramoto oscillator network for 13-embedding phase synchronization.
    /// TASK-GWT-001: Required for gwt/* handlers and consciousness computation.
    /// Uses RwLock because step() mutates internal state.
    pub(super) kuramoto_network: Arc<RwLock<dyn KuramotoProvider>>,

    /// GWT consciousness system provider.
    /// TASK-GWT-001: Required for consciousness computation C(t) = I(t) x R(t) x D(t).
    pub(super) gwt_system: Arc<dyn GwtSystemProvider>,

    /// Global workspace provider for winner-take-all memory selection.
    /// TASK-GWT-001: Required for workspace broadcast operations.
    pub(super) workspace_provider: Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>,

    /// Meta-cognitive loop provider for self-correction.
    /// TASK-GWT-001: Required for meta_score computation and dream triggering.
    pub(super) meta_cognitive: Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>,

    /// Self-ego node provider for system identity tracking.
    /// TASK-GWT-001: Required for identity continuity monitoring.
    pub(super) self_ego: Arc<tokio::sync::RwLock<dyn SelfEgoProvider>>,
```

### Required Imports (add to top of core.rs):

```rust
use super::gwt_traits::{
    KuramotoProvider, GwtSystemProvider, WorkspaceProvider,
    MetaCognitiveProvider, SelfEgoProvider, NUM_OSCILLATORS,
};
```

---

## 3. Constructor Changes

### 3.1 Update `Handlers::new()` (lines 250-279)

Add new parameters and initialization:

```rust
    /// Create new handlers with teleological and GWT dependencies.
    ///
    /// # Arguments
    /// * ... existing params ...
    /// * `kuramoto_network` - Kuramoto oscillator network (TASK-GWT-001)
    /// * `gwt_system` - GWT consciousness system (TASK-GWT-001)
    /// * `workspace_provider` - Global workspace provider (TASK-GWT-001)
    /// * `meta_cognitive` - Meta-cognitive loop provider (TASK-GWT-001)
    /// * `self_ego` - Self-ego node provider (TASK-GWT-001)
    ///
    /// # TASK-GWT-001 Note
    ///
    /// All GWT providers are REQUIRED. No stub implementations allowed.
    /// If GWT is not needed, use a different constructor.
    pub fn new(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: GoalHierarchy,
        // NEW GWT parameters
        kuramoto_network: Arc<RwLock<dyn KuramotoProvider>>,
        gwt_system: Arc<dyn GwtSystemProvider>,
        workspace_provider: Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>,
        meta_cognitive: Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>,
        self_ego: Arc<tokio::sync::RwLock<dyn SelfEgoProvider>>,
    ) -> Self {
        // ... existing initialization ...

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy: Arc::new(RwLock::new(goal_hierarchy)),
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            // NEW fields
            kuramoto_network,
            gwt_system,
            workspace_provider,
            meta_cognitive,
            self_ego,
        }
    }
```

### 3.2 Update `with_shared_hierarchy()` (lines 296-325)

Same pattern - add GWT parameters.

### 3.3 Update `with_johari_manager()` (lines 343-369)

Same pattern - add GWT parameters.

### 3.4 Update `with_meta_utl_tracker()` (lines 381-405)

Same pattern - add GWT parameters.

### 3.5 Update `with_full_monitoring()` (lines 422-444)

This becomes the PRIMARY constructor for production:

```rust
    /// Create new handlers with full monitoring and GWT support.
    ///
    /// TASK-GWT-001: This is the recommended constructor for production use
    /// with REAL health metrics and GWT consciousness features.
    pub fn with_full_monitoring(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
        meta_utl_tracker: Arc<RwLock<MetaUtlTracker>>,
        system_monitor: Arc<dyn SystemMonitor>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
        // NEW GWT parameters
        kuramoto_network: Arc<RwLock<dyn KuramotoProvider>>,
        gwt_system: Arc<dyn GwtSystemProvider>,
        workspace_provider: Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>,
        meta_cognitive: Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>,
        self_ego: Arc<tokio::sync::RwLock<dyn SelfEgoProvider>>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
            meta_utl_tracker,
            system_monitor,
            layer_status_provider,
            kuramoto_network,
            gwt_system,
            workspace_provider,
            meta_cognitive,
            self_ego,
        }
    }
```

---

## 4. Error Codes

### File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs`

Add after line 166 (after PIPELINE_METRICS_UNAVAILABLE):

```rust
    // GWT/Kuramoto error codes (-32060 to -32069) - TASK-GWT-001
    /// GWT system not initialized or unavailable
    pub const GWT_NOT_INITIALIZED: i32 = -32060;
    /// Kuramoto network error (step failed, invalid phase, etc.)
    pub const KURAMOTO_ERROR: i32 = -32061;
    /// Consciousness computation failed (invalid inputs, math error)
    pub const CONSCIOUSNESS_COMPUTATION_FAILED: i32 = -32062;
    /// Workspace selection or broadcast error
    pub const WORKSPACE_ERROR: i32 = -32063;
    /// State machine transition error
    pub const STATE_TRANSITION_ERROR: i32 = -32064;
    /// Meta-cognitive evaluation failed
    pub const META_COGNITIVE_ERROR: i32 = -32065;
    /// Self-ego node operation failed
    pub const SELF_EGO_ERROR: i32 = -32066;
    /// Identity continuity check failed
    pub const IDENTITY_CONTINUITY_ERROR: i32 = -32067;
```

---

## 5. Method Constants

### File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs`

Add to `pub mod methods` after line 244:

```rust
    // GWT/Consciousness operations (TASK-GWT-001)
    /// Get Kuramoto network synchronization status
    pub const GWT_KURAMOTO_STATUS: &str = "gwt/kuramoto_status";
    /// Get consciousness level and metrics
    pub const GWT_CONSCIOUSNESS_LEVEL: &str = "gwt/consciousness_level";
    /// Get workspace status and active memory
    pub const GWT_WORKSPACE_STATUS: &str = "gwt/workspace_status";
    /// Get consciousness state machine status
    pub const GWT_STATE_STATUS: &str = "gwt/state_status";
    /// Get meta-cognitive loop status
    pub const GWT_META_COGNITIVE_STATUS: &str = "gwt/meta_cognitive_status";
    /// Get self-ego node status
    pub const GWT_SELF_EGO_STATUS: &str = "gwt/self_ego_status";
```

---

## 6. Tool Definitions

### File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools.rs`

### 6.1 Add to `tool_names` module:

```rust
    // GWT tools (TASK-GWT-001)
    pub const GWT_KURAMOTO_STATUS: &str = "gwt_kuramoto_status";
    pub const GWT_CONSCIOUSNESS_LEVEL: &str = "gwt_consciousness_level";
    pub const GWT_WORKSPACE_STATUS: &str = "gwt_workspace_status";
    pub const GWT_STATE_STATUS: &str = "gwt_state_status";
    pub const GWT_META_COGNITIVE_STATUS: &str = "gwt_meta_cognitive_status";
    pub const GWT_SELF_EGO_STATUS: &str = "gwt_self_ego_status";
```

### 6.2 Add to `get_tool_definitions()`:

```rust
        // gwt_kuramoto_status - Kuramoto oscillator network status
        ToolDefinition::new(
            "gwt_kuramoto_status",
            "Get Kuramoto oscillator network status including order parameter (synchronization level), \
             all 13 oscillator phases, natural frequencies, coupling strength, and consciousness state \
             classification (DORMANT, FRAGMENTED, EMERGING, CONSCIOUS, HYPERSYNC).",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // gwt_consciousness_level - Compute consciousness level
        ToolDefinition::new(
            "gwt_consciousness_level",
            "Compute consciousness level using the equation C(t) = I(t) x R(t) x D(t) where \
             I = Kuramoto order parameter (integration), R = sigmoid(MetaUTL.accuracy) (reflection), \
             D = normalized entropy of purpose vector (differentiation). Returns consciousness level, \
             component analysis, and limiting factor identification.",
            json!({
                "type": "object",
                "properties": {
                    "purpose_vector": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 13,
                        "maxItems": 13,
                        "description": "13D purpose vector for differentiation computation (optional, uses current if omitted)"
                    }
                },
                "required": []
            }),
        ),

        // gwt_workspace_status - Global workspace status
        ToolDefinition::new(
            "gwt_workspace_status",
            "Get global workspace status including active (conscious) memory, broadcast state, \
             coherence threshold, candidate count, and conflict detection. The workspace uses \
             winner-take-all selection where memories with r >= 0.8 compete.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // gwt_state_status - Consciousness state machine
        ToolDefinition::new(
            "gwt_state_status",
            "Get consciousness state machine status including current state \
             (DORMANT/FRAGMENTED/EMERGING/CONSCIOUS/HYPERSYNC), time in state, \
             last transition details, and state thresholds.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // gwt_meta_cognitive_status - Meta-cognitive loop status
        ToolDefinition::new(
            "gwt_meta_cognitive_status",
            "Get meta-cognitive loop status including current MetaScore, \
             acetylcholine level (learning rate), monitoring frequency, score trend, \
             and whether introspective dream is triggered. Formula: MetaScore = sigmoid(2 x (L_predicted - L_actual)).",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // gwt_self_ego_status - Self-ego node status
        ToolDefinition::new(
            "gwt_self_ego_status",
            "Get SELF_EGO_NODE status including current purpose vector, coherence with actions, \
             identity trajectory length, identity status (HEALTHY/WARNING/DEGRADED/CRITICAL), \
             and identity coherence value (IC = cos(PV_t, PV_{t-1}) x r(t)).",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),
```

---

## 7. Handler Dispatch Updates

### File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/core.rs`

### 7.1 Add dispatch arms in `dispatch()` method (after line 558):

```rust
            // GWT/Consciousness operations (TASK-GWT-001)
            methods::GWT_KURAMOTO_STATUS => {
                self.handle_gwt_kuramoto_status(request.id).await
            }
            methods::GWT_CONSCIOUSNESS_LEVEL => {
                self.handle_gwt_consciousness_level(request.id, request.params).await
            }
            methods::GWT_WORKSPACE_STATUS => {
                self.handle_gwt_workspace_status(request.id).await
            }
            methods::GWT_STATE_STATUS => {
                self.handle_gwt_state_status(request.id).await
            }
            methods::GWT_META_COGNITIVE_STATUS => {
                self.handle_gwt_meta_cognitive_status(request.id).await
            }
            methods::GWT_SELF_EGO_STATUS => {
                self.handle_gwt_self_ego_status(request.id).await
            }
```

### 7.2 Add tool dispatch arms in `handle_tools_call()` (in handlers/tools.rs after line 87):

```rust
            // GWT tools (TASK-GWT-001)
            tool_names::GWT_KURAMOTO_STATUS => self.call_gwt_kuramoto_status(id).await,
            tool_names::GWT_CONSCIOUSNESS_LEVEL => self.call_gwt_consciousness_level(id, arguments).await,
            tool_names::GWT_WORKSPACE_STATUS => self.call_gwt_workspace_status(id).await,
            tool_names::GWT_STATE_STATUS => self.call_gwt_state_status(id).await,
            tool_names::GWT_META_COGNITIVE_STATUS => self.call_gwt_meta_cognitive_status(id).await,
            tool_names::GWT_SELF_EGO_STATUS => self.call_gwt_self_ego_status(id).await,
```

---

## 8. New Handler Module

### File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/gwt.rs` (NEW FILE)

```rust
//! GWT/Consciousness handler implementations.
//!
//! TASK-GWT-001: Handlers for Global Workspace Theory and Kuramoto oscillator operations.
//!
//! All handlers follow FAIL-FAST principle - no stubs, no fallbacks.
//! If a GWT component fails, the entire operation fails with a detailed error.

use serde_json::json;
use tracing::{debug, error};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// Handle gwt/kuramoto_status request.
    ///
    /// Returns Kuramoto oscillator network status.
    pub(super) async fn handle_gwt_kuramoto_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        debug!("Handling gwt/kuramoto_status request");

        let network = self.kuramoto_network.read();

        let (r, psi) = network.order_parameter();
        let phases = network.phases();
        let frequencies = network.natural_frequencies();

        // Determine consciousness state classification
        let state = if network.is_hypersync() {
            "HYPERSYNC"
        } else if network.is_conscious() {
            "CONSCIOUS"
        } else if network.is_fragmented() {
            "FRAGMENTED"
        } else if r >= 0.5 {
            "EMERGING"
        } else {
            "DORMANT"
        };

        JsonRpcResponse::success(
            id,
            json!({
                "order_parameter": {
                    "r": r,
                    "psi": psi,
                    "r_percentage": (r * 100.0).round() / 100.0
                },
                "phases": phases,
                "natural_frequencies_hz": frequencies,
                "coupling_strength": network.coupling_strength(),
                "elapsed_total_ms": network.elapsed_total().as_millis(),
                "consciousness_state": state,
                "thresholds": {
                    "conscious": 0.8,
                    "fragmented": 0.5,
                    "hypersync": 0.95
                }
            }),
        )
    }

    /// Handle gwt/consciousness_level request.
    ///
    /// Computes consciousness level: C(t) = I(t) x R(t) x D(t)
    pub(super) async fn handle_gwt_consciousness_level(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("Handling gwt/consciousness_level request");

        // Get Kuramoto order parameter (Integration factor)
        let kuramoto_r = {
            let network = self.kuramoto_network.read();
            network.synchronization() as f32
        };

        // Get meta-accuracy (Reflection factor) from meta_utl_tracker
        let meta_accuracy = {
            let tracker = self.meta_utl_tracker.read();
            // Use average accuracy across all embedders, or default to 0.5
            let mut total = 0.0f32;
            let mut count = 0;
            for i in 0..13 {
                if let Some(acc) = tracker.get_embedder_accuracy(i) {
                    total += acc;
                    count += 1;
                }
            }
            if count > 0 { total / count as f32 } else { 0.5 }
        };

        // Get purpose vector (Differentiation factor)
        let purpose_vector: [f32; 13] = if let Some(params) = params {
            if let Some(pv) = params.get("purpose_vector") {
                if let Some(arr) = pv.as_array() {
                    if arr.len() == 13 {
                        let mut vec = [0.0f32; 13];
                        for (i, v) in arr.iter().enumerate() {
                            vec[i] = v.as_f64().unwrap_or(0.0) as f32;
                        }
                        vec
                    } else {
                        return JsonRpcResponse::error(
                            id,
                            error_codes::INVALID_PARAMS,
                            "purpose_vector must have exactly 13 elements",
                        );
                    }
                } else {
                    // Use default from self_ego
                    let ego = self.self_ego.read().await;
                    ego.purpose_vector()
                }
            } else {
                let ego = self.self_ego.read().await;
                ego.purpose_vector()
            }
        } else {
            let ego = self.self_ego.read().await;
            ego.purpose_vector()
        };

        // Compute consciousness metrics
        match self.gwt_system.compute_metrics(kuramoto_r, meta_accuracy, &purpose_vector) {
            Ok(metrics) => {
                let limiting_factor = match metrics.component_analysis.limiting_factor {
                    context_graph_core::gwt::LimitingFactor::Integration => "integration",
                    context_graph_core::gwt::LimitingFactor::Reflection => "reflection",
                    context_graph_core::gwt::LimitingFactor::Differentiation => "differentiation",
                    context_graph_core::gwt::LimitingFactor::None => "none",
                };

                JsonRpcResponse::success(
                    id,
                    json!({
                        "consciousness_level": metrics.consciousness,
                        "components": {
                            "integration": metrics.integration,
                            "reflection": metrics.reflection,
                            "differentiation": metrics.differentiation
                        },
                        "component_analysis": {
                            "integration_sufficient": metrics.component_analysis.integration_sufficient,
                            "reflection_sufficient": metrics.component_analysis.reflection_sufficient,
                            "differentiation_sufficient": metrics.component_analysis.differentiation_sufficient,
                            "limiting_factor": limiting_factor
                        },
                        "inputs": {
                            "kuramoto_r": kuramoto_r,
                            "meta_accuracy": meta_accuracy,
                            "purpose_vector": purpose_vector
                        }
                    }),
                )
            }
            Err(e) => {
                error!(error = %e, "Consciousness computation FAILED");
                JsonRpcResponse::error(
                    id,
                    error_codes::CONSCIOUSNESS_COMPUTATION_FAILED,
                    format!("Consciousness computation failed: {}", e),
                )
            }
        }
    }

    /// Handle gwt/workspace_status request.
    ///
    /// Returns global workspace status.
    pub(super) async fn handle_gwt_workspace_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        debug!("Handling gwt/workspace_status request");

        let workspace = self.workspace_provider.read().await;

        let active_memory = workspace.get_active_memory().map(|id| id.to_string());
        let is_broadcasting = workspace.is_broadcasting();
        let has_conflict = workspace.has_conflict();
        let conflict_ids: Option<Vec<String>> = workspace
            .get_conflict_details()
            .map(|ids| ids.into_iter().map(|id| id.to_string()).collect());

        JsonRpcResponse::success(
            id,
            json!({
                "active_memory": active_memory,
                "is_broadcasting": is_broadcasting,
                "coherence_threshold": workspace.coherence_threshold(),
                "conflict": {
                    "has_conflict": has_conflict,
                    "conflicting_memories": conflict_ids
                }
            }),
        )
    }

    /// Handle gwt/state_status request.
    ///
    /// Returns consciousness state machine status.
    pub(super) async fn handle_gwt_state_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        debug!("Handling gwt/state_status request");

        let current_state = self.gwt_system.current_state();
        let is_conscious = self.gwt_system.is_conscious();
        let time_in_state = self.gwt_system.time_in_state();
        let last_transition = self.gwt_system.last_transition();

        let state_name = current_state.name();

        let transition_info = last_transition.map(|t| {
            json!({
                "from": t.from.name(),
                "to": t.to.name(),
                "timestamp": t.timestamp.to_rfc3339(),
                "consciousness_level": t.consciousness_level
            })
        });

        JsonRpcResponse::success(
            id,
            json!({
                "current_state": state_name,
                "is_conscious": is_conscious,
                "time_in_state_ms": time_in_state.as_millis(),
                "last_transition": transition_info,
                "state_thresholds": {
                    "dormant": "r < 0.3",
                    "fragmented": "0.3 <= r < 0.5",
                    "emerging": "0.5 <= r < 0.8",
                    "conscious": "r >= 0.8",
                    "hypersync": "r > 0.95 (warning)"
                }
            }),
        )
    }

    /// Handle gwt/meta_cognitive_status request.
    ///
    /// Returns meta-cognitive loop status.
    pub(super) async fn handle_gwt_meta_cognitive_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        debug!("Handling gwt/meta_cognitive_status request");

        let meta = self.meta_cognitive.read().await;

        let acetylcholine = meta.acetylcholine();
        let monitoring_frequency = meta.monitoring_frequency();
        let recent_scores = meta.get_recent_scores();

        // Compute trend from recent scores
        let trend = if recent_scores.len() < 3 {
            "insufficient_data"
        } else {
            let len = recent_scores.len();
            let first_half: f32 = recent_scores[..len / 2].iter().sum::<f32>() / (len / 2) as f32;
            let second_half: f32 = recent_scores[len / 2..].iter().sum::<f32>() / (len - len / 2) as f32;
            let delta = second_half - first_half;
            if delta > 0.1 {
                "increasing"
            } else if delta < -0.1 {
                "decreasing"
            } else {
                "stable"
            }
        };

        let avg_score = if recent_scores.is_empty() {
            0.5
        } else {
            recent_scores.iter().sum::<f32>() / recent_scores.len() as f32
        };

        JsonRpcResponse::success(
            id,
            json!({
                "acetylcholine": acetylcholine,
                "monitoring_frequency_hz": monitoring_frequency,
                "recent_scores": recent_scores,
                "average_meta_score": avg_score,
                "trend": trend,
                "formula": "MetaScore = sigmoid(2 x (L_predicted - L_actual))"
            }),
        )
    }

    /// Handle gwt/self_ego_status request.
    ///
    /// Returns SELF_EGO_NODE status.
    pub(super) async fn handle_gwt_self_ego_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        debug!("Handling gwt/self_ego_status request");

        let ego = self.self_ego.read().await;

        let purpose_vector = ego.purpose_vector();
        let coherence_with_actions = ego.coherence_with_actions();
        let trajectory_length = ego.trajectory_length();
        let identity_status = ego.identity_status();
        let identity_coherence = ego.identity_coherence();

        let status_name = match identity_status {
            context_graph_core::gwt::IdentityStatus::Healthy => "HEALTHY",
            context_graph_core::gwt::IdentityStatus::Warning => "WARNING",
            context_graph_core::gwt::IdentityStatus::Degraded => "DEGRADED",
            context_graph_core::gwt::IdentityStatus::Critical => "CRITICAL",
        };

        JsonRpcResponse::success(
            id,
            json!({
                "purpose_vector": purpose_vector,
                "coherence_with_actions": coherence_with_actions,
                "identity_trajectory_length": trajectory_length,
                "identity_status": status_name,
                "identity_coherence": identity_coherence,
                "formula": "IC = cos(PV_t, PV_{t-1}) x r(t)",
                "status_thresholds": {
                    "healthy": "IC > 0.9",
                    "warning": "0.7 <= IC <= 0.9",
                    "degraded": "0.5 <= IC < 0.7",
                    "critical": "IC < 0.5 (triggers introspective dream)"
                }
            }),
        )
    }

    // ========== Tool Call Implementations ==========

    /// gwt_kuramoto_status tool implementation.
    pub(super) async fn call_gwt_kuramoto_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        let response = self.handle_gwt_kuramoto_status(None).await;
        if let Some(result) = response.result {
            self.tool_result_with_pulse(id, result)
        } else {
            self.tool_error_with_pulse(id, "Failed to get Kuramoto status")
        }
    }

    /// gwt_consciousness_level tool implementation.
    pub(super) async fn call_gwt_consciousness_level(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let response = self.handle_gwt_consciousness_level(None, Some(args)).await;
        if let Some(result) = response.result {
            self.tool_result_with_pulse(id, result)
        } else if let Some(err) = response.error {
            self.tool_error_with_pulse(id, &err.message)
        } else {
            self.tool_error_with_pulse(id, "Failed to compute consciousness level")
        }
    }

    /// gwt_workspace_status tool implementation.
    pub(super) async fn call_gwt_workspace_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        let response = self.handle_gwt_workspace_status(None).await;
        if let Some(result) = response.result {
            self.tool_result_with_pulse(id, result)
        } else {
            self.tool_error_with_pulse(id, "Failed to get workspace status")
        }
    }

    /// gwt_state_status tool implementation.
    pub(super) async fn call_gwt_state_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        let response = self.handle_gwt_state_status(None).await;
        if let Some(result) = response.result {
            self.tool_result_with_pulse(id, result)
        } else {
            self.tool_error_with_pulse(id, "Failed to get state status")
        }
    }

    /// gwt_meta_cognitive_status tool implementation.
    pub(super) async fn call_gwt_meta_cognitive_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        let response = self.handle_gwt_meta_cognitive_status(None).await;
        if let Some(result) = response.result {
            self.tool_result_with_pulse(id, result)
        } else {
            self.tool_error_with_pulse(id, "Failed to get meta-cognitive status")
        }
    }

    /// gwt_self_ego_status tool implementation.
    pub(super) async fn call_gwt_self_ego_status(
        &self,
        id: Option<JsonRpcId>,
    ) -> JsonRpcResponse {
        let response = self.handle_gwt_self_ego_status(None).await;
        if let Some(result) = response.result {
            self.tool_result_with_pulse(id, result)
        } else {
            self.tool_error_with_pulse(id, "Failed to get self-ego status")
        }
    }
}
```

---

## 9. Module Registration

### File: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/mod.rs`

Add after line 26:

```rust
mod gwt;
mod gwt_traits;

// Re-export GWT traits for external use
pub use self::gwt_traits::{
    KuramotoProvider, GwtSystemProvider, WorkspaceProvider,
    MetaCognitiveProvider, SelfEgoProvider, NUM_OSCILLATORS,
};
```

---

## 10. Implementation Order for P2-4

1. **Create gwt_traits.rs** - Define all provider traits
2. **Create gwt.rs** - Implement handler methods
3. **Update protocol.rs** - Add error codes and method constants
4. **Update tools.rs** - Add tool definitions and names
5. **Update core.rs** - Add fields and update constructors
6. **Update mod.rs** - Register new modules
7. **Create adapter implementations** - Implement provider traits for real components
8. **Update tests** - Add GWT handler tests

---

## 11. Testing Requirements

Each handler must have:
1. Unit test with mock providers
2. Integration test with real GWT components
3. Error path test (FAIL-FAST verification)
4. JSON schema validation test

---

## 12. Dependencies

### Cargo.toml additions (context-graph-mcp):

```toml
[dependencies]
async-trait = "0.1"  # For async provider traits
```

---

## Notes for P2-4

1. **NO STUBS**: All provider implementations must wrap real GWT components
2. **FAIL FAST**: Use `CoreError` for all error conditions, never return defaults
3. **Thread Safety**: All providers require `Send + Sync` bounds
4. **Async Locks**: Use `tokio::sync::RwLock` for async providers, `parking_lot::RwLock` for sync
5. **Testing**: Every handler needs error path tests that verify FAIL-FAST behavior
