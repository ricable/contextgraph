# TASK-DREAM-P0-001: Dream Layer Types and Interfaces

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-001 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 1 (Foundation) |
| **Priority** | P0 - Critical |
| **Effort** | 2 hours |
| **Dependencies** | None |
| **Blocks** | TASK-DREAM-P0-002, TASK-DREAM-P0-003, TASK-DREAM-P0-004, TASK-DREAM-P0-005 |

---

## 1. Objective

Define all new types, configuration structs, and trait interfaces required for the Dream Layer implementation. This establishes the foundation for Hebbian learning, hyperbolic walks, and trigger mechanisms.

---

## 2. Input Context Files

```yaml
must_read:
  - path: crates/context-graph-core/src/dream/mod.rs
    purpose: Understand existing dream module structure and exports
  - path: crates/context-graph-core/src/dream/nrem.rs
    purpose: See existing NremPhase struct for extension
  - path: crates/context-graph-core/src/dream/rem.rs
    purpose: See existing RemPhase struct for extension
  - path: crates/context-graph-core/src/dream/scheduler.rs
    purpose: See existing DreamScheduler for extension
  - path: crates/context-graph-graph/src/hyperbolic/poincare/types.rs
    purpose: Reference PoincarePoint type for integration

should_read:
  - path: crates/context-graph-core/src/types/graph_edge/edge.rs
    purpose: Understand GraphEdge for Hebbian updates
  - path: crates/context-graph-core/src/error.rs
    purpose: Understand error types for new error variants
```

---

## 3. Files to Create/Modify

### 3.1 Create: `crates/context-graph-core/src/dream/types.rs`

```rust
//! Dream Layer Types
//!
//! Defines types for Hebbian learning, hyperbolic walks, and dream triggers.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-export PoincarePoint for convenience (from graph crate)
// Note: This will be imported by dependent modules

/// Configuration for Hebbian learning in NREM phase.
///
/// Constitution Reference: Section dream.phases.nrem
///
/// # Example
///
/// ```
/// use context_graph_core::dream::HebbianConfig;
///
/// let config = HebbianConfig::default();
/// assert_eq!(config.learning_rate, 0.01);
/// assert_eq!(config.weight_decay, 0.001);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HebbianConfig {
    /// Learning rate (eta) for weight updates.
    /// Constitution: default 0.01
    pub learning_rate: f32,

    /// Weight decay factor applied per NREM cycle.
    /// Constitution: 0.001
    pub weight_decay: f32,

    /// Minimum weight before edge is marked for pruning.
    /// Constitution: 0.05
    pub weight_floor: f32,

    /// Maximum weight cap to prevent runaway strengthening.
    /// Constitution: 1.0
    pub weight_cap: f32,

    /// Kuramoto coupling strength for neural synchronization.
    /// Constitution: 10.0 during NREM
    pub coupling_strength: f32,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            weight_decay: 0.001,
            weight_floor: 0.05,
            weight_cap: 1.0,
            coupling_strength: 10.0,
        }
    }
}

/// Activation (phi) value for a node during replay.
///
/// Represents the "firing" level of a node during NREM replay.
/// Used in Hebbian update formula: dw_ij = eta * phi_i * phi_j
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeActivation {
    /// Node identifier
    pub node_id: Uuid,

    /// Activation level [0.0, 1.0]
    /// 0.0 = not firing, 1.0 = maximum firing
    pub phi: f32,

    /// Timestamp when activation was recorded
    #[serde(skip)]
    pub timestamp: Option<Instant>,
}

impl NodeActivation {
    /// Create a new node activation.
    ///
    /// # Arguments
    ///
    /// * `node_id` - The node's UUID
    /// * `phi` - Activation level (clamped to [0.0, 1.0])
    pub fn new(node_id: Uuid, phi: f32) -> Self {
        Self {
            node_id,
            phi: phi.clamp(0.0, 1.0),
            timestamp: Some(Instant::now()),
        }
    }

    /// Check if this is a significant activation (> 0.1)
    pub fn is_significant(&self) -> bool {
        self.phi > 0.1
    }
}

/// Configuration for hyperbolic random walk in REM phase.
///
/// Constitution Reference: Section dream.phases.rem
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperbolicWalkConfig {
    /// Step size in Poincare ball (Euclidean distance before Mobius add)
    /// Default: 0.1
    pub step_size: f32,

    /// Maximum steps per walk before termination
    /// Default: 50
    pub max_steps: usize,

    /// Exploration temperature for direction sampling
    /// Constitution: 2.0
    pub temperature: f32,

    /// Minimum distance from nearest memory to consider a blind spot
    /// Constitution: semantic_leap >= 0.7
    pub min_blind_spot_distance: f32,

    /// Number of random direction samples per step
    /// Default: 8
    pub direction_samples: usize,
}

impl Default for HyperbolicWalkConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            max_steps: 50,
            temperature: 2.0,
            min_blind_spot_distance: 0.7,
            direction_samples: 8,
        }
    }
}

/// A single step in the hyperbolic random walk.
///
/// Records position, direction, and distance for trajectory analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkStep {
    /// Current position in Poincare ball (64D)
    /// Stored as flat array for serialization
    pub position: [f32; 64],

    /// Direction of the step taken (unit vector before scaling)
    pub step_direction: [f32; 64],

    /// Geodesic distance from walk start position
    pub distance_from_start: f32,

    /// Step index in the walk (0-based)
    pub step_index: usize,

    /// Whether a blind spot was detected at this position
    pub blind_spot_detected: bool,
}

impl WalkStep {
    /// Create a new walk step.
    pub fn new(
        position: [f32; 64],
        direction: [f32; 64],
        distance: f32,
        index: usize,
    ) -> Self {
        Self {
            position,
            step_direction: direction,
            distance_from_start: distance,
            step_index: index,
            blind_spot_detected: false,
        }
    }

    /// Mark this step as containing a blind spot.
    pub fn mark_blind_spot(&mut self) {
        self.blind_spot_detected = true;
    }
}

/// Entropy tracking window for dream trigger.
///
/// Monitors system entropy over a sliding window to detect
/// sustained high entropy (> 0.7 for 5 minutes).
#[derive(Debug, Clone)]
pub struct EntropyWindow {
    /// Entropy samples with timestamps
    samples: VecDeque<(Instant, f32)>,

    /// Duration of the sliding window
    /// Constitution: 5 minutes
    pub window_duration: Duration,

    /// Entropy threshold for trigger
    /// Constitution: 0.7
    pub threshold: f32,
}

impl EntropyWindow {
    /// Create a new entropy window with constitution defaults.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(300), // 5 min at 1 sample/sec
            window_duration: Duration::from_secs(300), // 5 minutes
            threshold: 0.7,
        }
    }

    /// Create with custom parameters (for testing).
    pub fn with_params(window_duration: Duration, threshold: f32) -> Self {
        let capacity = (window_duration.as_secs() + 1) as usize;
        Self {
            samples: VecDeque::with_capacity(capacity),
            window_duration,
            threshold,
        }
    }

    /// Add an entropy sample.
    pub fn push(&mut self, entropy: f32) {
        let now = Instant::now();

        // Add new sample
        self.samples.push_back((now, entropy.clamp(0.0, 1.0)));

        // Remove samples outside window
        self.prune_old_samples(now);
    }

    /// Check if entropy trigger condition is met.
    ///
    /// Returns true if:
    /// 1. Window is full (5 minutes of data)
    /// 2. All samples are above threshold
    pub fn should_trigger(&self) -> bool {
        if self.samples.is_empty() {
            return false;
        }

        // Check if we have enough samples (window is full)
        let oldest = self.samples.front().map(|(t, _)| *t);
        let newest = self.samples.back().map(|(t, _)| *t);

        if let (Some(oldest), Some(newest)) = (oldest, newest) {
            let duration = newest.duration_since(oldest);
            if duration < self.window_duration {
                return false; // Not enough data yet
            }
        } else {
            return false;
        }

        // Check if all samples are above threshold
        self.samples.iter().all(|(_, e)| *e > self.threshold)
    }

    /// Get current average entropy.
    pub fn average(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().map(|(_, e)| e).sum();
        sum / self.samples.len() as f32
    }

    /// Get minimum entropy in window.
    pub fn minimum(&self) -> f32 {
        self.samples
            .iter()
            .map(|(_, e)| *e)
            .fold(f32::INFINITY, f32::min)
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Remove samples outside the window.
    fn prune_old_samples(&mut self, now: Instant) {
        while let Some((timestamp, _)) = self.samples.front() {
            if now.duration_since(*timestamp) > self.window_duration {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}

impl Default for EntropyWindow {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU utilization trigger state.
///
/// Monitors GPU usage to trigger dream cycles when load is high.
#[derive(Debug, Clone)]
pub struct GpuTriggerState {
    /// Current GPU usage [0.0, 1.0]
    pub current_usage: f32,

    /// Threshold for trigger
    /// Constitution: 0.80 (80%)
    pub threshold: f32,

    /// Recent usage samples for smoothing
    samples: VecDeque<f32>,

    /// Maximum samples to retain
    max_samples: usize,

    /// Whether trigger has fired (reset after dream)
    pub triggered: bool,
}

impl GpuTriggerState {
    /// Create a new GPU trigger state with constitution defaults.
    pub fn new() -> Self {
        Self {
            current_usage: 0.0,
            threshold: 0.80,
            samples: VecDeque::with_capacity(10),
            max_samples: 10,
            triggered: false,
        }
    }

    /// Create with custom threshold (for testing).
    pub fn with_threshold(threshold: f32) -> Self {
        let mut state = Self::new();
        state.threshold = threshold.clamp(0.0, 1.0);
        state
    }

    /// Update with new GPU usage reading.
    pub fn update(&mut self, usage: f32) {
        let usage = usage.clamp(0.0, 1.0);

        self.samples.push_back(usage);
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }

        // Use smoothed average for stability
        self.current_usage = self.average();
    }

    /// Check if GPU trigger condition is met.
    pub fn should_trigger(&self) -> bool {
        !self.triggered && self.current_usage >= self.threshold
    }

    /// Mark trigger as fired.
    pub fn mark_triggered(&mut self) {
        self.triggered = true;
    }

    /// Reset trigger state after dream completes.
    pub fn reset(&mut self) {
        self.triggered = false;
        self.samples.clear();
        self.current_usage = 0.0;
    }

    /// Get smoothed average usage.
    pub fn average(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().sum();
        sum / self.samples.len() as f32
    }
}

impl Default for GpuTriggerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Reason for triggering a dream cycle (extended).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtendedTriggerReason {
    /// Activity below threshold for idle duration
    IdleTimeout,

    /// Entropy above 0.7 for 5 minutes
    HighEntropy,

    /// GPU usage above 80%
    GpuOverload,

    /// Memory pressure requires consolidation
    MemoryPressure,

    /// Manual trigger by user/system
    Manual,

    /// Scheduled dream time
    Scheduled,
}

impl std::fmt::Display for ExtendedTriggerReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdleTimeout => write!(f, "idle_timeout"),
            Self::HighEntropy => write!(f, "high_entropy"),
            Self::GpuOverload => write!(f, "gpu_overload"),
            Self::MemoryPressure => write!(f, "memory_pressure"),
            Self::Manual => write!(f, "manual"),
            Self::Scheduled => write!(f, "scheduled"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_hebbian_config_default() {
        let config = HebbianConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.weight_decay, 0.001);
        assert_eq!(config.weight_floor, 0.05);
        assert_eq!(config.weight_cap, 1.0);
        assert_eq!(config.coupling_strength, 10.0);
    }

    #[test]
    fn test_node_activation_clamping() {
        let activation = NodeActivation::new(Uuid::new_v4(), 1.5);
        assert_eq!(activation.phi, 1.0);

        let activation = NodeActivation::new(Uuid::new_v4(), -0.5);
        assert_eq!(activation.phi, 0.0);
    }

    #[test]
    fn test_node_activation_significance() {
        let significant = NodeActivation::new(Uuid::new_v4(), 0.5);
        assert!(significant.is_significant());

        let not_significant = NodeActivation::new(Uuid::new_v4(), 0.05);
        assert!(!not_significant.is_significant());
    }

    #[test]
    fn test_hyperbolic_walk_config_default() {
        let config = HyperbolicWalkConfig::default();
        assert_eq!(config.step_size, 0.1);
        assert_eq!(config.max_steps, 50);
        assert_eq!(config.temperature, 2.0);
        assert_eq!(config.min_blind_spot_distance, 0.7);
    }

    #[test]
    fn test_walk_step_creation() {
        let position = [0.0; 64];
        let direction = [0.1; 64];
        let step = WalkStep::new(position, direction, 0.5, 3);

        assert_eq!(step.step_index, 3);
        assert_eq!(step.distance_from_start, 0.5);
        assert!(!step.blind_spot_detected);
    }

    #[test]
    fn test_walk_step_blind_spot_marking() {
        let mut step = WalkStep::new([0.0; 64], [0.0; 64], 0.0, 0);
        assert!(!step.blind_spot_detected);

        step.mark_blind_spot();
        assert!(step.blind_spot_detected);
    }

    #[test]
    fn test_entropy_window_empty() {
        let window = EntropyWindow::new();
        assert!(!window.should_trigger());
        assert_eq!(window.average(), 0.0);
    }

    #[test]
    fn test_entropy_window_push_and_average() {
        let mut window = EntropyWindow::new();
        window.push(0.8);
        window.push(0.9);
        window.push(0.85);

        assert!((window.average() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_entropy_window_threshold_check() {
        // Use short window for testing
        let mut window = EntropyWindow::with_params(Duration::from_millis(50), 0.7);

        // Add samples above threshold
        window.push(0.8);
        thread::sleep(Duration::from_millis(60)); // Wait for window duration
        window.push(0.9);

        // Should trigger after window is full
        assert!(window.should_trigger());
    }

    #[test]
    fn test_entropy_window_below_threshold() {
        let mut window = EntropyWindow::with_params(Duration::from_millis(50), 0.7);

        window.push(0.5); // Below threshold
        thread::sleep(Duration::from_millis(60));
        window.push(0.8);

        // Should NOT trigger because not all samples above threshold
        assert!(!window.should_trigger());
    }

    #[test]
    fn test_gpu_trigger_state_default() {
        let state = GpuTriggerState::new();
        assert_eq!(state.threshold, 0.80);
        assert_eq!(state.current_usage, 0.0);
        assert!(!state.triggered);
    }

    #[test]
    fn test_gpu_trigger_threshold() {
        let mut state = GpuTriggerState::new();

        state.update(0.5);
        assert!(!state.should_trigger());

        state.update(0.85);
        assert!(state.should_trigger());
    }

    #[test]
    fn test_gpu_trigger_once_only() {
        let mut state = GpuTriggerState::new();

        state.update(0.85);
        assert!(state.should_trigger());

        state.mark_triggered();
        assert!(!state.should_trigger()); // Should not trigger again
    }

    #[test]
    fn test_gpu_trigger_reset() {
        let mut state = GpuTriggerState::new();

        state.update(0.85);
        state.mark_triggered();
        assert!(!state.should_trigger());

        state.reset();
        state.update(0.85);
        assert!(state.should_trigger()); // Can trigger again after reset
    }

    #[test]
    fn test_extended_trigger_reason_display() {
        assert_eq!(ExtendedTriggerReason::IdleTimeout.to_string(), "idle_timeout");
        assert_eq!(ExtendedTriggerReason::HighEntropy.to_string(), "high_entropy");
        assert_eq!(ExtendedTriggerReason::GpuOverload.to_string(), "gpu_overload");
    }
}
```

### 3.2 Modify: `crates/context-graph-core/src/dream/mod.rs`

Add export for new types module:

```rust
// Add after line 51:
pub mod types;

// Add to re-exports after line 58:
pub use types::{
    HebbianConfig,
    NodeActivation,
    HyperbolicWalkConfig,
    WalkStep,
    EntropyWindow,
    GpuTriggerState,
    ExtendedTriggerReason,
};
```

---

## 4. Definition of Done

### 4.1 Type Signatures (Exact)

```rust
// All types must have these exact signatures:

pub struct HebbianConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub weight_floor: f32,
    pub weight_cap: f32,
    pub coupling_strength: f32,
}

pub struct NodeActivation {
    pub node_id: Uuid,
    pub phi: f32,
    pub timestamp: Option<Instant>,
}

pub struct HyperbolicWalkConfig {
    pub step_size: f32,
    pub max_steps: usize,
    pub temperature: f32,
    pub min_blind_spot_distance: f32,
    pub direction_samples: usize,
}

pub struct WalkStep {
    pub position: [f32; 64],
    pub step_direction: [f32; 64],
    pub distance_from_start: f32,
    pub step_index: usize,
    pub blind_spot_detected: bool,
}

pub struct EntropyWindow { /* internal fields */ }
impl EntropyWindow {
    pub fn new() -> Self;
    pub fn push(&mut self, entropy: f32);
    pub fn should_trigger(&self) -> bool;
    pub fn average(&self) -> f32;
    pub fn minimum(&self) -> f32;
    pub fn clear(&mut self);
}

pub struct GpuTriggerState { /* internal fields */ }
impl GpuTriggerState {
    pub fn new() -> Self;
    pub fn update(&mut self, usage: f32);
    pub fn should_trigger(&self) -> bool;
    pub fn mark_triggered(&mut self);
    pub fn reset(&mut self);
}

pub enum ExtendedTriggerReason {
    IdleTimeout,
    HighEntropy,
    GpuOverload,
    MemoryPressure,
    Manual,
    Scheduled,
}
```

### 4.2 Validation Criteria

| Criterion | Check |
|-----------|-------|
| Compiles without errors | `cargo build -p context-graph-core` |
| All tests pass | `cargo test -p context-graph-core dream::types` |
| No clippy warnings | `cargo clippy -p context-graph-core -- -D warnings` |
| Types are exported | Import works from `context_graph_core::dream::*` |
| Default implementations | All structs have `Default` impl where required |
| Serde derives | All public structs serialize/deserialize |
| Constitution constants | HebbianConfig defaults match constitution |
| EntropyWindow window | 5 minutes, threshold 0.7 |
| GpuTriggerState threshold | 0.80 (80%) |

### 4.3 Test Coverage Requirements

- [ ] Unit test for each public function
- [ ] Test HebbianConfig default values match constitution
- [ ] Test NodeActivation phi clamping [0.0, 1.0]
- [ ] Test HyperbolicWalkConfig default temperature = 2.0
- [ ] Test EntropyWindow sliding window behavior
- [ ] Test EntropyWindow trigger only when window is full
- [ ] Test GpuTriggerState threshold at exactly 80%
- [ ] Test GpuTriggerState triggers only once until reset

---

## 5. Implementation Notes

### 5.1 Key Decisions

1. **64D Position Arrays**: Use `[f32; 64]` for Poincare positions to match `PoincarePoint` without cross-crate dependency in types module.

2. **Optional Timestamp**: `NodeActivation::timestamp` is `Option<Instant>` to support serialization (Instant is not serializable).

3. **Internal Fields**: `EntropyWindow` and `GpuTriggerState` hide internal `VecDeque` to allow future optimization.

4. **Capacity Hints**: Pre-allocate `VecDeque` capacity based on expected window sizes.

### 5.2 Constitution Compliance Checklist

- [x] `learning_rate = 0.01` (HebbianConfig default)
- [x] `weight_decay = 0.001` (HebbianConfig default)
- [x] `weight_floor = 0.05` (HebbianConfig default)
- [x] `weight_cap = 1.0` (HebbianConfig default)
- [x] `coupling_strength = 10.0` (HebbianConfig default, Kuramoto K)
- [x] `temperature = 2.0` (HyperbolicWalkConfig default)
- [x] `min_blind_spot_distance = 0.7` (HyperbolicWalkConfig default, semantic_leap)
- [x] `window_duration = 5 minutes` (EntropyWindow default)
- [x] `entropy_threshold = 0.7` (EntropyWindow default)
- [x] `gpu_threshold = 0.80` (GpuTriggerState default)

---

## 6. Estimated Effort Breakdown

| Phase | Duration |
|-------|----------|
| Type definitions | 30 min |
| Implementation (methods) | 45 min |
| Unit tests | 30 min |
| Documentation | 10 min |
| Integration with mod.rs | 5 min |
| **Total** | **2 hours** |
