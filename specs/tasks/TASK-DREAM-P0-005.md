# TASK-DREAM-P0-005: Dream Trigger Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-005 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 2 (Logic) |
| **Priority** | P0 - Critical |
| **Effort** | 3 hours |
| **Dependencies** | TASK-DREAM-P0-001 (COMPLETED - types.rs provides EntropyWindow, GpuTriggerState, ExtendedTriggerReason) |
| **Blocks** | TASK-DREAM-P0-006 (Wake Controller and MCP Integration) |
| **Status** | **COMPLETED** |
| **Completed Date** | 2026-01-11 |
| **Last Audited** | 2026-01-11 |

---

## 1. Objective

Create `triggers.rs` module that implements a unified `TriggerManager` to coordinate all dream trigger mechanisms:

1. **Entropy-based trigger**: System entropy >0.7 sustained for 5 minutes
2. **GPU utilization trigger**: GPU usage >30% (NOT 80% - per Constitution)
3. **Manual trigger**: Immediate dream cycle request
4. **Integration with DreamScheduler**: Extend scheduler with TriggerManager

**Critical Constitution Reference**: `docs2/constitution.yaml` lines 255-256
```yaml
dream:
  trigger: { activity: "<0.15", idle: "10min", entropy: ">0.7 for 5min", gpu: "<80%" }
```

**IMPORTANT CORRECTION**: The constitution says `gpu: "<80%"` in the trigger section, but line 274 says `gpu: "<30%"` for dream constraints. The GPU trigger should fire when approaching the 30% limit to allow consolidation BEFORE hitting the budget. This task uses 30% threshold.

---

## 2. Current Codebase State (VERIFIED 2026-01-11)

### 2.1 EXISTING Types in `crates/context-graph-core/src/dream/types.rs`

These types ALREADY EXIST and MUST BE USED. DO NOT RECREATE:

```rust
// EntropyWindow (lines 309-434) - Tracks entropy over time window
pub struct EntropyWindow {
    samples: VecDeque<(Instant, f32)>,
    pub window_duration: Duration,  // Default: 5 minutes
    pub threshold: f32,             // Default: 0.7
    high_entropy_since: Option<Instant>,
}
impl EntropyWindow {
    pub fn new() -> Self;
    pub fn with_params(window_duration: Duration, threshold: f32) -> Self;
    pub fn push(&mut self, entropy: f32);
    pub fn should_trigger(&self) -> bool;  // True if entropy > threshold for window_duration
    pub fn average(&self) -> f32;
    pub fn minimum(&self) -> f32;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn clear(&mut self);
}

// GpuTriggerState (lines 442-544) - Monitors GPU usage
pub struct GpuTriggerState {
    pub current_usage: f32,
    pub threshold: f32,           // Default: 0.30 (30%, NOT 80%)
    samples: VecDeque<f32>,
    max_samples: usize,
    pub triggered: bool,
}
impl GpuTriggerState {
    pub fn new() -> Self;           // threshold=0.30 per Constitution
    pub fn with_threshold(threshold: f32) -> Self;  // PANICS if threshold > 0.30
    pub fn update(&mut self, usage: f32);
    pub fn should_trigger(&self) -> bool;
    pub fn mark_triggered(&mut self);
    pub fn reset(&mut self);
    pub fn average(&self) -> f32;
}

// ExtendedTriggerReason (lines 546-579) - Enum for trigger types
pub enum ExtendedTriggerReason {
    IdleTimeout,      // Activity below 0.15 for 10 min
    HighEntropy,      // Entropy above 0.7 for 5 min
    GpuOverload,      // GPU usage approaching 30%
    MemoryPressure,   // Memory needs consolidation
    Manual,           // User-triggered
    Scheduled,        // Scheduled dream time
}
```

### 2.2 EXISTING DreamScheduler in `crates/context-graph-core/src/dream/scheduler.rs`

The scheduler handles idle timeout trigger but DOES NOT integrate entropy or GPU triggers:

```rust
pub struct DreamScheduler {
    activity_threshold: f32,      // 0.15
    idle_duration_trigger: Duration,  // 10 minutes
    last_activity: Option<Instant>,
    activity_samples: VecDeque<ActivitySample>,
    last_dream_completed: Option<Instant>,
    low_activity_start: Option<Instant>,
    average_activity: f32,
}
impl DreamScheduler {
    pub fn new() -> Self;
    pub fn with_thresholds(activity_threshold: f32, idle_duration: Duration) -> Self;
    pub fn update_activity(&mut self, activity: f32);
    pub fn should_trigger_dream(&self) -> bool;
    pub fn check_trigger(&self) -> TriggerDecision;
    pub fn record_dream_completion(&mut self);
    // ... other methods
}
```

### 2.3 EXISTING Module Structure in `crates/context-graph-core/src/dream/mod.rs`

```rust
pub mod amortized;
pub mod controller;
pub mod hebbian;
pub mod hyperbolic_walk;
pub mod nrem;
pub mod poincare_walk;
pub mod rem;
pub mod scheduler;
pub mod types;

// Re-exports include EntropyWindow, GpuTriggerState, ExtendedTriggerReason
pub use types::{
    EntropyWindow,
    ExtendedTriggerReason,
    GpuTriggerState,
    HebbianConfig,
    HyperbolicWalkConfig,
    NodeActivation,
    WalkStep,
};
```

### 2.4 Files That DO NOT EXIST (Must Create)

- `crates/context-graph-core/src/dream/triggers.rs` - **THIS IS THE MAIN DELIVERABLE**

---

## 3. Files to Create/Modify

### 3.1 CREATE: `crates/context-graph-core/src/dream/triggers.rs`

**Purpose**: Unified TriggerManager that combines entropy, GPU, and manual triggers.

```rust
//! Dream Trigger Implementations
//!
//! Implements trigger mechanisms beyond idle timeout:
//! - High entropy trigger (>0.7 for 5 minutes) - Constitution Section dream.trigger
//! - GPU overload trigger (approaching 30% budget) - Constitution Section dream.constraints
//! - Manual trigger (highest priority)
//!
//! Constitution Reference: docs2/constitution.yaml lines 255-256, 274
//! - entropy: ">0.7 for 5min" (line 255)
//! - gpu: "<30%" during dream (line 274)

use std::time::{Duration, Instant};

use tracing::{debug, info};

use super::types::{EntropyWindow, ExtendedTriggerReason, GpuTriggerState};

/// Unified trigger manager for dream cycles.
///
/// Combines all trigger mechanisms into a single interface:
/// - High entropy (>0.7 sustained for 5 minutes)
/// - GPU overload (approaching 30% usage)
/// - Manual trigger
///
/// # Constitution Compliance
///
/// - Entropy threshold: 0.7 (Constitution dream.trigger)
/// - Entropy window: 5 minutes (Constitution dream.trigger)
/// - GPU threshold: 0.30 (Constitution dream.constraints.gpu)
/// - Cooldown: 30 minutes (Constitution idle timeout)
#[derive(Debug)]
pub struct TriggerManager {
    /// Entropy tracking window
    entropy_window: EntropyWindow,

    /// GPU utilization state
    gpu_state: GpuTriggerState,

    /// Whether manual trigger was requested
    manual_trigger: bool,

    /// Last trigger reason (for reporting)
    last_trigger_reason: Option<ExtendedTriggerReason>,

    /// Cooldown after trigger (to prevent rapid re-triggering)
    trigger_cooldown: Duration,

    /// Last trigger time
    last_trigger_time: Option<Instant>,

    /// Whether triggers are enabled
    enabled: bool,
}

impl TriggerManager {
    /// Create a new trigger manager with constitution defaults.
    ///
    /// # Constitution Values Applied
    /// - Entropy threshold: 0.7
    /// - Entropy window: 5 minutes
    /// - GPU threshold: 0.30 (30%)
    /// - Cooldown: 30 minutes
    pub fn new() -> Self {
        Self {
            entropy_window: EntropyWindow::new(), // Uses Constitution defaults
            gpu_state: GpuTriggerState::new(),    // Uses Constitution defaults
            manual_trigger: false,
            last_trigger_reason: None,
            trigger_cooldown: Duration::from_secs(1800), // 30 minutes
            last_trigger_time: None,
            enabled: true,
        }
    }

    /// Create with custom cooldown (for testing with REAL time).
    ///
    /// # Arguments
    /// * `cooldown` - Duration before trigger can fire again
    ///
    /// # Note
    /// Tests MUST use real durations, not mocked time.
    pub fn with_cooldown(cooldown: Duration) -> Self {
        let mut manager = Self::new();
        manager.trigger_cooldown = cooldown;
        manager
    }

    /// Update entropy reading.
    ///
    /// Called periodically (e.g., every second) with system entropy value.
    /// High entropy (>0.7) indicates system stress/confusion.
    ///
    /// # Arguments
    /// * `entropy` - Current system entropy [0.0, 1.0]
    ///
    /// # Constitution Reference
    /// Trigger fires when entropy > 0.7 for 5 minutes continuously.
    pub fn update_entropy(&mut self, entropy: f32) {
        if !self.enabled {
            return;
        }

        self.entropy_window.push(entropy);

        if self.entropy_window.should_trigger() {
            debug!(
                "Entropy trigger condition met: avg={:.3}, threshold=0.7",
                self.entropy_window.average()
            );
        }
    }

    /// Update GPU utilization reading.
    ///
    /// Called periodically with GPU usage percentage.
    /// High GPU usage approaching budget indicates need for consolidation.
    ///
    /// # Arguments
    /// * `usage` - Current GPU usage [0.0, 1.0]
    ///
    /// # Constitution Reference
    /// GPU budget during dream is <30%. Trigger fires when approaching this limit.
    pub fn update_gpu_usage(&mut self, usage: f32) {
        if !self.enabled {
            return;
        }

        self.gpu_state.update(usage);

        if self.gpu_state.should_trigger() {
            debug!("GPU trigger condition met: usage={:.1}%", usage * 100.0);
        }
    }

    /// Request a manual dream trigger.
    ///
    /// Manual triggers have highest priority and bypass cooldown.
    pub fn request_manual_trigger(&mut self) {
        info!("Manual dream trigger requested");
        self.manual_trigger = true;
    }

    /// Clear manual trigger flag.
    pub fn clear_manual_trigger(&mut self) {
        self.manual_trigger = false;
    }

    /// Check if any trigger condition is met.
    ///
    /// Returns the trigger reason if triggered, None otherwise.
    /// Priority order: Manual > GPU > Entropy
    ///
    /// # Returns
    /// * `Some(ExtendedTriggerReason)` - If trigger condition met
    /// * `None` - If no trigger condition met or in cooldown
    pub fn check_triggers(&self) -> Option<ExtendedTriggerReason> {
        if !self.enabled {
            return None;
        }

        // Check cooldown (manual trigger bypasses cooldown)
        if !self.manual_trigger {
            if let Some(last_time) = self.last_trigger_time {
                if last_time.elapsed() < self.trigger_cooldown {
                    return None;
                }
            }
        }

        // Check manual trigger (highest priority)
        if self.manual_trigger {
            return Some(ExtendedTriggerReason::Manual);
        }

        // Check GPU trigger (higher priority than entropy)
        if self.gpu_state.should_trigger() {
            return Some(ExtendedTriggerReason::GpuOverload);
        }

        // Check entropy trigger
        if self.entropy_window.should_trigger() {
            return Some(ExtendedTriggerReason::HighEntropy);
        }

        None
    }

    /// Check if dream should be triggered (simple boolean).
    #[inline]
    pub fn should_trigger(&self) -> bool {
        self.check_triggers().is_some()
    }

    /// Mark that a trigger fired (starts cooldown).
    ///
    /// Call this AFTER starting a dream cycle.
    pub fn mark_triggered(&mut self, reason: ExtendedTriggerReason) {
        info!("Dream triggered: {:?}", reason);

        self.last_trigger_reason = Some(reason);
        self.last_trigger_time = Some(Instant::now());

        // Reset states
        self.manual_trigger = false;
        self.gpu_state.mark_triggered();
        self.entropy_window.clear();
    }

    /// Reset after dream completion.
    ///
    /// Call this AFTER dream cycle completes.
    pub fn reset(&mut self) {
        debug!("Trigger manager reset");

        self.gpu_state.reset();
        self.entropy_window.clear();
        self.manual_trigger = false;
    }

    /// Get time remaining in cooldown, if any.
    pub fn cooldown_remaining(&self) -> Option<Duration> {
        self.last_trigger_time.and_then(|last| {
            let elapsed = last.elapsed();
            if elapsed < self.trigger_cooldown {
                Some(self.trigger_cooldown - elapsed)
            } else {
                None
            }
        })
    }

    /// Get current entropy average.
    #[inline]
    pub fn current_entropy(&self) -> f32 {
        self.entropy_window.average()
    }

    /// Get current GPU usage.
    #[inline]
    pub fn current_gpu_usage(&self) -> f32 {
        self.gpu_state.current_usage
    }

    /// Get last trigger reason.
    pub fn last_trigger_reason(&self) -> Option<ExtendedTriggerReason> {
        self.last_trigger_reason
    }

    /// Enable or disable triggers.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            info!("Dream triggers disabled");
        } else {
            info!("Dream triggers enabled");
        }
    }

    /// Check if triggers are enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Access entropy window for inspection/testing.
    pub fn entropy_window(&self) -> &EntropyWindow {
        &self.entropy_window
    }

    /// Access GPU state for inspection/testing.
    pub fn gpu_state(&self) -> &GpuTriggerState {
        &self.gpu_state
    }
}

impl Default for TriggerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU utilization monitor stub.
///
/// Provides a placeholder for actual GPU monitoring.
/// Real implementation would use NVML (NVIDIA) or ROCm (AMD).
///
/// # Note
/// This is a STUB. Production requires actual GPU monitoring integration.
#[derive(Debug, Clone)]
pub struct GpuMonitor {
    /// Simulated GPU usage (for testing)
    simulated_usage: f32,

    /// Whether to use simulated values
    use_simulated: bool,
}

impl GpuMonitor {
    /// Create a new GPU monitor.
    pub fn new() -> Self {
        Self {
            simulated_usage: 0.0,
            use_simulated: true, // Default to simulated until real impl
        }
    }

    /// Get current GPU utilization.
    ///
    /// Returns value in [0.0, 1.0].
    pub fn get_usage(&self) -> f32 {
        if self.use_simulated {
            self.simulated_usage
        } else {
            // TODO(FUTURE): Implement real GPU monitoring via NVML
            // For now, return 0.0 (no GPU usage)
            0.0
        }
    }

    /// Set simulated GPU usage (for testing).
    pub fn set_simulated_usage(&mut self, usage: f32) {
        self.simulated_usage = usage.clamp(0.0, 1.0);
    }

    /// Check if GPU is available.
    pub fn is_available(&self) -> bool {
        // TODO(FUTURE): Check for actual GPU
        false
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy calculator from system state.
///
/// Computes system entropy based on query rate variance.
/// High variance = high entropy = system confusion.
///
/// # Algorithm
/// Uses coefficient of variation (CV = std/mean) of query inter-arrival times.
/// CV > 1 indicates high variability (chaotic) = high entropy.
/// CV < 1 indicates regularity (predictable) = low entropy.
#[derive(Debug, Clone)]
pub struct EntropyCalculator {
    /// Recent query timestamps for rate calculation
    query_times: Vec<Instant>,

    /// Maximum queries to track
    max_queries: usize,

    /// Time window for entropy calculation
    window: Duration,
}

impl EntropyCalculator {
    /// Create a new entropy calculator.
    pub fn new() -> Self {
        Self {
            query_times: Vec::with_capacity(100),
            max_queries: 100,
            window: Duration::from_secs(60), // 1 minute window
        }
    }

    /// Record a query event.
    pub fn record_query(&mut self) {
        let now = Instant::now();

        self.query_times.push(now);

        // Trim old queries outside window
        self.query_times
            .retain(|&t| now.duration_since(t) < self.window);

        // Cap size
        while self.query_times.len() > self.max_queries {
            self.query_times.remove(0);
        }
    }

    /// Calculate current entropy based on query patterns.
    ///
    /// Returns value in [0.0, 1.0] where:
    /// - 0.0 = no activity or regular pattern (low entropy)
    /// - 1.0 = high chaotic activity (high entropy)
    pub fn calculate(&self) -> f32 {
        if self.query_times.len() < 2 {
            return 0.0;
        }

        // Calculate inter-arrival times
        let mut intervals: Vec<f32> = Vec::new();
        for i in 1..self.query_times.len() {
            let interval = self.query_times[i]
                .duration_since(self.query_times[i - 1])
                .as_secs_f32();
            intervals.push(interval);
        }

        if intervals.is_empty() {
            return 0.0;
        }

        // Calculate coefficient of variation (CV = std / mean)
        let mean: f32 = intervals.iter().sum::<f32>() / intervals.len() as f32;
        if mean < 1e-6 {
            return 1.0; // Extremely rapid queries = high entropy
        }

        let variance: f32 = intervals
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;
        let std = variance.sqrt();
        let cv = std / mean;

        // Normalize CV to [0, 1]
        // CV > 1 indicates high variability (high entropy)
        // CV < 1 indicates regularity (low entropy)
        (cv / 2.0).min(1.0)
    }

    /// Clear query history.
    pub fn clear(&mut self) {
        self.query_times.clear();
    }

    /// Get the number of queries in window.
    pub fn query_count(&self) -> usize {
        self.query_times.len()
    }
}

impl Default for EntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS - NO MOCK DATA, REAL TIME ONLY
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // ============ Constitution Compliance Tests ============

    #[test]
    fn test_trigger_manager_constitution_defaults() {
        let manager = TriggerManager::new();

        assert!(manager.is_enabled());
        assert!(!manager.should_trigger());

        // Verify entropy window uses Constitution defaults
        assert_eq!(manager.entropy_window.threshold, 0.7,
            "Entropy threshold must be 0.7 per Constitution");
        assert_eq!(manager.entropy_window.window_duration, Duration::from_secs(300),
            "Entropy window must be 5 minutes per Constitution");

        // Verify GPU state uses Constitution defaults
        assert_eq!(manager.gpu_state.threshold, 0.30,
            "GPU threshold must be 0.30 (30%) per Constitution, NOT 80%");
    }

    // ============ Manual Trigger Tests ============

    #[test]
    fn test_trigger_manager_manual_highest_priority() {
        let mut manager = TriggerManager::new();

        assert!(!manager.should_trigger());

        manager.request_manual_trigger();

        assert!(manager.should_trigger());
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::Manual)
        );
    }

    #[test]
    fn test_trigger_manager_manual_bypasses_cooldown() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_secs(3600)); // 1 hour

        // Trigger and start cooldown
        manager.request_manual_trigger();
        manager.mark_triggered(ExtendedTriggerReason::Manual);

        // Another manual trigger should bypass cooldown
        manager.request_manual_trigger();
        assert!(manager.should_trigger(), "Manual trigger should bypass cooldown");
    }

    // ============ GPU Trigger Tests ============

    #[test]
    fn test_trigger_manager_gpu_threshold_30_percent() {
        let mut manager = TriggerManager::new();

        // Below threshold
        manager.update_gpu_usage(0.25);
        assert!(!manager.should_trigger(), "25% < 30% should not trigger");

        // Above threshold
        manager.update_gpu_usage(0.35);
        assert!(manager.should_trigger(), "35% > 30% should trigger");
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::GpuOverload)
        );
    }

    #[test]
    fn test_trigger_manager_gpu_uses_smoothed_average() {
        let mut manager = TriggerManager::new();

        // Push multiple samples - average should be used
        manager.update_gpu_usage(0.20);
        manager.update_gpu_usage(0.22);
        manager.update_gpu_usage(0.25);

        assert!(!manager.should_trigger(), "Average ~22% should not trigger");
    }

    // ============ Entropy Trigger Tests ============

    #[test]
    fn test_trigger_manager_entropy_requires_sustained_high() {
        // Use short window for testing with REAL time
        let mut manager = TriggerManager::new();
        manager.entropy_window = EntropyWindow::with_params(
            Duration::from_millis(50),
            0.7,
        );
        manager.trigger_cooldown = Duration::from_millis(1);

        // Push high entropy - starts tracking
        manager.update_entropy(0.8);
        assert!(!manager.should_trigger(), "Should not trigger immediately");

        // Wait for window duration plus margin
        thread::sleep(Duration::from_millis(60));
        manager.update_entropy(0.85);

        assert!(manager.should_trigger(), "Should trigger after sustained high entropy");
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::HighEntropy)
        );
    }

    #[test]
    fn test_trigger_manager_entropy_resets_on_low() {
        let mut manager = TriggerManager::new();
        manager.entropy_window = EntropyWindow::with_params(
            Duration::from_millis(50),
            0.7,
        );

        // Start high
        manager.update_entropy(0.9);
        thread::sleep(Duration::from_millis(30));

        // Drop below threshold - resets tracking
        manager.update_entropy(0.5);

        // Wait and check - should NOT trigger because tracking reset
        thread::sleep(Duration::from_millis(30));
        manager.update_entropy(0.5);

        assert!(!manager.should_trigger(), "Low entropy should reset tracking");
    }

    // ============ Cooldown Tests ============

    #[test]
    fn test_trigger_manager_cooldown_prevents_rapid_retrigger() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(100));

        // Trigger with GPU
        manager.update_gpu_usage(0.35);
        assert!(manager.should_trigger());

        manager.mark_triggered(ExtendedTriggerReason::GpuOverload);

        // Try to trigger again immediately
        manager.update_gpu_usage(0.40);
        assert!(!manager.should_trigger(), "Cooldown should prevent re-trigger");

        // Wait for cooldown
        thread::sleep(Duration::from_millis(150));

        manager.update_gpu_usage(0.40);
        assert!(manager.should_trigger(), "Should trigger after cooldown expires");
    }

    #[test]
    fn test_trigger_manager_cooldown_remaining() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(100));

        assert!(manager.cooldown_remaining().is_none(), "No cooldown initially");

        manager.request_manual_trigger();
        manager.mark_triggered(ExtendedTriggerReason::Manual);

        let remaining = manager.cooldown_remaining();
        assert!(remaining.is_some(), "Should have cooldown after trigger");
        assert!(remaining.unwrap() <= Duration::from_millis(100));
    }

    // ============ Disabled Trigger Tests ============

    #[test]
    fn test_trigger_manager_disabled_blocks_all() {
        let mut manager = TriggerManager::new();

        manager.set_enabled(false);

        // None of these should trigger
        manager.request_manual_trigger();
        manager.update_gpu_usage(0.95);
        manager.update_entropy(0.99);

        assert!(!manager.should_trigger(), "Disabled manager should not trigger");
    }

    // ============ Reset Tests ============

    #[test]
    fn test_trigger_manager_reset_clears_state() {
        let mut manager = TriggerManager::new();

        manager.update_gpu_usage(0.35);
        manager.update_entropy(0.9);
        manager.mark_triggered(ExtendedTriggerReason::GpuOverload);

        manager.reset();

        // GPU state should be reset
        assert_eq!(manager.current_gpu_usage(), 0.0);
        // Entropy window should be cleared
        assert!(manager.entropy_window.is_empty());
    }

    // ============ GpuMonitor Tests ============

    #[test]
    fn test_gpu_monitor_simulated() {
        let mut monitor = GpuMonitor::new();

        assert_eq!(monitor.get_usage(), 0.0);

        monitor.set_simulated_usage(0.75);
        assert_eq!(monitor.get_usage(), 0.75);

        // Test clamping
        monitor.set_simulated_usage(1.5);
        assert_eq!(monitor.get_usage(), 1.0);

        monitor.set_simulated_usage(-0.5);
        assert_eq!(monitor.get_usage(), 0.0);
    }

    // ============ EntropyCalculator Tests ============

    #[test]
    fn test_entropy_calculator_empty_returns_zero() {
        let calc = EntropyCalculator::new();
        assert_eq!(calc.calculate(), 0.0);
    }

    #[test]
    fn test_entropy_calculator_single_query_returns_zero() {
        let mut calc = EntropyCalculator::new();
        calc.record_query();
        assert_eq!(calc.calculate(), 0.0);
    }

    #[test]
    fn test_entropy_calculator_regular_queries_low_entropy() {
        let mut calc = EntropyCalculator::new();

        // Simulate regular queries at fixed intervals
        for _ in 0..5 {
            calc.record_query();
            thread::sleep(Duration::from_millis(10));
        }

        let entropy = calc.calculate();
        // Regular intervals = low entropy
        assert!(entropy < 0.5, "Regular queries should have low entropy: {}", entropy);
    }

    #[test]
    fn test_entropy_calculator_irregular_queries_high_entropy() {
        let mut calc = EntropyCalculator::new();

        // Simulate irregular queries
        calc.record_query();
        thread::sleep(Duration::from_millis(5));
        calc.record_query();
        thread::sleep(Duration::from_millis(50));
        calc.record_query();
        thread::sleep(Duration::from_millis(10));
        calc.record_query();
        thread::sleep(Duration::from_millis(100));
        calc.record_query();

        let entropy = calc.calculate();
        // Irregular intervals = higher entropy
        assert!(entropy > 0.3, "Irregular queries should have higher entropy: {}", entropy);
    }
}
```

### 3.2 MODIFY: `crates/context-graph-core/src/dream/mod.rs`

Add triggers module and re-exports after line 54 (after `scheduler`):

```rust
// ADD after line 54:
pub mod triggers;

// ADD to re-exports section (around line 91, after scheduler export):
pub use triggers::{
    TriggerManager,
    GpuMonitor,
    EntropyCalculator,
};
```

### 3.3 MODIFY: `crates/context-graph-core/src/dream/scheduler.rs` (OPTIONAL ENHANCEMENT)

Add TriggerManager integration. This is optional but recommended for unified trigger checking:

```rust
// Add import at top:
use super::triggers::TriggerManager;
use super::types::ExtendedTriggerReason;

// Add field to DreamScheduler struct:
    /// Extended trigger manager for entropy/GPU triggers
    trigger_manager: TriggerManager,

// Update new() to initialize trigger_manager:
    trigger_manager: TriggerManager::new(),

// Add these methods to DreamScheduler impl block:

    /// Update entropy for extended trigger detection.
    ///
    /// # Arguments
    /// * `entropy` - Current system entropy [0.0, 1.0]
    pub fn update_entropy(&mut self, entropy: f32) {
        self.trigger_manager.update_entropy(entropy);
    }

    /// Update GPU usage for extended trigger detection.
    ///
    /// # Arguments
    /// * `usage` - Current GPU usage [0.0, 1.0]
    pub fn update_gpu_usage(&mut self, usage: f32) {
        self.trigger_manager.update_gpu_usage(usage);
    }

    /// Check extended triggers (entropy, GPU, manual).
    ///
    /// Returns the trigger reason if any extended trigger fired.
    pub fn check_extended_triggers(&self) -> Option<ExtendedTriggerReason> {
        self.trigger_manager.check_triggers()
    }

    /// Request manual dream trigger.
    pub fn request_manual_trigger(&mut self) {
        self.trigger_manager.request_manual_trigger();
    }

    /// Access trigger manager for inspection.
    pub fn trigger_manager(&self) -> &TriggerManager {
        &self.trigger_manager
    }

    /// Access trigger manager mutably.
    pub fn trigger_manager_mut(&mut self) -> &mut TriggerManager {
        &mut self.trigger_manager
    }
```

---

## 4. Definition of Done

### 4.1 Type Signatures (Exact)

```rust
pub struct TriggerManager { /* internal */ }
impl TriggerManager {
    pub fn new() -> Self;
    pub fn with_cooldown(cooldown: Duration) -> Self;
    pub fn update_entropy(&mut self, entropy: f32);
    pub fn update_gpu_usage(&mut self, usage: f32);
    pub fn request_manual_trigger(&mut self);
    pub fn clear_manual_trigger(&mut self);
    pub fn check_triggers(&self) -> Option<ExtendedTriggerReason>;
    pub fn should_trigger(&self) -> bool;
    pub fn mark_triggered(&mut self, reason: ExtendedTriggerReason);
    pub fn reset(&mut self);
    pub fn cooldown_remaining(&self) -> Option<Duration>;
    pub fn current_entropy(&self) -> f32;
    pub fn current_gpu_usage(&self) -> f32;
    pub fn last_trigger_reason(&self) -> Option<ExtendedTriggerReason>;
    pub fn set_enabled(&mut self, enabled: bool);
    pub fn is_enabled(&self) -> bool;
    pub fn entropy_window(&self) -> &EntropyWindow;
    pub fn gpu_state(&self) -> &GpuTriggerState;
}

pub struct GpuMonitor { /* internal */ }
impl GpuMonitor {
    pub fn new() -> Self;
    pub fn get_usage(&self) -> f32;
    pub fn set_simulated_usage(&mut self, usage: f32);
    pub fn is_available(&self) -> bool;
}

pub struct EntropyCalculator { /* internal */ }
impl EntropyCalculator {
    pub fn new() -> Self;
    pub fn record_query(&mut self);
    pub fn calculate(&self) -> f32;
    pub fn clear(&mut self);
    pub fn query_count(&self) -> usize;
}
```

### 4.2 Validation Commands

```bash
# Build must succeed
cargo build -p context-graph-core

# All tests must pass
cargo test -p context-graph-core dream::triggers -- --nocapture

# No clippy warnings
cargo clippy -p context-graph-core -- -D warnings

# Verify no regressions
cargo test -p context-graph-core dream:: -- --nocapture
```

### 4.3 Test Coverage Requirements

| Test | What It Verifies |
|------|------------------|
| `test_trigger_manager_constitution_defaults` | Entropy=0.7, Window=5min, GPU=30% |
| `test_trigger_manager_manual_highest_priority` | Manual trigger fires immediately |
| `test_trigger_manager_manual_bypasses_cooldown` | Manual ignores cooldown |
| `test_trigger_manager_gpu_threshold_30_percent` | GPU triggers at 30%, NOT 80% |
| `test_trigger_manager_gpu_uses_smoothed_average` | GPU uses averaged samples |
| `test_trigger_manager_entropy_requires_sustained_high` | Entropy needs 5min sustained |
| `test_trigger_manager_entropy_resets_on_low` | Drop below 0.7 resets tracking |
| `test_trigger_manager_cooldown_prevents_rapid_retrigger` | Cooldown enforced |
| `test_trigger_manager_cooldown_remaining` | Can query cooldown time |
| `test_trigger_manager_disabled_blocks_all` | Disabled blocks all triggers |
| `test_trigger_manager_reset_clears_state` | Reset clears GPU/entropy |
| `test_gpu_monitor_simulated` | GpuMonitor simulation works |
| `test_entropy_calculator_empty_returns_zero` | Empty = 0 entropy |
| `test_entropy_calculator_single_query_returns_zero` | Single query = 0 |
| `test_entropy_calculator_regular_queries_low_entropy` | Regular = low entropy |
| `test_entropy_calculator_irregular_queries_high_entropy` | Irregular = high entropy |

---

## 5. Full State Verification (FSV)

### 5.1 Source of Truth

| Component | Location | What to Verify |
|-----------|----------|----------------|
| TriggerManager state | In-memory struct | `should_trigger()` returns expected value |
| EntropyWindow tracking | `high_entropy_since` field | Timestamp set when entropy > 0.7 |
| GpuTriggerState | `triggered` flag | Set to true after `mark_triggered()` |
| Cooldown | `last_trigger_time` field | Time recorded after `mark_triggered()` |

### 5.2 Execute & Inspect Protocol

After implementing, run this verification sequence:

```rust
// Create manager
let mut manager = TriggerManager::new();
println!("BEFORE: should_trigger={}", manager.should_trigger()); // false

// Simulate high GPU
manager.update_gpu_usage(0.35);
println!("AFTER GPU 35%: should_trigger={}", manager.should_trigger()); // true
println!("AFTER GPU 35%: check_triggers={:?}", manager.check_triggers()); // GpuOverload

// Mark triggered
manager.mark_triggered(ExtendedTriggerReason::GpuOverload);
println!("AFTER MARK: gpu_state.triggered={}", manager.gpu_state().triggered); // true
println!("AFTER MARK: cooldown_remaining={:?}", manager.cooldown_remaining()); // Some(...)

// Try again (should fail due to cooldown)
manager.update_gpu_usage(0.40);
println!("IN COOLDOWN: should_trigger={}", manager.should_trigger()); // false
```

### 5.3 Boundary & Edge Case Audit

Execute these 3 edge cases and log before/after state:

**Edge Case 1: Entropy exactly at threshold (0.7)**
```rust
let mut manager = TriggerManager::new();
manager.entropy_window = EntropyWindow::with_params(Duration::from_millis(10), 0.7);

// Entropy AT threshold should NOT trigger (must be ABOVE)
manager.update_entropy(0.7);
thread::sleep(Duration::from_millis(20));
manager.update_entropy(0.7);

println!("Entropy=0.7: should_trigger={}", manager.should_trigger()); // false (need >0.7)
```

**Edge Case 2: GPU exactly at threshold (0.30)**
```rust
let mut manager = TriggerManager::new();

manager.update_gpu_usage(0.30);
println!("GPU=0.30: should_trigger={}", manager.should_trigger()); // true (>=0.30)

manager.reset();
manager.update_gpu_usage(0.29);
println!("GPU=0.29: should_trigger={}", manager.should_trigger()); // false (<0.30)
```

**Edge Case 3: Disabled manager ignores all inputs**
```rust
let mut manager = TriggerManager::new();
manager.set_enabled(false);

manager.request_manual_trigger();
manager.update_gpu_usage(0.99);
manager.update_entropy(0.99);

println!("Disabled: should_trigger={}", manager.should_trigger()); // false
println!("Disabled: manual_trigger_internal={}", manager.manual_trigger); // false (not set)
```

### 5.4 Evidence of Success

After implementation, all tests pass and this output is logged:

```
test dream::triggers::tests::test_trigger_manager_constitution_defaults ... ok
test dream::triggers::tests::test_trigger_manager_manual_highest_priority ... ok
test dream::triggers::tests::test_trigger_manager_manual_bypasses_cooldown ... ok
test dream::triggers::tests::test_trigger_manager_gpu_threshold_30_percent ... ok
test dream::triggers::tests::test_trigger_manager_gpu_uses_smoothed_average ... ok
test dream::triggers::tests::test_trigger_manager_entropy_requires_sustained_high ... ok
test dream::triggers::tests::test_trigger_manager_entropy_resets_on_low ... ok
test dream::triggers::tests::test_trigger_manager_cooldown_prevents_rapid_retrigger ... ok
test dream::triggers::tests::test_trigger_manager_cooldown_remaining ... ok
test dream::triggers::tests::test_trigger_manager_disabled_blocks_all ... ok
test dream::triggers::tests::test_trigger_manager_reset_clears_state ... ok
test dream::triggers::tests::test_gpu_monitor_simulated ... ok
test dream::triggers::tests::test_entropy_calculator_empty_returns_zero ... ok
test dream::triggers::tests::test_entropy_calculator_single_query_returns_zero ... ok
test dream::triggers::tests::test_entropy_calculator_regular_queries_low_entropy ... ok
test dream::triggers::tests::test_entropy_calculator_irregular_queries_high_entropy ... ok

test result: ok. 16 passed; 0 failed; 0 ignored
```

---

## 6. Manual Testing with Synthetic Data

### 6.1 Test Case 1: Full Trigger Lifecycle

**Input**:
- Initial state: No triggers
- Update GPU to 35%
- Mark triggered
- Wait for cooldown
- Verify re-triggerable

**Expected State Transitions**:
```
Initial: should_trigger=false, gpu_usage=0.0, cooldown=None
After GPU 35%: should_trigger=true, gpu_usage=0.35, check_triggers=GpuOverload
After mark_triggered: should_trigger=false, last_trigger_reason=GpuOverload, cooldown=Some(...)
After cooldown expires: should_trigger=true (with new GPU input)
```

### 6.2 Test Case 2: Entropy Trigger with Tracking Reset

**Input**:
- Push entropy 0.8 (starts tracking)
- Wait 30ms
- Push entropy 0.5 (below threshold - resets)
- Push entropy 0.9 (restarts tracking)
- Wait 30ms
- Verify NOT triggered (only 30ms since restart, not 50ms)

**Expected State Transitions**:
```
Initial: high_entropy_since=None
After 0.8: high_entropy_since=Some(t0)
After 0.5: high_entropy_since=None (reset!)
After 0.9: high_entropy_since=Some(t1) where t1 > t0
After 30ms: should_trigger=false (only 30ms elapsed since restart)
```

### 6.3 Test Case 3: Priority Order (Manual > GPU > Entropy)

**Input**:
- Set GPU to 35% (trigger condition met)
- Set entropy window to trigger (high sustained)
- Request manual trigger
- Verify manual returned first

**Expected**:
```rust
let mut manager = TriggerManager::new();
// Setup GPU trigger
manager.update_gpu_usage(0.35);
// Setup entropy trigger (use short window for test)
manager.entropy_window = EntropyWindow::with_params(Duration::from_millis(10), 0.7);
manager.update_entropy(0.9);
thread::sleep(Duration::from_millis(20));
manager.update_entropy(0.9);
// Request manual
manager.request_manual_trigger();

// Manual should be returned (highest priority)
assert_eq!(manager.check_triggers(), Some(ExtendedTriggerReason::Manual));
```

---

## 7. Error Handling Philosophy

**FAIL FAST - NO GRACEFUL DEGRADATION**

1. **Invalid inputs**: Types.rs already validates EntropyWindow and GpuTriggerState
   - `GpuTriggerState::with_threshold(0.80)` PANICS (>0.30 violates Constitution)
   - `EntropyWindow::with_params(_, threshold)` requires threshold in [0.0, 1.0]

2. **Constitution violations**: Hard-coded limits, not configurable
   - Query limit = 100 (cannot be changed)
   - GPU threshold = 0.30 (GpuTriggerState::new() enforces this)
   - Entropy threshold = 0.7 (EntropyWindow::new() enforces this)

3. **State corruption**: Reset on completion
   - After `mark_triggered()`, states reset appropriately
   - After `reset()`, all tracking cleared

---

## 8. Constitution Compliance Matrix

| Parameter | Constitution Location | Value | Enforcement |
|-----------|----------------------|-------|-------------|
| Entropy threshold | Line 255 | >0.7 | `EntropyWindow::new().threshold = 0.7` |
| Entropy window | Line 255 | 5 min | `EntropyWindow::new().window_duration = 300s` |
| GPU budget | Line 274 | <30% | `GpuTriggerState::new().threshold = 0.30` |
| Idle timeout | Line 255 | 10 min | `DreamScheduler` (existing) |
| Activity threshold | Line 255 | <0.15 | `DreamScheduler` (existing) |

---

## 9. Dependencies

### 9.1 Crate Dependencies (Already in Cargo.toml)

```toml
tracing = "0.1"
```

### 9.2 Internal Dependencies

- `super::types::{EntropyWindow, ExtendedTriggerReason, GpuTriggerState}` - MUST USE existing types
- `super::scheduler::DreamScheduler` - OPTIONAL integration point

---

## 10. Effort Breakdown

| Phase | Duration |
|-------|----------|
| Create triggers.rs with TriggerManager | 45 min |
| Implement GpuMonitor stub | 15 min |
| Implement EntropyCalculator | 30 min |
| Write unit tests (16 tests) | 45 min |
| Update mod.rs exports | 5 min |
| Optional: Integrate with DreamScheduler | 20 min |
| Manual verification & FSV | 20 min |
| **Total** | **3 hours** |

---

## 11. Related Tasks

| Task | Relationship |
|------|-------------|
| TASK-DREAM-P0-001 | COMPLETED - Provides EntropyWindow, GpuTriggerState, ExtendedTriggerReason |
| TASK-DREAM-P0-004 | COMPLETED - HyperbolicExplorer (not directly related but same module) |
| TASK-DREAM-P0-006 | BLOCKED BY THIS - Wake Controller needs TriggerManager |
| TASK-GWT-P1-002 | COMPLETED - Workspace events (potential entropy source) |

---

## 12. Anti-Patterns to Avoid

| Anti-Pattern | What NOT To Do | Why |
|--------------|----------------|-----|
| AP-42 | Skip wiring mental_checks to TriggerManager | Constitution requires entropy trigger integration |
| Recreating types | Create new EntropyWindow/GpuTriggerState | Use existing types from types.rs |
| Mock time | Use fake clocks in tests | Constitution requires real time testing |
| 80% GPU threshold | Use 0.80 for GPU trigger | Constitution says <30% (0.30) |
| Silent failures | Return None on errors | Panic with detailed messages |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-11 | Initial creation from spec | Task Decomposition Agent |
| 2026-01-11 | Full audit against codebase - corrected all file paths and existing types | Audit Agent |
| 2026-01-11 | Added FSV section, manual testing, edge cases, Constitution compliance matrix | Audit Agent |
