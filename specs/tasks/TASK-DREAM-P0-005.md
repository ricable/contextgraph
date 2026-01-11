# TASK-DREAM-P0-005: Dream Trigger Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-005 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 2 (Logic) |
| **Priority** | P0 - Critical |
| **Effort** | 3 hours |
| **Dependencies** | TASK-DREAM-P0-001 |
| **Blocks** | TASK-DREAM-P0-006 |

---

## 1. Objective

Implement the entropy-based and GPU-based dream triggers. This extends the existing DreamScheduler to support:
1. Entropy threshold trigger (>0.7 sustained for 5 minutes)
2. GPU utilization trigger (>80%)
3. Integration with GWT mental checks subsystem

---

## 2. Input Context Files

```yaml
must_read:
  - path: crates/context-graph-core/src/dream/scheduler.rs
    purpose: Existing DreamScheduler to be extended
  - path: crates/context-graph-core/src/dream/types.rs
    purpose: EntropyWindow and GpuTriggerState types (from TASK-001)
  - path: crates/context-graph-core/src/dream/mod.rs
    purpose: Dream module structure and TriggerReason enum

should_read:
  - path: crates/context-graph-core/src/gwt/mod.rs
    purpose: GWT mental checks for entropy integration (if exists)
```

---

## 3. Files to Create/Modify

### 3.1 Create: `crates/context-graph-core/src/dream/triggers.rs`

```rust
//! Dream Trigger Implementations
//!
//! Implements additional trigger mechanisms beyond idle timeout:
//! - High entropy trigger (>0.7 for 5 minutes)
//! - GPU overload trigger (>80% utilization)
//!
//! Constitution Reference: Section gwt.mental_checks, Section dream.trigger

use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use super::types::{EntropyWindow, ExtendedTriggerReason, GpuTriggerState};

/// Unified trigger manager for dream cycles.
///
/// Combines all trigger mechanisms into a single interface:
/// - Idle timeout (from DreamScheduler)
/// - High entropy (>0.7 sustained for 5 minutes)
/// - GPU overload (>80% utilization)
/// - Manual trigger
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
    pub fn new() -> Self {
        Self {
            entropy_window: EntropyWindow::new(),
            gpu_state: GpuTriggerState::new(),
            manual_trigger: false,
            last_trigger_reason: None,
            trigger_cooldown: Duration::from_secs(1800), // 30 minutes
            last_trigger_time: None,
            enabled: true,
        }
    }

    /// Create with custom cooldown (for testing).
    pub fn with_cooldown(cooldown: Duration) -> Self {
        let mut manager = Self::new();
        manager.trigger_cooldown = cooldown;
        manager
    }

    /// Update entropy reading.
    ///
    /// Called periodically (e.g., every second) with system entropy value.
    ///
    /// # Arguments
    ///
    /// * `entropy` - Current system entropy [0.0, 1.0]
    pub fn update_entropy(&mut self, entropy: f32) {
        if !self.enabled {
            return;
        }

        self.entropy_window.push(entropy);

        if self.entropy_window.should_trigger() {
            debug!(
                "Entropy trigger condition met: avg={:.3}, min={:.3}",
                self.entropy_window.average(),
                self.entropy_window.minimum()
            );
        }
    }

    /// Update GPU utilization reading.
    ///
    /// Called periodically with GPU usage percentage.
    ///
    /// # Arguments
    ///
    /// * `usage` - Current GPU usage [0.0, 1.0]
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
    pub fn check_triggers(&self) -> Option<ExtendedTriggerReason> {
        if !self.enabled {
            return None;
        }

        // Check cooldown
        if let Some(last_time) = self.last_trigger_time {
            if last_time.elapsed() < self.trigger_cooldown {
                return None;
            }
        }

        // Check manual trigger (highest priority)
        if self.manual_trigger {
            return Some(ExtendedTriggerReason::Manual);
        }

        // Check GPU trigger
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
    pub fn should_trigger(&self) -> bool {
        self.check_triggers().is_some()
    }

    /// Mark that a trigger fired (starts cooldown).
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
    pub fn current_entropy(&self) -> f32 {
        self.entropy_window.average()
    }

    /// Get current GPU usage.
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
        }
    }

    /// Check if triggers are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
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
/// Real implementation would use NVML or similar.
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
            // TODO: Implement real GPU monitoring via NVML
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
        // TODO: Check for actual GPU
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
/// Computes system entropy based on:
/// - Query rate variance
/// - Memory access patterns
/// - Context switching frequency
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
            window: Duration::from_secs(60),
        }
    }

    /// Record a query event.
    pub fn record_query(&mut self) {
        let now = Instant::now();

        self.query_times.push(now);

        // Trim old queries
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
    /// - 0.0 = no activity (low entropy)
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

    /// Get the number of queries in the window.
    pub fn query_count(&self) -> usize {
        self.query_times.len()
    }
}

impl Default for EntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration with GWT mental checks subsystem.
///
/// Bridges dream triggers with the Global Workspace Theory
/// consciousness model's mental check events.
pub struct GwtIntegration {
    /// Trigger manager reference
    trigger_manager: TriggerManager,

    /// Entropy calculator
    entropy_calc: EntropyCalculator,

    /// GPU monitor
    gpu_monitor: GpuMonitor,

    /// Update interval
    update_interval: Duration,

    /// Last update time
    last_update: Option<Instant>,
}

impl GwtIntegration {
    /// Create a new GWT integration.
    pub fn new() -> Self {
        Self {
            trigger_manager: TriggerManager::new(),
            entropy_calc: EntropyCalculator::new(),
            gpu_monitor: GpuMonitor::new(),
            update_interval: Duration::from_secs(1),
            last_update: None,
        }
    }

    /// Record a query event (for entropy calculation).
    pub fn on_query(&mut self) {
        self.entropy_calc.record_query();
        self.maybe_update();
    }

    /// Force an update of all metrics.
    pub fn update(&mut self) {
        // Update entropy
        let entropy = self.entropy_calc.calculate();
        self.trigger_manager.update_entropy(entropy);

        // Update GPU usage
        let gpu_usage = self.gpu_monitor.get_usage();
        self.trigger_manager.update_gpu_usage(gpu_usage);

        self.last_update = Some(Instant::now());
    }

    /// Maybe update if interval has passed.
    fn maybe_update(&mut self) {
        let should_update = self
            .last_update
            .map(|t| t.elapsed() >= self.update_interval)
            .unwrap_or(true);

        if should_update {
            self.update();
        }
    }

    /// Check if dream should be triggered.
    pub fn should_trigger_dream(&self) -> bool {
        self.trigger_manager.should_trigger()
    }

    /// Get trigger reason if triggered.
    pub fn check_trigger(&self) -> Option<ExtendedTriggerReason> {
        self.trigger_manager.check_triggers()
    }

    /// Mark dream as triggered.
    pub fn on_dream_triggered(&mut self, reason: ExtendedTriggerReason) {
        self.trigger_manager.mark_triggered(reason);
    }

    /// Reset after dream completion.
    pub fn on_dream_completed(&mut self) {
        self.trigger_manager.reset();
    }

    /// Get current system entropy.
    pub fn current_entropy(&self) -> f32 {
        self.entropy_calc.calculate()
    }

    /// Get current GPU usage.
    pub fn current_gpu_usage(&self) -> f32 {
        self.gpu_monitor.get_usage()
    }

    /// Access trigger manager directly.
    pub fn trigger_manager(&self) -> &TriggerManager {
        &self.trigger_manager
    }

    /// Access trigger manager mutably.
    pub fn trigger_manager_mut(&mut self) -> &mut TriggerManager {
        &mut self.trigger_manager
    }

    /// Access GPU monitor for testing.
    pub fn gpu_monitor_mut(&mut self) -> &mut GpuMonitor {
        &mut self.gpu_monitor
    }
}

impl Default for GwtIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_trigger_manager_creation() {
        let manager = TriggerManager::new();
        assert!(manager.is_enabled());
        assert!(!manager.should_trigger());
    }

    #[test]
    fn test_trigger_manager_manual() {
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
    fn test_trigger_manager_gpu() {
        let mut manager = TriggerManager::new();

        // Below threshold
        manager.update_gpu_usage(0.5);
        assert!(!manager.should_trigger());

        // Above threshold
        manager.update_gpu_usage(0.85);
        assert!(manager.should_trigger());
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::GpuOverload)
        );
    }

    #[test]
    fn test_trigger_manager_entropy() {
        // Use short window for testing
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(1));
        manager.entropy_window = EntropyWindow::with_params(
            Duration::from_millis(50),
            0.7,
        );

        // Push high entropy values
        manager.update_entropy(0.8);
        thread::sleep(Duration::from_millis(60));
        manager.update_entropy(0.9);

        assert!(manager.should_trigger());
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::HighEntropy)
        );
    }

    #[test]
    fn test_trigger_manager_cooldown() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(100));

        manager.request_manual_trigger();
        assert!(manager.should_trigger());

        // Mark as triggered
        manager.mark_triggered(ExtendedTriggerReason::Manual);

        // Request again immediately
        manager.request_manual_trigger();
        assert!(!manager.should_trigger()); // Cooldown active

        // Wait for cooldown
        thread::sleep(Duration::from_millis(150));

        manager.request_manual_trigger();
        assert!(manager.should_trigger()); // Cooldown expired
    }

    #[test]
    fn test_trigger_manager_disabled() {
        let mut manager = TriggerManager::new();

        manager.set_enabled(false);

        manager.request_manual_trigger();
        manager.update_gpu_usage(0.95);

        assert!(!manager.should_trigger());
    }

    #[test]
    fn test_gpu_monitor() {
        let mut monitor = GpuMonitor::new();

        assert_eq!(monitor.get_usage(), 0.0);

        monitor.set_simulated_usage(0.75);
        assert_eq!(monitor.get_usage(), 0.75);

        // Test clamping
        monitor.set_simulated_usage(1.5);
        assert_eq!(monitor.get_usage(), 1.0);
    }

    #[test]
    fn test_entropy_calculator_empty() {
        let calc = EntropyCalculator::new();
        assert_eq!(calc.calculate(), 0.0);
    }

    #[test]
    fn test_entropy_calculator_regular_queries() {
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
    fn test_entropy_calculator_irregular_queries() {
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

    #[test]
    fn test_gwt_integration_basic() {
        let mut integration = GwtIntegration::new();

        // Initially no trigger
        assert!(!integration.should_trigger_dream());

        // Simulate high GPU usage
        integration.gpu_monitor_mut().set_simulated_usage(0.9);
        integration.update();

        assert!(integration.should_trigger_dream());
    }

    #[test]
    fn test_gwt_integration_query_entropy() {
        let mut integration = GwtIntegration::new();

        // Record many queries in bursts
        for _ in 0..10 {
            integration.on_query();
        }

        // Entropy should be calculated
        let entropy = integration.current_entropy();
        assert!(entropy >= 0.0 && entropy <= 1.0);
    }
}
```

### 3.2 Modify: `crates/context-graph-core/src/dream/scheduler.rs`

Add integration with TriggerManager:

```rust
// Add imports:
use super::triggers::TriggerManager;
use super::types::ExtendedTriggerReason;

// Add field to DreamScheduler:
    /// Extended trigger manager
    trigger_manager: TriggerManager,

// Update new() to initialize trigger_manager:
    trigger_manager: TriggerManager::new(),

// Add methods to DreamScheduler:

    /// Update entropy for trigger detection.
    pub fn update_entropy(&mut self, entropy: f32) {
        self.trigger_manager.update_entropy(entropy);
    }

    /// Update GPU usage for trigger detection.
    pub fn update_gpu_usage(&mut self, usage: f32) {
        self.trigger_manager.update_gpu_usage(usage);
    }

    /// Check extended triggers (entropy, GPU, manual).
    pub fn check_extended_triggers(&self) -> Option<ExtendedTriggerReason> {
        self.trigger_manager.check_triggers()
    }

    /// Request manual trigger.
    pub fn request_manual_trigger(&mut self) {
        self.trigger_manager.request_manual_trigger();
    }
```

### 3.3 Modify: `crates/context-graph-core/src/dream/mod.rs`

Add triggers module export:

```rust
// Add after hyperbolic_walk module:
pub mod triggers;

// Add to re-exports:
pub use triggers::{
    TriggerManager,
    GpuMonitor,
    EntropyCalculator,
    GwtIntegration,
};
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

pub struct GwtIntegration { /* internal */ }
impl GwtIntegration {
    pub fn new() -> Self;
    pub fn on_query(&mut self);
    pub fn update(&mut self);
    pub fn should_trigger_dream(&self) -> bool;
    pub fn check_trigger(&self) -> Option<ExtendedTriggerReason>;
    pub fn on_dream_triggered(&mut self, reason: ExtendedTriggerReason);
    pub fn on_dream_completed(&mut self);
    pub fn current_entropy(&self) -> f32;
    pub fn current_gpu_usage(&self) -> f32;
}
```

### 4.2 Validation Criteria

| Criterion | Check |
|-----------|-------|
| Compiles | `cargo build -p context-graph-core` |
| Tests pass | `cargo test -p context-graph-core dream::triggers` |
| No clippy warnings | `cargo clippy -p context-graph-core` |
| Entropy threshold | Triggers at >0.7 sustained for 5min |
| GPU threshold | Triggers at >=80% |
| Cooldown works | No re-trigger during cooldown |
| Manual trigger | Highest priority |
| Disable works | No triggers when disabled |

### 4.3 Test Coverage Requirements

- [ ] TriggerManager creation
- [ ] Manual trigger priority
- [ ] GPU threshold at 80%
- [ ] Entropy threshold at 0.7
- [ ] Cooldown prevents re-trigger
- [ ] Cooldown expiration allows re-trigger
- [ ] Disabled manager blocks all triggers
- [ ] GpuMonitor simulated usage
- [ ] EntropyCalculator empty returns 0
- [ ] EntropyCalculator regular queries = low entropy
- [ ] EntropyCalculator irregular queries = high entropy
- [ ] GwtIntegration basic flow

---

## 5. Implementation Notes

### 5.1 Entropy Calculation

Based on query inter-arrival time variability:
- Calculate coefficient of variation (CV = std/mean)
- CV > 1: High variability = high entropy
- CV < 1: Regular pattern = low entropy
- Normalize to [0, 1]

### 5.2 Constitution Compliance

- Entropy threshold: 0.7
- Entropy window: 5 minutes
- GPU threshold: 80%
- Cooldown: 30 minutes (same as idle timeout)

### 5.3 GPU Monitoring

Currently stub implementation. Real implementation would use:
- NVML (NVIDIA Management Library)
- ROCm for AMD GPUs
- Metal Performance Shaders for Apple

---

## 6. Estimated Effort Breakdown

| Phase | Duration |
|-------|----------|
| TriggerManager core | 45 min |
| GpuMonitor stub | 15 min |
| EntropyCalculator | 30 min |
| GwtIntegration | 30 min |
| DreamScheduler integration | 15 min |
| Unit tests | 45 min |
| **Total** | **3 hours** |
