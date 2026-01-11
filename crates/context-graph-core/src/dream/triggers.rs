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
        if enabled {
            info!("Dream triggers enabled");
        } else {
            info!("Dream triggers disabled");
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

    /// Access entropy window mutably for testing with custom parameters.
    ///
    /// # Note
    /// This method is primarily for testing purposes to configure
    /// shorter window durations with REAL time (not mocked).
    pub fn entropy_window_mut(&mut self) -> &mut EntropyWindow {
        &mut self.entropy_window
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
        assert_eq!(
            manager.entropy_window.threshold, 0.7,
            "Entropy threshold must be 0.7 per Constitution"
        );
        assert_eq!(
            manager.entropy_window.window_duration,
            Duration::from_secs(300),
            "Entropy window must be 5 minutes per Constitution"
        );

        // Verify GPU state uses Constitution defaults
        assert_eq!(
            manager.gpu_state.threshold, 0.30,
            "GPU threshold must be 0.30 (30%) per Constitution, NOT 80%"
        );
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
        assert!(
            manager.should_trigger(),
            "Manual trigger should bypass cooldown"
        );
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

        assert!(
            !manager.should_trigger(),
            "Average ~22% should not trigger"
        );
    }

    // ============ Entropy Trigger Tests ============

    #[test]
    fn test_trigger_manager_entropy_requires_sustained_high() {
        // Use short window for testing with REAL time
        let mut manager = TriggerManager::new();
        manager.entropy_window = EntropyWindow::with_params(Duration::from_millis(50), 0.7);
        manager.trigger_cooldown = Duration::from_millis(1);

        // Push high entropy - starts tracking
        manager.update_entropy(0.8);
        assert!(!manager.should_trigger(), "Should not trigger immediately");

        // Wait for window duration plus margin
        thread::sleep(Duration::from_millis(60));
        manager.update_entropy(0.85);

        assert!(
            manager.should_trigger(),
            "Should trigger after sustained high entropy"
        );
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::HighEntropy)
        );
    }

    #[test]
    fn test_trigger_manager_entropy_resets_on_low() {
        let mut manager = TriggerManager::new();
        manager.entropy_window = EntropyWindow::with_params(Duration::from_millis(50), 0.7);

        // Start high
        manager.update_entropy(0.9);
        thread::sleep(Duration::from_millis(30));

        // Drop below threshold - resets tracking
        manager.update_entropy(0.5);

        // Wait and check - should NOT trigger because tracking reset
        thread::sleep(Duration::from_millis(30));
        manager.update_entropy(0.5);

        assert!(
            !manager.should_trigger(),
            "Low entropy should reset tracking"
        );
    }

    // ============ Cooldown Tests ============

    #[test]
    fn test_trigger_manager_cooldown_prevents_rapid_retrigger() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(100));

        // Trigger with GPU
        manager.update_gpu_usage(0.35);
        assert!(manager.should_trigger());

        manager.mark_triggered(ExtendedTriggerReason::GpuOverload);

        // Try to trigger again immediately (within cooldown)
        // Even after reset(), cooldown should prevent re-trigger
        manager.reset(); // Simulate dream cycle completion
        manager.update_gpu_usage(0.40);
        assert!(
            !manager.should_trigger(),
            "Cooldown should prevent re-trigger even after reset"
        );

        // Wait for cooldown to expire
        thread::sleep(Duration::from_millis(150));

        // Now with cooldown expired, new GPU trigger should fire
        manager.update_gpu_usage(0.40);
        assert!(
            manager.should_trigger(),
            "Should trigger after cooldown expires"
        );
    }

    #[test]
    fn test_trigger_manager_cooldown_remaining() {
        let mut manager = TriggerManager::with_cooldown(Duration::from_millis(100));

        assert!(
            manager.cooldown_remaining().is_none(),
            "No cooldown initially"
        );

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

        assert!(
            !manager.should_trigger(),
            "Disabled manager should not trigger"
        );
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
        assert!(
            entropy < 0.5,
            "Regular queries should have low entropy: {}",
            entropy
        );
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
        assert!(
            entropy > 0.3,
            "Irregular queries should have higher entropy: {}",
            entropy
        );
    }

    // ============ Priority Order Tests ============

    #[test]
    fn test_trigger_manager_priority_manual_over_gpu_over_entropy() {
        let mut manager = TriggerManager::new();

        // Setup GPU trigger
        manager.update_gpu_usage(0.35);
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::GpuOverload)
        );

        // Add manual - should take priority
        manager.request_manual_trigger();
        assert_eq!(
            manager.check_triggers(),
            Some(ExtendedTriggerReason::Manual)
        );
    }
}
