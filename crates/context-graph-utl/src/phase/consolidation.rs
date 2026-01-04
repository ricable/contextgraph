//! Memory consolidation phase detection.
//!
//! Implements consolidation phase detection based on activity levels,
//! inspired by sleep stage dynamics (NREM/REM) for memory consolidation.

use std::f32::consts::PI;
use std::time::{Duration, Instant};

use crate::config::PhaseConfig;
use crate::error::{UtlError, UtlResult};

/// Memory consolidation phases based on sleep-inspired dynamics.
///
/// Each phase has different characteristics for memory processing:
/// - **NREM**: Non-REM phase for replay and tight coupling
/// - **REM**: REM phase for exploring attractor dynamics
/// - **Wake**: Normal waking operation
///
/// # Constitution Reference
///
/// - NREM: Replay + tight coupling (recency_bias: 0.8)
/// - REM: Explore attractors (temp: 2.0)
/// - Wake: Normal operation (balanced processing)
///
/// # Example
///
/// ```
/// use context_graph_utl::phase::ConsolidationPhase;
///
/// let phase = ConsolidationPhase::NREM;
///
/// // Get phase-specific parameters
/// assert_eq!(phase.recency_bias(), 0.8);
/// assert!(phase.is_consolidation_phase());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ConsolidationPhase {
    /// Non-REM sleep phase: Memory replay with tight coupling.
    ///
    /// Characteristics:
    /// - High recency bias (0.8)
    /// - Strong coupling strength
    /// - Focus on recent memory consolidation
    NREM,

    /// REM sleep phase: Attractor exploration.
    ///
    /// Characteristics:
    /// - High temperature (2.0) for exploration
    /// - Loose coupling
    /// - Creative association and pattern discovery
    REM,

    /// Waking phase: Normal operation.
    ///
    /// Characteristics:
    /// - Balanced processing
    /// - Standard temperature (1.0)
    /// - Active learning and interaction
    #[default]
    Wake,
}

impl ConsolidationPhase {
    /// Get the recency bias for this phase.
    ///
    /// Higher values prioritize recent memories during replay.
    ///
    /// # Returns
    ///
    /// Recency bias in `[0, 1]`:
    /// - NREM: 0.8 (strong recent memory bias)
    /// - REM: 0.4 (moderate bias, more exploration)
    /// - Wake: 0.5 (balanced)
    #[inline]
    pub fn recency_bias(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.8,
            ConsolidationPhase::REM => 0.4,
            ConsolidationPhase::Wake => 0.5,
        }
    }

    /// Get the temperature parameter for this phase.
    ///
    /// Temperature controls exploration vs. exploitation:
    /// - High temperature: More random/exploratory
    /// - Low temperature: More deterministic/focused
    ///
    /// # Returns
    ///
    /// Temperature parameter:
    /// - NREM: 0.5 (focused replay)
    /// - REM: 2.0 (high exploration)
    /// - Wake: 1.0 (balanced)
    #[inline]
    pub fn temperature(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.5,
            ConsolidationPhase::REM => 2.0,
            ConsolidationPhase::Wake => 1.0,
        }
    }

    /// Get the coupling strength for this phase.
    ///
    /// Controls how strongly different memory systems are coupled.
    ///
    /// # Returns
    ///
    /// Coupling strength in `[0, 1]`:
    /// - NREM: 0.9 (tight coupling for replay)
    /// - REM: 0.3 (loose coupling for exploration)
    /// - Wake: 0.6 (moderate coupling)
    #[inline]
    pub fn coupling_strength(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.9,
            ConsolidationPhase::REM => 0.3,
            ConsolidationPhase::Wake => 0.6,
        }
    }

    /// Get the learning rate modifier for this phase.
    ///
    /// # Returns
    ///
    /// Learning rate multiplier:
    /// - NREM: 0.3 (reduced active learning)
    /// - REM: 0.5 (moderate learning from associations)
    /// - Wake: 1.0 (full learning rate)
    #[inline]
    pub fn learning_rate_modifier(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.3,
            ConsolidationPhase::REM => 0.5,
            ConsolidationPhase::Wake => 1.0,
        }
    }

    /// Check if this is a consolidation phase (NREM or REM).
    ///
    /// # Returns
    ///
    /// `true` if NREM or REM, `false` if Wake.
    #[inline]
    pub fn is_consolidation_phase(&self) -> bool {
        matches!(self, ConsolidationPhase::NREM | ConsolidationPhase::REM)
    }

    /// Check if this is the waking phase.
    #[inline]
    pub fn is_wake(&self) -> bool {
        matches!(self, ConsolidationPhase::Wake)
    }

    /// Get the default phase angle for this consolidation state.
    ///
    /// # Returns
    ///
    /// Phase angle in `[0, π]`:
    /// - NREM: 0 (synchronized, cos = 1)
    /// - REM: π (anti-phase, cos = -1)
    /// - Wake: π/2 (orthogonal, cos = 0)
    #[inline]
    pub fn default_phase_angle(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.0,
            ConsolidationPhase::REM => PI,
            ConsolidationPhase::Wake => PI / 2.0,
        }
    }

    /// Get a human-readable name for this phase.
    pub fn name(&self) -> &'static str {
        match self {
            ConsolidationPhase::NREM => "NREM",
            ConsolidationPhase::REM => "REM",
            ConsolidationPhase::Wake => "Wake",
        }
    }

    /// Get a description of this phase.
    pub fn description(&self) -> &'static str {
        match self {
            ConsolidationPhase::NREM => "Memory replay with tight coupling",
            ConsolidationPhase::REM => "Attractor exploration with loose coupling",
            ConsolidationPhase::Wake => "Normal waking operation",
        }
    }
}

impl std::fmt::Display for ConsolidationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Detector for current consolidation phase based on activity levels.
///
/// The phase detector analyzes activity metrics to determine whether
/// the system should be in NREM, REM, or Wake phase.
///
/// # Detection Logic
///
/// - **High activity** (> 0.7): Wake phase
/// - **Medium activity** (0.3-0.7): REM phase (dreaming/exploration)
/// - **Low activity** (< 0.3): NREM phase (deep consolidation)
///
/// # Example
///
/// ```
/// use context_graph_utl::phase::{PhaseDetector, ConsolidationPhase};
///
/// let detector = PhaseDetector::new();
///
/// // Low activity -> NREM
/// assert_eq!(detector.detect_phase(0.1), ConsolidationPhase::NREM);
///
/// // Medium activity -> REM
/// assert_eq!(detector.detect_phase(0.5), ConsolidationPhase::REM);
///
/// // High activity -> Wake
/// assert_eq!(detector.detect_phase(0.9), ConsolidationPhase::Wake);
/// ```
#[derive(Debug, Clone)]
pub struct PhaseDetector {
    /// Threshold below which NREM is detected.
    nrem_threshold: f32,

    /// Threshold above which Wake is detected.
    wake_threshold: f32,

    /// Current detected phase.
    current_phase: ConsolidationPhase,

    /// Activity history for smoothing.
    activity_history: Vec<f32>,

    /// Maximum history size for EMA calculation.
    history_size: usize,

    /// Minimum time in phase before transition (hysteresis).
    min_phase_duration: Duration,

    /// Time entered current phase.
    phase_entered: Option<Instant>,

    /// Smoothing factor for activity EMA.
    ema_alpha: f32,

    /// Current smoothed activity level.
    smoothed_activity: f32,
}

impl PhaseDetector {
    /// Create a new phase detector with default thresholds.
    ///
    /// Default thresholds:
    /// - NREM: activity < 0.3
    /// - REM: 0.3 <= activity <= 0.7
    /// - Wake: activity > 0.7
    pub fn new() -> Self {
        Self {
            nrem_threshold: 0.3,
            wake_threshold: 0.7,
            current_phase: ConsolidationPhase::Wake,
            activity_history: Vec::with_capacity(10),
            history_size: 10,
            min_phase_duration: Duration::from_secs(5),
            phase_entered: Some(Instant::now()),
            ema_alpha: 0.3,
            smoothed_activity: 0.5,
        }
    }

    /// Create a phase detector with custom thresholds.
    ///
    /// # Arguments
    ///
    /// * `nrem_threshold` - Activity level below which NREM is detected
    /// * `wake_threshold` - Activity level above which Wake is detected
    ///
    /// # Errors
    ///
    /// Returns `UtlError::PhaseError` if thresholds are invalid.
    pub fn with_thresholds(nrem_threshold: f32, wake_threshold: f32) -> UtlResult<Self> {
        if !(0.0..=1.0).contains(&nrem_threshold) {
            return Err(UtlError::PhaseError(format!(
                "NREM threshold must be in [0, 1], got {}",
                nrem_threshold
            )));
        }
        if !(0.0..=1.0).contains(&wake_threshold) {
            return Err(UtlError::PhaseError(format!(
                "Wake threshold must be in [0, 1], got {}",
                wake_threshold
            )));
        }
        if nrem_threshold >= wake_threshold {
            return Err(UtlError::PhaseError(format!(
                "NREM threshold ({}) must be less than Wake threshold ({})",
                nrem_threshold, wake_threshold
            )));
        }

        let mut detector = Self::new();
        detector.nrem_threshold = nrem_threshold;
        detector.wake_threshold = wake_threshold;
        Ok(detector)
    }

    /// Create a phase detector from phase configuration.
    pub fn from_config(config: &PhaseConfig) -> Self {
        let mut detector = Self::new();
        // Use sync_threshold to influence wake threshold
        detector.wake_threshold = config.sync_threshold;
        detector.nrem_threshold = config.sync_threshold * 0.4;
        detector
    }

    /// Detect the consolidation phase based on activity level.
    ///
    /// This is a simple instantaneous detection without smoothing.
    ///
    /// # Arguments
    ///
    /// * `activity` - Current activity level in `[0, 1]`
    ///
    /// # Returns
    ///
    /// The detected consolidation phase.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::phase::{PhaseDetector, ConsolidationPhase};
    ///
    /// let detector = PhaseDetector::new();
    ///
    /// assert_eq!(detector.detect_phase(0.2), ConsolidationPhase::NREM);
    /// assert_eq!(detector.detect_phase(0.5), ConsolidationPhase::REM);
    /// assert_eq!(detector.detect_phase(0.8), ConsolidationPhase::Wake);
    /// ```
    pub fn detect_phase(&self, activity: f32) -> ConsolidationPhase {
        let activity = activity.clamp(0.0, 1.0);

        if activity < self.nrem_threshold {
            ConsolidationPhase::NREM
        } else if activity > self.wake_threshold {
            ConsolidationPhase::Wake
        } else {
            ConsolidationPhase::REM
        }
    }

    /// Update the detector with a new activity measurement.
    ///
    /// Uses exponential moving average (EMA) for smoothing and
    /// applies hysteresis to prevent rapid phase transitions.
    ///
    /// # Arguments
    ///
    /// * `activity` - Current activity level in `[0, 1]`
    ///
    /// # Returns
    ///
    /// The current (possibly unchanged) consolidation phase.
    pub fn update(&mut self, activity: f32) -> ConsolidationPhase {
        let activity = activity.clamp(0.0, 1.0);

        // Update EMA
        self.smoothed_activity =
            self.ema_alpha * activity + (1.0 - self.ema_alpha) * self.smoothed_activity;

        // Add to history
        if self.activity_history.len() >= self.history_size {
            self.activity_history.remove(0);
        }
        self.activity_history.push(activity);

        // Detect new phase
        let detected = self.detect_phase(self.smoothed_activity);

        // Apply hysteresis
        if detected != self.current_phase {
            let can_transition = self
                .phase_entered
                .map(|entered| entered.elapsed() >= self.min_phase_duration)
                .unwrap_or(true);

            if can_transition {
                self.current_phase = detected;
                self.phase_entered = Some(Instant::now());
            }
        }

        self.current_phase
    }

    /// Get the current detected phase.
    #[inline]
    pub fn current_phase(&self) -> ConsolidationPhase {
        self.current_phase
    }

    /// Get the current smoothed activity level.
    #[inline]
    pub fn smoothed_activity(&self) -> f32 {
        self.smoothed_activity
    }

    /// Get the average activity from history.
    pub fn average_activity(&self) -> f32 {
        if self.activity_history.is_empty() {
            return self.smoothed_activity;
        }
        let sum: f32 = self.activity_history.iter().sum();
        sum / self.activity_history.len() as f32
    }

    /// Get the activity variance from history.
    pub fn activity_variance(&self) -> f32 {
        if self.activity_history.len() < 2 {
            return 0.0;
        }
        let mean = self.average_activity();
        let variance_sum: f32 = self
            .activity_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        variance_sum / (self.activity_history.len() - 1) as f32
    }

    /// Get how long the current phase has been active.
    pub fn phase_duration(&self) -> Duration {
        self.phase_entered
            .map(|entered| entered.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Force a specific phase (bypasses detection).
    ///
    /// Use sparingly; prefer `update()` for normal operation.
    pub fn force_phase(&mut self, phase: ConsolidationPhase) {
        self.current_phase = phase;
        self.phase_entered = Some(Instant::now());
    }

    /// Reset the detector to initial state.
    pub fn reset(&mut self) {
        self.current_phase = ConsolidationPhase::Wake;
        self.activity_history.clear();
        self.smoothed_activity = 0.5;
        self.phase_entered = Some(Instant::now());
    }

    /// Set the minimum phase duration (hysteresis).
    pub fn set_min_phase_duration(&mut self, duration: Duration) {
        self.min_phase_duration = duration;
    }

    /// Set the EMA smoothing factor.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing factor in `[0, 1]`. Higher = less smoothing.
    pub fn set_ema_alpha(&mut self, alpha: f32) {
        self.ema_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Get the NREM threshold.
    #[inline]
    pub fn nrem_threshold(&self) -> f32 {
        self.nrem_threshold
    }

    /// Get the Wake threshold.
    #[inline]
    pub fn wake_threshold(&self) -> f32 {
        self.wake_threshold
    }

    /// Check if currently in a consolidation phase (NREM or REM).
    pub fn is_consolidating(&self) -> bool {
        self.current_phase.is_consolidation_phase()
    }

    /// Get the recommended phase for a given time of day.
    ///
    /// Simulates circadian rhythm:
    /// - Night (0:00-6:00): NREM/REM cycle
    /// - Day (6:00-22:00): Wake
    /// - Evening (22:00-24:00): Transition to NREM
    ///
    /// # Arguments
    ///
    /// * `hour` - Hour of day (0-23)
    ///
    /// # Returns
    ///
    /// Recommended phase based on circadian model.
    pub fn circadian_phase(hour: u8) -> ConsolidationPhase {
        match hour {
            0..=2 => ConsolidationPhase::NREM,
            3..=4 => ConsolidationPhase::REM,
            5 => ConsolidationPhase::NREM,
            6..=21 => ConsolidationPhase::Wake,
            22..=23 => ConsolidationPhase::NREM,
            _ => ConsolidationPhase::Wake,
        }
    }
}

impl Default for PhaseDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_phase_recency_bias() {
        assert_eq!(ConsolidationPhase::NREM.recency_bias(), 0.8);
        assert_eq!(ConsolidationPhase::REM.recency_bias(), 0.4);
        assert_eq!(ConsolidationPhase::Wake.recency_bias(), 0.5);
    }

    #[test]
    fn test_consolidation_phase_temperature() {
        assert_eq!(ConsolidationPhase::NREM.temperature(), 0.5);
        assert_eq!(ConsolidationPhase::REM.temperature(), 2.0);
        assert_eq!(ConsolidationPhase::Wake.temperature(), 1.0);
    }

    #[test]
    fn test_consolidation_phase_coupling() {
        assert_eq!(ConsolidationPhase::NREM.coupling_strength(), 0.9);
        assert_eq!(ConsolidationPhase::REM.coupling_strength(), 0.3);
        assert_eq!(ConsolidationPhase::Wake.coupling_strength(), 0.6);
    }

    #[test]
    fn test_consolidation_phase_learning_rate() {
        assert_eq!(ConsolidationPhase::NREM.learning_rate_modifier(), 0.3);
        assert_eq!(ConsolidationPhase::REM.learning_rate_modifier(), 0.5);
        assert_eq!(ConsolidationPhase::Wake.learning_rate_modifier(), 1.0);
    }

    #[test]
    fn test_is_consolidation_phase() {
        assert!(ConsolidationPhase::NREM.is_consolidation_phase());
        assert!(ConsolidationPhase::REM.is_consolidation_phase());
        assert!(!ConsolidationPhase::Wake.is_consolidation_phase());
    }

    #[test]
    fn test_is_wake() {
        assert!(!ConsolidationPhase::NREM.is_wake());
        assert!(!ConsolidationPhase::REM.is_wake());
        assert!(ConsolidationPhase::Wake.is_wake());
    }

    #[test]
    fn test_default_phase_angle() {
        assert_eq!(ConsolidationPhase::NREM.default_phase_angle(), 0.0);
        assert_eq!(ConsolidationPhase::REM.default_phase_angle(), PI);
        assert!((ConsolidationPhase::Wake.default_phase_angle() - PI / 2.0).abs() < 0.001);
    }

    #[test]
    fn test_phase_name_and_description() {
        assert_eq!(ConsolidationPhase::NREM.name(), "NREM");
        assert!(!ConsolidationPhase::NREM.description().is_empty());

        assert_eq!(ConsolidationPhase::REM.name(), "REM");
        assert!(!ConsolidationPhase::REM.description().is_empty());

        assert_eq!(ConsolidationPhase::Wake.name(), "Wake");
        assert!(!ConsolidationPhase::Wake.description().is_empty());
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", ConsolidationPhase::NREM), "NREM");
        assert_eq!(format!("{}", ConsolidationPhase::REM), "REM");
        assert_eq!(format!("{}", ConsolidationPhase::Wake), "Wake");
    }

    #[test]
    fn test_phase_default() {
        assert_eq!(ConsolidationPhase::default(), ConsolidationPhase::Wake);
    }

    #[test]
    fn test_detector_creation() {
        let detector = PhaseDetector::new();
        assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);
    }

    #[test]
    fn test_detector_with_thresholds_valid() {
        let detector = PhaseDetector::with_thresholds(0.2, 0.8).unwrap();
        assert_eq!(detector.nrem_threshold(), 0.2);
        assert_eq!(detector.wake_threshold(), 0.8);
    }

    #[test]
    fn test_detector_with_thresholds_invalid() {
        // NREM threshold too high
        assert!(PhaseDetector::with_thresholds(1.5, 0.8).is_err());

        // Wake threshold too low
        assert!(PhaseDetector::with_thresholds(0.2, -0.1).is_err());

        // NREM >= Wake
        assert!(PhaseDetector::with_thresholds(0.8, 0.3).is_err());

        // Equal thresholds
        assert!(PhaseDetector::with_thresholds(0.5, 0.5).is_err());
    }

    #[test]
    fn test_detect_phase_nrem() {
        let detector = PhaseDetector::new();
        assert_eq!(detector.detect_phase(0.1), ConsolidationPhase::NREM);
        assert_eq!(detector.detect_phase(0.0), ConsolidationPhase::NREM);
        assert_eq!(detector.detect_phase(0.29), ConsolidationPhase::NREM);
    }

    #[test]
    fn test_detect_phase_rem() {
        let detector = PhaseDetector::new();
        assert_eq!(detector.detect_phase(0.5), ConsolidationPhase::REM);
        assert_eq!(detector.detect_phase(0.31), ConsolidationPhase::REM);
        assert_eq!(detector.detect_phase(0.69), ConsolidationPhase::REM);
    }

    #[test]
    fn test_detect_phase_wake() {
        let detector = PhaseDetector::new();
        assert_eq!(detector.detect_phase(0.8), ConsolidationPhase::Wake);
        assert_eq!(detector.detect_phase(0.9), ConsolidationPhase::Wake);
        assert_eq!(detector.detect_phase(1.0), ConsolidationPhase::Wake);
    }

    #[test]
    fn test_detect_phase_clamping() {
        let detector = PhaseDetector::new();

        // Values outside [0, 1] should be clamped
        assert_eq!(detector.detect_phase(-0.5), ConsolidationPhase::NREM);
        assert_eq!(detector.detect_phase(1.5), ConsolidationPhase::Wake);
    }

    #[test]
    fn test_update_smoothing() {
        let mut detector = PhaseDetector::new();
        detector.set_min_phase_duration(Duration::ZERO); // Disable hysteresis for test

        // Start with high activity
        for _ in 0..5 {
            detector.update(0.9);
        }
        assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);

        // Gradually lower activity
        for _ in 0..20 {
            detector.update(0.1);
        }

        // Should eventually transition to NREM
        // (EMA smoothing means it takes multiple updates)
        assert!(detector.smoothed_activity() < 0.3);
    }

    #[test]
    fn test_average_activity() {
        let mut detector = PhaseDetector::new();

        detector.update(0.2);
        detector.update(0.4);
        detector.update(0.6);
        detector.update(0.8);

        let avg = detector.average_activity();
        assert!((avg - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_activity_variance() {
        let mut detector = PhaseDetector::new();

        // Same value multiple times = zero variance
        for _ in 0..5 {
            detector.update(0.5);
        }
        assert!(detector.activity_variance() < 0.01);

        // Different values = non-zero variance
        detector.reset();
        detector.update(0.0);
        detector.update(1.0);
        assert!(detector.activity_variance() > 0.0);
    }

    #[test]
    fn test_force_phase() {
        let mut detector = PhaseDetector::new();

        detector.force_phase(ConsolidationPhase::NREM);
        assert_eq!(detector.current_phase(), ConsolidationPhase::NREM);

        detector.force_phase(ConsolidationPhase::REM);
        assert_eq!(detector.current_phase(), ConsolidationPhase::REM);
    }

    #[test]
    fn test_reset() {
        let mut detector = PhaseDetector::new();

        detector.force_phase(ConsolidationPhase::NREM);
        detector.update(0.1);

        detector.reset();

        assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);
        assert!((detector.smoothed_activity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_is_consolidating() {
        let mut detector = PhaseDetector::new();

        detector.force_phase(ConsolidationPhase::Wake);
        assert!(!detector.is_consolidating());

        detector.force_phase(ConsolidationPhase::NREM);
        assert!(detector.is_consolidating());

        detector.force_phase(ConsolidationPhase::REM);
        assert!(detector.is_consolidating());
    }

    #[test]
    fn test_circadian_phase() {
        // Night hours
        assert_eq!(PhaseDetector::circadian_phase(1), ConsolidationPhase::NREM);
        assert_eq!(PhaseDetector::circadian_phase(3), ConsolidationPhase::REM);

        // Day hours
        assert_eq!(PhaseDetector::circadian_phase(12), ConsolidationPhase::Wake);
        assert_eq!(PhaseDetector::circadian_phase(18), ConsolidationPhase::Wake);

        // Evening
        assert_eq!(PhaseDetector::circadian_phase(23), ConsolidationPhase::NREM);
    }

    #[test]
    fn test_set_ema_alpha() {
        let mut detector = PhaseDetector::new();

        detector.set_ema_alpha(0.8);
        // Alpha is internal, just verify no panic

        // Test clamping
        detector.set_ema_alpha(1.5);
        detector.set_ema_alpha(-0.2);
    }

    #[test]
    fn test_from_config() {
        let config = PhaseConfig::default();
        let detector = PhaseDetector::from_config(&config);

        // Should use sync_threshold to derive thresholds
        assert!(detector.wake_threshold() > 0.0);
        assert!(detector.nrem_threshold() > 0.0);
        assert!(detector.nrem_threshold() < detector.wake_threshold());
    }

    #[test]
    fn test_phase_equality() {
        assert_eq!(ConsolidationPhase::NREM, ConsolidationPhase::NREM);
        assert_ne!(ConsolidationPhase::NREM, ConsolidationPhase::REM);
        assert_ne!(ConsolidationPhase::REM, ConsolidationPhase::Wake);
    }

    #[test]
    fn test_phase_clone() {
        let phase = ConsolidationPhase::REM;
        let cloned = phase;
        assert_eq!(phase, cloned);
    }

    #[test]
    fn test_detector_default() {
        let detector = PhaseDetector::default();
        assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);
    }
}
