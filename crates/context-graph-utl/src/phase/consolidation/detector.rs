//! PhaseDetector for consolidation phase detection.
//!
//! Implements detection logic for NREM, REM, and Wake phases based on activity levels.

use std::time::{Duration, Instant};

use crate::config::PhaseConfig;
use crate::error::{UtlError, UtlResult};

use super::ConsolidationPhase;

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
/// assert_eq!(detector.detect_phase(0.1), ConsolidationPhase::NREM);
/// assert_eq!(detector.detect_phase(0.5), ConsolidationPhase::REM);
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
    /// Default: NREM < 0.3, REM 0.3-0.7, Wake > 0.7
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
        detector.wake_threshold = config.sync_threshold;
        detector.nrem_threshold = config.sync_threshold * 0.4;
        detector
    }

    /// Detect the consolidation phase based on activity level (instantaneous, no smoothing).
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::phase::{PhaseDetector, ConsolidationPhase};
    ///
    /// let detector = PhaseDetector::new();
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
    /// Uses EMA for smoothing and hysteresis to prevent rapid phase transitions.
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

    /// Force a specific phase (bypasses detection). Use sparingly.
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

    /// Set the EMA smoothing factor. Higher = less smoothing.
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

    /// Get recommended phase for a given time of day (circadian rhythm simulation).
    ///
    /// Night (0:00-6:00): NREM/REM cycle, Day (6:00-22:00): Wake, Evening: NREM
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
