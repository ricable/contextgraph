//! NORTH-010: DriftDetector Service
//!
//! Detects alignment drift from North Star goal using rolling statistical analysis.
//! Monitors alignment observations over a configurable window and calculates
//! drift severity, trend direction, and provides actionable recommendations.
//!
//! # Architecture
//!
//! The DriftDetector operates on alignment observations (scores between 0.0 and 1.0)
//! and maintains rolling statistics to detect when alignment drifts from baseline.
//!
//! Key metrics:
//! - Rolling mean: EWMA of recent alignment scores
//! - Rolling variance: Measure of alignment stability
//! - Trend: Direction of alignment change (Improving/Declining/Stable)
//! - Severity: Classification of drift magnitude (None/Mild/Moderate/Severe)
//!
//! # Example
//!
//! ```rust
//! use context_graph_core::autonomous::services::drift_detector::{DriftDetector, DriftRecommendation};
//! use context_graph_core::autonomous::drift::DriftConfig;
//!
//! let mut detector = DriftDetector::new();
//!
//! // Add alignment observations
//! detector.add_observation(0.80, 1000);
//! detector.add_observation(0.75, 2000);
//! detector.add_observation(0.70, 3000);
//!
//! // Check drift severity
//! let severity = detector.detect_drift();
//! let trend = detector.compute_trend();
//!
//! if detector.requires_attention() {
//!     let recommendation = detector.get_recommendation();
//!     // Handle based on recommendation
//! }
//! ```

use crate::autonomous::drift::{DriftConfig, DriftMonitoring, DriftSeverity, DriftTrend};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Minimum samples required for reliable statistics
const MIN_SAMPLES_DEFAULT: usize = 10;

/// A single data point for drift detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorDataPoint {
    /// Timestamp of observation (Unix epoch millis or monotonic counter)
    pub timestamp: u64,
    /// Alignment score at this point
    pub alignment: f32,
    /// Delta from the rolling mean at time of observation
    pub delta_from_mean: f32,
}

/// Recommendation for handling detected drift
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftRecommendation {
    /// No action needed - alignment is stable
    NoAction,
    /// Continue monitoring - minor drift detected
    Monitor,
    /// Review recent memories - moderate drift may indicate quality issues
    ReviewMemories,
    /// Adjust thresholds - drift may indicate threshold miscalibration
    AdjustThresholds,
    /// Recalibrate baseline - severe drift requires re-establishing baseline
    RecalibrateBaseline,
    /// User intervention required - critical drift detected
    UserIntervention,
}

/// Internal state for drift detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorState {
    /// Current alignment (most recent observation)
    pub current_alignment: f32,
    /// Rolling mean of alignment scores
    pub rolling_mean: f32,
    /// Rolling variance of alignment scores
    pub rolling_variance: f32,
    /// Current trend direction
    pub trend: DriftTrend,
    /// Current severity level
    pub severity: DriftSeverity,
    /// Historical data points within the window
    pub data_points: VecDeque<DetectorDataPoint>,
    /// Baseline alignment to compare against
    pub baseline: f32,
}

impl Default for DetectorState {
    fn default() -> Self {
        Self {
            current_alignment: 0.75,
            rolling_mean: 0.75,
            rolling_variance: 0.0,
            trend: DriftTrend::Stable,
            severity: DriftSeverity::None,
            data_points: VecDeque::with_capacity(256),
            baseline: 0.75,
        }
    }
}

/// Service for detecting alignment drift from North Star goal
#[derive(Clone, Debug)]
pub struct DriftDetector {
    config: DriftConfig,
    state: DetectorState,
    min_samples: usize,
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DriftDetector {
    /// Create a new DriftDetector with default configuration
    pub fn new() -> Self {
        Self {
            config: DriftConfig::default(),
            state: DetectorState::default(),
            min_samples: MIN_SAMPLES_DEFAULT,
        }
    }

    /// Create a new DriftDetector with custom configuration
    pub fn with_config(config: DriftConfig) -> Self {
        Self {
            config,
            state: DetectorState::default(),
            min_samples: MIN_SAMPLES_DEFAULT,
        }
    }

    /// Set custom minimum samples requirement
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        if min_samples == 0 {
            panic!("FAIL FAST: min_samples must be > 0");
        }
        self.min_samples = min_samples;
        self
    }

    /// Set initial baseline alignment
    pub fn with_baseline(mut self, baseline: f32) -> Self {
        if !(0.0..=1.0).contains(&baseline) {
            panic!(
                "FAIL FAST: baseline must be in range [0.0, 1.0], got {}",
                baseline
            );
        }
        self.state.baseline = baseline;
        self.state.rolling_mean = baseline;
        self.state.current_alignment = baseline;
        self
    }

    /// Add a new alignment observation
    ///
    /// # Panics
    /// Panics if alignment is not in range [0.0, 1.0]
    pub fn add_observation(&mut self, alignment: f32, timestamp: u64) {
        if !(0.0..=1.0).contains(&alignment) {
            panic!(
                "FAIL FAST: alignment must be in range [0.0, 1.0], got {}",
                alignment
            );
        }

        // Compute delta from current rolling mean before updating
        let delta_from_mean = alignment - self.state.rolling_mean;

        // Create and store data point
        let point = DetectorDataPoint {
            timestamp,
            alignment,
            delta_from_mean,
        };
        self.state.data_points.push_back(point);

        // Update current alignment
        self.state.current_alignment = alignment;

        // Trim old data points beyond window
        self.trim_window(timestamp);

        // Recompute rolling statistics
        self.compute_rolling_stats();
    }

    /// Trim data points outside the rolling window
    fn trim_window(&mut self, current_timestamp: u64) {
        // Window in milliseconds (assuming timestamp is millis)
        let window_ms = self.config.window_days as u64 * 24 * 60 * 60 * 1000;
        let cutoff = current_timestamp.saturating_sub(window_ms);

        while let Some(front) = self.state.data_points.front() {
            if front.timestamp < cutoff {
                self.state.data_points.pop_front();
            } else {
                break;
            }
        }
    }

    /// Compute rolling mean and variance from data points
    pub fn compute_rolling_stats(&mut self) {
        let n = self.state.data_points.len();
        if n == 0 {
            return;
        }

        // Compute mean
        let sum: f32 = self.state.data_points.iter().map(|p| p.alignment).sum();
        self.state.rolling_mean = sum / n as f32;

        // Compute variance using Welford's online algorithm for numerical stability
        if n < 2 {
            self.state.rolling_variance = 0.0;
        } else {
            let variance_sum: f32 = self
                .state
                .data_points
                .iter()
                .map(|p| {
                    let diff = p.alignment - self.state.rolling_mean;
                    diff * diff
                })
                .sum();
            self.state.rolling_variance = variance_sum / (n - 1) as f32;
        }

        // Update severity and trend
        self.update_severity();
        self.update_trend();
    }

    /// Update severity classification based on drift from baseline
    fn update_severity(&mut self) {
        let drift = (self.state.baseline - self.state.rolling_mean).abs();

        self.state.severity = if drift >= self.config.severe_threshold {
            DriftSeverity::Severe
        } else if drift >= self.config.alert_threshold {
            DriftSeverity::Moderate
        } else if drift > 0.01 {
            DriftSeverity::Mild
        } else {
            DriftSeverity::None
        };
    }

    /// Update trend based on recent data points
    fn update_trend(&mut self) {
        let n = self.state.data_points.len();
        if n < 3 {
            self.state.trend = DriftTrend::Stable;
            return;
        }

        // Use linear regression slope on recent points to determine trend
        // We use the last min(n, 10) points for trend detection
        let trend_window = n.min(10);
        let recent: Vec<f32> = self
            .state
            .data_points
            .iter()
            .skip(n - trend_window)
            .map(|p| p.alignment)
            .collect();

        // Compute slope using least squares
        let slope = self.compute_slope(&recent);

        // Classify trend based on slope magnitude
        const SLOPE_THRESHOLD: f32 = 0.005;
        self.state.trend = if slope > SLOPE_THRESHOLD {
            DriftTrend::Improving
        } else if slope < -SLOPE_THRESHOLD {
            DriftTrend::Declining
        } else {
            DriftTrend::Stable
        };
    }

    /// Compute slope using least squares regression
    fn compute_slope(&self, values: &[f32]) -> f32 {
        let n = values.len();
        if n < 2 {
            return 0.0;
        }

        let n_f = n as f32;

        // x values are indices 0, 1, 2, ...
        // sum_x = n*(n-1)/2
        let sum_x: f32 = (0..n).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..n).map(|i| (i * i) as f32).sum();

        let denominator = n_f * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return 0.0;
        }

        (n_f * sum_xy - sum_x * sum_y) / denominator
    }

    /// Detect current drift severity
    pub fn detect_drift(&self) -> DriftSeverity {
        self.state.severity.clone()
    }

    /// Compute current trend direction
    pub fn compute_trend(&self) -> DriftTrend {
        self.state.trend
    }

    /// Get the current drift score (distance from baseline)
    pub fn get_drift_score(&self) -> f32 {
        (self.state.baseline - self.state.rolling_mean).abs()
    }

    /// Check if drift requires attention (moderate or severe)
    pub fn requires_attention(&self) -> bool {
        matches!(
            self.state.severity,
            DriftSeverity::Moderate | DriftSeverity::Severe
        )
    }

    /// Check if drift requires user intervention (severe only)
    pub fn requires_intervention(&self) -> bool {
        matches!(self.state.severity, DriftSeverity::Severe)
    }

    /// Get recommendation based on current drift state
    pub fn get_recommendation(&self) -> DriftRecommendation {
        // Not enough samples for reliable analysis
        if self.state.data_points.len() < self.min_samples {
            return DriftRecommendation::Monitor;
        }

        match (&self.state.severity, &self.state.trend) {
            // No drift - no action
            (DriftSeverity::None, _) => DriftRecommendation::NoAction,

            // Mild drift - just monitor
            (DriftSeverity::Mild, DriftTrend::Improving) => DriftRecommendation::NoAction,
            (DriftSeverity::Mild, _) => DriftRecommendation::Monitor,

            // Moderate drift - depends on trend
            (DriftSeverity::Moderate, DriftTrend::Improving) => DriftRecommendation::Monitor,
            (DriftSeverity::Moderate, DriftTrend::Stable) => DriftRecommendation::ReviewMemories,
            (DriftSeverity::Moderate, DriftTrend::Declining | DriftTrend::Worsening) => {
                DriftRecommendation::AdjustThresholds
            }

            // Severe drift - serious action needed
            (DriftSeverity::Severe, DriftTrend::Improving) => {
                DriftRecommendation::RecalibrateBaseline
            }
            (DriftSeverity::Severe, DriftTrend::Stable) => DriftRecommendation::RecalibrateBaseline,
            (DriftSeverity::Severe, DriftTrend::Declining | DriftTrend::Worsening) => {
                DriftRecommendation::UserIntervention
            }
        }
    }

    /// Get the current rolling mean
    pub fn rolling_mean(&self) -> f32 {
        self.state.rolling_mean
    }

    /// Get the current rolling variance
    pub fn rolling_variance(&self) -> f32 {
        self.state.rolling_variance
    }

    /// Get the current baseline
    pub fn baseline(&self) -> f32 {
        self.state.baseline
    }

    /// Get number of data points in the window
    pub fn data_point_count(&self) -> usize {
        self.state.data_points.len()
    }

    /// Reset the baseline to the current rolling mean
    pub fn reset_baseline(&mut self) {
        self.state.baseline = self.state.rolling_mean;
        self.update_severity();
    }

    /// Check if continuous monitoring is enabled
    pub fn is_continuous_monitoring(&self) -> bool {
        matches!(self.config.monitoring, DriftMonitoring::Continuous)
    }

    /// Get a reference to the internal state (for testing/inspection)
    pub fn state(&self) -> &DetectorState {
        &self.state
    }

    /// Get a reference to the config
    pub fn config(&self) -> &DriftConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_detector_new() {
        let detector = DriftDetector::new();
        assert_eq!(detector.data_point_count(), 0);
        assert!((detector.rolling_mean() - 0.75).abs() < f32::EPSILON);
        assert!((detector.baseline() - 0.75).abs() < f32::EPSILON);
        assert_eq!(detector.detect_drift(), DriftSeverity::None);
        println!("[PASS] test_drift_detector_new");
    }

    #[test]
    fn test_drift_detector_with_config() {
        let config = DriftConfig {
            monitoring: DriftMonitoring::Periodic { interval_hours: 6 },
            alert_threshold: 0.03,
            auto_correct: false,
            severe_threshold: 0.08,
            window_days: 14,
        };
        let detector = DriftDetector::with_config(config);
        assert!(!detector.is_continuous_monitoring());
        assert!((detector.config().alert_threshold - 0.03).abs() < f32::EPSILON);
        assert!((detector.config().severe_threshold - 0.08).abs() < f32::EPSILON);
        println!("[PASS] test_drift_detector_with_config");
    }

    #[test]
    fn test_drift_detector_with_baseline() {
        let detector = DriftDetector::new().with_baseline(0.85);
        assert!((detector.baseline() - 0.85).abs() < f32::EPSILON);
        assert!((detector.rolling_mean() - 0.85).abs() < f32::EPSILON);
        println!("[PASS] test_drift_detector_with_baseline");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST: baseline must be in range")]
    fn test_drift_detector_invalid_baseline() {
        DriftDetector::new().with_baseline(1.5);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST: min_samples must be > 0")]
    fn test_drift_detector_zero_min_samples() {
        DriftDetector::new().with_min_samples(0);
    }

    #[test]
    fn test_add_observation_basic() {
        let mut detector = DriftDetector::new().with_baseline(0.80);

        detector.add_observation(0.75, 1000);
        assert_eq!(detector.data_point_count(), 1);
        assert!((detector.state().current_alignment - 0.75).abs() < f32::EPSILON);
        assert!((detector.rolling_mean() - 0.75).abs() < f32::EPSILON);
        println!("[PASS] test_add_observation_basic");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST: alignment must be in range")]
    fn test_add_observation_invalid_alignment() {
        let mut detector = DriftDetector::new();
        detector.add_observation(-0.1, 1000);
    }

    #[test]
    fn test_rolling_mean_calculation() {
        let mut detector = DriftDetector::new().with_baseline(0.80);

        detector.add_observation(0.90, 1000);
        detector.add_observation(0.80, 2000);
        detector.add_observation(0.70, 3000);

        // Mean should be (0.90 + 0.80 + 0.70) / 3 = 0.80
        let expected_mean = 0.80;
        assert!(
            (detector.rolling_mean() - expected_mean).abs() < 1e-6,
            "Expected mean {}, got {}",
            expected_mean,
            detector.rolling_mean()
        );
        println!("[PASS] test_rolling_mean_calculation");
    }

    #[test]
    fn test_rolling_variance_calculation() {
        let mut detector = DriftDetector::new().with_baseline(0.80);

        // Add values with known variance: [0.70, 0.80, 0.90]
        // Mean = 0.80
        // Variance = ((0.70-0.80)^2 + (0.80-0.80)^2 + (0.90-0.80)^2) / 2
        //          = (0.01 + 0 + 0.01) / 2 = 0.01
        detector.add_observation(0.70, 1000);
        detector.add_observation(0.80, 2000);
        detector.add_observation(0.90, 3000);

        let expected_variance = 0.01;
        assert!(
            (detector.rolling_variance() - expected_variance).abs() < 1e-6,
            "Expected variance {}, got {}",
            expected_variance,
            detector.rolling_variance()
        );
        println!("[PASS] test_rolling_variance_calculation");
    }

    #[test]
    fn test_variance_with_identical_values() {
        let mut detector = DriftDetector::new();

        // All same values should have zero variance
        for i in 0..5 {
            detector.add_observation(0.75, i * 1000);
        }

        assert!(
            detector.rolling_variance().abs() < f32::EPSILON,
            "Variance should be 0 for identical values, got {}",
            detector.rolling_variance()
        );
        println!("[PASS] test_variance_with_identical_values");
    }

    #[test]
    fn test_drift_severity_none() {
        let mut detector = DriftDetector::new().with_baseline(0.75);
        detector.add_observation(0.75, 1000);

        assert_eq!(detector.detect_drift(), DriftSeverity::None);
        assert!(!detector.requires_attention());
        assert!(!detector.requires_intervention());
        println!("[PASS] test_drift_severity_none");
    }

    #[test]
    fn test_drift_severity_mild() {
        let mut detector = DriftDetector::new().with_baseline(0.80);
        // Drift of 0.02 is mild (< 0.05 alert threshold)
        detector.add_observation(0.78, 1000);

        assert_eq!(detector.detect_drift(), DriftSeverity::Mild);
        assert!(!detector.requires_attention());
        assert!(!detector.requires_intervention());
        println!("[PASS] test_drift_severity_mild");
    }

    #[test]
    fn test_drift_severity_moderate() {
        let mut detector = DriftDetector::new().with_baseline(0.80);
        // Drift of 0.07 is moderate (>= 0.05, < 0.10)
        detector.add_observation(0.73, 1000);

        assert_eq!(detector.detect_drift(), DriftSeverity::Moderate);
        assert!(detector.requires_attention());
        assert!(!detector.requires_intervention());
        println!("[PASS] test_drift_severity_moderate");
    }

    #[test]
    fn test_drift_severity_severe() {
        let mut detector = DriftDetector::new().with_baseline(0.80);
        // Drift of 0.15 is severe (>= 0.10)
        detector.add_observation(0.65, 1000);

        assert_eq!(detector.detect_drift(), DriftSeverity::Severe);
        assert!(detector.requires_attention());
        assert!(detector.requires_intervention());
        println!("[PASS] test_drift_severity_severe");
    }

    #[test]
    fn test_drift_score_calculation() {
        let mut detector = DriftDetector::new().with_baseline(0.80);
        detector.add_observation(0.70, 1000);

        let drift_score = detector.get_drift_score();
        assert!(
            (drift_score - 0.10).abs() < f32::EPSILON,
            "Expected drift score 0.10, got {}",
            drift_score
        );
        println!("[PASS] test_drift_score_calculation");
    }

    #[test]
    fn test_trend_stable() {
        let mut detector = DriftDetector::new().with_baseline(0.75);

        // Add stable values
        for i in 0..10 {
            detector.add_observation(0.75, i * 1000);
        }

        assert_eq!(detector.compute_trend(), DriftTrend::Stable);
        println!("[PASS] test_trend_stable");
    }

    #[test]
    fn test_trend_improving() {
        let mut detector = DriftDetector::new().with_baseline(0.70);

        // Add increasing values
        for i in 0..10 {
            let alignment = 0.70 + (i as f32 * 0.02);
            detector.add_observation(alignment, i * 1000);
        }

        assert_eq!(detector.compute_trend(), DriftTrend::Improving);
        println!("[PASS] test_trend_improving");
    }

    #[test]
    fn test_trend_declining() {
        let mut detector = DriftDetector::new().with_baseline(0.90);

        // Add decreasing values
        for i in 0..10 {
            let alignment = 0.90 - (i as f32 * 0.02);
            detector.add_observation(alignment, i * 1000);
        }

        assert_eq!(detector.compute_trend(), DriftTrend::Declining);
        println!("[PASS] test_trend_declining");
    }

    #[test]
    fn test_trend_insufficient_data() {
        let mut detector = DriftDetector::new();
        detector.add_observation(0.80, 1000);
        detector.add_observation(0.75, 2000);

        // Less than 3 points should be stable
        assert_eq!(detector.compute_trend(), DriftTrend::Stable);
        println!("[PASS] test_trend_insufficient_data");
    }

    #[test]
    fn test_recommendation_no_action() {
        let mut detector = DriftDetector::new().with_baseline(0.75).with_min_samples(5);

        // Add enough stable observations
        for i in 0..10 {
            detector.add_observation(0.75, i * 1000);
        }

        assert_eq!(detector.get_recommendation(), DriftRecommendation::NoAction);
        println!("[PASS] test_recommendation_no_action");
    }

    #[test]
    fn test_recommendation_monitor_insufficient_samples() {
        let mut detector = DriftDetector::new().with_min_samples(10);

        // Only 5 samples
        for i in 0..5 {
            detector.add_observation(0.75, i * 1000);
        }

        assert_eq!(detector.get_recommendation(), DriftRecommendation::Monitor);
        println!("[PASS] test_recommendation_monitor_insufficient_samples");
    }

    #[test]
    fn test_recommendation_user_intervention() {
        let mut detector = DriftDetector::new().with_baseline(0.90).with_min_samples(5);

        // Add declining observations that create severe drift
        for i in 0..10 {
            let alignment = 0.70 - (i as f32 * 0.02);
            detector.add_observation(alignment.max(0.0), i * 1000);
        }

        assert_eq!(detector.detect_drift(), DriftSeverity::Severe);
        assert_eq!(detector.compute_trend(), DriftTrend::Declining);
        assert_eq!(
            detector.get_recommendation(),
            DriftRecommendation::UserIntervention
        );
        println!("[PASS] test_recommendation_user_intervention");
    }

    #[test]
    fn test_recommendation_adjust_thresholds() {
        let mut detector = DriftDetector::new().with_baseline(0.80).with_min_samples(5);

        // Create moderate declining drift
        for i in 0..10 {
            let alignment = 0.75 - (i as f32 * 0.005);
            detector.add_observation(alignment, i * 1000);
        }

        assert_eq!(detector.detect_drift(), DriftSeverity::Moderate);
        assert_eq!(detector.compute_trend(), DriftTrend::Declining);
        assert_eq!(
            detector.get_recommendation(),
            DriftRecommendation::AdjustThresholds
        );
        println!("[PASS] test_recommendation_adjust_thresholds");
    }

    #[test]
    fn test_reset_baseline() {
        let mut detector = DriftDetector::new().with_baseline(0.80);

        // Create drift
        for i in 0..5 {
            detector.add_observation(0.70, i * 1000);
        }

        assert_eq!(detector.detect_drift(), DriftSeverity::Severe);

        // Reset baseline to current mean
        detector.reset_baseline();

        assert!((detector.baseline() - 0.70).abs() < f32::EPSILON);
        assert_eq!(detector.detect_drift(), DriftSeverity::None);
        println!("[PASS] test_reset_baseline");
    }

    #[test]
    fn test_window_trimming() {
        let config = DriftConfig {
            window_days: 1, // 1 day window
            ..Default::default()
        };
        let mut detector = DriftDetector::with_config(config);

        // Add observations spread across 2 days (in milliseconds)
        let day_ms = 24 * 60 * 60 * 1000u64;

        // Old observations (before window)
        for i in 0..5 {
            detector.add_observation(0.50, i * 1000);
        }

        // Jump forward past window
        let current_time = 2 * day_ms;
        for i in 0..5 {
            detector.add_observation(0.80, current_time + i * 1000);
        }

        // Only recent observations should remain
        assert_eq!(detector.data_point_count(), 5);

        // Mean should be based only on recent observations
        assert!(
            (detector.rolling_mean() - 0.80).abs() < f32::EPSILON,
            "Mean should be 0.80, got {}",
            detector.rolling_mean()
        );
        println!("[PASS] test_window_trimming");
    }

    #[test]
    fn test_linear_regression_slope_positive() {
        let detector = DriftDetector::new();

        // Clearly increasing: 0.1, 0.2, 0.3, 0.4, 0.5
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let slope = detector.compute_slope(&values);

        // Slope should be 0.1 per step
        assert!(
            (slope - 0.1).abs() < 1e-6,
            "Expected slope 0.1, got {}",
            slope
        );
        println!("[PASS] test_linear_regression_slope_positive");
    }

    #[test]
    fn test_linear_regression_slope_negative() {
        let detector = DriftDetector::new();

        // Clearly decreasing: 0.5, 0.4, 0.3, 0.2, 0.1
        let values = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let slope = detector.compute_slope(&values);

        // Slope should be -0.1 per step
        assert!(
            (slope - (-0.1)).abs() < 1e-6,
            "Expected slope -0.1, got {}",
            slope
        );
        println!("[PASS] test_linear_regression_slope_negative");
    }

    #[test]
    fn test_linear_regression_slope_zero() {
        let detector = DriftDetector::new();

        // Flat: all same values
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let slope = detector.compute_slope(&values);

        assert!(
            slope.abs() < f32::EPSILON,
            "Expected slope 0, got {}",
            slope
        );
        println!("[PASS] test_linear_regression_slope_zero");
    }

    #[test]
    fn test_linear_regression_single_value() {
        let detector = DriftDetector::new();
        let values = vec![0.5];
        let slope = detector.compute_slope(&values);
        assert!((slope - 0.0).abs() < f32::EPSILON);
        println!("[PASS] test_linear_regression_single_value");
    }

    #[test]
    fn test_continuous_monitoring_flag() {
        let detector_continuous = DriftDetector::new();
        assert!(detector_continuous.is_continuous_monitoring());

        let config = DriftConfig {
            monitoring: DriftMonitoring::Manual,
            ..Default::default()
        };
        let detector_manual = DriftDetector::with_config(config);
        assert!(!detector_manual.is_continuous_monitoring());
        println!("[PASS] test_continuous_monitoring_flag");
    }

    #[test]
    fn test_negative_drift_detection() {
        // When alignment improves beyond baseline
        let mut detector = DriftDetector::new().with_baseline(0.70);
        detector.add_observation(0.85, 1000);

        // Drift score should still be positive (absolute value)
        assert!(
            (detector.get_drift_score() - 0.15).abs() < f32::EPSILON,
            "Drift score should be 0.15, got {}",
            detector.get_drift_score()
        );

        // Severity is based on absolute drift
        assert_eq!(detector.detect_drift(), DriftSeverity::Severe);
        println!("[PASS] test_negative_drift_detection");
    }

    #[test]
    fn test_custom_thresholds() {
        let config = DriftConfig {
            alert_threshold: 0.02,
            severe_threshold: 0.05,
            ..Default::default()
        };
        let mut detector = DriftDetector::with_config(config).with_baseline(0.80);

        // 0.03 drift should be moderate with custom thresholds
        detector.add_observation(0.77, 1000);
        assert_eq!(detector.detect_drift(), DriftSeverity::Moderate);

        // 0.06 drift should be severe with custom thresholds
        let mut detector2 = DriftDetector::with_config(DriftConfig {
            alert_threshold: 0.02,
            severe_threshold: 0.05,
            ..Default::default()
        })
        .with_baseline(0.80);
        detector2.add_observation(0.74, 1000);
        assert_eq!(detector2.detect_drift(), DriftSeverity::Severe);
        println!("[PASS] test_custom_thresholds");
    }

    #[test]
    fn test_statistical_precision() {
        let mut detector = DriftDetector::new().with_baseline(0.50);

        // Add many observations to test numerical stability
        for i in 0..1000 {
            let alignment = 0.50 + 0.001 * (i % 10) as f32;
            detector.add_observation(alignment, i * 1000);
        }

        // Mean should be close to 0.5045 (average of 0.50 to 0.509)
        let expected_mean = 0.5045;
        assert!(
            (detector.rolling_mean() - expected_mean).abs() < 0.001,
            "Expected mean ~{}, got {}",
            expected_mean,
            detector.rolling_mean()
        );
        println!("[PASS] test_statistical_precision");
    }

    #[test]
    fn test_data_point_delta_from_mean() {
        let mut detector = DriftDetector::new().with_baseline(0.75);

        detector.add_observation(0.80, 1000);

        let point = detector.state().data_points.front().unwrap();
        // First observation: delta = 0.80 - 0.75 (initial mean) = 0.05
        assert!(
            (point.delta_from_mean - 0.05).abs() < f32::EPSILON,
            "Expected delta 0.05, got {}",
            point.delta_from_mean
        );
        println!("[PASS] test_data_point_delta_from_mean");
    }

    #[test]
    fn test_state_serialization() {
        let mut detector = DriftDetector::new().with_baseline(0.80);
        detector.add_observation(0.75, 1000);
        detector.add_observation(0.70, 2000);

        let state = detector.state();
        let json = serde_json::to_string(state).expect("Failed to serialize state");
        let deserialized: DetectorState =
            serde_json::from_str(&json).expect("Failed to deserialize state");

        assert!((deserialized.rolling_mean - state.rolling_mean).abs() < f32::EPSILON);
        assert_eq!(deserialized.data_points.len(), state.data_points.len());
        println!("[PASS] test_state_serialization");
    }

    #[test]
    fn test_recommendation_serialization() {
        let recommendations = vec![
            DriftRecommendation::NoAction,
            DriftRecommendation::Monitor,
            DriftRecommendation::ReviewMemories,
            DriftRecommendation::AdjustThresholds,
            DriftRecommendation::RecalibrateBaseline,
            DriftRecommendation::UserIntervention,
        ];

        for rec in recommendations {
            let json = serde_json::to_string(&rec).expect("Failed to serialize");
            let deserialized: DriftRecommendation =
                serde_json::from_str(&json).expect("Failed to deserialize");
            assert_eq!(rec, deserialized);
        }
        println!("[PASS] test_recommendation_serialization");
    }

    #[test]
    fn test_boundary_alignment_values() {
        let mut detector = DriftDetector::new();

        // Test boundary values
        detector.add_observation(0.0, 1000);
        assert!((detector.state().current_alignment - 0.0).abs() < f32::EPSILON);

        detector.add_observation(1.0, 2000);
        assert!((detector.state().current_alignment - 1.0).abs() < f32::EPSILON);
        println!("[PASS] test_boundary_alignment_values");
    }

    #[test]
    fn test_full_scenario_improving_from_severe() {
        let mut detector = DriftDetector::new().with_baseline(0.80).with_min_samples(3);

        // Start with severe drift
        detector.add_observation(0.60, 1000);
        detector.add_observation(0.60, 2000);
        detector.add_observation(0.60, 3000);
        assert_eq!(detector.detect_drift(), DriftSeverity::Severe);

        // Start improving
        for i in 4..15 {
            let alignment = 0.60 + ((i - 3) as f32 * 0.02);
            detector.add_observation(alignment.min(0.80), i * 1000);
        }

        // Trend should be improving
        assert_eq!(detector.compute_trend(), DriftTrend::Improving);
        println!("[PASS] test_full_scenario_improving_from_severe");
    }

    #[test]
    fn test_all_severity_levels_accessible() {
        // Verify each severity level is reachable through actual observations
        let config = DriftConfig::default();

        // None
        let mut d = DriftDetector::with_config(config.clone()).with_baseline(0.75);
        d.add_observation(0.75, 1000);
        assert_eq!(d.detect_drift(), DriftSeverity::None);

        // Mild (> 0.01, < 0.05)
        let mut d = DriftDetector::with_config(config.clone()).with_baseline(0.75);
        d.add_observation(0.73, 1000); // 0.02 drift
        assert_eq!(d.detect_drift(), DriftSeverity::Mild);

        // Moderate (>= 0.05, < 0.10)
        let mut d = DriftDetector::with_config(config.clone()).with_baseline(0.75);
        d.add_observation(0.68, 1000); // 0.07 drift
        assert_eq!(d.detect_drift(), DriftSeverity::Moderate);

        // Severe (>= 0.10)
        let mut d = DriftDetector::with_config(config).with_baseline(0.75);
        d.add_observation(0.60, 1000); // 0.15 drift
        assert_eq!(d.detect_drift(), DriftSeverity::Severe);

        println!("[PASS] test_all_severity_levels_accessible");
    }
}
