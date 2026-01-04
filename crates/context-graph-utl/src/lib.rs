//! UTL (Unified Theory of Learning) computation engine for Context Graph.
//!
//! This crate provides the core UTL computation engine that implements the
//! learning score formula: `L = f((ΔS × ΔC) · wₑ · cos φ)`
//!
//! # Modules
//!
//! - [`config`]: Configuration types for all UTL subsystems
//! - [`error`]: Error types and result aliases
//! - [`surprise`]: Surprise (ΔS) computation using KL divergence and embedding distance
//! - [`coherence`]: Coherence (ΔC) computation with structural and graph-based metrics
//! - [`emotional`]: Emotional weight (wₑ) computation with lexicon and arousal/valence
//! - [`phase`]: Phase oscillation (φ) and consolidation detection (NREM/REM)
//! - [`johari`]: Johari quadrant classification and retrieval weighting
//! - [`lifecycle`]: Lifecycle stage management with Marblestone lambda weights
//!
//! # Constitution Reference
//!
//! The UTL formula is defined as:
//! - `ΔS`: Entropy/novelty change in range `[0, 1]`
//! - `ΔC`: Coherence/understanding change in range `[0, 1]`
//! - `wₑ`: Emotional weight in range `[0.5, 1.5]`
//! - `φ`: Phase synchronization angle in range `[0, π]`
//!
//! # Example
//!
//! ```
//! use context_graph_utl::{compute_learning_magnitude, UtlConfig};
//!
//! // Compute learning magnitude from UTL components
//! let learning = compute_learning_magnitude(0.7, 0.6, 1.2, 0.0);
//! assert!(learning >= 0.0 && learning <= 1.0);
//!
//! // Or use the configuration-based approach
//! let config = UtlConfig::default();
//! assert!(config.validate().is_ok());
//! ```

pub mod config;
pub mod error;
pub mod surprise;

// Core UTL modules
pub mod coherence;
pub mod emotional;
pub mod johari;
pub mod lifecycle;
pub mod metrics;
pub mod phase;
pub mod processor;

// Re-export commonly used types from this crate
pub use config::UtlConfig;
pub use error::{UtlError, UtlResult};

// Re-export core types from context-graph-core (DO NOT DUPLICATE)
pub use context_graph_core::types::{EmotionalState, UtlContext, UtlMetrics};

// Re-export lifecycle types for convenience
pub use lifecycle::{LifecycleLambdaWeights, LifecycleManager, LifecycleStage};

// Re-export johari types for convenience
pub use johari::{classify_quadrant, JohariClassifier, JohariQuadrant, SuggestedAction};

// Re-export phase types for convenience
pub use phase::{ConsolidationPhase, PhaseDetector, PhaseOscillator};

// Re-export processor types for convenience
pub use processor::{SessionContext, UtlProcessor};

// Re-export metrics types for convenience
pub use metrics::{
    QuadrantDistribution, StageThresholds, ThresholdsResponse, UtlComputationMetrics, UtlStatus,
    UtlStatusResponse,
};

/// Compute the learning magnitude using the UTL formula.
///
/// Implements: `L = (ΔS × ΔC) · wₑ · cos φ`
///
/// # Arguments
///
/// * `delta_s` - Surprise/entropy change in range `[0.0, 1.0]`
/// * `delta_c` - Coherence change in range `[0.0, 1.0]`
/// * `w_e` - Emotional weight in range `[0.5, 1.5]`
/// * `phi` - Phase angle in radians, range `[0, π]`
///
/// # Returns
///
/// Learning magnitude clamped to `[0.0, 1.0]`
///
/// # Example
///
/// ```
/// use context_graph_utl::compute_learning_magnitude;
///
/// // High surprise, high coherence, engaged emotional state, synchronized phase
/// let learning = compute_learning_magnitude(0.8, 0.7, 1.2, 0.0);
/// assert!(learning > 0.5); // Strong learning signal
///
/// // Low surprise, low coherence - minimal learning
/// let learning = compute_learning_magnitude(0.1, 0.1, 1.0, 0.0);
/// assert!(learning < 0.1);
///
/// // Anti-phase (φ = π) - learning suppressed
/// let learning = compute_learning_magnitude(0.8, 0.7, 1.2, std::f32::consts::PI);
/// assert!(learning < 0.0 || learning == 0.0); // Negative or clamped to 0
/// ```
#[inline]
pub fn compute_learning_magnitude(delta_s: f32, delta_c: f32, w_e: f32, phi: f32) -> f32 {
    let raw = (delta_s * delta_c) * w_e * phi.cos();
    raw.clamp(0.0, 1.0)
}

/// Compute learning magnitude with validation.
///
/// Like [`compute_learning_magnitude`] but validates input ranges
/// and returns an error for invalid inputs.
///
/// # Arguments
///
/// * `delta_s` - Surprise/entropy change in range `[0.0, 1.0]`
/// * `delta_c` - Coherence change in range `[0.0, 1.0]`
/// * `w_e` - Emotional weight in range `[0.5, 1.5]`
/// * `phi` - Phase angle in radians, range `[0, π]`
///
/// # Returns
///
/// `Ok(f32)` with learning magnitude, or `Err(UtlError)` if inputs invalid.
///
/// # Example
///
/// ```
/// use context_graph_utl::compute_learning_magnitude_validated;
///
/// // Valid inputs
/// let result = compute_learning_magnitude_validated(0.5, 0.6, 1.0, 0.0);
/// assert!(result.is_ok());
///
/// // Invalid delta_s (out of range)
/// let result = compute_learning_magnitude_validated(1.5, 0.6, 1.0, 0.0);
/// assert!(result.is_err());
/// ```
pub fn compute_learning_magnitude_validated(
    delta_s: f32,
    delta_c: f32,
    w_e: f32,
    phi: f32,
) -> UtlResult<f32> {
    // Validate delta_s range
    if !(0.0..=1.0).contains(&delta_s) {
        return Err(UtlError::InvalidParameter {
            name: "delta_s".to_string(),
            value: delta_s.to_string(),
            reason: "Must be in range [0.0, 1.0]".to_string(),
        });
    }

    // Validate delta_c range
    if !(0.0..=1.0).contains(&delta_c) {
        return Err(UtlError::InvalidParameter {
            name: "delta_c".to_string(),
            value: delta_c.to_string(),
            reason: "Must be in range [0.0, 1.0]".to_string(),
        });
    }

    // Validate emotional weight range (constitution: [0.5, 1.5])
    if !(0.5..=1.5).contains(&w_e) {
        return Err(UtlError::InvalidParameter {
            name: "w_e".to_string(),
            value: w_e.to_string(),
            reason: "Must be in range [0.5, 1.5]".to_string(),
        });
    }

    // Validate phase angle range (constitution: [0, π])
    if !(0.0..=std::f32::consts::PI).contains(&phi) {
        return Err(UtlError::InvalidParameter {
            name: "phi".to_string(),
            value: phi.to_string(),
            reason: "Must be in range [0, π]".to_string(),
        });
    }

    let raw = (delta_s * delta_c) * w_e * phi.cos();

    // Check for NaN/Infinity
    if raw.is_nan() {
        return Err(UtlError::nan_result(delta_s, delta_c, w_e, phi));
    }
    if raw.is_infinite() {
        return Err(UtlError::infinite_result(delta_s, delta_c, w_e, phi));
    }

    Ok(raw.clamp(0.0, 1.0))
}

// ============= LearningIntensity, LearningSignal, UtlState =============

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Learning intensity category for quick classification.
///
/// Categorizes learning magnitude into Low, Medium, or High buckets
/// for efficient filtering and prioritization.
///
/// # Constitution Reference
///
/// Thresholds based on UTL learning score ranges:
/// - Low: < 0.3 (minimal learning potential)
/// - Medium: 0.3 - 0.7 (moderate learning)
/// - High: > 0.7 (strong learning signal)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum LearningIntensity {
    /// Learning magnitude < 0.3
    Low = 0,
    /// Learning magnitude 0.3 - 0.7
    Medium = 1,
    /// Learning magnitude > 0.7
    High = 2,
}

impl LearningIntensity {
    /// Returns all intensity variants.
    pub fn all() -> [LearningIntensity; 3] {
        [Self::Low, Self::Medium, Self::High]
    }

    /// Returns a human-readable description of this intensity.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Low => "Low learning potential (magnitude < 0.3)",
            Self::Medium => "Moderate learning (magnitude 0.3 - 0.7)",
            Self::High => "Strong learning signal (magnitude > 0.7)",
        }
    }
}

impl std::fmt::Display for LearningIntensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
        }
    }
}

/// Complete UTL computation output with all component values and derived decisions.
///
/// This struct captures the full result of a UTL computation including:
/// - Raw component values (delta_s, delta_c, w_e, phi)
/// - Computed learning magnitude
/// - Johari classification and suggested action
/// - Storage/consolidation recommendations
/// - Performance metrics (latency)
///
/// # Constitution Reference
///
/// - ΔS: [0,1] entropy/novelty (constitution.yaml:154)
/// - ΔC: [0,1] coherence/understanding (constitution.yaml:155)
/// - wₑ: [0.5,1.5] emotional weight (constitution.yaml:156)
/// - φ: [0,π] phase sync (constitution.yaml:157)
///
/// # Example
///
/// ```
/// use context_graph_utl::{LearningSignal, JohariQuadrant, SuggestedAction};
///
/// let signal = LearningSignal::new(
///     0.7, 0.6, 0.8, 1.2, 0.5,
///     None,
///     JohariQuadrant::Open,
///     SuggestedAction::DirectRecall,
///     false, true, 1500,
/// ).expect("Valid signal");
///
/// assert!(signal.magnitude >= 0.0 && signal.magnitude <= 1.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSignal {
    /// Learning magnitude L in [0, 1], computed from formula
    pub magnitude: f32,

    /// Surprise (entropy) value [0, 1]
    pub delta_s: f32,

    /// Coherence value [0, 1]
    pub delta_c: f32,

    /// Emotional weight [0.5, 1.5]
    pub w_e: f32,

    /// Phase angle [0, PI]
    pub phi: f32,

    /// Marblestone lambda weights if lifecycle-adjusted
    pub lambda_weights: Option<LifecycleLambdaWeights>,

    /// Classified Johari quadrant
    pub quadrant: JohariQuadrant,

    /// Recommended retrieval action based on quadrant
    pub suggested_action: SuggestedAction,

    /// Whether consolidation is recommended (magnitude > 0.6)
    pub should_consolidate: bool,

    /// Whether storage is recommended (magnitude > 0.3)
    pub should_store: bool,

    /// When computation was performed (UTC)
    pub timestamp: DateTime<Utc>,

    /// Computation time in microseconds for performance tracking
    pub latency_us: u64,
}

impl LearningSignal {
    /// Create a new LearningSignal with validation.
    ///
    /// # Arguments
    /// * `magnitude` - Pre-computed learning magnitude [0, 1]
    /// * `delta_s` - Surprise component [0, 1]
    /// * `delta_c` - Coherence component [0, 1]
    /// * `w_e` - Emotional weight [0.5, 1.5]
    /// * `phi` - Phase angle [0, π]
    /// * `lambda_weights` - Optional lifecycle weights
    /// * `quadrant` - Johari classification
    /// * `suggested_action` - Recommended action
    /// * `should_consolidate` - Consolidation flag
    /// * `should_store` - Storage flag
    /// * `latency_us` - Computation time in microseconds
    ///
    /// # Returns
    /// `Ok(LearningSignal)` if valid, `Err(UtlError)` if magnitude is NaN/Infinity
    ///
    /// # Errors
    /// Returns `UtlError::InvalidComputation` if magnitude is NaN or Infinity
    pub fn new(
        magnitude: f32,
        delta_s: f32,
        delta_c: f32,
        w_e: f32,
        phi: f32,
        lambda_weights: Option<LifecycleLambdaWeights>,
        quadrant: JohariQuadrant,
        suggested_action: SuggestedAction,
        should_consolidate: bool,
        should_store: bool,
        latency_us: u64,
    ) -> UtlResult<Self> {
        let signal = Self {
            magnitude,
            delta_s,
            delta_c,
            w_e,
            phi,
            lambda_weights,
            quadrant,
            suggested_action,
            should_consolidate,
            should_store,
            timestamp: Utc::now(),
            latency_us,
        };

        signal.validate()?;
        Ok(signal)
    }

    /// Validate that magnitude and components are finite (not NaN or Infinity).
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(UtlError::InvalidComputation)` otherwise
    pub fn validate(&self) -> UtlResult<()> {
        if self.magnitude.is_nan() {
            return Err(UtlError::InvalidComputation {
                delta_s: self.delta_s,
                delta_c: self.delta_c,
                w_e: self.w_e,
                phi: self.phi,
                reason: "magnitude is NaN".to_string(),
            });
        }

        if self.magnitude.is_infinite() {
            return Err(UtlError::InvalidComputation {
                delta_s: self.delta_s,
                delta_c: self.delta_c,
                w_e: self.w_e,
                phi: self.phi,
                reason: "magnitude is Infinity".to_string(),
            });
        }

        // Also check component values for NaN (fail fast)
        if self.delta_s.is_nan()
            || self.delta_c.is_nan()
            || self.w_e.is_nan()
            || self.phi.is_nan()
        {
            return Err(UtlError::InvalidComputation {
                delta_s: self.delta_s,
                delta_c: self.delta_c,
                w_e: self.w_e,
                phi: self.phi,
                reason: "component value is NaN".to_string(),
            });
        }

        Ok(())
    }

    /// Check if this signal indicates high learning potential (> 0.7)
    #[inline]
    pub fn is_high_learning(&self) -> bool {
        self.magnitude > 0.7
    }

    /// Check if this signal indicates low learning potential (< 0.3)
    #[inline]
    pub fn is_low_learning(&self) -> bool {
        self.magnitude < 0.3
    }

    /// Get the learning intensity category
    #[inline]
    pub fn intensity_category(&self) -> LearningIntensity {
        if self.magnitude < 0.3 {
            LearningIntensity::Low
        } else if self.magnitude < 0.7 {
            LearningIntensity::Medium
        } else {
            LearningIntensity::High
        }
    }
}

/// Compact UTL state for storage in MemoryNode.
///
/// This is the persistent representation stored with each memory node,
/// containing only the essential values needed for retrieval decisions.
/// Use `UtlState::from_signal()` to convert from a full `LearningSignal`.
///
/// # Example
///
/// ```
/// use context_graph_utl::{LearningSignal, UtlState, JohariQuadrant, SuggestedAction};
///
/// let signal = LearningSignal::new(
///     0.7, 0.6, 0.8, 1.2, 0.5, None,
///     JohariQuadrant::Blind, SuggestedAction::TriggerDream,
///     true, true, 2000,
/// ).unwrap();
///
/// let state = UtlState::from_signal(&signal);
/// assert_eq!(state.learning_magnitude, 0.7);
/// assert_eq!(state.quadrant, JohariQuadrant::Blind);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtlState {
    /// Last computed surprise value [0, 1]
    pub delta_s: f32,

    /// Last computed coherence value [0, 1]
    pub delta_c: f32,

    /// Last computed emotional weight [0.5, 1.5]
    pub w_e: f32,

    /// Phase angle at computation [0, PI]
    pub phi: f32,

    /// Computed learning magnitude [0, 1]
    pub learning_magnitude: f32,

    /// Classified Johari quadrant
    pub quadrant: JohariQuadrant,

    /// When state was last updated (UTC)
    pub last_computed: DateTime<Utc>,
}

impl UtlState {
    /// Create a new UtlState from a LearningSignal.
    ///
    /// Extracts the essential values for persistent storage.
    pub fn from_signal(signal: &LearningSignal) -> Self {
        Self {
            delta_s: signal.delta_s,
            delta_c: signal.delta_c,
            w_e: signal.w_e,
            phi: signal.phi,
            learning_magnitude: signal.magnitude,
            quadrant: signal.quadrant,
            last_computed: signal.timestamp,
        }
    }

    /// Create a default/empty UtlState for new nodes.
    ///
    /// Uses neutral defaults:
    /// - delta_s = 0.0 (no surprise)
    /// - delta_c = 0.0 (no coherence established)
    /// - w_e = 1.0 (neutral emotional state)
    /// - phi = 0.0 (synchronized)
    /// - learning_magnitude = 0.5 (medium baseline)
    /// - quadrant = Hidden (low surprise, low coherence)
    pub fn empty() -> Self {
        Self {
            delta_s: 0.0,
            delta_c: 0.0,
            w_e: 1.0,
            phi: 0.0,
            learning_magnitude: 0.5,
            quadrant: JohariQuadrant::Hidden,
            last_computed: Utc::now(),
        }
    }

    /// Validate that all values are finite (not NaN or Infinity).
    ///
    /// # Returns
    /// `Ok(())` if all values are finite, `Err(UtlError::InvalidComputation)` otherwise
    pub fn validate(&self) -> UtlResult<()> {
        let values = [
            ("delta_s", self.delta_s),
            ("delta_c", self.delta_c),
            ("w_e", self.w_e),
            ("phi", self.phi),
            ("learning_magnitude", self.learning_magnitude),
        ];

        for (name, value) in values {
            if value.is_nan() || value.is_infinite() {
                return Err(UtlError::InvalidComputation {
                    delta_s: self.delta_s,
                    delta_c: self.delta_c,
                    w_e: self.w_e,
                    phi: self.phi,
                    reason: format!("{} is NaN or Infinity", name),
                });
            }
        }

        Ok(())
    }

    /// Check if this state is stale (not computed recently).
    ///
    /// # Arguments
    /// * `max_age_seconds` - Maximum age in seconds before considered stale
    ///
    /// # Returns
    /// `true` if state is older than `max_age_seconds`
    pub fn is_stale(&self, max_age_seconds: i64) -> bool {
        let age = Utc::now() - self.last_computed;
        age.num_seconds() > max_age_seconds
    }

    /// Get the age of this state in seconds.
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.last_computed).num_seconds()
    }
}

impl Default for UtlState {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_learning_magnitude_basic() {
        // Perfect conditions: high surprise, high coherence, engaged, synchronized
        let learning = compute_learning_magnitude(0.8, 0.7, 1.2, 0.0);
        // (0.8 * 0.7) * 1.2 * cos(0) = 0.56 * 1.2 * 1.0 = 0.672
        assert!((learning - 0.672).abs() < 0.001);
    }

    #[test]
    fn test_compute_learning_magnitude_zero_inputs() {
        // Zero surprise -> zero learning
        let learning = compute_learning_magnitude(0.0, 0.7, 1.2, 0.0);
        assert_eq!(learning, 0.0);

        // Zero coherence -> zero learning
        let learning = compute_learning_magnitude(0.8, 0.0, 1.2, 0.0);
        assert_eq!(learning, 0.0);
    }

    #[test]
    fn test_compute_learning_magnitude_anti_phase() {
        // Anti-phase (φ = π) -> cos(π) = -1, so negative learning (clamped to 0)
        let learning = compute_learning_magnitude(0.8, 0.7, 1.2, std::f32::consts::PI);
        assert_eq!(learning, 0.0); // Clamped to 0
    }

    #[test]
    fn test_compute_learning_magnitude_half_phase() {
        // φ = π/2 -> cos(π/2) = 0, so zero learning
        let learning = compute_learning_magnitude(0.8, 0.7, 1.2, std::f32::consts::FRAC_PI_2);
        assert!(learning.abs() < 0.001);
    }

    #[test]
    fn test_compute_learning_magnitude_clamping() {
        // Very high values that would exceed 1.0
        let learning = compute_learning_magnitude(1.0, 1.0, 1.5, 0.0);
        // (1.0 * 1.0) * 1.5 * 1.0 = 1.5, clamped to 1.0
        assert_eq!(learning, 1.0);
    }

    #[test]
    fn test_compute_learning_magnitude_validated_valid() {
        let result = compute_learning_magnitude_validated(0.5, 0.6, 1.0, 0.0);
        assert!(result.is_ok());
        let learning = result.unwrap();
        // (0.5 * 0.6) * 1.0 * 1.0 = 0.3
        assert!((learning - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_compute_learning_magnitude_validated_invalid_delta_s() {
        let result = compute_learning_magnitude_validated(1.5, 0.6, 1.0, 0.0);
        assert!(result.is_err());
        match result.unwrap_err() {
            UtlError::InvalidParameter { name, .. } => assert_eq!(name, "delta_s"),
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_compute_learning_magnitude_validated_invalid_delta_c() {
        let result = compute_learning_magnitude_validated(0.5, -0.1, 1.0, 0.0);
        assert!(result.is_err());
        match result.unwrap_err() {
            UtlError::InvalidParameter { name, .. } => assert_eq!(name, "delta_c"),
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_compute_learning_magnitude_validated_invalid_w_e() {
        let result = compute_learning_magnitude_validated(0.5, 0.6, 0.3, 0.0);
        assert!(result.is_err());
        match result.unwrap_err() {
            UtlError::InvalidParameter { name, .. } => assert_eq!(name, "w_e"),
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_compute_learning_magnitude_validated_invalid_phi() {
        let result = compute_learning_magnitude_validated(0.5, 0.6, 1.0, 4.0);
        assert!(result.is_err());
        match result.unwrap_err() {
            UtlError::InvalidParameter { name, .. } => assert_eq!(name, "phi"),
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_re_exports_exist() {
        // Verify core re-exports are accessible
        let _metrics = UtlMetrics::default();
        let _context = UtlContext::default();
        let _state = EmotionalState::default();
        let _quadrant = JohariQuadrant::default();
    }

    #[test]
    fn test_lifecycle_re_exports() {
        let stage = LifecycleStage::from_interaction_count(100);
        assert_eq!(stage, LifecycleStage::Growth);

        let weights = LifecycleLambdaWeights::for_stage(stage);
        assert!((weights.lambda_s() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_johari_re_exports() {
        let quadrant = classify_quadrant(0.3, 0.7);
        assert_eq!(quadrant, JohariQuadrant::Open);
    }

    #[test]
    fn test_phase_re_exports() {
        let detector = PhaseDetector::new();
        let phase = detector.detect_phase(0.3);
        assert!(matches!(
            phase,
            ConsolidationPhase::NREM | ConsolidationPhase::REM | ConsolidationPhase::Wake
        ));
    }

    #[test]
    fn test_emotional_state_weight_modifier() {
        assert_eq!(EmotionalState::Neutral.weight_modifier(), 1.0);
        assert_eq!(EmotionalState::Focused.weight_modifier(), 1.3);
        assert_eq!(EmotionalState::Fatigued.weight_modifier(), 0.6);
    }

    #[test]
    fn test_config_validation() {
        let config = UtlConfig::default();
        assert!(config.validate().is_ok());
    }

    // =============================================================================
    // LearningSignal Tests
    // =============================================================================

    #[test]
    fn test_learning_signal_creation_valid() {
        let signal = LearningSignal::new(
            0.7,
            0.6,
            0.8,
            1.2,
            0.5,
            None,
            JohariQuadrant::Open,
            SuggestedAction::DirectRecall,
            true,
            true,
            1500,
        );
        assert!(signal.is_ok());
        let signal = signal.unwrap();
        assert_eq!(signal.magnitude, 0.7);
        assert_eq!(signal.delta_s, 0.6);
        assert_eq!(signal.delta_c, 0.8);
        assert_eq!(signal.w_e, 1.2);
        assert_eq!(signal.phi, 0.5);
        assert!(signal.lambda_weights.is_none());
        assert_eq!(signal.quadrant, JohariQuadrant::Open);
        assert_eq!(signal.suggested_action, SuggestedAction::DirectRecall);
        assert!(signal.should_consolidate);
        assert!(signal.should_store);
        assert_eq!(signal.latency_us, 1500);
    }

    #[test]
    fn test_learning_signal_validation_nan_magnitude() {
        let signal = LearningSignal::new(
            f32::NAN,
            0.5,
            0.5,
            1.0,
            0.0,
            None,
            JohariQuadrant::Hidden,
            SuggestedAction::GetNeighborhood,
            false,
            true,
            100,
        );
        assert!(signal.is_err());
        match signal.unwrap_err() {
            UtlError::InvalidComputation { reason, .. } => {
                assert!(reason.contains("NaN"));
            }
            _ => panic!("Expected InvalidComputation error"),
        }
    }

    #[test]
    fn test_learning_signal_validation_infinity_magnitude() {
        let signal = LearningSignal::new(
            f32::INFINITY,
            0.5,
            0.5,
            1.0,
            0.0,
            None,
            JohariQuadrant::Hidden,
            SuggestedAction::GetNeighborhood,
            false,
            true,
            100,
        );
        assert!(signal.is_err());
        match signal.unwrap_err() {
            UtlError::InvalidComputation { reason, .. } => {
                assert!(reason.contains("Infinity"));
            }
            _ => panic!("Expected InvalidComputation error"),
        }
    }

    #[test]
    fn test_learning_signal_validation_nan_component() {
        let signal = LearningSignal::new(
            0.5,
            f32::NAN, // delta_s is NaN
            0.5,
            1.0,
            0.0,
            None,
            JohariQuadrant::Hidden,
            SuggestedAction::GetNeighborhood,
            false,
            true,
            100,
        );
        assert!(signal.is_err());
        match signal.unwrap_err() {
            UtlError::InvalidComputation { reason, .. } => {
                assert!(reason.contains("NaN"));
            }
            _ => panic!("Expected InvalidComputation error"),
        }
    }

    #[test]
    fn test_learning_intensity_categories() {
        // Test all three categories
        assert_eq!(LearningIntensity::Low as u8, 0);
        assert_eq!(LearningIntensity::Medium as u8, 1);
        assert_eq!(LearningIntensity::High as u8, 2);

        // Test all() helper
        let all = LearningIntensity::all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], LearningIntensity::Low);
        assert_eq!(all[1], LearningIntensity::Medium);
        assert_eq!(all[2], LearningIntensity::High);

        // Test description()
        assert!(LearningIntensity::Low.description().contains("Low"));
        assert!(LearningIntensity::Medium.description().contains("Moderate"));
        assert!(LearningIntensity::High.description().contains("Strong"));

        // Test Display
        assert_eq!(format!("{}", LearningIntensity::Low), "Low");
        assert_eq!(format!("{}", LearningIntensity::Medium), "Medium");
        assert_eq!(format!("{}", LearningIntensity::High), "High");
    }

    #[test]
    fn test_learning_intensity_boundary_values() {
        // Low: < 0.3
        let low_signal = LearningSignal::new(
            0.29, 0.5, 0.5, 1.0, 0.0, None,
            JohariQuadrant::Hidden, SuggestedAction::GetNeighborhood,
            false, false, 100,
        ).unwrap();
        assert_eq!(low_signal.intensity_category(), LearningIntensity::Low);
        assert!(low_signal.is_low_learning());
        assert!(!low_signal.is_high_learning());

        // Medium: 0.3 - 0.7
        let med_signal = LearningSignal::new(
            0.5, 0.5, 0.5, 1.0, 0.0, None,
            JohariQuadrant::Hidden, SuggestedAction::GetNeighborhood,
            false, true, 100,
        ).unwrap();
        assert_eq!(med_signal.intensity_category(), LearningIntensity::Medium);
        assert!(!med_signal.is_low_learning());
        assert!(!med_signal.is_high_learning());

        // High: > 0.7
        let high_signal = LearningSignal::new(
            0.85, 0.5, 0.5, 1.0, 0.0, None,
            JohariQuadrant::Open, SuggestedAction::DirectRecall,
            true, true, 100,
        ).unwrap();
        assert_eq!(high_signal.intensity_category(), LearningIntensity::High);
        assert!(!high_signal.is_low_learning());
        assert!(high_signal.is_high_learning());

        // Boundary at 0.3 (should be Medium)
        let boundary_low = LearningSignal::new(
            0.3, 0.5, 0.5, 1.0, 0.0, None,
            JohariQuadrant::Hidden, SuggestedAction::GetNeighborhood,
            false, true, 100,
        ).unwrap();
        assert_eq!(boundary_low.intensity_category(), LearningIntensity::Medium);

        // Boundary at 0.7 (should be High since magnitude >= 0.7)
        let boundary_high = LearningSignal::new(
            0.7, 0.5, 0.5, 1.0, 0.0, None,
            JohariQuadrant::Open, SuggestedAction::DirectRecall,
            true, true, 100,
        ).unwrap();
        assert_eq!(boundary_high.intensity_category(), LearningIntensity::High);
    }

    // =============================================================================
    // UtlState Tests
    // =============================================================================

    #[test]
    fn test_utl_state_from_signal() {
        let signal = LearningSignal::new(
            0.7, 0.6, 0.8, 1.2, 0.5, None,
            JohariQuadrant::Blind, SuggestedAction::TriggerDream,
            true, true, 2000,
        ).unwrap();

        let state = UtlState::from_signal(&signal);

        assert_eq!(state.delta_s, 0.6);
        assert_eq!(state.delta_c, 0.8);
        assert_eq!(state.w_e, 1.2);
        assert_eq!(state.phi, 0.5);
        assert_eq!(state.learning_magnitude, 0.7);
        assert_eq!(state.quadrant, JohariQuadrant::Blind);
        assert_eq!(state.last_computed, signal.timestamp);
    }

    #[test]
    fn test_utl_state_empty() {
        let state = UtlState::empty();

        assert_eq!(state.delta_s, 0.0);
        assert_eq!(state.delta_c, 0.0);
        assert_eq!(state.w_e, 1.0);
        assert_eq!(state.phi, 0.0);
        assert_eq!(state.learning_magnitude, 0.5);
        assert_eq!(state.quadrant, JohariQuadrant::Hidden);
    }

    #[test]
    fn test_utl_state_default_matches_empty() {
        let default_state = UtlState::default();
        let empty_state = UtlState::empty();

        assert_eq!(default_state.delta_s, empty_state.delta_s);
        assert_eq!(default_state.delta_c, empty_state.delta_c);
        assert_eq!(default_state.w_e, empty_state.w_e);
        assert_eq!(default_state.phi, empty_state.phi);
        assert_eq!(default_state.learning_magnitude, empty_state.learning_magnitude);
        assert_eq!(default_state.quadrant, empty_state.quadrant);
    }

    #[test]
    fn test_utl_state_staleness() {
        // Fresh state should not be stale
        let state = UtlState::empty();
        assert!(!state.is_stale(60)); // Not stale within 60 seconds

        // age_seconds() should be very small (just created)
        let age = state.age_seconds();
        assert!(age >= 0 && age < 2); // Should be 0 or 1 second old
    }

    #[test]
    fn test_utl_state_validation_success() {
        let state = UtlState::empty();
        assert!(state.validate().is_ok());
    }

    #[test]
    fn test_utl_state_validation_nan() {
        let state = UtlState {
            delta_s: f32::NAN,
            delta_c: 0.5,
            w_e: 1.0,
            phi: 0.0,
            learning_magnitude: 0.5,
            quadrant: JohariQuadrant::Hidden,
            last_computed: Utc::now(),
        };
        assert!(state.validate().is_err());
    }

    #[test]
    fn test_utl_state_validation_infinity() {
        let state = UtlState {
            delta_s: 0.5,
            delta_c: f32::INFINITY,
            w_e: 1.0,
            phi: 0.0,
            learning_magnitude: 0.5,
            quadrant: JohariQuadrant::Hidden,
            last_computed: Utc::now(),
        };
        assert!(state.validate().is_err());
    }

    #[test]
    fn test_utl_state_serialization_roundtrip() {
        let original = UtlState {
            delta_s: 0.6,
            delta_c: 0.8,
            w_e: 1.2,
            phi: 0.5,
            learning_magnitude: 0.7,
            quadrant: JohariQuadrant::Blind,
            last_computed: Utc::now(),
        };

        let json = serde_json::to_string(&original).expect("Serialization failed");
        let deserialized: UtlState = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(deserialized.delta_s, original.delta_s);
        assert_eq!(deserialized.delta_c, original.delta_c);
        assert_eq!(deserialized.w_e, original.w_e);
        assert_eq!(deserialized.phi, original.phi);
        assert_eq!(deserialized.learning_magnitude, original.learning_magnitude);
        assert_eq!(deserialized.quadrant, original.quadrant);
    }

    #[test]
    fn test_learning_signal_serialization_roundtrip() {
        let original = LearningSignal::new(
            0.7, 0.6, 0.8, 1.2, 0.5, None,
            JohariQuadrant::Open, SuggestedAction::DirectRecall,
            true, true, 1500,
        ).unwrap();

        let json = serde_json::to_string(&original).expect("Serialization failed");
        let deserialized: LearningSignal = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(deserialized.magnitude, original.magnitude);
        assert_eq!(deserialized.delta_s, original.delta_s);
        assert_eq!(deserialized.delta_c, original.delta_c);
        assert_eq!(deserialized.w_e, original.w_e);
        assert_eq!(deserialized.phi, original.phi);
        assert!(deserialized.lambda_weights.is_none());
        assert_eq!(deserialized.quadrant, original.quadrant);
        assert_eq!(deserialized.suggested_action, original.suggested_action);
        assert_eq!(deserialized.should_consolidate, original.should_consolidate);
        assert_eq!(deserialized.should_store, original.should_store);
        assert_eq!(deserialized.latency_us, original.latency_us);
    }

    #[test]
    fn test_learning_signal_with_lambda_weights() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let signal = LearningSignal::new(
            0.7, 0.6, 0.8, 1.2, 0.5,
            Some(weights.clone()),
            JohariQuadrant::Open, SuggestedAction::DirectRecall,
            true, true, 1500,
        ).unwrap();

        assert!(signal.lambda_weights.is_some());
        let lw = signal.lambda_weights.unwrap();
        assert!((lw.lambda_s() - weights.lambda_s()).abs() < 0.001);
        assert!((lw.lambda_c() - weights.lambda_c()).abs() < 0.001);
    }
}
