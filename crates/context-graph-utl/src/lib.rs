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
pub mod phase;

// Re-export commonly used types from this crate
pub use config::UtlConfig;
pub use error::{UtlError, UtlResult};

// Re-export core types from context-graph-core (DO NOT DUPLICATE)
pub use context_graph_core::types::{EmotionalState, UtlContext, UtlMetrics};

// Re-export lifecycle types for convenience
pub use lifecycle::{LifecycleLambdaWeights, LifecycleManager, LifecycleStage};

// Re-export johari types for convenience
pub use johari::{classify_quadrant, JohariClassifier, JohariQuadrant};

// Re-export phase types for convenience
pub use phase::{ConsolidationPhase, PhaseDetector, PhaseOscillator};

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
}
