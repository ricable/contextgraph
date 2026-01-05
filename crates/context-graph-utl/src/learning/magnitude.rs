//! Learning magnitude computation using the UTL formula.
//!
//! Implements: `L = (ΔS × ΔC) · wₑ · cos φ`

use crate::error::{UtlError, UtlResult};

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
}
