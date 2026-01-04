//! Johari quadrant classifier based on UTL metrics.
//!
//! This module provides classification of knowledge items into Johari Window
//! quadrants based on their surprise (delta_s) and coherence (delta_c) values.
//!
//! # Classification Logic
//!
//! Per constitution.yaml specifications:
//! - **Open**: delta_s < threshold AND delta_c > threshold (low surprise, high coherence)
//! - **Blind**: delta_s > threshold AND delta_c < threshold (high surprise, low coherence)
//! - **Hidden**: delta_s < threshold AND delta_c < threshold (low surprise, low coherence)
//! - **Unknown**: delta_s > threshold AND delta_c > threshold (high surprise, high coherence)

use context_graph_core::types::JohariQuadrant;

use crate::config::JohariConfig;

/// Classifies surprise and coherence values into a Johari quadrant using default thresholds.
///
/// This is a convenience function that uses the default threshold of 0.5 for both
/// surprise and coherence classification.
///
/// # Arguments
///
/// * `delta_s` - Surprise/entropy value in range [0.0, 1.0]
/// * `delta_c` - Coherence/understanding value in range [0.0, 1.0]
///
/// # Returns
///
/// The appropriate `JohariQuadrant` based on the classification rules:
/// - Open: delta_s < 0.5, delta_c > 0.5
/// - Blind: delta_s > 0.5, delta_c < 0.5
/// - Hidden: delta_s < 0.5, delta_c < 0.5
/// - Unknown: delta_s > 0.5, delta_c > 0.5
///
/// # Example
///
/// ```
/// use context_graph_utl::johari::{classify_quadrant, JohariQuadrant};
///
/// // Low surprise, high coherence -> Open
/// assert_eq!(classify_quadrant(0.2, 0.8), JohariQuadrant::Open);
///
/// // High surprise, low coherence -> Blind
/// assert_eq!(classify_quadrant(0.8, 0.2), JohariQuadrant::Blind);
///
/// // Low surprise, low coherence -> Hidden
/// assert_eq!(classify_quadrant(0.2, 0.2), JohariQuadrant::Hidden);
///
/// // High surprise, high coherence -> Unknown
/// assert_eq!(classify_quadrant(0.8, 0.8), JohariQuadrant::Unknown);
/// ```
#[inline]
pub fn classify_quadrant(delta_s: f32, delta_c: f32) -> JohariQuadrant {
    const DEFAULT_THRESHOLD: f32 = 0.5;
    classify_with_thresholds(delta_s, delta_c, DEFAULT_THRESHOLD, DEFAULT_THRESHOLD)
}

/// Classifies surprise and coherence values into a Johari quadrant with custom thresholds.
///
/// This internal function allows specifying custom thresholds for classification.
///
/// # Arguments
///
/// * `delta_s` - Surprise/entropy value in range [0.0, 1.0]
/// * `delta_c` - Coherence/understanding value in range [0.0, 1.0]
/// * `surprise_threshold` - Threshold for surprise classification
/// * `coherence_threshold` - Threshold for coherence classification
#[inline]
fn classify_with_thresholds(
    delta_s: f32,
    delta_c: f32,
    surprise_threshold: f32,
    coherence_threshold: f32,
) -> JohariQuadrant {
    let low_surprise = delta_s < surprise_threshold;
    let high_coherence = delta_c > coherence_threshold;

    match (low_surprise, high_coherence) {
        (true, true) => JohariQuadrant::Open, // Low S, High C -> direct recall
        (false, false) => JohariQuadrant::Blind, // High S, Low C -> discovery
        (true, false) => JohariQuadrant::Hidden, // Low S, Low C -> private
        (false, true) => JohariQuadrant::Unknown, // High S, High C -> frontier
    }
}

/// A configurable Johari quadrant classifier.
///
/// This struct provides a stateful classifier that uses configuration-defined
/// thresholds and optionally supports fuzzy boundary classification.
///
/// # Configuration
///
/// The classifier uses `JohariConfig` to determine:
/// - `surprise_threshold`: The threshold for classifying surprise as high/low
/// - `coherence_threshold`: The threshold for classifying coherence as high/low
/// - `fuzzy_boundaries`: Whether to use fuzzy classification near boundaries
/// - `boundary_width`: The width of the fuzzy boundary region
///
/// # Example
///
/// ```
/// use context_graph_utl::johari::{JohariClassifier, JohariQuadrant};
/// use context_graph_utl::config::JohariConfig;
///
/// let config = JohariConfig::default();
/// let classifier = JohariClassifier::new(&config);
///
/// let quadrant = classifier.classify(0.3, 0.8);
/// assert_eq!(quadrant, JohariQuadrant::Open);
/// ```
#[derive(Debug, Clone)]
pub struct JohariClassifier {
    /// Threshold for surprise classification (default: 0.5)
    surprise_threshold: f32,
    /// Threshold for coherence classification (default: 0.5)
    coherence_threshold: f32,
    /// Whether to use fuzzy boundaries
    fuzzy_boundaries: bool,
    /// Width of fuzzy boundary region
    boundary_width: f32,
}

impl JohariClassifier {
    /// Creates a new classifier with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Johari configuration containing thresholds and boundary settings
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::JohariClassifier;
    /// use context_graph_utl::config::JohariConfig;
    ///
    /// let config = JohariConfig::default();
    /// let classifier = JohariClassifier::new(&config);
    /// ```
    pub fn new(config: &JohariConfig) -> Self {
        Self {
            surprise_threshold: config.surprise_threshold,
            coherence_threshold: config.coherence_threshold,
            fuzzy_boundaries: config.fuzzy_boundaries,
            boundary_width: config.boundary_width,
        }
    }

    /// Creates a classifier with default thresholds (0.5 for both).
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::JohariClassifier;
    ///
    /// let classifier = JohariClassifier::default();
    /// ```
    pub fn with_default_thresholds() -> Self {
        Self {
            surprise_threshold: 0.5,
            coherence_threshold: 0.5,
            fuzzy_boundaries: false,
            boundary_width: 0.1,
        }
    }

    /// Creates a classifier with custom thresholds.
    ///
    /// # Arguments
    ///
    /// * `surprise_threshold` - Threshold for surprise classification
    /// * `coherence_threshold` - Threshold for coherence classification
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::JohariClassifier;
    ///
    /// let classifier = JohariClassifier::with_thresholds(0.4, 0.6);
    /// ```
    pub fn with_thresholds(surprise_threshold: f32, coherence_threshold: f32) -> Self {
        Self {
            surprise_threshold,
            coherence_threshold,
            fuzzy_boundaries: false,
            boundary_width: 0.1,
        }
    }

    /// Enables fuzzy boundary classification.
    ///
    /// When enabled, values near the threshold boundaries will have smoother
    /// transitions between quadrants.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the fuzzy boundary region
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::JohariClassifier;
    ///
    /// let classifier = JohariClassifier::with_default_thresholds()
    ///     .with_fuzzy_boundaries(0.1);
    /// ```
    pub fn with_fuzzy_boundaries(mut self, width: f32) -> Self {
        self.fuzzy_boundaries = true;
        self.boundary_width = width.clamp(0.0, 0.2);
        self
    }

    /// Classifies surprise and coherence values into a Johari quadrant.
    ///
    /// # Arguments
    ///
    /// * `delta_s` - Surprise/entropy value in range [0.0, 1.0]
    /// * `delta_c` - Coherence/understanding value in range [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// The appropriate `JohariQuadrant` based on the classifier's configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::{JohariClassifier, JohariQuadrant};
    /// use context_graph_utl::config::JohariConfig;
    ///
    /// let classifier = JohariClassifier::with_default_thresholds();
    /// assert_eq!(classifier.classify(0.3, 0.7), JohariQuadrant::Open);
    /// assert_eq!(classifier.classify(0.7, 0.3), JohariQuadrant::Blind);
    /// ```
    #[inline]
    pub fn classify(&self, delta_s: f32, delta_c: f32) -> JohariQuadrant {
        // Clamp inputs to valid range
        let delta_s = delta_s.clamp(0.0, 1.0);
        let delta_c = delta_c.clamp(0.0, 1.0);

        classify_with_thresholds(
            delta_s,
            delta_c,
            self.surprise_threshold,
            self.coherence_threshold,
        )
    }

    /// Classifies with confidence scores for each quadrant.
    ///
    /// When fuzzy boundaries are enabled, this method returns confidence scores
    /// indicating how strongly the input belongs to each quadrant. Values near
    /// boundaries will have lower confidence for the primary quadrant.
    ///
    /// # Arguments
    ///
    /// * `delta_s` - Surprise/entropy value in range [0.0, 1.0]
    /// * `delta_c` - Coherence/understanding value in range [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// A tuple of `(quadrant, confidence)` where confidence is in range [0.0, 1.0].
    /// Higher confidence indicates the point is further from the decision boundaries.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::{JohariClassifier, JohariQuadrant};
    ///
    /// let classifier = JohariClassifier::with_default_thresholds()
    ///     .with_fuzzy_boundaries(0.1);
    ///
    /// // Clear Open quadrant - high confidence
    /// let (quadrant, confidence) = classifier.classify_with_confidence(0.1, 0.9);
    /// assert_eq!(quadrant, JohariQuadrant::Open);
    /// assert!(confidence > 0.8);
    ///
    /// // Near boundary - lower confidence (0.05 from both thresholds -> confidence = 0.5)
    /// let (quadrant, confidence) = classifier.classify_with_confidence(0.45, 0.55);
    /// assert!((confidence - 0.5).abs() < 0.01); // Exactly at 50% confidence
    /// ```
    pub fn classify_with_confidence(&self, delta_s: f32, delta_c: f32) -> (JohariQuadrant, f32) {
        let delta_s = delta_s.clamp(0.0, 1.0);
        let delta_c = delta_c.clamp(0.0, 1.0);

        let quadrant = classify_with_thresholds(
            delta_s,
            delta_c,
            self.surprise_threshold,
            self.coherence_threshold,
        );

        if !self.fuzzy_boundaries {
            return (quadrant, 1.0);
        }

        // Calculate distance from boundaries
        let s_distance = (delta_s - self.surprise_threshold).abs();
        let c_distance = (delta_c - self.coherence_threshold).abs();
        let min_distance = s_distance.min(c_distance);

        // Calculate confidence based on distance from boundary
        let confidence = if min_distance >= self.boundary_width {
            1.0
        } else {
            min_distance / self.boundary_width
        };

        (quadrant, confidence)
    }

    /// Checks if the given values are near a quadrant boundary.
    ///
    /// # Arguments
    ///
    /// * `delta_s` - Surprise/entropy value
    /// * `delta_c` - Coherence/understanding value
    ///
    /// # Returns
    ///
    /// `true` if either value is within `boundary_width` of its threshold.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::JohariClassifier;
    ///
    /// let classifier = JohariClassifier::with_default_thresholds()
    ///     .with_fuzzy_boundaries(0.1);
    ///
    /// assert!(classifier.is_near_boundary(0.45, 0.8));   // Near surprise threshold
    /// assert!(!classifier.is_near_boundary(0.2, 0.8));   // Far from boundaries
    /// ```
    pub fn is_near_boundary(&self, delta_s: f32, delta_c: f32) -> bool {
        let s_distance = (delta_s - self.surprise_threshold).abs();
        let c_distance = (delta_c - self.coherence_threshold).abs();
        s_distance < self.boundary_width || c_distance < self.boundary_width
    }

    /// Returns the current surprise threshold.
    #[inline]
    pub fn surprise_threshold(&self) -> f32 {
        self.surprise_threshold
    }

    /// Returns the current coherence threshold.
    #[inline]
    pub fn coherence_threshold(&self) -> f32 {
        self.coherence_threshold
    }

    /// Returns the boundary width for fuzzy classification.
    #[inline]
    pub fn boundary_width(&self) -> f32 {
        self.boundary_width
    }

    /// Returns whether fuzzy boundaries are enabled.
    #[inline]
    pub fn has_fuzzy_boundaries(&self) -> bool {
        self.fuzzy_boundaries
    }
}

impl Default for JohariClassifier {
    fn default() -> Self {
        Self::with_default_thresholds()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_quadrant_open() {
        // Low surprise, high coherence -> Open
        assert_eq!(classify_quadrant(0.0, 1.0), JohariQuadrant::Open);
        assert_eq!(classify_quadrant(0.2, 0.8), JohariQuadrant::Open);
        assert_eq!(classify_quadrant(0.49, 0.51), JohariQuadrant::Open);
    }

    #[test]
    fn test_classify_quadrant_blind() {
        // High surprise, low coherence -> Blind
        assert_eq!(classify_quadrant(1.0, 0.0), JohariQuadrant::Blind);
        assert_eq!(classify_quadrant(0.8, 0.2), JohariQuadrant::Blind);
        assert_eq!(classify_quadrant(0.51, 0.49), JohariQuadrant::Blind);
    }

    #[test]
    fn test_classify_quadrant_hidden() {
        // Low surprise, low coherence -> Hidden
        assert_eq!(classify_quadrant(0.0, 0.0), JohariQuadrant::Hidden);
        assert_eq!(classify_quadrant(0.2, 0.2), JohariQuadrant::Hidden);
        assert_eq!(classify_quadrant(0.49, 0.49), JohariQuadrant::Hidden);
    }

    #[test]
    fn test_classify_quadrant_unknown() {
        // High surprise, high coherence -> Unknown
        assert_eq!(classify_quadrant(1.0, 1.0), JohariQuadrant::Unknown);
        assert_eq!(classify_quadrant(0.8, 0.8), JohariQuadrant::Unknown);
        assert_eq!(classify_quadrant(0.51, 0.51), JohariQuadrant::Unknown);
    }

    #[test]
    fn test_classify_quadrant_boundary() {
        // Exact boundary cases (0.5)
        // At threshold: surprise >= 0.5 is "high", coherence <= 0.5 is "low"
        assert_eq!(classify_quadrant(0.5, 0.5), JohariQuadrant::Blind); // High S, Low C
    }

    #[test]
    fn test_classifier_new() {
        let config = JohariConfig::default();
        let classifier = JohariClassifier::new(&config);
        assert_eq!(classifier.surprise_threshold(), 0.5);
        assert_eq!(classifier.coherence_threshold(), 0.5);
    }

    #[test]
    fn test_classifier_with_custom_thresholds() {
        let classifier = JohariClassifier::with_thresholds(0.3, 0.7);
        assert_eq!(classifier.surprise_threshold(), 0.3);
        assert_eq!(classifier.coherence_threshold(), 0.7);

        // With custom thresholds: 0.25 is low surprise (< 0.3), 0.75 is high coherence (> 0.7)
        assert_eq!(classifier.classify(0.25, 0.75), JohariQuadrant::Open);
    }

    #[test]
    fn test_classifier_classify_all_quadrants() {
        let classifier = JohariClassifier::default();

        assert_eq!(classifier.classify(0.2, 0.8), JohariQuadrant::Open);
        assert_eq!(classifier.classify(0.8, 0.2), JohariQuadrant::Blind);
        assert_eq!(classifier.classify(0.2, 0.2), JohariQuadrant::Hidden);
        assert_eq!(classifier.classify(0.8, 0.8), JohariQuadrant::Unknown);
    }

    #[test]
    fn test_classifier_clamps_input() {
        let classifier = JohariClassifier::default();

        // Values outside [0, 1] should be clamped
        assert_eq!(classifier.classify(-0.5, 1.5), JohariQuadrant::Open);
        assert_eq!(classifier.classify(1.5, -0.5), JohariQuadrant::Blind);
    }

    #[test]
    fn test_classifier_with_fuzzy_boundaries() {
        let classifier = JohariClassifier::with_default_thresholds().with_fuzzy_boundaries(0.1);
        assert!(classifier.has_fuzzy_boundaries());
        assert_eq!(classifier.boundary_width(), 0.1);
    }

    #[test]
    fn test_classify_with_confidence_high() {
        let classifier = JohariClassifier::with_default_thresholds().with_fuzzy_boundaries(0.1);

        // Far from boundaries -> high confidence
        let (quadrant, confidence) = classifier.classify_with_confidence(0.1, 0.9);
        assert_eq!(quadrant, JohariQuadrant::Open);
        assert!(
            confidence > 0.9,
            "Expected high confidence, got {}",
            confidence
        );
    }

    #[test]
    fn test_classify_with_confidence_low() {
        let classifier = JohariClassifier::with_default_thresholds().with_fuzzy_boundaries(0.1);

        // Near surprise boundary -> lower confidence
        let (quadrant, confidence) = classifier.classify_with_confidence(0.45, 0.9);
        assert_eq!(quadrant, JohariQuadrant::Open);
        assert!(confidence < 1.0, "Expected lower confidence near boundary");
    }

    #[test]
    fn test_classify_with_confidence_no_fuzzy() {
        let classifier = JohariClassifier::with_default_thresholds();

        // Without fuzzy boundaries, confidence is always 1.0
        let (_, confidence) = classifier.classify_with_confidence(0.45, 0.55);
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_is_near_boundary() {
        let classifier = JohariClassifier::with_default_thresholds().with_fuzzy_boundaries(0.1);

        assert!(classifier.is_near_boundary(0.45, 0.8)); // Near surprise boundary
        assert!(classifier.is_near_boundary(0.2, 0.55)); // Near coherence boundary
        assert!(classifier.is_near_boundary(0.45, 0.55)); // Near both
        assert!(!classifier.is_near_boundary(0.2, 0.8)); // Far from both
    }

    #[test]
    fn test_classifier_default() {
        let classifier = JohariClassifier::default();
        assert_eq!(classifier.surprise_threshold(), 0.5);
        assert_eq!(classifier.coherence_threshold(), 0.5);
        assert!(!classifier.has_fuzzy_boundaries());
    }

    #[test]
    fn test_constitution_compliance() {
        // Verify classification matches constitution.yaml specification
        let classifier = JohariClassifier::default();

        // Open: delta_s < 0.5, delta_c > 0.5 -> direct recall
        assert_eq!(classifier.classify(0.3, 0.7), JohariQuadrant::Open);

        // Blind: delta_s > 0.5, delta_c < 0.5 -> discovery
        assert_eq!(classifier.classify(0.7, 0.3), JohariQuadrant::Blind);

        // Hidden: delta_s < 0.5, delta_c < 0.5 -> private
        assert_eq!(classifier.classify(0.3, 0.3), JohariQuadrant::Hidden);

        // Unknown: delta_s > 0.5, delta_c > 0.5 -> frontier
        assert_eq!(classifier.classify(0.7, 0.7), JohariQuadrant::Unknown);
    }

    #[test]
    fn test_boundary_width_clamped() {
        let classifier = JohariClassifier::with_default_thresholds().with_fuzzy_boundaries(0.5);

        // Boundary width should be clamped to max 0.2
        assert_eq!(classifier.boundary_width(), 0.2);
    }

    #[test]
    fn test_extreme_values() {
        let classifier = JohariClassifier::default();

        // Test extreme corners
        assert_eq!(classifier.classify(0.0, 1.0), JohariQuadrant::Open);
        assert_eq!(classifier.classify(1.0, 0.0), JohariQuadrant::Blind);
        assert_eq!(classifier.classify(0.0, 0.0), JohariQuadrant::Hidden);
        assert_eq!(classifier.classify(1.0, 1.0), JohariQuadrant::Unknown);
    }
}
