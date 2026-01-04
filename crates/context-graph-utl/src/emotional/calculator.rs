//! Emotional weight calculator.
//!
//! Computes the emotional weight (wₑ) component of the UTL formula by combining
//! lexicon-based sentiment analysis with the current emotional state's weight modifier.

use crate::config::EmotionalConfig;
use crate::error::{UtlError, UtlResult};

use super::lexicon::{SentimentLexicon, SentimentScore};
use super::EmotionalState;

/// Calculator for emotional weight (wₑ) in the UTL formula.
///
/// Combines sentiment analysis from text content with the current emotional state
/// to produce a weight value in the constitution-mandated range `[0.5, 1.5]`.
///
/// # Algorithm
///
/// 1. Analyze text sentiment using the lexicon to get a score in `[-1, 1]`
/// 2. Get the state's weight modifier (e.g., Focused = 1.3, Fatigued = 0.6)
/// 3. Combine: `base_weight = default + (sentiment * sensitivity * state_modifier)`
/// 4. Apply arousal/valence modulation if enabled
/// 5. Clamp to `[min_weight, max_weight]` (default `[0.5, 1.5]`)
///
/// # Example
///
/// ```
/// use context_graph_utl::emotional::{EmotionalWeightCalculator, EmotionalState};
/// use context_graph_utl::config::EmotionalConfig;
///
/// let config = EmotionalConfig::default();
/// let calculator = EmotionalWeightCalculator::new(&config);
///
/// // Positive sentiment with engaged state
/// let weight = calculator.compute_emotional_weight(
///     "This discovery is amazing and enlightening!",
///     EmotionalState::Engaged,
/// );
/// assert!(weight > 1.0); // Positive sentiment + engaged state = amplified
///
/// // Negative sentiment with fatigued state
/// let weight = calculator.compute_emotional_weight(
///     "This is frustrating and confusing.",
///     EmotionalState::Fatigued,
/// );
/// assert!(weight < 1.0); // Negative sentiment + fatigued state = dampened
/// ```
#[derive(Debug, Clone)]
pub struct EmotionalWeightCalculator {
    /// The sentiment lexicon for text analysis.
    lexicon: SentimentLexicon,

    /// Minimum emotional weight (constitution: 0.5).
    min_weight: f32,

    /// Maximum emotional weight (constitution: 1.5).
    max_weight: f32,

    /// Default/neutral emotional weight.
    default_weight: f32,

    /// Enable arousal-based modulation.
    arousal_modulation: bool,

    /// Arousal sensitivity factor.
    arousal_sensitivity: f32,

    /// Enable valence-based modulation.
    valence_modulation: bool,

    /// Valence sensitivity factor.
    valence_sensitivity: f32,
}

impl EmotionalWeightCalculator {
    /// Create a new emotional weight calculator with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Emotional weight configuration from [`EmotionalConfig`]
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::EmotionalWeightCalculator;
    /// use context_graph_utl::config::EmotionalConfig;
    ///
    /// let config = EmotionalConfig::default();
    /// let calculator = EmotionalWeightCalculator::new(&config);
    /// ```
    pub fn new(config: &EmotionalConfig) -> Self {
        Self {
            lexicon: SentimentLexicon::default(),
            min_weight: config.min_weight,
            max_weight: config.max_weight,
            default_weight: config.default_weight,
            arousal_modulation: config.arousal_modulation,
            arousal_sensitivity: config.arousal_sensitivity,
            valence_modulation: config.valence_modulation,
            valence_sensitivity: config.valence_sensitivity,
        }
    }

    /// Create a calculator with a custom sentiment lexicon.
    ///
    /// # Arguments
    ///
    /// * `config` - Emotional weight configuration
    /// * `lexicon` - Custom sentiment lexicon
    pub fn with_lexicon(config: &EmotionalConfig, lexicon: SentimentLexicon) -> Self {
        let mut calc = Self::new(config);
        calc.lexicon = lexicon;
        calc
    }

    /// Compute the emotional weight (wₑ) from text content and current state.
    ///
    /// This is the main entry point for emotional weight calculation. It combines
    /// the sentiment of the text with the current emotional state to produce
    /// a weight value clamped to `[0.5, 1.5]`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to analyze for sentiment
    /// * `current_state` - The current emotional state of the system
    ///
    /// # Returns
    ///
    /// The computed emotional weight in the range `[0.5, 1.5]`.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::{EmotionalWeightCalculator, EmotionalState};
    /// use context_graph_utl::config::EmotionalConfig;
    ///
    /// let calculator = EmotionalWeightCalculator::new(&EmotionalConfig::default());
    ///
    /// let weight = calculator.compute_emotional_weight(
    ///     "I understand this concept clearly now!",
    ///     EmotionalState::Focused,
    /// );
    ///
    /// assert!(weight >= 0.5 && weight <= 1.5);
    /// ```
    pub fn compute_emotional_weight(&self, text: &str, current_state: EmotionalState) -> f32 {
        // Step 1: Get sentiment from text
        let sentiment = self.lexicon.analyze(text);

        // Step 2: Get state weight modifier
        let state_modifier = current_state.weight_modifier();

        // Step 3: Compute base weight from sentiment and state
        let sentiment_contribution =
            self.compute_sentiment_contribution(&sentiment, state_modifier);

        // Step 4: Apply arousal modulation if enabled
        let arousal_contribution = if self.arousal_modulation {
            self.compute_arousal_contribution(&sentiment, state_modifier)
        } else {
            0.0
        };

        // Step 5: Combine contributions
        let raw_weight = self.default_weight + sentiment_contribution + arousal_contribution;

        // Step 6: Clamp to valid range
        self.clamp_weight(raw_weight)
    }

    /// Compute the emotional weight from a pre-computed sentiment score.
    ///
    /// Use this when you've already analyzed the text and have a sentiment score.
    ///
    /// # Arguments
    ///
    /// * `sentiment` - Pre-computed sentiment score
    /// * `current_state` - The current emotional state
    ///
    /// # Returns
    ///
    /// The computed emotional weight in the range `[0.5, 1.5]`.
    pub fn compute_from_sentiment(
        &self,
        sentiment: &SentimentScore,
        current_state: EmotionalState,
    ) -> f32 {
        let state_modifier = current_state.weight_modifier();
        let sentiment_contribution = self.compute_sentiment_contribution(sentiment, state_modifier);
        let arousal_contribution = if self.arousal_modulation {
            self.compute_arousal_contribution(sentiment, state_modifier)
        } else {
            0.0
        };

        self.clamp_weight(self.default_weight + sentiment_contribution + arousal_contribution)
    }

    /// Compute the weight contribution from sentiment analysis.
    ///
    /// Uses valence (positive-negative) to modulate the weight.
    fn compute_sentiment_contribution(
        &self,
        sentiment: &SentimentScore,
        state_modifier: f32,
    ) -> f32 {
        if !self.valence_modulation {
            return 0.0;
        }

        // Net sentiment: positive - negative (range: [-1, 1])
        let net_sentiment = sentiment.net_sentiment();

        // Scale by valence sensitivity and state modifier
        // State modifier affects how strongly the sentiment impacts weight
        let state_effect = (state_modifier - 1.0) * 0.5 + 1.0; // Normalize state effect

        net_sentiment * self.valence_sensitivity * 0.25 * state_effect
    }

    /// Compute the weight contribution from arousal (intensity).
    ///
    /// Uses the magnitude of emotional content to modulate the weight.
    fn compute_arousal_contribution(&self, sentiment: &SentimentScore, state_modifier: f32) -> f32 {
        // Arousal is based on the intensity of emotion (positive + negative)
        // High arousal = more emotional content = potentially stronger learning signal
        let arousal = sentiment.arousal();

        // States with high modifiers (Focused, Curious) amplify arousal effect
        // States with low modifiers (Fatigued, Stressed) dampen it
        let state_effect = state_modifier;

        arousal * self.arousal_sensitivity * 0.15 * state_effect
    }

    /// Clamp a weight value to the valid constitution range.
    ///
    /// # Arguments
    ///
    /// * `weight` - The raw weight value
    ///
    /// # Returns
    ///
    /// Weight clamped to `[min_weight, max_weight]` (default `[0.5, 1.5]`).
    #[inline]
    pub fn clamp_weight(&self, weight: f32) -> f32 {
        weight.clamp(self.min_weight, self.max_weight)
    }

    /// Get the neutral/default weight value.
    #[inline]
    pub fn default_weight(&self) -> f32 {
        self.default_weight
    }

    /// Get the minimum valid weight.
    #[inline]
    pub fn min_weight(&self) -> f32 {
        self.min_weight
    }

    /// Get the maximum valid weight.
    #[inline]
    pub fn max_weight(&self) -> f32 {
        self.max_weight
    }

    /// Validate that a weight is within the valid range.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with details if not.
    pub fn validate_weight(&self, weight: f32) -> UtlResult<()> {
        if weight.is_nan() {
            return Err(UtlError::InvalidParameter {
                name: "emotional_weight".to_string(),
                value: "NaN".to_string(),
                reason: "Emotional weight cannot be NaN".to_string(),
            });
        }

        if weight.is_infinite() {
            return Err(UtlError::InvalidParameter {
                name: "emotional_weight".to_string(),
                value: weight.to_string(),
                reason: "Emotional weight cannot be infinite".to_string(),
            });
        }

        if weight < self.min_weight || weight > self.max_weight {
            return Err(UtlError::InvalidParameter {
                name: "emotional_weight".to_string(),
                value: weight.to_string(),
                reason: format!(
                    "Weight must be in [{}, {}]",
                    self.min_weight, self.max_weight
                ),
            });
        }

        Ok(())
    }
}

impl Default for EmotionalWeightCalculator {
    fn default() -> Self {
        Self::new(&EmotionalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let config = EmotionalConfig::default();
        let calc = EmotionalWeightCalculator::new(&config);

        assert_eq!(calc.min_weight(), 0.5);
        assert_eq!(calc.max_weight(), 1.5);
        assert_eq!(calc.default_weight(), 1.0);
    }

    #[test]
    fn test_neutral_state_neutral_text() {
        let calc = EmotionalWeightCalculator::default();

        // Neutral text with neutral state should give approximately default weight
        let weight = calc.compute_emotional_weight("The sky is blue.", EmotionalState::Neutral);

        // Should be close to 1.0 (within arousal contribution)
        assert!(weight >= 0.9 && weight <= 1.1);
    }

    #[test]
    fn test_positive_sentiment_increases_weight() {
        let calc = EmotionalWeightCalculator::default();

        let neutral_weight =
            calc.compute_emotional_weight("The data exists.", EmotionalState::Neutral);
        let positive_weight = calc
            .compute_emotional_weight("This is excellent and wonderful!", EmotionalState::Neutral);

        assert!(positive_weight > neutral_weight);
    }

    #[test]
    fn test_negative_sentiment_decreases_weight() {
        let calc = EmotionalWeightCalculator::default();

        let neutral_weight =
            calc.compute_emotional_weight("The data exists.", EmotionalState::Neutral);
        let negative_weight = calc.compute_emotional_weight(
            "This is terrible and disappointing.",
            EmotionalState::Neutral,
        );

        assert!(negative_weight < neutral_weight);
    }

    #[test]
    fn test_focused_state_amplifies() {
        let calc = EmotionalWeightCalculator::default();

        let text = "This is an interesting discovery.";
        let neutral = calc.compute_emotional_weight(text, EmotionalState::Neutral);
        let focused = calc.compute_emotional_weight(text, EmotionalState::Focused);

        // Focused has higher modifier (1.3 vs 1.0), so should have higher weight
        assert!(focused >= neutral);
    }

    #[test]
    fn test_fatigued_state_dampens() {
        let calc = EmotionalWeightCalculator::default();

        let text = "This is an interesting discovery.";
        let neutral = calc.compute_emotional_weight(text, EmotionalState::Neutral);
        let fatigued = calc.compute_emotional_weight(text, EmotionalState::Fatigued);

        // Fatigued has lower modifier (0.6 vs 1.0), so arousal contribution is lower
        assert!(fatigued <= neutral);
    }

    #[test]
    fn test_weight_always_clamped() {
        let calc = EmotionalWeightCalculator::default();

        // Even with extreme inputs, weight should be clamped
        let very_positive = calc.compute_emotional_weight(
            "amazing wonderful excellent perfect brilliant outstanding",
            EmotionalState::Focused,
        );
        let very_negative = calc.compute_emotional_weight(
            "terrible horrible awful bad disappointing frustrating",
            EmotionalState::Fatigued,
        );

        assert!(very_positive >= 0.5 && very_positive <= 1.5);
        assert!(very_negative >= 0.5 && very_negative <= 1.5);
    }

    #[test]
    fn test_empty_text() {
        let calc = EmotionalWeightCalculator::default();

        // Empty text should give weight based purely on state
        let weight = calc.compute_emotional_weight("", EmotionalState::Curious);

        assert!(weight >= 0.5 && weight <= 1.5);
    }

    #[test]
    fn test_compute_from_sentiment() {
        let calc = EmotionalWeightCalculator::default();

        let sentiment = SentimentScore::new(0.8, 0.0, 0.2);
        let weight = calc.compute_from_sentiment(&sentiment, EmotionalState::Engaged);

        assert!(weight > 1.0); // Positive sentiment + engaged state
        assert!(weight <= 1.5);
    }

    #[test]
    fn test_validate_weight_valid() {
        let calc = EmotionalWeightCalculator::default();

        assert!(calc.validate_weight(0.5).is_ok());
        assert!(calc.validate_weight(1.0).is_ok());
        assert!(calc.validate_weight(1.5).is_ok());
    }

    #[test]
    fn test_validate_weight_invalid() {
        let calc = EmotionalWeightCalculator::default();

        assert!(calc.validate_weight(0.4).is_err());
        assert!(calc.validate_weight(1.6).is_err());
        assert!(calc.validate_weight(f32::NAN).is_err());
        assert!(calc.validate_weight(f32::INFINITY).is_err());
    }

    #[test]
    fn test_all_emotional_states() {
        let calc = EmotionalWeightCalculator::default();
        let text = "This is a test sentence.";

        let states = [
            EmotionalState::Neutral,
            EmotionalState::Curious,
            EmotionalState::Focused,
            EmotionalState::Stressed,
            EmotionalState::Fatigued,
            EmotionalState::Engaged,
            EmotionalState::Confused,
        ];

        for state in states {
            let weight = calc.compute_emotional_weight(text, state);
            assert!(
                weight >= 0.5 && weight <= 1.5,
                "State {:?} produced invalid weight {}",
                state,
                weight
            );
        }
    }

    #[test]
    fn test_with_custom_lexicon() {
        let config = EmotionalConfig::default();
        let mut lexicon = SentimentLexicon::new();
        lexicon.add_positive("customword", 0.9);

        let calc = EmotionalWeightCalculator::with_lexicon(&config, lexicon);

        let weight = calc.compute_emotional_weight("customword", EmotionalState::Neutral);
        assert!(weight > 1.0);
    }

    #[test]
    fn test_disabled_modulation() {
        let config = EmotionalConfig {
            arousal_modulation: false,
            valence_modulation: false,
            ..Default::default()
        };

        let calc = EmotionalWeightCalculator::new(&config);

        // With all modulation disabled, should always return default weight
        let weight = calc.compute_emotional_weight(
            "amazing wonderful terrible horrible",
            EmotionalState::Focused,
        );

        assert_eq!(weight, 1.0);
    }

    #[test]
    fn test_constitution_compliance() {
        let calc = EmotionalWeightCalculator::default();

        // Constitution specifies wₑ range as [0.5, 1.5]
        assert_eq!(calc.min_weight(), 0.5);
        assert_eq!(calc.max_weight(), 1.5);
        assert_eq!(calc.default_weight(), 1.0);

        // All possible outputs should be within this range
        for _ in 0..100 {
            let text = "random text with varying emotional content amazing terrible neutral";
            for state in [
                EmotionalState::Neutral,
                EmotionalState::Curious,
                EmotionalState::Focused,
                EmotionalState::Stressed,
                EmotionalState::Fatigued,
                EmotionalState::Engaged,
                EmotionalState::Confused,
            ] {
                let weight = calc.compute_emotional_weight(text, state);
                assert!(weight >= 0.5 && weight <= 1.5);
            }
        }
    }
}
