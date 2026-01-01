//! UTL (Unified Theory of Learning) types and metrics.
//!
//! Implements the learning equation: L = f((ΔS × ΔC) · wₑ · cos φ)

use serde::{Deserialize, Serialize};

/// UTL (Unified Theory of Learning) metrics.
///
/// Captures all components of the UTL equation for measuring learning effectiveness.
///
/// # The UTL Equation
///
/// `L = f((ΔS × ΔC) · wₑ · cos φ)`
///
/// Where:
/// - `ΔS` (delta_s): Surprise/entropy change - information gain
/// - `ΔC` (delta_c): Coherence change - understanding gain
/// - `wₑ`: Emotional weight - attention/motivation
/// - `cos φ`: Goal alignment - how well learning aligns with objectives
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlMetrics {
    /// Entropy measure [0.0, 1.0]
    pub entropy: f32,

    /// Coherence measure [0.0, 1.0]
    pub coherence: f32,

    /// Computed learning score
    pub learning_score: f32,

    /// Surprise component (delta_S)
    pub surprise: f32,

    /// Coherence change component (delta_C)
    pub coherence_change: f32,

    /// Emotional weight (w_e)
    pub emotional_weight: f32,

    /// Alignment angle cosine (cos phi)
    pub alignment: f32,
}

impl Default for UtlMetrics {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            learning_score: 0.0,
            surprise: 0.0,
            coherence_change: 0.0,
            emotional_weight: 1.0,
            alignment: 1.0,
        }
    }
}

impl UtlMetrics {
    /// Compute the learning score from current components.
    ///
    /// Uses the UTL equation: L = (ΔS × ΔC) · wₑ · cos φ
    pub fn compute_learning_score(&mut self) {
        self.learning_score =
            (self.surprise * self.coherence_change) * self.emotional_weight * self.alignment;
        self.learning_score = self.learning_score.clamp(0.0, 1.0);
    }

    /// Check if this represents an optimal learning state.
    ///
    /// Optimal learning occurs when both surprise and coherence are balanced
    /// (the "Aha!" moment - not too easy, not too confusing).
    pub fn is_optimal(&self) -> bool {
        self.entropy > 0.3 && self.entropy < 0.7 && self.coherence > 0.4 && self.coherence < 0.8
    }
}

/// UTL computation context.
///
/// Provides the contextual information needed to compute UTL metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UtlContext {
    /// Prior beliefs/expectations (prior entropy)
    pub prior_entropy: f32,

    /// Current system coherence
    pub current_coherence: f32,

    /// Emotional state modifier
    pub emotional_state: EmotionalState,

    /// Goal alignment vector
    pub goal_vector: Option<Vec<f32>>,
}

/// Represents the cognitive-emotional state of the system.
/// Each state has a weight modifier that affects UTL calculations.
///
/// The weight modifier feeds into the UTL formula:
/// `L = f((ΔS × ΔC) · wₑ · cos φ)`
/// Where `wₑ` (emotional weight) is derived from `EmotionalState::weight_modifier()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmotionalState {
    /// Baseline state with no modulation (weight: 1.0)
    #[default]
    Neutral,
    /// Active exploration state - enhanced novelty seeking (weight: 1.2)
    Curious,
    /// Deep concentration state - enhanced coherence (weight: 1.3)
    Focused,
    /// High cognitive load - reduced processing capacity (weight: 0.8)
    Stressed,
    /// Low energy state - reduced overall performance (weight: 0.6)
    Fatigued,
    /// Active engagement - balanced enhancement (weight: 1.15)
    Engaged,
    /// Uncertainty state - reduced confidence (weight: 0.9)
    Confused,
}

impl EmotionalState {
    /// Returns the weight modifier for UTL calculations.
    ///
    /// Modifiers:
    /// - Neutral: 1.0 (no modification)
    /// - Curious: 1.2 (enhanced novelty seeking)
    /// - Focused: 1.3 (enhanced coherence)
    /// - Stressed: 0.8 (reduced capacity)
    /// - Fatigued: 0.6 (reduced performance)
    /// - Engaged: 1.15 (balanced enhancement)
    /// - Confused: 0.9 (reduced confidence)
    pub fn weight_modifier(&self) -> f32 {
        match self {
            Self::Neutral => 1.0,
            Self::Curious => 1.2,
            Self::Focused => 1.3,
            Self::Stressed => 0.8,
            Self::Fatigued => 0.6,
            Self::Engaged => 1.15,
            Self::Confused => 0.9,
        }
    }

    /// Returns a human-readable description of this state.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Neutral => "Baseline cognitive state with no modulation",
            Self::Curious => "Active exploration with enhanced novelty seeking",
            Self::Focused => "Deep concentration with enhanced coherence processing",
            Self::Stressed => "High cognitive load reducing processing capacity",
            Self::Fatigued => "Low energy state with reduced overall performance",
            Self::Engaged => "Active engagement with balanced enhancement",
            Self::Confused => "Uncertainty state with reduced confidence",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utl_default() {
        let metrics = UtlMetrics::default();
        assert_eq!(metrics.entropy, 0.5);
        assert_eq!(metrics.coherence, 0.5);
        assert_eq!(metrics.learning_score, 0.0);
    }

    #[test]
    fn test_compute_learning_score() {
        let mut metrics = UtlMetrics {
            surprise: 0.5,
            coherence_change: 0.6,
            emotional_weight: 1.2,
            alignment: 0.9,
            ..Default::default()
        };
        metrics.compute_learning_score();

        // (0.5 * 0.6) * 1.2 * 0.9 = 0.324
        assert!((metrics.learning_score - 0.324).abs() < 0.001);
    }

    #[test]
    fn test_emotional_state_weight_modifiers() {
        assert_eq!(EmotionalState::Neutral.weight_modifier(), 1.0);
        assert_eq!(EmotionalState::Curious.weight_modifier(), 1.2);
        assert_eq!(EmotionalState::Focused.weight_modifier(), 1.3);
        assert_eq!(EmotionalState::Stressed.weight_modifier(), 0.8);
        assert_eq!(EmotionalState::Fatigued.weight_modifier(), 0.6);
        assert_eq!(EmotionalState::Engaged.weight_modifier(), 1.15);
        assert_eq!(EmotionalState::Confused.weight_modifier(), 0.9);
    }

    #[test]
    fn test_emotional_state_default_is_neutral() {
        let state = EmotionalState::default();
        assert_eq!(state, EmotionalState::Neutral);
        assert_eq!(state.weight_modifier(), 1.0);
    }

    #[test]
    fn test_emotional_state_serde_roundtrip() {
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
            let json = serde_json::to_string(&state).unwrap();
            let parsed: EmotionalState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, parsed);
        }
    }

    #[test]
    fn test_emotional_state_serde_snake_case() {
        // Verify snake_case serialization for multi-word variants
        // Note: single-word variants serialize as lowercase
        let neutral_json = serde_json::to_string(&EmotionalState::Neutral).unwrap();
        assert_eq!(neutral_json, "\"neutral\"");

        // All variants should roundtrip correctly
        for state in [
            EmotionalState::Neutral,
            EmotionalState::Curious,
            EmotionalState::Focused,
            EmotionalState::Stressed,
            EmotionalState::Fatigued,
            EmotionalState::Engaged,
            EmotionalState::Confused,
        ] {
            let json = serde_json::to_string(&state).unwrap();
            let parsed: EmotionalState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, parsed);
        }
    }

    #[test]
    fn test_emotional_state_description_not_empty() {
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
            let desc = state.description();
            assert!(!desc.is_empty(), "{:?} has empty description", state);
            assert!(desc.len() > 10, "{:?} description too short", state);
        }
    }

    #[test]
    fn test_emotional_state_hash_and_eq() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(EmotionalState::Neutral);
        set.insert(EmotionalState::Curious);
        set.insert(EmotionalState::Neutral); // duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&EmotionalState::Neutral));
        assert!(set.contains(&EmotionalState::Curious));
    }

    #[test]
    fn test_emotional_state_copy() {
        let state = EmotionalState::Focused;
        let copied = state; // Copy, not move
        assert_eq!(state, copied);
        assert_eq!(state.weight_modifier(), copied.weight_modifier());
    }

    #[test]
    fn test_weight_modifiers_are_positive() {
        for state in [
            EmotionalState::Neutral,
            EmotionalState::Curious,
            EmotionalState::Focused,
            EmotionalState::Stressed,
            EmotionalState::Fatigued,
            EmotionalState::Engaged,
            EmotionalState::Confused,
        ] {
            assert!(
                state.weight_modifier() > 0.0,
                "{:?} has non-positive weight",
                state
            );
        }
    }

    #[test]
    fn test_emotional_state_deserialize_unknown_variant() {
        let result: Result<EmotionalState, _> = serde_json::from_str("\"invalid_state\"");
        assert!(result.is_err());
    }
}
