//! CognitivePulse struct for meta-cognitive state tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::utl::{EmotionalState, UtlMetrics};

use super::action::SuggestedAction;

/// Cognitive Pulse header included in all tool responses.
///
/// Provides meta-cognitive state information to help agents
/// understand system state and decide on next actions.
///
/// The 6-field structure captures:
/// - Core metrics: entropy, coherence, coherence_delta
/// - Emotional context: emotional_weight (from EmotionalState)
/// - Action guidance: suggested_action
/// - Temporal context: timestamp
///
/// # Example Response
///
/// ```json
/// {
///   "entropy": 0.45,
///   "coherence": 0.72,
///   "coherence_delta": 0.05,
///   "emotional_weight": 1.2,
///   "suggested_action": "continue",
///   "timestamp": "2025-01-01T12:00:00Z"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitivePulse {
    /// Current entropy level [0.0, 1.0]
    /// Higher values indicate more uncertainty/novelty
    pub entropy: f32,

    /// Current coherence level [0.0, 1.0]
    /// Higher values indicate better integration/understanding
    pub coherence: f32,

    /// Change in coherence from previous measurement [−1.0, 1.0]
    /// Positive values indicate improving understanding
    pub coherence_delta: f32,

    /// Emotional weight modifier from EmotionalState [0.0, 2.0]
    /// Derived from EmotionalState::weight_modifier()
    pub emotional_weight: f32,

    /// Suggested action based on current state
    pub suggested_action: SuggestedAction,

    /// UTC timestamp when this pulse was created
    pub timestamp: DateTime<Utc>,
}

impl Default for CognitivePulse {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            coherence_delta: 0.0,
            emotional_weight: 1.0,
            suggested_action: SuggestedAction::Continue,
            timestamp: Utc::now(),
        }
    }
}

impl CognitivePulse {
    /// Create a new pulse with explicit values for all 6 fields.
    ///
    /// Values are clamped to valid ranges:
    /// - entropy: [0.0, 1.0]
    /// - coherence: [0.0, 1.0]
    /// - coherence_delta: [-1.0, 1.0]
    /// - emotional_weight: [0.0, 2.0]
    pub fn new(
        entropy: f32,
        coherence: f32,
        coherence_delta: f32,
        emotional_weight: f32,
        suggested_action: SuggestedAction,
    ) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let coherence_delta = coherence_delta.clamp(-1.0, 1.0);
        let emotional_weight = emotional_weight.clamp(0.0, 2.0);

        Self {
            entropy,
            coherence,
            coherence_delta,
            emotional_weight,
            suggested_action,
            timestamp: Utc::now(),
        }
    }

    /// Create a new pulse computed from UTL metrics.
    ///
    /// Derives all 6 fields from the provided metrics:
    /// - entropy: from metrics.entropy
    /// - coherence: from metrics.coherence
    /// - coherence_delta: from metrics.coherence_change
    /// - emotional_weight: from metrics.emotional_weight
    /// - suggested_action: computed from entropy/coherence
    /// - timestamp: current UTC time
    pub fn computed(metrics: &UtlMetrics) -> Self {
        let entropy = metrics.entropy.clamp(0.0, 1.0);
        let coherence = metrics.coherence.clamp(0.0, 1.0);
        let coherence_delta = metrics.coherence_change.clamp(-1.0, 1.0);
        let emotional_weight = metrics.emotional_weight.clamp(0.0, 2.0);
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            coherence_delta,
            emotional_weight,
            suggested_action,
            timestamp: Utc::now(),
        }
    }

    /// Create a simple pulse from entropy and coherence values.
    ///
    /// Uses default values for other fields:
    /// - coherence_delta: 0.0
    /// - emotional_weight: 1.0
    /// - suggested_action: computed from entropy/coherence
    pub fn from_values(entropy: f32, coherence: f32) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            coherence_delta: 0.0,
            emotional_weight: 1.0,
            suggested_action,
            timestamp: Utc::now(),
        }
    }

    /// Compute the suggested action based on entropy and coherence.
    pub(crate) fn compute_action(entropy: f32, coherence: f32) -> SuggestedAction {
        match (entropy, coherence) {
            // High entropy, low coherence - needs stabilization
            (e, c) if e > 0.7 && c < 0.4 => SuggestedAction::Stabilize,
            // High entropy, high coherence - exploration frontier
            (e, c) if e > 0.6 && c > 0.5 => SuggestedAction::Explore,
            // Low entropy, high coherence - well understood, ready
            (e, c) if e < 0.4 && c > 0.6 => SuggestedAction::Ready,
            // Low coherence - needs consolidation
            (_, c) if c < 0.4 => SuggestedAction::Consolidate,
            // High entropy - consider pruning
            (e, _) if e > 0.8 => SuggestedAction::Prune,
            // Review needed
            (e, c) if e > 0.5 && c < 0.5 => SuggestedAction::Review,
            // Default: continue
            _ => SuggestedAction::Continue,
        }
    }

    /// Returns true if the system is in a healthy state.
    pub fn is_healthy(&self) -> bool {
        self.entropy < 0.8 && self.coherence > 0.3
    }

    /// Updates entropy and coherence by applying deltas.
    ///
    /// Recomputes suggested_action based on new values.
    /// All values are clamped to valid ranges.
    ///
    /// # Arguments
    /// * `delta_entropy` - Change to apply to entropy
    /// * `delta_coherence` - Change to apply to coherence
    pub fn update(&mut self, delta_entropy: f32, delta_coherence: f32) {
        // Store old coherence for delta calculation
        let old_coherence = self.coherence;

        // Apply deltas with clamping
        self.entropy = (self.entropy + delta_entropy).clamp(0.0, 1.0);
        self.coherence = (self.coherence + delta_coherence).clamp(0.0, 1.0);

        // Update coherence_delta to reflect this change
        self.coherence_delta = (self.coherence - old_coherence).clamp(-1.0, 1.0);

        // Recompute suggested action
        self.suggested_action = Self::compute_action(self.entropy, self.coherence);

        // Update timestamp to now
        self.timestamp = Utc::now();
    }

    /// Linearly interpolates between two pulses.
    ///
    /// Creates a new pulse that is a blend of `self` and `other`.
    /// The blend factor `t` determines the weight:
    /// - t = 0.0 -> result equals self
    /// - t = 1.0 -> result equals other
    /// - t = 0.5 -> result is midpoint
    ///
    /// # Arguments
    /// * `other` - The other pulse to blend with
    /// * `t` - Blend factor clamped to [0.0, 1.0]
    ///
    /// # Returns
    /// A new CognitivePulse with interpolated values.
    pub fn blend(&self, other: &CognitivePulse, t: f32) -> CognitivePulse {
        let t = t.clamp(0.0, 1.0);

        // Linear interpolation helper
        let lerp = |a: f32, b: f32| a + t * (b - a);

        // Interpolate numeric fields
        let entropy = lerp(self.entropy, other.entropy);
        let coherence = lerp(self.coherence, other.coherence);
        let coherence_delta = lerp(self.coherence_delta, other.coherence_delta);
        let emotional_weight = lerp(self.emotional_weight, other.emotional_weight);

        // Compute new action from blended values
        let suggested_action = Self::compute_action(entropy, coherence);

        CognitivePulse {
            entropy,
            coherence,
            coherence_delta,
            emotional_weight,
            suggested_action,
            timestamp: Utc::now(),
        }
    }

    /// Create a pulse with a specific emotional state.
    ///
    /// Derives emotional_weight from the provided EmotionalState.
    pub fn with_emotion(
        entropy: f32,
        coherence: f32,
        emotional_state: EmotionalState,
    ) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let emotional_weight = emotional_state.weight_modifier();
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            coherence_delta: 0.0,
            emotional_weight,
            suggested_action,
            timestamp: Utc::now(),
        }
    }
}
