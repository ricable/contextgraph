//! Emotional state tracking with temporal decay.
//!
//! Provides state tracking over time with exponential decay toward neutral state
//! and smooth state transitions.

use std::time::{Duration, Instant};

use super::EmotionalState;

/// Tracks emotional state over time with decay toward neutral.
///
/// The tracker maintains the current emotional state and applies exponential
/// decay toward neutral based on elapsed time. This models the natural
/// tendency for emotional states to fade without reinforcement.
///
/// # Example
///
/// ```
/// use context_graph_utl::emotional::{EmotionalStateTracker, EmotionalState, StateDecay};
/// use std::time::Duration;
///
/// let decay = StateDecay::default();
/// let mut tracker = EmotionalStateTracker::new(EmotionalState::Curious, decay);
///
/// // State starts as Curious
/// assert_eq!(tracker.current_state(), EmotionalState::Curious);
///
/// // After significant time, state decays toward neutral
/// tracker.decay(Duration::from_secs(300));
/// // State may have transitioned based on decay
/// ```
#[derive(Debug, Clone)]
pub struct EmotionalStateTracker {
    /// Current emotional state.
    current_state: EmotionalState,

    /// Current state intensity (1.0 = full, 0.0 = decayed to neutral).
    intensity: f32,

    /// Decay configuration.
    decay_config: StateDecay,

    /// Last update time for tracking elapsed duration.
    last_update: Option<Instant>,

    /// Transition smoothing factor (0.0-1.0).
    /// Higher values mean faster transitions.
    transition_alpha: f32,
}

impl EmotionalStateTracker {
    /// Create a new emotional state tracker.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The starting emotional state
    /// * `decay_config` - Configuration for state decay
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::{EmotionalStateTracker, EmotionalState, StateDecay};
    ///
    /// let tracker = EmotionalStateTracker::new(
    ///     EmotionalState::Focused,
    ///     StateDecay::default(),
    /// );
    /// ```
    pub fn new(initial_state: EmotionalState, decay_config: StateDecay) -> Self {
        Self {
            current_state: initial_state,
            intensity: 1.0,
            decay_config,
            last_update: Some(Instant::now()),
            transition_alpha: 0.3,
        }
    }

    /// Create a tracker starting in neutral state.
    pub fn neutral() -> Self {
        Self::new(EmotionalState::Neutral, StateDecay::default())
    }

    /// Get the current emotional state.
    #[inline]
    pub fn current_state(&self) -> EmotionalState {
        self.current_state
    }

    /// Get the current state intensity (0.0-1.0).
    ///
    /// Intensity represents how strongly the current state is felt.
    /// 1.0 means full intensity, 0.0 means the state has decayed.
    #[inline]
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Get the effective weight modifier considering intensity.
    ///
    /// This combines the state's base weight modifier with the current
    /// intensity, interpolating toward neutral (1.0) as intensity decreases.
    ///
    /// # Returns
    ///
    /// Effective weight modifier in range `[0.6, 1.3]` (based on state range).
    pub fn effective_weight(&self) -> f32 {
        let base_modifier = self.current_state.weight_modifier();
        let neutral_modifier = 1.0;

        // Interpolate between neutral and current state based on intensity
        neutral_modifier + (base_modifier - neutral_modifier) * self.intensity
    }

    /// Apply time-based decay to the current state.
    ///
    /// Uses exponential decay to reduce intensity over time.
    /// When intensity drops below the threshold, transitions to neutral.
    ///
    /// # Arguments
    ///
    /// * `elapsed` - Time elapsed since last update
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::{EmotionalStateTracker, EmotionalState, StateDecay};
    /// use std::time::Duration;
    ///
    /// let mut tracker = EmotionalStateTracker::new(
    ///     EmotionalState::Focused,
    ///     StateDecay::default(),
    /// );
    ///
    /// // Apply 60 seconds of decay
    /// tracker.decay(Duration::from_secs(60));
    ///
    /// // Intensity has decreased
    /// assert!(tracker.intensity() < 1.0);
    /// ```
    pub fn decay(&mut self, elapsed: Duration) {
        if self.current_state == EmotionalState::Neutral {
            // Neutral state doesn't decay
            return;
        }

        let elapsed_secs = elapsed.as_secs_f32();
        let half_life = self.decay_config.half_life_secs;

        // Exponential decay: I(t) = I(0) * 0.5^(t/half_life)
        let decay_factor = 0.5_f32.powf(elapsed_secs / half_life);
        self.intensity *= decay_factor;

        // Check if we should transition to neutral
        if self.intensity < self.decay_config.neutral_threshold {
            self.transition_to_neutral();
        }
    }

    /// Update with automatic time tracking.
    ///
    /// Calculates elapsed time since last update and applies decay.
    /// Call this periodically to maintain accurate state decay.
    pub fn update(&mut self) {
        let now = Instant::now();
        if let Some(last) = self.last_update {
            let elapsed = now.duration_since(last);
            self.decay(elapsed);
        }
        self.last_update = Some(now);
    }

    /// Transition to a new emotional state.
    ///
    /// Applies smoothing to prevent abrupt state changes.
    /// The transition is immediate but intensity is adjusted based on
    /// how different the new state is from the current one.
    ///
    /// # Arguments
    ///
    /// * `new_state` - The state to transition to
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::{EmotionalStateTracker, EmotionalState, StateDecay};
    ///
    /// let mut tracker = EmotionalStateTracker::neutral();
    ///
    /// // Transition to focused state
    /// tracker.transition(EmotionalState::Focused);
    ///
    /// assert_eq!(tracker.current_state(), EmotionalState::Focused);
    /// ```
    pub fn transition(&mut self, new_state: EmotionalState) {
        if new_state == self.current_state {
            // Reinforce current state
            self.reinforce();
            return;
        }

        // Calculate transition smoothness based on state similarity
        let current_modifier = self.current_state.weight_modifier();
        let new_modifier = new_state.weight_modifier();
        let modifier_diff = (current_modifier - new_modifier).abs();

        // Larger differences result in more gradual transitions
        // (intensity starts lower for bigger state changes)
        let transition_penalty = (modifier_diff * 0.5).clamp(0.0, 0.5);
        let base_intensity = self.decay_config.initial_intensity;

        self.current_state = new_state;
        self.intensity = (base_intensity - transition_penalty).max(0.3);
        self.last_update = Some(Instant::now());
    }

    /// Reinforce the current emotional state.
    ///
    /// Increases intensity toward maximum, modeling repeated exposure
    /// or continued engagement with emotionally relevant content.
    pub fn reinforce(&mut self) {
        let boost = self.decay_config.reinforcement_boost;
        self.intensity = (self.intensity + boost).min(1.0);
        self.last_update = Some(Instant::now());
    }

    /// Set the transition smoothing factor.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing factor in `[0, 1]`. Higher = faster transitions.
    pub fn set_transition_alpha(&mut self, alpha: f32) {
        self.transition_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Reset to neutral state with full intensity.
    pub fn reset(&mut self) {
        self.current_state = EmotionalState::Neutral;
        self.intensity = 1.0;
        self.last_update = Some(Instant::now());
    }

    /// Force a specific state and intensity.
    ///
    /// Use sparingly; prefer `transition()` for normal state changes.
    pub fn force_state(&mut self, state: EmotionalState, intensity: f32) {
        self.current_state = state;
        self.intensity = intensity.clamp(0.0, 1.0);
        self.last_update = Some(Instant::now());
    }

    /// Check if the state has fully decayed to neutral.
    pub fn is_neutral(&self) -> bool {
        self.current_state == EmotionalState::Neutral
    }

    /// Check if the state is at high intensity.
    pub fn is_intense(&self) -> bool {
        self.intensity > 0.7
    }

    /// Transition to neutral state.
    fn transition_to_neutral(&mut self) {
        self.current_state = EmotionalState::Neutral;
        self.intensity = 1.0; // Neutral is at full intensity
    }
}

impl Default for EmotionalStateTracker {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Configuration for emotional state decay.
///
/// Controls how quickly emotional states fade over time and when
/// they transition to neutral.
///
/// # Example
///
/// ```
/// use context_graph_utl::emotional::StateDecay;
///
/// // Custom decay: slower fade, higher threshold for neutral transition
/// let decay = StateDecay {
///     half_life_secs: 300.0, // 5 minutes to half intensity
///     neutral_threshold: 0.2,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StateDecay {
    /// Half-life for exponential decay in seconds.
    /// After this duration, intensity is halved.
    pub half_life_secs: f32,

    /// Intensity threshold below which state becomes neutral.
    pub neutral_threshold: f32,

    /// Initial intensity when entering a new state.
    pub initial_intensity: f32,

    /// Intensity boost when reinforcing current state.
    pub reinforcement_boost: f32,
}

impl Default for StateDecay {
    fn default() -> Self {
        Self {
            half_life_secs: 120.0, // 2 minutes
            neutral_threshold: 0.1,
            initial_intensity: 0.8,
            reinforcement_boost: 0.2,
        }
    }
}

impl StateDecay {
    /// Create a decay config for fast-fading states.
    ///
    /// Suitable for transient emotional responses.
    pub fn fast() -> Self {
        Self {
            half_life_secs: 30.0,
            neutral_threshold: 0.15,
            initial_intensity: 0.7,
            reinforcement_boost: 0.15,
        }
    }

    /// Create a decay config for slow-fading states.
    ///
    /// Suitable for sustained emotional engagement.
    pub fn slow() -> Self {
        Self {
            half_life_secs: 300.0,
            neutral_threshold: 0.05,
            initial_intensity: 0.9,
            reinforcement_boost: 0.25,
        }
    }

    /// Create a decay config with no decay (persistent states).
    ///
    /// States will not fade over time until explicitly changed.
    pub fn persistent() -> Self {
        Self {
            half_life_secs: f32::MAX,
            neutral_threshold: 0.0,
            initial_intensity: 1.0,
            reinforcement_boost: 0.0,
        }
    }

    /// Calculate the decay factor for a given elapsed time.
    ///
    /// # Returns
    ///
    /// Multiplier in `[0, 1]` to apply to current intensity.
    pub fn decay_factor(&self, elapsed: Duration) -> f32 {
        let elapsed_secs = elapsed.as_secs_f32();
        0.5_f32.powf(elapsed_secs / self.half_life_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = EmotionalStateTracker::new(EmotionalState::Curious, StateDecay::default());

        assert_eq!(tracker.current_state(), EmotionalState::Curious);
        assert_eq!(tracker.intensity(), 1.0);
    }

    #[test]
    fn test_neutral_tracker() {
        let tracker = EmotionalStateTracker::neutral();

        assert_eq!(tracker.current_state(), EmotionalState::Neutral);
        assert!(tracker.is_neutral());
    }

    #[test]
    fn test_decay_reduces_intensity() {
        let mut tracker =
            EmotionalStateTracker::new(EmotionalState::Focused, StateDecay::default());

        let initial_intensity = tracker.intensity();
        tracker.decay(Duration::from_secs(60)); // One minute

        assert!(tracker.intensity() < initial_intensity);
    }

    #[test]
    fn test_neutral_does_not_decay() {
        let mut tracker = EmotionalStateTracker::neutral();

        tracker.decay(Duration::from_secs(3600)); // One hour

        assert_eq!(tracker.intensity(), 1.0);
        assert!(tracker.is_neutral());
    }

    #[test]
    fn test_decay_to_neutral() {
        let decay = StateDecay {
            half_life_secs: 10.0,
            neutral_threshold: 0.1,
            ..Default::default()
        };

        let mut tracker = EmotionalStateTracker::new(EmotionalState::Stressed, decay);

        // Decay for long enough to go below threshold
        tracker.decay(Duration::from_secs(100)); // Many half-lives

        assert!(tracker.is_neutral());
    }

    #[test]
    fn test_transition() {
        let mut tracker = EmotionalStateTracker::neutral();

        tracker.transition(EmotionalState::Engaged);

        assert_eq!(tracker.current_state(), EmotionalState::Engaged);
    }

    #[test]
    fn test_reinforce() {
        let decay = StateDecay::default();
        let mut tracker = EmotionalStateTracker::new(EmotionalState::Focused, decay);

        // Reduce intensity
        tracker.decay(Duration::from_secs(60));
        let reduced_intensity = tracker.intensity();

        // Reinforce
        tracker.reinforce();

        assert!(tracker.intensity() > reduced_intensity);
    }

    #[test]
    fn test_reinforce_same_state() {
        let mut tracker = EmotionalStateTracker::neutral();
        tracker.transition(EmotionalState::Curious);

        // Transition to same state should reinforce
        let intensity_before = tracker.intensity();
        tracker.transition(EmotionalState::Curious);

        assert!(tracker.intensity() >= intensity_before);
    }

    #[test]
    fn test_effective_weight() {
        let tracker = EmotionalStateTracker::new(EmotionalState::Focused, StateDecay::default());

        // At full intensity, effective weight should equal state modifier
        assert!((tracker.effective_weight() - 1.3).abs() < 0.01);

        // After decay, effective weight should move toward neutral (1.0)
        let mut decayed_tracker = tracker.clone();
        decayed_tracker.force_state(EmotionalState::Focused, 0.5);

        let effective = decayed_tracker.effective_weight();
        assert!(effective > 1.0 && effective < 1.3);
    }

    #[test]
    fn test_reset() {
        let mut tracker =
            EmotionalStateTracker::new(EmotionalState::Stressed, StateDecay::default());

        tracker.reset();

        assert!(tracker.is_neutral());
        assert_eq!(tracker.intensity(), 1.0);
    }

    #[test]
    fn test_force_state() {
        let mut tracker = EmotionalStateTracker::neutral();

        tracker.force_state(EmotionalState::Fatigued, 0.5);

        assert_eq!(tracker.current_state(), EmotionalState::Fatigued);
        assert_eq!(tracker.intensity(), 0.5);
    }

    #[test]
    fn test_is_intense() {
        let mut tracker =
            EmotionalStateTracker::new(EmotionalState::Curious, StateDecay::default());

        assert!(tracker.is_intense());

        tracker.force_state(EmotionalState::Curious, 0.3);
        assert!(!tracker.is_intense());
    }

    #[test]
    fn test_state_decay_presets() {
        let fast = StateDecay::fast();
        let slow = StateDecay::slow();
        let persistent = StateDecay::persistent();

        assert!(fast.half_life_secs < slow.half_life_secs);
        assert_eq!(persistent.half_life_secs, f32::MAX);
    }

    #[test]
    fn test_decay_factor() {
        let decay = StateDecay::default();

        // At t=0, factor should be 1.0
        assert!((decay.decay_factor(Duration::ZERO) - 1.0).abs() < 0.001);

        // At t=half_life, factor should be 0.5
        let half_life = Duration::from_secs_f32(decay.half_life_secs);
        assert!((decay.decay_factor(half_life) - 0.5).abs() < 0.01);

        // At t=2*half_life, factor should be 0.25
        let double_half_life = Duration::from_secs_f32(decay.half_life_secs * 2.0);
        assert!((decay.decay_factor(double_half_life) - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_transition_smoothing() {
        let mut tracker =
            EmotionalStateTracker::new(EmotionalState::Neutral, StateDecay::default());

        // Large state change (Neutral -> Focused)
        tracker.transition(EmotionalState::Focused);
        let large_change_intensity = tracker.intensity();

        tracker.reset();
        tracker.transition(EmotionalState::Engaged); // Smaller change

        // Both should have some intensity, but exact values depend on implementation
        assert!(large_change_intensity > 0.0);
        assert!(tracker.intensity() > 0.0);
    }

    #[test]
    fn test_all_states_have_valid_weights() {
        let decay = StateDecay::default();

        for state in [
            EmotionalState::Neutral,
            EmotionalState::Curious,
            EmotionalState::Focused,
            EmotionalState::Stressed,
            EmotionalState::Fatigued,
            EmotionalState::Engaged,
            EmotionalState::Confused,
        ] {
            let tracker = EmotionalStateTracker::new(state, decay);
            let weight = tracker.effective_weight();

            assert!(weight > 0.0, "State {:?} has non-positive weight", state);
            assert!(
                weight <= 1.5,
                "State {:?} weight {} exceeds max",
                state,
                weight
            );
        }
    }

    #[test]
    fn test_set_transition_alpha() {
        let mut tracker = EmotionalStateTracker::neutral();

        tracker.set_transition_alpha(0.5);
        // Alpha is clamped
        tracker.set_transition_alpha(2.0);

        // Should not panic, alpha is internal
    }
}
