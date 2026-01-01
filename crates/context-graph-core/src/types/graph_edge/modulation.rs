//! Neurotransmitter and steering modulation methods for GraphEdge.

use super::edge::GraphEdge;

impl GraphEdge {
    /// Get the modulated weight considering NT weights and steering reward.
    ///
    /// # Formula
    /// ```text
    /// nt_factor = neurotransmitter_weights.compute_effective_weight(weight)
    /// modulated = (nt_factor * (1.0 + steering_reward * 0.2)).clamp(0.0, 1.0)
    /// ```
    ///
    /// Per constitution.yaml dopamine feedback: `pos: "+=r×0.2", neg: "-=|r|×0.1"`
    ///
    /// # Returns
    /// Effective weight in [0.0, 1.0], never NaN or Infinity.
    #[inline]
    pub fn get_modulated_weight(&self) -> f32 {
        let nt_factor = self
            .neurotransmitter_weights
            .compute_effective_weight(self.weight);
        (nt_factor * (1.0 + self.steering_reward * 0.2)).clamp(0.0, 1.0)
    }

    /// Apply a steering reward signal from the Steering Subsystem.
    ///
    /// Accumulates reward (additive), clamped to [-1.0, 1.0].
    /// Per constitution.yaml steering.reward: `range: "[-1,1]"`
    ///
    /// # Arguments
    /// * `reward` - Reward signal to add (positive reinforces, negative discourages)
    #[inline]
    pub fn apply_steering_reward(&mut self, reward: f32) {
        self.steering_reward = (self.steering_reward + reward).clamp(-1.0, 1.0);
    }

    /// Decay the steering reward by a factor.
    ///
    /// Used to gradually reduce influence of old rewards over time.
    /// Does NOT clamp - assumes decay_factor is in [0.0, 1.0].
    ///
    /// # Arguments
    /// * `decay_factor` - Multiplicative decay (e.g., 0.9 reduces by 10%)
    #[inline]
    pub fn decay_steering(&mut self, decay_factor: f32) {
        self.steering_reward *= decay_factor;
    }
}
