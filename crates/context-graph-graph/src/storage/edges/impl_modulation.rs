//! GraphEdge modulation and traversal methods.
//!
//! This module contains the canonical weight modulation formula and
//! traversal tracking functionality.

use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{Domain, GraphEdge, NeurotransmitterWeights};

impl GraphEdge {
    /// Get modulated weight for a specific query domain.
    ///
    /// # CANONICAL FORMULA
    ///
    /// ```text
    /// net_activation = excitatory - inhibitory + (modulatory * 0.5)
    /// domain_bonus = 0.1 if edge_domain == query_domain else 0.0
    /// steering_factor = 0.5 + steering_reward  // Range [0.5, 1.5]
    /// w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
    /// ```
    /// Result clamped to [0.0, 1.0]
    ///
    /// # Arguments
    /// * `query_domain` - The domain of the current query/traversal
    ///
    /// # Returns
    /// Effective weight after modulation, clamped to [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_graph::storage::edges::{GraphEdge, EdgeType, Domain};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(1, Uuid::new_v4(), Uuid::new_v4(), EdgeType::Semantic, 0.5, Domain::Code);
    /// edge.steering_reward = 0.5;  // steering_factor = 1.0
    ///
    /// // Query same domain gets bonus
    /// let w_code = edge.get_modulated_weight(Domain::Code);
    /// let w_legal = edge.get_modulated_weight(Domain::Legal);
    /// assert!(w_code > w_legal);  // Domain bonus applies
    /// ```
    #[inline]
    pub fn get_modulated_weight(&self, query_domain: Domain) -> f32 {
        let nt = &self.neurotransmitter_weights;

        // Net activation from NT weights
        let net_activation = nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5);

        // Domain bonus: +0.1 if query matches edge domain
        let domain_bonus = if self.domain == query_domain {
            0.1
        } else {
            0.0
        };

        // Steering factor: 0.5 + steering_reward puts it in [0.5, 1.5] range
        // (assuming steering_reward is [0.0, 1.0])
        let steering_factor = 0.5 + self.steering_reward;

        // Apply canonical formula
        let w_eff = self.weight * (1.0 + net_activation + domain_bonus) * steering_factor;

        // Clamp to valid range per AP-009
        w_eff.clamp(0.0, 1.0)
    }

    /// Get unmodulated base weight.
    #[inline]
    pub fn base_weight(&self) -> f32 {
        self.weight
    }

    /// Record a traversal of this edge.
    ///
    /// Updates traversal count, timestamp, and steering reward via EMA.
    ///
    /// # Arguments
    /// * `success` - Whether the traversal led to successful retrieval
    /// * `alpha` - EMA smoothing factor [0, 1], typical value 0.1
    ///
    /// # EMA Formula
    /// ```text
    /// reward = 1.0 if success else 0.0
    /// steering_reward = (1 - alpha) * steering_reward + alpha * reward
    /// ```
    pub fn record_traversal(&mut self, success: bool, alpha: f32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.traversal_count += 1;
        self.last_traversed_at = now;

        // Update steering reward via EMA
        let reward = if success { 1.0 } else { 0.0 };
        self.steering_reward = (1.0 - alpha) * self.steering_reward + alpha * reward;
        self.steering_reward = self.steering_reward.clamp(0.0, 1.0);
    }

    /// Record traversal with default EMA alpha (0.1).
    #[inline]
    pub fn record_traversal_default(&mut self, success: bool) {
        self.record_traversal(success, 0.1);
    }

    /// Mark this edge as an amortized shortcut (learned during dream consolidation).
    #[inline]
    pub fn mark_as_shortcut(&mut self) {
        self.is_amortized_shortcut = true;
    }

    /// Update confidence score.
    #[inline]
    pub fn update_confidence(&mut self, new_confidence: f32) {
        self.confidence = new_confidence.clamp(0.0, 1.0);
    }

    /// Update neurotransmitter weights.
    #[inline]
    pub fn update_nt_weights(&mut self, weights: NeurotransmitterWeights) {
        self.neurotransmitter_weights = weights;
    }

    /// Check if edge has been traversed since given timestamp.
    ///
    /// # Arguments
    /// * `since` - Unix timestamp to check against
    ///
    /// # Returns
    /// true if last_traversed_at > since
    #[inline]
    pub fn traversed_since(&self, since: u64) -> bool {
        self.last_traversed_at > since
    }

    /// Get edge "freshness" - seconds since last traversal.
    ///
    /// Returns u64::MAX if never traversed.
    pub fn freshness(&self) -> u64 {
        if self.last_traversed_at == 0 {
            return u64::MAX;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        now.saturating_sub(self.last_traversed_at)
    }

    /// Get composite score combining modulated weight and confidence.
    #[inline]
    pub fn composite_score(&self, query_domain: Domain) -> f32 {
        self.get_modulated_weight(query_domain) * self.confidence
    }
}
