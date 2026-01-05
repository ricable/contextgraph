//! Core GraphEdge implementation - constructors and factory methods.

use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

use super::types::{Domain, EdgeId, EdgeType, GraphEdge, NeurotransmitterWeights, NodeId};

impl GraphEdge {
    /// Create a new edge with domain-appropriate NT weights.
    ///
    /// # Arguments
    /// * `id` - Unique edge identifier
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `edge_type` - Type of relationship
    /// * `weight` - Base edge weight [0, 1]
    /// * `domain` - Cognitive domain
    ///
    /// # Returns
    /// New GraphEdge with:
    /// - NT weights from `NeurotransmitterWeights::for_domain(domain)`
    /// - confidence = 1.0
    /// - steering_reward = 0.5 (neutral)
    /// - traversal_count = 0
    /// - is_amortized_shortcut = false
    pub fn new(
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        edge_type: EdgeType,
        weight: f32,
        domain: Domain,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id,
            source,
            target,
            edge_type,
            weight: weight.clamp(0.0, 1.0),
            confidence: 1.0,
            domain,
            neurotransmitter_weights: NeurotransmitterWeights::for_domain(domain),
            is_amortized_shortcut: false,
            steering_reward: 0.5, // Neutral starting value
            traversal_count: 0,
            created_at: now,
            last_traversed_at: 0,
        }
    }

    /// Create a simple semantic edge with General domain.
    #[inline]
    pub fn semantic(id: EdgeId, source: NodeId, target: NodeId, weight: f32) -> Self {
        Self::new(
            id,
            source,
            target,
            EdgeType::Semantic,
            weight,
            Domain::General,
        )
    }

    /// Create a hierarchical edge with General domain.
    #[inline]
    pub fn hierarchical(id: EdgeId, parent: NodeId, child: NodeId, weight: f32) -> Self {
        Self::new(
            id,
            parent,
            child,
            EdgeType::Hierarchical,
            weight,
            Domain::General,
        )
    }

    /// Create a contradiction edge with inhibitory-heavy NT modulation.
    ///
    /// # M04-T26: NT Modulation for Contradiction Detection
    ///
    /// Contradiction edges use inhibitory-heavy NT profile to suppress
    /// contradicting content during retrieval:
    /// - excitatory: 0.2 (low activation)
    /// - inhibitory: 0.7 (high suppression)
    /// - modulatory: 0.1 (minimal modulation)
    ///
    /// Uses EdgeType::Contradicts default weight (0.3).
    ///
    /// # Arguments
    /// * `id` - Unique edge identifier
    /// * `source` - First node in contradiction pair
    /// * `target` - Second node in contradiction pair
    /// * `domain` - Cognitive domain of the contradiction
    ///
    /// # Returns
    /// GraphEdge with Contradicts type and inhibitory-heavy NT weights
    #[inline]
    pub fn contradiction(id: EdgeId, source: NodeId, target: NodeId, domain: Domain) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id,
            source,
            target,
            edge_type: EdgeType::Contradicts,
            weight: EdgeType::Contradicts.default_weight(), // 0.3
            confidence: 1.0,
            domain,
            // Inhibitory-heavy NT profile per M04-T26 spec
            neurotransmitter_weights: NeurotransmitterWeights {
                excitatory: 0.2,
                inhibitory: 0.7,
                modulatory: 0.1,
            },
            is_amortized_shortcut: false,
            steering_reward: 0.5,
            traversal_count: 0,
            created_at: now,
            last_traversed_at: 0,
        }
    }
}

impl Default for GraphEdge {
    fn default() -> Self {
        Self::new(
            0,
            Uuid::nil(),
            Uuid::nil(),
            EdgeType::Semantic,
            0.5,
            Domain::General,
        )
    }
}
