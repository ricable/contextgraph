//! Graph edge connecting two memory nodes with Marblestone architecture support.
//!
//! This module provides the GraphEdge struct which represents directed relationships
//! between MemoryNodes in the Context Graph. It implements the Marblestone architecture
//! features for neurotransmitter-based weight modulation, amortized shortcuts, and
//! steering rewards.
//!
//! # Constitution Reference
//! - edge_model: Full edge specification
//! - edge_model.nt_weights: Neurotransmitter modulation formula
//! - edge_model.amortized: Shortcut learning criteria
//! - edge_model.steering_reward: [-1,1] reward signal

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::NodeId;
use crate::marblestone::{Domain, EdgeType, NeurotransmitterWeights};

/// Type alias for edge identifiers (UUID v4).
pub type EdgeId = Uuid;

/// A directed edge between two nodes in the Context Graph.
///
/// Implements Marblestone architecture features:
/// - Neurotransmitter-based weight modulation
/// - Amortized shortcuts (learned during dream consolidation)
/// - Steering rewards for reinforcement learning
///
/// # Fields
/// All 13 fields per PRD Section 4.2 and constitution.yaml edge_model:
/// - `id`: Unique edge identifier (UUID v4)
/// - `source_id`: Source node UUID
/// - `target_id`: Target node UUID
/// - `edge_type`: Relationship type (Semantic|Temporal|Causal|Hierarchical)
/// - `weight`: Base edge weight [0.0, 1.0]
/// - `confidence`: Confidence in validity [0.0, 1.0]
/// - `domain`: Knowledge domain for context-aware retrieval
/// - `neurotransmitter_weights`: NT modulation weights
/// - `is_amortized_shortcut`: True if learned during dream consolidation
/// - `steering_reward`: Steering Subsystem feedback [-1.0, 1.0]
/// - `traversal_count`: Number of times edge was traversed
/// - `created_at`: Creation timestamp
/// - `last_traversed_at`: Last traversal timestamp (None until first traversal)
///
/// # Performance Characteristics
/// - Serialized size: ~200 bytes
/// - Traversal latency target: <50μs
///
/// # Example
/// ```rust
/// use context_graph_core::types::GraphEdge;
/// use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
/// use uuid::Uuid;
///
/// let source = Uuid::new_v4();
/// let target = Uuid::new_v4();
///
/// // Create edge with all Marblestone fields
/// let edge = GraphEdge {
///     id: Uuid::new_v4(),
///     source_id: source,
///     target_id: target,
///     edge_type: EdgeType::Causal,
///     weight: 0.8,
///     confidence: 0.9,
///     domain: Domain::Code,
///     neurotransmitter_weights: NeurotransmitterWeights::for_domain(Domain::Code),
///     is_amortized_shortcut: false,
///     steering_reward: 0.0,
///     traversal_count: 0,
///     created_at: chrono::Utc::now(),
///     last_traversed_at: None,
/// };
///
/// assert!(edge.weight >= 0.0 && edge.weight <= 1.0);
/// assert!(edge.confidence >= 0.0 && edge.confidence <= 1.0);
/// assert!(edge.steering_reward >= -1.0 && edge.steering_reward <= 1.0);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique identifier for this edge (UUID v4).
    pub id: EdgeId,

    /// Source node ID (edge starts here).
    pub source_id: NodeId,

    /// Target node ID (edge ends here).
    pub target_id: NodeId,

    /// Type of relationship this edge represents.
    pub edge_type: EdgeType,

    /// Base weight of the edge [0.0, 1.0].
    /// Higher weight = stronger connection.
    pub weight: f32,

    /// Confidence in this edge's validity [0.0, 1.0].
    /// Higher confidence = more reliable relationship.
    pub confidence: f32,

    /// Knowledge domain this edge belongs to.
    /// Used for context-aware retrieval weighting.
    pub domain: Domain,

    /// Neurotransmitter weights for modulation.
    /// Applied via: w_eff = base × (1 + excitatory - inhibitory + 0.5×modulatory)
    pub neurotransmitter_weights: NeurotransmitterWeights,

    /// Whether this edge is an amortized shortcut (learned during dreams).
    /// Shortcuts are created when 3+ hop paths are traversed ≥5 times.
    pub is_amortized_shortcut: bool,

    /// Steering reward signal from the Steering Subsystem [-1.0, 1.0].
    /// Positive = reinforce, Negative = discourage, Zero = neutral.
    pub steering_reward: f32,

    /// Number of times this edge has been traversed.
    /// Used for amortized shortcut detection.
    pub traversal_count: u64,

    /// Timestamp when this edge was created.
    pub created_at: DateTime<Utc>,

    /// Timestamp when this edge was last traversed.
    /// None until the first traversal occurs.
    pub last_traversed_at: Option<DateTime<Utc>>,
}

impl GraphEdge {
    /// Create a new edge with default values for the given domain.
    ///
    /// # Arguments
    /// * `source_id` - Source node UUID
    /// * `target_id` - Target node UUID
    /// * `edge_type` - Type of relationship
    /// * `domain` - Knowledge domain (determines NT weights)
    ///
    /// # Returns
    /// New GraphEdge with:
    /// - weight = edge_type.default_weight()
    /// - confidence = 0.5
    /// - neurotransmitter_weights = NeurotransmitterWeights::for_domain(domain)
    /// - steering_reward = 0.0
    /// - traversal_count = 0
    /// - is_amortized_shortcut = false
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Causal,
    ///     Domain::Code,
    /// );
    /// assert_eq!(edge.weight, 0.8); // Causal default
    /// ```
    pub fn new(source_id: NodeId, target_id: NodeId, edge_type: EdgeType, domain: Domain) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            source_id,
            target_id,
            edge_type,
            weight: edge_type.default_weight(),
            confidence: 0.5,
            domain,
            neurotransmitter_weights: NeurotransmitterWeights::for_domain(domain),
            is_amortized_shortcut: false,
            steering_reward: 0.0,
            traversal_count: 0,
            created_at: now,
            last_traversed_at: None,
        }
    }

    /// Create a new edge with explicit weight and confidence.
    ///
    /// # Arguments
    /// * `source_id` - Source node UUID
    /// * `target_id` - Target node UUID
    /// * `edge_type` - Type of relationship
    /// * `domain` - Knowledge domain
    /// * `weight` - Base edge weight (will be clamped to [0.0, 1.0])
    /// * `confidence` - Confidence level (will be clamped to [0.0, 1.0])
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let edge = GraphEdge::with_weight(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::Research,
    ///     0.75,
    ///     0.95,
    /// );
    /// assert_eq!(edge.weight, 0.75);
    /// assert_eq!(edge.confidence, 0.95);
    /// ```
    pub fn with_weight(
        source_id: NodeId,
        target_id: NodeId,
        edge_type: EdgeType,
        domain: Domain,
        weight: f32,
        confidence: f32,
    ) -> Self {
        let mut edge = Self::new(source_id, target_id, edge_type, domain);
        edge.weight = weight.clamp(0.0, 1.0);
        edge.confidence = confidence.clamp(0.0, 1.0);
        edge
    }

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
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    /// edge.steering_reward = 0.5;
    /// let modulated = edge.get_modulated_weight();
    /// assert!(modulated >= 0.0 && modulated <= 1.0);
    /// ```
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
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    /// edge.apply_steering_reward(0.5);
    /// assert_eq!(edge.steering_reward, 0.5);
    /// edge.apply_steering_reward(0.8);
    /// assert_eq!(edge.steering_reward, 1.0); // Clamped
    /// ```
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
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    /// edge.steering_reward = 1.0;
    /// edge.decay_steering(0.5);
    /// assert_eq!(edge.steering_reward, 0.5);
    /// ```
    #[inline]
    pub fn decay_steering(&mut self, decay_factor: f32) {
        self.steering_reward *= decay_factor;
    }

    /// Record a traversal of this edge.
    ///
    /// Updates traversal_count (saturating) and last_traversed_at timestamp.
    /// Per constitution.yaml dream.amortized: `trigger: "3+ hop path traversed ≥5×"`
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    /// assert!(edge.last_traversed_at.is_none());
    /// edge.record_traversal();
    /// assert_eq!(edge.traversal_count, 1);
    /// assert!(edge.last_traversed_at.is_some());
    /// ```
    #[inline]
    pub fn record_traversal(&mut self) {
        self.traversal_count = self.traversal_count.saturating_add(1);
        self.last_traversed_at = Some(Utc::now());
    }

    /// Check if this edge is a reliable amortized shortcut.
    ///
    /// Per constitution.yaml edge_model.amortized:
    /// - `trigger: "3+ hop path traversed ≥5×"` (we check traversal_count >= 3)
    /// - `confidence: "≥0.7"`
    /// - `steering_reward > 0.3` (positive feedback)
    /// - `is_amortized_shortcut == true`
    ///
    /// # Returns
    /// `true` if ALL conditions are met:
    /// 1. `is_amortized_shortcut == true`
    /// 2. `traversal_count >= 3`
    /// 3. `steering_reward > 0.3`
    /// 4. `confidence >= 0.7`
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    /// assert!(!edge.is_reliable_shortcut()); // Not a shortcut
    ///
    /// edge.is_amortized_shortcut = true;
    /// edge.traversal_count = 5;
    /// edge.steering_reward = 0.5;
    /// edge.confidence = 0.8;
    /// assert!(edge.is_reliable_shortcut());
    /// ```
    #[inline]
    pub fn is_reliable_shortcut(&self) -> bool {
        self.is_amortized_shortcut
            && self.traversal_count >= 3
            && self.steering_reward > 0.3
            && self.confidence >= 0.7
    }

    /// Mark this edge as an amortized shortcut.
    ///
    /// Called during dream consolidation when a 3+ hop path has been
    /// traversed enough times to warrant a direct shortcut edge.
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    /// assert!(!edge.is_amortized_shortcut);
    /// edge.mark_as_shortcut();
    /// assert!(edge.is_amortized_shortcut);
    /// ```
    #[inline]
    pub fn mark_as_shortcut(&mut self) {
        self.is_amortized_shortcut = true;
    }

    /// Get the age of this edge in seconds since creation.
    ///
    /// # Returns
    /// Number of seconds since created_at. Always >= 0.
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let edge = GraphEdge::new(
    ///     Uuid::new_v4(),
    ///     Uuid::new_v4(),
    ///     EdgeType::Semantic,
    ///     Domain::General,
    /// );
    /// let age = edge.age_seconds();
    /// assert!(age >= 0);
    /// ```
    #[inline]
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }
}

impl Default for GraphEdge {
    /// Creates a default edge with nil UUIDs.
    ///
    /// Uses:
    /// - source_id/target_id: Uuid::nil()
    /// - edge_type: EdgeType::Semantic (default)
    /// - domain: Domain::General (default)
    ///
    /// # Example
    /// ```rust
    /// use context_graph_core::types::GraphEdge;
    /// use context_graph_core::marblestone::{Domain, EdgeType};
    /// use uuid::Uuid;
    ///
    /// let edge = GraphEdge::default();
    /// assert_eq!(edge.source_id, Uuid::nil());
    /// assert_eq!(edge.target_id, Uuid::nil());
    /// assert_eq!(edge.edge_type, EdgeType::Semantic);
    /// assert_eq!(edge.domain, Domain::General);
    /// ```
    fn default() -> Self {
        Self::new(
            Uuid::nil(),
            Uuid::nil(),
            EdgeType::default(),
            Domain::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // =========================================================================
    // Struct Field Existence Tests
    // =========================================================================

    #[test]
    fn test_graph_edge_has_all_13_fields() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();

        // This test verifies all 13 fields compile and are accessible
        let edge = GraphEdge {
            id: Uuid::new_v4(),
            source_id: source,
            target_id: target,
            edge_type: EdgeType::Semantic,
            weight: 0.5,
            confidence: 0.8,
            domain: Domain::General,
            neurotransmitter_weights: NeurotransmitterWeights::default(),
            is_amortized_shortcut: false,
            steering_reward: 0.0,
            traversal_count: 0,
            created_at: Utc::now(),
            last_traversed_at: None,
        };

        // Verify all fields are accessible
        let _id: EdgeId = edge.id;
        let _src: NodeId = edge.source_id;
        let _tgt: NodeId = edge.target_id;
        let _et: EdgeType = edge.edge_type;
        let _w: f32 = edge.weight;
        let _c: f32 = edge.confidence;
        let _d: Domain = edge.domain;
        let _nt: NeurotransmitterWeights = edge.neurotransmitter_weights;
        let _short: bool = edge.is_amortized_shortcut;
        let _sr: f32 = edge.steering_reward;
        let _tc: u64 = edge.traversal_count;
        let _ca: DateTime<Utc> = edge.created_at;
        let _lt: Option<DateTime<Utc>> = edge.last_traversed_at;
    }

    #[test]
    fn test_edge_id_is_uuid() {
        let edge_id: EdgeId = Uuid::new_v4();
        assert_eq!(edge_id.get_version_num(), 4);
    }

    // =========================================================================
    // Field Type Tests
    // =========================================================================

    #[test]
    fn test_source_id_is_node_id() {
        let source: NodeId = Uuid::new_v4();
        let edge = create_test_edge();
        let _: NodeId = edge.source_id;
        assert_ne!(source, edge.source_id); // Just verifying type compatibility
    }

    #[test]
    fn test_target_id_is_node_id() {
        let edge = create_test_edge();
        let _: NodeId = edge.target_id;
    }

    #[test]
    fn test_edge_type_uses_marblestone_enum() {
        let edge = create_test_edge();
        // Verify it's the Marblestone EdgeType (has default_weight method)
        let _weight = edge.edge_type.default_weight();
    }

    #[test]
    fn test_domain_uses_marblestone_enum() {
        let edge = create_test_edge();
        // Verify it's the Marblestone Domain (has description method)
        let _desc = edge.domain.description();
    }

    #[test]
    fn test_nt_weights_uses_marblestone_struct() {
        let edge = create_test_edge();
        // Verify it's the Marblestone NeurotransmitterWeights
        let _eff = edge.neurotransmitter_weights.compute_effective_weight(0.5);
    }

    // =========================================================================
    // Serde Serialization Tests
    // =========================================================================

    #[test]
    fn test_serde_roundtrip() {
        let edge = create_test_edge();
        let json = serde_json::to_string(&edge).expect("serialize failed");
        let restored: GraphEdge = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(edge, restored);
    }

    #[test]
    fn test_serde_json_contains_all_fields() {
        let edge = create_test_edge();
        let json = serde_json::to_string(&edge).unwrap();

        // Verify all field names appear in JSON
        assert!(json.contains("\"id\""), "JSON missing id field");
        assert!(
            json.contains("\"source_id\""),
            "JSON missing source_id field"
        );
        assert!(
            json.contains("\"target_id\""),
            "JSON missing target_id field"
        );
        assert!(
            json.contains("\"edge_type\""),
            "JSON missing edge_type field"
        );
        assert!(json.contains("\"weight\""), "JSON missing weight field");
        assert!(
            json.contains("\"confidence\""),
            "JSON missing confidence field"
        );
        assert!(json.contains("\"domain\""), "JSON missing domain field");
        assert!(
            json.contains("\"neurotransmitter_weights\""),
            "JSON missing neurotransmitter_weights field"
        );
        assert!(
            json.contains("\"is_amortized_shortcut\""),
            "JSON missing is_amortized_shortcut field"
        );
        assert!(
            json.contains("\"steering_reward\""),
            "JSON missing steering_reward field"
        );
        assert!(
            json.contains("\"traversal_count\""),
            "JSON missing traversal_count field"
        );
        assert!(
            json.contains("\"created_at\""),
            "JSON missing created_at field"
        );
        assert!(
            json.contains("\"last_traversed_at\""),
            "JSON missing last_traversed_at field"
        );
    }

    #[test]
    fn test_serde_with_last_traversed_at_some() {
        let mut edge = create_test_edge();
        edge.last_traversed_at = Some(Utc::now());

        let json = serde_json::to_string(&edge).unwrap();
        let restored: GraphEdge = serde_json::from_str(&json).unwrap();

        assert!(restored.last_traversed_at.is_some());
    }

    #[test]
    fn test_serde_with_last_traversed_at_none() {
        let edge = create_test_edge();
        assert!(edge.last_traversed_at.is_none());

        let json = serde_json::to_string(&edge).unwrap();
        let restored: GraphEdge = serde_json::from_str(&json).unwrap();

        assert!(restored.last_traversed_at.is_none());
    }

    #[test]
    fn test_serde_edge_type_snake_case() {
        let mut edge = create_test_edge();
        edge.edge_type = EdgeType::Hierarchical;

        let json = serde_json::to_string(&edge).unwrap();
        assert!(
            json.contains("\"hierarchical\""),
            "EdgeType should serialize to snake_case"
        );
    }

    #[test]
    fn test_serde_domain_snake_case() {
        let mut edge = create_test_edge();
        edge.domain = Domain::Medical;

        let json = serde_json::to_string(&edge).unwrap();
        assert!(
            json.contains("\"medical\""),
            "Domain should serialize to snake_case"
        );
    }

    // =========================================================================
    // Derive Trait Tests
    // =========================================================================

    #[test]
    fn test_debug_format() {
        let edge = create_test_edge();
        let debug = format!("{:?}", edge);
        assert!(debug.contains("GraphEdge"));
        assert!(debug.contains("source_id"));
        assert!(debug.contains("target_id"));
    }

    #[test]
    fn test_clone() {
        let edge = create_test_edge();
        let cloned = edge.clone();
        assert_eq!(edge, cloned);
    }

    #[test]
    fn test_partial_eq() {
        let edge1 = create_test_edge();
        let edge2 = edge1.clone();
        assert_eq!(edge1, edge2);

        let mut edge3 = edge1.clone();
        edge3.weight = 0.9;
        assert_ne!(edge1, edge3);
    }

    // =========================================================================
    // Field Value Range Tests
    // =========================================================================

    #[test]
    fn test_weight_boundary_zero() {
        let mut edge = create_test_edge();
        edge.weight = 0.0;
        assert_eq!(edge.weight, 0.0);
    }

    #[test]
    fn test_weight_boundary_one() {
        let mut edge = create_test_edge();
        edge.weight = 1.0;
        assert_eq!(edge.weight, 1.0);
    }

    #[test]
    fn test_confidence_boundary_zero() {
        let mut edge = create_test_edge();
        edge.confidence = 0.0;
        assert_eq!(edge.confidence, 0.0);
    }

    #[test]
    fn test_confidence_boundary_one() {
        let mut edge = create_test_edge();
        edge.confidence = 1.0;
        assert_eq!(edge.confidence, 1.0);
    }

    #[test]
    fn test_steering_reward_boundary_negative_one() {
        let mut edge = create_test_edge();
        edge.steering_reward = -1.0;
        assert_eq!(edge.steering_reward, -1.0);
    }

    #[test]
    fn test_steering_reward_boundary_positive_one() {
        let mut edge = create_test_edge();
        edge.steering_reward = 1.0;
        assert_eq!(edge.steering_reward, 1.0);
    }

    #[test]
    fn test_steering_reward_zero_is_neutral() {
        let edge = create_test_edge();
        assert_eq!(edge.steering_reward, 0.0);
    }

    #[test]
    fn test_traversal_count_starts_at_zero() {
        let edge = create_test_edge();
        assert_eq!(edge.traversal_count, 0);
    }

    #[test]
    fn test_is_amortized_shortcut_defaults_false() {
        let edge = create_test_edge();
        assert!(!edge.is_amortized_shortcut);
    }

    #[test]
    fn test_is_amortized_shortcut_can_be_true() {
        let mut edge = create_test_edge();
        edge.is_amortized_shortcut = true;
        assert!(edge.is_amortized_shortcut);
    }

    // =========================================================================
    // All EdgeType Variants Test
    // =========================================================================

    #[test]
    fn test_all_edge_types_work() {
        for edge_type in EdgeType::all() {
            let mut edge = create_test_edge();
            edge.edge_type = edge_type;

            let json = serde_json::to_string(&edge).unwrap();
            let restored: GraphEdge = serde_json::from_str(&json).unwrap();
            assert_eq!(restored.edge_type, edge_type);
        }
    }

    // =========================================================================
    // All Domain Variants Test
    // =========================================================================

    #[test]
    fn test_all_domains_work() {
        for domain in Domain::all() {
            let mut edge = create_test_edge();
            edge.domain = domain;
            edge.neurotransmitter_weights = NeurotransmitterWeights::for_domain(domain);

            let json = serde_json::to_string(&edge).unwrap();
            let restored: GraphEdge = serde_json::from_str(&json).unwrap();
            assert_eq!(restored.domain, domain);
        }
    }

    // =========================================================================
    // Timestamp Tests
    // =========================================================================

    #[test]
    fn test_created_at_is_required() {
        let edge = create_test_edge();
        let _: DateTime<Utc> = edge.created_at;
    }

    #[test]
    fn test_timestamps_preserved_through_serde() {
        let edge = create_test_edge();
        let original_created = edge.created_at;

        let json = serde_json::to_string(&edge).unwrap();
        let restored: GraphEdge = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.created_at, original_created);
    }

    // =========================================================================
    // UUID Tests
    // =========================================================================

    #[test]
    fn test_id_is_v4_uuid() {
        let edge = create_test_edge();
        assert_eq!(edge.id.get_version_num(), 4);
    }

    #[test]
    fn test_source_and_target_are_different() {
        let edge = create_test_edge();
        assert_ne!(
            edge.source_id, edge.target_id,
            "Source and target should be different UUIDs"
        );
    }

    // =========================================================================
    // Helper Function
    // =========================================================================

    fn create_test_edge() -> GraphEdge {
        GraphEdge {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            target_id: Uuid::new_v4(),
            edge_type: EdgeType::Semantic,
            weight: 0.5,
            confidence: 0.8,
            domain: Domain::General,
            neurotransmitter_weights: NeurotransmitterWeights::default(),
            is_amortized_shortcut: false,
            steering_reward: 0.0,
            traversal_count: 0,
            created_at: Utc::now(),
            last_traversed_at: None,
        }
    }

    // =========================================================================
    // GraphEdge Method Tests (TASK-M02-011)
    // =========================================================================

    // --- new() Constructor Tests ---

    #[test]
    fn test_new_creates_edge_with_domain_nt_weights() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = GraphEdge::new(source, target, EdgeType::Semantic, Domain::Code);

        let expected_nt = NeurotransmitterWeights::for_domain(Domain::Code);
        assert_eq!(edge.neurotransmitter_weights, expected_nt);
    }

    #[test]
    fn test_new_uses_edge_type_default_weight() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Causal,
            Domain::General,
        );
        assert_eq!(edge.weight, EdgeType::Causal.default_weight());
        assert_eq!(edge.weight, 0.8);
    }

    #[test]
    fn test_new_sets_confidence_to_half() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert_eq!(edge.confidence, 0.5);
    }

    #[test]
    fn test_new_sets_steering_reward_to_zero() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert_eq!(edge.steering_reward, 0.0);
    }

    #[test]
    fn test_new_sets_traversal_count_to_zero() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert_eq!(edge.traversal_count, 0);
    }

    #[test]
    fn test_new_sets_is_amortized_shortcut_false() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert!(!edge.is_amortized_shortcut);
    }

    #[test]
    fn test_new_sets_last_traversed_at_none() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert!(edge.last_traversed_at.is_none());
    }

    #[test]
    fn test_new_generates_unique_id() {
        let edge1 = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        let edge2 = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert_ne!(edge1.id, edge2.id);
    }

    #[test]
    fn test_new_all_edge_types() {
        for edge_type in EdgeType::all() {
            let edge = GraphEdge::new(Uuid::new_v4(), Uuid::new_v4(), edge_type, Domain::General);
            assert_eq!(edge.weight, edge_type.default_weight());
        }
    }

    #[test]
    fn test_new_all_domains() {
        for domain in Domain::all() {
            let edge = GraphEdge::new(Uuid::new_v4(), Uuid::new_v4(), EdgeType::Semantic, domain);
            assert_eq!(edge.domain, domain);
            assert_eq!(
                edge.neurotransmitter_weights,
                NeurotransmitterWeights::for_domain(domain)
            );
        }
    }

    // --- with_weight() Constructor Tests ---

    #[test]
    fn test_with_weight_sets_explicit_values() {
        let edge = GraphEdge::with_weight(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
            0.75,
            0.95,
        );
        assert_eq!(edge.weight, 0.75);
        assert_eq!(edge.confidence, 0.95);
    }

    #[test]
    fn test_with_weight_clamps_weight_high() {
        let edge = GraphEdge::with_weight(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
            1.5,
            0.5,
        );
        assert_eq!(edge.weight, 1.0);
    }

    #[test]
    fn test_with_weight_clamps_weight_low() {
        let edge = GraphEdge::with_weight(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
            -0.5,
            0.5,
        );
        assert_eq!(edge.weight, 0.0);
    }

    #[test]
    fn test_with_weight_clamps_confidence_high() {
        let edge = GraphEdge::with_weight(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
            0.5,
            1.5,
        );
        assert_eq!(edge.confidence, 1.0);
    }

    #[test]
    fn test_with_weight_clamps_confidence_low() {
        let edge = GraphEdge::with_weight(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
            0.5,
            -0.5,
        );
        assert_eq!(edge.confidence, 0.0);
    }

    // --- get_modulated_weight() Tests ---

    #[test]
    fn test_get_modulated_weight_applies_nt() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        // General NT: excitatory=0.5, inhibitory=0.2, modulatory=0.3
        // Base weight for Semantic: 0.5
        // NT factor: (0.5*0.5 - 0.5*0.2) * (1 + (0.3-0.5)*0.4) = 0.15 * 0.92 = 0.138
        // With steering_reward=0: 0.138 * (1 + 0*0.2) = 0.138
        let modulated = edge.get_modulated_weight();
        assert!(
            (modulated - 0.138).abs() < 0.001,
            "Expected ~0.138, got {}",
            modulated
        );
    }

    #[test]
    fn test_get_modulated_weight_applies_steering_positive() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.steering_reward = 1.0;
        // NT factor ~0.138 (from above)
        // With steering_reward=1.0: 0.138 * (1 + 1.0*0.2) = 0.138 * 1.2 = 0.1656
        let modulated = edge.get_modulated_weight();
        assert!(
            modulated > 0.138,
            "Positive steering should increase weight"
        );
    }

    #[test]
    fn test_get_modulated_weight_applies_steering_negative() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.steering_reward = -1.0;
        // NT factor ~0.138
        // With steering_reward=-1.0: 0.138 * (1 + (-1.0)*0.2) = 0.138 * 0.8 = 0.1104
        let modulated = edge.get_modulated_weight();
        assert!(
            modulated < 0.138,
            "Negative steering should decrease weight"
        );
    }

    #[test]
    fn test_get_modulated_weight_always_in_range() {
        for domain in Domain::all() {
            for edge_type in EdgeType::all() {
                for sr in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
                    let mut edge =
                        GraphEdge::new(Uuid::new_v4(), Uuid::new_v4(), edge_type, domain);
                    edge.steering_reward = sr;
                    let modulated = edge.get_modulated_weight();
                    assert!(
                        modulated >= 0.0 && modulated <= 1.0,
                        "Out of range: domain={:?}, edge_type={:?}, sr={}, modulated={}",
                        domain,
                        edge_type,
                        sr,
                        modulated
                    );
                }
            }
        }
    }

    // --- apply_steering_reward() Tests ---

    #[test]
    fn test_apply_steering_reward_adds() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.apply_steering_reward(0.3);
        assert_eq!(edge.steering_reward, 0.3);
        edge.apply_steering_reward(0.2);
        assert_eq!(edge.steering_reward, 0.5);
    }

    #[test]
    fn test_apply_steering_reward_clamps_positive() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.apply_steering_reward(0.8);
        edge.apply_steering_reward(0.5);
        assert_eq!(edge.steering_reward, 1.0);
    }

    #[test]
    fn test_apply_steering_reward_clamps_negative() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.apply_steering_reward(-0.8);
        edge.apply_steering_reward(-0.5);
        assert_eq!(edge.steering_reward, -1.0);
    }

    #[test]
    fn test_apply_steering_reward_handles_negative() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.apply_steering_reward(-0.5);
        assert_eq!(edge.steering_reward, -0.5);
    }

    // --- decay_steering() Tests ---

    #[test]
    fn test_decay_steering_multiplies() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.steering_reward = 1.0;
        edge.decay_steering(0.5);
        assert_eq!(edge.steering_reward, 0.5);
    }

    #[test]
    fn test_decay_steering_to_zero() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.steering_reward = 0.5;
        edge.decay_steering(0.0);
        assert_eq!(edge.steering_reward, 0.0);
    }

    #[test]
    fn test_decay_steering_negative() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.steering_reward = -1.0;
        edge.decay_steering(0.5);
        assert_eq!(edge.steering_reward, -0.5);
    }

    // --- record_traversal() Tests ---

    #[test]
    fn test_record_traversal_increments_count() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert_eq!(edge.traversal_count, 0);
        edge.record_traversal();
        assert_eq!(edge.traversal_count, 1);
        edge.record_traversal();
        assert_eq!(edge.traversal_count, 2);
    }

    #[test]
    fn test_record_traversal_updates_timestamp() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert!(edge.last_traversed_at.is_none());
        edge.record_traversal();
        assert!(edge.last_traversed_at.is_some());
    }

    #[test]
    fn test_record_traversal_saturates() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.traversal_count = u64::MAX;
        edge.record_traversal();
        assert_eq!(edge.traversal_count, u64::MAX);
    }

    // --- is_reliable_shortcut() Tests ---

    #[test]
    fn test_is_reliable_shortcut_all_conditions_met() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = true;
        edge.traversal_count = 5;
        edge.steering_reward = 0.5;
        edge.confidence = 0.8;
        assert!(edge.is_reliable_shortcut());
    }

    #[test]
    fn test_is_reliable_shortcut_fails_not_shortcut() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = false; // Fails here
        edge.traversal_count = 5;
        edge.steering_reward = 0.5;
        edge.confidence = 0.8;
        assert!(!edge.is_reliable_shortcut());
    }

    #[test]
    fn test_is_reliable_shortcut_fails_low_traversal() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = true;
        edge.traversal_count = 2; // Fails here (need >= 3)
        edge.steering_reward = 0.5;
        edge.confidence = 0.8;
        assert!(!edge.is_reliable_shortcut());
    }

    #[test]
    fn test_is_reliable_shortcut_fails_low_reward() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = true;
        edge.traversal_count = 5;
        edge.steering_reward = 0.2; // Fails here (need > 0.3)
        edge.confidence = 0.8;
        assert!(!edge.is_reliable_shortcut());
    }

    #[test]
    fn test_is_reliable_shortcut_fails_low_confidence() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = true;
        edge.traversal_count = 5;
        edge.steering_reward = 0.5;
        edge.confidence = 0.6; // Fails here (need >= 0.7)
        assert!(!edge.is_reliable_shortcut());
    }

    #[test]
    fn test_is_reliable_shortcut_boundary_traversal() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = true;
        edge.traversal_count = 3; // Exactly 3
        edge.steering_reward = 0.5;
        edge.confidence = 0.8;
        assert!(edge.is_reliable_shortcut());
    }

    #[test]
    fn test_is_reliable_shortcut_boundary_reward() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = true;
        edge.traversal_count = 5;
        edge.steering_reward = 0.3; // Exactly 0.3 - should FAIL (need > 0.3)
        edge.confidence = 0.8;
        assert!(!edge.is_reliable_shortcut());
    }

    #[test]
    fn test_is_reliable_shortcut_boundary_confidence() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.is_amortized_shortcut = true;
        edge.traversal_count = 5;
        edge.steering_reward = 0.5;
        edge.confidence = 0.7; // Exactly 0.7
        assert!(edge.is_reliable_shortcut());
    }

    // --- mark_as_shortcut() Tests ---

    #[test]
    fn test_mark_as_shortcut() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert!(!edge.is_amortized_shortcut);
        edge.mark_as_shortcut();
        assert!(edge.is_amortized_shortcut);
    }

    #[test]
    fn test_mark_as_shortcut_idempotent() {
        let mut edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        edge.mark_as_shortcut();
        edge.mark_as_shortcut();
        assert!(edge.is_amortized_shortcut);
    }

    // --- age_seconds() Tests ---

    #[test]
    fn test_age_seconds_non_negative() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        assert!(edge.age_seconds() >= 0);
    }

    #[test]
    fn test_age_seconds_increases() {
        let edge = GraphEdge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            Domain::General,
        );
        let age1 = edge.age_seconds();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let age2 = edge.age_seconds();
        assert!(age2 >= age1);
    }

    // --- Default Trait Tests ---

    #[test]
    fn test_default_uses_nil_uuids() {
        let edge = GraphEdge::default();
        assert_eq!(edge.source_id, Uuid::nil());
        assert_eq!(edge.target_id, Uuid::nil());
    }

    #[test]
    fn test_default_uses_semantic_edge_type() {
        let edge = GraphEdge::default();
        assert_eq!(edge.edge_type, EdgeType::Semantic);
    }

    #[test]
    fn test_default_uses_general_domain() {
        let edge = GraphEdge::default();
        assert_eq!(edge.domain, Domain::General);
    }

    #[test]
    fn test_default_has_valid_nt_weights() {
        let edge = GraphEdge::default();
        assert!(edge.neurotransmitter_weights.validate());
    }

    #[test]
    fn test_default_weight_matches_semantic() {
        let edge = GraphEdge::default();
        assert_eq!(edge.weight, EdgeType::Semantic.default_weight());
    }
}
