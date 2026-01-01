//! Unit tests for GraphEdge.

use super::*;
use chrono::{DateTime, Utc};
use crate::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
use crate::types::NodeId;
use uuid::Uuid;

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
