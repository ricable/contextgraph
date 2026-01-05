//! Tests for GraphEdge construction and factory methods.

use uuid::Uuid;

use crate::storage::edges::{Domain, EdgeType, GraphEdge, NeurotransmitterWeights};

#[test]
fn test_edge_creation() {
    let source = Uuid::new_v4();
    let target = Uuid::new_v4();
    let edge = GraphEdge::new(1, source, target, EdgeType::Semantic, 0.8, Domain::Code);

    assert_eq!(edge.id, 1);
    assert_eq!(edge.source, source);
    assert_eq!(edge.target, target);
    assert_eq!(edge.edge_type, EdgeType::Semantic);
    assert!((edge.weight - 0.8).abs() < 1e-6);
    assert_eq!(edge.domain, Domain::Code);
    assert!(!edge.is_amortized_shortcut);
    assert!((edge.steering_reward - 0.5).abs() < 1e-6);
    assert_eq!(edge.traversal_count, 0);
    assert!(edge.created_at > 0);
    assert_eq!(edge.last_traversed_at, 0);
}

#[test]
fn test_edge_nt_weights_match_domain() {
    let edge = GraphEdge::new(
        1,
        Uuid::nil(),
        Uuid::nil(),
        EdgeType::Semantic,
        0.5,
        Domain::Code,
    );
    let expected = NeurotransmitterWeights::for_domain(Domain::Code);

    assert_eq!(
        edge.neurotransmitter_weights.excitatory,
        expected.excitatory
    );
    assert_eq!(
        edge.neurotransmitter_weights.inhibitory,
        expected.inhibitory
    );
    assert_eq!(
        edge.neurotransmitter_weights.modulatory,
        expected.modulatory
    );
}

#[test]
fn test_semantic_helper() {
    let edge = GraphEdge::semantic(42, Uuid::nil(), Uuid::nil(), 0.7);
    assert_eq!(edge.id, 42);
    assert_eq!(edge.edge_type, EdgeType::Semantic);
    assert_eq!(edge.domain, Domain::General);
    assert!((edge.weight - 0.7).abs() < 1e-6);
}

#[test]
fn test_hierarchical_helper() {
    let edge = GraphEdge::hierarchical(99, Uuid::nil(), Uuid::nil(), 0.9);
    assert_eq!(edge.id, 99);
    assert_eq!(edge.edge_type, EdgeType::Hierarchical);
}

#[test]
fn test_contradiction_edge_helper() {
    let source = Uuid::new_v4();
    let target = Uuid::new_v4();
    let edge = GraphEdge::contradiction(1, source, target, Domain::General);

    assert_eq!(edge.id, 1);
    assert_eq!(edge.source, source);
    assert_eq!(edge.target, target);
    assert_eq!(edge.edge_type, EdgeType::Contradicts);
    assert!(edge.edge_type.is_contradiction());
    // Contradicts has default weight 0.3
    assert!((edge.weight - 0.3).abs() < 1e-6);
    // Inhibitory-heavy NT profile
    assert!((edge.neurotransmitter_weights.excitatory - 0.2).abs() < 1e-6);
    assert!((edge.neurotransmitter_weights.inhibitory - 0.7).abs() < 1e-6);
    assert!((edge.neurotransmitter_weights.modulatory - 0.1).abs() < 1e-6);
}
