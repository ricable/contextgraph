//! Unit tests for contradiction detection.

#![allow(clippy::field_reassign_with_default)]

use uuid::Uuid;

use crate::error::GraphError;

use super::helpers::{
    compute_confidence, generate_edge_id, infer_type_from_similarity, uuid_to_i64,
};
use super::types::{CandidateInfo, ContradictionParams, ContradictionResult, ContradictionType};

// ========== ContradictionParams Tests ==========

#[test]
fn test_contradiction_params_default() {
    let params = ContradictionParams::default();

    assert!((params.threshold - 0.5).abs() < 1e-6);
    assert_eq!(params.semantic_k, 50);
    assert!((params.min_similarity - 0.3).abs() < 1e-6);
    assert_eq!(params.graph_depth, 2);
    assert!((params.explicit_edge_weight - 0.6).abs() < 1e-6);
}

#[test]
fn test_contradiction_params_builder() {
    let params = ContradictionParams::default()
        .threshold(0.7)
        .semantic_k(100)
        .min_similarity(0.4)
        .graph_depth(3);

    assert!((params.threshold - 0.7).abs() < 1e-6);
    assert_eq!(params.semantic_k, 100);
    assert!((params.min_similarity - 0.4).abs() < 1e-6);
    assert_eq!(params.graph_depth, 3);
}

#[test]
fn test_high_sensitivity() {
    let params = ContradictionParams::default().high_sensitivity();

    assert!(params.threshold < 0.5);
    assert!(params.semantic_k > 50);
    assert!(params.min_similarity < 0.3);
}

#[test]
fn test_low_sensitivity() {
    let params = ContradictionParams::default().low_sensitivity();

    assert!(params.threshold > 0.5);
    assert!(params.semantic_k < 50);
    assert!(params.min_similarity > 0.3);
}

#[test]
fn test_threshold_clamping() {
    let params = ContradictionParams::default().threshold(1.5);
    assert!((params.threshold - 1.0).abs() < 1e-6);

    let params = ContradictionParams::default().threshold(-0.5);
    assert!((params.threshold - 0.0).abs() < 1e-6);
}

#[test]
fn test_validate_params_valid() {
    let params = ContradictionParams::default();
    assert!(params.validate().is_ok());
}

#[test]
fn test_validate_params_zero_k() {
    let mut params = ContradictionParams::default();
    params.semantic_k = 0;

    let result = params.validate();
    assert!(result.is_err());
    match result {
        Err(GraphError::InvalidInput(msg)) => {
            assert!(msg.contains("semantic_k"));
        }
        _ => panic!("Expected InvalidInput error"),
    }
}

// ========== ContradictionResult Tests ==========

#[test]
fn test_contradiction_result_severity() {
    let result = ContradictionResult {
        contradicting_node_id: Uuid::new_v4(),
        contradiction_type: ContradictionType::DirectOpposition,
        confidence: 0.8,
        semantic_similarity: 0.9,
        edge_weight: Some(0.85),
        has_explicit_edge: true,
        evidence: vec![],
    };

    // DirectOpposition has severity 1.0
    // severity = 0.8 * 1.0 = 0.8
    assert!((result.severity() - 0.8).abs() < 1e-6);
}

#[test]
fn test_is_high_confidence() {
    let result = ContradictionResult {
        contradicting_node_id: Uuid::new_v4(),
        contradiction_type: ContradictionType::DirectOpposition,
        confidence: 0.7,
        semantic_similarity: 0.0,
        edge_weight: None,
        has_explicit_edge: false,
        evidence: vec![],
    };

    assert!(result.is_high_confidence(0.5));
    assert!(result.is_high_confidence(0.7));
    assert!(!result.is_high_confidence(0.8));
}

// ========== Confidence Calculation Tests ==========

#[test]
fn test_compute_confidence_semantic_only() {
    let info = CandidateInfo {
        semantic_similarity: 0.8,
        has_explicit_edge: false,
        edge_weight: None,
        edge_type: None,
    };

    let params = ContradictionParams::default();
    let confidence = compute_confidence(&info, &params);

    // semantic_component = 0.8 * (1 - 0.6) = 0.8 * 0.4 = 0.32
    // No edge component, no boost
    assert!((confidence - 0.32).abs() < 1e-6);
}

#[test]
fn test_compute_confidence_explicit_only() {
    let info = CandidateInfo {
        semantic_similarity: 0.0,
        has_explicit_edge: true,
        edge_weight: Some(0.8),
        edge_type: None,
    };

    let params = ContradictionParams::default();
    let confidence = compute_confidence(&info, &params);

    // edge_component = 0.8 * 0.6 = 0.48
    // No boost (semantic_similarity <= 0.5)
    assert!((confidence - 0.48).abs() < 1e-6);
}

#[test]
fn test_compute_confidence_dual_evidence_boost() {
    let info = CandidateInfo {
        semantic_similarity: 0.8,
        has_explicit_edge: true,
        edge_weight: Some(0.8),
        edge_type: None,
    };

    let params = ContradictionParams::default();
    let confidence = compute_confidence(&info, &params);

    // semantic_component = 0.8 * 0.4 = 0.32
    // edge_component = 0.8 * 0.6 = 0.48
    // combined = 0.32 + 0.48 = 0.80
    // boost = 1.2 (dual evidence)
    // result = 0.80 * 1.2 = 0.96
    assert!((confidence - 0.96).abs() < 1e-6);
}

// ========== Contradiction Type Tests ==========

#[test]
fn test_infer_type_from_high_similarity() {
    let t = infer_type_from_similarity(0.95);
    assert_eq!(t, ContradictionType::DirectOpposition);
}

#[test]
fn test_infer_type_from_medium_similarity() {
    let t = infer_type_from_similarity(0.75);
    assert_eq!(t, ContradictionType::LogicalInconsistency);
}

#[test]
fn test_infer_type_from_low_similarity() {
    let t = infer_type_from_similarity(0.4);
    assert_eq!(t, ContradictionType::CausalConflict);
}

// ========== Helper Function Tests ==========

#[test]
fn test_uuid_to_i64_roundtrip() {
    // Create a UUID from an i64
    let original: i64 = 12345678;
    let uuid = Uuid::from_u64_pair(original as u64, 0);

    // Convert back
    let recovered = uuid_to_i64(&uuid);
    assert_eq!(recovered, original);
}

#[test]
fn test_generate_edge_id_unique() {
    let id1 = generate_edge_id();
    std::thread::sleep(std::time::Duration::from_nanos(1));
    let id2 = generate_edge_id();

    // IDs should be different (time-based)
    assert_ne!(id1, id2);
}

// ========== Contradiction Type Severity Tests ==========

#[test]
fn test_contradiction_type_severity_ordering() {
    assert!(
        ContradictionType::DirectOpposition.severity()
            > ContradictionType::LogicalInconsistency.severity()
    );
    assert!(
        ContradictionType::LogicalInconsistency.severity()
            > ContradictionType::TemporalConflict.severity()
    );
    assert!(
        ContradictionType::TemporalConflict.severity()
            > ContradictionType::CausalConflict.severity()
    );
}

#[test]
fn test_contradiction_type_severity_values() {
    assert!((ContradictionType::DirectOpposition.severity() - 1.0).abs() < 1e-6);
    assert!((ContradictionType::LogicalInconsistency.severity() - 0.8).abs() < 1e-6);
    assert!((ContradictionType::TemporalConflict.severity() - 0.7).abs() < 1e-6);
    assert!((ContradictionType::CausalConflict.severity() - 0.6).abs() < 1e-6);
}
