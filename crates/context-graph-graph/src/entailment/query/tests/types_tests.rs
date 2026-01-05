//! Tests for entailment query types.

use crate::config::HyperbolicConfig;
use crate::entailment::query::{
    EntailmentDirection, EntailmentQueryParams, EntailmentResult, LcaResult,
};

use super::helpers::{create_test_cone, create_test_point};

// ========== EntailmentDirection Tests ==========

#[test]
fn test_entailment_direction_equality() {
    assert_eq!(
        EntailmentDirection::Ancestors,
        EntailmentDirection::Ancestors
    );
    assert_eq!(
        EntailmentDirection::Descendants,
        EntailmentDirection::Descendants
    );
    assert_ne!(
        EntailmentDirection::Ancestors,
        EntailmentDirection::Descendants
    );
}

#[test]
fn test_entailment_direction_copy() {
    let dir = EntailmentDirection::Ancestors;
    let copy = dir; // Copy trait
    assert_eq!(dir, copy);
}

// ========== EntailmentQueryParams Tests ==========

#[test]
fn test_params_default() {
    let params = EntailmentQueryParams::default();
    assert_eq!(params.max_depth, 3);
    assert_eq!(params.max_results, 100);
    assert!((params.min_membership_score - 0.7).abs() < 1e-6);
}

#[test]
fn test_params_builder() {
    let params = EntailmentQueryParams::default()
        .with_max_depth(5)
        .with_max_results(50)
        .with_min_score(0.5);

    assert_eq!(params.max_depth, 5);
    assert_eq!(params.max_results, 50);
    assert!((params.min_membership_score - 0.5).abs() < 1e-6);
}

#[test]
fn test_params_with_hyperbolic_config() {
    let config = HyperbolicConfig::with_curvature(-0.5);
    let params = EntailmentQueryParams::default().with_hyperbolic_config(config);

    assert_eq!(params.hyperbolic_config.curvature, -0.5);
}

// ========== LcaResult Tests ==========

#[test]
fn test_lca_result_no_common_ancestor() {
    let result = LcaResult {
        lca_id: None,
        lca_point: None,
        lca_cone: None,
        depth_from_a: 0,
        depth_from_b: 0,
    };
    assert!(result.lca_id.is_none());
}

#[test]
fn test_lca_result_with_ancestor() {
    let point = create_test_point(0.3);
    let cone = create_test_cone(0.3, 1.0, 2);

    let result = LcaResult {
        lca_id: Some(42),
        lca_point: Some(point),
        lca_cone: Some(cone),
        depth_from_a: 2,
        depth_from_b: 1,
    };

    assert_eq!(result.lca_id, Some(42));
    assert_eq!(result.depth_from_a, 2);
    assert_eq!(result.depth_from_b, 1);
}

// ========== EntailmentResult Tests ==========

#[test]
fn test_entailment_result_construction() {
    let point = create_test_point(0.5);
    let cone = create_test_cone(0.5, 0.8, 3);

    let result = EntailmentResult {
        node_id: 42,
        point: point.clone(),
        cone: cone.clone(),
        membership_score: 0.95,
        depth: 3,
        is_direct: true,
    };

    assert_eq!(result.node_id, 42);
    assert!((result.membership_score - 0.95).abs() < 1e-6);
    assert_eq!(result.depth, 3);
    assert!(result.is_direct);
}
