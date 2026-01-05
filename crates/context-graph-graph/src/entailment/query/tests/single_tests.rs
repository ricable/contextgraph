//! Tests for single-pair entailment operations.

use crate::config::HyperbolicConfig;
use crate::entailment::cones::EntailmentCone;
use crate::entailment::query::{entailment_score, is_entailed_by};
use crate::error::GraphError;
use crate::hyperbolic::poincare::PoincarePoint;

use super::helpers::{
    create_test_cone, create_test_point, create_test_storage, store_cone, store_point,
};

// ========== is_entailed_by Tests ==========

#[test]
fn test_is_entailed_by_missing_ancestor_cone() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // No data stored - should fail fast
    let result = is_entailed_by(&storage, 1, 2, &config);
    assert!(matches!(result, Err(GraphError::MissingHyperbolicData(1))));
}

#[test]
fn test_is_entailed_by_missing_descendant_point() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Store ancestor cone but no descendant point
    let ancestor_cone = create_test_cone(0.0, 1.0, 0);
    store_cone(&storage, 1, &ancestor_cone);

    let result = is_entailed_by(&storage, 1, 2, &config);
    assert!(matches!(result, Err(GraphError::MissingHyperbolicData(2))));
}

#[test]
fn test_is_entailed_by_contained() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Ancestor at origin with wide cone
    let ancestor_cone = create_test_cone(0.0, 1.5, 0);
    store_cone(&storage, 1, &ancestor_cone);

    // Descendant point inside cone
    let descendant_point = create_test_point(0.3);
    store_point(&storage, 2, &descendant_point);

    let result = is_entailed_by(&storage, 1, 2, &config).expect("check should succeed");
    assert!(result, "Descendant should be contained in ancestor's cone");
}

#[test]
fn test_is_entailed_by_not_contained() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Ancestor with narrow cone away from origin
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.5;
    let apex = PoincarePoint::from_coords(apex_coords);
    let ancestor_cone = EntailmentCone::with_aperture(apex, 0.2, 1).expect("valid cone");
    store_cone(&storage, 1, &ancestor_cone);

    // Descendant point perpendicular to cone axis
    let mut desc_coords = [0.0f32; 64];
    desc_coords[1] = 0.5; // Perpendicular direction
    let descendant_point = PoincarePoint::from_coords(desc_coords);
    store_point(&storage, 2, &descendant_point);

    let result = is_entailed_by(&storage, 1, 2, &config).expect("check should succeed");
    assert!(!result, "Perpendicular point should not be contained");
}

// ========== entailment_score Tests ==========

#[test]
fn test_entailment_score_fully_contained() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Ancestor at origin with wide cone
    let ancestor_cone = create_test_cone(0.0, 1.5, 0);
    store_cone(&storage, 1, &ancestor_cone);

    // Descendant point inside cone (score should be 1.0)
    let descendant_point = create_test_point(0.3);
    store_point(&storage, 2, &descendant_point);

    let score = entailment_score(&storage, 1, 2, &config).expect("score should succeed");
    assert_eq!(score, 1.0, "Fully contained point should have score 1.0");
}

#[test]
fn test_entailment_score_partial() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Ancestor with narrow cone
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.5;
    let apex = PoincarePoint::from_coords(apex_coords);
    let ancestor_cone = EntailmentCone::with_aperture(apex, 0.3, 1).expect("valid cone");
    store_cone(&storage, 1, &ancestor_cone);

    // Descendant point outside but near cone
    let mut desc_coords = [0.0f32; 64];
    desc_coords[1] = 0.4;
    let descendant_point = PoincarePoint::from_coords(desc_coords);
    store_point(&storage, 2, &descendant_point);

    let score = entailment_score(&storage, 1, 2, &config).expect("score should succeed");
    assert!(score > 0.0, "Nearby point should have positive score");
    assert!(score < 1.0, "Outside point should have score < 1.0");
}

// ========== Edge Cases ==========

#[test]
fn test_self_entailment() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Node with both cone and point
    let point = create_test_point(0.3);
    let apex = point.clone();
    let cone = EntailmentCone::with_aperture(apex, 1.0, 1).expect("valid cone");

    store_cone(&storage, 1, &cone);
    store_point(&storage, 1, &point);

    // Node should entail itself (point at apex is always contained)
    let result = is_entailed_by(&storage, 1, 1, &config).expect("check should succeed");
    assert!(result, "Node should entail itself");

    let score = entailment_score(&storage, 1, 1, &config).expect("score should succeed");
    assert_eq!(score, 1.0, "Self-entailment score should be 1.0");
}
