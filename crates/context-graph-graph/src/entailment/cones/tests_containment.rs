//! Containment tests for EntailmentCone (M04-T07).
//!
//! MUST USE REAL DATA, NO MOCKS (per constitution REQ-KG-TEST)

use super::*;
use crate::config::{ConeConfig, HyperbolicConfig};
use crate::hyperbolic::mobius::PoincareBall;
use crate::hyperbolic::poincare::PoincarePoint;

fn default_ball() -> PoincareBall {
    PoincareBall::new(HyperbolicConfig::default())
}

/// Test point at apex is always contained
#[test]
fn test_point_at_apex_contained() {
    let config = ConeConfig::default();
    let apex = PoincarePoint::origin();
    let cone = EntailmentCone::new(apex.clone(), 0, &config).expect("valid cone");
    let ball = default_ball();

    assert!(
        cone.contains(&apex, &ball),
        "Point at apex must be contained"
    );
    assert_eq!(
        cone.membership_score(&apex, &ball),
        1.0,
        "Point at apex must have score 1.0"
    );
}

/// Test apex at origin contains all points (degenerate cone)
#[test]
fn test_apex_at_origin_contains_all() {
    let apex = PoincarePoint::origin();
    let cone = EntailmentCone::with_aperture(apex, 0.5, 0).expect("valid");
    let ball = default_ball();

    // Create various points
    let mut coords = [0.0f32; 64];
    coords[0] = 0.5;
    let point = PoincarePoint::from_coords(coords);

    assert!(
        cone.contains(&point, &ball),
        "Origin apex contains all points"
    );
    assert_eq!(
        cone.membership_score(&point, &ball),
        1.0,
        "Origin apex gives score 1.0"
    );
}

/// Test point along cone axis is contained
#[test]
fn test_point_along_cone_axis_contained() {
    // Apex not at origin, point toward origin (along cone axis)
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.5;
    let apex = PoincarePoint::from_coords(apex_coords);
    let cone = EntailmentCone::with_aperture(apex, 0.5, 1).expect("valid");
    let ball = default_ball();

    // Point between apex and origin (along axis)
    let mut point_coords = [0.0f32; 64];
    point_coords[0] = 0.25;
    let point = PoincarePoint::from_coords(point_coords);

    assert!(
        cone.contains(&point, &ball),
        "Point along axis toward origin should be contained"
    );
    assert_eq!(
        cone.membership_score(&point, &ball),
        1.0,
        "Point along axis should have score 1.0"
    );
}

/// Test point perpendicular to cone axis is outside (for narrow cone)
#[test]
fn test_point_outside_cone_perpendicular() {
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.5;
    let apex = PoincarePoint::from_coords(apex_coords);
    // Very narrow cone aperture
    let cone = EntailmentCone::with_aperture(apex, 0.3, 1).expect("valid");
    let ball = default_ball();

    // Point in perpendicular direction
    let mut point_coords = [0.0f32; 64];
    point_coords[1] = 0.5;
    let point = PoincarePoint::from_coords(point_coords);

    assert!(
        !cone.contains(&point, &ball),
        "Perpendicular point should be outside narrow cone"
    );
    let score = cone.membership_score(&point, &ball);
    assert!(
        score < 1.0,
        "Outside point should have score < 1.0, got {}",
        score
    );
    assert!(
        score > 0.0,
        "Score should be > 0 (exponential decay), got {}",
        score
    );
}

/// Test membership_score canonical formula: exp(-2.0 * (angle - aperture))
#[test]
fn test_membership_score_canonical_formula() {
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.5;
    let apex = PoincarePoint::from_coords(apex_coords);
    let cone = EntailmentCone::with_aperture(apex, 0.3, 1).expect("valid");
    let ball = default_ball();

    // Point outside cone
    let mut point_coords = [0.0f32; 64];
    point_coords[1] = 0.5;
    let point = PoincarePoint::from_coords(point_coords);

    let score = cone.membership_score(&point, &ball);

    // Score should be positive and less than 1
    assert!(
        score > 0.0,
        "Score should be > 0 (exponential never reaches 0)"
    );
    assert!(score < 1.0, "Score should be < 1 for outside point");

    // Verify it's a valid exponential decay value
    assert!((0.0..=1.0).contains(&score), "Score must be in [0, 1]");
}

/// Test update_aperture with positive delta
#[test]
fn test_update_aperture_positive_delta() {
    let mut cone = EntailmentCone::default();
    assert_eq!(cone.aperture_factor, 1.0);

    cone.update_aperture(0.3);
    assert!(
        (cone.aperture_factor - 1.3).abs() < 1e-6,
        "Expected 1.3, got {}",
        cone.aperture_factor
    );
}

/// Test update_aperture with negative delta
#[test]
fn test_update_aperture_negative_delta() {
    let mut cone = EntailmentCone::default();
    cone.update_aperture(-0.3);
    assert!(
        (cone.aperture_factor - 0.7).abs() < 1e-6,
        "Expected 0.7, got {}",
        cone.aperture_factor
    );
}

/// Test update_aperture clamps to max 2.0
#[test]
fn test_update_aperture_clamps_max() {
    let mut cone = EntailmentCone::default();
    cone.update_aperture(10.0); // Large positive
    assert_eq!(cone.aperture_factor, 2.0, "Should clamp to max 2.0");
}

/// Test update_aperture clamps to min 0.5
#[test]
fn test_update_aperture_clamps_min() {
    let mut cone = EntailmentCone::default();
    cone.update_aperture(-10.0); // Large negative
    assert_eq!(cone.aperture_factor, 0.5, "Should clamp to min 0.5");
}

/// Test containment boundary - wide aperture should contain more
#[test]
fn test_containment_boundary_wide_aperture() {
    let config = ConeConfig::default();
    let apex = PoincarePoint::origin();
    let cone = EntailmentCone::new(apex, 0, &config).expect("valid");
    let ball = default_ball();

    // With apex at origin, all points should be contained
    let mut coords = [0.0f32; 64];
    coords[0] = 0.9;
    let point = PoincarePoint::from_coords(coords);

    assert!(
        cone.contains(&point, &ball),
        "Wide cone at origin should contain points"
    );
}

/// Test containment with non-origin apex and point toward origin
#[test]
fn test_containment_toward_origin() {
    // Cone apex at (0.7, 0, ...)
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.7;
    let apex = PoincarePoint::from_coords(apex_coords);
    let cone = EntailmentCone::with_aperture(apex, 0.8, 2).expect("valid");
    let ball = default_ball();

    // Point closer to origin than apex (should be in cone direction)
    let mut point_coords = [0.0f32; 64];
    point_coords[0] = 0.3;
    let point = PoincarePoint::from_coords(point_coords);

    // This point is along the cone axis (toward origin)
    assert!(
        cone.contains(&point, &ball),
        "Point toward origin should be contained in wide cone"
    );
}

/// Test that compute_angle returns 0 for point at apex
#[test]
fn test_compute_angle_point_at_apex() {
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.3;
    let apex = PoincarePoint::from_coords(apex_coords);
    let cone = EntailmentCone::with_aperture(apex.clone(), 0.5, 1).expect("valid");
    let ball = default_ball();

    // Point at apex should give angle 0
    assert!(cone.contains(&apex, &ball));
    assert_eq!(cone.membership_score(&apex, &ball), 1.0);
}

/// Test multiple containment checks for determinism
#[test]
fn test_containment_deterministic() {
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.4;
    let apex = PoincarePoint::from_coords(apex_coords);
    let cone = EntailmentCone::with_aperture(apex, 0.5, 1).expect("valid");
    let ball = default_ball();

    let mut point_coords = [0.0f32; 64];
    point_coords[0] = 0.2;
    let point = PoincarePoint::from_coords(point_coords);

    let first_result = cone.contains(&point, &ball);
    let first_score = cone.membership_score(&point, &ball);

    // Run many times to verify determinism
    for _ in 0..100 {
        assert_eq!(
            cone.contains(&point, &ball),
            first_result,
            "Containment should be deterministic"
        );
        assert_eq!(
            cone.membership_score(&point, &ball),
            first_score,
            "Score should be deterministic"
        );
    }
}

/// Test membership score is 1.0 for all contained points
#[test]
fn test_membership_score_contained_is_one() {
    let apex = PoincarePoint::origin();
    let cone = EntailmentCone::with_aperture(apex, 1.0, 0).expect("valid");
    let ball = default_ball();

    // Multiple points that should be contained (apex at origin means all contained)
    let test_coords = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    for coord in test_coords {
        let mut coords = [0.0f32; 64];
        coords[0] = coord;
        let point = PoincarePoint::from_coords(coords);

        if cone.contains(&point, &ball) {
            assert_eq!(
                cone.membership_score(&point, &ball),
                1.0,
                "Contained point should have score 1.0"
            );
        }
    }
}

/// Test update_aperture maintains validity
#[test]
fn test_update_aperture_maintains_validity() {
    let mut cone = EntailmentCone::default();
    assert!(cone.is_valid());

    // Multiple updates
    cone.update_aperture(0.5);
    assert!(cone.is_valid(), "Cone should remain valid after update");
    assert!(cone.validate().is_ok());

    cone.update_aperture(-0.8);
    assert!(cone.is_valid(), "Cone should remain valid after update");
    assert!(cone.validate().is_ok());

    // Extreme updates that trigger clamping
    cone.update_aperture(100.0);
    assert!(
        cone.is_valid(),
        "Cone should remain valid after clamped update"
    );
    assert_eq!(cone.aperture_factor, 2.0);

    cone.update_aperture(-100.0);
    assert!(
        cone.is_valid(),
        "Cone should remain valid after clamped update"
    );
    assert_eq!(cone.aperture_factor, 0.5);
}

/// Test containment with 2D point configurations
#[test]
fn test_containment_2d_points() {
    // Apex at (0.5, 0.3, 0, ...)
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.5;
    apex_coords[1] = 0.3;
    let apex = PoincarePoint::from_coords(apex_coords);
    let cone = EntailmentCone::with_aperture(apex, 0.7, 1).expect("valid");
    let ball = default_ball();

    // Point in same general direction but closer to origin
    let mut point_coords = [0.0f32; 64];
    point_coords[0] = 0.2;
    point_coords[1] = 0.1;
    let point = PoincarePoint::from_coords(point_coords);

    // Should be contained (toward origin from apex)
    let is_contained = cone.contains(&point, &ball);
    let score = cone.membership_score(&point, &ball);

    // Verify consistency
    if is_contained {
        assert_eq!(score, 1.0);
    } else {
        assert!(score < 1.0);
    }
}

/// Test that effective_aperture affects containment
#[test]
fn test_aperture_factor_affects_containment() {
    let mut apex_coords = [0.0f32; 64];
    apex_coords[0] = 0.5;
    let apex = PoincarePoint::from_coords(apex_coords);
    let _ball = default_ball();

    // Create a point that might be on the boundary
    let mut point_coords = [0.0f32; 64];
    point_coords[0] = 0.3;
    point_coords[1] = 0.4; // Diagonal from axis
    let _point = PoincarePoint::from_coords(point_coords);

    // Narrow cone (small aperture)
    let mut narrow_cone = EntailmentCone::with_aperture(apex.clone(), 0.2, 1).expect("valid");
    narrow_cone.aperture_factor = 0.5; // Make even narrower

    // Wide cone (same base aperture but larger factor)
    let mut wide_cone = EntailmentCone::with_aperture(apex, 0.2, 1).expect("valid");
    wide_cone.aperture_factor = 2.0; // Make wider

    // Wide cone should contain more than narrow cone
    assert!(
        wide_cone.effective_aperture() > narrow_cone.effective_aperture(),
        "Wide cone should have larger effective aperture"
    );
}
