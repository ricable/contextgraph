//! Validity tests for PoincarePoint.

use crate::config::HyperbolicConfig;
use crate::hyperbolic::PoincarePoint;

#[test]
fn test_is_valid_origin() {
    let origin = PoincarePoint::origin();
    assert!(origin.is_valid());
}

#[test]
fn test_is_valid_inside_ball() {
    let coords = [0.05f32; 64]; // norm approx 0.4
    let point = PoincarePoint::from_coords(coords);
    assert!(point.is_valid());
}

#[test]
fn test_is_valid_at_boundary_false() {
    let mut coords = [0.0f32; 64];
    coords[0] = 1.0; // norm = 1.0 exactly
    let point = PoincarePoint::from_coords(coords);
    assert!(!point.is_valid(), "Point AT boundary is invalid");
}

#[test]
fn test_is_valid_outside_ball_false() {
    let coords = [0.2f32; 64]; // norm approx 1.6
    let point = PoincarePoint::from_coords(coords);
    assert!(!point.is_valid());
}

#[test]
fn test_is_valid_for_config() {
    let config = HyperbolicConfig::default(); // max_norm = 0.99999
    let coords = [0.12f32; 64]; // norm approx 0.96
    let point = PoincarePoint::from_coords(coords);
    assert!(point.is_valid_for_config(&config));
}
