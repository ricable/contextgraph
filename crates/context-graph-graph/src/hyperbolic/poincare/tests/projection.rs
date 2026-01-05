//! Projection tests for PoincarePoint.

use crate::config::HyperbolicConfig;
use crate::hyperbolic::PoincarePoint;

#[test]
fn test_project_inside_ball_unchanged() {
    let config = HyperbolicConfig::default();
    let coords = [0.05f32; 64]; // norm = sqrt(64*0.0025) = 0.4
    let mut point = PoincarePoint::from_coords(coords);
    let original_norm = point.norm();
    point.project(&config);
    assert!((point.norm() - original_norm).abs() < 1e-6);
}

#[test]
fn test_project_outside_ball_rescaled() {
    let config = HyperbolicConfig::default();
    let coords = [0.2f32; 64]; // norm = sqrt(64*0.04) = 1.6
    let mut point = PoincarePoint::from_coords(coords);
    assert!(!point.is_valid()); // outside ball
    point.project(&config);
    assert!(point.is_valid());
    assert!((point.norm() - config.max_norm).abs() < 1e-6);
}

#[test]
fn test_project_at_boundary() {
    let config = HyperbolicConfig::default();
    // Create point exactly at boundary (norm = 1.0)
    let mut coords = [0.0f32; 64];
    coords[0] = 1.0;
    let mut point = PoincarePoint::from_coords(coords);
    assert!(!point.is_valid()); // AT boundary = invalid
    point.project(&config);
    assert!(point.is_valid());
}

#[test]
fn test_projected_returns_new_point() {
    let config = HyperbolicConfig::default();
    let coords = [0.2f32; 64];
    let original = PoincarePoint::from_coords(coords);
    let projected = original.projected(&config);
    // Original unchanged
    assert!((original.norm() - 1.6).abs() < 0.1);
    // Projected is valid
    assert!(projected.is_valid());
}

#[test]
fn test_project_preserves_direction() {
    let config = HyperbolicConfig::default();
    let mut coords = [0.0f32; 64];
    coords[0] = 2.0;
    coords[1] = 1.0;
    let mut point = PoincarePoint::from_coords(coords);
    point.project(&config);
    // Ratio should be preserved
    let ratio = point.coords[0] / point.coords[1];
    assert!((ratio - 2.0).abs() < 1e-5);
}
