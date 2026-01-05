//! Edge case tests for PoincarePoint.

use crate::config::HyperbolicConfig;
use crate::hyperbolic::PoincarePoint;

#[test]
fn test_edge_case_very_small_norm() {
    let coords = [1e-10f32; 64];
    let point = PoincarePoint::from_coords(coords);
    assert!(point.is_valid());
    assert!(point.norm() > 0.0);
}

#[test]
fn test_edge_case_near_max_norm() {
    let config = HyperbolicConfig::default();
    // norm approx 0.999 (just under max_norm)
    let scale = 0.999 / (64.0_f32).sqrt();
    let coords = [scale; 64];
    let point = PoincarePoint::from_coords(coords);
    assert!(point.is_valid_for_config(&config));
}

#[test]
fn test_edge_case_negative_coords() {
    let mut coords = [0.0f32; 64];
    coords[0] = -0.5;
    coords[1] = 0.5;
    let point = PoincarePoint::from_coords(coords);
    // norm = sqrt(0.25 + 0.25) = sqrt(0.5) approx 0.707
    assert!(point.is_valid());
    assert!((point.norm() - (0.5_f32).sqrt()).abs() < 1e-6);
}

#[test]
fn test_edge_case_project_zero_vector() {
    let config = HyperbolicConfig::default();
    let mut origin = PoincarePoint::origin();
    origin.project(&config); // Should not panic
    assert_eq!(origin.norm(), 0.0);
}

#[test]
fn test_edge_case_nan_detection() {
    let mut coords = [0.0f32; 64];
    coords[0] = f32::NAN;
    let point = PoincarePoint::from_coords(coords);
    // norm_squared and norm will be NaN
    assert!(point.norm().is_nan());
    // is_valid should return false for NaN
    assert!(!point.is_valid()); // NaN < 1.0 is false
}

#[test]
fn test_edge_case_infinity() {
    let mut coords = [0.0f32; 64];
    coords[0] = f32::INFINITY;
    let point = PoincarePoint::from_coords(coords);
    assert!(!point.is_valid());
}
