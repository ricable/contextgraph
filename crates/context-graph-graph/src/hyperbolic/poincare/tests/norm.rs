//! Norm calculation tests for PoincarePoint.

use crate::hyperbolic::PoincarePoint;

#[test]
fn test_norm_squared_single_nonzero() {
    let mut coords = [0.0f32; 64];
    coords[0] = 0.5;
    let point = PoincarePoint::from_coords(coords);
    assert_eq!(point.norm_squared(), 0.25);
}

#[test]
fn test_norm_squared_multiple_nonzero() {
    let mut coords = [0.0f32; 64];
    coords[0] = 0.3;
    coords[1] = 0.4;
    let point = PoincarePoint::from_coords(coords);
    // 0.09 + 0.16 = 0.25
    assert!((point.norm_squared() - 0.25).abs() < 1e-6);
}

#[test]
fn test_norm_pythagorean() {
    let mut coords = [0.0f32; 64];
    coords[0] = 0.6;
    coords[1] = 0.8;
    let point = PoincarePoint::from_coords(coords);
    // sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0
    assert!((point.norm() - 1.0).abs() < 1e-6);
}

#[test]
fn test_norm_uniform_coords() {
    let coords = [0.1f32; 64];
    let point = PoincarePoint::from_coords(coords);
    // sqrt(64 * 0.01) = sqrt(0.64) = 0.8
    assert!((point.norm() - 0.8).abs() < 1e-6);
}
