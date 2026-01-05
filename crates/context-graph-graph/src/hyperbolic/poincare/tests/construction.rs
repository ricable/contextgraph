//! Construction tests for PoincarePoint.

use crate::config::HyperbolicConfig;
use crate::hyperbolic::PoincarePoint;

#[test]
fn test_origin_is_zero_vector() {
    let origin = PoincarePoint::origin();
    for &c in &origin.coords {
        assert_eq!(c, 0.0, "Origin must have all zero coordinates");
    }
    assert_eq!(origin.coords.len(), 64, "Must have exactly 64 dimensions");
}

#[test]
fn test_origin_has_zero_norm() {
    let origin = PoincarePoint::origin();
    assert_eq!(origin.norm(), 0.0);
    assert_eq!(origin.norm_squared(), 0.0);
}

#[test]
fn test_default_is_origin() {
    let default = PoincarePoint::default();
    let origin = PoincarePoint::origin();
    assert_eq!(default, origin);
}

#[test]
fn test_from_coords_preserves_values() {
    let mut coords = [0.0f32; 64];
    coords[0] = 0.1;
    coords[63] = 0.2;
    let point = PoincarePoint::from_coords(coords);
    assert_eq!(point.coords[0], 0.1);
    assert_eq!(point.coords[63], 0.2);
    assert_eq!(point.coords[32], 0.0);
}

#[test]
fn test_from_coords_projected_ensures_validity() {
    let config = HyperbolicConfig::default();
    let coords = [1.0f32; 64]; // norm = 8.0
    let point = PoincarePoint::from_coords_projected(coords, &config);
    assert!(point.is_valid(), "Projected point must be valid");
    // After projection, norm is at most max_norm (may be exactly equal due to floating point)
    assert!(point.norm() <= config.max_norm + 1e-6);
}
