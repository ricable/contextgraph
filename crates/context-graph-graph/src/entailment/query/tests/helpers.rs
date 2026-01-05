//! Test helpers and utilities for entailment query tests.

use crate::entailment::cones::EntailmentCone;
use crate::entailment::query::conversion::test_helpers::{
    entailment_to_storage_cone, hyperbolic_to_storage_point,
};
use crate::hyperbolic::poincare::PoincarePoint;
use crate::storage::{GraphStorage, StorageConfig};
use tempfile::TempDir;

/// Create a test storage instance with a temporary directory.
pub fn create_test_storage() -> (GraphStorage, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = StorageConfig::default();
    let storage = GraphStorage::open(temp_dir.path(), config).expect("Failed to open storage");
    (storage, temp_dir)
}

/// Create a test point with the given x coordinate.
pub fn create_test_point(x: f32) -> PoincarePoint {
    let mut coords = [0.0f32; 64];
    coords[0] = x;
    PoincarePoint::from_coords(coords)
}

/// Create a test cone with the given apex x coordinate, aperture, and depth.
pub fn create_test_cone(apex_x: f32, aperture: f32, depth: u32) -> EntailmentCone {
    let apex = create_test_point(apex_x);
    EntailmentCone::with_aperture(apex, aperture, depth).expect("valid cone")
}

/// Store entailment cone (converts to storage type).
pub fn store_cone(storage: &GraphStorage, node_id: i64, cone: &EntailmentCone) {
    let storage_cone = entailment_to_storage_cone(cone);
    storage.put_cone(node_id, &storage_cone).expect("put cone");
}

/// Store hyperbolic point (converts to storage type).
pub fn store_point(storage: &GraphStorage, node_id: i64, point: &PoincarePoint) {
    let storage_point = hyperbolic_to_storage_point(point);
    storage
        .put_hyperbolic(node_id, &storage_point)
        .expect("put hyperbolic");
}
