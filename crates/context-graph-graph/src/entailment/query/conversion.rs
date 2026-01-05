//! Type conversion helpers between storage and domain types.
//!
//! This module provides conversions between the storage layer types
//! and the hyperbolic/entailment domain types.

use crate::entailment::cones::EntailmentCone;
use crate::error::GraphResult;
use crate::hyperbolic::poincare::PoincarePoint;
use crate::storage::storage_impl::{
    EntailmentCone as StorageEntailmentCone, PoincarePoint as StoragePoincarePoint,
};

/// Convert storage PoincarePoint to hyperbolic PoincarePoint.
pub fn storage_to_hyperbolic_point(storage_point: &StoragePoincarePoint) -> PoincarePoint {
    PoincarePoint::from_coords(storage_point.coords)
}

/// Convert storage EntailmentCone to entailment EntailmentCone.
pub fn storage_to_entailment_cone(
    storage_cone: &StorageEntailmentCone,
) -> GraphResult<EntailmentCone> {
    let apex = storage_to_hyperbolic_point(&storage_cone.apex);
    let mut cone = EntailmentCone::with_aperture(apex, storage_cone.aperture, storage_cone.depth)?;
    cone.aperture_factor = storage_cone.aperture_factor;
    Ok(cone)
}

#[cfg(test)]
pub mod test_helpers {
    //! Test-only type conversion helpers (reverse direction).

    use super::*;

    /// Convert hyperbolic PoincarePoint to storage PoincarePoint (test-only).
    pub fn hyperbolic_to_storage_point(hyp_point: &PoincarePoint) -> StoragePoincarePoint {
        StoragePoincarePoint {
            coords: hyp_point.coords,
        }
    }

    /// Convert entailment EntailmentCone to storage EntailmentCone (test-only).
    pub fn entailment_to_storage_cone(ent_cone: &EntailmentCone) -> StorageEntailmentCone {
        StorageEntailmentCone {
            apex: hyperbolic_to_storage_point(&ent_cone.apex),
            aperture: ent_cone.aperture,
            aperture_factor: ent_cone.aperture_factor,
            depth: ent_cone.depth,
        }
    }
}
