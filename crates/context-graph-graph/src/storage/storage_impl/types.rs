//! Core types for graph storage.
//!
//! Defines `PoincarePoint`, `EntailmentCone`, `LegacyGraphEdge`, and `NodeId`.
//!
//! # Binary Formats
//!
//! - PoincarePoint: 256 bytes (64 f32 little-endian)
//! - EntailmentCone: 268 bytes (256 apex + 4 aperture + 4 factor + 4 depth)
//! - NodeId: 8 bytes (i64 little-endian)

use crate::error::{GraphError, GraphResult};

// ========== Type Aliases ==========

/// Node ID type (8 bytes, little-endian)
pub type NodeId = i64;

// ========== Core Types ==========

/// 64D Poincare ball coordinates
///
/// Represents a point in hyperbolic space using the Poincare ball model.
/// All coordinates must satisfy ||x|| < 1 (open ball constraint).
///
/// # Binary Format
///
/// 256 bytes: 64 f32 values in little-endian order.
#[derive(Debug, Clone, PartialEq)]
pub struct PoincarePoint {
    /// 64-dimensional coordinates in Poincare ball.
    pub coords: [f32; 64],
}

impl PoincarePoint {
    /// Create a point at the origin (all zeros).
    #[must_use]
    pub fn origin() -> Self {
        Self { coords: [0.0; 64] }
    }

    /// Create a point from a slice of f32 values.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::DimensionMismatch` if slice length != 64.
    pub fn from_slice(slice: &[f32]) -> GraphResult<Self> {
        if slice.len() != 64 {
            return Err(GraphError::DimensionMismatch {
                expected: 64,
                actual: slice.len(),
            });
        }
        let mut coords = [0.0f32; 64];
        coords.copy_from_slice(slice);
        Ok(Self { coords })
    }

    /// Compute the Euclidean norm of the point.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.coords.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Entailment cone for hierarchical reasoning.
///
/// Represents an entailment cone in hyperbolic space, used for
/// efficient O(1) hierarchy queries via cone containment.
///
/// # Binary Format
///
/// 268 bytes total:
/// - apex: 256 bytes (64 f32 little-endian)
/// - aperture: 4 bytes (f32 little-endian)
/// - aperture_factor: 4 bytes (f32 little-endian)
/// - depth: 4 bytes (u32 little-endian)
#[derive(Debug, Clone)]
pub struct EntailmentCone {
    /// Apex point of the cone in hyperbolic space.
    pub apex: PoincarePoint,
    /// Half-angle aperture in radians (0, pi).
    pub aperture: f32,
    /// Factor for adaptive aperture computation.
    pub aperture_factor: f32,
    /// Hierarchy depth (0 = root).
    pub depth: u32,
}

impl EntailmentCone {
    /// Create a default cone at origin with standard aperture.
    #[must_use]
    pub fn default_at_origin() -> Self {
        Self {
            apex: PoincarePoint::origin(),
            aperture: std::f32::consts::FRAC_PI_4, // 45 degrees
            aperture_factor: 1.0,
            depth: 0,
        }
    }
}

/// Legacy graph edge (placeholder before M04-T15).
///
/// This is the minimal edge representation used in storage_impl.
/// For the full Marblestone-aware GraphEdge with NT weights, use
/// `crate::storage::edges::GraphEdge` instead.
///
/// NOTE: This type is kept for backwards compatibility with existing
/// storage operations until they are migrated to use the full GraphEdge.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LegacyGraphEdge {
    /// Target node ID.
    pub target: NodeId,
    /// Edge type identifier.
    pub edge_type: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_point_origin() {
        let point = PoincarePoint::origin();
        assert_eq!(point.coords, [0.0; 64]);
        assert!((point.norm() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_poincare_point_from_slice() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let point = PoincarePoint::from_slice(&data).expect("valid 64D slice");
        assert!((point.coords[0] - 0.0).abs() < 0.0001);
        assert!((point.coords[63] - 0.63).abs() < 0.0001);
    }

    #[test]
    fn test_poincare_point_from_slice_wrong_dim() {
        let data: Vec<f32> = vec![1.0; 32];
        let result = PoincarePoint::from_slice(&data);
        assert!(result.is_err());
        match result {
            Err(GraphError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 32);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_poincare_point_norm() {
        let mut point = PoincarePoint::origin();
        point.coords[0] = 0.6;
        point.coords[1] = 0.8;
        let norm = point.norm();
        assert!((norm - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_entailment_cone_default() {
        let cone = EntailmentCone::default_at_origin();
        assert_eq!(cone.apex.coords, [0.0; 64]);
        assert!((cone.aperture - std::f32::consts::FRAC_PI_4).abs() < 0.0001);
        assert!((cone.aperture_factor - 1.0).abs() < 0.0001);
        assert_eq!(cone.depth, 0);
    }

    #[test]
    fn test_legacy_graph_edge_serialization() {
        let edge = LegacyGraphEdge {
            target: 42,
            edge_type: 1,
        };
        let bytes = bincode::serialize(&edge).expect("serialize");
        let deserialized: LegacyGraphEdge = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(deserialized.target, 42);
        assert_eq!(deserialized.edge_type, 1);
    }
}
