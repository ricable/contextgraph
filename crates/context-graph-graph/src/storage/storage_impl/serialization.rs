//! Serialization utilities for graph storage types.
//!
//! Provides binary serialization for `PoincarePoint` and `EntailmentCone`.
//!
//! # Binary Formats
//!
//! - PoincarePoint: 256 bytes (64 f32 little-endian)
//! - EntailmentCone: 268 bytes (256 apex + 4 aperture + 4 factor + 4 depth)

use super::types::{EntailmentCone, PoincarePoint};
use crate::error::{GraphError, GraphResult};

/// Serialize PoincarePoint to exactly 256 bytes.
pub fn serialize_point(point: &PoincarePoint) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(256);
    for coord in &point.coords {
        bytes.extend_from_slice(&coord.to_le_bytes());
    }
    debug_assert_eq!(bytes.len(), 256);
    bytes
}

/// Deserialize PoincarePoint from 256 bytes.
pub fn deserialize_point(bytes: &[u8]) -> GraphResult<PoincarePoint> {
    if bytes.len() != 256 {
        return Err(GraphError::CorruptedData {
            location: "PoincarePoint".to_string(),
            details: format!("Expected 256 bytes, got {}", bytes.len()),
        });
    }

    let mut coords = [0.0f32; 64];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        coords[i] = f32::from_le_bytes(
            chunk
                .try_into()
                .expect("chunks_exact(4) guarantees 4 bytes"),
        );
    }

    Ok(PoincarePoint { coords })
}

/// Serialize EntailmentCone to exactly 268 bytes.
pub fn serialize_cone(cone: &EntailmentCone) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(268);

    // Apex coordinates (256 bytes)
    for coord in &cone.apex.coords {
        bytes.extend_from_slice(&coord.to_le_bytes());
    }

    // Aperture (4 bytes)
    bytes.extend_from_slice(&cone.aperture.to_le_bytes());

    // Aperture factor (4 bytes)
    bytes.extend_from_slice(&cone.aperture_factor.to_le_bytes());

    // Depth (4 bytes)
    bytes.extend_from_slice(&cone.depth.to_le_bytes());

    debug_assert_eq!(bytes.len(), 268);
    bytes
}

/// Deserialize EntailmentCone from 268 bytes.
pub fn deserialize_cone(bytes: &[u8]) -> GraphResult<EntailmentCone> {
    if bytes.len() != 268 {
        return Err(GraphError::CorruptedData {
            location: "EntailmentCone".to_string(),
            details: format!("Expected 268 bytes, got {}", bytes.len()),
        });
    }

    let apex = deserialize_point(&bytes[..256])?;
    let aperture = f32::from_le_bytes(
        bytes[256..260]
            .try_into()
            .expect("slice bounds verified above"),
    );
    let aperture_factor = f32::from_le_bytes(
        bytes[260..264]
            .try_into()
            .expect("slice bounds verified above"),
    );
    let depth = u32::from_le_bytes(
        bytes[264..268]
            .try_into()
            .expect("slice bounds verified above"),
    );

    Ok(EntailmentCone {
        apex,
        aperture,
        aperture_factor,
        depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_point_256_bytes() {
        let point = PoincarePoint::origin();
        let bytes = serialize_point(&point);
        assert_eq!(bytes.len(), 256);
    }

    #[test]
    fn test_serialize_cone_268_bytes() {
        let cone = EntailmentCone::default_at_origin();
        let bytes = serialize_cone(&cone);
        assert_eq!(bytes.len(), 268);
    }

    #[test]
    fn test_point_roundtrip() {
        let mut point = PoincarePoint::origin();
        point.coords[0] = 0.5;
        point.coords[63] = -0.3;

        let bytes = serialize_point(&point);
        let restored = deserialize_point(&bytes).expect("deserialize");

        assert!((restored.coords[0] - 0.5).abs() < 0.0001);
        assert!((restored.coords[63] - (-0.3)).abs() < 0.0001);
    }

    #[test]
    fn test_cone_roundtrip() {
        let mut cone = EntailmentCone::default_at_origin();
        cone.apex.coords[0] = 0.1;
        cone.aperture = 0.5;
        cone.aperture_factor = 2.0;
        cone.depth = 5;

        let bytes = serialize_cone(&cone);
        let restored = deserialize_cone(&bytes).expect("deserialize");

        assert!((restored.apex.coords[0] - 0.1).abs() < 0.0001);
        assert!((restored.aperture - 0.5).abs() < 0.0001);
        assert!((restored.aperture_factor - 2.0).abs() < 0.0001);
        assert_eq!(restored.depth, 5);
    }

    #[test]
    fn test_deserialize_point_wrong_size() {
        let bytes = vec![0u8; 100]; // Wrong size
        let result = deserialize_point(&bytes);
        assert!(result.is_err());
        match result {
            Err(GraphError::CorruptedData { details, .. }) => {
                assert!(details.contains("256"));
            }
            _ => panic!("Expected CorruptedData error"),
        }
    }

    #[test]
    fn test_deserialize_cone_wrong_size() {
        let bytes = vec![0u8; 200]; // Wrong size
        let result = deserialize_cone(&bytes);
        assert!(result.is_err());
        match result {
            Err(GraphError::CorruptedData { details, .. }) => {
                assert!(details.contains("268"));
            }
            _ => panic!("Expected CorruptedData error"),
        }
    }
}
