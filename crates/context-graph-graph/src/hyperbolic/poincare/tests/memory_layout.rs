//! Memory layout tests for PoincarePoint.

use crate::hyperbolic::PoincarePoint;

#[test]
fn test_size_is_256_bytes() {
    assert_eq!(
        std::mem::size_of::<PoincarePoint>(),
        256,
        "PoincarePoint must be 256 bytes (64 * f32)"
    );
}

#[test]
fn test_alignment_is_64_bytes() {
    assert_eq!(
        std::mem::align_of::<PoincarePoint>(),
        64,
        "PoincarePoint must be 64-byte aligned for SIMD"
    );
}
