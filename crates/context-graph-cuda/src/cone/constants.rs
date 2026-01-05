//! Constants for CUDA cone operations.
//!
//! These constants define dimensions and parameters that must match
//! the CUDA kernel definitions.

/// Cone data dimension (64 apex coords + 1 aperture).
/// Must match kernel's CONE_DATA_DIM constant.
pub const CONE_DATA_DIM: usize = 65;

/// Point dimension (must match POINCARE_DIM and kernel's POINT_DIM).
pub const POINT_DIM: usize = 64;

/// Default curvature (negative, per hyperbolic geometry).
/// Standard unit hyperbolic space has curvature -1.0.
pub const DEFAULT_CURVATURE: f32 = -1.0;

/// Numerical epsilon for stability (matches kernel).
pub const CONE_EPS: f32 = 1e-7;
