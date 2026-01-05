//! Constants for Poincare ball CUDA operations.
//!
//! These constants define the fixed parameters for CUDA kernel compatibility.

/// Default Poincare ball dimension (fixed for SIMD alignment).
/// Must match the kernel's POINT_DIM constant.
pub const POINCARE_DIM: usize = 64;

/// Default curvature (negative, per hyperbolic geometry).
/// Standard unit hyperbolic space has curvature -1.0.
pub const DEFAULT_CURVATURE: f32 = -1.0;

/// Numerical epsilon for stability (matches kernel).
pub const POINCARE_EPS: f32 = 1e-7;
