//! CUDA-accelerated entailment cone membership checking.
//!
//! Provides GPU-accelerated batch cone membership computation for hyperbolic geometry.
//!
//! # CUDA Required
//!
//! This module requires CUDA support (RTX 5090 / Blackwell).
//! There are NO fallback stubs - the system will fail fast if CUDA is unavailable.
//!
//! # Performance
//!
//! - GPU (RTX 5090): <2ms for 1K x 1K membership matrix
//! - CPU fallback: Uses reference implementation
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004 Section 10.2: Cone CUDA Kernel
//! - perf.latency.entailment_check: <1ms
//! - perf.latency.cone_containment_gpu: <2ms for 1K x 1K batch
//!
//! # CANONICAL Membership Score Formula
//!
//! ```text
//! - If angle <= aperture: score = 1.0
//! - If angle > aperture: score = exp(-2.0 * (angle - aperture))
//! ```
//!
//! # Angle Computation Algorithm
//!
//! ```text
//! 1. tangent = log_map(apex, point) - direction to point in tangent space
//! 2. to_origin = log_map(apex, origin) - cone axis direction (toward origin)
//! 3. cos_angle = dot(tangent, to_origin) / (||tangent|| * ||to_origin||)
//! 4. angle = acos(cos_angle.clamp(-1.0, 1.0))
//!
//! Edge cases that return angle = 0.0 (score = 1.0):
//! - Point at apex (distance < eps)
//! - Apex at origin (norm < eps)
//! - Zero-length tangent or to_origin vectors
//! ```

mod config;
mod constants;
mod cpu;
mod ffi;
mod gpu;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public items - CUDA is ALWAYS required
pub use config::ConeCudaConfig;
pub use constants::{CONE_DATA_DIM, CONE_EPS, DEFAULT_CURVATURE, POINT_DIM};
pub use cpu::{cone_check_batch_cpu, cone_membership_score_cpu};
pub use gpu::{
    cone_check_batch_gpu, cone_check_single_gpu, get_cone_kernel_info, is_cone_gpu_available,
};
pub use types::{ConeData, ConeKernelInfo};
