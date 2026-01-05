//! CUDA-accelerated Poincare ball distance computation.
//!
//! Provides GPU-accelerated batch distance computation for hyperbolic geometry.
//!
//! # CUDA Required
//!
//! This module requires CUDA support (RTX 5090 / Blackwell).
//! There are NO fallback stubs - the system will fail fast if CUDA is unavailable.
//!
//! # Performance
//!
//! - GPU (RTX 5090): <1ms for 1K x 1K distance matrix
//! - CPU fallback: Uses context-graph-graph PoincareBall
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004 Section 10.1: Poincare CUDA Kernel
//! - perf.latency.poincare_distance_gpu: <1ms for 1K x 1K
//! - stack.gpu: RTX 5090, 32GB GDDR7, compute: 12.0
//!
//! # Mathematics
//!
//! Poincare ball distance formula (direct, GPU-friendly):
//! ```text
//! d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c * ||x-y||^2 / ((1-c*||x||^2)(1-c*||y||^2))))
//! ```
//!
//! where c = |curvature| (always positive for hyperbolic space).
//!
//! # Module Structure
//!
//! - [`constants`]: Dimension, curvature, and epsilon constants
//! - [`config`]: Configuration types and validation
//! - [`kernel`]: Kernel info and GPU availability
//! - [`gpu`]: GPU-accelerated distance functions
//! - [`cpu`]: CPU reference implementations

// Submodules
mod config;
mod constants;
mod cpu;
mod ffi;
mod gpu;
mod kernel;

#[cfg(test)]
mod tests;

// Re-export all public items for backwards compatibility
// Constants
pub use constants::{DEFAULT_CURVATURE, POINCARE_DIM, POINCARE_EPS};

// Configuration
pub use config::PoincareCudaConfig;

// Kernel info and GPU availability
pub use kernel::{get_kernel_info, is_poincare_gpu_available, PoincareKernelInfo};

// GPU functions
pub use gpu::{poincare_distance_batch_gpu, poincare_distance_single_gpu};

// CPU reference implementations
pub use cpu::{poincare_distance_batch_cpu, poincare_distance_cpu};
