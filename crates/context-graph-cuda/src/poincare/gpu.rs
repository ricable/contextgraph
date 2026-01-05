//! GPU-accelerated Poincare distance computation.
//!
//! Provides batch and single-pair distance computation on GPU.
//!
//! # CUDA Required
//!
//! This module requires CUDA support (RTX 5090 / Blackwell).
//! There are NO fallback stubs - the system will fail fast if CUDA is unavailable.
//!
//! # Performance
//!
//! Target: <1ms for 1K x 1K distance matrix on RTX 5090.

use std::ffi::c_void;

use crate::error::{CudaError, CudaResult};

use super::config::PoincareCudaConfig;
use super::ffi;

/// Compute batch Poincare distances on GPU.
///
/// # Safety
///
/// - `d_queries`, `d_database`, `d_distances` must be valid device pointers
/// - Arrays must be properly sized: queries\[n_queries\]\[64\], database\[n_database\]\[64\]
/// - Output distances\[n_queries\]\[n_database\] must be pre-allocated
/// - Pointers must be aligned for float32 access
///
/// # Arguments
///
/// * `d_queries` - Device pointer to query vectors \[n_queries\]\[64\]
/// * `d_database` - Device pointer to database vectors \[n_database\]\[64\]
/// * `d_distances` - Device pointer to output matrix \[n_queries\]\[n_database\]
/// * `n_queries` - Number of query points
/// * `n_database` - Number of database points
/// * `config` - Poincare configuration
/// * `stream` - CUDA stream (None for default stream)
///
/// # Errors
///
/// Returns `CudaError` if:
/// - Configuration is invalid
/// - Kernel launch fails
/// - CUDA runtime error occurs
///
/// # Performance
///
/// Target: <1ms for 1K x 1K distance matrix on RTX 5090.
///
/// # Example
///
/// ```ignore
/// use context_graph_cuda::poincare::{poincare_distance_batch_gpu, PoincareCudaConfig};
///
/// // Assume d_queries, d_database, d_distances are valid device pointers
/// unsafe {
///     let config = PoincareCudaConfig::default();
///     poincare_distance_batch_gpu(
///         d_queries, d_database, d_distances,
///         1000, 1000, &config, None
///     )?;
/// }
/// ```
pub unsafe fn poincare_distance_batch_gpu(
    d_queries: *const f32,
    d_database: *const f32,
    d_distances: *mut f32,
    n_queries: usize,
    n_database: usize,
    config: &PoincareCudaConfig,
    stream: Option<*mut c_void>,
) -> CudaResult<()> {
    // Validate configuration
    config.validate()?;

    // Validate pointer arguments
    if d_queries.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_queries pointer is null".to_string(),
        ));
    }
    if d_database.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_database pointer is null".to_string(),
        ));
    }
    if d_distances.is_null() {
        return Err(CudaError::InvalidConfig(
            "d_distances pointer is null".to_string(),
        ));
    }

    // Validate sizes
    if n_queries == 0 || n_database == 0 {
        return Err(CudaError::InvalidConfig(
            "n_queries and n_database must be positive".to_string(),
        ));
    }

    let stream_ptr = stream.unwrap_or(std::ptr::null_mut());

    let result = ffi::launch_poincare_distance(
        d_queries,
        d_database,
        d_distances,
        n_queries as i32,
        n_database as i32,
        config.curvature,
        stream_ptr,
    );

    if result != 0 {
        return Err(CudaError::KernelError(format!(
            "Poincare distance kernel failed with CUDA error code {}",
            result
        )));
    }

    Ok(())
}

/// Compute single-pair Poincare distance on GPU.
///
/// # Safety
///
/// Same requirements as `poincare_distance_batch_gpu`.
///
/// # Note
///
/// For efficiency, prefer `poincare_distance_batch_gpu` when computing
/// multiple distances. Single-pair calls have significant kernel launch overhead.
pub unsafe fn poincare_distance_single_gpu(
    d_point_a: *const f32,
    d_point_b: *const f32,
    d_distance: *mut f32,
    config: &PoincareCudaConfig,
    stream: Option<*mut c_void>,
) -> CudaResult<()> {
    config.validate()?;

    if d_point_a.is_null() || d_point_b.is_null() || d_distance.is_null() {
        return Err(CudaError::InvalidConfig(
            "Device pointers cannot be null".to_string(),
        ));
    }

    let stream_ptr = stream.unwrap_or(std::ptr::null_mut());

    let result = ffi::poincare_distance_single(
        d_point_a,
        d_point_b,
        d_distance,
        config.curvature,
        stream_ptr,
    );

    if result != 0 {
        return Err(CudaError::KernelError(format!(
            "Poincare distance single kernel failed with CUDA error code {}",
            result
        )));
    }

    Ok(())
}
