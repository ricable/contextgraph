//! FFI declarations for Poincare CUDA kernel functions.
//!
//! # CUDA Required
//!
//! This module requires CUDA support (RTX 5090 / Blackwell).
//! There are NO fallback stubs - linking will fail if CUDA libraries are unavailable.

use std::ffi::c_void;
use std::os::raw::c_int;

#[link(name = "poincare_distance", kind = "static")]
extern "C" {
    /// Launch batch Poincare distance computation.
    pub fn launch_poincare_distance(
        d_queries: *const f32,
        d_database: *const f32,
        d_distances: *mut f32,
        n_queries: c_int,
        n_database: c_int,
        curvature: f32,
        stream: *mut c_void,
    ) -> c_int;

    /// Single-pair distance (delegates to batch with n=1).
    pub fn poincare_distance_single(
        d_point_a: *const f32,
        d_point_b: *const f32,
        d_distance: *mut f32,
        curvature: f32,
        stream: *mut c_void,
    ) -> c_int;

    /// Get kernel configuration info.
    pub fn get_poincare_kernel_config(
        block_dim_x: *mut c_int,
        block_dim_y: *mut c_int,
        point_dim: *mut c_int,
        shared_mem: *mut c_int,
    );
}
