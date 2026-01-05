//! FFI declarations for CUDA cone kernel functions.
//!
//! # CUDA Required
//!
//! This module requires CUDA support (RTX 5090 / Blackwell).
//! There are NO fallback stubs - linking will fail if CUDA libraries are unavailable.

use std::ffi::c_void;
use std::os::raw::c_int;

#[link(name = "cone_check", kind = "static")]
extern "C" {
    /// Launch batch cone membership computation.
    pub fn launch_cone_check(
        d_cones: *const f32,
        d_points: *const f32,
        d_scores: *mut f32,
        n_cones: c_int,
        n_points: c_int,
        curvature: f32,
        stream: *mut c_void,
    ) -> c_int;

    /// Single cone membership score (delegates to batch with n=1).
    pub fn cone_check_single(
        d_cone: *const f32,
        d_point: *const f32,
        d_score: *mut f32,
        curvature: f32,
        stream: *mut c_void,
    ) -> c_int;

    /// Get kernel configuration info.
    pub fn get_cone_kernel_config(
        block_dim_x: *mut c_int,
        block_dim_y: *mut c_int,
        point_dim: *mut c_int,
        cone_data_dim: *mut c_int,
        shared_mem: *mut c_int,
    );
}
