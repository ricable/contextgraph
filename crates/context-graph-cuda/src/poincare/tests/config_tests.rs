//! Configuration and constant tests for Poincare operations.

#![allow(clippy::assertions_on_constants)]

use crate::error::CudaError;
use crate::poincare::{
    get_kernel_info, is_poincare_gpu_available, PoincareCudaConfig, DEFAULT_CURVATURE,
    POINCARE_DIM, POINCARE_EPS,
};

// ========== Configuration Tests ==========

#[test]
fn test_config_default() {
    let config = PoincareCudaConfig::default();
    assert_eq!(config.dim, 64);
    assert!((config.curvature - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_config_with_curvature_valid() {
    let config = PoincareCudaConfig::with_curvature(-0.5).unwrap();
    assert!((config.curvature - (-0.5)).abs() < 1e-6);
    assert_eq!(config.dim, 64);
}

#[test]
fn test_config_with_curvature_invalid_positive() {
    let result = PoincareCudaConfig::with_curvature(0.5);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, CudaError::InvalidConfig(_)));
}

#[test]
fn test_config_with_curvature_invalid_zero() {
    let result = PoincareCudaConfig::with_curvature(0.0);
    assert!(result.is_err());
}

#[test]
fn test_config_with_curvature_invalid_nan() {
    let result = PoincareCudaConfig::with_curvature(f32::NAN);
    assert!(result.is_err());
}

#[test]
fn test_config_validate_default() {
    let config = PoincareCudaConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_wrong_dimension() {
    let bad_config = PoincareCudaConfig {
        dim: 128,
        curvature: -1.0,
    };
    let result = bad_config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("64"));
}

#[test]
fn test_config_abs_curvature() {
    let config = PoincareCudaConfig::with_curvature(-2.5).unwrap();
    assert!((config.abs_curvature() - 2.5).abs() < 1e-6);
}

#[test]
fn test_constants() {
    assert_eq!(POINCARE_DIM, 64);
    assert!((DEFAULT_CURVATURE - (-1.0)).abs() < 1e-6);
    assert!(POINCARE_EPS > 0.0);
    assert!(POINCARE_EPS < 1e-5);
}

// ========== GPU Availability Test ==========

#[test]
fn test_is_gpu_available_returns_bool() {
    // This test just ensures the function doesn't crash
    let _available = is_poincare_gpu_available();
    // Result depends on hardware - we just check it returns a bool
}

#[test]
fn test_kernel_info_format() {
    // Check kernel info returns expected format
    if let Some(info) = get_kernel_info() {
        assert!(info.block_dim_x > 0);
        assert!(info.block_dim_y > 0);
        assert_eq!(info.point_dim, 64);
        assert!(info.shared_mem_bytes > 0);
    }
    // If None, CUDA feature is disabled - that's okay
}
