//! Tests for CUDA allocation module.

use super::*;

// ============================================================================
// GpuInfo Tests
// ============================================================================

#[test]
fn test_gpu_info_construction() {
    let info = GpuInfo::new(
        0,
        "NVIDIA GeForce RTX 5090".to_string(),
        (12, 0),
        32 * GB,
        "13.1.0".to_string(),
    );

    assert_eq!(info.device_id, 0);
    assert_eq!(info.name, "NVIDIA GeForce RTX 5090");
    assert_eq!(info.compute_capability, (12, 0));
    assert_eq!(info.total_memory_bytes, 32 * GB);
    assert_eq!(info.driver_version, "13.1.0");
}

#[test]
fn test_gpu_info_default() {
    let info = GpuInfo::default();

    assert_eq!(info.device_id, 0);
    assert_eq!(info.name, "No GPU");
    assert_eq!(info.compute_capability, (0, 0));
    assert_eq!(info.total_memory_bytes, 0);
    assert_eq!(info.driver_version, "N/A");
}

#[test]
fn test_gpu_info_compute_capability_string() {
    let info = GpuInfo::new(
        0,
        "RTX 5090".to_string(),
        (12, 0),
        32 * GB,
        "13.1.0".to_string(),
    );

    assert_eq!(info.compute_capability_string(), "12.0");

    let info_89 = GpuInfo::new(
        0,
        "RTX 4090".to_string(),
        (8, 9),
        24 * GB,
        "12.0.0".to_string(),
    );

    assert_eq!(info_89.compute_capability_string(), "8.9");
}

#[test]
fn test_gpu_info_total_memory_gb() {
    let info = GpuInfo::new(
        0,
        "RTX 5090".to_string(),
        (12, 0),
        32 * GB,
        "13.1.0".to_string(),
    );

    assert!((info.total_memory_gb() - 32.0).abs() < 0.01);
}

#[test]
fn test_gpu_info_meets_compute_requirement() {
    let rtx_5090 = GpuInfo::new(
        0,
        "RTX 5090".to_string(),
        (12, 0),
        32 * GB,
        "13.1.0".to_string(),
    );

    // Exact match
    assert!(rtx_5090.meets_compute_requirement(12, 0));

    // Higher major version meets requirement
    assert!(rtx_5090.meets_compute_requirement(11, 0));
    assert!(rtx_5090.meets_compute_requirement(8, 9));

    // Same major, lower minor meets requirement
    // (12.0 >= 12.0, so this is true)

    // Higher requirement not met
    assert!(!rtx_5090.meets_compute_requirement(13, 0));
    assert!(!rtx_5090.meets_compute_requirement(12, 1));

    // RTX 4090 case
    let rtx_4090 = GpuInfo::new(
        0,
        "RTX 4090".to_string(),
        (8, 9),
        24 * GB,
        "12.0.0".to_string(),
    );

    assert!(rtx_4090.meets_compute_requirement(8, 9));
    assert!(rtx_4090.meets_compute_requirement(8, 0));
    assert!(rtx_4090.meets_compute_requirement(7, 5));
    assert!(!rtx_4090.meets_compute_requirement(12, 0));
}

#[test]
fn test_gpu_info_meets_rtx_5090_requirements() {
    let rtx_5090 = GpuInfo::new(
        0,
        "RTX 5090".to_string(),
        (12, 0),
        32 * GB,
        "13.1.0".to_string(),
    );

    assert!(rtx_5090.meets_rtx_5090_requirements());

    // Insufficient VRAM
    let low_vram = GpuInfo::new(
        0,
        "RTX 5090".to_string(),
        (12, 0),
        24 * GB, // Only 24GB
        "13.1.0".to_string(),
    );

    assert!(!low_vram.meets_rtx_5090_requirements());

    // Insufficient compute capability
    let old_gpu = GpuInfo::new(
        0,
        "RTX 4090".to_string(),
        (8, 9),
        32 * GB,
        "12.0.0".to_string(),
    );

    assert!(!old_gpu.meets_rtx_5090_requirements());
}

// ============================================================================
// VramAllocation Tests
// ============================================================================

#[test]
fn test_vram_allocation_protected() {
    let alloc = VramAllocation::new_protected(0x1000_0000, 800_000_000, 0);

    assert_eq!(alloc.ptr, 0x1000_0000);
    assert_eq!(alloc.size_bytes, 800_000_000);
    assert_eq!(alloc.device_id, 0);
    assert!(alloc.is_protected);
    assert!(alloc.is_valid());
}

#[test]
fn test_vram_allocation_evictable() {
    let alloc = VramAllocation::new_evictable(0x2000_0000, 1_000_000, 1);

    assert_eq!(alloc.ptr, 0x2000_0000);
    assert_eq!(alloc.size_bytes, 1_000_000);
    assert_eq!(alloc.device_id, 1);
    assert!(!alloc.is_protected);
    assert!(alloc.is_valid());
}

#[test]
fn test_vram_allocation_default() {
    let alloc = VramAllocation::default();

    assert_eq!(alloc.ptr, 0);
    assert_eq!(alloc.size_bytes, 0);
    assert_eq!(alloc.device_id, 0);
    assert!(!alloc.is_protected);
    assert!(!alloc.is_valid()); // Null pointer is invalid
}

#[test]
fn test_vram_allocation_size_conversions() {
    let alloc = VramAllocation::new_protected(0x1000, 1_073_741_824, 0); // 1GB

    assert!((alloc.size_mb() - 1024.0).abs() < 0.01);
    assert!((alloc.size_gb() - 1.0).abs() < 0.01);
}

// ============================================================================
// NOTE: Stub tests REMOVED - CUDA is ALWAYS required (RTX 5090)
// Per constitution: No fallback stubs, fail-fast architecture
// ============================================================================

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_format_bytes() {
    assert_eq!(format_bytes(0), "0B");
    assert_eq!(format_bytes(512), "512B");
    assert_eq!(format_bytes(1024), "1.00KB");
    assert_eq!(format_bytes(1536), "1.50KB");
    assert_eq!(format_bytes(1024 * 1024), "1.00MB");
    assert_eq!(format_bytes(1500 * 1024 * 1024), "1.46GB");
    assert_eq!(format_bytes(32 * GB), "32.00GB");
}

// ============================================================================
// Constant Tests
// ============================================================================

#[test]
fn test_constants() {
    assert_eq!(REQUIRED_COMPUTE_MAJOR, 12);
    assert_eq!(REQUIRED_COMPUTE_MINOR, 0);
    assert_eq!(MINIMUM_VRAM_BYTES, 32 * 1024 * 1024 * 1024);
}
