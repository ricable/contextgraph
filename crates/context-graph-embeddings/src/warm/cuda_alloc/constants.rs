//! Constants for CUDA allocation requirements.
//!
//! Defines hardware requirements for RTX 5090 (Blackwell) target hardware.

/// Required compute capability major version for RTX 5090 (Blackwell).
///
/// RTX 5090 has compute capability 12.0. We require this as the minimum
/// to ensure Blackwell-specific optimizations are available.
pub const REQUIRED_COMPUTE_MAJOR: u32 = 12;

/// Required compute capability minor version for RTX 5090.
pub const REQUIRED_COMPUTE_MINOR: u32 = 0;

/// Minimum VRAM required in bytes (32GB for RTX 5090).
///
/// This is the total VRAM on an RTX 5090. We require the full amount
/// to ensure all 12 embedding models can be loaded.
pub const MINIMUM_VRAM_BYTES: usize = 32 * 1024 * 1024 * 1024;

/// One gigabyte in bytes.
pub const GB: usize = 1024 * 1024 * 1024;

/// Maximum number of allocation history entries to keep.
pub const MAX_ALLOCATION_HISTORY: usize = 100;
