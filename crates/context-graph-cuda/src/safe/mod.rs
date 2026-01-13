//! Safe RAII wrappers for CUDA resources.
//!
//! This module provides memory-safe wrappers with automatic cleanup via Drop.
//!
//! # Constitution Compliance
//!
//! - ARCH-06: CUDA FFI only in context-graph-cuda
//! - AP-14: No .unwrap() - all errors propagated via Result

pub mod device;

pub use device::GpuDevice;
