//! GPU Inference Engine for Warm Model Validation.
//!
//! This module provides real GPU inference capabilities for validating
//! warm-loaded embedding models. Per Constitution AP-007, NO FAKE DATA.
//!
//! # Error Code
//!
//! EMB-E011: Inference failures (both init and execution)
//!
//! # Design Philosophy
//!
//! FAIL FAST. NO FALLBACKS. REAL INFERENCE ONLY.

mod engine;

pub use engine::InferenceEngine;
