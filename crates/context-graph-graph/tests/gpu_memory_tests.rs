//! GPU Memory Manager tests - M04-T28
//!
//! NO MOCKS - tests actual allocation tracking behavior.
//!
//! # Full State Verification Requirements
//!
//! After each operation, verify internal state matches external API:
//! - Source of truth: `ManagerInner.allocations` HashMap
//! - Each test prints state BEFORE and AFTER operations
//! - Edge cases: zero-size, max allocation, invalid config
//!
//! # Constitution Reference
//!
//! - AP-001: Fail fast, never unwrap() in prod
//! - AP-015: GPU alloc without pool -> use CUDA memory pool
//! - perf.memory.gpu: <24GB (8GB headroom)
//!
//! # Module Structure
//!
//! - `basic_tests`: Core allocation/deallocation, budget enforcement, thread safety
//! - `edge_case_tests`: Zero-size, max allocation, boundary conditions, invalid config
//! - `category_tests`: Category budgets, low memory detection, multi-category allocations
//! - `stats_tests`: Memory statistics, usage tracking, usage percentage
//! - `serialization_tests`: Config, stats, and enum serialization

#[path = "gpu_memory_tests/basic_tests.rs"]
mod basic_tests;

#[path = "gpu_memory_tests/category_tests.rs"]
mod category_tests;

#[path = "gpu_memory_tests/edge_case_tests.rs"]
mod edge_case_tests;

#[path = "gpu_memory_tests/serialization_tests.rs"]
mod serialization_tests;

#[path = "gpu_memory_tests/stats_tests.rs"]
mod stats_tests;
