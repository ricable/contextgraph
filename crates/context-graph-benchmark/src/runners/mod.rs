//! Benchmark runners for executing comparative analysis.
//!
//! This module provides the main harness for running benchmarks and
//! comparing multi-space vs single-embedder approaches.

pub mod comparative;
pub mod retrieval;
pub mod scaling;
pub mod topic;

pub use comparative::{BenchmarkHarness, BenchmarkResults, ComparativeResults};
pub use retrieval::RetrievalRunner;
pub use scaling::ScalingRunner;
pub use topic::TopicRunner;
