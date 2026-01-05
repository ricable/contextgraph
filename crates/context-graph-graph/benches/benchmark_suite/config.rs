//! Benchmark configuration constants.
//!
//! All values derived from constitution and task spec.

#![allow(dead_code)]

/// Poincare ball dimension (from constitution)
pub const POINCARE_DIM: usize = 64;

/// Poincare ball curvature (from constitution)
pub const POINCARE_CURVATURE: f32 = -1.0;

/// Batch sizes for throughput benchmarks
pub const BATCH_SIZES: &[usize] = &[1, 10, 100, 1000];

/// Graph sizes for traversal benchmarks
pub const GRAPH_SIZES: &[usize] = &[100, 1000, 10000];

/// BFS max depth for benchmarks
pub const BFS_MAX_DEPTH: u32 = 6;

/// GPU memory budget (from constitution: 24GB safe limit)
pub const GPU_MEMORY_BUDGET_GB: usize = 24;

/// GPU memory budget in bytes
pub const GPU_MEMORY_BUDGET_BYTES: usize = GPU_MEMORY_BUDGET_GB * 1024 * 1024 * 1024;

/// Performance targets (in microseconds)
#[allow(dead_code)]
pub mod targets {
    pub const POINCARE_SINGLE_US: u64 = 10;
    pub const POINCARE_1K_BATCH_CPU_US: u64 = 100_000;
    pub const POINCARE_1K_BATCH_GPU_US: u64 = 1_000;
    pub const CONE_SINGLE_US: u64 = 15;
    pub const CONE_1K_BATCH_CPU_US: u64 = 200_000;
    pub const CONE_1K_BATCH_GPU_US: u64 = 2_000;
    pub const BFS_DEPTH6_US: u64 = 100_000;
    pub const DOMAIN_SEARCH_US: u64 = 10_000;
}
