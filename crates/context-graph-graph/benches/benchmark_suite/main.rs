//! Comprehensive benchmark suite for context-graph-graph crate.
//!
//! # Task: M04-T29 - Benchmark Suite Implementation
//!
//! This benchmark suite provides CPU + GPU performance validation for:
//! - Poincare distance computation (hyperbolic geometry)
//! - Entailment cone membership scoring
//! - BFS graph traversal
//! - GPU memory manager operations
//! - Domain search with NT modulation
//!
//! # Performance Targets (from constitution)
//!
//! | Benchmark | Target |
//! |-----------|--------|
//! | poincare_single | <10us |
//! | poincare_1k_batch | <1ms GPU / <100ms CPU |
//! | cone_single | <15us |
//! | cone_1k_batch | <2ms GPU / <200ms CPU |
//! | bfs_depth6 | <100ms |
//! | faiss_1M_k100 | <2ms |
//! | domain_search | <10ms |
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench --package context-graph-graph
//!
//! # Run specific benchmark
//! cargo bench --package context-graph-graph -- poincare
//!
//! # With GPU features
//! cargo bench --package context-graph-graph --features faiss-gpu
//! ```
//!
//! # JSON Export for CI
//!
//! Results are exported to `target/criterion/benchmark_results.json`
//! for CI integration and regression detection.
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004: Performance specifications
//! - AP-009: NaN/Infinity handling
//! - perf.latency.*: All latency targets

use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

mod config;
mod generators;

mod cone_benches;
mod domain_search_benches;
mod gpu_memory_benches;
mod poincare_benches;
mod stability_benches;
mod traversal_benches;

// ============================================================================
// CRITERION SETUP
// ============================================================================

criterion_group!(
    name = poincare_bench_group;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = poincare_benches::bench_poincare_distance
);

criterion_group!(
    name = cone_bench_group;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5));
    targets = cone_benches::bench_cone_membership
);

criterion_group!(
    name = traversal_bench_group;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(5));
    targets = traversal_benches::bench_bfs_traversal
);

criterion_group!(
    name = gpu_memory_bench_group;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(3));
    targets = gpu_memory_benches::bench_gpu_memory_manager
);

criterion_group!(
    name = domain_search_bench_group;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(3));
    targets = domain_search_benches::bench_domain_search_modulation
);

criterion_group!(
    name = stability_bench_group;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(3));
    targets = stability_benches::bench_numerical_stability
);

criterion_main!(
    poincare_bench_group,
    cone_bench_group,
    traversal_bench_group,
    gpu_memory_bench_group,
    domain_search_bench_group,
    stability_bench_group
);
