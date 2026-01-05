//! Numerical stability benchmarks (AP-009).

use criterion::{black_box, Criterion};
use std::time::Duration;

use super::generators::generate_poincare_point_fixed;

/// Benchmark numerical stability edge cases.
pub fn bench_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    group.measurement_time(Duration::from_secs(3));

    // NaN handling
    group.bench_function("nan_check", |b| {
        let values = [0.5f32, f32::NAN, 0.7, f32::INFINITY, 0.3];
        b.iter(|| {
            let valid: Vec<f32> = values
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .collect();
            black_box(valid)
        })
    });

    // Clamping benchmarks
    group.bench_function("clamp_0_1", |b| {
        let values: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 250.0).collect();
        b.iter(|| {
            let clamped: Vec<f32> = values.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
            black_box(clamped)
        })
    });

    // Poincare ball constraint (||x|| < 1)
    group.bench_function("poincare_constraint", |b| {
        let points: Vec<[f32; 64]> = (0..100)
            .map(|i| generate_poincare_point_fixed(i as u64 * 123))
            .collect();
        b.iter(|| {
            let valid: Vec<bool> = points
                .iter()
                .map(|p| {
                    let norm_sq: f32 = p.iter().map(|x| x * x).sum();
                    norm_sq < 1.0
                })
                .collect();
            black_box(valid)
        })
    });

    // Epsilon comparisons
    group.bench_function("epsilon_compare", |b| {
        let a = 0.1f32 + 0.2f32;
        let b_val = 0.3f32;
        let epsilon = 1e-6f32;
        b.iter(|| {
            let equal = (black_box(a) - black_box(b_val)).abs() < black_box(epsilon);
            black_box(equal)
        })
    });

    group.finish();
}
