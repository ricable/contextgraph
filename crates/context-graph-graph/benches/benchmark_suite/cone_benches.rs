//! Entailment cone membership benchmarks.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use super::config;
use super::generators::{
    generate_cone_batch_flat, generate_cone_data_fixed, generate_poincare_batch_flat,
    generate_poincare_point_fixed,
};

/// Benchmark entailment cone membership scoring.
pub fn bench_cone_membership(c: &mut Criterion) {
    let mut group = c.benchmark_group("cone_membership");
    group.measurement_time(Duration::from_secs(5));

    // Single cone check
    let (apex, aperture) = generate_cone_data_fixed(1);
    let point = generate_poincare_point_fixed(100);

    group.bench_function("single_cpu", |b| {
        b.iter(|| {
            context_graph_cuda::cone_membership_score_cpu(
                black_box(&apex),
                black_box(aperture),
                black_box(&point),
                black_box(config::POINCARE_CURVATURE),
            )
        })
    });

    // Batch cone checks
    for &batch_size in config::BATCH_SIZES {
        let cones = generate_cone_batch_flat(batch_size);
        let points = generate_poincare_batch_flat(batch_size);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_cpu", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    context_graph_cuda::cone_check_batch_cpu(
                        black_box(&cones),
                        black_box(&points),
                        black_box(bs),
                        black_box(bs),
                        black_box(config::POINCARE_CURVATURE),
                    )
                })
            },
        );
    }

    // Aperture variation benchmarks
    for aperture_val in [0.1, 0.5, 1.0, std::f32::consts::FRAC_PI_2 - 0.1] {
        group.bench_with_input(
            BenchmarkId::new("aperture", format!("{:.2}", aperture_val)),
            &aperture_val,
            |b, &ap| {
                b.iter(|| {
                    context_graph_cuda::cone_membership_score_cpu(
                        black_box(&apex),
                        black_box(ap),
                        black_box(&point),
                        black_box(config::POINCARE_CURVATURE),
                    )
                })
            },
        );
    }

    group.finish();
}
