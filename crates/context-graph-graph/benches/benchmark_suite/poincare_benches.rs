//! Poincare distance benchmarks.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use super::config;
use super::generators::{generate_poincare_batch_flat, generate_poincare_point_fixed};

/// Benchmark Poincare distance computation.
pub fn bench_poincare_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("poincare_distance");
    group.measurement_time(Duration::from_secs(5));

    // Single distance computation
    let p1 = generate_poincare_point_fixed(1);
    let p2 = generate_poincare_point_fixed(2);

    group.bench_function("single_cpu", |b| {
        b.iter(|| {
            context_graph_cuda::poincare_distance_cpu(
                black_box(&p1),
                black_box(&p2),
                black_box(config::POINCARE_CURVATURE),
            )
        })
    });

    // Batch distance computation
    for &batch_size in config::BATCH_SIZES {
        let queries = generate_poincare_batch_flat(batch_size);
        let database = generate_poincare_batch_flat(batch_size);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_cpu", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    context_graph_cuda::poincare_distance_batch_cpu(
                        black_box(&queries),
                        black_box(&database),
                        black_box(bs),
                        black_box(bs),
                        black_box(config::POINCARE_CURVATURE),
                    )
                })
            },
        );
    }

    // Edge cases for numerical stability (AP-009)
    group.bench_function("near_origin", |b| {
        let mut origin_near = [0.0f32; 64];
        origin_near[0] = 1e-7;
        let p = generate_poincare_point_fixed(100);
        b.iter(|| {
            context_graph_cuda::poincare_distance_cpu(
                black_box(&origin_near),
                black_box(&p),
                black_box(config::POINCARE_CURVATURE),
            )
        })
    });

    group.bench_function("near_boundary", |b| {
        let mut boundary_near = generate_poincare_point_fixed(200);
        // Scale to be very close to boundary
        let norm: f32 = boundary_near.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = 0.999 / norm;
        for val in &mut boundary_near {
            *val *= scale;
        }
        let p = generate_poincare_point_fixed(201);

        b.iter(|| {
            context_graph_cuda::poincare_distance_cpu(
                black_box(&boundary_near),
                black_box(&p),
                black_box(config::POINCARE_CURVATURE),
            )
        })
    });

    group.finish();
}
