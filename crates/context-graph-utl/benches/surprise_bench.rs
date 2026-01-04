//! Surprise computation benchmark suite.

use context_graph_utl::config::SurpriseConfig;
use context_graph_utl::surprise::{
    compute_cosine_distance, compute_kl_divergence, SurpriseCalculator,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn surprise_benchmarks(c: &mut Criterion) {
    let config = SurpriseConfig::default();
    let calculator = SurpriseCalculator::new(&config);

    // Generate test data
    let current: Vec<f32> = (0..384).map(|i| (i as f32 / 384.0)).collect();
    let history: Vec<Vec<f32>> = (0..10)
        .map(|j| (0..384).map(|i| ((i + j) as f32 / 384.0)).collect())
        .collect();

    c.bench_function("compute_surprise", |b| {
        b.iter(|| calculator.compute_surprise(black_box(&current), black_box(&history)))
    });

    let p: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];
    let q: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

    c.bench_function("compute_kl_divergence", |b| {
        b.iter(|| compute_kl_divergence(black_box(&p), black_box(&q), 1e-10))
    });

    let a: Vec<f32> = (0..384).map(|i| (i as f32 / 384.0)).collect();
    let vec_b: Vec<f32> = (0..384).map(|i| ((i + 10) as f32 / 384.0)).collect();

    c.bench_function("compute_cosine_distance", |b| {
        b.iter(|| compute_cosine_distance(black_box(&a), black_box(&vec_b)))
    });
}

criterion_group!(benches, surprise_benchmarks);
criterion_main!(benches);
