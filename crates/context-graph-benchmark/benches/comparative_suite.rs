//! Comparative benchmark suite using Criterion.
//!
//! Compares multi-space vs single-embedder performance at various scales.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use context_graph_benchmark::baseline::SingleEmbedderBaseline;
use context_graph_benchmark::config::{Tier, TierConfig};
use context_graph_benchmark::datasets::{DatasetGenerator, GeneratorConfig};
use context_graph_benchmark::runners::{RetrievalRunner, TopicRunner};

/// Benchmark retrieval for single-embedder baseline.
fn bench_single_embedder_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_embedder_retrieval");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for tier in [Tier::Tier0, Tier::Tier1] {
        let config = TierConfig::for_tier(tier);
        let gen_config = GeneratorConfig {
            seed: 42,
            ..Default::default()
        };
        let mut generator = DatasetGenerator::with_config(gen_config);
        let dataset = generator.generate_dataset(&config);

        // Build baseline
        let mut baseline = SingleEmbedderBaseline::new();
        for (id, fp) in &dataset.fingerprints {
            baseline.insert(*id, &fp.e1_semantic);
        }

        group.throughput(Throughput::Elements(dataset.queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("retrieval", tier.to_string()),
            &(&baseline, &dataset),
            |b, (baseline, dataset)| {
                b.iter(|| {
                    for query in &dataset.queries {
                        let _ = black_box(baseline.search(&query.embedding, 10));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark retrieval for multi-space system.
fn bench_multi_space_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_space_retrieval");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for tier in [Tier::Tier0, Tier::Tier1] {
        let config = TierConfig::for_tier(tier);
        let gen_config = GeneratorConfig {
            seed: 42,
            ..Default::default()
        };
        let mut generator = DatasetGenerator::with_config(gen_config);
        let dataset = generator.generate_dataset(&config);

        let runner = RetrievalRunner::new().with_warmup(0);

        group.throughput(Throughput::Elements(dataset.queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("retrieval", tier.to_string()),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let _ = black_box(runner.run_multi_space(dataset));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark topic detection for single-embedder baseline.
fn bench_single_embedder_topic(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_embedder_topic");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for tier in [Tier::Tier0, Tier::Tier1] {
        let config = TierConfig::for_tier(tier);
        let gen_config = GeneratorConfig {
            seed: 42,
            ..Default::default()
        };
        let mut generator = DatasetGenerator::with_config(gen_config);
        let dataset = generator.generate_dataset(&config);

        // Build baseline
        let mut baseline = SingleEmbedderBaseline::new();
        for (id, fp) in &dataset.fingerprints {
            baseline.insert(*id, &fp.e1_semantic);
        }

        group.throughput(Throughput::Elements(dataset.fingerprints.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("kmeans", tier.to_string()),
            &(&baseline, config.topic_count),
            |b, (baseline, n_topics)| {
                b.iter(|| {
                    let _ = black_box(baseline.detect_topics_kmeans(*n_topics, 50));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark topic detection for multi-space system.
fn bench_multi_space_topic(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_space_topic");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for tier in [Tier::Tier0, Tier::Tier1] {
        let config = TierConfig::for_tier(tier);
        let gen_config = GeneratorConfig {
            seed: 42,
            ..Default::default()
        };
        let mut generator = DatasetGenerator::with_config(gen_config);
        let dataset = generator.generate_dataset(&config);

        let runner = TopicRunner::new().with_expected_topics(config.topic_count);

        group.throughput(Throughput::Elements(dataset.fingerprints.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("clustering", tier.to_string()),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let _ = black_box(runner.run_multi_space(dataset));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark dataset generation.
fn bench_dataset_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_generation");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    for tier in [Tier::Tier0, Tier::Tier1] {
        let config = TierConfig::for_tier(tier);

        group.throughput(Throughput::Elements(config.memory_count as u64));

        group.bench_with_input(
            BenchmarkId::new("generate", tier.to_string()),
            &config,
            |b, config| {
                b.iter(|| {
                    let gen_config = GeneratorConfig {
                        seed: 42,
                        ..Default::default()
                    };
                    let mut generator = DatasetGenerator::with_config(gen_config);
                    let _ = black_box(generator.generate_dataset(config));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    retrieval_benches,
    bench_single_embedder_retrieval,
    bench_multi_space_retrieval
);

criterion_group!(
    topic_benches,
    bench_single_embedder_topic,
    bench_multi_space_topic
);

criterion_group!(generation_benches, bench_dataset_generation);

criterion_main!(retrieval_benches, topic_benches, generation_benches);
