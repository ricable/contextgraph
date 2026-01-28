//! Quality benchmarks for E11 entity search in causal relationships.
//!
//! Measures QUALITY metrics (not speed):
//! - Entity recall: Does E11 find entity-related relationships E1 misses?
//! - Blind spot discovery: Results E11 finds that E1 misses
//! - Source diversity: Unique sources in E11 results
//!
//! # Hypothesis
//! E11 (KEPLER) knows entity relationships that E1 misses.
//! Example: Query "Diesel" - E11 knows it's a Rust ORM, E1 may return diesel fuel content.
//!
//! # Usage
//! cargo bench -p context-graph-storage --bench e11_quality_bench

use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use tempfile::TempDir;
use uuid::Uuid;

use context_graph_core::types::CausalRelationship;
use context_graph_storage::teleological::RocksDbTeleologicalStore;

// ============================================================================
// CONSTANTS - Embedding dimensions per constitution
// ============================================================================

/// E1 embedding dimension per constitution.yaml
const E1_DIM: usize = 1024;

/// E5 embedding dimension per constitution.yaml
const E5_DIM: usize = 768;

/// E11 entity embedding dimension (768D KEPLER per constitution.yaml)
const E11_DIM: usize = 768;

// ============================================================================
// TEST DATA GENERATION
// ============================================================================

/// Generate a random normalized E1 embedding (1024D).
fn generate_e1_embedding(rng: &mut StdRng) -> Vec<f32> {
    let mut embedding: Vec<f32> = (0..E1_DIM).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

/// Generate a random normalized E5 embedding (768D).
fn generate_e5_embedding(rng: &mut StdRng) -> Vec<f32> {
    let mut embedding: Vec<f32> = (0..E5_DIM).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

/// Generate a random normalized E11 embedding (768D KEPLER).
fn generate_e11_embedding(rng: &mut StdRng) -> Vec<f32> {
    let mut embedding: Vec<f32> = (0..E11_DIM).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

/// Sample causal explanations for benchmark diversity.
const SAMPLE_EXPLANATIONS: &[&str] = &[
    "Chronic stress activates the HPA axis, leading to elevated cortisol levels.",
    "Database connection pool exhaustion causes query latency spikes.",
    "Memory leaks in the authentication service cause cascading failures.",
    "Race conditions in mutex usage corrupt shared state.",
    "Network partitions trigger split-brain in distributed consensus.",
];

const SAMPLE_CAUSES: &[&str] = &[
    "Chronic stress activates HPA axis",
    "Connection pool exhaustion",
    "Unclosed database connections",
    "Missing mutex synchronization",
    "Network partition without quorum",
];

const SAMPLE_EFFECTS: &[&str] = &[
    "Hippocampal neurodegeneration",
    "Query latency bottleneck",
    "Cascading service failures",
    "Shared state corruption",
    "Split-brain data inconsistency",
];

const MECHANISM_TYPES: &[&str] = &["direct", "mediated", "feedback", "temporal"];

/// Create a test CausalRelationship with E11 embedding.
fn create_test_relationship(rng: &mut StdRng, source_id: Uuid, index: usize) -> CausalRelationship {
    CausalRelationship::new(
        SAMPLE_CAUSES[index % SAMPLE_CAUSES.len()].to_string(),
        SAMPLE_EFFECTS[index % SAMPLE_EFFECTS.len()].to_string(),
        SAMPLE_EXPLANATIONS[index % SAMPLE_EXPLANATIONS.len()].to_string(),
        generate_e5_embedding(rng),
        generate_e5_embedding(rng),
        generate_e1_embedding(rng),
        format!("Source content for relationship {}", index),
        source_id,
        0.75 + (index % 25) as f32 * 0.01,
        MECHANISM_TYPES[index % MECHANISM_TYPES.len()].to_string(),
    )
    .with_entity_embedding(generate_e11_embedding(rng))
}

// ============================================================================
// QUALITY METRICS
// ============================================================================

/// Calculate blind spot count: results E11 finds that E1 misses.
fn blind_spot_count(e1_results: &[(Uuid, f32)], e11_results: &[(Uuid, f32)]) -> usize {
    let e1_ids: HashSet<_> = e1_results.iter().map(|(id, _)| *id).collect();
    let e11_ids: HashSet<_> = e11_results.iter().map(|(id, _)| *id).collect();
    e11_ids.difference(&e1_ids).count()
}

/// Calculate result overlap (Jaccard similarity).
fn result_overlap(a: &[(Uuid, f32)], b: &[(Uuid, f32)]) -> f32 {
    let set_a: HashSet<_> = a.iter().map(|(id, _)| *id).collect();
    let set_b: HashSet<_> = b.iter().map(|(id, _)| *id).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Calculate source diversity: unique source fingerprints in results.
fn source_diversity(
    results: &[(Uuid, f32)],
    store: &RocksDbTeleologicalStore,
    rt: &tokio::runtime::Runtime,
) -> usize {
    let mut unique_sources = HashSet::new();
    for (id, _) in results {
        if let Ok(Some(rel)) = rt.block_on(async { store.get_causal_relationship(*id).await }) {
            unique_sources.insert(rel.source_fingerprint_id);
        }
    }
    unique_sources.len()
}

// ============================================================================
// QUALITY BENCHMARKS
// ============================================================================

/// Benchmark: E11 blind spot discovery.
/// Measures what E11 finds that E1 misses.
fn bench_e11_blind_spots(c: &mut Criterion) {
    let temp_dir = TempDir::new().expect("BENCH ERROR: Failed to create temp directory");
    let store = RocksDbTeleologicalStore::open(temp_dir.path())
        .expect("BENCH ERROR: Failed to open RocksDB store");

    let rt = tokio::runtime::Runtime::new().expect("BENCH ERROR: Failed to create tokio runtime");
    let mut rng = StdRng::seed_from_u64(42);

    // Pre-populate with 500 relationships with E11 embeddings
    println!("Populating 500 causal relationships for E11 quality analysis...");
    for i in 0..500 {
        let source_id = Uuid::new_v4();
        let rel = create_test_relationship(&mut rng, source_id, i);
        rt.block_on(async { store.store_causal_relationship(&rel).await })
            .expect("BENCH ERROR: Failed to pre-store relationship");
    }

    // Generate query embeddings (different RNG state for query vs stored data)
    let e1_query = generate_e1_embedding(&mut rng);
    let e11_query = generate_e11_embedding(&mut rng);

    // Run both searches
    let e1_results = rt
        .block_on(async { store.search_causal_relationships(&e1_query, 20, None).await })
        .expect("E1 search failed");

    let e11_results = rt
        .block_on(async { store.search_causal_e11(&e11_query, 20).await })
        .expect("E11 search failed");

    // Calculate quality metrics
    let blind_spots = blind_spot_count(&e1_results, &e11_results);
    let overlap = result_overlap(&e1_results, &e11_results);
    let e1_diversity = source_diversity(&e1_results, &store, &rt);
    let e11_diversity = source_diversity(&e11_results, &store, &rt);

    println!("=== E11 Quality Metrics ===");
    println!("E1 results: {}", e1_results.len());
    println!("E11 results: {}", e11_results.len());
    println!("Blind spots (E11 finds, E1 misses): {}", blind_spots);
    println!("Result overlap (Jaccard): {:.2}%", overlap * 100.0);
    println!("E1 source diversity: {} unique sources", e1_diversity);
    println!("E11 source diversity: {} unique sources", e11_diversity);
    println!(
        "E11 blind spot rate: {:.2}%",
        (blind_spots as f32 / e11_results.len().max(1) as f32) * 100.0
    );

    // Benchmark the quality measurement itself (should be fast)
    c.bench_function("quality/e11_blind_spot_analysis", |b| {
        b.iter(|| {
            let e1_res = rt
                .block_on(async { store.search_causal_relationships(&e1_query, 20, None).await })
                .unwrap();
            let e11_res = rt
                .block_on(async { store.search_causal_e11(&e11_query, 20).await })
                .unwrap();
            let spots = blind_spot_count(&e1_res, &e11_res);
            let ovlp = result_overlap(&e1_res, &e11_res);
            criterion::black_box((spots, ovlp))
        })
    });
}

/// Benchmark: E11 vs E1 result diversity.
/// Measures unique sources in search results.
fn bench_e11_result_diversity(c: &mut Criterion) {
    let temp_dir = TempDir::new().expect("BENCH ERROR: Failed to create temp directory");
    let store = RocksDbTeleologicalStore::open(temp_dir.path())
        .expect("BENCH ERROR: Failed to open RocksDB store");

    let rt = tokio::runtime::Runtime::new().expect("BENCH ERROR: Failed to create tokio runtime");
    let mut rng = StdRng::seed_from_u64(42);

    // Pre-populate with relationships from different sources
    println!("Populating 500 causal relationships for diversity analysis...");
    let mut source_ids = Vec::new();
    for i in 0..500 {
        // Use fewer unique sources (50) to test clustering behavior
        let source_id = if i % 10 == 0 {
            let id = Uuid::new_v4();
            source_ids.push(id);
            id
        } else {
            source_ids[source_ids.len() - 1]
        };
        let rel = create_test_relationship(&mut rng, source_id, i);
        rt.block_on(async { store.store_causal_relationship(&rel).await })
            .expect("BENCH ERROR: Failed to pre-store relationship");
    }

    let e1_query = generate_e1_embedding(&mut rng);
    let e11_query = generate_e11_embedding(&mut rng);

    c.bench_function("quality/e11_diversity_analysis", |b| {
        b.iter(|| {
            let e1_res = rt
                .block_on(async { store.search_causal_relationships(&e1_query, 20, None).await })
                .unwrap();
            let e11_res = rt
                .block_on(async { store.search_causal_e11(&e11_query, 20).await })
                .unwrap();

            let e1_div = source_diversity(&e1_res, &store, &rt);
            let e11_div = source_diversity(&e11_res, &store, &rt);

            criterion::black_box((e1_div, e11_div))
        })
    });

    // Final diversity analysis
    let e1_results = rt
        .block_on(async { store.search_causal_relationships(&e1_query, 20, None).await })
        .unwrap();
    let e11_results = rt
        .block_on(async { store.search_causal_e11(&e11_query, 20).await })
        .unwrap();

    println!("=== E11 Diversity Analysis ===");
    println!(
        "E1 source diversity: {} unique sources in top 20",
        source_diversity(&e1_results, &store, &rt)
    );
    println!(
        "E11 source diversity: {} unique sources in top 20",
        source_diversity(&e11_results, &store, &rt)
    );
}

/// Benchmark: E11 ranking quality.
/// Compares top-k similarity scores between E1 and E11.
fn bench_e11_ranking_quality(c: &mut Criterion) {
    let temp_dir = TempDir::new().expect("BENCH ERROR: Failed to create temp directory");
    let store = RocksDbTeleologicalStore::open(temp_dir.path())
        .expect("BENCH ERROR: Failed to open RocksDB store");

    let rt = tokio::runtime::Runtime::new().expect("BENCH ERROR: Failed to create tokio runtime");
    let mut rng = StdRng::seed_from_u64(42);

    // Pre-populate
    for i in 0..500 {
        let source_id = Uuid::new_v4();
        let rel = create_test_relationship(&mut rng, source_id, i);
        rt.block_on(async { store.store_causal_relationship(&rel).await })
            .expect("BENCH ERROR: Failed to pre-store relationship");
    }

    let e1_query = generate_e1_embedding(&mut rng);
    let e11_query = generate_e11_embedding(&mut rng);

    c.bench_function("quality/e11_ranking_quality", |b| {
        b.iter(|| {
            let e1_res = rt
                .block_on(async { store.search_causal_relationships(&e1_query, 10, None).await })
                .unwrap();
            let e11_res = rt
                .block_on(async { store.search_causal_e11(&e11_query, 10).await })
                .unwrap();

            // Calculate mean similarity for top-10
            let e1_mean: f32 =
                e1_res.iter().map(|(_, s)| s).sum::<f32>() / e1_res.len().max(1) as f32;
            let e11_mean: f32 =
                e11_res.iter().map(|(_, s)| s).sum::<f32>() / e11_res.len().max(1) as f32;

            criterion::black_box((e1_mean, e11_mean))
        })
    });

    // Print quality stats
    let e1_results = rt
        .block_on(async { store.search_causal_relationships(&e1_query, 10, None).await })
        .unwrap();
    let e11_results = rt
        .block_on(async { store.search_causal_e11(&e11_query, 10).await })
        .unwrap();

    let e1_mean: f32 =
        e1_results.iter().map(|(_, s)| s).sum::<f32>() / e1_results.len().max(1) as f32;
    let e11_mean: f32 =
        e11_results.iter().map(|(_, s)| s).sum::<f32>() / e11_results.len().max(1) as f32;

    println!("=== E11 Ranking Quality ===");
    println!("E1 mean top-10 similarity: {:.4}", e1_mean);
    println!("E11 mean top-10 similarity: {:.4}", e11_mean);
    println!(
        "E1 top result similarity: {:.4}",
        e1_results.first().map(|(_, s)| *s).unwrap_or(0.0)
    );
    println!(
        "E11 top result similarity: {:.4}",
        e11_results.first().map(|(_, s)| *s).unwrap_or(0.0)
    );
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    benches,
    bench_e11_blind_spots,
    bench_e11_result_diversity,
    bench_e11_ranking_quality,
);
criterion_main!(benches);
