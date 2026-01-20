//! Corpus builder for creating datasets at various scales.
//!
//! Provides utilities for building test corpora at different tiers,
//! with incremental building support for memory efficiency.

use std::collections::HashMap;
use uuid::Uuid;

use context_graph_core::types::fingerprint::SemanticFingerprint;

use crate::config::{Tier, TierConfig};
use crate::datasets::{BenchmarkDataset, DatasetGenerator, GeneratorConfig};

/// Builder for creating corpora at various scales.
#[allow(dead_code)]
pub struct CorpusBuilder {
    /// Random seed for reproducibility.
    seed: u64,
    /// Generator configuration.
    gen_config: GeneratorConfig,
    /// Cached datasets by tier.
    cache: HashMap<Tier, BenchmarkDataset>,
}

impl CorpusBuilder {
    /// Create a new corpus builder.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            gen_config: GeneratorConfig {
                seed,
                ..Default::default()
            },
            cache: HashMap::new(),
        }
    }

    /// Create with custom generator config.
    pub fn with_config(gen_config: GeneratorConfig) -> Self {
        Self {
            seed: gen_config.seed,
            gen_config,
            cache: HashMap::new(),
        }
    }

    /// Build dataset for a specific tier.
    pub fn build_tier(&mut self, tier: Tier) -> &BenchmarkDataset {
        if !self.cache.contains_key(&tier) {
            let config = TierConfig::for_tier(tier);
            let mut generator = DatasetGenerator::with_config(self.gen_config.clone());
            let dataset = generator.generate_dataset(&config);
            self.cache.insert(tier, dataset);
        }

        self.cache.get(&tier).unwrap()
    }

    /// Get cached dataset if available.
    pub fn get_cached(&self, tier: Tier) -> Option<&BenchmarkDataset> {
        self.cache.get(&tier)
    }

    /// Clear cache to free memory.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Clear specific tier from cache.
    pub fn clear_tier(&mut self, tier: Tier) {
        self.cache.remove(&tier);
    }

    /// Get memory usage estimate for a tier.
    pub fn estimate_memory_bytes(tier: Tier) -> usize {
        let config = TierConfig::for_tier(tier);

        // Estimate per-fingerprint: ~46KB (from constitution)
        let fp_bytes = 46_000;

        // Queries are smaller (E1 only + metadata)
        let query_bytes = 1024 * 4 + 200; // 1024 floats + metadata

        config.memory_count * fp_bytes + config.query_count * query_bytes
    }

    /// Check if system has enough memory for a tier.
    pub fn can_fit_in_memory(tier: Tier, available_bytes: usize) -> bool {
        Self::estimate_memory_bytes(tier) < available_bytes
    }
}

/// Incremental corpus builder for very large datasets.
///
/// Builds the corpus in chunks to avoid memory exhaustion.
pub struct IncrementalCorpusBuilder {
    seed: u64,
    chunk_size: usize,
    current_chunk: usize,
    total_chunks: usize,
}

impl IncrementalCorpusBuilder {
    /// Create a new incremental builder.
    pub fn new(seed: u64, total_items: usize, chunk_size: usize) -> Self {
        let total_chunks = (total_items + chunk_size - 1) / chunk_size;

        Self {
            seed,
            chunk_size,
            current_chunk: 0,
            total_chunks,
        }
    }

    /// Check if there are more chunks to process.
    pub fn has_next(&self) -> bool {
        self.current_chunk < self.total_chunks
    }

    /// Get next chunk of fingerprints.
    ///
    /// Returns None when all chunks have been processed.
    pub fn next_chunk(&mut self) -> Option<Vec<(Uuid, SemanticFingerprint)>> {
        if !self.has_next() {
            return None;
        }

        // Generate chunk with deterministic seed based on chunk index
        let chunk_seed = self.seed + self.current_chunk as u64 * 1000;
        let gen_config = GeneratorConfig {
            seed: chunk_seed,
            ..Default::default()
        };

        // Use a small tier config for each chunk
        let mut tier_config = TierConfig::for_tier(Tier::Tier0);
        tier_config.memory_count = self.chunk_size;
        tier_config.topic_count = 10; // Use consistent topic count

        let mut generator = DatasetGenerator::with_config(gen_config);
        let dataset = generator.generate_dataset(&tier_config);

        self.current_chunk += 1;

        Some(dataset.fingerprints)
    }

    /// Get progress as (current, total) chunks.
    pub fn progress(&self) -> (usize, usize) {
        (self.current_chunk, self.total_chunks)
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.current_chunk = 0;
    }
}

/// Corpus statistics.
#[derive(Debug, Clone)]
pub struct CorpusStats {
    /// Total number of documents.
    pub document_count: usize,
    /// Number of topics.
    pub topic_count: usize,
    /// Documents per topic (min, max, mean).
    pub docs_per_topic: (usize, usize, f64),
    /// Total memory usage (bytes).
    pub memory_bytes: usize,
    /// Average fingerprint size (bytes).
    pub avg_fingerprint_bytes: usize,
}

impl CorpusStats {
    /// Compute statistics for a dataset.
    pub fn from_dataset(dataset: &BenchmarkDataset) -> Self {
        let mut topic_counts: HashMap<usize, usize> = HashMap::new();
        let mut total_fp_bytes = 0usize;

        for (_, fp) in &dataset.fingerprints {
            total_fp_bytes += fp.storage_size();
        }

        for topic in dataset.topic_assignments.values() {
            *topic_counts.entry(*topic).or_insert(0) += 1;
        }

        let counts: Vec<usize> = topic_counts.values().copied().collect();
        let min_docs = counts.iter().min().copied().unwrap_or(0);
        let max_docs = counts.iter().max().copied().unwrap_or(0);
        let mean_docs = if counts.is_empty() {
            0.0
        } else {
            counts.iter().sum::<usize>() as f64 / counts.len() as f64
        };

        let avg_fp_bytes = if dataset.document_count() > 0 {
            total_fp_bytes / dataset.document_count()
        } else {
            0
        };

        Self {
            document_count: dataset.document_count(),
            topic_count: dataset.topic_count(),
            docs_per_topic: (min_docs, max_docs, mean_docs),
            memory_bytes: total_fp_bytes,
            avg_fingerprint_bytes: avg_fp_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_builder() {
        let mut builder = CorpusBuilder::new(42);

        // Build and check tier 0
        let doc_count = {
            let dataset = builder.build_tier(Tier::Tier0);
            assert_eq!(dataset.document_count(), 100);
            assert_eq!(dataset.topic_count(), 5);
            dataset.document_count()
        };

        // Should be cached - verify via get_cached
        let cached = builder.get_cached(Tier::Tier0);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().document_count(), doc_count);
    }

    #[test]
    fn test_memory_estimation() {
        let tier0_bytes = CorpusBuilder::estimate_memory_bytes(Tier::Tier0);
        let tier1_bytes = CorpusBuilder::estimate_memory_bytes(Tier::Tier1);

        // Tier 1 should be ~10x Tier 0
        assert!(tier1_bytes > tier0_bytes * 5);
        assert!(tier1_bytes < tier0_bytes * 15);
    }

    #[test]
    fn test_incremental_builder() {
        let mut builder = IncrementalCorpusBuilder::new(42, 1000, 100);

        let mut chunk_count = 0;
        while let Some(chunk) = builder.next_chunk() {
            chunk_count += 1;
            assert!(chunk.len() <= 100);
        }

        // Should have processed all chunks (1000/100 = 10)
        assert_eq!(chunk_count, 10);
        assert!(!builder.has_next());
    }

    #[test]
    fn test_corpus_stats() {
        let mut builder = CorpusBuilder::new(42);
        let dataset = builder.build_tier(Tier::Tier0);
        let stats = CorpusStats::from_dataset(dataset);

        assert_eq!(stats.document_count, 100);
        assert_eq!(stats.topic_count, 5);
        assert!(stats.memory_bytes > 0);
        assert!(stats.avg_fingerprint_bytes > 0);
    }
}
