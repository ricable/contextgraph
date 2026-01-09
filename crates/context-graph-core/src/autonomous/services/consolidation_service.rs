//! NORTH-013: Memory Consolidation Service
//!
//! This service merges similar memories to reduce redundancy while preserving
//! alignment with the North Star goal. It uses cosine similarity for content
//! comparison and respects configurable thresholds for merging decisions.

use crate::autonomous::curation::{ConsolidationConfig, ConsolidationReport, MemoryId};
use serde::{Deserialize, Serialize};

/// Content of a memory for consolidation purposes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryContent {
    /// Unique identifier
    pub id: MemoryId,
    /// Embedding vector (normalized)
    pub embedding: Vec<f32>,
    /// Text content
    pub text: String,
    /// North Star alignment score
    pub alignment: f32,
    /// Access count for importance weighting
    pub access_count: u32,
}

impl MemoryContent {
    /// Create a new memory content
    pub fn new(id: MemoryId, embedding: Vec<f32>, text: String, alignment: f32) -> Self {
        Self {
            id,
            embedding,
            text,
            alignment,
            access_count: 0,
        }
    }

    /// Create with access count
    pub fn with_access_count(mut self, count: u32) -> Self {
        self.access_count = count;
        self
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }
}

/// A pair of memories to evaluate for consolidation
#[derive(Clone, Debug)]
pub struct MemoryPair {
    /// First memory
    pub a: MemoryContent,
    /// Second memory
    pub b: MemoryContent,
}

impl MemoryPair {
    /// Create a new memory pair
    pub fn new(a: MemoryContent, b: MemoryContent) -> Self {
        Self { a, b }
    }

    /// Get alignment difference between the two memories
    pub fn alignment_diff(&self) -> f32 {
        (self.a.alignment - self.b.alignment).abs()
    }
}

/// Candidate for consolidation with computed metrics
#[derive(Clone, Debug)]
pub struct ServiceConsolidationCandidate {
    /// Source memory IDs to merge
    pub source_ids: Vec<MemoryId>,
    /// Target memory ID (result of merge)
    pub target_id: MemoryId,
    /// Similarity score between sources
    pub similarity: f32,
    /// Combined alignment of merged memory
    pub combined_alignment: f32,
}

impl ServiceConsolidationCandidate {
    /// Create a new consolidation candidate
    pub fn new(
        source_ids: Vec<MemoryId>,
        target_id: MemoryId,
        similarity: f32,
        combined_alignment: f32,
    ) -> Self {
        Self {
            source_ids,
            target_id,
            similarity,
            combined_alignment,
        }
    }
}

/// Service for consolidating similar memories
#[derive(Clone, Debug)]
pub struct ConsolidationService {
    /// Configuration
    config: ConsolidationConfig,
    /// Daily merge counter
    daily_merges: u32,
}

impl Default for ConsolidationService {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolidationService {
    /// Create a new consolidation service with default config
    pub fn new() -> Self {
        Self {
            config: ConsolidationConfig::default(),
            daily_merges: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ConsolidationConfig) -> Self {
        Self {
            config,
            daily_merges: 0,
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }

    /// Get the daily merge count
    pub fn daily_merges(&self) -> u32 {
        self.daily_merges
    }

    /// Reset daily merge counter (called at start of new day)
    pub fn reset_daily_counter(&mut self) {
        self.daily_merges = 0;
    }

    /// Find consolidation candidates from memory pairs
    ///
    /// Evaluates each pair and returns candidates that meet the threshold criteria.
    pub fn find_consolidation_candidates(
        &self,
        memories: &[MemoryPair],
    ) -> Vec<ServiceConsolidationCandidate> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut candidates = Vec::new();

        for pair in memories {
            let similarity = self.compute_similarity(&pair.a, &pair.b);
            let alignment_diff = pair.alignment_diff();

            if self.should_consolidate(similarity, alignment_diff) {
                let combined_alignment =
                    self.compute_combined_alignment(&[pair.a.alignment, pair.b.alignment]);
                let target_id = MemoryId::new();

                candidates.push(ServiceConsolidationCandidate::new(
                    vec![pair.a.id.clone(), pair.b.id.clone()],
                    target_id,
                    similarity,
                    combined_alignment,
                ));
            }
        }

        candidates
    }

    /// Compute cosine similarity between two memory contents
    ///
    /// Returns a value in [0, 1] where 1 means identical.
    /// Fails fast if embeddings have different dimensions.
    pub fn compute_similarity(&self, a: &MemoryContent, b: &MemoryContent) -> f32 {
        if a.embedding.len() != b.embedding.len() {
            return 0.0; // Fail fast: incompatible dimensions
        }

        if a.embedding.is_empty() {
            return 0.0; // Fail fast: empty embeddings
        }

        // Compute dot product
        let dot: f32 = a
            .embedding
            .iter()
            .zip(b.embedding.iter())
            .map(|(x, y)| x * y)
            .sum();

        // Compute magnitudes
        let mag_a: f32 = a.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
            return 0.0; // Fail fast: zero magnitude
        }

        let similarity = dot / (mag_a * mag_b);

        // Clamp to [0, 1] (cosine can be negative but we only care about positive similarity)
        similarity.clamp(0.0, 1.0)
    }

    /// Determine if two memories should be consolidated
    ///
    /// Returns true if similarity exceeds threshold AND alignment difference is within tolerance.
    pub fn should_consolidate(&self, similarity: f32, alignment_diff: f32) -> bool {
        similarity >= self.config.similarity_threshold
            && alignment_diff <= self.config.theta_diff_threshold
    }

    /// Perform consolidation on a set of candidates
    ///
    /// Respects daily merge limits and returns a report of actions taken.
    pub fn consolidate(
        &mut self,
        candidates: &[ServiceConsolidationCandidate],
    ) -> ConsolidationReport {
        let mut report = ConsolidationReport {
            candidates_found: candidates.len(),
            merged: 0,
            skipped: 0,
            daily_limit_reached: false,
        };

        if !self.config.enabled {
            report.skipped = candidates.len();
            return report;
        }

        for _candidate in candidates {
            if self.daily_merges >= self.config.max_daily_merges {
                report.daily_limit_reached = true;
                report.skipped += 1;
                continue;
            }

            // Perform merge (in real implementation, this would update storage)
            self.daily_merges += 1;
            report.merged += 1;
        }

        report
    }

    /// Merge multiple memory contents into one
    ///
    /// Combines text and averages embeddings weighted by access count.
    /// The merged memory gets a new ID.
    pub fn merge_memories(&self, sources: &[MemoryContent]) -> MemoryContent {
        if sources.is_empty() {
            return MemoryContent::new(MemoryId::new(), Vec::new(), String::new(), 0.0);
        }

        if sources.len() == 1 {
            return sources[0].clone();
        }

        // Determine embedding dimension
        let dim = sources.iter().map(|s| s.dimension()).max().unwrap_or(0);

        if dim == 0 {
            // No valid embeddings, just merge text
            let combined_text = sources
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" | ");

            let alignments: Vec<f32> = sources.iter().map(|s| s.alignment).collect();
            let combined_alignment = self.compute_combined_alignment(&alignments);
            let total_access: u32 = sources.iter().map(|s| s.access_count).sum();

            return MemoryContent::new(
                MemoryId::new(),
                Vec::new(),
                combined_text,
                combined_alignment,
            )
            .with_access_count(total_access);
        }

        // Compute weighted average embedding
        let total_weight: f32 = sources
            .iter()
            .map(|s| s.access_count as f32 + 1.0) // +1 to avoid zero weight
            .sum();

        let mut merged_embedding = vec![0.0f32; dim];

        for source in sources {
            if source.dimension() != dim {
                continue; // Skip incompatible dimensions
            }

            let weight = (source.access_count as f32 + 1.0) / total_weight;
            for (i, val) in source.embedding.iter().enumerate() {
                merged_embedding[i] += val * weight;
            }
        }

        // Normalize the merged embedding
        let magnitude: f32 = merged_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > f32::EPSILON {
            for val in &mut merged_embedding {
                *val /= magnitude;
            }
        }

        // Combine text content
        let combined_text = sources
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ");

        // Compute combined alignment
        let alignments: Vec<f32> = sources.iter().map(|s| s.alignment).collect();
        let combined_alignment = self.compute_combined_alignment(&alignments);

        // Sum access counts
        let total_access: u32 = sources.iter().map(|s| s.access_count).sum();

        MemoryContent::new(
            MemoryId::new(),
            merged_embedding,
            combined_text,
            combined_alignment,
        )
        .with_access_count(total_access)
    }

    /// Compute combined alignment from multiple alignment scores
    ///
    /// Uses weighted average favoring higher alignments.
    pub fn compute_combined_alignment(&self, alignments: &[f32]) -> f32 {
        if alignments.is_empty() {
            return 0.0;
        }

        if alignments.len() == 1 {
            return alignments[0];
        }

        // Weight by alignment^2 to favor higher alignments
        let weights: Vec<f32> = alignments.iter().map(|a| a * a).collect();
        let total_weight: f32 = weights.iter().sum();

        if total_weight < f32::EPSILON {
            // All alignments near zero, use simple average
            return alignments.iter().sum::<f32>() / alignments.len() as f32;
        }

        let weighted_sum: f32 = alignments
            .iter()
            .zip(weights.iter())
            .map(|(a, w)| a * w)
            .sum();

        weighted_sum / total_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_memory(embedding: Vec<f32>, text: &str, alignment: f32) -> MemoryContent {
        MemoryContent::new(MemoryId::new(), embedding, text.to_string(), alignment)
    }

    // ========== MemoryContent Tests ==========

    #[test]
    fn test_memory_content_new() {
        let embedding = vec![0.5, 0.5, 0.5];
        let mem = create_test_memory(embedding.clone(), "test content", 0.8);

        assert_eq!(mem.embedding, embedding);
        assert_eq!(mem.text, "test content");
        assert!((mem.alignment - 0.8).abs() < f32::EPSILON);
        assert_eq!(mem.access_count, 0);

        println!("[PASS] test_memory_content_new");
    }

    #[test]
    fn test_memory_content_with_access_count() {
        let mem = create_test_memory(vec![1.0, 0.0], "test", 0.5).with_access_count(42);

        assert_eq!(mem.access_count, 42);

        println!("[PASS] test_memory_content_with_access_count");
    }

    #[test]
    fn test_memory_content_dimension() {
        let mem = create_test_memory(vec![1.0, 2.0, 3.0, 4.0], "test", 0.7);
        assert_eq!(mem.dimension(), 4);

        let empty_mem = create_test_memory(vec![], "empty", 0.5);
        assert_eq!(empty_mem.dimension(), 0);

        println!("[PASS] test_memory_content_dimension");
    }

    // ========== MemoryPair Tests ==========

    #[test]
    fn test_memory_pair_new() {
        let a = create_test_memory(vec![1.0], "a", 0.8);
        let b = create_test_memory(vec![1.0], "b", 0.6);

        let pair = MemoryPair::new(a.clone(), b.clone());
        assert_eq!(pair.a.text, "a");
        assert_eq!(pair.b.text, "b");

        println!("[PASS] test_memory_pair_new");
    }

    #[test]
    fn test_memory_pair_alignment_diff() {
        let a = create_test_memory(vec![1.0], "a", 0.9);
        let b = create_test_memory(vec![1.0], "b", 0.7);

        let pair = MemoryPair::new(a, b);
        assert!((pair.alignment_diff() - 0.2).abs() < f32::EPSILON);

        // Test symmetry
        let c = create_test_memory(vec![1.0], "c", 0.3);
        let d = create_test_memory(vec![1.0], "d", 0.8);
        let pair2 = MemoryPair::new(c, d);
        assert!((pair2.alignment_diff() - 0.5).abs() < f32::EPSILON);

        println!("[PASS] test_memory_pair_alignment_diff");
    }

    // ========== ServiceConsolidationCandidate Tests ==========

    #[test]
    fn test_consolidation_candidate_new() {
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let target = MemoryId::new();

        let candidate = ServiceConsolidationCandidate::new(
            vec![id1.clone(), id2.clone()],
            target.clone(),
            0.95,
            0.85,
        );

        assert_eq!(candidate.source_ids.len(), 2);
        assert!((candidate.similarity - 0.95).abs() < f32::EPSILON);
        assert!((candidate.combined_alignment - 0.85).abs() < f32::EPSILON);

        println!("[PASS] test_consolidation_candidate_new");
    }

    // ========== ConsolidationService Construction Tests ==========

    #[test]
    fn test_service_new() {
        let service = ConsolidationService::new();
        let config = service.config();

        assert!(config.enabled);
        assert!((config.similarity_threshold - 0.92).abs() < f32::EPSILON);
        assert_eq!(config.max_daily_merges, 50);
        assert!((config.theta_diff_threshold - 0.05).abs() < f32::EPSILON);
        assert_eq!(service.daily_merges(), 0);

        println!("[PASS] test_service_new");
    }

    #[test]
    fn test_service_default() {
        let service = ConsolidationService::default();
        assert!(service.config().enabled);
        assert_eq!(service.daily_merges(), 0);

        println!("[PASS] test_service_default");
    }

    #[test]
    fn test_service_with_config() {
        let config = ConsolidationConfig {
            enabled: false,
            similarity_threshold: 0.85,
            max_daily_merges: 100,
            theta_diff_threshold: 0.10,
        };

        let service = ConsolidationService::with_config(config);

        assert!(!service.config().enabled);
        assert!((service.config().similarity_threshold - 0.85).abs() < f32::EPSILON);
        assert_eq!(service.config().max_daily_merges, 100);

        println!("[PASS] test_service_with_config");
    }

    #[test]
    fn test_service_reset_daily_counter() {
        let config = ConsolidationConfig::default();
        let mut service = ConsolidationService::with_config(config);

        // Create and consolidate some candidates
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let candidates = vec![ServiceConsolidationCandidate::new(
            vec![id1, id2],
            MemoryId::new(),
            0.95,
            0.8,
        )];

        service.consolidate(&candidates);
        assert_eq!(service.daily_merges(), 1);

        service.reset_daily_counter();
        assert_eq!(service.daily_merges(), 0);

        println!("[PASS] test_service_reset_daily_counter");
    }

    // ========== compute_similarity Tests ==========

    #[test]
    fn test_compute_similarity_identical() {
        let service = ConsolidationService::new();

        // Normalized vector
        let embedding = vec![0.6, 0.8]; // magnitude = 1.0
        let a = create_test_memory(embedding.clone(), "a", 0.8);
        let b = create_test_memory(embedding, "b", 0.8);

        let sim = service.compute_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Identical vectors should have similarity 1.0, got {}",
            sim
        );

        println!("[PASS] test_compute_similarity_identical");
    }

    #[test]
    fn test_compute_similarity_orthogonal() {
        let service = ConsolidationService::new();

        let a = create_test_memory(vec![1.0, 0.0], "a", 0.8);
        let b = create_test_memory(vec![0.0, 1.0], "b", 0.8);

        let sim = service.compute_similarity(&a, &b);
        assert!(
            sim.abs() < 0.001,
            "Orthogonal vectors should have similarity 0.0, got {}",
            sim
        );

        println!("[PASS] test_compute_similarity_orthogonal");
    }

    #[test]
    fn test_compute_similarity_real_calculation() {
        let service = ConsolidationService::new();

        // Two similar but not identical vectors
        let a = create_test_memory(vec![0.8, 0.6, 0.0], "a", 0.8);
        let b = create_test_memory(vec![0.7, 0.7, 0.1], "b", 0.8);

        // Manual calculation:
        // dot = 0.8*0.7 + 0.6*0.7 + 0.0*0.1 = 0.56 + 0.42 + 0.0 = 0.98
        // |a| = sqrt(0.64 + 0.36 + 0.0) = 1.0
        // |b| = sqrt(0.49 + 0.49 + 0.01) = sqrt(0.99) ≈ 0.995
        // sim = 0.98 / (1.0 * 0.995) ≈ 0.985

        let sim = service.compute_similarity(&a, &b);
        assert!(
            (sim - 0.985).abs() < 0.01,
            "Expected similarity ~0.985, got {}",
            sim
        );

        println!("[PASS] test_compute_similarity_real_calculation");
    }

    #[test]
    fn test_compute_similarity_empty_embedding() {
        let service = ConsolidationService::new();

        let a = create_test_memory(vec![], "a", 0.8);
        let b = create_test_memory(vec![], "b", 0.8);

        let sim = service.compute_similarity(&a, &b);
        assert!(
            sim.abs() < f32::EPSILON,
            "Empty embeddings should return 0.0"
        );

        println!("[PASS] test_compute_similarity_empty_embedding");
    }

    #[test]
    fn test_compute_similarity_different_dimensions() {
        let service = ConsolidationService::new();

        let a = create_test_memory(vec![1.0, 0.0, 0.0], "a", 0.8);
        let b = create_test_memory(vec![1.0, 0.0], "b", 0.8);

        let sim = service.compute_similarity(&a, &b);
        assert!(
            sim.abs() < f32::EPSILON,
            "Different dimensions should return 0.0"
        );

        println!("[PASS] test_compute_similarity_different_dimensions");
    }

    #[test]
    fn test_compute_similarity_zero_magnitude() {
        let service = ConsolidationService::new();

        let a = create_test_memory(vec![0.0, 0.0, 0.0], "a", 0.8);
        let b = create_test_memory(vec![1.0, 0.0, 0.0], "b", 0.8);

        let sim = service.compute_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON, "Zero magnitude should return 0.0");

        println!("[PASS] test_compute_similarity_zero_magnitude");
    }

    #[test]
    fn test_compute_similarity_high_dimensional() {
        let service = ConsolidationService::new();

        // Simulate realistic 384-dim embedding
        let mut vec_a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut vec_b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01 + 0.1).sin()).collect();

        // Normalize
        let mag_a: f32 = vec_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = vec_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut vec_a {
            *v /= mag_a;
        }
        for v in &mut vec_b {
            *v /= mag_b;
        }

        let a = create_test_memory(vec_a, "a", 0.8);
        let b = create_test_memory(vec_b, "b", 0.8);

        let sim = service.compute_similarity(&a, &b);
        assert!(
            sim > 0.9 && sim < 1.0,
            "Similar high-dim vectors should have high similarity: {}",
            sim
        );

        println!("[PASS] test_compute_similarity_high_dimensional");
    }

    // ========== should_consolidate Tests ==========

    #[test]
    fn test_should_consolidate_true() {
        let service = ConsolidationService::new();
        // Default: similarity >= 0.92, theta_diff <= 0.05

        assert!(service.should_consolidate(0.95, 0.03));
        assert!(service.should_consolidate(0.92, 0.05)); // Boundary

        println!("[PASS] test_should_consolidate_true");
    }

    #[test]
    fn test_should_consolidate_false_low_similarity() {
        let service = ConsolidationService::new();

        assert!(!service.should_consolidate(0.90, 0.03));
        assert!(!service.should_consolidate(0.919, 0.03)); // Just below

        println!("[PASS] test_should_consolidate_false_low_similarity");
    }

    #[test]
    fn test_should_consolidate_false_high_theta_diff() {
        let service = ConsolidationService::new();

        assert!(!service.should_consolidate(0.95, 0.10));
        assert!(!service.should_consolidate(0.95, 0.051)); // Just above

        println!("[PASS] test_should_consolidate_false_high_theta_diff");
    }

    #[test]
    fn test_should_consolidate_custom_thresholds() {
        let config = ConsolidationConfig {
            enabled: true,
            similarity_threshold: 0.80,
            max_daily_merges: 50,
            theta_diff_threshold: 0.15,
        };
        let service = ConsolidationService::with_config(config);

        assert!(service.should_consolidate(0.85, 0.10));
        assert!(!service.should_consolidate(0.75, 0.10));
        assert!(!service.should_consolidate(0.85, 0.20));

        println!("[PASS] test_should_consolidate_custom_thresholds");
    }

    // ========== find_consolidation_candidates Tests ==========

    #[test]
    fn test_find_candidates_empty() {
        let service = ConsolidationService::new();
        let candidates = service.find_consolidation_candidates(&[]);
        assert!(candidates.is_empty());

        println!("[PASS] test_find_candidates_empty");
    }

    #[test]
    fn test_find_candidates_disabled() {
        let config = ConsolidationConfig {
            enabled: false,
            ..Default::default()
        };
        let service = ConsolidationService::with_config(config);

        // Even with identical memories, should return empty
        let embedding = vec![1.0, 0.0, 0.0];
        let a = create_test_memory(embedding.clone(), "a", 0.8);
        let b = create_test_memory(embedding, "b", 0.8);

        let pairs = vec![MemoryPair::new(a, b)];
        let candidates = service.find_consolidation_candidates(&pairs);
        assert!(candidates.is_empty());

        println!("[PASS] test_find_candidates_disabled");
    }

    #[test]
    fn test_find_candidates_found() {
        let service = ConsolidationService::new();

        // Create nearly identical memories
        let embedding = vec![0.6, 0.8, 0.0];
        let a = create_test_memory(embedding.clone(), "a", 0.82);
        let b = create_test_memory(embedding, "b", 0.80); // diff = 0.02 < 0.05

        let pairs = vec![MemoryPair::new(a, b)];
        let candidates = service.find_consolidation_candidates(&pairs);

        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].similarity - 1.0).abs() < 0.001);
        assert_eq!(candidates[0].source_ids.len(), 2);

        println!("[PASS] test_find_candidates_found");
    }

    #[test]
    fn test_find_candidates_filtered() {
        let service = ConsolidationService::new();

        // Pair 1: High similarity, low diff -> should be found
        let a1 = create_test_memory(vec![1.0, 0.0], "a1", 0.80);
        let b1 = create_test_memory(vec![1.0, 0.0], "b1", 0.78);

        // Pair 2: Low similarity -> should not be found
        let a2 = create_test_memory(vec![1.0, 0.0], "a2", 0.80);
        let b2 = create_test_memory(vec![0.0, 1.0], "b2", 0.80);

        // Pair 3: High similarity but high diff -> should not be found
        let a3 = create_test_memory(vec![1.0, 0.0], "a3", 0.90);
        let b3 = create_test_memory(vec![1.0, 0.0], "b3", 0.70);

        let pairs = vec![
            MemoryPair::new(a1, b1),
            MemoryPair::new(a2, b2),
            MemoryPair::new(a3, b3),
        ];

        let candidates = service.find_consolidation_candidates(&pairs);
        assert_eq!(candidates.len(), 1);

        println!("[PASS] test_find_candidates_filtered");
    }

    // ========== consolidate Tests ==========

    #[test]
    fn test_consolidate_empty() {
        let mut service = ConsolidationService::new();
        let report = service.consolidate(&[]);

        assert_eq!(report.candidates_found, 0);
        assert_eq!(report.merged, 0);
        assert_eq!(report.skipped, 0);
        assert!(!report.daily_limit_reached);

        println!("[PASS] test_consolidate_empty");
    }

    #[test]
    fn test_consolidate_disabled() {
        let config = ConsolidationConfig {
            enabled: false,
            ..Default::default()
        };
        let mut service = ConsolidationService::with_config(config);

        let candidates = vec![ServiceConsolidationCandidate::new(
            vec![MemoryId::new(), MemoryId::new()],
            MemoryId::new(),
            0.95,
            0.8,
        )];

        let report = service.consolidate(&candidates);
        assert_eq!(report.merged, 0);
        assert_eq!(report.skipped, 1);

        println!("[PASS] test_consolidate_disabled");
    }

    #[test]
    fn test_consolidate_merges() {
        let mut service = ConsolidationService::new();

        let candidates = vec![
            ServiceConsolidationCandidate::new(
                vec![MemoryId::new(), MemoryId::new()],
                MemoryId::new(),
                0.95,
                0.8,
            ),
            ServiceConsolidationCandidate::new(
                vec![MemoryId::new(), MemoryId::new()],
                MemoryId::new(),
                0.93,
                0.85,
            ),
        ];

        let report = service.consolidate(&candidates);
        assert_eq!(report.candidates_found, 2);
        assert_eq!(report.merged, 2);
        assert_eq!(report.skipped, 0);
        assert_eq!(service.daily_merges(), 2);

        println!("[PASS] test_consolidate_merges");
    }

    #[test]
    fn test_consolidate_daily_limit() {
        let config = ConsolidationConfig {
            enabled: true,
            similarity_threshold: 0.92,
            max_daily_merges: 2,
            theta_diff_threshold: 0.05,
        };
        let mut service = ConsolidationService::with_config(config);

        let candidates: Vec<_> = (0..5)
            .map(|_| {
                ServiceConsolidationCandidate::new(
                    vec![MemoryId::new(), MemoryId::new()],
                    MemoryId::new(),
                    0.95,
                    0.8,
                )
            })
            .collect();

        let report = service.consolidate(&candidates);

        assert_eq!(report.candidates_found, 5);
        assert_eq!(report.merged, 2);
        assert_eq!(report.skipped, 3);
        assert!(report.daily_limit_reached);

        println!("[PASS] test_consolidate_daily_limit");
    }

    // ========== merge_memories Tests ==========

    #[test]
    fn test_merge_memories_empty() {
        let service = ConsolidationService::new();
        let merged = service.merge_memories(&[]);

        assert!(merged.embedding.is_empty());
        assert!(merged.text.is_empty());
        assert!(merged.alignment.abs() < f32::EPSILON);

        println!("[PASS] test_merge_memories_empty");
    }

    #[test]
    fn test_merge_memories_single() {
        let service = ConsolidationService::new();

        let mem = create_test_memory(vec![1.0, 0.0], "original", 0.8).with_access_count(10);

        let merged = service.merge_memories(std::slice::from_ref(&mem));

        assert_eq!(merged.embedding, mem.embedding);
        assert_eq!(merged.text, "original");
        assert!((merged.alignment - 0.8).abs() < f32::EPSILON);

        println!("[PASS] test_merge_memories_single");
    }

    #[test]
    fn test_merge_memories_text_combined() {
        let service = ConsolidationService::new();

        let a = create_test_memory(vec![1.0, 0.0], "first", 0.8);
        let b = create_test_memory(vec![1.0, 0.0], "second", 0.8);
        let c = create_test_memory(vec![1.0, 0.0], "third", 0.8);

        let merged = service.merge_memories(&[a, b, c]);

        assert!(merged.text.contains("first"));
        assert!(merged.text.contains("second"));
        assert!(merged.text.contains("third"));
        assert!(merged.text.contains(" | "));

        println!("[PASS] test_merge_memories_text_combined");
    }

    #[test]
    fn test_merge_memories_embedding_averaged() {
        let service = ConsolidationService::new();

        // Two perpendicular unit vectors with equal weight
        let a = create_test_memory(vec![1.0, 0.0], "a", 0.8).with_access_count(0);
        let b = create_test_memory(vec![0.0, 1.0], "b", 0.8).with_access_count(0);

        let merged = service.merge_memories(&[a, b]);

        // Should be normalized average: [0.5, 0.5] normalized = [0.707, 0.707]
        assert_eq!(merged.dimension(), 2);
        let expected = 1.0 / (2.0_f32).sqrt();
        assert!(
            (merged.embedding[0] - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            merged.embedding[0]
        );
        assert!(
            (merged.embedding[1] - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            merged.embedding[1]
        );

        println!("[PASS] test_merge_memories_embedding_averaged");
    }

    #[test]
    fn test_merge_memories_weighted_by_access() {
        let service = ConsolidationService::new();

        // Higher access count should have more influence
        let a = create_test_memory(vec![1.0, 0.0], "a", 0.8).with_access_count(9); // weight = 10
        let b = create_test_memory(vec![0.0, 1.0], "b", 0.8).with_access_count(0); // weight = 1

        let merged = service.merge_memories(&[a, b]);

        // a has 10x weight, so embedding should be closer to [1, 0]
        // Weighted: [10/11, 1/11] normalized
        assert!(
            merged.embedding[0] > merged.embedding[1],
            "First component should dominate: {:?}",
            merged.embedding
        );

        println!("[PASS] test_merge_memories_weighted_by_access");
    }

    #[test]
    fn test_merge_memories_access_count_summed() {
        let service = ConsolidationService::new();

        let a = create_test_memory(vec![1.0], "a", 0.8).with_access_count(5);
        let b = create_test_memory(vec![1.0], "b", 0.8).with_access_count(10);
        let c = create_test_memory(vec![1.0], "c", 0.8).with_access_count(7);

        let merged = service.merge_memories(&[a, b, c]);
        assert_eq!(merged.access_count, 22);

        println!("[PASS] test_merge_memories_access_count_summed");
    }

    #[test]
    fn test_merge_memories_empty_embeddings() {
        let service = ConsolidationService::new();

        let a = create_test_memory(vec![], "first", 0.8);
        let b = create_test_memory(vec![], "second", 0.7);

        let merged = service.merge_memories(&[a, b]);

        assert!(merged.embedding.is_empty());
        assert!(merged.text.contains("first"));
        assert!(merged.text.contains("second"));

        println!("[PASS] test_merge_memories_empty_embeddings");
    }

    // ========== compute_combined_alignment Tests ==========

    #[test]
    fn test_combined_alignment_empty() {
        let service = ConsolidationService::new();
        let result = service.compute_combined_alignment(&[]);
        assert!(result.abs() < f32::EPSILON);

        println!("[PASS] test_combined_alignment_empty");
    }

    #[test]
    fn test_combined_alignment_single() {
        let service = ConsolidationService::new();
        let result = service.compute_combined_alignment(&[0.85]);
        assert!((result - 0.85).abs() < f32::EPSILON);

        println!("[PASS] test_combined_alignment_single");
    }

    #[test]
    fn test_combined_alignment_equal() {
        let service = ConsolidationService::new();
        // Equal alignments should give that alignment
        let result = service.compute_combined_alignment(&[0.80, 0.80, 0.80]);
        assert!((result - 0.80).abs() < f32::EPSILON);

        println!("[PASS] test_combined_alignment_equal");
    }

    #[test]
    fn test_combined_alignment_favors_higher() {
        let service = ConsolidationService::new();

        // With [0.9, 0.5], weights are [0.81, 0.25]
        // weighted sum = 0.9*0.81 + 0.5*0.25 = 0.729 + 0.125 = 0.854
        // total weight = 1.06
        // result = 0.854 / 1.06 ≈ 0.806
        let result = service.compute_combined_alignment(&[0.9, 0.5]);

        // Result should be closer to 0.9 than 0.5
        assert!(
            result > 0.7,
            "Result should favor higher alignment: {}",
            result
        );
        assert!(result < 0.9, "Result should not exceed highest: {}", result);

        println!("[PASS] test_combined_alignment_favors_higher");
    }

    #[test]
    fn test_combined_alignment_zero_weights() {
        let service = ConsolidationService::new();
        // All zeros should use simple average = 0.0
        let result = service.compute_combined_alignment(&[0.0, 0.0]);
        assert!(result.abs() < f32::EPSILON);

        println!("[PASS] test_combined_alignment_zero_weights");
    }

    #[test]
    fn test_combined_alignment_real_calculation() {
        let service = ConsolidationService::new();

        // Manual calculation: [0.8, 0.7, 0.6]
        // weights = [0.64, 0.49, 0.36] sum = 1.49
        // weighted = 0.8*0.64 + 0.7*0.49 + 0.6*0.36 = 0.512 + 0.343 + 0.216 = 1.071
        // result = 1.071 / 1.49 ≈ 0.7188

        let result = service.compute_combined_alignment(&[0.8, 0.7, 0.6]);
        assert!(
            (result - 0.7188).abs() < 0.01,
            "Expected ~0.7188, got {}",
            result
        );

        println!("[PASS] test_combined_alignment_real_calculation");
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_full_consolidation_workflow() {
        let mut service = ConsolidationService::new();

        // Create similar memories
        let embedding = vec![0.6, 0.8, 0.0];
        let mem1 =
            create_test_memory(embedding.clone(), "The quick brown fox", 0.85).with_access_count(5);
        let mem2 =
            create_test_memory(embedding.clone(), "The fast brown fox", 0.83).with_access_count(3);

        // Create dissimilar memory
        let mem3 = create_test_memory(vec![0.0, 0.0, 1.0], "Something unrelated", 0.90)
            .with_access_count(10);

        // Build pairs
        let pairs = vec![
            MemoryPair::new(mem1.clone(), mem2.clone()),
            MemoryPair::new(mem1.clone(), mem3.clone()),
        ];

        // Find candidates
        let candidates = service.find_consolidation_candidates(&pairs);
        assert_eq!(candidates.len(), 1, "Should find exactly one similar pair");

        // Verify candidate properties
        let candidate = &candidates[0];
        assert!(candidate.similarity > 0.99);
        assert!(candidate.combined_alignment > 0.83);

        // Consolidate
        let report = service.consolidate(&candidates);
        assert_eq!(report.merged, 1);
        assert_eq!(report.skipped, 0);

        // Verify merge result
        let merged = service.merge_memories(&[mem1.clone(), mem2.clone()]);
        assert!(merged.text.contains("quick"));
        assert!(merged.text.contains("fast"));
        assert_eq!(merged.access_count, 8);

        println!("[PASS] test_full_consolidation_workflow");
    }

    #[test]
    fn test_batch_consolidation() {
        let config = ConsolidationConfig {
            enabled: true,
            similarity_threshold: 0.90,
            max_daily_merges: 100,
            theta_diff_threshold: 0.10,
        };
        let mut service = ConsolidationService::with_config(config);

        // Create multiple similar pairs
        let mut pairs = Vec::new();
        for i in 0..10 {
            let embedding = vec![1.0, 0.0, 0.0];
            let a = create_test_memory(embedding.clone(), &format!("Memory A{}", i), 0.80);
            let b = create_test_memory(embedding, &format!("Memory B{}", i), 0.78);
            pairs.push(MemoryPair::new(a, b));
        }

        let candidates = service.find_consolidation_candidates(&pairs);
        assert_eq!(candidates.len(), 10);

        let report = service.consolidate(&candidates);
        assert_eq!(report.merged, 10);
        assert!(!report.daily_limit_reached);

        println!("[PASS] test_batch_consolidation");
    }

    #[test]
    fn test_edge_case_all_same_alignment() {
        let service = ConsolidationService::new();

        let embedding = vec![1.0, 0.0];
        let mems: Vec<MemoryContent> = (0..5)
            .map(|i| create_test_memory(embedding.clone(), &format!("mem{}", i), 0.75))
            .collect();

        let result = service
            .compute_combined_alignment(&mems.iter().map(|m| m.alignment).collect::<Vec<_>>());
        assert!((result - 0.75).abs() < f32::EPSILON);

        println!("[PASS] test_edge_case_all_same_alignment");
    }
}
