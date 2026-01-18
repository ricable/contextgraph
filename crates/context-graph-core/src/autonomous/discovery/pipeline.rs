//! Goal discovery pipeline for autonomous goal emergence.
//!
//! This module implements the main GoalDiscoveryPipeline that orchestrates
//! clustering and goal scoring to discover emergent goals from teleological arrays.

use uuid::Uuid;

use crate::autonomous::evolution::GoalLevel;
use crate::teleological::comparator::TeleologicalComparator;
use crate::teleological::{Embedder, NUM_EMBEDDERS};
use crate::types::fingerprint::TeleologicalArray;

use super::centroid::{compute_centroid, l2_norm};
use super::clustering::cluster_arrays;
use super::types::{
    Cluster, DiscoveredGoal, DiscoveryConfig, DiscoveryResult, GoalCandidate, GoalRelationship,
};

/// Goal discovery pipeline for autonomous goal emergence.
///
/// Uses K-means clustering on TeleologicalArrays to discover emergent goals.
/// Compares arrays using TeleologicalComparator (apples-to-apples per embedder).
pub struct GoalDiscoveryPipeline {
    comparator: TeleologicalComparator,
}

impl GoalDiscoveryPipeline {
    /// Create a new GoalDiscoveryPipeline with default comparator.
    pub fn new(comparator: TeleologicalComparator) -> Self {
        Self { comparator }
    }

    /// Discover goals from teleological arrays.
    ///
    /// FAILS FAST on any error - no recovery attempts.
    ///
    /// # Panics
    ///
    /// - If input arrays is empty
    /// - If input is smaller than min_cluster_size
    /// - If clustering fails
    /// - If no valid clusters found
    pub fn discover(
        &self,
        arrays: &[TeleologicalArray],
        config: &DiscoveryConfig,
    ) -> DiscoveryResult {
        // FAIL FAST: Check minimum data requirements
        assert!(
            !arrays.is_empty(),
            "FAIL FAST: Insufficient arrays for goal discovery. Got 0 arrays, need at least {}",
            config.min_cluster_size
        );

        assert!(
            arrays.len() >= config.min_cluster_size,
            "FAIL FAST: Insufficient arrays for clustering. Got {} arrays, minimum required: {}",
            arrays.len(),
            config.min_cluster_size
        );

        // Sample if needed
        let sampled_arrays: Vec<&TeleologicalArray> = if arrays.len() > config.sample_size {
            // Simple deterministic sampling using stride
            let stride = arrays.len() / config.sample_size;
            arrays
                .iter()
                .step_by(stride)
                .take(config.sample_size)
                .collect()
        } else {
            arrays.iter().collect()
        };

        eprintln!(
            "[GoalDiscoveryPipeline] Analyzing {} arrays (sampled from {})",
            sampled_arrays.len(),
            arrays.len()
        );

        // Perform clustering
        let clusters = cluster_arrays(&sampled_arrays, config, &self.comparator);

        // FAIL FAST: Verify clusters found
        assert!(
            !clusters.is_empty(),
            "FAIL FAST: No clusters found with min_cluster_size={} and min_coherence={}",
            config.min_cluster_size,
            config.min_coherence
        );

        eprintln!("[GoalDiscoveryPipeline] Found {} clusters", clusters.len());

        // Score clusters and create goal candidates
        let mut candidates: Vec<GoalCandidate> = clusters
            .iter()
            .filter(|c| c.members.len() >= config.min_cluster_size)
            .filter(|c| c.coherence >= config.min_coherence)
            .map(|c| self.score_cluster(c))
            .collect();

        // Assign goal levels
        for candidate in &mut candidates {
            candidate.level = self.assign_level(candidate);
        }

        // Build hierarchy
        let hierarchy = self.build_hierarchy(&candidates);

        // candidates is Vec<GoalCandidate> which is Vec<DiscoveredGoal>
        let discovered_goals: Vec<DiscoveredGoal> = candidates;

        eprintln!(
            "[GoalDiscoveryPipeline] Discovered {} goals with {} hierarchy relationships",
            discovered_goals.len(),
            hierarchy.len()
        );

        DiscoveryResult {
            clusters_found: discovered_goals.len(),
            total_arrays_analyzed: sampled_arrays.len(),
            discovered_goals,
            hierarchy,
        }
    }

    /// Compute centroid for a cluster (delegates to centroid module).
    pub fn compute_centroid(&self, members: &[&TeleologicalArray]) -> TeleologicalArray {
        compute_centroid(members)
    }

    /// Score cluster suitability as a goal.
    fn score_cluster(&self, cluster: &Cluster) -> GoalCandidate {
        let size_score = (cluster.members.len() as f32 / 50.0).min(1.0);
        let coherence_score = cluster.coherence;

        // Find dominant embedders
        let dominant_embedders = self.find_dominant_embedders(&cluster.centroid);
        let embedder_diversity = (dominant_embedders.len() as f32) / 3.0;

        // Combined confidence: 40% coherence, 30% size, 30% embedder distribution
        let confidence = 0.4 * coherence_score + 0.3 * size_score + 0.3 * embedder_diversity;

        GoalCandidate {
            goal_id: Uuid::new_v4().to_string(),
            description: format!(
                "Goal cluster (size={}, coherence={:.2})",
                cluster.members.len(),
                cluster.coherence
            ),
            level: GoalLevel::Operational, // Will be reassigned
            confidence,
            member_count: cluster.members.len(),
            centroid: cluster.centroid.clone(),
            dominant_embedders,
            coherence_score: cluster.coherence,
        }
    }

    /// Find top 3 embedders by magnitude.
    fn find_dominant_embedders(&self, centroid: &TeleologicalArray) -> Vec<Embedder> {
        let mut embedder_magnitudes: Vec<(Embedder, f32)> = Vec::with_capacity(NUM_EMBEDDERS);

        // Dense embedders
        embedder_magnitudes.push((Embedder::Semantic, l2_norm(&centroid.e1_semantic)));
        embedder_magnitudes.push((
            Embedder::TemporalRecent,
            l2_norm(&centroid.e2_temporal_recent),
        ));
        embedder_magnitudes.push((
            Embedder::TemporalPeriodic,
            l2_norm(&centroid.e3_temporal_periodic),
        ));
        embedder_magnitudes.push((
            Embedder::TemporalPositional,
            l2_norm(&centroid.e4_temporal_positional),
        ));
        embedder_magnitudes.push((Embedder::Causal, l2_norm(&centroid.e5_causal)));
        embedder_magnitudes.push((Embedder::Code, l2_norm(&centroid.e7_code)));
        embedder_magnitudes.push((Embedder::Emotional, l2_norm(&centroid.e8_graph)));
        embedder_magnitudes.push((Embedder::Hdc, l2_norm(&centroid.e9_hdc)));
        embedder_magnitudes.push((Embedder::Multimodal, l2_norm(&centroid.e10_multimodal)));
        embedder_magnitudes.push((Embedder::Entity, l2_norm(&centroid.e11_entity)));

        // Sparse embedders
        embedder_magnitudes.push((Embedder::Sparse, centroid.e6_sparse.l2_norm()));
        embedder_magnitudes.push((Embedder::KeywordSplade, centroid.e13_splade.l2_norm()));

        // Token-level embedder
        let e12_magnitude: f32 = centroid
            .e12_late_interaction
            .iter()
            .map(|t| l2_norm(t))
            .sum::<f32>()
            / centroid.e12_late_interaction.len().max(1) as f32;
        embedder_magnitudes.push((Embedder::LateInteraction, e12_magnitude));

        // Sort by magnitude descending and take top 3
        embedder_magnitudes
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        embedder_magnitudes
            .into_iter()
            .take(3)
            .map(|(e, _)| e)
            .collect()
    }

    /// Assign goal level based on size and coherence thresholds.
    ///
    /// Thresholds:
    /// - Strategic (top): size >= 50 AND coherence >= 0.85
    /// - Strategic: size >= 20 AND coherence >= 0.80
    /// - Tactical: size >= 10 AND coherence >= 0.75
    /// - Operational: everything else
    pub fn assign_level(&self, candidate: &GoalCandidate) -> GoalLevel {
        let size = candidate.member_count;
        let coherence = candidate.coherence_score;

        if size >= 50 && coherence >= 0.85 {
            GoalLevel::Strategic
        } else if size >= 20 && coherence >= 0.80 {
            GoalLevel::Tactical
        } else if size >= 10 && coherence >= 0.75 {
            GoalLevel::Tactical
        } else {
            GoalLevel::Operational
        }
    }

    /// Build parent-child relationships based on centroid similarity.
    ///
    /// Higher-level goals (larger, more coherent) become parents of
    /// lower-level goals with similar centroids.
    pub fn build_hierarchy(&self, candidates: &[GoalCandidate]) -> Vec<GoalRelationship> {
        let mut relationships = Vec::new();

        // Sort candidates by level (higher first) and then by size
        let mut sorted: Vec<(usize, &GoalCandidate)> = candidates.iter().enumerate().collect();
        sorted.sort_by(|a, b| {
            let cmp = Self::level_order(&a.1.level).cmp(&Self::level_order(&b.1.level));
            if cmp == std::cmp::Ordering::Equal {
                b.1.member_count.cmp(&a.1.member_count)
            } else {
                cmp
            }
        });

        // For each candidate, find the best parent (higher level, most similar)
        for i in 0..sorted.len() {
            let child = sorted[i].1;
            let child_level = Self::level_order(&child.level);

            let mut best_parent: Option<(&GoalCandidate, f32)> = None;

            for (_, parent) in sorted.iter().take(i) {
                let parent_level = Self::level_order(&parent.level);

                // Parent must be higher level
                if parent_level >= child_level {
                    continue;
                }

                // Compute similarity
                let result = self.comparator.compare(&parent.centroid, &child.centroid);
                let similarity = result.map(|r| r.overall).unwrap_or(0.0);

                // Threshold: at least 0.5 similarity to form relationship
                if similarity >= 0.5 {
                    if let Some((_, best_sim)) = best_parent {
                        if similarity > best_sim {
                            best_parent = Some((parent, similarity));
                        }
                    } else {
                        best_parent = Some((parent, similarity));
                    }
                }
            }

            if let Some((parent, similarity)) = best_parent {
                relationships.push(GoalRelationship {
                    parent_id: parent.goal_id.clone(),
                    child_id: child.goal_id.clone(),
                    similarity,
                });
            }
        }

        relationships
    }

    /// Convert level to ordering number.
    /// TASK-P0-001: Uses 3-level hierarchy, Strategic is now 0.
    fn level_order(level: &GoalLevel) -> u8 {
        match level {
            GoalLevel::Strategic => 0,
            GoalLevel::Tactical => 1,
            GoalLevel::Operational => 2,
        }
    }
}
