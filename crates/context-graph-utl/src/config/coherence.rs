//! Coherence (delta-C) computation settings.
//!
//! Controls how coherence/understanding is computed for knowledge items.
//! Coherence measures how well information fits with existing knowledge.

use serde::{Deserialize, Serialize};

/// Coherence (delta-C) computation settings.
///
/// Controls how coherence/understanding is computed for knowledge items.
/// Coherence measures how well information fits with existing knowledge.
///
/// # Constitution Reference
///
/// - `delta-C` range: `[0, 1]` representing coherence/understanding
/// - Higher values indicate better integration with existing knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Weight for semantic similarity contribution.
    /// Range: `[0.0, 1.0]`
    pub similarity_weight: f32,

    /// Weight for consistency with existing knowledge.
    /// Range: `[0.0, 1.0]`
    pub consistency_weight: f32,

    /// Minimum coherence threshold.
    /// Range: `[0.0, 0.5]`
    pub min_threshold: f32,

    /// Decay rate for coherence over time without reinforcement.
    /// Range: `[0.0, 1.0]`
    pub decay_rate: f32,

    /// Number of neighbors to consider for coherence calculation.
    pub neighbor_count: usize,

    /// Minimum similarity for neighbor consideration.
    /// Range: `[0.0, 1.0]`
    pub neighbor_similarity_min: f32,

    /// Enable graph-based coherence computation.
    pub use_graph_coherence: bool,

    /// Weight for graph-based coherence.
    /// Range: `[0.0, 1.0]`
    pub graph_weight: f32,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 0.5,
            consistency_weight: 0.7,
            min_threshold: 0.1,
            decay_rate: 0.05,
            neighbor_count: 10,
            neighbor_similarity_min: 0.3,
            use_graph_coherence: true,
            graph_weight: 0.4,
        }
    }
}

impl CoherenceConfig {
    /// Validate the coherence configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.similarity_weight) {
            return Err(format!(
                "similarity_weight must be in [0, 1], got {}",
                self.similarity_weight
            ));
        }
        if !(0.0..=1.0).contains(&self.consistency_weight) {
            return Err(format!(
                "consistency_weight must be in [0, 1], got {}",
                self.consistency_weight
            ));
        }
        if !(0.0..=0.5).contains(&self.min_threshold) {
            return Err(format!(
                "min_threshold must be in [0, 0.5], got {}",
                self.min_threshold
            ));
        }
        if !(0.0..=1.0).contains(&self.decay_rate) {
            return Err(format!(
                "decay_rate must be in [0, 1], got {}",
                self.decay_rate
            ));
        }
        if self.neighbor_count == 0 {
            return Err("neighbor_count must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.neighbor_similarity_min) {
            return Err(format!(
                "neighbor_similarity_min must be in [0, 1], got {}",
                self.neighbor_similarity_min
            ));
        }
        if !(0.0..=1.0).contains(&self.graph_weight) {
            return Err(format!(
                "graph_weight must be in [0, 1], got {}",
                self.graph_weight
            ));
        }
        Ok(())
    }
}
