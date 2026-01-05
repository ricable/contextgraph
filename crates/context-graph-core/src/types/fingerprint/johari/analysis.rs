//! Analysis methods for JohariFingerprint.
//!
//! This module provides methods for analyzing cross-space patterns,
//! predicting transitions, and computing aggregate metrics.

use crate::types::JohariQuadrant;

use super::core::JohariFingerprint;
use super::NUM_EMBEDDERS;

impl JohariFingerprint {
    /// Find blind spots: cross-space gaps where one embedder understands but another doesn't.
    ///
    /// Specifically finds embedders with high Blind weight while E1 (semantic) has high Open weight.
    /// This indicates understanding at the semantic level but lack of awareness in other dimensions.
    ///
    /// # Returns
    /// Vector of `(embedder_idx, blind_severity)` pairs sorted by severity descending.
    /// Blind severity = Open[E1] x Blind[embedder]
    ///
    /// # Interpretation
    /// High severity means:
    /// - E1 (semantic) strongly classifies as Open (well understood)
    /// - The target embedder strongly classifies as Blind (discovery opportunity)
    /// - This is a "blind spot" - semantic understanding without dimensional insight
    pub fn find_blind_spots(&self) -> Vec<(usize, f32)> {
        let e1_open_weight = self.quadrants[0][Self::OPEN_IDX];

        let mut blind_spots: Vec<(usize, f32)> = (0..NUM_EMBEDDERS)
            .filter_map(|idx| {
                let blind_weight = self.quadrants[idx][Self::BLIND_IDX];
                let severity = e1_open_weight * blind_weight;

                if severity > f32::EPSILON {
                    Some((idx, severity))
                } else {
                    None
                }
            })
            .collect();

        // Sort by severity descending
        blind_spots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        blind_spots
    }

    /// Predict most likely next quadrant given current state.
    ///
    /// Uses the transition probability matrix to determine the most probable
    /// next quadrant from the current quadrant.
    ///
    /// # Arguments
    /// * `embedder_idx` - Index of embedder (0-12)
    /// * `current` - Current quadrant state
    ///
    /// # Returns
    /// The `JohariQuadrant` with highest transition probability from the current state.
    ///
    /// # Panics
    /// Panics if `embedder_idx >= NUM_EMBEDDERS` (13)
    pub fn predict_transition(
        &self,
        embedder_idx: usize,
        current: JohariQuadrant,
    ) -> JohariQuadrant {
        assert!(
            embedder_idx < NUM_EMBEDDERS,
            "embedder_idx {} out of bounds (max {})",
            embedder_idx,
            NUM_EMBEDDERS - 1
        );

        let from_idx = Self::quadrant_to_idx(current);
        let probs = &self.transition_probs[embedder_idx][from_idx];

        let mut max_idx = 0;
        let mut max_prob = probs[0];

        for (idx, &prob) in probs.iter().enumerate().skip(1) {
            if prob > max_prob {
                max_prob = prob;
                max_idx = idx;
            }
        }

        Self::idx_to_quadrant(max_idx)
    }

    /// Compute overall openness (fraction of embedders with Open dominant).
    ///
    /// # Returns
    /// Value in [0.0, 1.0] representing the fraction of embedders where Open is dominant.
    pub fn openness(&self) -> f32 {
        let open_count = (0..NUM_EMBEDDERS)
            .filter(|&idx| self.dominant_quadrant(idx) == JohariQuadrant::Open)
            .count();
        open_count as f32 / NUM_EMBEDDERS as f32
    }

    /// Check if overall awareness is healthy (majority Open/Hidden dominant).
    ///
    /// A memory is considered "aware" if more than half of its embedders
    /// are in self-aware quadrants (Open or Hidden).
    ///
    /// # Returns
    /// `true` if more than 50% of embedders have Open or Hidden dominant.
    pub fn is_aware(&self) -> bool {
        let aware_count = (0..NUM_EMBEDDERS)
            .filter(|&idx| {
                let dom = self.dominant_quadrant(idx);
                dom == JohariQuadrant::Open || dom == JohariQuadrant::Hidden
            })
            .count();

        aware_count as f32 / NUM_EMBEDDERS as f32 >= 0.5
    }

    /// Set transition probabilities for an embedder.
    ///
    /// Normalizes each row to sum to 1.0.
    ///
    /// # Arguments
    /// * `embedder_idx` - Index of embedder (0-12)
    /// * `matrix` - 4x4 transition probability matrix [from][to]
    ///
    /// # Panics
    /// Panics if `embedder_idx >= NUM_EMBEDDERS` (13)
    pub fn set_transition_probs(&mut self, embedder_idx: usize, matrix: [[f32; 4]; 4]) {
        assert!(
            embedder_idx < NUM_EMBEDDERS,
            "embedder_idx {} out of bounds (max {})",
            embedder_idx,
            NUM_EMBEDDERS - 1
        );

        for (from_idx, row) in matrix.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            if sum < f32::EPSILON {
                // All zero: use uniform
                self.transition_probs[embedder_idx][from_idx] = [0.25; 4];
            } else {
                for (to_idx, &val) in row.iter().enumerate() {
                    self.transition_probs[embedder_idx][from_idx][to_idx] = val / sum;
                }
            }
        }
    }
}
