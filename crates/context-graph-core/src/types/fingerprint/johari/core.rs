//! Core JohariFingerprint struct definition and constructors.
//!
//! This module contains the main `JohariFingerprint` struct along with
//! its constructors and associated constants.

use serde::{Deserialize, Serialize};

use super::NUM_EMBEDDERS;

/// Per-embedder Johari Window classification with soft weights.
///
/// Unlike the simple `JohariQuadrant` enum, this provides:
/// - 4 weights per embedder (sum to 1.0) for soft classification
/// - Confidence score per embedder
/// - Transition probability matrix for evolution prediction
/// - Cross-space analysis methods
///
/// # Invariants
/// - All `quadrants[i]` arrays MUST sum to 1.0 (enforced by `set_quadrant`)
/// - All `confidence[i]` values MUST be in [0.0, 1.0]
/// - All `transition_probs[i][j]` rows MUST sum to 1.0
///
/// # Memory Layout
/// - quadrants: 13 x 4 x 4 bytes = 208 bytes
/// - confidence: 13 x 4 bytes = 52 bytes
/// - transition_probs: 13 x 4 x 4 x 4 bytes = 832 bytes
/// - Total: ~1092 bytes per JohariFingerprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohariFingerprint {
    /// Soft quadrant weights per embedder: [Open, Hidden, Blind, Unknown]
    /// Each inner array MUST sum to 1.0 (enforced by set_quadrant)
    /// Index 0-12 maps to E1-E13
    pub quadrants: [[f32; 4]; NUM_EMBEDDERS],

    /// Confidence of classification per embedder [0.0, 1.0]
    /// Low confidence = classification is uncertain
    pub confidence: [f32; NUM_EMBEDDERS],

    /// Transition probability matrix per embedder
    /// `transition_probs[embedder][from_quadrant][to_quadrant]`
    /// Each row (from_quadrant) MUST sum to 1.0
    pub transition_probs: [[[f32; 4]; 4]; NUM_EMBEDDERS],
}

impl JohariFingerprint {
    /// Entropy threshold for Johari classification (from constitution.yaml line 192)
    pub const ENTROPY_THRESHOLD: f32 = 0.5;

    /// Coherence threshold for Johari classification (from constitution.yaml line 193)
    pub const COHERENCE_THRESHOLD: f32 = 0.5;

    /// Quadrant index mapping (matches JohariQuadrant enum order)
    pub const OPEN_IDX: usize = 0;
    pub const HIDDEN_IDX: usize = 1;
    pub const BLIND_IDX: usize = 2;
    pub const UNKNOWN_IDX: usize = 3;

    /// Create with all zeros for quadrants/confidence and uniform transition priors (0.25 each).
    ///
    /// This is the recommended starting point for new fingerprints.
    /// Use `set_quadrant()` to populate with actual classifications.
    ///
    /// # Returns
    /// A `JohariFingerprint` with:
    /// - All quadrant weights set to [0.0, 0.0, 0.0, 0.0]
    /// - All confidence values set to 0.0
    /// - All transition probabilities set to uniform (0.25)
    pub fn zeroed() -> Self {
        // Uniform transition probabilities: 0.25 to each quadrant
        let uniform_transitions = [[0.25f32; 4]; 4];

        Self {
            quadrants: [[0.0f32; 4]; NUM_EMBEDDERS],
            confidence: [0.0f32; NUM_EMBEDDERS],
            transition_probs: [uniform_transitions; NUM_EMBEDDERS],
        }
    }

    /// Create stub with all Unknown dominant (backwards compat during migration).
    ///
    /// **DEPRECATED**: Use `zeroed()` for new code.
    ///
    /// Sets all embedders to 100% Unknown weight with full confidence.
    /// This matches the old stub behavior for backwards compatibility.
    #[deprecated(since = "2.0.0", note = "Use zeroed() for new code")]
    pub fn stub() -> Self {
        let mut fp = Self::zeroed();
        for embedder_idx in 0..NUM_EMBEDDERS {
            // Set 100% Unknown weight, 100% confidence
            fp.quadrants[embedder_idx] = [0.0, 0.0, 0.0, 1.0];
            fp.confidence[embedder_idx] = 1.0;
        }
        fp
    }
}
