//! Trait implementations for JohariFingerprint.
//!
//! This module provides Default and PartialEq implementations.

use super::core::JohariFingerprint;
use super::NUM_EMBEDDERS;

impl Default for JohariFingerprint {
    /// Default to zeroed (not stub) for new code
    fn default() -> Self {
        Self::zeroed()
    }
}

impl PartialEq for JohariFingerprint {
    /// Compare all fields with f32 epsilon tolerance
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f32 = 1e-6;

        // Compare quadrants
        for i in 0..NUM_EMBEDDERS {
            for j in 0..4 {
                if (self.quadrants[i][j] - other.quadrants[i][j]).abs() > EPSILON {
                    return false;
                }
            }
        }

        // Compare confidence
        for i in 0..NUM_EMBEDDERS {
            if (self.confidence[i] - other.confidence[i]).abs() > EPSILON {
                return false;
            }
        }

        // Compare transition_probs
        for i in 0..NUM_EMBEDDERS {
            for j in 0..4 {
                for k in 0..4 {
                    if (self.transition_probs[i][j][k] - other.transition_probs[i][j][k]).abs()
                        > EPSILON
                    {
                        return false;
                    }
                }
            }
        }

        true
    }
}
