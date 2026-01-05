//! Serialization and validation methods for JohariFingerprint.
//!
//! This module provides compact byte encoding/decoding and validation.

use super::core::JohariFingerprint;
use super::NUM_EMBEDDERS;

impl JohariFingerprint {
    /// Encode dominant quadrants as compact bytes.
    ///
    /// 2 bits per quadrant = 26 bits for 13 embedders.
    /// Fits in 4 bytes with 6 bits unused.
    ///
    /// Encoding: 00=Open, 01=Hidden, 10=Blind, 11=Unknown
    ///
    /// # Byte Layout
    /// - byte[0]: E1[1:0], E2[3:2], E3[5:4], E4[7:6]
    /// - byte[1]: E5[1:0], E6[3:2], E7[5:4], E8[7:6]
    /// - byte[2]: E9[1:0], E10[3:2], E11[5:4], E12[7:6]
    /// - byte[3]: E13[1:0], unused[7:2]
    pub fn to_compact_bytes(&self) -> [u8; 4] {
        let mut bytes = [0u8; 4];

        for embedder_idx in 0..NUM_EMBEDDERS {
            let quadrant = self.dominant_quadrant(embedder_idx);
            let bits = Self::quadrant_to_idx(quadrant) as u8;

            let byte_idx = embedder_idx / 4;
            let bit_offset = (embedder_idx % 4) * 2;

            bytes[byte_idx] |= bits << bit_offset;
        }

        bytes
    }

    /// Decode from compact bytes to JohariFingerprint.
    ///
    /// Sets dominant quadrant to 1.0 weight, others to 0.0.
    /// Sets confidence to 1.0 (hard classification from compact).
    ///
    /// # Arguments
    /// * `bytes` - 4-byte compact representation
    ///
    /// # Returns
    /// A `JohariFingerprint` with hard classifications derived from the compact encoding.
    pub fn from_compact_bytes(bytes: [u8; 4]) -> Self {
        let mut fp = Self::zeroed();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let byte_idx = embedder_idx / 4;
            let bit_offset = (embedder_idx % 4) * 2;

            let bits = (bytes[byte_idx] >> bit_offset) & 0b11;
            let quadrant_idx = bits as usize;

            // Set 100% weight for the decoded quadrant
            fp.quadrants[embedder_idx] = [0.0; 4];
            fp.quadrants[embedder_idx][quadrant_idx] = 1.0;
            fp.confidence[embedder_idx] = 1.0; // Hard classification
        }

        fp
    }

    /// Validate all invariants. Returns Err with description if invalid.
    ///
    /// Checks:
    /// - All quadrant weight rows sum to 1.0 (+/-0.001 tolerance)
    /// - All confidence values in [0.0, 1.0]
    /// - All transition probability rows sum to 1.0 (+/-0.001 tolerance)
    /// - No NaN or infinite values
    ///
    /// # Returns
    /// - `Ok(())` if all invariants hold
    /// - `Err(String)` with detailed error message if any invariant is violated
    pub fn validate(&self) -> Result<(), String> {
        const TOLERANCE: f32 = 0.001;

        // Check quadrant weights
        for (embedder_idx, weights) in self.quadrants.iter().enumerate() {
            // Check for NaN/Inf
            for (quad_idx, &weight) in weights.iter().enumerate() {
                if weight.is_nan() {
                    return Err(format!(
                        "JohariFingerprint validation failed: quadrants[{}][{}] is NaN",
                        embedder_idx, quad_idx
                    ));
                }
                if weight.is_infinite() {
                    return Err(format!(
                        "JohariFingerprint validation failed: quadrants[{}][{}] is infinite",
                        embedder_idx, quad_idx
                    ));
                }
                if weight < 0.0 {
                    return Err(format!(
                        "JohariFingerprint validation failed: quadrants[{}][{}] is negative ({})",
                        embedder_idx, quad_idx, weight
                    ));
                }
            }

            let sum: f32 = weights.iter().sum();
            // Allow either all-zeros (not yet set) or normalized sum
            if sum > f32::EPSILON && (sum - 1.0).abs() > TOLERANCE {
                return Err(format!(
                    "JohariFingerprint validation failed at embedder {}: quadrant weights sum to {} (expected 1.0)",
                    embedder_idx, sum
                ));
            }
        }

        // Check confidence values
        for (embedder_idx, &conf) in self.confidence.iter().enumerate() {
            if conf.is_nan() {
                return Err(format!(
                    "JohariFingerprint validation failed: confidence[{}] is NaN",
                    embedder_idx
                ));
            }
            if conf.is_infinite() {
                return Err(format!(
                    "JohariFingerprint validation failed: confidence[{}] is infinite",
                    embedder_idx
                ));
            }
            if !(0.0..=1.0).contains(&conf) {
                return Err(format!(
                    "JohariFingerprint validation failed: confidence[{}] = {} not in [0.0, 1.0]",
                    embedder_idx, conf
                ));
            }
        }

        // Check transition probability matrices
        for (embedder_idx, matrix) in self.transition_probs.iter().enumerate() {
            for (from_idx, row) in matrix.iter().enumerate() {
                // Check for NaN/Inf
                for (to_idx, &prob) in row.iter().enumerate() {
                    if prob.is_nan() {
                        return Err(format!(
                            "JohariFingerprint validation failed: transition_probs[{}][{}][{}] is NaN",
                            embedder_idx, from_idx, to_idx
                        ));
                    }
                    if prob.is_infinite() {
                        return Err(format!(
                            "JohariFingerprint validation failed: transition_probs[{}][{}][{}] is infinite",
                            embedder_idx, from_idx, to_idx
                        ));
                    }
                    if prob < 0.0 {
                        return Err(format!(
                            "JohariFingerprint validation failed: transition_probs[{}][{}][{}] is negative ({})",
                            embedder_idx, from_idx, to_idx, prob
                        ));
                    }
                }

                let sum: f32 = row.iter().sum();
                if (sum - 1.0).abs() > TOLERANCE {
                    return Err(format!(
                        "JohariFingerprint validation failed: transition_probs[{}][{}] sums to {} (expected 1.0)",
                        embedder_idx, from_idx, sum
                    ));
                }
            }
        }

        Ok(())
    }
}
