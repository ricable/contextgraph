//! JohariFingerprint: Per-embedder soft classification with transition probabilities.
//!
//! **STATUS: TASK-F003 COMPLETE - Full implementation**
//!
//! This module provides per-embedder Johari Window classification with:
//! - Soft classification: 4 weights per embedder (sum to 1.0)
//! - Confidence scores per embedder
//! - Transition probability matrix for evolution prediction
//! - Cross-space analysis methods (find_blind_spots)
//!
//! From constitution.yaml (lines 184-194), Johari quadrants map to UTL states:
//! - **Open**: delta S < 0.5, delta C > 0.5 -> Known to self AND others (direct recall)
//! - **Hidden**: delta S < 0.5, delta C < 0.5 -> Known to self, NOT others (private)
//! - **Blind**: delta S > 0.5, delta C < 0.5 -> NOT known to self, known to others (discovery)
//! - **Unknown**: delta S > 0.5, delta C > 0.5 -> NOT known to self OR others (frontier)
//!
//! Cross-space capability (constitution.yaml line 81):
//! > "Memory can be Open(semantic/E1) but Blind(causal/E5)"

// Re-export NUM_EMBEDDERS for submodules
pub(crate) use super::purpose::NUM_EMBEDDERS;

// Submodules
mod analysis;
mod classification;
mod core;
mod impls;
mod serialization;

#[cfg(test)]
mod tests;

// Re-export the main type
pub use self::core::JohariFingerprint;
