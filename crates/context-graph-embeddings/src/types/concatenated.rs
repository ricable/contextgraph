//! Aggregated embedding from all 12 models for FuseMoE input.
//!
//! The `ConcatenatedEmbedding` struct collects individual `ModelEmbedding` outputs
//! and concatenates them into a single 8320-dimensional vector for FuseMoE processing.
//!
//! # Pipeline Position
//!
//! ```text
//! Individual Models (E1-E12)
//!          ↓
//!     ModelEmbedding (per model)
//!          ↓
//!     ConcatenatedEmbedding (this module) ← collects all 12
//!          ↓
//!     FuseMoE (8320D → 1536D)
//!          ↓
//!     FusedEmbedding (final output)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::types::{ConcatenatedEmbedding, ModelEmbedding, ModelId};
//!
//! let mut concat = ConcatenatedEmbedding::new();
//!
//! // Add embeddings from each model
//! for model_id in ModelId::all() {
//!     let dim = model_id.projected_dimension();
//!     let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
//!     emb.set_projected(true);
//!     concat.set(emb);
//! }
//!
//! // Now build the concatenated vector
//! concat.concatenate();
//! assert_eq!(concat.concatenated.len(), 8320);
//! ```

use crate::error::EmbeddingResult;
use crate::types::dimensions::{self, MODEL_COUNT, TOTAL_CONCATENATED};
use crate::types::{ModelEmbedding, ModelId};
use xxhash_rust::xxh64::xxh64;

/// Aggregates outputs from all 12 embedding models.
///
/// This struct collects individual `ModelEmbedding` outputs and concatenates
/// them into a single 8320D vector for FuseMoE input.
///
/// # Invariants
/// - `embeddings` is indexed by `ModelId as u8` (0-11)
/// - `is_complete()` returns true only when all 12 slots are filled
/// - `concatenated` vector is built in model order (E1-E12)
/// - `content_hash` is deterministic: same embeddings → same hash
///
/// # Fail-Fast Behavior
/// - `set()` panics if embedding dimension doesn't match projected dimension
/// - `concatenate()` panics if not all 12 slots are filled
/// - No fallbacks or workarounds - errors must be addressed
#[derive(Debug, Clone)]
pub struct ConcatenatedEmbedding {
    /// Individual model embeddings indexed by `ModelId as u8`.
    /// Array of 12 slots, each `Option<ModelEmbedding>`.
    pub embeddings: [Option<ModelEmbedding>; MODEL_COUNT],

    /// The concatenated 8320D vector (built by `concatenate()`).
    /// Empty until `concatenate()` is called.
    pub concatenated: Vec<f32>,

    /// Sum of all individual model latencies in microseconds.
    /// Updated incrementally as embeddings are set.
    pub total_latency_us: u64,

    /// xxHash64 of concatenated vector bytes for caching.
    /// Zero until `concatenate()` is called.
    pub content_hash: u64,
}

impl ConcatenatedEmbedding {
    /// Creates a new `ConcatenatedEmbedding` with all slots empty.
    ///
    /// # Returns
    /// A new instance with:
    /// - All 12 embedding slots set to `None`
    /// - Empty `concatenated` vector
    /// - `total_latency_us = 0`
    /// - `content_hash = 0`
    #[must_use]
    pub fn new() -> Self {
        Self {
            embeddings: std::array::from_fn(|_| None),
            concatenated: Vec::new(),
            total_latency_us: 0,
            content_hash: 0,
        }
    }

    /// Sets the embedding at the index matching `embedding.model_id`.
    ///
    /// Updates `total_latency_us` by adding the embedding's latency.
    /// If overwriting an existing embedding, the old latency is subtracted first.
    ///
    /// # Arguments
    /// * `embedding` - The model embedding to store. Must have `is_projected = true`
    ///   and vector length matching `model_id.projected_dimension()`.
    ///
    /// # Panics
    /// Panics if `embedding.vector.len() != embedding.model_id.projected_dimension()`.
    /// This is a fail-fast design - incorrect dimensions must be fixed, not worked around.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut concat = ConcatenatedEmbedding::new();
    /// let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 1000);
    /// emb.set_projected(true);
    /// concat.set(emb);
    /// assert_eq!(concat.filled_count(), 1);
    /// ```
    pub fn set(&mut self, embedding: ModelEmbedding) {
        let model_id = embedding.model_id;
        let expected_dim = model_id.projected_dimension();
        let actual_dim = embedding.vector.len();

        // Fail-fast: dimension must match projected dimension
        assert!(
            actual_dim == expected_dim,
            "Dimension mismatch for {:?}: expected {}, got {}. \
             Embeddings must be projected to projected_dimension() before concatenation.",
            model_id,
            expected_dim,
            actual_dim
        );

        let index = model_id as u8 as usize;

        // If overwriting, subtract old latency first
        if let Some(ref old_emb) = self.embeddings[index] {
            self.total_latency_us = self.total_latency_us.saturating_sub(old_emb.latency_us);
        }

        // Add new latency (with saturating add to handle u64::MAX)
        self.total_latency_us = self.total_latency_us.saturating_add(embedding.latency_us);

        self.embeddings[index] = Some(embedding);
    }

    /// Gets the embedding for the specified model, if present.
    ///
    /// # Arguments
    /// * `model_id` - The model to retrieve
    ///
    /// # Returns
    /// - `Some(&ModelEmbedding)` if the model's embedding has been set
    /// - `None` if not yet set
    #[inline]
    #[must_use]
    pub fn get(&self, model_id: ModelId) -> Option<&ModelEmbedding> {
        let index = model_id as u8 as usize;
        self.embeddings[index].as_ref()
    }

    /// Returns `true` only if all 12 slots are filled.
    ///
    /// This is a prerequisite for calling `concatenate()`.
    #[inline]
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.embeddings.iter().all(|e| e.is_some())
    }

    /// Returns the list of `ModelId` variants not yet set.
    ///
    /// Useful for error reporting and timeout handling.
    ///
    /// # Returns
    /// A vector of missing model IDs. Empty if `is_complete()` is true.
    #[must_use]
    pub fn missing_models(&self) -> Vec<ModelId> {
        ModelId::all()
            .iter()
            .copied()
            .filter(|model_id| self.embeddings[*model_id as u8 as usize].is_none())
            .collect()
    }

    /// Returns the count of filled slots (0-12).
    #[inline]
    #[must_use]
    pub fn filled_count(&self) -> usize {
        self.embeddings.iter().filter(|e| e.is_some()).count()
    }

    /// Builds the concatenated vector from embeddings in model order (E1-E12).
    ///
    /// After this call:
    /// - `self.concatenated.len() == 8320`
    /// - `self.content_hash` is computed via xxHash64
    ///
    /// # Panics
    /// Panics if `is_complete() == false`. All 12 models must be present.
    /// This is fail-fast by design - partial concatenation is not supported.
    ///
    /// # Performance
    /// - Preallocates the full 8320-element vector
    /// - Uses `extend_from_slice` for efficient copying
    /// - Hash computation is O(n) on the final vector
    pub fn concatenate(&mut self) {
        assert!(
            self.is_complete(),
            "Cannot concatenate: {} of 12 models missing. Missing: {:?}",
            MODEL_COUNT - self.filled_count(),
            self.missing_models()
        );

        // Preallocate exact size
        self.concatenated = Vec::with_capacity(TOTAL_CONCATENATED);

        // Concatenate in E1-E12 order
        for model_id in ModelId::all() {
            let embedding = self.embeddings[*model_id as u8 as usize]
                .as_ref()
                .expect("Embedding should exist after is_complete() check");
            self.concatenated.extend_from_slice(&embedding.vector);
        }

        // Compute content hash for caching
        self.content_hash = Self::compute_hash(&self.concatenated);
    }

    /// Returns the total dimension of the concatenated vector.
    ///
    /// Returns `TOTAL_CONCATENATED` (8320) when complete, or the sum of
    /// filled embedding dimensions when incomplete.
    #[must_use]
    pub fn total_dimension(&self) -> usize {
        if !self.concatenated.is_empty() {
            return self.concatenated.len();
        }

        // Sum dimensions of filled embeddings
        self.embeddings
            .iter()
            .filter_map(|e| e.as_ref())
            .map(|e| e.vector.len())
            .sum()
    }

    /// Validates all embeddings against their model requirements.
    ///
    /// Calls `validate()` on each present embedding.
    ///
    /// # Errors
    /// Returns the first validation error encountered.
    /// Does not validate missing embeddings (use `is_complete()` for that).
    pub fn validate(&self) -> EmbeddingResult<()> {
        for embedding in self.embeddings.iter().flatten() {
            embedding.validate()?;
        }
        Ok(())
    }

    /// Computes xxHash64 of concatenated vector bytes.
    ///
    /// This hash is deterministic: same embeddings → same hash.
    /// Used for caching and deduplication.
    ///
    /// # Arguments
    /// * `data` - The f32 slice to hash
    ///
    /// # Returns
    /// The 64-bit hash value
    fn compute_hash(data: &[f32]) -> u64 {
        // Convert f32 slice to bytes for hashing
        // SAFETY: f32 and [u8; 4] have the same size and alignment requirements
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
            )
        };

        // Use seed 0 for deterministic hashing
        xxh64(bytes, 0)
    }

    /// Returns slice of the concatenated vector at the given model's offset.
    ///
    /// Useful for extracting individual model contributions after concatenation.
    ///
    /// # Arguments
    /// * `model_id` - The model whose slice to extract
    ///
    /// # Returns
    /// - `Some(&[f32])` if concatenated vector exists
    /// - `None` if `concatenate()` hasn't been called
    ///
    /// # Panics
    /// Panics if the concatenated vector is malformed (slice bounds error).
    #[must_use]
    pub fn get_slice(&self, model_id: ModelId) -> Option<&[f32]> {
        if self.concatenated.is_empty() {
            return None;
        }

        let index = model_id as u8 as usize;
        let offset = dimensions::offset_by_index(index);
        let dim = dimensions::projected_dimension_by_index(index);

        Some(&self.concatenated[offset..offset + dim])
    }
}

impl Default for ConcatenatedEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EmbeddingError;

    // ========== Construction Tests ==========

    #[test]
    fn test_new_creates_empty_struct() {
        let ce = ConcatenatedEmbedding::new();

        assert!(ce.embeddings.iter().all(|e| e.is_none()));
        assert!(ce.concatenated.is_empty());
        assert_eq!(ce.total_latency_us, 0);
        assert_eq!(ce.content_hash, 0);
        assert!(!ce.is_complete());
        assert_eq!(ce.filled_count(), 0);
    }

    #[test]
    fn test_default_equals_new() {
        let ce1 = ConcatenatedEmbedding::new();
        let ce2 = ConcatenatedEmbedding::default();

        assert_eq!(ce1.total_latency_us, ce2.total_latency_us);
        assert_eq!(ce1.content_hash, ce2.content_hash);
        assert_eq!(ce1.filled_count(), ce2.filled_count());
    }

    // ========== Set Tests ==========

    #[test]
    fn test_set_places_at_correct_index() {
        let mut ce = ConcatenatedEmbedding::new();
        let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 1000);
        emb.set_projected(true);
        ce.set(emb);

        assert!(ce.embeddings[0].is_some()); // Semantic = 0
        assert_eq!(ce.filled_count(), 1);
        assert_eq!(ce.total_latency_us, 1000);
    }

    #[test]
    fn test_set_all_models() {
        let mut ce = ConcatenatedEmbedding::new();

        for model_id in ModelId::all() {
            let dim = model_id.projected_dimension();
            let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
            emb.set_projected(true);
            ce.set(emb);
        }

        assert!(ce.is_complete());
        assert_eq!(ce.filled_count(), 12);
        assert_eq!(ce.total_latency_us, 1200); // 12 * 100
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_set_wrong_dimension_panics() {
        let mut ce = ConcatenatedEmbedding::new();
        // Semantic requires 1024, but we provide 512
        let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 512], 1000);
        emb.set_projected(true);
        ce.set(emb); // Should panic
    }

    // ========== Get Tests ==========

    #[test]
    fn test_get_returns_correct_embedding() {
        let mut ce = ConcatenatedEmbedding::new();
        let mut emb = ModelEmbedding::new(ModelId::Causal, vec![0.1; 768], 500);
        emb.set_projected(true);
        ce.set(emb);

        let got = ce.get(ModelId::Causal);
        assert!(got.is_some());
        assert_eq!(got.unwrap().model_id, ModelId::Causal);
    }

    #[test]
    fn test_get_returns_none_for_missing() {
        let ce = ConcatenatedEmbedding::new();
        assert!(ce.get(ModelId::Semantic).is_none());
    }

    // ========== Completion Tests ==========

    #[test]
    fn test_is_complete_only_when_all_12() {
        let mut ce = ConcatenatedEmbedding::new();

        // Fill 11 models
        for model_id in ModelId::all().iter().take(11) {
            let dim = model_id.projected_dimension();
            let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
            emb.set_projected(true);
            ce.set(emb);
        }
        assert!(!ce.is_complete());
        assert_eq!(ce.filled_count(), 11);

        // Fill last model
        let mut emb = ModelEmbedding::new(ModelId::LateInteraction, vec![0.1; 128], 100);
        emb.set_projected(true);
        ce.set(emb);
        assert!(ce.is_complete());
        assert_eq!(ce.filled_count(), 12);
    }

    #[test]
    fn test_missing_models_returns_correct_list() {
        let mut ce = ConcatenatedEmbedding::new();
        let missing = ce.missing_models();
        assert_eq!(missing.len(), 12);

        // Set one model
        let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 100);
        emb.set_projected(true);
        ce.set(emb);

        let missing = ce.missing_models();
        assert_eq!(missing.len(), 11);
        assert!(!missing.contains(&ModelId::Semantic));
    }

    // ========== Concatenation Tests ==========

    #[test]
    fn test_concatenate_produces_8320_vector() {
        let mut ce = create_complete_embedding();
        ce.concatenate();

        assert_eq!(ce.concatenated.len(), dimensions::TOTAL_CONCATENATED);
        assert_eq!(ce.concatenated.len(), 8320);
    }

    #[test]
    fn test_concatenate_order_matches_model_order() {
        let mut ce = ConcatenatedEmbedding::new();

        // Set each model with unique value equal to its index
        for (i, model_id) in ModelId::all().iter().enumerate() {
            let dim = model_id.projected_dimension();
            let mut emb = ModelEmbedding::new(*model_id, vec![i as f32; dim], 100);
            emb.set_projected(true);
            ce.set(emb);
        }

        ce.concatenate();

        // Verify order: first 1024 elements should be 0.0 (Semantic)
        assert_eq!(ce.concatenated[0], 0.0);
        // Next 512 should be 1.0 (TemporalRecent)
        assert_eq!(ce.concatenated[1024], 1.0);
        // Last 128 should be 11.0 (LateInteraction)
        assert_eq!(ce.concatenated[8320 - 1], 11.0);
    }

    #[test]
    fn test_content_hash_deterministic() {
        let mut ce1 = create_complete_embedding();
        let mut ce2 = create_complete_embedding();

        ce1.concatenate();
        ce2.concatenate();

        assert_eq!(ce1.content_hash, ce2.content_hash);
        assert_ne!(ce1.content_hash, 0);
    }

    #[test]
    fn test_content_hash_differs_for_different_data() {
        let mut ce1 = create_complete_embedding();
        ce1.concatenate();
        let hash1 = ce1.content_hash;

        // Create another with different values
        let mut ce2 = ConcatenatedEmbedding::new();
        for (i, model_id) in ModelId::all().iter().enumerate() {
            let dim = model_id.projected_dimension();
            let mut emb = ModelEmbedding::new(*model_id, vec![(i as f32) * 0.1; dim], 100);
            emb.set_projected(true);
            ce2.set(emb);
        }
        ce2.concatenate();
        let hash2 = ce2.content_hash;

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_total_latency_sums_all() {
        let mut ce = ConcatenatedEmbedding::new();

        for model_id in ModelId::all() {
            let dim = model_id.projected_dimension();
            let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
            emb.set_projected(true);
            ce.set(emb);
        }

        assert_eq!(ce.total_latency_us, 1200); // 12 * 100
    }

    #[test]
    #[should_panic(expected = "Cannot concatenate")]
    fn test_concatenate_panics_when_incomplete() {
        let mut ce = ConcatenatedEmbedding::new();
        ce.concatenate(); // Should panic
    }

    #[test]
    fn test_total_dimension() {
        let mut ce = create_complete_embedding();
        ce.concatenate();
        assert_eq!(ce.total_dimension(), 8320);
    }

    #[test]
    fn test_total_dimension_partial() {
        let mut ce = ConcatenatedEmbedding::new();
        let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 100);
        emb.set_projected(true);
        ce.set(emb);

        assert_eq!(ce.total_dimension(), 1024);
    }

    // ========== Validation Tests ==========

    #[test]
    fn test_validate_succeeds_for_valid_embeddings() {
        let ce = create_complete_embedding();
        assert!(ce.validate().is_ok());
    }

    #[test]
    fn test_validate_detects_nan() {
        let mut ce = ConcatenatedEmbedding::new();
        let mut vector = vec![0.1; 1024];
        vector[500] = f32::NAN;
        let mut emb = ModelEmbedding::new(ModelId::Semantic, vector, 100);
        emb.set_projected(true);
        ce.set(emb);

        let result = ce.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidInput(msg) => assert!(msg.contains("NaN")),
            _ => panic!("Expected InvalidInput error"),
        }
    }

    // ========== Get Slice Tests ==========

    #[test]
    fn test_get_slice_returns_correct_segment() {
        let mut ce = create_complete_embedding();
        ce.concatenate();

        let semantic_slice = ce.get_slice(ModelId::Semantic).unwrap();
        assert_eq!(semantic_slice.len(), 1024);
        assert!(semantic_slice.iter().all(|&v| (v - 0.5).abs() < 1e-6));

        let late_interaction_slice = ce.get_slice(ModelId::LateInteraction).unwrap();
        assert_eq!(late_interaction_slice.len(), 128);
    }

    #[test]
    fn test_get_slice_none_before_concatenation() {
        let ce = ConcatenatedEmbedding::new();
        assert!(ce.get_slice(ModelId::Semantic).is_none());
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn edge_case_empty_struct() {
        let ce = ConcatenatedEmbedding::new();
        println!(
            "BEFORE: filled={}, complete={}",
            ce.filled_count(),
            ce.is_complete()
        );

        let missing = ce.missing_models();

        println!("AFTER: missing_count={}", missing.len());
        assert_eq!(missing.len(), 12);
        println!("Edge Case 1 PASSED: Empty struct returns all 12 models as missing");
    }

    #[test]
    fn edge_case_overwrite() {
        let mut ce = ConcatenatedEmbedding::new();
        let mut emb1 = ModelEmbedding::new(ModelId::Semantic, vec![1.0; 1024], 100);
        emb1.set_projected(true);
        ce.set(emb1);

        println!(
            "BEFORE: latency={}, first_value={}",
            ce.total_latency_us,
            ce.embeddings[0].as_ref().unwrap().vector[0]
        );

        let mut emb2 = ModelEmbedding::new(ModelId::Semantic, vec![2.0; 1024], 200);
        emb2.set_projected(true);
        ce.set(emb2);

        println!(
            "AFTER: latency={}, first_value={}",
            ce.total_latency_us,
            ce.embeddings[0].as_ref().unwrap().vector[0]
        );

        // Latency should be replaced (old subtracted, new added)
        assert_eq!(ce.total_latency_us, 200);
        assert_eq!(ce.embeddings[0].as_ref().unwrap().vector[0], 2.0);
        println!("Edge Case 2 PASSED: Overwrite replaces embedding and updates latency correctly");
    }

    #[test]
    fn edge_case_max_latency() {
        let mut ce = ConcatenatedEmbedding::new();
        let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], u64::MAX);
        emb.set_projected(true);

        println!("BEFORE: total_latency={}", ce.total_latency_us);
        ce.set(emb);
        println!("AFTER: total_latency={}", ce.total_latency_us);

        assert_eq!(ce.total_latency_us, u64::MAX);
        println!("Edge Case 3 PASSED: u64::MAX latency handled correctly");
    }

    // ========== Source of Truth Verification ==========

    #[test]
    fn verify_source_of_truth() {
        // The concatenated vector IS the source of truth
        let mut ce = create_complete_embedding();
        ce.concatenate();

        // 1. Verify vector exists in memory
        assert!(!ce.concatenated.is_empty());
        println!(
            "SOURCE OF TRUTH: concatenated.len() = {}",
            ce.concatenated.len()
        );

        // 2. Verify dimensions match specification
        assert_eq!(ce.concatenated.len(), dimensions::TOTAL_CONCATENATED);
        println!(
            "DIMENSION CHECK: {} == {} (expected)",
            ce.concatenated.len(),
            dimensions::TOTAL_CONCATENATED
        );

        // 3. Verify hash is non-zero
        assert_ne!(ce.content_hash, 0);
        println!(
            "HASH CHECK: content_hash = {} (non-zero)",
            ce.content_hash
        );

        // 4. Read back individual slices
        for (i, model_id) in ModelId::all().iter().enumerate() {
            let offset = dimensions::offset_by_index(i);
            let dim = dimensions::projected_dimension_by_index(i);
            let slice = &ce.concatenated[offset..offset + dim];

            println!(
                "MODEL {:?}: offset={}, dim={}, slice_len={}",
                model_id,
                offset,
                dim,
                slice.len()
            );
            assert_eq!(slice.len(), dim);
        }

        println!("VERIFICATION COMPLETE: All checks passed");
    }

    // ========== Helper Functions ==========

    fn create_complete_embedding() -> ConcatenatedEmbedding {
        let mut ce = ConcatenatedEmbedding::new();
        for model_id in ModelId::all() {
            let dim = model_id.projected_dimension();
            let mut emb = ModelEmbedding::new(*model_id, vec![0.5; dim], 100);
            emb.set_projected(true);
            ce.set(emb);
        }
        ce
    }
}
