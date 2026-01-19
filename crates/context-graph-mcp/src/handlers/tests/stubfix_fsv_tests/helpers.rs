//! Synthetic data helpers for FSV tests

use sha2::{Digest, Sha256};

use context_graph_core::types::fingerprint::{
    SemanticFingerprint, SparseVector, TeleologicalFingerprint,
};

// ============================================================================
// Synthetic Data Helpers
// ============================================================================

/// Embedding dimensions (subset needed for tests)
pub(crate) const E1_DIM: usize = 1024;
pub(crate) const E2_DIM: usize = 512;
pub(crate) const E3_DIM: usize = 512;
pub(crate) const E4_DIM: usize = 512;
pub(crate) const E5_DIM: usize = 768;
pub(crate) const E7_DIM: usize = 1536;
pub(crate) const E8_DIM: usize = 384;
pub(crate) const E9_DIM: usize = 1024;
pub(crate) const E10_DIM: usize = 768;
pub(crate) const E11_DIM: usize = 384;
pub(crate) const E12_TOKEN_DIM: usize = 128;
pub(crate) const NUM_EMBEDDERS: usize = 13;

/// Create a synthetic fingerprint with configurable access count.
///
/// Parameters:
/// - content: Used to generate content hash and embeddings deterministically
/// - _theta: Unused parameter (preserved for API compatibility)
/// - access_count: Number of times this fingerprint has been accessed
pub(crate) fn create_test_fingerprint(
    content: &str,
    _purpose_value: f32,
    access_count: u64,
) -> TeleologicalFingerprint {
    let content_hash = {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hasher.finalize().into()
    };

    let semantic = create_test_semantic(&content_hash);

    let mut fp = TeleologicalFingerprint::new(semantic, content_hash);
    fp.access_count = access_count;
    fp
}

/// Create a synthetic fingerprint with specific created_at time for temporal tests.
pub(crate) fn create_test_fingerprint_with_age(
    content: &str,
    theta: f32,
    access_count: u64,
    days_old: i64,
) -> TeleologicalFingerprint {
    let mut fp = create_test_fingerprint(content, theta, access_count);
    fp.created_at = chrono::Utc::now() - chrono::Duration::days(days_old);
    fp.last_updated = fp.created_at;
    fp
}

/// Create semantic fingerprint with deterministic embeddings from hash.
pub(crate) fn create_test_semantic(hash: &[u8; 32]) -> SemanticFingerprint {
    let seed = |offset: usize| -> f32 {
        let idx = offset % 32;
        let val = hash[idx] as f32 / 255.0;
        (val * 2.0) - 1.0
    };

    let e1_semantic: Vec<f32> = (0..E1_DIM).map(|i| seed(i) * 0.5).collect();
    let e2_temporal_recent: Vec<f32> = (0..E2_DIM).map(|i| seed(i + 1000)).collect();
    let e3_temporal_periodic: Vec<f32> = (0..E3_DIM).map(|i| seed(i + 2000) * 0.8).collect();
    let e4_temporal_positional: Vec<f32> = (0..E4_DIM).map(|i| seed(i + 3000) * 0.6).collect();
    let e5_causal: Vec<f32> = (0..E5_DIM).map(|i| seed(i + 4000)).collect();
    let e7_code: Vec<f32> = (0..E7_DIM).map(|i| seed(i + 6000)).collect();
    let e8_graph: Vec<f32> = (0..E8_DIM).map(|i| seed(i + 7000)).collect();
    let e9_hdc: Vec<f32> = (0..E9_DIM)
        .map(|i| if seed(i + 8000) > 0.0 { 1.0 } else { -1.0 })
        .collect();
    let e10_multimodal: Vec<f32> = (0..E10_DIM).map(|i| seed(i + 9000)).collect();
    let e11_entity: Vec<f32> = (0..E11_DIM).map(|i| seed(i + 10000)).collect();

    // E12: Late interaction - 10 tokens
    let e12_late_interaction: Vec<Vec<f32>> = (0..10)
        .map(|t| {
            (0..E12_TOKEN_DIM)
                .map(|i| seed(i + t * 128 + 11000))
                .collect()
        })
        .collect();

    // Sparse vectors
    let mut e6_indices: Vec<u16> = (0..20)
        .map(|i| ((hash[i % 32] as u32 * 100) % 30000) as u16)
        .collect();
    e6_indices.sort();
    e6_indices.dedup();
    let e6_values: Vec<f32> = (0..e6_indices.len())
        .map(|i| seed(i + 5000).abs() * 2.0)
        .collect();

    let mut e13_indices: Vec<u16> = (0..30)
        .map(|i| ((hash[(i + 10) % 32] as u32 * 200) % 30000) as u16)
        .collect();
    e13_indices.sort();
    e13_indices.dedup();
    let e13_values: Vec<f32> = (0..e13_indices.len())
        .map(|i| seed(i + 12000).abs() * 3.0)
        .collect();

    SemanticFingerprint {
        e1_semantic,
        e2_temporal_recent,
        e3_temporal_periodic,
        e4_temporal_positional,
        e5_causal,
        e6_sparse: SparseVector {
            indices: e6_indices,
            values: e6_values,
        },
        e7_code,
        e8_graph,
        e9_hdc,
        e10_multimodal,
        e11_entity,
        e12_late_interaction,
        e13_splade: SparseVector {
            indices: e13_indices,
            values: e13_values,
        },
    }
}
