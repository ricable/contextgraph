//! Helper Functions: Deterministic Data Generation (NO MOCKS)

/// Generate deterministic embedding using mathematical functions.
/// Uses sin-based generation for reproducible, normalized values.
pub fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (i as f64 + seed as f64) * 0.1;
            ((x.sin() + 1.0) / 2.0) as f32 // Normalized [0, 1]
        })
        .collect()
}

/// Generate multiple context embeddings for comparison.
pub fn generate_context(count: usize, dim: usize, base_seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| generate_embedding(dim, base_seed + i as u64))
        .collect()
}

/// Generate embedding with a specific value pattern (for edge case testing).
pub fn uniform_embedding(dim: usize, value: f32) -> Vec<f32> {
    vec![value; dim]
}
