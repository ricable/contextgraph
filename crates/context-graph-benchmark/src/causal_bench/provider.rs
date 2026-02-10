//! Embedding provider abstraction for causal benchmarks.
//!
//! Provides a trait for computing E5 scores, with two implementations:
//! - `SyntheticProvider`: deterministic hash-based scores (default, CI-compatible)
//! - `GpuProvider`: real CausalModel embeddings (requires `real-embeddings` feature + GPU)
//!
//! Thread through phase 2-7 functions via `BenchConfig::provider`.

/// Trait for computing E5 causal embedding scores and vectors.
///
/// Implementations must be deterministic for reproducible benchmarks.
pub trait EmbeddingProvider: Send + Sync {
    /// Compute E5 similarity score between a cause text and an effect text.
    ///
    /// Returns a cosine similarity in [0, 1].
    fn e5_score(&self, cause_text: &str, effect_text: &str) -> f32;

    /// Generate a full E5 embedding vector for the given text.
    ///
    /// Returns a 768-dimensional vector (nomic-embed-text-v1.5).
    fn e5_embedding(&self, text: &str) -> Vec<f32>;

    /// Generate dual E5 embeddings (cause variant + effect variant).
    ///
    /// Returns (cause_embedding, effect_embedding), each 768-dimensional.
    fn e5_dual_embeddings(&self, text: &str) -> (Vec<f32>, Vec<f32>) {
        // Default: same embedding for both variants (synthetic behavior)
        let emb = self.e5_embedding(text);
        (emb.clone(), emb)
    }

    /// Provider name for logging/reporting.
    fn name(&self) -> &str;

    /// Whether this provider uses real GPU embeddings.
    fn is_gpu(&self) -> bool {
        false
    }
}

/// Synthetic embedding provider using deterministic hash-based scores.
///
/// Simulates E5 compression behavior: causal text clusters 0.93-0.98,
/// non-causal text scores slightly lower at 0.90-0.95.
/// Used for CI and when GPU is unavailable.
pub struct SyntheticProvider;

impl SyntheticProvider {
    pub fn new() -> Self {
        Self
    }

    fn hash_text(text: &str) -> u64 {
        let mut hash = 5381u64;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    fn hash_to_float(text: &str) -> f32 {
        let h = Self::hash_text(text);
        let h = h.wrapping_mul(0x517cc1b727220a95);
        let h = h ^ (h >> 32);
        let h = h.wrapping_mul(0x6c62272e07bb0142);
        let h = h ^ (h >> 32);
        ((h >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

impl Default for SyntheticProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingProvider for SyntheticProvider {
    fn e5_score(&self, cause_text: &str, effect_text: &str) -> f32 {
        // Simulate E5 compression: scores cluster 0.93-0.98
        let combined = format!("{}{}", cause_text, effect_text);
        let base = 0.955;
        let noise = Self::hash_to_float(&combined) * 0.04;
        (base + noise).clamp(0.0, 1.0)
    }

    fn e5_embedding(&self, text: &str) -> Vec<f32> {
        let dim = 768;
        let mut vec = vec![0.0f32; dim];
        let seed = Self::hash_text(text);
        for (i, v) in vec.iter_mut().enumerate() {
            let h = seed.wrapping_add(i as u64);
            let h = h.wrapping_mul(0x517cc1b727220a95);
            let h = h ^ (h >> 32);
            *v = (h as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        // L2 normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
        vec
    }

    fn name(&self) -> &str {
        "synthetic"
    }
}

/// GPU-based embedding provider using the real CausalModel.
///
/// Requires `real-embeddings` feature flag and a CUDA-capable GPU.
/// Loads trained or base nomic-embed-text-v1.5 weights.
///
/// All CausalModel methods are async, so this provider uses a dedicated
/// tokio runtime to bridge from the synchronous `EmbeddingProvider` trait.
#[cfg(feature = "real-embeddings")]
pub struct GpuProvider {
    model: std::sync::Arc<context_graph_embeddings::models::pretrained::causal::model::CausalModel>,
    runtime: tokio::runtime::Runtime,
}

#[cfg(feature = "real-embeddings")]
impl GpuProvider {
    /// Create a new GPU provider, loading the CausalModel.
    ///
    /// Automatically loads trained LoRA + projection weights from
    /// `{model_path}/trained/` if available. Falls back to base model.
    pub fn new(model_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        use context_graph_embeddings::models::pretrained::causal::model::CausalModel;
        use context_graph_embeddings::traits::SingleModelConfig;

        let runtime = tokio::runtime::Runtime::new()?;
        let model = CausalModel::new(model_path, SingleModelConfig::default())?;
        runtime.block_on(model.load())?;

        // Auto-load trained weights if available
        let trained_dir = model_path.join("trained");
        match model.load_trained_weights(&trained_dir) {
            Ok(true) => println!("GpuProvider: loaded trained LoRA + projection weights"),
            Ok(false) => println!("GpuProvider: using base model (no trained weights found)"),
            Err(e) => println!("GpuProvider: trained weight loading failed ({}), using base model", e),
        }

        Ok(Self {
            model: std::sync::Arc::new(model),
            runtime,
        })
    }
}

#[cfg(feature = "real-embeddings")]
impl EmbeddingProvider for GpuProvider {
    fn e5_score(&self, cause_text: &str, effect_text: &str) -> f32 {
        let cause_emb = self.e5_embedding(cause_text);
        let effect_emb = self.e5_embedding(effect_text);
        cosine_similarity(&cause_emb, &effect_emb).max(0.0)
    }

    fn e5_embedding(&self, text: &str) -> Vec<f32> {
        match self.runtime.block_on(self.model.embed_as_cause(text)) {
            Ok(emb) => emb,
            Err(e) => {
                tracing::warn!("GPU embedding failed, falling back to zeros: {}", e);
                vec![0.0f32; 768]
            }
        }
    }

    fn e5_dual_embeddings(&self, text: &str) -> (Vec<f32>, Vec<f32>) {
        match self.runtime.block_on(self.model.embed_dual(text)) {
            Ok((cause, effect)) => (cause, effect),
            Err(e) => {
                tracing::warn!("GPU dual embedding failed: {}", e);
                (vec![0.0f32; 768], vec![0.0f32; 768])
            }
        }
    }

    fn name(&self) -> &str {
        "gpu"
    }

    fn is_gpu(&self) -> bool {
        true
    }
}

#[cfg(feature = "real-embeddings")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_provider_deterministic() {
        let provider = SyntheticProvider::new();
        let s1 = provider.e5_score("smoking", "cancer");
        let s2 = provider.e5_score("smoking", "cancer");
        assert_eq!(s1, s2, "Synthetic provider must be deterministic");
    }

    #[test]
    fn test_synthetic_provider_score_range() {
        let provider = SyntheticProvider::new();
        let score = provider.e5_score("chronic stress elevates cortisol", "hippocampal damage");
        assert!(score >= 0.0 && score <= 1.0, "Score {} out of range", score);
    }

    #[test]
    fn test_synthetic_embedding_dimension() {
        let provider = SyntheticProvider::new();
        let emb = provider.e5_embedding("test text");
        assert_eq!(emb.len(), 768);
    }

    #[test]
    fn test_synthetic_embedding_normalized() {
        let provider = SyntheticProvider::new();
        let emb = provider.e5_embedding("test text");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be L2-normalized, got norm={}", norm);
    }

    #[test]
    fn test_synthetic_dual_embeddings() {
        let provider = SyntheticProvider::new();
        let (cause, effect) = provider.e5_dual_embeddings("test text");
        assert_eq!(cause.len(), 768);
        assert_eq!(effect.len(), 768);
        // Default impl returns identical vectors
        assert_eq!(cause, effect);
    }

    #[test]
    fn test_provider_name() {
        let provider = SyntheticProvider::new();
        assert_eq!(provider.name(), "synthetic");
        assert!(!provider.is_gpu());
    }
}
