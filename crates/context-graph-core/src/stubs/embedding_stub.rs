//! Stub embedding provider for testing.
//!
//! Generates DETERMINISTIC embeddings based on content hash.
//! NOT for production - use real Candle provider (M06-T04).
//!
//! # How It Works
//!
//! 1. Hash content using DefaultHasher
//! 2. Seed LCG PRNG with hash
//! 3. Generate deterministic vector from seeded RNG
//! 4. Normalize to unit length
//!
//! This ensures:
//! - Same content → same embedding (deterministic tests)
//! - Different content → different embedding (similarity works)
//! - Embeddings are normalized (cosine similarity valid)
//!
//! # NEVER use vec![0.1; 1536] - all nodes would be identical!

use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use crate::error::{CoreError, CoreResult};
use crate::traits::{EmbeddingOutput, EmbeddingProvider};

/// Stub embedding provider for testing.
///
/// Generates hash-based deterministic embeddings.
/// Different content produces different (but repeatable) vectors.
///
/// # Example
///
/// ```rust
/// use context_graph_core::stubs::StubEmbeddingProvider;
/// use context_graph_core::traits::EmbeddingProvider;
///
/// let provider = StubEmbeddingProvider::new();
/// assert_eq!(provider.dimensions(), 1536);
/// assert!(provider.is_ready());
/// ```
pub struct StubEmbeddingProvider {
    dimensions: usize,
    model_id: String,
}

impl Default for StubEmbeddingProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl StubEmbeddingProvider {
    /// Create with default 1536 dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_core::stubs::StubEmbeddingProvider;
    ///
    /// let provider = StubEmbeddingProvider::new();
    /// assert_eq!(provider.dimensions(), 1536);
    /// ```
    pub fn new() -> Self {
        Self {
            dimensions: 1536,
            model_id: "stub-embedding-v1".to_string(),
        }
    }

    /// Create with custom dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_core::stubs::StubEmbeddingProvider;
    ///
    /// let provider = StubEmbeddingProvider::with_dimensions(768);
    /// assert_eq!(provider.dimensions(), 768);
    /// ```
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self {
            dimensions,
            model_id: format!("stub-embedding-v1-d{}", dimensions),
        }
    }

    /// Get the configured dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Generate deterministic embedding from content hash.
    ///
    /// Uses DefaultHasher for hashing, seeds LCG PRNG, generates normalized vector.
    ///
    /// # Algorithm
    ///
    /// 1. Hash content string to u64
    /// 2. Use hash as seed for Linear Congruential Generator
    /// 3. Generate `dimensions` values in [-1, 1]
    /// 4. Normalize to unit length (magnitude = 1.0)
    ///
    /// # Determinism Guarantee
    ///
    /// Same content always produces identical embedding.
    fn generate_embedding(&self, content: &str) -> Vec<f32> {
        // Hash content
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        let hash = hasher.finish();

        // Seed simple LCG PRNG
        // LCG parameters from Knuth MMIX
        let mut seed = hash;
        let mut vector = Vec::with_capacity(self.dimensions);

        for _ in 0..self.dimensions {
            // LCG: next = (a * seed + c) mod m
            // Using Knuth MMIX parameters: a=6364136223846793005, c=1
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to f32 in [-1, 1]
            let value = (seed as f64 / u64::MAX as f64) * 2.0 - 1.0;
            vector.push(value as f32);
        }

        // Normalize to unit length
        let magnitude: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut vector {
                *v /= magnitude;
            }
        }

        vector
    }
}

#[async_trait]
impl EmbeddingProvider for StubEmbeddingProvider {
    /// Generate embedding for single text content.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::Embedding` if content is empty.
    async fn embed(&self, content: &str) -> CoreResult<EmbeddingOutput> {
        let start = Instant::now();

        if content.is_empty() {
            return Err(CoreError::Embedding("Empty content".into()));
        }

        let vector = self.generate_embedding(content);
        let latency = start.elapsed();

        EmbeddingOutput::new(vector, &self.model_id, latency)
    }

    /// Generate embeddings for batch of texts.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::Embedding` if any content is empty.
    async fn embed_batch(&self, contents: &[String]) -> CoreResult<Vec<EmbeddingOutput>> {
        let mut results = Vec::with_capacity(contents.len());
        for content in contents {
            results.push(self.embed(content).await?);
        }
        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn is_ready(&self) -> bool {
        true // Stub is always ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stub_produces_correct_dimensions() {
        let provider = StubEmbeddingProvider::new();
        let result = provider.embed("test content").await.unwrap();
        assert_eq!(result.dimensions, 1536);
        assert_eq!(result.vector.len(), 1536);
    }

    #[tokio::test]
    async fn test_stub_different_content_different_embedding() {
        let provider = StubEmbeddingProvider::new();

        let emb1 = provider.embed("first content").await.unwrap();
        let emb2 = provider.embed("second content").await.unwrap();

        // Vectors must be different
        assert_ne!(emb1.vector, emb2.vector);

        // Cosine similarity should not be 1.0 (they should be different)
        let dot: f32 = emb1
            .vector
            .iter()
            .zip(&emb2.vector)
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot < 0.99,
            "Different content should have different embeddings, got dot={}",
            dot
        );
    }

    #[tokio::test]
    async fn test_stub_same_content_same_embedding() {
        let provider = StubEmbeddingProvider::new();

        let emb1 = provider.embed("same content").await.unwrap();
        let emb2 = provider.embed("same content").await.unwrap();

        // Vectors must be identical (deterministic)
        assert_eq!(emb1.vector, emb2.vector);
    }

    #[tokio::test]
    async fn test_stub_embedding_normalized() {
        let provider = StubEmbeddingProvider::new();
        let result = provider.embed("test content").await.unwrap();

        let magnitude: f32 = result.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.001,
            "Embedding should be normalized, magnitude={}",
            magnitude
        );
    }

    #[tokio::test]
    async fn test_stub_empty_content_fails() {
        let provider = StubEmbeddingProvider::new();
        let result = provider.embed("").await;
        assert!(result.is_err());
        match result {
            Err(CoreError::Embedding(msg)) => {
                assert!(msg.contains("Empty"));
            }
            _ => panic!("Expected CoreError::Embedding"),
        }
    }

    #[tokio::test]
    async fn test_stub_batch_embedding() {
        let provider = StubEmbeddingProvider::new();
        let contents = vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ];

        let results = provider.embed_batch(&contents).await.unwrap();
        assert_eq!(results.len(), 3);

        // All different
        assert_ne!(results[0].vector, results[1].vector);
        assert_ne!(results[1].vector, results[2].vector);
    }

    #[tokio::test]
    async fn test_stub_custom_dimensions() {
        let provider = StubEmbeddingProvider::with_dimensions(768);
        let result = provider.embed("test").await.unwrap();
        assert_eq!(result.dimensions, 768);
        assert_eq!(result.vector.len(), 768);
    }

    #[tokio::test]
    async fn test_stub_is_ready() {
        let provider = StubEmbeddingProvider::new();
        assert!(provider.is_ready());
    }

    #[tokio::test]
    async fn test_stub_latency_tracked() {
        let provider = StubEmbeddingProvider::new();
        let result = provider.embed("test content").await.unwrap();
        // Latency should be non-zero (even if tiny)
        assert!(result.latency.as_nanos() > 0);
    }

    #[tokio::test]
    async fn test_stub_model_id() {
        let provider = StubEmbeddingProvider::new();
        assert_eq!(provider.model_id(), "stub-embedding-v1");

        let custom = StubEmbeddingProvider::with_dimensions(768);
        assert_eq!(custom.model_id(), "stub-embedding-v1-d768");
    }

    #[tokio::test]
    async fn test_stub_unicode_content() {
        let provider = StubEmbeddingProvider::new();

        let emb_unicode = provider.embed("日本語テスト🚀").await.unwrap();
        let emb_ascii = provider.embed("Japanese Test").await.unwrap();

        // Unicode produces different embedding than ASCII
        assert_ne!(emb_unicode.vector, emb_ascii.vector);

        // Both are properly normalized
        assert!((emb_unicode.magnitude() - 1.0).abs() < 0.001);
        assert!((emb_ascii.magnitude() - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_stub_long_content() {
        let provider = StubEmbeddingProvider::new();

        // 100KB string
        let long_content = "x".repeat(100_000);
        let result = provider.embed(&long_content).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.dimensions, 1536);
        assert!((output.magnitude() - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_stub_consistency_across_instances() {
        // Two different instances should produce same embedding for same content
        let provider1 = StubEmbeddingProvider::new();
        let provider2 = StubEmbeddingProvider::new();

        let emb1 = provider1.embed("consistency test").await.unwrap();
        let emb2 = provider2.embed("consistency test").await.unwrap();

        assert_eq!(emb1.vector, emb2.vector);
    }

    /// Source of truth verification test.
    ///
    /// Verifies that provider state matches output.
    #[tokio::test]
    async fn test_source_of_truth_verification() {
        let provider = StubEmbeddingProvider::new();

        // BEFORE: Check provider state
        let expected_dim = provider.dimensions();
        let expected_model = provider.model_id().to_string();
        let is_ready = provider.is_ready();

        assert!(is_ready, "Provider should be ready");
        assert_eq!(expected_dim, 1536);
        assert_eq!(expected_model, "stub-embedding-v1");

        // EXECUTE: Generate embedding
        let result = provider.embed("verification content").await.unwrap();

        // AFTER: Verify output matches source of truth
        assert_eq!(
            result.dimensions, expected_dim,
            "Output dimensions must match provider dimensions"
        );
        assert_eq!(
            result.model_id, expected_model,
            "Output model_id must match provider model_id"
        );
        assert_eq!(
            result.vector.len(),
            result.dimensions,
            "Vector length must match dimensions"
        );
    }

    /// Edge case audit: empty, long, unicode
    #[tokio::test]
    async fn test_edge_case_audit() {
        let provider = StubEmbeddingProvider::new();

        // EDGE CASE 1: Empty content
        let result = provider.embed("").await;
        assert!(result.is_err(), "Empty content must fail");

        // EDGE CASE 2: Very long content
        let long_content = "x".repeat(100_000);
        let result = provider.embed(&long_content).await;
        assert!(result.is_ok(), "Long content must succeed");

        // EDGE CASE 3: Unicode content
        let unicode = "日本語テスト🚀";
        let ascii = "Japanese Test";
        let emb_unicode = provider.embed(unicode).await.unwrap();
        let emb_ascii = provider.embed(ascii).await.unwrap();
        assert_ne!(
            emb_unicode.vector, emb_ascii.vector,
            "Unicode must produce different embedding than ASCII"
        );
    }
}
