//! Topic cluster generation for controlled benchmarks.
//!
//! Generates topic centroids with controlled inter-topic distance
//! to ensure distinct, well-separated clusters.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

use context_graph_core::types::fingerprint::{
    E11_DIM, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E7_DIM, E8_DIM, E9_DIM,
};

use crate::util::cosine_similarity;

/// A topic cluster with centroid embeddings.
#[derive(Debug, Clone)]
pub struct TopicCluster {
    /// Topic ID.
    pub id: usize,
    /// Topic name (synthetic).
    pub name: String,
    /// Centroid for E1 (semantic) - 1024D.
    pub e1_centroid: Vec<f32>,
    /// Centroid for E5 (causal) - 768D.
    pub e5_centroid: Vec<f32>,
    /// Centroid for E7 (code) - 1536D.
    pub e7_centroid: Vec<f32>,
    /// Centroid for E10 (multimodal) - 768D.
    pub e10_centroid: Vec<f32>,
    /// Noise standard deviation for this topic.
    pub noise_std: f32,
}

impl TopicCluster {
    /// Get E1 centroid dimension.
    pub fn e1_dim() -> usize {
        E1_DIM
    }

    /// Generate a point within this cluster (E1 only, for baseline comparison).
    pub fn sample_e1(&self, rng: &mut ChaCha8Rng, noise_std: f32) -> Vec<f32> {
        sample_from_centroid(&self.e1_centroid, noise_std, rng)
    }

    /// Generate a complete point within this cluster (all embeddings).
    pub fn sample_all(
        &self,
        rng: &mut ChaCha8Rng,
        noise_std: f32,
    ) -> AllEmbeddings {
        AllEmbeddings {
            e1: sample_from_centroid(&self.e1_centroid, noise_std, rng),
            e2: random_embedding(E2_DIM, rng), // Temporal - not topic-specific
            e3: random_embedding(E3_DIM, rng),
            e4: random_embedding(E4_DIM, rng),
            e5: sample_from_centroid(&self.e5_centroid, noise_std, rng),
            e7: sample_from_centroid(&self.e7_centroid, noise_std, rng),
            e8: random_embedding(E8_DIM, rng),
            e9: random_embedding(E9_DIM, rng),
            e10: sample_from_centroid(&self.e10_centroid, noise_std, rng),
            e11: random_embedding(E11_DIM, rng),
        }
    }
}

/// All dense embeddings for a generated point.
#[derive(Debug, Clone)]
pub struct AllEmbeddings {
    pub e1: Vec<f32>,
    pub e2: Vec<f32>,
    pub e3: Vec<f32>,
    pub e4: Vec<f32>,
    pub e5: Vec<f32>,
    pub e7: Vec<f32>,
    pub e8: Vec<f32>,
    pub e9: Vec<f32>,
    pub e10: Vec<f32>,
    pub e11: Vec<f32>,
}

/// Generator for topic clusters.
pub struct TopicGenerator {
    rng: ChaCha8Rng,
    /// Minimum cosine similarity between topic centroids.
    min_inter_topic_distance: f32,
    /// Maximum cosine similarity between topic centroids.
    max_inter_topic_similarity: f32,
    /// Next topic ID for divergent topics.
    next_id: usize,
}

impl TopicGenerator {
    /// Create a new generator with seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            min_inter_topic_distance: 0.5,
            max_inter_topic_similarity: 0.3,
            next_id: 1000, // Start divergent topic IDs at 1000
        }
    }

    /// Set minimum distance between topic centroids.
    pub fn with_min_distance(mut self, min_distance: f32) -> Self {
        self.min_inter_topic_distance = min_distance;
        self
    }

    /// Set maximum similarity between topic centroids.
    pub fn with_max_similarity(mut self, max_similarity: f32) -> Self {
        self.max_inter_topic_similarity = max_similarity;
        self
    }

    /// Generate N topic clusters with controlled separation.
    pub fn generate(&mut self, n_topics: usize, noise_std: f32) -> Vec<TopicCluster> {
        let mut clusters = Vec::with_capacity(n_topics);

        for i in 0..n_topics {
            let cluster = self.generate_single_topic(i, &clusters, noise_std);
            clusters.push(cluster);
        }

        clusters
    }

    /// Generate a single topic with controlled distance from existing topics.
    fn generate_single_topic(
        &mut self,
        id: usize,
        existing: &[TopicCluster],
        noise_std: f32,
    ) -> TopicCluster {
        const MAX_ATTEMPTS: usize = 100;

        for _ in 0..MAX_ATTEMPTS {
            let e1_centroid = random_unit_embedding(E1_DIM, &mut self.rng);
            let e5_centroid = random_unit_embedding(768, &mut self.rng);
            let e7_centroid = random_unit_embedding(E7_DIM, &mut self.rng);
            let e10_centroid = random_unit_embedding(768, &mut self.rng);

            // Check distance from all existing topics
            let mut valid = true;
            for existing_cluster in existing {
                let sim = cosine_similarity(&e1_centroid, &existing_cluster.e1_centroid);
                if sim > self.max_inter_topic_similarity {
                    valid = false;
                    break;
                }
            }

            if valid {
                return TopicCluster {
                    id,
                    name: format!("Topic_{}", id),
                    e1_centroid,
                    e5_centroid,
                    e7_centroid,
                    e10_centroid,
                    noise_std,
                };
            }
        }

        // If we couldn't find a well-separated point, just use random
        // This can happen with many topics in high dimensions
        TopicCluster {
            id,
            name: format!("Topic_{}", id),
            e1_centroid: random_unit_embedding(E1_DIM, &mut self.rng),
            e5_centroid: random_unit_embedding(768, &mut self.rng),
            e7_centroid: random_unit_embedding(E7_DIM, &mut self.rng),
            e10_centroid: random_unit_embedding(768, &mut self.rng),
            noise_std,
        }
    }

    /// Generate a divergent query centroid (not similar to any existing topic).
    pub fn generate_divergent_centroid(&mut self, existing: &[TopicCluster]) -> Vec<f32> {
        const MAX_ATTEMPTS: usize = 100;

        for _ in 0..MAX_ATTEMPTS {
            let centroid = random_unit_embedding(E1_DIM, &mut self.rng);

            let max_sim = existing
                .iter()
                .map(|c| cosine_similarity(&centroid, &c.e1_centroid))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            if max_sim < self.max_inter_topic_similarity {
                return centroid;
            }
        }

        // Fallback
        random_unit_embedding(E1_DIM, &mut self.rng)
    }

    /// Generate a full divergent topic cluster (not similar to any existing topic).
    /// Used for creating full fingerprints for divergent queries.
    pub fn generate_divergent_topic(&mut self, existing: &[TopicCluster]) -> TopicCluster {
        let id = self.next_id;
        self.next_id += 1;

        let e1_centroid = self.generate_divergent_centroid(existing);

        TopicCluster {
            id,
            name: format!("Divergent_{}", id),
            e1_centroid,
            e5_centroid: random_unit_embedding(768, &mut self.rng),
            e7_centroid: random_unit_embedding(E7_DIM, &mut self.rng),
            e10_centroid: random_unit_embedding(768, &mut self.rng),
            noise_std: 0.1, // Lower noise for divergent topics
        }
    }
}

/// Generate a random unit-length embedding.
fn random_unit_embedding(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut embedding: Vec<f32> = (0..dim).map(|_| normal.sample(rng) as f32).collect();

    // Normalize to unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

/// Generate a random embedding (not necessarily unit length).
fn random_embedding(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let normal = Normal::new(0.0, 0.5).unwrap();
    (0..dim).map(|_| normal.sample(rng) as f32).collect()
}

/// Sample from a centroid with Gaussian noise.
fn sample_from_centroid(centroid: &[f32], noise_std: f32, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let normal = Normal::new(0.0, noise_std as f64).unwrap();

    let mut sample: Vec<f32> = centroid
        .iter()
        .map(|&x| x + normal.sample(rng) as f32)
        .collect();

    // Normalize to unit length
    let norm: f32 = sample.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut sample {
            *x /= norm;
        }
    }

    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_generation() {
        let mut generator = TopicGenerator::new(42);
        let topics = generator.generate(5, 0.1);

        assert_eq!(topics.len(), 5);

        // Check all topics have correct dimensions
        for topic in &topics {
            assert_eq!(topic.e1_centroid.len(), E1_DIM);
            assert_eq!(topic.e5_centroid.len(), 768);
            assert_eq!(topic.e7_centroid.len(), E7_DIM);
        }
    }

    #[test]
    fn test_topic_separation() {
        let mut generator = TopicGenerator::new(42).with_max_similarity(0.3);
        let topics = generator.generate(5, 0.1);

        // Check pairwise similarity
        for i in 0..topics.len() {
            for j in (i + 1)..topics.len() {
                let sim = cosine_similarity(&topics[i].e1_centroid, &topics[j].e1_centroid);
                // Allow some tolerance since we may not always achieve perfect separation
                assert!(
                    sim < 0.5,
                    "Topics {} and {} too similar: {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_sample_from_topic() {
        let mut generator = TopicGenerator::new(42);
        let topics = generator.generate(2, 0.1);

        let mut rng = ChaCha8Rng::seed_from_u64(123);

        // Sample from topic 0
        let sample0 = topics[0].sample_e1(&mut rng, 0.1);
        let _sample1 = topics[1].sample_e1(&mut rng, 0.1);

        // Samples should be more similar to their own centroid
        let sim00 = cosine_similarity(&sample0, &topics[0].e1_centroid);
        let sim01 = cosine_similarity(&sample0, &topics[1].e1_centroid);

        assert!(
            sim00 > sim01,
            "Sample should be more similar to own centroid"
        );
    }
}
