//! SONA (Self-Organizing Neural Adaptation) learning hooks for RVF.
//!
//! This module implements the 3-loop architecture for continuous learning:
//! - **Loop A** (<100μs): Per-query confidence scoring
//! - **Loop B** (batch): Retrain retrieval weights → OVERLAY_SEG (LoRA)
//! - **Loop C** (consolidation): HNSW topology → GRAPH_SEG
//!
//! # Architecture
//!
//! ```ignore
//! Query → Loop A (confidence) → [if low] → Loop B (adapt) → [periodic] → Loop C (consolidate)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use context_graph_storage::rvf::sona::{SonaLearning, SonaLoop, SonaConfidence};
//!
//! let sona = SonaLearning::new(config);
//! let confidence = sona.evaluate_confidence(&query, &results).await?;
//! if confidence.score < 0.7 {
//!     sona.adapt(&query, &results, feedback).await?;
//! }
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// SONA learning loop types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SonaLoop {
    /// Per-query confidence scoring (<100μs)
    LoopA,
    /// Batch retraining of retrieval weights (LoRA)
    LoopB,
    /// Consolidation: HNSW topology → Graph (periodic)
    LoopC,
}

impl SonaLoop {
    /// Get the latency target for this loop.
    pub fn latency_target_us(&self) -> u64 {
        match self {
            Self::LoopA => 100,  // <100μs
            Self::LoopB => 10_000_000, // ~10s batch
            Self::LoopC => 300_000_000, // ~5min consolidation
        }
    }

    /// Get the human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::LoopA => "Loop A (Confidence)",
            Self::LoopB => "Loop B (Adaptation)",
            Self::LoopC => "Loop C (Consolidation)",
        }
    }
}

/// Confidence score for query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaConfidence {
    /// Overall confidence score [0.0, 1.0]
    pub score: f32,
    /// Relevance component [0.0, 1.0]
    pub relevance: f32,
    /// Diversity component [0.0, 1.0]
    pub diversity: f32,
    /// Novelty component [0.0, 1.0]
    pub novelty: f32,
    /// Source loop that generated this confidence
    pub source_loop: SonaLoop,
    /// Timestamp (Unix epoch ms)
    pub timestamp_ms: i64,
}

impl Default for SonaConfidence {
    fn default() -> Self {
        Self {
            score: 0.5,
            relevance: 0.5,
            diversity: 0.5,
            novelty: 0.5,
            source_loop: SonaLoop::LoopA,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }
}

impl SonaConfidence {
    /// Create a new confidence score.
    pub fn new(relevance: f32, diversity: f32, novelty: f32) -> Self {
        // Weighted combination: 60% relevance, 25% diversity, 15% novelty
        let score = 0.6 * relevance + 0.25 * diversity + 0.15 * novelty;
        Self {
            score: score.clamp(0.0, 1.0),
            relevance: relevance.clamp(0.0, 1.0),
            diversity: diversity.clamp(0.0, 1.0),
            novelty: novelty.clamp(0.0, 1.0),
            source_loop: SonaLoop::LoopA,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Check if confidence is high enough (>= threshold).
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.score >= threshold
    }

    /// Get recommendations based on confidence level.
    pub fn recommendation(&self) -> SonaRecommendation {
        if self.score >= 0.85 {
            SonaRecommendation::Accept
        } else if self.score >= 0.7 {
            SonaRecommendation::AcceptWithMinorReview
        } else if self.score >= 0.5 {
            SonaRecommendation::NeedsAdaptation
        } else {
            SonaRecommendation::NeedsConsolidation
        }
    }
}

/// Recommendation based on confidence score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SonaRecommendation {
    /// High confidence - accept results
    Accept,
    /// Good confidence - accept with minor review
    AcceptWithMinorReview,
    /// Medium confidence - needs Loop B adaptation
    NeedsAdaptation,
    /// Low confidence - needs Loop C consolidation
    NeedsConsolidation,
}

/// Feedback from query result evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaFeedback {
    /// Query that was executed
    pub query_id: Uuid,
    /// Results that were returned
    pub result_ids: Vec<Uuid>,
    /// User feedback (positive/negative/none)
    pub feedback_type: FeedbackType,
    /// Optional rating [0.0, 1.0]
    pub rating: Option<f32>,
    /// Timestamp
    pub timestamp_ms: i64,
}

impl SonaFeedback {
    /// Create positive feedback.
    pub fn positive(query_id: Uuid, result_ids: Vec<Uuid>) -> Self {
        Self {
            query_id,
            result_ids,
            feedback_type: FeedbackType::Positive,
            rating: Some(1.0),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create negative feedback.
    pub fn negative(query_id: Uuid, result_ids: Vec<Uuid>) -> Self {
        Self {
            query_id,
            result_ids,
            feedback_type: FeedbackType::Negative,
            rating: Some(0.0),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }
}

/// Type of user feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackType {
    Positive,
    Negative,
    Neutral,
}

/// SONA learning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaConfig {
    /// Confidence threshold for accepting results (default: 0.7)
    pub confidence_threshold: f32,
    /// Minimum samples before Loop B triggers (default: 10)
    pub loop_b_min_samples: usize,
    /// Minimum samples before Loop C triggers (default: 100)
    pub loop_c_min_samples: usize,
    /// Learning rate for Loop B (default: 0.01)
    pub learning_rate: f32,
    /// EWC lambda for catastrophic forgetting prevention (default: 0.5)
    pub ewc_lambda: f32,
    /// LoRA rank for overlay adaptation (default: 8)
    pub lora_rank: usize,
    /// Consolidation interval in ms (default: 300000 = 5min)
    pub consolidation_interval_ms: i64,
    /// Enable hyperbolic (Poincaré) distance (default: true)
    pub hyperbolic_enabled: bool,
    /// Poincaré curvature for hyperbolic space (default: 1.0)
    pub poincare_curvature: f32,
}

impl Default for SonaConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            loop_b_min_samples: 10,
            loop_c_min_samples: 100,
            learning_rate: 0.01,
            ewc_lambda: 0.5,
            lora_rank: 8,
            consolidation_interval_ms: 300_000,
            hyperbolic_enabled: true,
            poincare_curvature: 1.0,
        }
    }
}

/// SONA learning state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaState {
    /// Total queries processed
    pub queries_processed: u64,
    /// Total adaptations performed (Loop B)
    pub adaptations_performed: u64,
    /// Total consolidations performed (Loop C)
    pub consolidations_performed: u64,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Last consolidation timestamp
    pub last_consolidation_ms: i64,
    /// Loop B queue size
    pub loop_b_queue_size: usize,
    /// Loop C queue size
    pub loop_c_queue_size: usize,
}

impl Default for SonaState {
    fn default() -> Self {
        Self {
            queries_processed: 0,
            adaptations_performed: 0,
            consolidations_performed: 0,
            avg_confidence: 0.5,
            last_consolidation_ms: 0,
            loop_b_queue_size: 0,
            loop_c_queue_size: 0,
        }
    }
}

/// SONA learning engine.
///
/// Implements the 3-loop architecture for continuous learning.
pub struct SonaLearning {
    config: SonaConfig,
    state: RwLock<SonaState>,
    // Feedback queue for Loop B
    feedback_queue: RwLock<Vec<SonaFeedback>>,
    // Query embeddings for consolidation
    query_history: RwLock<HashMap<Uuid, QueryRecord>>,
    // Learned weights for retrieval
    learned_weights: RwLock<HashMap<String, f32>>,
}

#[derive(Debug, Clone)]
struct QueryRecord {
    embedding: Vec<f32>,
    results: Vec<(Uuid, f32)>,
    confidence: f32,
    timestamp_ms: i64,
}

impl SonaLearning {
    /// Create a new SONA learning engine.
    pub fn new(config: SonaConfig) -> Self {
        Self {
            config,
            state: RwLock::new(SonaState::default()),
            feedback_queue: RwLock::new(Vec::new()),
            query_history: RwLock::new(HashMap::new()),
            learned_weights: RwLock::new(HashMap::new()),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &SonaConfig {
        &self.config
    }

    /// Get the current state.
    pub fn state(&self) -> SonaState {
        self.state.read().clone()
    }

    /// Evaluate confidence for query results (Loop A).
    ///
    /// This is the fast path (<100μs) that scores query results
    /// and determines if further learning is needed.
    pub fn evaluate_confidence(
        &self,
        query_embedding: &[f32],
        result_embeddings: &[Vec<f32>],
        _result_ids: &[Uuid],
    ) -> SonaConfidence {
        if result_embeddings.is_empty() {
            return SonaConfidence::default();
        }

        // Calculate relevance: average similarity to query
        let relevance = Self::calculate_relevance(query_embedding, result_embeddings);

        // Calculate diversity: how different are the results from each other
        let diversity = Self::calculate_diversity(result_embeddings);

        // Calculate novelty: how different from recent history
        let history = self.query_history.read();
        let novelty = if history.is_empty() {
            0.5
        } else {
            let recent: Vec<_> = history.values().take(10).collect();
            Self::calculate_novelty(query_embedding, &recent)
        };

        let mut confidence = SonaConfidence::new(relevance, diversity, novelty);
        confidence.source_loop = SonaLoop::LoopA;

        // Update state
        {
            let mut state = self.state.write();
            state.queries_processed += 1;
            // Exponential moving average of confidence
            state.avg_confidence = 0.95 * state.avg_confidence + 0.05 * confidence.score;
        }

        confidence
    }

    /// Add feedback to the adaptation queue (Loop B trigger).
    pub fn add_feedback(&self, feedback: SonaFeedback) {
        let mut queue = self.feedback_queue.write();
        queue.push(feedback);

        let mut state = self.state.write();
        state.loop_b_queue_size = queue.len();
    }

    /// Check if Loop B should trigger.
    pub fn should_run_loop_b(&self) -> bool {
        let state = self.state.read();
        state.loop_b_queue_size >= self.config.loop_b_min_samples
    }

    /// Run Loop B: batch adaptation with LoRA overlay.
    ///
    /// Returns the number of weights updated.
    pub fn run_loop_b(&self) -> usize {
        // Drain feedback queue
        let feedback: Vec<_> = {
            let mut queue = self.feedback_queue.write();
            let items: Vec<_> = queue.drain(..).collect();
            queue.clear();
            items
        };

        if feedback.is_empty() {
            return 0;
        }

        let mut state = self.state.write();
        state.loop_b_queue_size = 0;
        state.adaptations_performed += 1;

        // Simplified LoRA weight update
        // In production, this would train actual LoRA matrices
        let mut weights = self.learned_weights.write();
        let mut updated = 0;

        for fb in &feedback {
            // Extract key terms from feedback (simplified)
            let key = format!("query_{}", fb.query_id);
            let delta = match fb.feedback_type {
                FeedbackType::Positive => self.config.learning_rate,
                FeedbackType::Negative => -self.config.learning_rate,
                FeedbackType::Neutral => 0.0,
            };

            let current = weights.get(&key).copied().unwrap_or(0.0);
            let new_weight = (current + delta).clamp(-1.0, 1.0);
            weights.insert(key, new_weight);
            updated += 1;
        }

        updated
    }

    /// Check if Loop C should trigger.
    pub fn should_run_loop_c(&self) -> bool {
        let state = self.state.read();
        let now = chrono::Utc::now().timestamp_millis();
        let time_since_consolidation = now - state.last_consolidation_ms;

        // Check both queue size and time interval
        let queue_threshold = state.loop_c_queue_size >= self.config.loop_c_min_samples;
        let time_threshold = time_since_consolidation >= self.config.consolidation_interval_ms;

        queue_threshold || time_threshold
    }

    /// Run Loop C: consolidate query history into graph topology.
    ///
    /// This updates the HNSW topology based on frequently co-occurring results.
    pub fn run_loop_c(&self) -> (usize, usize) {
        let history: Vec<_> = {
            let mut h = self.query_history.write();
            let items: Vec<_> = h.drain().collect();
            // Keep last 1000 for next iteration
            for (id, record) in items.iter().take(1000) {
                h.insert(*id, record.clone());
            }
            items
        };

        let mut state = self.state.write();
        state.loop_c_queue_size = 0;
        state.last_consolidation_ms = chrono::Utc::now().timestamp_millis();
        state.consolidations_performed += 1;

        // Simplified: return (queries_analyzed, edges_identified)
        let queries = history.len();
        let edges = queries.saturating_sub(1); // N-1 edges for N queries

        (queries, edges)
    }

    /// Queue a query for Loop C analysis.
    pub fn queue_for_consolidation(
        &self,
        query_id: Uuid,
        query_embedding: Vec<f32>,
        results: Vec<(Uuid, f32)>,
        confidence: f32,
    ) {
        let record = QueryRecord {
            embedding: query_embedding,
            results,
            confidence,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        };

        let mut history = self.query_history.write();
        history.insert(query_id, record);

        let mut state = self.state.write();
        state.loop_c_queue_size = history.len();
    }

    /// Get learned weight for a query key.
    pub fn get_weight(&self, key: &str) -> f32 {
        self.learned_weights
            .read()
            .get(key)
            .copied()
            .unwrap_or(0.0)
    }

    /// Apply learned weights to search scores.
    pub fn apply_weights(&self, base_scores: &mut [f32], query_keys: &[String]) {
        let weights = self.learned_weights.read();
        for (score, key) in base_scores.iter_mut().zip(query_keys.iter()) {
            if let Some(weight) = weights.get(key) {
                *score += *weight * 0.1; // Small weight influence
            }
        }
    }

    /// Calculate Poincaré distance for hyperbolic embedding space.
    pub fn poincare_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if !self.config.hyperbolic_enabled {
            // Fall back to Euclidean
            return a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
        }

        // Poincaré ball model distance
        // d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
        let norm_u = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_v = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_u >= 1.0 || norm_v >= 1.0 {
            return f32::MAX; // Points outside the unit ball
        }

        let diff_sq = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>();

        let denominator = (1.0 - norm_u * norm_u) * (1.0 - norm_v * norm_v);
        if denominator <= 0.0 {
            return f32::MAX;
        }

        let arg = 1.0 + 2.0 * diff_sq / denominator;
        // arcosh(x) = ln(x + sqrt(x^2 - 1))
        let distance = (arg + (arg * arg - 1.0).sqrt()).ln();

        distance * self.config.poincare_curvature
    }

    // Helper functions

    fn calculate_relevance(query: &[f32], results: &[Vec<f32>]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        let sum: f32 = results
            .iter()
            .map(|r| Self::cosine_similarity(query, r))
            .sum();

        sum / results.len() as f32
    }

    fn calculate_diversity(results: &[Vec<f32>]) -> f32 {
        if results.len() < 2 {
            return 1.0;
        }

        // Average pairwise distance
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let dist = Self::euclidean_distance(&results[i], &results[j]);
                total_dist += dist;
                count += 1;
            }
        }

        if count == 0 {
            return 1.0;
        }

        // Normalize to [0, 1]
        (total_dist / count as f32).min(1.0)
    }

    fn calculate_novelty(query: &[f32], history: &[&QueryRecord]) -> f32 {
        if history.is_empty() {
            return 0.5;
        }

        // Average distance to recent queries
        let avg_dist: f32 = history
            .iter()
            .map(|r| Self::euclidean_distance(query, &r.embedding))
            .sum::<f32>()
            / history.len() as f32;

        // Higher distance = more novel
        (avg_dist / 2.0).min(1.0)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_calculation() {
        let confidence = SonaConfidence::new(0.8, 0.6, 0.4);
        assert!((confidence.score - 0.67).abs() < 0.01);
    }

    #[test]
    fn test_confidence_recommendation() {
        let high = SonaConfidence::new(0.9, 0.5, 0.5);
        assert_eq!(high.recommendation(), SonaRecommendation::Accept);

        let low = SonaConfidence::new(0.3, 0.5, 0.5);
        assert_eq!(low.recommendation(), SonaRecommendation::NeedsConsolidation);
    }

    #[test]
    fn test_sona_loop_latency() {
        assert_eq!(SonaLoop::LoopA.latency_target_us(), 100);
        assert_eq!(SonaLoop::LoopB.latency_target_us(), 10_000_000);
        assert_eq!(SonaLoop::LoopC.latency_target_us(), 300_000_000);
    }

    #[test]
    fn test_feedback_creation() {
        let query_id = Uuid::new_v4();
        let results = vec![Uuid::new_v4(), Uuid::new_v4()];

        let positive = SonaFeedback::positive(query_id, results.clone());
        assert_eq!(positive.feedback_type, FeedbackType::Positive);
        assert_eq!(positive.rating, Some(1.0));

        let negative = SonaFeedback::negative(query_id, results);
        assert_eq!(negative.feedback_type, FeedbackType::Negative);
        assert_eq!(negative.rating, Some(0.0));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((SonaLearning::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((SonaLearning::cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
}
