//! Retrieval quality metrics: P@K, R@K, MRR, NDCG, MAP.
//!
//! These metrics evaluate how well the retrieval system returns relevant documents
//! for given queries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Retrieval quality metrics for a benchmark run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    /// Precision at K (fraction of top-K results that are relevant).
    pub precision_at: HashMap<usize, f64>,

    /// Recall at K (fraction of relevant documents in top-K).
    pub recall_at: HashMap<usize, f64>,

    /// Mean Reciprocal Rank (average of 1/rank of first relevant result).
    pub mrr: f64,

    /// Normalized Discounted Cumulative Gain at K.
    pub ndcg_at: HashMap<usize, f64>,

    /// Mean Average Precision.
    pub map: f64,

    /// Number of queries used to compute these metrics.
    pub query_count: usize,
}

impl RetrievalMetrics {
    /// Create new metrics with all values.
    pub fn new(
        precision_at: HashMap<usize, f64>,
        recall_at: HashMap<usize, f64>,
        mrr: f64,
        ndcg_at: HashMap<usize, f64>,
        map: f64,
        query_count: usize,
    ) -> Self {
        Self {
            precision_at,
            recall_at,
            mrr,
            ndcg_at,
            map,
            query_count,
        }
    }

    /// Overall retrieval score (weighted combination).
    pub fn overall_score(&self) -> f64 {
        let p10 = self.precision_at.get(&10).copied().unwrap_or(0.0);
        let r10 = self.recall_at.get(&10).copied().unwrap_or(0.0);
        let ndcg10 = self.ndcg_at.get(&10).copied().unwrap_or(0.0);

        // 30% MRR + 25% P@10 + 25% R@10 + 20% NDCG@10
        0.30 * self.mrr + 0.25 * p10 + 0.25 * r10 + 0.20 * ndcg10
    }

    /// F1 score at K.
    pub fn f1_at(&self, k: usize) -> f64 {
        let p = self.precision_at.get(&k).copied().unwrap_or(0.0);
        let r = self.recall_at.get(&k).copied().unwrap_or(0.0);

        if p + r < f64::EPSILON {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// Compute Precision@K.
///
/// P@K = (number of relevant docs in top K) / K
///
/// # Arguments
/// * `retrieved` - Retrieved document IDs in ranked order
/// * `relevant` - Set of relevant document IDs (ground truth)
/// * `k` - Number of top results to consider
pub fn precision_at_k<T: Eq + std::hash::Hash>(
    retrieved: &[T],
    relevant: &std::collections::HashSet<T>,
    k: usize,
) -> f64 {
    if k == 0 {
        return 0.0;
    }

    let top_k = retrieved.iter().take(k);
    let hits = top_k.filter(|doc| relevant.contains(doc)).count();

    hits as f64 / k as f64
}

/// Compute Recall@K.
///
/// R@K = (number of relevant docs in top K) / (total relevant docs)
///
/// # Arguments
/// * `retrieved` - Retrieved document IDs in ranked order
/// * `relevant` - Set of relevant document IDs (ground truth)
/// * `k` - Number of top results to consider
pub fn recall_at_k<T: Eq + std::hash::Hash>(
    retrieved: &[T],
    relevant: &std::collections::HashSet<T>,
    k: usize,
) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let top_k = retrieved.iter().take(k);
    let hits = top_k.filter(|doc| relevant.contains(doc)).count();

    hits as f64 / relevant.len() as f64
}

/// Compute Mean Reciprocal Rank (MRR).
///
/// MRR = average over queries of 1/rank of first relevant result
///
/// # Arguments
/// * `query_results` - For each query: (retrieved docs, relevant docs)
pub fn mean_reciprocal_rank<T: Eq + std::hash::Hash>(
    query_results: &[(Vec<T>, std::collections::HashSet<T>)],
) -> f64 {
    if query_results.is_empty() {
        return 0.0;
    }

    let sum: f64 = query_results
        .iter()
        .map(|(retrieved, relevant)| {
            retrieved
                .iter()
                .position(|doc| relevant.contains(doc))
                .map(|pos| 1.0 / (pos + 1) as f64)
                .unwrap_or(0.0)
        })
        .sum();

    sum / query_results.len() as f64
}

/// Compute reciprocal rank for a single query.
pub fn reciprocal_rank<T: Eq + std::hash::Hash>(
    retrieved: &[T],
    relevant: &std::collections::HashSet<T>,
) -> f64 {
    retrieved
        .iter()
        .position(|doc| relevant.contains(doc))
        .map(|pos| 1.0 / (pos + 1) as f64)
        .unwrap_or(0.0)
}

/// Compute Normalized Discounted Cumulative Gain at K.
///
/// NDCG@K = DCG@K / IDCG@K
///
/// Where DCG@K = sum over i in 1..K of rel(i) / log2(i+1)
/// and IDCG@K is DCG@K for the ideal ranking.
///
/// # Arguments
/// * `retrieved` - Retrieved document IDs in ranked order
/// * `relevance_scores` - Map from doc ID to relevance score (higher = more relevant)
/// * `k` - Number of top results to consider
pub fn ndcg_at_k<T: Eq + std::hash::Hash + Clone>(
    retrieved: &[T],
    relevance_scores: &HashMap<T, f64>,
    k: usize,
) -> f64 {
    if k == 0 {
        return 0.0;
    }

    // Compute DCG@K
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, doc)| {
            let rel = relevance_scores.get(doc).copied().unwrap_or(0.0);
            rel / (i as f64 + 2.0).log2()
        })
        .sum();

    // Compute IDCG@K (ideal DCG with perfect ranking)
    let mut ideal_scores: Vec<f64> = relevance_scores.values().copied().collect();
    ideal_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let idcg: f64 = ideal_scores
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| rel / (i as f64 + 2.0).log2())
        .sum();

    if idcg < f64::EPSILON {
        0.0
    } else {
        dcg / idcg
    }
}

/// Compute Average Precision for a single query.
///
/// AP = (1/|R|) * sum over k of (P@k * rel(k))
///
/// Where rel(k) = 1 if document at position k is relevant.
pub fn average_precision<T: Eq + std::hash::Hash>(
    retrieved: &[T],
    relevant: &std::collections::HashSet<T>,
) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut relevant_count = 0;

    for (i, doc) in retrieved.iter().enumerate() {
        if relevant.contains(doc) {
            relevant_count += 1;
            sum += relevant_count as f64 / (i + 1) as f64;
        }
    }

    sum / relevant.len() as f64
}

/// Compute Mean Average Precision (MAP).
///
/// MAP = average over all queries of Average Precision.
pub fn mean_average_precision<T: Eq + std::hash::Hash>(
    query_results: &[(Vec<T>, std::collections::HashSet<T>)],
) -> f64 {
    if query_results.is_empty() {
        return 0.0;
    }

    let sum: f64 = query_results
        .iter()
        .map(|(retrieved, relevant)| average_precision(retrieved, relevant))
        .sum();

    sum / query_results.len() as f64
}

/// Compute all retrieval metrics for a set of queries.
pub fn compute_all_metrics<T: Eq + std::hash::Hash + Clone>(
    query_results: &[(Vec<T>, std::collections::HashSet<T>)],
    k_values: &[usize],
) -> RetrievalMetrics {
    let mut precision_at = HashMap::new();
    let mut recall_at = HashMap::new();
    let mut ndcg_at = HashMap::new();

    // For NDCG, we'll use binary relevance (1.0 for relevant, 0.0 otherwise)
    for &k in k_values {
        let mut p_sum = 0.0;
        let mut r_sum = 0.0;
        let mut ndcg_sum = 0.0;

        for (retrieved, relevant) in query_results {
            p_sum += precision_at_k(retrieved, relevant, k);
            r_sum += recall_at_k(retrieved, relevant, k);

            // Binary relevance for NDCG
            let relevance: HashMap<T, f64> =
                relevant.iter().map(|doc| (doc.clone(), 1.0)).collect();
            ndcg_sum += ndcg_at_k(retrieved, &relevance, k);
        }

        let n = query_results.len() as f64;
        precision_at.insert(k, p_sum / n);
        recall_at.insert(k, r_sum / n);
        ndcg_at.insert(k, ndcg_sum / n);
    }

    let mrr = mean_reciprocal_rank(query_results);
    let map = mean_average_precision(query_results);

    RetrievalMetrics {
        precision_at,
        recall_at,
        mrr,
        ndcg_at,
        map,
        query_count: query_results.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_precision_at_k() {
        let retrieved = vec![1, 2, 3, 4, 5];
        let relevant: HashSet<i32> = [1, 3, 5, 7, 9].into_iter().collect();

        // At K=5, 3 of 5 are relevant
        assert!((precision_at_k(&retrieved, &relevant, 5) - 0.6).abs() < 0.01);

        // At K=2, 1 of 2 is relevant
        assert!((precision_at_k(&retrieved, &relevant, 2) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_recall_at_k() {
        let retrieved = vec![1, 2, 3, 4, 5];
        let relevant: HashSet<i32> = [1, 3, 5, 7, 9].into_iter().collect();

        // At K=5, 3 of 5 relevant docs are found
        assert!((recall_at_k(&retrieved, &relevant, 5) - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_reciprocal_rank() {
        let retrieved = vec![2, 1, 3, 4, 5]; // First relevant at position 2
        let relevant: HashSet<i32> = [1, 3, 5].into_iter().collect();

        // First relevant (1) is at position 2 (0-indexed), so RR = 1/2
        assert!((reciprocal_rank(&retrieved, &relevant) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_average_precision() {
        let retrieved = vec![1, 2, 3, 4, 5];
        let relevant: HashSet<i32> = [1, 3, 5].into_iter().collect();

        // Positions 1, 3, 5 are relevant (0-indexed: 0, 2, 4)
        // AP = (1/3) * (1/1 + 2/3 + 3/5) = (1/3) * (1 + 0.667 + 0.6) = 0.756
        let ap = average_precision(&retrieved, &relevant);
        assert!((ap - 0.756).abs() < 0.01);
    }
}
