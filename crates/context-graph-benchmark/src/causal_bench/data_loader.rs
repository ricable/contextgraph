//! Data loader for causal benchmark datasets.
//!
//! Loads ground truth pairs, queries, and multi-hop chains from JSONL files.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A benchmark pair from ground_truth.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkPair {
    pub id: String,
    pub cause_text: String,
    pub effect_text: String,
    pub direction: String,
    pub confidence: f32,
    pub mechanism: String,
    pub domain: String,
    pub hard_negative: String,
    #[serde(default)]
    pub confounder: String,
    #[serde(default)]
    pub difficulty: f32,
}

impl BenchmarkPair {
    /// Whether this pair represents a causal relationship.
    pub fn is_causal(&self) -> bool {
        self.direction != "none" && self.confidence >= 0.5
    }

    /// Whether this pair is forward-direction.
    pub fn is_forward(&self) -> bool {
        self.direction == "forward"
    }

    /// Whether this pair is backward-direction.
    pub fn is_backward(&self) -> bool {
        self.direction == "backward"
    }

    /// Whether this pair is bidirectional.
    pub fn is_bidirectional(&self) -> bool {
        self.direction == "bidirectional"
    }
}

/// A benchmark query from queries.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkQuery {
    pub id: String,
    pub query: String,
    pub expected_direction: String,
    pub expected_domain: String,
    pub expected_top1_id: String,
    pub expected_top5_ids: Vec<String>,
    #[serde(default)]
    pub is_negation: bool,
    #[serde(default)]
    pub is_multi_hop: bool,
}

/// A single hop in a causal chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainHop {
    pub cause: String,
    pub effect: String,
    pub mechanism: String,
}

/// A multi-hop causal chain from multi_hop_chains.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHopChain {
    pub id: String,
    pub hops: Vec<ChainHop>,
    pub domain: String,
    pub total_hops: usize,
}

/// Load ground truth pairs from a JSONL file.
pub fn load_ground_truth(path: &Path) -> anyhow::Result<Vec<BenchmarkPair>> {
    load_jsonl(path)
}

/// Load benchmark queries from a JSONL file.
pub fn load_queries(path: &Path) -> anyhow::Result<Vec<BenchmarkQuery>> {
    load_jsonl(path)
}

/// Load multi-hop chains from a JSONL file.
pub fn load_chains(path: &Path) -> anyhow::Result<Vec<MultiHopChain>> {
    load_jsonl(path)
}

/// Split pairs into train and held-out sets by domain.
pub fn split_by_domain(
    pairs: &[BenchmarkPair],
    held_out_domains: &[&str],
) -> (Vec<BenchmarkPair>, Vec<BenchmarkPair>) {
    let mut train = Vec::new();
    let mut held_out = Vec::new();

    for pair in pairs {
        if held_out_domains.contains(&pair.domain.as_str()) {
            held_out.push(pair.clone());
        } else {
            train.push(pair.clone());
        }
    }

    (train, held_out)
}

/// Filter pairs by direction.
pub fn filter_by_direction<'a>(
    pairs: &'a [BenchmarkPair],
    direction: &str,
) -> Vec<&'a BenchmarkPair> {
    pairs.iter().filter(|p| p.direction == direction).collect()
}

/// Filter pairs to only causal ones (direction != "none", confidence >= 0.5).
pub fn filter_causal(pairs: &[BenchmarkPair]) -> Vec<&BenchmarkPair> {
    pairs.iter().filter(|p| p.is_causal()).collect()
}

/// Filter pairs to only non-causal ones.
pub fn filter_non_causal(pairs: &[BenchmarkPair]) -> Vec<&BenchmarkPair> {
    pairs.iter().filter(|p| !p.is_causal()).collect()
}

/// Get unique domains from pairs.
pub fn unique_domains(pairs: &[BenchmarkPair]) -> Vec<String> {
    let mut domains: Vec<String> = pairs
        .iter()
        .map(|p| p.domain.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    domains.sort();
    domains
}

/// Subsample pairs for quick mode (deterministic).
pub fn subsample(pairs: &[BenchmarkPair], fraction: f32) -> Vec<BenchmarkPair> {
    let count = ((pairs.len() as f32 * fraction).ceil() as usize).max(1);
    // Take every Nth element for even distribution
    let step = (pairs.len() as f32 / count as f32).ceil() as usize;
    pairs
        .iter()
        .step_by(step.max(1))
        .take(count)
        .cloned()
        .collect()
}

/// Subsample queries for quick mode.
pub fn subsample_queries(queries: &[BenchmarkQuery], fraction: f32) -> Vec<BenchmarkQuery> {
    let count = ((queries.len() as f32 * fraction).ceil() as usize).max(1);
    let step = (queries.len() as f32 / count as f32).ceil() as usize;
    queries
        .iter()
        .step_by(step.max(1))
        .take(count)
        .cloned()
        .collect()
}

/// Generic JSONL loader.
fn load_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    use std::io::BufRead;

    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", path.display(), e))?;
    let reader = std::io::BufReader::new(file);
    let mut items = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let item: T = serde_json::from_str(trimmed).map_err(|e| {
            anyhow::anyhow!("Parse error at {}:{}: {}", path.display(), line_num + 1, e)
        })?;
        items.push(item);
    }

    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pairs() -> Vec<BenchmarkPair> {
        vec![
            BenchmarkPair {
                id: "causal_001".into(),
                cause_text: "Stress elevates cortisol".into(),
                effect_text: "Cortisol damages neurons".into(),
                direction: "forward".into(),
                confidence: 0.92,
                mechanism: "biological".into(),
                domain: "health".into(),
                hard_negative: "Hippocampus aids navigation".into(),
                confounder: "Sleep also affects memory".into(),
                difficulty: 0.3,
            },
            BenchmarkPair {
                id: "causal_221".into(),
                cause_text: "Trauma causes PTSD".into(),
                effect_text: "PTSD impairs daily functioning".into(),
                direction: "forward".into(),
                confidence: 0.88,
                mechanism: "psychological".into(),
                domain: "psychology".into(),
                hard_negative: "Therapy helps PTSD".into(),
                confounder: "Genetics affect resilience".into(),
                difficulty: 0.5,
            },
            BenchmarkPair {
                id: "causal_181".into(),
                cause_text: "The sun is a star".into(),
                effect_text: "GDP measures economic output".into(),
                direction: "none".into(),
                confidence: 0.05,
                mechanism: "none".into(),
                domain: "none".into(),
                hard_negative: String::new(),
                confounder: String::new(),
                difficulty: 0.0,
            },
        ]
    }

    #[test]
    fn test_split_by_domain() {
        let pairs = sample_pairs();
        let (train, held_out) = split_by_domain(&pairs, &["psychology"]);
        assert_eq!(train.len(), 2);
        assert_eq!(held_out.len(), 1);
        assert_eq!(held_out[0].domain, "psychology");
    }

    #[test]
    fn test_filter_by_direction() {
        let pairs = sample_pairs();
        let forward = filter_by_direction(&pairs, "forward");
        assert_eq!(forward.len(), 2);
        let none = filter_by_direction(&pairs, "none");
        assert_eq!(none.len(), 1);
    }

    #[test]
    fn test_filter_causal() {
        let pairs = sample_pairs();
        let causal = filter_causal(&pairs);
        assert_eq!(causal.len(), 2);
        let non_causal = filter_non_causal(&pairs);
        assert_eq!(non_causal.len(), 1);
    }

    #[test]
    fn test_subsample() {
        let pairs = sample_pairs();
        let sub = subsample(&pairs, 0.5);
        assert!(sub.len() >= 1);
        assert!(sub.len() <= pairs.len());
    }
}
