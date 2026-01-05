# TASK-L007: Cross-Space Similarity Engine

```yaml
metadata:
  id: "TASK-L007"
  title: "Cross-Space Similarity Engine"
  layer: "logic"
  priority: "P0"
  estimated_hours: 8
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-L001"  # Multi-Embedding Query Executor
    - "TASK-L002"  # Purpose Vector Computation
    - "TASK-L005"  # Per-Space HNSW Index Builder
  spec_refs:
    - "projectionplan1.md:cross-space-similarity"
    - "projectionplan2.md:multi-utl-formula"
```

## Problem Statement

Implement an engine that computes unified similarity scores across multiple embedding spaces, incorporating purpose alignment and configurable weighting strategies for the Multi-Array Teleological Fingerprint architecture.

## Context

Unlike single-space similarity, cross-space similarity must:
- Combine scores from 12 independent embedding spaces
- Weight spaces by purpose alignment
- Handle missing embeddings gracefully
- Support multiple combination strategies (fusion vs. late interaction)
- Enable explainability (which spaces contributed)

The Multi-UTL formula extends the standard UTL with:
```
L_multi = sigmoid(2.0 * (SUM_i tau_i * lambda_S * Delta_S_i) * (SUM_j tau_j * lambda_C * Delta_C_j) * w_e * cos(phi))
```

## Technical Specification

### Data Structures

```rust
/// Configuration for cross-space similarity computation
#[derive(Clone, Debug)]
pub struct CrossSpaceConfig {
    /// Weighting strategy for combining spaces
    pub weighting_strategy: WeightingStrategy,

    /// Minimum spaces required for valid similarity
    pub min_active_spaces: usize,

    /// Whether to apply purpose weighting
    pub use_purpose_weighting: bool,

    /// Fallback value for missing spaces
    pub missing_space_handling: MissingSpaceHandling,

    /// Whether to include detailed breakdown
    pub include_breakdown: bool,
}

/// Strategy for weighting embedding spaces
#[derive(Clone, Debug)]
pub enum WeightingStrategy {
    /// Equal weight to all active spaces
    Uniform,

    /// Static weights per space
    Static([f32; 12]),

    /// Weight by purpose vector alignment
    PurposeAligned,

    /// Learned weights from historical relevance
    Learned { model_id: String },

    /// Dynamic weights based on query characteristics
    QueryAdaptive,

    /// Late interaction (ColBERT-style MaxSim)
    LateInteraction,
}

/// How to handle missing embeddings
#[derive(Clone, Copy, Debug)]
pub enum MissingSpaceHandling {
    /// Skip missing spaces in aggregation
    Skip,

    /// Use zero similarity for missing
    ZeroFill,

    /// Use average of other spaces
    AverageFill,

    /// Fail if any required space is missing
    RequireAll,
}

/// Result of cross-space similarity computation
#[derive(Clone, Debug)]
pub struct CrossSpaceSimilarity {
    /// Final aggregated similarity score [0, 1]
    pub score: f32,

    /// Raw score before sigmoid (for Multi-UTL)
    pub raw_score: f32,

    /// Per-space similarity breakdown
    pub space_scores: Option<[Option<f32>; 12]>,

    /// Which spaces contributed
    pub active_spaces: u16,  // Bitmask

    /// Contribution weight per space
    pub space_weights: Option<[f32; 12]>,

    /// Purpose alignment influence
    pub purpose_contribution: Option<f32>,

    /// Confidence in the score
    pub confidence: f32,
}

/// Multi-UTL extended parameters
#[derive(Clone, Debug)]
pub struct MultiUtlParams {
    /// Per-space semantic deltas (Delta_S_i)
    pub semantic_deltas: [f32; 12],

    /// Per-space coherence deltas (Delta_C_j)
    pub coherence_deltas: [f32; 12],

    /// Per-space tau weights
    pub tau_weights: [f32; 12],

    /// Lambda for semantic term
    pub lambda_s: f32,

    /// Lambda for coherence term
    pub lambda_c: f32,

    /// Embedding weight
    pub w_e: f32,

    /// Phase angle (goal alignment)
    pub phi: f32,
}
```

### Core Trait

```rust
/// Engine for computing cross-space similarity
#[async_trait]
pub trait CrossSpaceSimilarityEngine: Send + Sync {
    /// Compute similarity between two teleological fingerprints
    async fn compute_similarity(
        &self,
        fp1: &TeleologicalFingerprint,
        fp2: &TeleologicalFingerprint,
        config: &CrossSpaceConfig,
    ) -> Result<CrossSpaceSimilarity, SimilarityError>;

    /// Compute similarity between query and multiple candidates
    async fn compute_batch(
        &self,
        query: &TeleologicalFingerprint,
        candidates: &[TeleologicalFingerprint],
        config: &CrossSpaceConfig,
    ) -> Result<Vec<CrossSpaceSimilarity>, SimilarityError>;

    /// Compute Multi-UTL score
    async fn compute_multi_utl(
        &self,
        params: &MultiUtlParams,
    ) -> f32;

    /// Update learned weights from feedback
    async fn update_weights(
        &mut self,
        query: &TeleologicalFingerprint,
        relevant_ids: &[MemoryId],
        irrelevant_ids: &[MemoryId],
    ) -> Result<(), SimilarityError>;

    /// Get current weighting model
    fn get_weights(&self, strategy: &WeightingStrategy) -> [f32; 12];

    /// Explain similarity (which spaces contributed most)
    fn explain(
        &self,
        similarity: &CrossSpaceSimilarity,
    ) -> SimilarityExplanation;
}

/// Human-readable similarity explanation
#[derive(Clone, Debug)]
pub struct SimilarityExplanation {
    pub summary: String,
    pub top_contributing_spaces: Vec<(usize, &'static str, f32)>,
    pub missing_spaces: Vec<usize>,
    pub purpose_alignment: Option<String>,
    pub confidence_factors: Vec<String>,
}
```

### Implementation

```rust
pub struct DefaultCrossSpaceEngine {
    learned_weights: Option<[f32; 12]>,
    purpose_computer: Arc<dyn PurposeVectorComputer>,
}

impl DefaultCrossSpaceEngine {
    /// Compute per-space cosine similarity
    fn compute_space_similarity(
        &self,
        emb1: Option<&Vec<f32>>,
        emb2: Option<&Vec<f32>>,
    ) -> Option<f32> {
        match (emb1, emb2) {
            (Some(e1), Some(e2)) => Some(cosine_similarity(e1, e2)),
            _ => None,
        }
    }

    /// Apply weighting strategy
    fn get_space_weights(
        &self,
        fp1: &TeleologicalFingerprint,
        fp2: &TeleologicalFingerprint,
        config: &CrossSpaceConfig,
    ) -> [f32; 12] {
        match &config.weighting_strategy {
            WeightingStrategy::Uniform => [1.0 / 12.0; 12],

            WeightingStrategy::Static(weights) => *weights,

            WeightingStrategy::PurposeAligned => {
                // Average the purpose vectors and normalize
                let mut weights = [0.0; 12];
                for i in 0..12 {
                    weights[i] = (fp1.purpose_vector.alignment[i]
                        + fp2.purpose_vector.alignment[i]) / 2.0;
                }
                normalize_weights(&mut weights);
                weights
            }

            WeightingStrategy::Learned { model_id: _ } => {
                self.learned_weights.unwrap_or([1.0 / 12.0; 12])
            }

            WeightingStrategy::QueryAdaptive => {
                // Weight by query embedding magnitude (importance)
                let mut weights = [0.0; 12];
                for (i, emb) in fp1.semantic_fingerprint.embeddings.iter().enumerate() {
                    if let Some(e) = emb {
                        weights[i] = vector_magnitude(e);
                    }
                }
                normalize_weights(&mut weights);
                weights
            }

            WeightingStrategy::LateInteraction => {
                // Late interaction uses MaxSim, weights not directly applicable
                [1.0 / 12.0; 12]
            }
        }
    }
}

#[async_trait]
impl CrossSpaceSimilarityEngine for DefaultCrossSpaceEngine {
    async fn compute_similarity(
        &self,
        fp1: &TeleologicalFingerprint,
        fp2: &TeleologicalFingerprint,
        config: &CrossSpaceConfig,
    ) -> Result<CrossSpaceSimilarity, SimilarityError> {
        // Compute per-space similarities
        let mut space_scores = [None; 12];
        let mut active_mask = 0u16;

        for i in 0..12 {
            let emb1 = fp1.semantic_fingerprint.embeddings[i].as_ref();
            let emb2 = fp2.semantic_fingerprint.embeddings[i].as_ref();

            if let Some(sim) = self.compute_space_similarity(emb1, emb2) {
                space_scores[i] = Some(sim);
                active_mask |= 1 << i;
            }
        }

        let active_count = active_mask.count_ones() as usize;

        // Check minimum spaces requirement
        if active_count < config.min_active_spaces {
            return Err(SimilarityError::InsufficientSpaces {
                required: config.min_active_spaces,
                active: active_count,
            });
        }

        // Get weights
        let weights = self.get_space_weights(fp1, fp2, config);

        // Aggregate based on strategy
        let (raw_score, score) = if matches!(config.weighting_strategy, WeightingStrategy::LateInteraction) {
            self.compute_late_interaction(fp1, fp2)?
        } else {
            self.compute_weighted_average(&space_scores, &weights, config)?
        };

        // Compute confidence
        let confidence = self.compute_confidence(active_count, &space_scores);

        Ok(CrossSpaceSimilarity {
            score,
            raw_score,
            space_scores: if config.include_breakdown { Some(space_scores) } else { None },
            active_spaces: active_mask,
            space_weights: if config.include_breakdown { Some(weights) } else { None },
            purpose_contribution: if config.use_purpose_weighting {
                Some(self.compute_purpose_contribution(fp1, fp2))
            } else {
                None
            },
            confidence,
        })
    }

    async fn compute_multi_utl(&self, params: &MultiUtlParams) -> f32 {
        // L_multi = sigmoid(2.0 * (SUM_i tau_i * lambda_S * Delta_S_i) *
        //                          (SUM_j tau_j * lambda_C * Delta_C_j) *
        //                          w_e * cos(phi))

        let semantic_sum: f32 = params.semantic_deltas.iter()
            .zip(params.tau_weights.iter())
            .map(|(delta, tau)| tau * params.lambda_s * delta)
            .sum();

        let coherence_sum: f32 = params.coherence_deltas.iter()
            .zip(params.tau_weights.iter())
            .map(|(delta, tau)| tau * params.lambda_c * delta)
            .sum();

        let raw = 2.0 * semantic_sum * coherence_sum * params.w_e * params.phi.cos();

        sigmoid(raw)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn normalize_weights(weights: &mut [f32; 12]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-L001 complete (Multi-embedding query)
- [ ] TASK-L002 complete (Purpose vector computation)
- [ ] TASK-L005 complete (HNSW indexes)

### Scope

#### In Scope

- Per-space similarity computation
- Multiple weighting strategies
- Multi-UTL formula implementation
- Late interaction (MaxSim) support
- Similarity explanation
- Weight learning from feedback

#### Out of Scope

- Retrieval pipeline orchestration (TASK-L008)
- Index operations (TASK-L005)
- Purpose computation (TASK-L002)

### Constraints

- Similarity computation < 5ms for pair
- Batch processing < 50ms for 100 candidates
- Thread-safe for concurrent use
- Deterministic for same inputs

## Pseudo Code

```
FUNCTION compute_cross_space_similarity(fp1, fp2, config):
    // Step 1: Compute per-space similarities
    space_scores = [None; 12]
    active_mask = 0

    FOR i IN 0..12:
        emb1 = fp1.semantic_fingerprint.embeddings[i]
        emb2 = fp2.semantic_fingerprint.embeddings[i]

        IF emb1 IS NOT NULL AND emb2 IS NOT NULL:
            space_scores[i] = cosine_similarity(emb1, emb2)
            active_mask |= (1 << i)

    active_count = popcount(active_mask)

    // Step 2: Check minimum requirement
    IF active_count < config.min_active_spaces:
        RETURN Error("Insufficient spaces")

    // Step 3: Get weights based on strategy
    weights = get_space_weights(fp1, fp2, config.weighting_strategy)

    // Step 4: Aggregate
    IF config.weighting_strategy == LateInteraction:
        // MaxSim: max over token similarities
        score = compute_late_interaction(fp1, fp2)
    ELSE:
        // Weighted average
        weighted_sum = 0.0
        weight_total = 0.0

        FOR i IN 0..12:
            IF space_scores[i] IS NOT NULL:
                weighted_sum += weights[i] * space_scores[i]
                weight_total += weights[i]
            ELSE:
                MATCH config.missing_space_handling:
                    Skip: CONTINUE
                    ZeroFill: weighted_sum += 0.0; weight_total += weights[i]
                    AverageFill: // Handle separately
                    RequireAll: RETURN Error("Missing required space")

        raw_score = weighted_sum / weight_total
        score = sigmoid(raw_score * 2.0)  // Scale for sigmoid

    // Step 5: Purpose contribution
    purpose_contrib = None
    IF config.use_purpose_weighting:
        // Boost by purpose alignment similarity
        purpose_sim = cosine_similarity(
            fp1.purpose_vector.alignment,
            fp2.purpose_vector.alignment
        )
        score = score * 0.7 + purpose_sim * 0.3
        purpose_contrib = purpose_sim

    // Step 6: Confidence
    confidence = compute_confidence(active_count, space_scores)

    RETURN CrossSpaceSimilarity {
        score,
        raw_score,
        space_scores: IF config.include_breakdown THEN Some(space_scores) ELSE None,
        active_spaces: active_mask,
        space_weights: IF config.include_breakdown THEN Some(weights) ELSE None,
        purpose_contribution: purpose_contrib,
        confidence
    }

FUNCTION compute_multi_utl(params):
    // Extended UTL formula
    semantic_sum = SUM(params.tau_weights[i] * params.lambda_s * params.semantic_deltas[i])
    coherence_sum = SUM(params.tau_weights[j] * params.lambda_c * params.coherence_deltas[j])

    raw = 2.0 * semantic_sum * coherence_sum * params.w_e * cos(params.phi)

    RETURN sigmoid(raw)

FUNCTION compute_late_interaction(fp1, fp2):
    // ColBERT-style MaxSim over E10 (late interaction) space
    tokens1 = fp1.semantic_fingerprint.embeddings[9]  // E10
    tokens2 = fp2.semantic_fingerprint.embeddings[9]

    IF tokens1 IS NULL OR tokens2 IS NULL:
        RETURN (0.0, 0.0)

    // MaxSim: for each query token, max similarity to any doc token
    max_sims = []
    FOR t1 IN tokens1:
        max_sim = 0.0
        FOR t2 IN tokens2:
            sim = dot_product(t1, t2)
            max_sim = max(max_sim, sim)
        max_sims.push(max_sim)

    raw = mean(max_sims)
    RETURN (raw, sigmoid(raw))
```

## Definition of Done

### Implementation Checklist

- [ ] `CrossSpaceConfig` struct
- [ ] `WeightingStrategy` enum with all variants
- [ ] `CrossSpaceSimilarity` result type
- [ ] `MultiUtlParams` for extended UTL
- [ ] `CrossSpaceSimilarityEngine` trait
- [ ] Default implementation
- [ ] Uniform weighting
- [ ] Static weighting
- [ ] Purpose-aligned weighting
- [ ] Late interaction (MaxSim)
- [ ] Multi-UTL formula
- [ ] Similarity explanation
- [ ] Weight learning interface

### Testing Requirements

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_uniform_weighting() {
        let engine = DefaultCrossSpaceEngine::new();
        let fp1 = create_test_fingerprint();
        let fp2 = create_similar_fingerprint(&fp1, 0.8);

        let config = CrossSpaceConfig {
            weighting_strategy: WeightingStrategy::Uniform,
            ..Default::default()
        };

        let sim = engine.compute_similarity(&fp1, &fp2, &config).await.unwrap();

        assert!(sim.score >= 0.0 && sim.score <= 1.0);
        assert!(sim.score > 0.5); // Should be similar
    }

    #[tokio::test]
    async fn test_purpose_weighting() {
        let engine = DefaultCrossSpaceEngine::new();
        let fp1 = create_fingerprint_with_purpose([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let fp2 = create_fingerprint_with_purpose([0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);

        let config = CrossSpaceConfig {
            weighting_strategy: WeightingStrategy::PurposeAligned,
            use_purpose_weighting: true,
            ..Default::default()
        };

        let sim = engine.compute_similarity(&fp1, &fp2, &config).await.unwrap();

        // Purpose contribution should be high
        assert!(sim.purpose_contribution.unwrap() > 0.7);
    }

    #[tokio::test]
    async fn test_multi_utl() {
        let engine = DefaultCrossSpaceEngine::new();

        let params = MultiUtlParams {
            semantic_deltas: [0.1; 12],
            coherence_deltas: [0.1; 12],
            tau_weights: [1.0; 12],
            lambda_s: 1.0,
            lambda_c: 1.0,
            w_e: 1.0,
            phi: 0.0,  // cos(0) = 1
        };

        let score = engine.compute_multi_utl(&params).await;

        assert!(score >= 0.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_missing_space_handling() {
        let engine = DefaultCrossSpaceEngine::new();
        let fp1 = create_sparse_fingerprint([0, 1, 2]); // Only 3 spaces
        let fp2 = create_sparse_fingerprint([0, 1, 2]);

        let config = CrossSpaceConfig {
            min_active_spaces: 3,
            missing_space_handling: MissingSpaceHandling::Skip,
            ..Default::default()
        };

        let sim = engine.compute_similarity(&fp1, &fp2, &config).await.unwrap();
        assert_eq!(sim.active_spaces.count_ones(), 3);
    }

    #[tokio::test]
    async fn test_batch_computation() {
        let engine = DefaultCrossSpaceEngine::new();
        let query = create_test_fingerprint();
        let candidates: Vec<_> = (0..100)
            .map(|i| create_varied_fingerprint(i))
            .collect();

        let config = CrossSpaceConfig::default();

        let results = engine.compute_batch(&query, &candidates, &config).await.unwrap();

        assert_eq!(results.len(), 100);
    }

    #[tokio::test]
    async fn test_explanation() {
        let engine = DefaultCrossSpaceEngine::new();
        let fp1 = create_test_fingerprint();
        let fp2 = create_similar_fingerprint(&fp1, 0.8);

        let config = CrossSpaceConfig {
            include_breakdown: true,
            ..Default::default()
        };

        let sim = engine.compute_similarity(&fp1, &fp2, &config).await.unwrap();
        let explanation = engine.explain(&sim);

        assert!(!explanation.summary.is_empty());
        assert!(!explanation.top_contributing_spaces.is_empty());
    }
}
```

### Verification Commands

```bash
# Run unit tests
cargo test -p context-graph-core cross_space_similarity

# Benchmark computation
cargo bench -p context-graph-core -- cross_space

# Test with different weighting strategies
cargo test -p context-graph-core weighting_strategy -- --nocapture
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-core/src/similarity/mod.rs` | Similarity module |
| `crates/context-graph-core/src/similarity/cross_space.rs` | CrossSpaceSimilarityEngine |
| `crates/context-graph-core/src/similarity/weighting.rs` | Weighting strategies |
| `crates/context-graph-core/src/similarity/multi_utl.rs` | Multi-UTL implementation |
| `crates/context-graph-core/src/similarity/explanation.rs` | Similarity explanation |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-core/src/lib.rs` | Add `pub mod similarity` |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| Cross-space similarity | projectionplan1.md:cross-space | Complete |
| Multi-UTL formula | projectionplan2.md:multi-utl | Complete |
| Purpose weighting | projectionplan1.md:purpose | Complete |
| Late interaction | projectionplan2.md:colbert | Complete |

---

*Task created: 2026-01-04*
*Layer: Logic*
*Priority: P0 - Core similarity computation*
