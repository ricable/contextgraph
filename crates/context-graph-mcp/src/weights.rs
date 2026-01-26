//! Weight profile configuration for multi-embedding search.
//!
//! TASK-S002: Provides predefined weight profiles for 13 embedding spaces.
//!
//! # 13 Embedding Spaces (NOT 12)
//!
//! | Index | Name | Purpose |
//! |-------|------|---------|
//! | 0 | E1_Semantic | General semantic similarity |
//! | 1 | E2_Temporal_Recent | Recent time proximity |
//! | 2 | E3_Temporal_Periodic | Recurring patterns |
//! | 3 | E4_Temporal_Positional | Document position encoding |
//! | 4 | E5_Causal | Cause-effect relationships |
//! | 5 | E6_Sparse | Keyword-level matching |
//! | 6 | E7_Code | Source code similarity |
//! | 7 | E8_Graph | Node2Vec structural |
//! | 8 | E9_HDC | Hyperdimensional computing |
//! | 9 | E10_Multimodal | Cross-modal alignment |
//! | 10 | E11_Entity | Named entity matching |
//! | 11 | E12_Late_Interaction | ColBERT-style token matching |
//! | 12 | E13_SPLADE | Sparse learned expansion (Stage 1) |
//!
//! # Error Handling
//!
//! FAIL FAST: Invalid weights return detailed error immediately.

use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

/// Predefined weight profiles per query type.
///
/// Each profile has 13 weights corresponding to:
/// [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13]
///
/// # IMPORTANT: Temporal Embedders (E2-E4)
///
/// Per AP-71 and research findings ([Pinecone Cascading Retrieval](https://www.pinecone.io/blog/cascading-retrieval/),
/// [ACM TOIS Fusion](https://dl.acm.org/doi/10.1145/3596512)):
///
/// **Temporal embedders (E2-E4) have weight 0.0 in semantic search profiles.**
///
/// Temporal proximity ≠ topical similarity. Documents created at the same time
/// are NOT necessarily on the same topic. Time should be a post-retrieval boost,
/// not a similarity measure.
///
/// # Profile Categories
///
/// - **Semantic Profiles** (E2-E4 = 0.0): semantic_search, code_search, causal_reasoning, fact_checking
/// - **Special Profiles**: temporal_navigation (for explicit time-based queries)
/// - **Category-Weighted**: category_weighted (constitution-compliant)
///
/// # IMPORTANT: Pipeline-Stage Embedders (E12, E13) - per ARCH-13
///
/// E12 (Late Interaction) and E13 (SPLADE) have weight 0.0 in ALL semantic scoring profiles
/// because they're used in specific pipeline stages, NOT for similarity scoring:
/// - E13: Stage 1 recall ONLY (inverted index) - per AP-74
/// - E12: Stage 3 re-ranking ONLY (MaxSim) - per AP-73
///
/// This ensures these embedders contribute through their proper roles in the pipeline,
/// not through score fusion which would misuse their specialized capabilities.
pub const WEIGHT_PROFILES: &[(&str, [f32; NUM_EMBEDDERS])] = &[
    // =========================================================================
    // SEMANTIC PROFILES - Temporal (E2-E4) = 0.0 per AP-71
    // =========================================================================

    // Semantic Search: General queries - E1 primary, E5/E7/E10 supporting
    // Sum of weights = 1.0 (excluding temporal which are 0.0)
    // E9 has minimal backup weight (0.02) for typo tolerance
    (
        "semantic_search",
        [
            0.33, // E1_Semantic (primary, reduced from 0.35 to make room for E9)
            0.0,  // E2_Temporal_Recent - NOT for semantic search
            0.0,  // E3_Temporal_Periodic - NOT for semantic search
            0.0,  // E4_Temporal_Positional - NOT for semantic search
            0.15, // E5_Causal
            0.05, // E6_Sparse (keyword backup)
            0.20, // E7_Code
            0.05, // E8_Graph (relational)
            0.02, // E9_HDC (noise-robust backup for typo tolerance)
            0.15, // E10_Multimodal
            0.05, // E11_Entity (relational)
            0.0,  // E12_Late_Interaction (Stage 3 rerank only)
            0.0,  // E13_SPLADE (Stage 1 recall only)
        ],
    ),

    // Causal Reasoning: "Why" questions - E5 primary
    (
        "causal_reasoning",
        [
            0.20, // E1_Semantic
            0.0,  // E2_Temporal_Recent - NOT for semantic search
            0.0,  // E3_Temporal_Periodic - NOT for semantic search
            0.0,  // E4_Temporal_Positional - NOT for semantic search
            0.45, // E5_Causal (primary)
            0.05, // E6_Sparse
            0.10, // E7_Code
            0.10, // E8_Graph (causal chains)
            0.0,  // E9_HDC
            0.05, // E10_Multimodal
            0.05, // E11_Entity
            0.0,  // E12_Late_Interaction
            0.0,  // E13_SPLADE
        ],
    ),

    // Code Search: Programming queries - E7 primary
    (
        "code_search",
        [
            0.20, // E1_Semantic
            0.0,  // E2_Temporal_Recent - NOT for semantic search
            0.0,  // E3_Temporal_Periodic - NOT for semantic search
            0.0,  // E4_Temporal_Positional - Use for line numbers via recency_boost instead
            0.10, // E5_Causal
            0.10, // E6_Sparse (keywords)
            0.40, // E7_Code (primary)
            0.0,  // E8_Graph
            0.0,  // E9_HDC
            0.10, // E10_Multimodal
            0.10, // E11_Entity (function names, etc.)
            0.0,  // E12_Late_Interaction
            0.0,  // E13_SPLADE
        ],
    ),

    // Fact Checking: Entity/fact queries - E11 primary, E6 for keywords
    (
        "fact_checking",
        [
            0.15, // E1_Semantic
            0.0,  // E2_Temporal_Recent - NOT for semantic search
            0.0,  // E3_Temporal_Periodic - NOT for semantic search
            0.0,  // E4_Temporal_Positional - NOT for semantic search
            0.15, // E5_Causal
            0.15, // E6_Sparse (keyword match)
            0.05, // E7_Code
            0.05, // E8_Graph
            0.0,  // E9_HDC
            0.05, // E10_Multimodal
            0.40, // E11_Entity (primary - named entities)
            0.0,  // E12_Late_Interaction
            0.0,  // E13_SPLADE
        ],
    ),

    // Intent Search: "What was the goal?" queries - E1 primary, E10 via multiplicative boost
    // Use for intent-aware retrieval: "what work had the same goal?",
    // "find memories with similar purpose", "what was trying to be accomplished?"
    //
    // ARCH-17 COMPLIANT: E10 is set to 0.0 in weighted fusion because E10 now operates
    // via multiplicative boost POST-RETRIEVAL, not in initial candidate scoring.
    // This prevents E10 from competing with E1 in fusion (it ENHANCES E1 instead).
    (
        "intent_search",
        [
            0.55, // E1_Semantic (foundation per ARCH-12, boosted - E10 weight redistributed)
            0.0,  // E2_Temporal_Recent - NOT for semantic search per AP-71
            0.0,  // E3_Temporal_Periodic - NOT for semantic search per AP-71
            0.0,  // E4_Temporal_Positional - NOT for semantic search per AP-71
            0.12, // E5_Causal (intent often has causal structure)
            0.05, // E6_Sparse (keyword backup)
            0.18, // E7_Code (code intent/purpose, boosted)
            0.05, // E8_Graph (relational)
            0.0,  // E9_HDC
            0.0,  // E10_Multimodal - NOW VIA MULTIPLICATIVE BOOST (ARCH-17)
            0.05, // E11_Entity (entities in intent)
            0.0,  // E12_Late_Interaction (Stage 3 rerank only per AP-73)
            0.0,  // E13_SPLADE (Stage 1 recall only per AP-74)
        ],
    ),

    // Intent Enhanced: E1 primary foundation, E10 via stronger multiplicative boost
    // Use when intentMode != "none" in search_graph for asymmetric E10 reranking
    //
    // ARCH-17 COMPLIANT: E10 is set to 0.0 in weighted fusion because E10 now operates
    // via multiplicative boost POST-RETRIEVAL with IntentBoostConfig::aggressive().
    // E1 weight increased to maintain strong semantic foundation.
    (
        "intent_enhanced",
        [
            0.55, // E1_Semantic (foundation per ARCH-12, boosted - E10 weight redistributed)
            0.0,  // E2_Temporal_Recent - NOT for semantic search per AP-71
            0.0,  // E3_Temporal_Periodic - NOT for semantic search per AP-71
            0.0,  // E4_Temporal_Positional - NOT for semantic search per AP-71
            0.12, // E5_Causal (intent often has causal structure)
            0.05, // E6_Sparse (keyword backup)
            0.18, // E7_Code (code intent/purpose, boosted)
            0.05, // E8_Graph (relational)
            0.0,  // E9_HDC
            0.0,  // E10_Multimodal - NOW VIA MULTIPLICATIVE BOOST (ARCH-17)
            0.05, // E11_Entity (entities in intent)
            0.0,  // E12_Late_Interaction (Stage 3 rerank only per AP-73)
            0.0,  // E13_SPLADE (Stage 1 recall only per AP-74)
        ],
    ),

    // =========================================================================
    // SPECIAL PROFILES
    // =========================================================================

    // Temporal Navigation: EXPLICIT time-based queries only
    // Use when user explicitly asks for "recent" or "from last week"
    // NOT for general semantic search
    (
        "temporal_navigation",
        [
            0.12, // E1_Semantic
            0.22, // E2_Temporal_Recent (primary)
            0.22, // E3_Temporal_Periodic (primary)
            0.22, // E4_Temporal_Positional (primary)
            0.03, // E5_Causal
            0.02, // E6_Sparse
            0.03, // E7_Code
            0.02, // E8_Graph
            0.03, // E9_HDC
            0.03, // E10_Multimodal
            0.02, // E11_Entity
            0.02, // E12_Late_Interaction
            0.02, // E13_SPLADE
        ],
    ),

    // =========================================================================
    // SEQUENCE NAVIGATION PROFILES - E4 focused
    // =========================================================================

    // Sequence Navigation: For explicit sequence traversal queries
    // Use for: "What did we discuss before X?", "Previous message", "Next turn"
    // E4 (V_ordering) is PRIMARY - 55% weight for document/conversation ordering
    // Research: MemoriesDB (2025), TG-RAG temporal ordering
    (
        "sequence_navigation",
        [
            0.20, // E1_Semantic (semantic backup)
            0.05, // E2_Temporal_Recent (mild recency signal)
            0.0,  // E3_Temporal_Periodic (no periodic patterns for sequence)
            0.55, // E4_Temporal_Positional (PRIMARY - sequence ordering)
            0.03, // E5_Causal
            0.02, // E6_Sparse
            0.03, // E7_Code
            0.02, // E8_Graph
            0.03, // E9_HDC
            0.03, // E10_Multimodal
            0.02, // E11_Entity
            0.0,  // E12_Late_Interaction (pipeline stage only)
            0.02, // E13_SPLADE
        ],
    ),

    // Conversation History: Balanced E4 + E1 for contextual recall
    // Use for: "What did we discuss about X?", "Earlier in our conversation"
    // Combines semantic similarity with sequence ordering
    // Research: Memoria Framework session-level + Episodic Memory RAG
    (
        "conversation_history",
        [
            0.30, // E1_Semantic (topic matching)
            0.05, // E2_Temporal_Recent (recent context helps)
            0.0,  // E3_Temporal_Periodic
            0.35, // E4_Temporal_Positional (conversation ordering)
            0.10, // E5_Causal (causal chains in conversation)
            0.03, // E6_Sparse
            0.05, // E7_Code
            0.02, // E8_Graph
            0.0,  // E9_HDC
            0.05, // E10_Multimodal
            0.03, // E11_Entity
            0.0,  // E12_Late_Interaction (pipeline stage only)
            0.02, // E13_SPLADE
        ],
    ),

    // Category-Weighted: Constitution-compliant weights per CLAUDE.md and ARCH-13
    // SEMANTIC (E1,E5,E6,E7,E10): weight 1.0
    // TEMPORAL (E2,E3,E4): weight 0.0 per AP-60
    // RELATIONAL (E8,E11): weight 0.5
    // STRUCTURAL (E9): weight 0.5
    // PIPELINE-STAGE (E12,E13): weight 0.0 per ARCH-13 (used in pipeline stages, not scoring)
    // max_weighted_agreement = 6.5 (5*1.0 + 3*0.5 = 6.5)
    (
        "category_weighted",
        [
            1.0 / 6.5,   // E1_Semantic (SEMANTIC)
            0.0,         // E2_Temporal_Recent (TEMPORAL - excluded per AP-60)
            0.0,         // E3_Temporal_Periodic (TEMPORAL - excluded per AP-60)
            0.0,         // E4_Temporal_Positional (TEMPORAL - excluded per AP-60)
            1.0 / 6.5,   // E5_Causal (SEMANTIC)
            1.0 / 6.5,   // E6_Sparse (SEMANTIC)
            1.0 / 6.5,   // E7_Code (SEMANTIC)
            0.5 / 6.5,   // E8_Graph (RELATIONAL)
            0.5 / 6.5,   // E9_HDC (STRUCTURAL)
            1.0 / 6.5,   // E10_Multimodal (SEMANTIC)
            0.5 / 6.5,   // E11_Entity (RELATIONAL)
            0.0,         // E12_Late_Interaction (PIPELINE-STAGE - rerank only per AP-73)
            0.0,         // E13_SPLADE (PIPELINE-STAGE - recall only per AP-74)
        ],
    ),

    // =========================================================================
    // TYPO-TOLERANT PROFILE - E9 primary for noisy queries
    // =========================================================================

    // Typo Tolerant: For queries with potential spelling errors or variations
    // Use when input may contain typos, misspellings, or character-level variations.
    // E9 (HDC) uses character trigrams which preserve similarity despite spelling errors.
    // Example: "authetication" matches "authentication" via character overlap.
    (
        "typo_tolerant",
        [
            0.30, // E1_Semantic (reduced - query might be noisy)
            0.0,  // E2_Temporal_Recent - NOT for semantic search
            0.0,  // E3_Temporal_Periodic - NOT for semantic search
            0.0,  // E4_Temporal_Positional - NOT for semantic search
            0.10, // E5_Causal
            0.05, // E6_Sparse (keyword backup for exact matches)
            0.15, // E7_Code (reduced to make room for E9)
            0.03, // E8_Graph (relational)
            0.15, // E9_HDC (PRIMARY for typo tolerance)
            0.12, // E10_Multimodal
            0.05, // E11_Entity (relational)
            0.03, // E12_Late_Interaction (can help with phrase matching)
            0.02, // E13_SPLADE (term expansion helps with variations)
        ],
    ),

    // =========================================================================
    // PIPELINE-AWARE PROFILES - Phase 5 E12/E13 Integration
    // =========================================================================

    // Pipeline Stage 1 Recall: E13-heavy for sparse retrieval
    // Use in SearchStrategy::Pipeline Stage 1 for maximum recall.
    // E13 SPLADE provides term expansion (fast→quick) and sparse retrieval.
    // Per ARCH-13/AP-74: E13 is for Stage 1 recall ONLY.
    //
    // This profile is used internally by the pipeline, not directly by users.
    // The weights are designed for candidate generation, not final ranking.
    (
        "pipeline_stage1_recall",
        [
            0.20, // E1_Semantic (backup for semantic overlap)
            0.0,  // E2_Temporal_Recent - NOT for recall stage
            0.0,  // E3_Temporal_Periodic - NOT for recall stage
            0.0,  // E4_Temporal_Positional - NOT for recall stage
            0.05, // E5_Causal (minimal)
            0.25, // E6_Sparse (keyword matching, supports E13)
            0.10, // E7_Code (for code queries)
            0.0,  // E8_Graph
            0.05, // E9_HDC (typo tolerance helps recall)
            0.05, // E10_Multimodal
            0.05, // E11_Entity (entity names)
            0.0,  // E12_Late_Interaction (Stage 3 rerank only per AP-73)
            0.25, // E13_SPLADE (PRIMARY - term expansion for recall)
        ],
    ),

    // Pipeline Stage 2 Scoring: E1-heavy for dense candidate scoring
    // Use in SearchStrategy::Pipeline Stage 2 after E13 recall.
    // E1 provides the semantic foundation for scoring retrieved candidates.
    // Per ARCH-12: E1 is THE semantic foundation.
    //
    // This profile is used internally by the pipeline, not directly by users.
    // E12 (ColBERT) is NOT included here - it's applied via MaxSim in Stage 3.
    (
        "pipeline_stage2_scoring",
        [
            0.50, // E1_Semantic (PRIMARY - semantic foundation per ARCH-12)
            0.0,  // E2_Temporal_Recent - NOT for scoring stage
            0.0,  // E3_Temporal_Periodic - NOT for scoring stage
            0.0,  // E4_Temporal_Positional - NOT for scoring stage
            0.12, // E5_Causal (causal relationships)
            0.05, // E6_Sparse (keyword precision)
            0.15, // E7_Code (code understanding)
            0.03, // E8_Graph (relational)
            0.02, // E9_HDC (noise tolerance)
            0.08, // E10_Multimodal (intent alignment via boost)
            0.05, // E11_Entity (entity matching)
            0.0,  // E12_Late_Interaction (Stage 3 rerank only per AP-73)
            0.0,  // E13_SPLADE (Stage 1 recall only per AP-74)
        ],
    ),

    // Pipeline Full: Combined profile for complete pipeline execution
    // Use when running full E13→E1→E12 pipeline but need fusion weights.
    // This balances recall (E6/E13) with precision (E1/E7).
    // E12 is applied separately via MaxSim reranking.
    (
        "pipeline_full",
        [
            0.40, // E1_Semantic (strong foundation)
            0.0,  // E2_Temporal_Recent - NOT for pipeline
            0.0,  // E3_Temporal_Periodic - NOT for pipeline
            0.0,  // E4_Temporal_Positional - NOT for pipeline
            0.10, // E5_Causal
            0.10, // E6_Sparse (keyword precision)
            0.15, // E7_Code
            0.03, // E8_Graph
            0.02, // E9_HDC
            0.08, // E10_Multimodal
            0.05, // E11_Entity
            0.0,  // E12_Late_Interaction (applied via MaxSim, not fusion)
            0.07, // E13_SPLADE (mild weight for fusion awareness)
        ],
    ),

    // Balanced: Equal weights across all 13 spaces (for testing/comparison)
    // NOTE: This is NOT recommended for production - temporal affects results
    (
        "balanced",
        [
            0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077,
            0.076, // E13 slightly lower to sum to 1.0
        ],
    ),
];

/// Get weight profile by name.
///
/// # Arguments
/// * `name` - Profile name (e.g., "semantic_search", "code_search")
///
/// # Returns
/// The 13-element weight array if found, None otherwise.
pub fn get_weight_profile(name: &str) -> Option<[f32; NUM_EMBEDDERS]> {
    WEIGHT_PROFILES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, w)| *w)
}

/// Get all available profile names.
pub(crate) fn get_profile_names() -> Vec<&'static str> {
    WEIGHT_PROFILES.iter().map(|(n, _)| *n).collect()
}

/// Validate that weights sum to ~1.0 and all are in [0.0, 1.0].
///
/// # FAIL FAST
/// Returns detailed error on validation failure.
pub(crate) fn validate_weights(weights: &[f32; NUM_EMBEDDERS]) -> Result<(), WeightValidationError> {
    // Check each weight is in range
    for (i, &w) in weights.iter().enumerate() {
        if !(0.0..=1.0).contains(&w) {
            return Err(WeightValidationError::OutOfRange {
                space_index: i,
                space_name: space_name(i),
                value: w,
            });
        }
    }

    // Check sum is ~1.0
    let sum: f32 = weights.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(WeightValidationError::InvalidSum {
            expected: 1.0,
            actual: sum,
            weights: weights.to_vec(),
        });
    }

    Ok(())
}

/// Get space name by index.
pub(crate) fn space_name(idx: usize) -> &'static str {
    match idx {
        0 => "E1_Semantic",
        1 => "E2_Temporal_Recent",
        2 => "E3_Temporal_Periodic",
        3 => "E4_Temporal_Positional",
        4 => "E5_Causal",
        5 => "E6_Sparse",
        6 => "E7_Code",
        7 => "E8_Graph",
        8 => "E9_HDC",
        9 => "E10_Multimodal",
        10 => "E11_Entity",
        11 => "E12_Late_Interaction",
        12 => "E13_SPLADE",
        _ => "Unknown",
    }
}

/// Get snake_case key name for JSON serialization.
pub(crate) fn space_json_key(idx: usize) -> &'static str {
    match idx {
        0 => "e1_semantic",
        1 => "e2_temporal_recent",
        2 => "e3_temporal_periodic",
        3 => "e4_temporal_positional",
        4 => "e5_causal",
        5 => "e6_sparse",
        6 => "e7_code",
        7 => "e8_graph",
        8 => "e9_hdc",
        9 => "e10_multimodal",
        10 => "e11_entity",
        11 => "e12_late_interaction",
        12 => "e13_splade",
        _ => "unknown",
    }
}

/// Weight validation error.
///
/// Provides detailed context for FAIL FAST error handling.
#[derive(Debug, Clone)]
pub(crate) enum WeightValidationError {
    /// A weight is outside the valid range [0.0, 1.0].
    OutOfRange {
        space_index: usize,
        space_name: &'static str,
        value: f32,
    },
    /// Weights do not sum to 1.0.
    InvalidSum {
        expected: f32,
        actual: f32,
        weights: Vec<f32>,
    },
    /// Wrong number of weights provided.
    WrongCount { expected: usize, actual: usize },
    /// Invalid weight value (not a number).
    InvalidValue {
        index: usize,
        space_name: &'static str,
        value: serde_json::Value,
    },
}

impl std::fmt::Display for WeightValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfRange {
                space_index,
                space_name,
                value,
            } => {
                write!(
                    f,
                    "Weight for space {} ({}) is out of range [0.0, 1.0]: {}",
                    space_index, space_name, value
                )
            }
            Self::InvalidSum {
                expected,
                actual,
                weights,
            } => {
                write!(
                    f,
                    "Weights must sum to {}, got {}. Weights: {:?}",
                    expected, actual, weights
                )
            }
            Self::WrongCount { expected, actual } => {
                write!(f, "Expected {} weights, got {}", expected, actual)
            }
            Self::InvalidValue { index, space_name, value } => {
                write!(
                    f,
                    "Invalid weight at index {} ({}): {:?} is not a number",
                    index, space_name, value
                )
            }
        }
    }
}

impl std::error::Error for WeightValidationError {}

/// Parse weights from JSON array.
///
/// # Arguments
/// * `arr` - JSON array of 13 numeric weights
///
/// # Returns
/// Validated weight array.
///
/// # Errors (FAIL FAST)
/// - `WrongCount`: Array has wrong number of elements
/// - `InvalidValue`: A value is not a number (NO SILENT 0.0 FALLBACK)
/// - `OutOfRange`: A weight is outside [0.0, 1.0]
/// - `InvalidSum`: Weights don't sum to 1.0
pub(crate) fn parse_weights_from_json(
    arr: &[serde_json::Value],
) -> Result<[f32; NUM_EMBEDDERS], WeightValidationError> {
    if arr.len() != NUM_EMBEDDERS {
        return Err(WeightValidationError::WrongCount {
            expected: NUM_EMBEDDERS,
            actual: arr.len(),
        });
    }

    let mut weights = [0.0f32; NUM_EMBEDDERS];
    for (i, v) in arr.iter().enumerate() {
        // FAIL FAST: Reject non-numeric values instead of silently using 0.0
        weights[i] = v.as_f64()
            .ok_or_else(|| WeightValidationError::InvalidValue {
                index: i,
                space_name: space_name(i),
                value: v.clone(),
            })? as f32;
    }

    validate_weights(&weights)?;
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_profiles_count() {
        assert!(
            WEIGHT_PROFILES.len() >= 6,
            "Should have at least 6 predefined profiles"
        );
        println!(
            "[VERIFIED] WEIGHT_PROFILES has {} profiles",
            WEIGHT_PROFILES.len()
        );
    }

    #[test]
    fn test_all_profiles_sum_to_one() {
        for (name, weights) in WEIGHT_PROFILES {
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Profile '{}' weights sum to {} (expected ~1.0)",
                name,
                sum
            );
            println!("[VERIFIED] Profile '{}' sums to {:.4}", name, sum);
        }
    }

    #[test]
    fn test_all_profiles_have_13_weights() {
        for (name, weights) in WEIGHT_PROFILES {
            assert_eq!(
                weights.len(),
                13,
                "Profile '{}' should have 13 weights",
                name
            );
        }
        println!("[VERIFIED] All profiles have exactly 13 weights");
    }

    #[test]
    fn test_get_weight_profile() {
        let semantic = get_weight_profile("semantic_search");
        assert!(semantic.is_some(), "semantic_search profile should exist");
        assert!(
            (semantic.unwrap()[0] - 0.33).abs() < 0.001,
            "E1 should be 0.33 in semantic_search profile"
        );

        let missing = get_weight_profile("nonexistent");
        assert!(missing.is_none(), "Unknown profile should return None");

        println!("[VERIFIED] get_weight_profile works correctly");
    }

    #[test]
    fn test_typo_tolerant_profile_exists() {
        let weights = get_weight_profile("typo_tolerant");
        assert!(weights.is_some(), "typo_tolerant profile should exist");
        println!("[VERIFIED] typo_tolerant profile exists");
    }

    #[test]
    fn test_typo_tolerant_e9_is_primary() {
        // E9 should have significant weight in typo_tolerant profile
        let weights = get_weight_profile("typo_tolerant").unwrap();

        // E9 should be >= 0.10 (substantial contribution)
        assert!(
            weights[8] >= 0.10,
            "E9 should be >= 0.10 in typo_tolerant (got {})",
            weights[8]
        );

        // E9 should be one of the highest weighted (top 3)
        let mut indexed_weights: Vec<(usize, f32)> = weights.iter().cloned().enumerate().collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_3_indices: Vec<usize> = indexed_weights.iter().take(3).map(|(i, _)| *i).collect();
        assert!(
            top_3_indices.contains(&8),
            "E9 (index 8) should be in top 3 weights for typo_tolerant. Top 3: {:?}",
            top_3_indices
        );

        println!(
            "[VERIFIED] typo_tolerant has E9={:.2} as primary structural embedder",
            weights[8]
        );
    }

    #[test]
    fn test_typo_tolerant_temporal_excluded() {
        // Temporal embedders should still be excluded per AP-71
        let weights = get_weight_profile("typo_tolerant").unwrap();

        assert_eq!(
            weights[1], 0.0,
            "E2 should be 0.0 in typo_tolerant per AP-71"
        );
        assert_eq!(
            weights[2], 0.0,
            "E3 should be 0.0 in typo_tolerant per AP-71"
        );
        assert_eq!(
            weights[3], 0.0,
            "E4 should be 0.0 in typo_tolerant per AP-71"
        );

        println!("[VERIFIED] typo_tolerant excludes temporal embedders (E2-E4)");
    }

    #[test]
    fn test_semantic_search_has_e9_backup() {
        // semantic_search should have minimal E9 weight for typo backup
        let weights = get_weight_profile("semantic_search").unwrap();

        assert!(
            weights[8] > 0.0,
            "E9 should have non-zero weight in semantic_search for typo backup"
        );
        assert!(
            weights[8] <= 0.05,
            "E9 should have minimal weight (<=0.05) in semantic_search, got {}",
            weights[8]
        );

        println!(
            "[VERIFIED] semantic_search has E9={:.2} as backup",
            weights[8]
        );
    }

    #[test]
    fn test_temporal_embedders_excluded_from_semantic_profiles() {
        // Per AP-71: Temporal embedders (E2-E4) MUST NOT be used in similarity scoring
        // All semantic search profiles should have E2-E4 = 0.0

        let semantic_profiles = ["semantic_search", "causal_reasoning", "code_search", "fact_checking", "category_weighted", "intent_search", "intent_enhanced", "typo_tolerant"];

        for profile_name in semantic_profiles {
            let weights = get_weight_profile(profile_name).expect(&format!(
                "Profile '{}' should exist",
                profile_name
            ));

            assert_eq!(
                weights[1], 0.0,
                "E2 (temporal recent) should be 0.0 in '{}' profile per AP-71",
                profile_name
            );
            assert_eq!(
                weights[2], 0.0,
                "E3 (temporal periodic) should be 0.0 in '{}' profile per AP-71",
                profile_name
            );
            assert_eq!(
                weights[3], 0.0,
                "E4 (temporal positional) should be 0.0 in '{}' profile per AP-71",
                profile_name
            );

            println!(
                "[VERIFIED] Profile '{}' has temporal embedders (E2-E4) = 0.0",
                profile_name
            );
        }
    }

    #[test]
    fn test_pipeline_stage_embedders_excluded_from_semantic_profiles() {
        // Per ARCH-13: E12 (rerank-only) and E13 (recall-only) MUST be 0.0 in semantic scoring profiles
        // E12 is used in Stage 3 re-ranking (ColBERT MaxSim) per AP-73
        // E13 is used in Stage 1 recall (SPLADE inverted index) per AP-74

        let semantic_profiles = ["semantic_search", "causal_reasoning", "code_search", "fact_checking", "category_weighted", "intent_search", "intent_enhanced"];

        for profile_name in semantic_profiles {
            let weights = get_weight_profile(profile_name).expect(&format!(
                "Profile '{}' should exist",
                profile_name
            ));

            assert_eq!(
                weights[11], 0.0,
                "E12 (late interaction) should be 0.0 in '{}' profile per ARCH-13 (rerank-only)",
                profile_name
            );
            assert_eq!(
                weights[12], 0.0,
                "E13 (SPLADE) should be 0.0 in '{}' profile per ARCH-13 (recall-only)",
                profile_name
            );

            println!(
                "[VERIFIED] Profile '{}' has pipeline-stage embedders (E12-E13) = 0.0 per ARCH-13",
                profile_name
            );
        }
    }

    #[test]
    fn test_category_weighted_profile() {
        // Per constitution and ARCH-13:
        // SEMANTIC (E1,E5,E6,E7,E10) = 1.0
        // TEMPORAL (E2-E4) = 0.0 per AP-60
        // RELATIONAL (E8,E11) = 0.5
        // STRUCTURAL (E9) = 0.5
        // PIPELINE-STAGE (E12,E13) = 0.0 per ARCH-13
        let weights = get_weight_profile("category_weighted").expect("category_weighted should exist");

        // Temporal (E2-E4) = 0.0 per AP-60
        assert_eq!(weights[1], 0.0, "E2 should be 0.0 (TEMPORAL)");
        assert_eq!(weights[2], 0.0, "E3 should be 0.0 (TEMPORAL)");
        assert_eq!(weights[3], 0.0, "E4 should be 0.0 (TEMPORAL)");

        // SEMANTIC embedders should have non-zero weights
        assert!(weights[0] > 0.0, "E1 should have non-zero weight (SEMANTIC)");
        assert!(weights[4] > 0.0, "E5 should have non-zero weight (SEMANTIC)");
        assert!(weights[5] > 0.0, "E6 should have non-zero weight (SEMANTIC)");
        assert!(weights[6] > 0.0, "E7 should have non-zero weight (SEMANTIC)");
        assert!(weights[9] > 0.0, "E10 should have non-zero weight (SEMANTIC)");

        // RELATIONAL embedders should have ~half the weight of SEMANTIC
        assert!(weights[7] > 0.0, "E8 should have non-zero weight (RELATIONAL)");
        assert!(weights[10] > 0.0, "E11 should have non-zero weight (RELATIONAL)");

        // STRUCTURAL embedder
        assert!(weights[8] > 0.0, "E9 should have non-zero weight (STRUCTURAL)");

        // Pipeline-stage embedders (E12, E13) = 0.0 per ARCH-13
        assert_eq!(weights[11], 0.0, "E12 should be 0.0 (rerank-only per ARCH-13, AP-73)");
        assert_eq!(weights[12], 0.0, "E13 should be 0.0 (recall-only per ARCH-13, AP-74)");

        println!("[VERIFIED] category_weighted profile follows constitution and ARCH-13");
    }

    #[test]
    fn test_validate_weights_valid() {
        let valid = get_weight_profile("semantic_search").unwrap();
        assert!(
            validate_weights(&valid).is_ok(),
            "Valid profile should pass validation"
        );
        println!("[VERIFIED] Valid weights pass validation");
    }

    #[test]
    fn test_validate_weights_out_of_range() {
        let mut weights = [0.077f32; NUM_EMBEDDERS];
        weights[0] = 1.5; // Out of range

        let result = validate_weights(&weights);
        assert!(result.is_err());

        match result.unwrap_err() {
            WeightValidationError::OutOfRange { space_index, .. } => {
                assert_eq!(space_index, 0);
            }
            _ => panic!("Expected OutOfRange error"),
        }
        println!("[VERIFIED] Out-of-range weight fails fast");
    }

    #[test]
    fn test_validate_weights_invalid_sum() {
        let weights = [0.5f32; NUM_EMBEDDERS]; // Sum = 6.5

        let result = validate_weights(&weights);
        assert!(result.is_err());

        match result.unwrap_err() {
            WeightValidationError::InvalidSum { actual, .. } => {
                assert!((actual - 6.5).abs() < 0.01);
            }
            _ => panic!("Expected InvalidSum error"),
        }
        println!("[VERIFIED] Invalid sum fails fast");
    }

    #[test]
    fn test_space_names() {
        assert_eq!(space_name(0), "E1_Semantic");
        assert_eq!(space_name(12), "E13_SPLADE");
        assert_eq!(space_name(13), "Unknown");
        println!("[VERIFIED] space_name returns correct names");
    }

    #[test]
    fn test_space_json_keys() {
        assert_eq!(space_json_key(0), "e1_semantic");
        assert_eq!(space_json_key(12), "e13_splade");
        println!("[VERIFIED] space_json_key returns correct keys");
    }

    #[test]
    fn test_parse_weights_from_json_valid() {
        let json_arr: Vec<serde_json::Value> = vec![
            0.28, 0.05, 0.05, 0.05, 0.10, 0.04, 0.18, 0.05, 0.05, 0.05, 0.03, 0.05, 0.02,
        ]
        .into_iter()
        .map(serde_json::Value::from)
        .collect();

        let result = parse_weights_from_json(&json_arr);
        assert!(result.is_ok());
        println!("[VERIFIED] parse_weights_from_json works for valid input");
    }

    #[test]
    fn test_parse_weights_from_json_wrong_count() {
        let json_arr: Vec<serde_json::Value> = vec![0.5, 0.5]
            .into_iter()
            .map(serde_json::Value::from)
            .collect();

        let result = parse_weights_from_json(&json_arr);
        assert!(result.is_err());

        match result.unwrap_err() {
            WeightValidationError::WrongCount { expected, actual } => {
                assert_eq!(expected, 13);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected WrongCount error"),
        }
        println!("[VERIFIED] Wrong count fails with clear error");
    }

    // =========================================================================
    // SEQUENCE NAVIGATION PROFILE TESTS
    // =========================================================================

    #[test]
    fn test_sequence_navigation_profile_exists() {
        let weights = get_weight_profile("sequence_navigation");
        assert!(
            weights.is_some(),
            "sequence_navigation profile should exist"
        );
        println!("[VERIFIED] sequence_navigation profile exists");
    }

    #[test]
    fn test_sequence_navigation_e4_is_primary() {
        // E4 should be the PRIMARY embedder for sequence navigation
        let weights = get_weight_profile("sequence_navigation").unwrap();

        // E4 should be >= 0.50 (dominant)
        assert!(
            weights[3] >= 0.50,
            "E4 should be >= 0.50 in sequence_navigation (got {})",
            weights[3]
        );

        // E4 should be the highest weighted embedder
        let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            (weights[3] - max_weight).abs() < 0.001,
            "E4 should be highest weighted in sequence_navigation"
        );

        println!(
            "[VERIFIED] sequence_navigation has E4={:.2} as primary",
            weights[3]
        );
    }

    #[test]
    fn test_sequence_navigation_pipeline_stage_excluded() {
        // E12 (rerank) and E13 (recall) should be excluded per ARCH-13
        let weights = get_weight_profile("sequence_navigation").unwrap();

        // E12 should be 0.0 (used only for reranking)
        assert_eq!(
            weights[11], 0.0,
            "E12 should be 0.0 in sequence_navigation per ARCH-13"
        );

        println!("[VERIFIED] sequence_navigation excludes pipeline-stage E12");
    }

    #[test]
    fn test_conversation_history_profile_exists() {
        let weights = get_weight_profile("conversation_history");
        assert!(
            weights.is_some(),
            "conversation_history profile should exist"
        );
        println!("[VERIFIED] conversation_history profile exists");
    }

    #[test]
    fn test_conversation_history_balanced_e1_e4() {
        // conversation_history should balance E1 (semantic) and E4 (sequence)
        let weights = get_weight_profile("conversation_history").unwrap();

        // E1 should be significant (>= 0.25)
        assert!(
            weights[0] >= 0.25,
            "E1 should be >= 0.25 in conversation_history (got {})",
            weights[0]
        );

        // E4 should be significant (>= 0.30)
        assert!(
            weights[3] >= 0.30,
            "E4 should be >= 0.30 in conversation_history (got {})",
            weights[3]
        );

        // Combined E1 + E4 should be >= 0.60 (they're the primary pair)
        let combined = weights[0] + weights[3];
        assert!(
            combined >= 0.60,
            "E1 + E4 should be >= 0.60 in conversation_history (got {})",
            combined
        );

        println!(
            "[VERIFIED] conversation_history has E1={:.2} + E4={:.2} = {:.2}",
            weights[0], weights[3], combined
        );
    }

    #[test]
    fn test_conversation_history_pipeline_stage_excluded() {
        // E12 (rerank) and E13 (recall) should be excluded per ARCH-13
        let weights = get_weight_profile("conversation_history").unwrap();

        // E12 should be 0.0
        assert_eq!(
            weights[11], 0.0,
            "E12 should be 0.0 in conversation_history per ARCH-13"
        );

        println!("[VERIFIED] conversation_history excludes pipeline-stage E12");
    }

    #[test]
    fn test_sequence_profiles_sum_to_one() {
        let sequence_profiles = ["sequence_navigation", "conversation_history"];

        for profile_name in sequence_profiles {
            let weights = get_weight_profile(profile_name).expect(&format!(
                "Profile '{}' should exist",
                profile_name
            ));

            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Profile '{}' weights sum to {} (expected ~1.0)",
                profile_name,
                sum
            );

            println!(
                "[VERIFIED] Profile '{}' sums to {:.4}",
                profile_name, sum
            );
        }
    }

    // =========================================================================
    // INTENT PROFILE TESTS
    // =========================================================================

    #[test]
    fn test_intent_search_profile_exists() {
        let weights = get_weight_profile("intent_search");
        assert!(weights.is_some(), "intent_search profile should exist");
        println!("[VERIFIED] intent_search profile exists");
    }

    #[test]
    fn test_intent_enhanced_profile_exists() {
        let weights = get_weight_profile("intent_enhanced");
        assert!(weights.is_some(), "intent_enhanced profile should exist");
        println!("[VERIFIED] intent_enhanced profile exists");
    }

    #[test]
    fn test_intent_search_e10_weight() {
        // E10 should be 0.0 in intent_search - per ARCH-17, E10 now operates via
        // multiplicative boost POST-RETRIEVAL, not in weighted fusion.
        // This prevents E10 from competing with E1 (it ENHANCES E1 instead).
        let weights = get_weight_profile("intent_search").unwrap();
        assert!(
            (weights[9] - 0.0).abs() < 0.001,
            "E10 should be 0.0 in intent_search (ARCH-17 multiplicative boost). Got {}",
            weights[9]
        );
        println!("[VERIFIED] intent_search has E10={:.2} (multiplicative boost via IntentBoostConfig)", weights[9]);
    }

    #[test]
    fn test_intent_enhanced_e10_weight() {
        // E10 should be 0.0 in intent_enhanced - per ARCH-17, E10 now operates via
        // multiplicative boost POST-RETRIEVAL with IntentBoostConfig::aggressive().
        let weights = get_weight_profile("intent_enhanced").unwrap();
        assert!(
            (weights[9] - 0.0).abs() < 0.001,
            "E10 should be 0.0 in intent_enhanced (ARCH-17 multiplicative boost). Got {}",
            weights[9]
        );
        println!("[VERIFIED] intent_enhanced has E10={:.2} (multiplicative boost via IntentBoostConfig::aggressive())", weights[9]);
    }

    #[test]
    fn test_intent_profiles_e1_boosted() {
        // With E10 moved to multiplicative boost, E1 should be boosted in intent profiles
        // to maintain strong semantic foundation (ARCH-12).
        let intent_search = get_weight_profile("intent_search").unwrap();
        let intent_enhanced = get_weight_profile("intent_enhanced").unwrap();
        let semantic_search = get_weight_profile("semantic_search").unwrap();

        // E1 in intent profiles should be >= E1 in semantic_search
        assert!(
            intent_search[0] >= semantic_search[0],
            "intent_search E1 ({}) should be >= semantic_search E1 ({})",
            intent_search[0], semantic_search[0]
        );
        assert!(
            intent_enhanced[0] >= semantic_search[0],
            "intent_enhanced E1 ({}) should be >= semantic_search E1 ({})",
            intent_enhanced[0], semantic_search[0]
        );
        println!(
            "[VERIFIED] Intent profiles have boosted E1: intent_search={:.2}, intent_enhanced={:.2}, semantic_search={:.2}",
            intent_search[0], intent_enhanced[0], semantic_search[0]
        );
    }

    // =========================================================================
    // PIPELINE-AWARE PROFILE TESTS - Phase 5 E12/E13 Integration
    // =========================================================================

    #[test]
    fn test_pipeline_stage1_recall_profile_exists() {
        let weights = get_weight_profile("pipeline_stage1_recall");
        assert!(
            weights.is_some(),
            "pipeline_stage1_recall profile should exist"
        );
        println!("[VERIFIED] pipeline_stage1_recall profile exists");
    }

    #[test]
    fn test_pipeline_stage1_e13_is_primary() {
        // E13 should have significant weight for sparse recall
        let weights = get_weight_profile("pipeline_stage1_recall").unwrap();

        // E13 should be one of the highest weighted
        assert!(
            weights[12] >= 0.20,
            "E13 should be >= 0.20 in pipeline_stage1_recall (got {})",
            weights[12]
        );

        // E6 (sparse keywords) should also have significant weight to support E13
        assert!(
            weights[5] >= 0.15,
            "E6 should be >= 0.15 in pipeline_stage1_recall (got {})",
            weights[5]
        );

        println!(
            "[VERIFIED] pipeline_stage1_recall has E13={:.2}, E6={:.2} for recall",
            weights[12], weights[5]
        );
    }

    #[test]
    fn test_pipeline_stage1_e12_excluded() {
        // E12 (ColBERT) should be 0.0 - it's for Stage 3 reranking only
        let weights = get_weight_profile("pipeline_stage1_recall").unwrap();

        assert_eq!(
            weights[11], 0.0,
            "E12 should be 0.0 in pipeline_stage1_recall (Stage 3 rerank only per AP-73)"
        );

        println!("[VERIFIED] pipeline_stage1_recall excludes E12 (rerank-only)");
    }

    #[test]
    fn test_pipeline_stage2_scoring_profile_exists() {
        let weights = get_weight_profile("pipeline_stage2_scoring");
        assert!(
            weights.is_some(),
            "pipeline_stage2_scoring profile should exist"
        );
        println!("[VERIFIED] pipeline_stage2_scoring profile exists");
    }

    #[test]
    fn test_pipeline_stage2_e1_is_primary() {
        // E1 should be the PRIMARY embedder for Stage 2 scoring per ARCH-12
        let weights = get_weight_profile("pipeline_stage2_scoring").unwrap();

        // E1 should be >= 0.45 (dominant)
        assert!(
            weights[0] >= 0.45,
            "E1 should be >= 0.45 in pipeline_stage2_scoring (got {})",
            weights[0]
        );

        // E1 should be the highest weighted embedder
        let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            (weights[0] - max_weight).abs() < 0.001,
            "E1 should be highest weighted in pipeline_stage2_scoring"
        );

        println!(
            "[VERIFIED] pipeline_stage2_scoring has E1={:.2} as primary (ARCH-12)",
            weights[0]
        );
    }

    #[test]
    fn test_pipeline_stage2_e12_e13_excluded() {
        // E12 and E13 should be 0.0 in Stage 2 scoring:
        // - E12 is applied via MaxSim in Stage 3
        // - E13 was used in Stage 1 recall
        let weights = get_weight_profile("pipeline_stage2_scoring").unwrap();

        assert_eq!(
            weights[11], 0.0,
            "E12 should be 0.0 in pipeline_stage2_scoring (Stage 3 rerank only per AP-73)"
        );
        assert_eq!(
            weights[12], 0.0,
            "E13 should be 0.0 in pipeline_stage2_scoring (Stage 1 recall only per AP-74)"
        );

        println!("[VERIFIED] pipeline_stage2_scoring excludes E12/E13 (pipeline-stage only)");
    }

    #[test]
    fn test_pipeline_full_profile_exists() {
        let weights = get_weight_profile("pipeline_full");
        assert!(weights.is_some(), "pipeline_full profile should exist");
        println!("[VERIFIED] pipeline_full profile exists");
    }

    #[test]
    fn test_pipeline_full_balanced() {
        // pipeline_full should balance E1 (semantic) with E6/E13 (keyword/recall)
        let weights = get_weight_profile("pipeline_full").unwrap();

        // E1 should be strong
        assert!(
            weights[0] >= 0.35,
            "E1 should be >= 0.35 in pipeline_full (got {})",
            weights[0]
        );

        // E12 should be 0.0 (applied via MaxSim separately)
        assert_eq!(
            weights[11], 0.0,
            "E12 should be 0.0 in pipeline_full (applied via MaxSim, not fusion)"
        );

        println!(
            "[VERIFIED] pipeline_full has E1={:.2}, E12={:.2} (E12 via MaxSim)",
            weights[0], weights[11]
        );
    }

    #[test]
    fn test_pipeline_profiles_temporal_excluded() {
        // All pipeline profiles should have E2-E4 = 0.0 per AP-71
        let pipeline_profiles = [
            "pipeline_stage1_recall",
            "pipeline_stage2_scoring",
            "pipeline_full",
        ];

        for profile_name in pipeline_profiles {
            let weights = get_weight_profile(profile_name)
                .expect(&format!("Profile '{}' should exist", profile_name));

            assert_eq!(
                weights[1], 0.0,
                "E2 should be 0.0 in '{}' per AP-71",
                profile_name
            );
            assert_eq!(
                weights[2], 0.0,
                "E3 should be 0.0 in '{}' per AP-71",
                profile_name
            );
            assert_eq!(
                weights[3], 0.0,
                "E4 should be 0.0 in '{}' per AP-71",
                profile_name
            );

            println!(
                "[VERIFIED] Profile '{}' has temporal embedders (E2-E4) = 0.0",
                profile_name
            );
        }
    }
}
