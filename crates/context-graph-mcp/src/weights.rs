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

#![allow(dead_code)]

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
/// E12 (Late Interaction) and E13 (SPLADE) have low weights in aggregation
/// because they're primarily used in pipeline stages:
/// - E13: Stage 1 recall (inverted index)
/// - E12: Stage 3 re-ranking (MaxSim)
pub const WEIGHT_PROFILES: &[(&str, [f32; NUM_EMBEDDERS])] = &[
    // =========================================================================
    // SEMANTIC PROFILES - Temporal (E2-E4) = 0.0 per AP-71
    // =========================================================================

    // Semantic Search: General queries - E1 primary, E5/E7/E10 supporting
    // Sum of weights = 1.0 (excluding temporal which are 0.0)
    (
        "semantic_search",
        [
            0.35, // E1_Semantic (primary)
            0.0,  // E2_Temporal_Recent - NOT for semantic search
            0.0,  // E3_Temporal_Periodic - NOT for semantic search
            0.0,  // E4_Temporal_Positional - NOT for semantic search
            0.15, // E5_Causal
            0.05, // E6_Sparse (keyword backup)
            0.20, // E7_Code
            0.05, // E8_Graph (relational)
            0.0,  // E9_HDC (noise-robust backup, not in fusion)
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

    // Category-Weighted: Constitution-compliant weights per CLAUDE.md
    // SEMANTIC (E1,E5,E6,E7,E10,E12,E13): weight 1.0
    // TEMPORAL (E2,E3,E4): weight 0.0 per AP-60
    // RELATIONAL (E8,E11): weight 0.5
    // STRUCTURAL (E9): weight 0.5
    // max_weighted_agreement = 8.5
    (
        "category_weighted",
        [
            1.0 / 8.5,   // E1_Semantic (SEMANTIC)
            0.0,         // E2_Temporal_Recent (TEMPORAL - excluded)
            0.0,         // E3_Temporal_Periodic (TEMPORAL - excluded)
            0.0,         // E4_Temporal_Positional (TEMPORAL - excluded)
            1.0 / 8.5,   // E5_Causal (SEMANTIC)
            1.0 / 8.5,   // E6_Sparse (SEMANTIC)
            1.0 / 8.5,   // E7_Code (SEMANTIC)
            0.5 / 8.5,   // E8_Graph (RELATIONAL)
            0.5 / 8.5,   // E9_HDC (STRUCTURAL)
            1.0 / 8.5,   // E10_Multimodal (SEMANTIC)
            0.5 / 8.5,   // E11_Entity (RELATIONAL)
            1.0 / 8.5,   // E12_Late_Interaction (SEMANTIC)
            1.0 / 8.5,   // E13_SPLADE (SEMANTIC)
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
pub fn get_profile_names() -> Vec<&'static str> {
    WEIGHT_PROFILES.iter().map(|(n, _)| *n).collect()
}

/// Validate that weights sum to ~1.0 and all are in [0.0, 1.0].
///
/// # FAIL FAST
/// Returns detailed error on validation failure.
pub fn validate_weights(weights: &[f32; NUM_EMBEDDERS]) -> Result<(), WeightValidationError> {
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
pub fn space_name(idx: usize) -> &'static str {
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
pub fn space_json_key(idx: usize) -> &'static str {
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
pub enum WeightValidationError {
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
/// # Errors
/// - `WrongCount`: Array has wrong number of elements
/// - `OutOfRange`: A weight is outside [0.0, 1.0]
/// - `InvalidSum`: Weights don't sum to 1.0
pub fn parse_weights_from_json(
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
        weights[i] = v.as_f64().unwrap_or(0.0) as f32;
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
            (semantic.unwrap()[0] - 0.35).abs() < 0.001,
            "E1 should be 0.35 in semantic_search profile"
        );

        let missing = get_weight_profile("nonexistent");
        assert!(missing.is_none(), "Unknown profile should return None");

        println!("[VERIFIED] get_weight_profile works correctly");
    }

    #[test]
    fn test_temporal_embedders_excluded_from_semantic_profiles() {
        // Per AP-71: Temporal embedders (E2-E4) MUST NOT be used in similarity scoring
        // All semantic search profiles should have E2-E4 = 0.0

        let semantic_profiles = ["semantic_search", "causal_reasoning", "code_search", "fact_checking"];

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
    fn test_category_weighted_profile() {
        // Per constitution: SEMANTIC=1.0, TEMPORAL=0.0, RELATIONAL=0.5, STRUCTURAL=0.5
        let weights = get_weight_profile("category_weighted").expect("category_weighted should exist");

        // Temporal (E2-E4) = 0.0
        assert_eq!(weights[1], 0.0, "E2 should be 0.0 (TEMPORAL)");
        assert_eq!(weights[2], 0.0, "E3 should be 0.0 (TEMPORAL)");
        assert_eq!(weights[3], 0.0, "E4 should be 0.0 (TEMPORAL)");

        // SEMANTIC embedders should have non-zero weights
        assert!(weights[0] > 0.0, "E1 should have non-zero weight (SEMANTIC)");
        assert!(weights[4] > 0.0, "E5 should have non-zero weight (SEMANTIC)");
        assert!(weights[6] > 0.0, "E7 should have non-zero weight (SEMANTIC)");
        assert!(weights[9] > 0.0, "E10 should have non-zero weight (SEMANTIC)");

        // RELATIONAL embedders should have ~half the weight of SEMANTIC
        assert!(weights[7] > 0.0, "E8 should have non-zero weight (RELATIONAL)");
        assert!(weights[10] > 0.0, "E11 should have non-zero weight (RELATIONAL)");

        println!("[VERIFIED] category_weighted profile follows constitution");
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
}
