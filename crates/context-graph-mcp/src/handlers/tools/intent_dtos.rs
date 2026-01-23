//! DTOs for intent-aware MCP tools.
//!
//! Per PRD v6 E10 Intent/Context Enhancement, these DTOs support:
//! - search_by_intent: Find memories with similar intent using asymmetric E10
//! - find_contextual_matches: Find memories relevant to a context using E10
//!
//! Constitution References:
//! - ARCH-15: Uses asymmetric E10 with separate intent/context encodings
//! - E10 ENHANCES E1 semantic search via blendWithSemantic parameter
//! - Direction modifiers: intent→context=1.2, context→intent=0.8

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default topK for intent search results.
pub const DEFAULT_INTENT_SEARCH_TOP_K: usize = 10;

/// Maximum topK for intent search results.
pub const MAX_INTENT_SEARCH_TOP_K: usize = 50;

/// Default minimum score threshold for intent search results.
pub const DEFAULT_MIN_INTENT_SCORE: f32 = 0.2;

/// Default blend weight for E10 vs E1 semantic.
/// 0.3 means 70% E1 semantic + 30% E10 intent/context.
pub const DEFAULT_BLEND_WITH_SEMANTIC: f32 = 0.3;

/// Intent→Context direction modifier.
/// Per plan: intent→context = 1.2x boost.
pub const INTENT_TO_CONTEXT_MODIFIER: f32 = 1.2;

/// Context→Intent direction modifier (dampening).
/// Per plan: context→intent = 0.8x dampening.
pub const CONTEXT_TO_INTENT_MODIFIER: f32 = 0.8;

// ============================================================================
// REQUEST DTOs
// ============================================================================

/// Request parameters for search_by_intent tool.
///
/// # Example JSON
/// ```json
/// {
///   "query": "Improve the performance of the database queries",
///   "topK": 10,
///   "minScore": 0.2,
///   "blendWithSemantic": 0.3,
///   "includeContent": true
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct SearchByIntentRequest {
    /// The intent or goal to search for (required).
    /// Describe what you're trying to accomplish.
    pub query: String,

    /// Maximum number of results to return (1-50, default: 10).
    #[serde(rename = "topK", default = "default_top_k")]
    pub top_k: usize,

    /// Minimum similarity score threshold (0-1, default: 0.2).
    #[serde(rename = "minScore", default = "default_min_score")]
    pub min_score: f32,

    /// Blend weight for E10 intent vs E1 semantic (0-1, default: 0.3).
    /// 0.0 = pure E1 semantic, 1.0 = pure E10 intent.
    #[serde(rename = "blendWithSemantic", default = "default_blend")]
    pub blend_with_semantic: f32,

    /// Whether to include full content text in results (default: false).
    #[serde(rename = "includeContent", default)]
    pub include_content: bool,
}

fn default_top_k() -> usize {
    DEFAULT_INTENT_SEARCH_TOP_K
}

fn default_min_score() -> f32 {
    DEFAULT_MIN_INTENT_SCORE
}

fn default_blend() -> f32 {
    DEFAULT_BLEND_WITH_SEMANTIC
}

impl Default for SearchByIntentRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            top_k: DEFAULT_INTENT_SEARCH_TOP_K,
            min_score: DEFAULT_MIN_INTENT_SCORE,
            blend_with_semantic: DEFAULT_BLEND_WITH_SEMANTIC,
            include_content: false,
        }
    }
}

impl SearchByIntentRequest {
    /// Validate the request parameters.
    ///
    /// # Errors
    /// Returns an error message if:
    /// - query is empty
    /// - topK is outside [1, 50]
    /// - minScore is outside [0, 1] or NaN/infinite
    /// - blendWithSemantic is outside [0, 1] or NaN/infinite
    pub fn validate(&self) -> Result<(), String> {
        if self.query.is_empty() {
            return Err("query is required and cannot be empty".to_string());
        }

        if self.top_k < 1 || self.top_k > MAX_INTENT_SEARCH_TOP_K {
            return Err(format!(
                "topK must be between 1 and {}, got {}",
                MAX_INTENT_SEARCH_TOP_K, self.top_k
            ));
        }

        if self.min_score.is_nan() || self.min_score.is_infinite() {
            return Err("minScore must be a finite number".to_string());
        }

        if self.min_score < 0.0 || self.min_score > 1.0 {
            return Err(format!(
                "minScore must be between 0.0 and 1.0, got {}",
                self.min_score
            ));
        }

        if self.blend_with_semantic.is_nan() || self.blend_with_semantic.is_infinite() {
            return Err("blendWithSemantic must be a finite number".to_string());
        }

        if self.blend_with_semantic < 0.0 || self.blend_with_semantic > 1.0 {
            return Err(format!(
                "blendWithSemantic must be between 0.0 and 1.0, got {}",
                self.blend_with_semantic
            ));
        }

        Ok(())
    }
}

/// Request parameters for find_contextual_matches tool.
///
/// # Example JSON
/// ```json
/// {
///   "context": "Working on database optimization for production system",
///   "topK": 10,
///   "minScore": 0.2,
///   "blendWithSemantic": 0.3,
///   "includeContent": true
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct FindContextualMatchesRequest {
    /// The context or situation to find relevant memories for (required).
    /// Describe the current situation.
    pub context: String,

    /// Maximum number of results to return (1-50, default: 10).
    #[serde(rename = "topK", default = "default_top_k")]
    pub top_k: usize,

    /// Minimum similarity score threshold (0-1, default: 0.2).
    #[serde(rename = "minScore", default = "default_min_score")]
    pub min_score: f32,

    /// Blend weight for E10 context vs E1 semantic (0-1, default: 0.3).
    /// 0.0 = pure E1 semantic, 1.0 = pure E10 context.
    #[serde(rename = "blendWithSemantic", default = "default_blend")]
    pub blend_with_semantic: f32,

    /// Whether to include full content text in results (default: false).
    #[serde(rename = "includeContent", default)]
    pub include_content: bool,
}

impl Default for FindContextualMatchesRequest {
    fn default() -> Self {
        Self {
            context: String::new(),
            top_k: DEFAULT_INTENT_SEARCH_TOP_K,
            min_score: DEFAULT_MIN_INTENT_SCORE,
            blend_with_semantic: DEFAULT_BLEND_WITH_SEMANTIC,
            include_content: false,
        }
    }
}

impl FindContextualMatchesRequest {
    /// Validate the request parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.context.is_empty() {
            return Err("context is required and cannot be empty".to_string());
        }

        if self.top_k < 1 || self.top_k > MAX_INTENT_SEARCH_TOP_K {
            return Err(format!(
                "topK must be between 1 and {}, got {}",
                MAX_INTENT_SEARCH_TOP_K, self.top_k
            ));
        }

        if self.min_score.is_nan() || self.min_score.is_infinite() {
            return Err("minScore must be a finite number".to_string());
        }

        if self.min_score < 0.0 || self.min_score > 1.0 {
            return Err(format!(
                "minScore must be between 0.0 and 1.0, got {}",
                self.min_score
            ));
        }

        if self.blend_with_semantic.is_nan() || self.blend_with_semantic.is_infinite() {
            return Err("blendWithSemantic must be a finite number".to_string());
        }

        if self.blend_with_semantic < 0.0 || self.blend_with_semantic > 1.0 {
            return Err(format!(
                "blendWithSemantic must be between 0.0 and 1.0, got {}",
                self.blend_with_semantic
            ));
        }

        Ok(())
    }
}

// ============================================================================
// RESPONSE DTOs
// ============================================================================

/// A single search result for intent/context search.
#[derive(Debug, Clone, Serialize)]
pub struct IntentSearchResult {
    /// UUID of the matched memory.
    #[serde(rename = "memoryId")]
    pub memory_id: Uuid,

    /// Blended score (E1 semantic + E10 intent/context).
    pub score: f32,

    /// Raw E1 semantic similarity (before blending).
    #[serde(rename = "e1Similarity")]
    pub e1_similarity: f32,

    /// Raw E10 intent/context similarity (before blending).
    #[serde(rename = "e10Similarity")]
    pub e10_similarity: f32,

    /// Full content text (if includeContent=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Source provenance information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
}

/// Source provenance information.
#[derive(Debug, Clone, Serialize)]
pub struct SourceInfo {
    /// Type of source (HookDescription, ClaudeResponse, MDFileChunk).
    #[serde(rename = "sourceType")]
    pub source_type: String,

    /// File path if from file source.
    #[serde(skip_serializing_if = "Option::is_none", rename = "filePath")]
    pub file_path: Option<String>,

    /// Hook type if from hook source.
    #[serde(skip_serializing_if = "Option::is_none", rename = "hookType")]
    pub hook_type: Option<String>,

    /// Tool name if from tool use.
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolName")]
    pub tool_name: Option<String>,
}

/// Response metadata for intent/context search.
#[derive(Debug, Clone, Serialize)]
pub struct IntentSearchMetadata {
    /// Number of candidates evaluated before filtering.
    #[serde(rename = "candidatesEvaluated")]
    pub candidates_evaluated: usize,

    /// Number of results filtered by score threshold.
    #[serde(rename = "filteredByScore")]
    pub filtered_by_score: usize,

    /// E10 blend weight used.
    #[serde(rename = "blendWeight")]
    pub blend_weight: f32,

    /// E1 weight (1.0 - blendWeight).
    #[serde(rename = "e1Weight")]
    pub e1_weight: f32,

    /// Direction modifier applied.
    #[serde(rename = "directionModifier")]
    pub direction_modifier: f32,
}

/// Response for search_by_intent tool.
#[derive(Debug, Clone, Serialize)]
pub struct SearchByIntentResponse {
    /// Original query.
    pub query: String,

    /// Matched results with blended scores.
    pub results: Vec<IntentSearchResult>,

    /// Number of results returned.
    pub count: usize,

    /// Metadata about the search.
    pub metadata: IntentSearchMetadata,
}

/// Response for find_contextual_matches tool.
#[derive(Debug, Clone, Serialize)]
pub struct FindContextualMatchesResponse {
    /// Original context query.
    pub context: String,

    /// Matched results with blended scores.
    pub results: Vec<IntentSearchResult>,

    /// Number of results returned.
    pub count: usize,

    /// Metadata about the search.
    pub metadata: IntentSearchMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_by_intent_validation_success() {
        let req = SearchByIntentRequest {
            query: "Find performance optimizations".to_string(),
            ..Default::default()
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_search_by_intent_empty_query() {
        let req = SearchByIntentRequest::default();
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_search_by_intent_invalid_blend() {
        let req = SearchByIntentRequest {
            query: "test".to_string(),
            blend_with_semantic: 1.5,
            ..Default::default()
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_find_contextual_matches_validation_success() {
        let req = FindContextualMatchesRequest {
            context: "Working on database optimization".to_string(),
            ..Default::default()
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_find_contextual_matches_empty_context() {
        let req = FindContextualMatchesRequest::default();
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_direction_modifiers() {
        assert!((INTENT_TO_CONTEXT_MODIFIER - 1.2).abs() < 0.001);
        assert!((CONTEXT_TO_INTENT_MODIFIER - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_default_blend_ensures_e1_dominant() {
        // Default 0.3 means 70% E1, 30% E10
        let e1_weight = 1.0 - DEFAULT_BLEND_WITH_SEMANTIC;
        assert!(e1_weight > DEFAULT_BLEND_WITH_SEMANTIC);
    }
}
