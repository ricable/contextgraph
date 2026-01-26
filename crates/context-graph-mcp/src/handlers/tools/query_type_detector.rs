//! Query Type Detection for Autonomous Multi-Embedder Enrichment.
//!
//! Detects query characteristics to select the most relevant enhancer embedders.
//! Per Constitution retrieval section:
//! - use_E5: "Causal queries (why, what caused)"
//! - use_E7: "Code queries (implementations, functions)"
//! - use_E10: "Intent queries (same goal, similar purpose)"
//! - use_E11: "Entity queries (specific named things)"
//! - use_E6_E13: "Keyword queries (exact terms, jargon)"
//!
//! # Design Philosophy
//!
//! The 13 embedders are 13 unique perspectives on every query. E1 (semantic)
//! is the foundation but has blind spots. The query type detector identifies
//! which enhancer embedders are most likely to find what E1 misses.
//!
//! # Examples
//!
//! - "why did the authentication fail" → Causal (E5)
//! - "tokio::spawn implementation" → Code (E7)
//! - "What databases work with Rust?" → Entity (E11)
//! - "What was the goal of that refactor?" → Intent (E10)
//! - "exact error message: ConnectionRefused" → Keyword (E6/E13)

use tracing::debug;

use super::embedder_dtos::EmbedderId;
use super::enrichment_dtos::{EnrichmentConfig, EnrichmentMode, QueryType};

// =============================================================================
// DETECTION PATTERNS
// =============================================================================

/// Causal query patterns - triggers E5 (V_causality) enhancement.
///
/// These patterns indicate the query is looking for cause-effect relationships.
/// E1 often loses causal direction through embedding averaging.
const CAUSAL_PATTERNS: &[&str] = &[
    // Why/Because patterns
    "why",
    "because",
    "caused",
    "cause",
    "causes",
    "causing",
    // Effect patterns
    "led to",
    "leads to",
    "result in",
    "results in",
    "resulted in",
    "resulting in",
    // Consequence patterns
    "consequence",
    "effect of",
    "effects of",
    "impact of",
    "due to",
    // Trigger patterns
    "triggered",
    "trigger",
    "what happened when",
    "what happens if",
    "what happens when",
    // Reason patterns
    "reason for",
    "reasons for",
    "root cause",
    "underlying cause",
    // Failure patterns (often causal)
    "failed because",
    "broke because",
    "crashed because",
    "error caused by",
];

/// Code query patterns - triggers E7 (V_correctness) enhancement.
///
/// These patterns indicate the query is about code/implementation.
/// E1 treats code as natural language, missing structural patterns.
const CODE_PATTERNS: &[&str] = &[
    // Function/method patterns
    "function",
    "method",
    "impl",
    "implementation",
    "fn ",
    "def ",
    "func ",
    // Code structure patterns
    "class",
    "struct",
    "enum",
    "trait",
    "interface",
    "module",
    // Import/dependency patterns
    "import",
    "use ",
    "require",
    "include",
    "dependency",
    "crate",
    // Code-specific terms
    "api",
    "endpoint",
    "handler",
    "controller",
    "middleware",
    "callback",
    // Syntax patterns (look for :: or .)
    "::",
    "->",
    "=>",
    ".await",
    ".unwrap",
    // Error handling
    "try",
    "catch",
    "except",
    "error handling",
    "panic",
    // Testing
    "#[test]",
    "assert",
    "expect",
    "should",
    "test case",
];

/// Entity query patterns - triggers E11 (V_factuality/KEPLER) enhancement.
///
/// These patterns indicate the query is about named entities or facts.
/// E1 misses that "Diesel" IS a database ORM without explicit mention.
const ENTITY_PATTERNS: &[&str] = &[
    // Factual queries
    "what is",
    "what are",
    "who is",
    "who are",
    "which",
    "what kind of",
    "what type of",
    // Definition queries
    "define",
    "definition",
    "meaning of",
    // Relationship queries
    "related to",
    "works with",
    "compatible with",
    "uses",
    "used by",
    "depends on",
    "requires",
    // Comparison queries
    "vs",
    "versus",
    "compared to",
    "difference between",
    "similarities between",
    // Entity-specific
    "library",
    "framework",
    "tool",
    "database",
    "language",
    "platform",
    "service",
    "provider",
];

/// Intent query patterns - triggers E10 (V_multimodality) enhancement.
///
/// These patterns indicate the query is about goals/purposes.
/// E1 may miss work with same intent but different words.
const INTENT_PATTERNS: &[&str] = &[
    // Goal patterns
    "goal",
    "goals",
    "objective",
    "objectives",
    "target",
    "targets",
    // Purpose patterns
    "purpose",
    "intent",
    "intention",
    "aim",
    "aiming",
    // Action patterns
    "trying to",
    "want to",
    "need to",
    "have to",
    "looking to",
    "planning to",
    "hoping to",
    // Achievement patterns
    "accomplish",
    "achieve",
    "complete",
    "finish",
    "solve",
    "fix",
    // Mission patterns
    "mission",
    "vision",
    "strategy",
    "plan",
    // Question forms
    "what was the plan",
    "what were we doing",
    "what is the intent",
    "what was the goal",
    "similar purpose",
    "same goal",
];

/// Keyword/exact match patterns - triggers E6/E13 enhancement.
///
/// These patterns indicate the query needs exact term matching.
/// E1 dilutes exact keywords through dense averaging.
const KEYWORD_PATTERNS: &[&str] = &[
    // Quoted terms (handled specially in detection)
    // Exact match requests
    "exact",
    "exactly",
    "literal",
    "literally",
    "verbatim",
    "specific",
    "specifically",
    // Error message patterns
    "error message",
    "error:",
    "exception:",
    "warning:",
    "fatal:",
    // Identifier patterns
    "named",
    "called",
    "identifier",
    "variable",
    "constant",
    // Technical terms often need exact matching
    "version",
    "v1.",
    "v2.",
    "v3.",
    // Config/setting patterns
    "config",
    "setting",
    "parameter",
    "flag",
    "option",
];

/// Temporal query patterns - E2-E4 are POST-RETRIEVAL ONLY per ARCH-25.
///
/// These patterns indicate time-based queries. Note that temporal embedders
/// are NOT used in similarity fusion, only for post-retrieval boosting.
const TEMPORAL_PATTERNS: &[&str] = &[
    // Relative time
    "yesterday",
    "today",
    "last week",
    "this week",
    "last month",
    "this month",
    "recently",
    "just now",
    // Sequence patterns
    "before",
    "after",
    "previous",
    "next",
    "earlier",
    "later",
    "first",
    "last",
    // Time-specific
    "when did",
    "when was",
    "how long ago",
    "time of",
    // Session patterns
    "earlier in this session",
    "previous conversation",
    "last time we",
    "in our last session",
];

// =============================================================================
// DETECTION LOGIC
// =============================================================================

/// Detect query types from query text.
///
/// Returns all matching query types, sorted by confidence (highest first).
/// May return multiple types for queries that span categories.
///
/// # Arguments
/// * `query` - The search query text
///
/// # Returns
/// Vector of detected query types. Returns [QueryType::General] if no specific type detected.
pub fn detect_query_types(query: &str) -> Vec<QueryType> {
    let query_lower = query.to_lowercase();
    let mut detected: Vec<(QueryType, usize)> = Vec::new();

    // Count pattern matches for each type
    let causal_count = count_pattern_matches(&query_lower, CAUSAL_PATTERNS);
    let code_count = count_pattern_matches(&query_lower, CODE_PATTERNS) + count_code_indicators(query);
    let entity_count = count_pattern_matches(&query_lower, ENTITY_PATTERNS) + count_capitalized_words(query);
    let intent_count = count_pattern_matches(&query_lower, INTENT_PATTERNS);
    let keyword_count = count_pattern_matches(&query_lower, KEYWORD_PATTERNS) + count_quoted_terms(query);
    let temporal_count = count_pattern_matches(&query_lower, TEMPORAL_PATTERNS);

    // Add types with at least 1 match
    if causal_count > 0 {
        detected.push((QueryType::Causal, causal_count));
    }
    if code_count > 0 {
        detected.push((QueryType::Code, code_count));
    }
    if entity_count > 0 {
        detected.push((QueryType::Entity, entity_count));
    }
    if intent_count > 0 {
        detected.push((QueryType::Intent, intent_count));
    }
    if keyword_count > 0 {
        detected.push((QueryType::Keyword, keyword_count));
    }
    if temporal_count > 0 {
        detected.push((QueryType::Temporal, temporal_count));
    }

    // Sort by match count (highest first)
    detected.sort_by(|a, b| b.1.cmp(&a.1));

    // Extract just the types
    let types: Vec<QueryType> = detected.into_iter().map(|(t, _)| t).collect();

    if types.is_empty() {
        vec![QueryType::General]
    } else {
        debug!(
            query_preview = %query.chars().take(50).collect::<String>(),
            detected_types = ?types,
            "Query type detection complete"
        );
        types
    }
}

/// Count how many patterns match in the query.
fn count_pattern_matches(query_lower: &str, patterns: &[&str]) -> usize {
    patterns.iter().filter(|p| query_lower.contains(*p)).count()
}

/// Count code-specific indicators (::, ->, .await, etc.)
fn count_code_indicators(query: &str) -> usize {
    let mut count = 0;

    // Double colon (Rust/C++ namespace)
    count += query.matches("::").count();

    // Arrow operators
    count += query.matches("->").count();
    count += query.matches("=>").count();

    // Rust-specific
    count += query.matches(".await").count();
    count += query.matches(".unwrap").count();
    count += query.matches("async fn").count();
    count += query.matches("pub fn").count();
    count += query.matches("impl ").count();

    // Python-specific
    count += query.matches("def ").count();
    count += query.matches("self.").count();

    // Generic code patterns
    count += query.matches("()").count();
    count += query.matches("{}").count();
    count += query.matches("[]").count();

    count
}

/// Count capitalized words that might be entity names.
///
/// Entities like "Diesel", "PostgreSQL", "Rust" are often capitalized.
fn count_capitalized_words(query: &str) -> usize {
    query
        .split_whitespace()
        .filter(|word| {
            let trimmed = word.trim_matches(|c: char| !c.is_alphabetic());
            !trimmed.is_empty()
                && trimmed.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                && trimmed.len() >= 2 // Avoid single letters like "I"
                && !is_common_start_word(trimmed) // Avoid sentence starters
        })
        .count()
}

/// Check if word is a common sentence-starting word (not an entity).
fn is_common_start_word(word: &str) -> bool {
    const COMMON_STARTERS: &[&str] = &[
        "The", "A", "An", "This", "That", "What", "Why", "How", "When", "Where",
        "Who", "Which", "It", "I", "We", "They", "He", "She", "You", "Is", "Are",
        "Was", "Were", "Has", "Have", "Had", "Do", "Does", "Did", "Can", "Could",
        "Would", "Should", "Will", "If", "Or", "And", "But", "For", "In", "On",
        "At", "To", "From", "With", "About", "As", "By",
    ];
    COMMON_STARTERS.iter().any(|&s| s.eq_ignore_ascii_case(word))
}

/// Count quoted terms in the query.
///
/// Quoted terms indicate exact match requirements.
pub fn count_quoted_terms(query: &str) -> usize {
    let mut count = 0;
    let mut in_quote = false;

    for c in query.chars() {
        if c == '"' || c == '\'' {
            if in_quote {
                count += 1; // Closing quote = one quoted term
            }
            in_quote = !in_quote;
        }
    }

    count
}

// =============================================================================
// AUTO-UPGRADE DETECTION
// =============================================================================

/// Minimum quoted terms to trigger auto-upgrade to Pipeline strategy.
const AUTO_UPGRADE_QUOTED_THRESHOLD: usize = 1;

/// Minimum keyword pattern matches to trigger auto-upgrade to Pipeline strategy.
const AUTO_UPGRADE_KEYWORD_THRESHOLD: usize = 2;

/// Determine if a query should be auto-upgraded to Pipeline strategy.
///
/// Per Phase 4 E12/E13 Integration Plan, precision queries benefit from:
/// - E13 SPLADE for Stage 1 recall (exact term matching)
/// - E12 ColBERT for final reranking (phrase-level precision)
///
/// A query is considered a "precision query" if:
/// 1. It contains quoted terms (e.g., "ConnectionRefused") - indicates exact match need
/// 2. It has multiple keyword patterns (exact, verbatim, error:, etc.)
///
/// # Arguments
/// * `query` - The search query text
///
/// # Returns
/// `true` if the query should use Pipeline strategy, `false` for MultiSpace.
///
/// # Examples
///
/// ```ignore
/// // Quoted terms → Pipeline
/// assert!(should_auto_upgrade_to_pipeline("find \"ConnectionRefused\" error"));
///
/// // Keyword patterns → Pipeline
/// assert!(should_auto_upgrade_to_pipeline("exact error message: timeout"));
///
/// // Generic query → MultiSpace
/// assert!(!should_auto_upgrade_to_pipeline("what is authentication"));
/// ```
pub fn should_auto_upgrade_to_pipeline(query: &str) -> bool {
    // Check for quoted terms - strong indicator of precision need
    let quoted_count = count_quoted_terms(query);
    if quoted_count >= AUTO_UPGRADE_QUOTED_THRESHOLD {
        debug!(
            query_preview = %query.chars().take(50).collect::<String>(),
            quoted_terms = quoted_count,
            "Auto-upgrading to Pipeline: quoted terms detected"
        );
        return true;
    }

    // Check for keyword patterns
    let query_lower = query.to_lowercase();
    let keyword_count = count_pattern_matches(&query_lower, KEYWORD_PATTERNS);
    if keyword_count >= AUTO_UPGRADE_KEYWORD_THRESHOLD {
        debug!(
            query_preview = %query.chars().take(50).collect::<String>(),
            keyword_patterns = keyword_count,
            "Auto-upgrading to Pipeline: keyword patterns detected"
        );
        return true;
    }

    false
}

// =============================================================================
// EMBEDDER SELECTION
// =============================================================================

/// Select enhancer embedders based on detected query types and enrichment mode.
///
/// Per Constitution ARCH-12: E1 is always used as foundation (not included here).
/// Per Constitution ARCH-13: E12/E13 are pipeline stages, not selected here.
///
/// # Arguments
/// * `types` - Detected query types
/// * `mode` - Enrichment mode (controls how many enhancers)
///
/// # Returns
/// Vector of enhancer embedder IDs to use (excludes E1).
pub fn select_embedders_for_types(types: &[QueryType], mode: EnrichmentMode) -> Vec<EmbedderId> {
    let max_enhancers = mode.max_enhancers();
    if max_enhancers == 0 {
        return vec![];
    }

    let mut embedders: Vec<EmbedderId> = Vec::new();

    // Collect primary enhancers from each type
    for query_type in types {
        for embedder in query_type.primary_enhancers() {
            if !embedders.contains(&embedder) {
                embedders.push(embedder);
            }
        }
    }

    // If we have room for more and mode is Full, add secondary enhancers
    if mode == EnrichmentMode::Full && embedders.len() < max_enhancers {
        for query_type in types {
            for embedder in query_type.secondary_enhancers() {
                if !embedders.contains(&embedder) && embedders.len() < max_enhancers {
                    embedders.push(embedder);
                }
            }
        }
    }

    // Truncate to max allowed
    embedders.truncate(max_enhancers);

    debug!(
        query_types = ?types,
        mode = ?mode,
        selected_embedders = ?embedders,
        "Embedder selection complete"
    );

    embedders
}

// =============================================================================
// ENRICHMENT CONFIG BUILDER
// =============================================================================

/// Build enrichment configuration from query text and mode.
///
/// This is the main entry point for query analysis. It:
/// 1. Detects query types
/// 2. Selects appropriate embedders
/// 3. Builds the enrichment config
///
/// # Arguments
/// * `query` - The search query text
/// * `mode` - Enrichment mode (Off, Light, Full)
///
/// # Returns
/// EnrichmentConfig ready for the enrichment pipeline.
pub fn build_enrichment_config(query: &str, mode: EnrichmentMode) -> EnrichmentConfig {
    use context_graph_core::traits::DecayFunction;

    if mode == EnrichmentMode::Off {
        return EnrichmentConfig::off();
    }

    let detected_types = detect_query_types(query);
    let selected_embedders = select_embedders_for_types(&detected_types, mode);
    let detect_blind_spots = mode.enables_blind_spots();

    // Enable temporal boost when Temporal query type is detected
    // Per ARCH-25: Temporal boosts POST-retrieval only, NOT in similarity fusion
    let has_temporal = detected_types.contains(&QueryType::Temporal);

    debug!(
        query_preview = %query.chars().take(50).collect::<String>(),
        mode = ?mode,
        detected_types = ?detected_types,
        selected_embedders = ?selected_embedders,
        detect_blind_spots = detect_blind_spots,
        temporal_boost_enabled = has_temporal,
        "Enrichment config built"
    );

    EnrichmentConfig {
        mode,
        detected_types,
        selected_embedders,
        detect_blind_spots,
        temporal_boost_enabled: has_temporal,
        temporal_weight: if has_temporal { 0.3 } else { 0.0 },
        decay_function: if has_temporal {
            DecayFunction::Exponential
        } else {
            DecayFunction::default()
        },
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CAUSAL DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_detect_causal_why() {
        let types = detect_query_types("why did the authentication fail");
        assert!(types.contains(&QueryType::Causal));
    }

    #[test]
    fn test_detect_causal_because() {
        let types = detect_query_types("the test failed because of a null pointer");
        assert!(types.contains(&QueryType::Causal));
    }

    #[test]
    fn test_detect_causal_caused() {
        let types = detect_query_types("what caused the memory leak");
        assert!(types.contains(&QueryType::Causal));
    }

    #[test]
    fn test_detect_causal_led_to() {
        let types = detect_query_types("the change led to performance issues");
        assert!(types.contains(&QueryType::Causal));
    }

    #[test]
    fn test_detect_causal_root_cause() {
        let types = detect_query_types("find the root cause of the crash");
        assert!(types.contains(&QueryType::Causal));
    }

    // =========================================================================
    // CODE DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_detect_code_function() {
        let types = detect_query_types("find the function that handles authentication");
        assert!(types.contains(&QueryType::Code));
    }

    #[test]
    fn test_detect_code_double_colon() {
        let types = detect_query_types("tokio::spawn usage");
        assert!(types.contains(&QueryType::Code));
    }

    #[test]
    fn test_detect_code_impl() {
        let types = detect_query_types("impl TeleologicalStore for PostgresBackend");
        assert!(types.contains(&QueryType::Code));
    }

    #[test]
    fn test_detect_code_async_fn() {
        let types = detect_query_types("async fn embed_all");
        assert!(types.contains(&QueryType::Code));
    }

    #[test]
    fn test_detect_code_await() {
        let types = detect_query_types("the .await call that blocks");
        assert!(types.contains(&QueryType::Code));
    }

    // =========================================================================
    // ENTITY DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_detect_entity_what_is() {
        let types = detect_query_types("what is Diesel");
        assert!(types.contains(&QueryType::Entity));
    }

    #[test]
    fn test_detect_entity_capitalized() {
        let types = detect_query_types("find PostgreSQL configuration");
        assert!(types.contains(&QueryType::Entity));
    }

    #[test]
    fn test_detect_entity_related_to() {
        let types = detect_query_types("libraries related to async runtime");
        assert!(types.contains(&QueryType::Entity));
    }

    #[test]
    fn test_detect_entity_works_with() {
        let types = detect_query_types("what databases work with Rust");
        assert!(types.contains(&QueryType::Entity));
    }

    #[test]
    fn test_detect_entity_framework() {
        let types = detect_query_types("which framework should we use");
        assert!(types.contains(&QueryType::Entity));
    }

    // =========================================================================
    // INTENT DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_detect_intent_goal() {
        let types = detect_query_types("what was the goal of the refactor");
        assert!(types.contains(&QueryType::Intent));
    }

    #[test]
    fn test_detect_intent_trying_to() {
        let types = detect_query_types("what were we trying to accomplish");
        assert!(types.contains(&QueryType::Intent));
    }

    #[test]
    fn test_detect_intent_purpose() {
        let types = detect_query_types("the purpose of this module");
        assert!(types.contains(&QueryType::Intent));
    }

    #[test]
    fn test_detect_intent_objective() {
        let types = detect_query_types("find memories with similar objectives");
        assert!(types.contains(&QueryType::Intent));
    }

    // =========================================================================
    // KEYWORD DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_detect_keyword_quoted() {
        let types = detect_query_types("find \"ConnectionRefused\" error");
        assert!(types.contains(&QueryType::Keyword));
    }

    #[test]
    fn test_detect_keyword_exact() {
        let types = detect_query_types("exact error message please");
        assert!(types.contains(&QueryType::Keyword));
    }

    #[test]
    fn test_detect_keyword_error_colon() {
        let types = detect_query_types("error: failed to connect");
        assert!(types.contains(&QueryType::Keyword));
    }

    #[test]
    fn test_detect_keyword_version() {
        let types = detect_query_types("v2.0 compatibility");
        assert!(types.contains(&QueryType::Keyword));
    }

    // =========================================================================
    // TEMPORAL DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_detect_temporal_yesterday() {
        let types = detect_query_types("what did we discuss yesterday");
        assert!(types.contains(&QueryType::Temporal));
    }

    #[test]
    fn test_detect_temporal_before() {
        let types = detect_query_types("messages before the crash");
        assert!(types.contains(&QueryType::Temporal));
    }

    #[test]
    fn test_detect_temporal_last_week() {
        let types = detect_query_types("changes from last week");
        assert!(types.contains(&QueryType::Temporal));
    }

    #[test]
    fn test_detect_temporal_recently() {
        let types = detect_query_types("recently modified files");
        assert!(types.contains(&QueryType::Temporal));
    }

    // =========================================================================
    // GENERAL/COMBINED DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_detect_general_no_patterns() {
        let types = detect_query_types("something random here");
        assert_eq!(types, vec![QueryType::General]);
    }

    #[test]
    fn test_detect_multiple_types() {
        // This query has both causal ("why") and code ("function")
        let types = detect_query_types("why did the function fail");
        assert!(types.len() >= 2);
        assert!(types.contains(&QueryType::Causal));
        assert!(types.contains(&QueryType::Code));
    }

    #[test]
    fn test_detect_entity_and_code() {
        let types = detect_query_types("what is tokio::spawn");
        assert!(types.contains(&QueryType::Entity));
        assert!(types.contains(&QueryType::Code));
    }

    // =========================================================================
    // EMBEDDER SELECTION TESTS
    // =========================================================================

    #[test]
    fn test_select_embedders_off_mode() {
        let embedders = select_embedders_for_types(&[QueryType::Causal], EnrichmentMode::Off);
        assert!(embedders.is_empty());
    }

    #[test]
    fn test_select_embedders_causal() {
        let embedders = select_embedders_for_types(&[QueryType::Causal], EnrichmentMode::Light);
        assert!(embedders.contains(&EmbedderId::E5));
    }

    #[test]
    fn test_select_embedders_code() {
        let embedders = select_embedders_for_types(&[QueryType::Code], EnrichmentMode::Light);
        assert!(embedders.contains(&EmbedderId::E7));
    }

    #[test]
    fn test_select_embedders_entity() {
        let embedders = select_embedders_for_types(&[QueryType::Entity], EnrichmentMode::Light);
        assert!(embedders.contains(&EmbedderId::E11));
    }

    #[test]
    fn test_select_embedders_intent() {
        let embedders = select_embedders_for_types(&[QueryType::Intent], EnrichmentMode::Light);
        assert!(embedders.contains(&EmbedderId::E10));
    }

    #[test]
    fn test_select_embedders_light_max_2() {
        let embedders = select_embedders_for_types(
            &[QueryType::Causal, QueryType::Code, QueryType::Entity],
            EnrichmentMode::Light,
        );
        assert!(embedders.len() <= 2);
    }

    #[test]
    fn test_select_embedders_full_includes_secondary() {
        let embedders = select_embedders_for_types(&[QueryType::Causal], EnrichmentMode::Full);
        assert!(embedders.contains(&EmbedderId::E5)); // Primary
        assert!(embedders.contains(&EmbedderId::E8)); // Secondary (graph for causal chains)
    }

    #[test]
    fn test_select_embedders_full_max_6() {
        let embedders = select_embedders_for_types(
            &[QueryType::Causal, QueryType::Code, QueryType::Entity, QueryType::Intent],
            EnrichmentMode::Full,
        );
        assert!(embedders.len() <= 6);
    }

    // =========================================================================
    // CONFIG BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_build_config_off() {
        let config = build_enrichment_config("test query", EnrichmentMode::Off);
        assert_eq!(config.mode, EnrichmentMode::Off);
        assert!(config.selected_embedders.is_empty());
        assert!(!config.detect_blind_spots);
    }

    #[test]
    fn test_build_config_light_causal() {
        let config = build_enrichment_config("why did this fail", EnrichmentMode::Light);
        assert_eq!(config.mode, EnrichmentMode::Light);
        assert!(config.detected_types.contains(&QueryType::Causal));
        assert!(config.selected_embedders.contains(&EmbedderId::E5));
        assert!(!config.detect_blind_spots);
    }

    #[test]
    fn test_build_config_full_enables_blind_spots() {
        let config = build_enrichment_config("what is Diesel", EnrichmentMode::Full);
        assert_eq!(config.mode, EnrichmentMode::Full);
        assert!(config.detect_blind_spots);
    }

    // =========================================================================
    // HELPER FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_count_quoted_terms() {
        assert_eq!(count_quoted_terms("\"hello\" world"), 1);
        assert_eq!(count_quoted_terms("\"one\" and \"two\""), 2);
        assert_eq!(count_quoted_terms("no quotes here"), 0);
    }

    #[test]
    fn test_count_code_indicators() {
        assert!(count_code_indicators("tokio::spawn") > 0);
        assert!(count_code_indicators("async fn main") > 0);
        assert!(count_code_indicators("result.await") > 0);
        assert_eq!(count_code_indicators("no code here"), 0);
    }

    #[test]
    fn test_count_capitalized_words() {
        assert!(count_capitalized_words("Diesel and PostgreSQL") >= 2);
        assert!(count_capitalized_words("lowercase only") == 0);
        // Should not count common starters
        assert!(count_capitalized_words("The quick brown fox") == 0);
    }

    #[test]
    fn test_is_common_start_word() {
        assert!(is_common_start_word("The"));
        assert!(is_common_start_word("What"));
        assert!(is_common_start_word("Why"));
        assert!(!is_common_start_word("Diesel"));
        assert!(!is_common_start_word("PostgreSQL"));
    }

    // =========================================================================
    // AUTO-UPGRADE DETECTION TESTS
    // =========================================================================

    #[test]
    fn test_auto_upgrade_quoted_terms() {
        // Single quoted term → upgrade
        assert!(should_auto_upgrade_to_pipeline("find \"ConnectionRefused\" error"));
        assert!(should_auto_upgrade_to_pipeline("search for 'exact match'"));
        println!("[PASS] Quoted terms trigger auto-upgrade to Pipeline");
    }

    #[test]
    fn test_auto_upgrade_multiple_quoted() {
        // Multiple quoted terms → upgrade
        assert!(should_auto_upgrade_to_pipeline("find \"error\" and \"timeout\""));
        println!("[PASS] Multiple quoted terms trigger auto-upgrade to Pipeline");
    }

    #[test]
    fn test_auto_upgrade_keyword_patterns() {
        // Multiple keyword patterns → upgrade
        // "exact" + "error message" = 2 patterns
        assert!(should_auto_upgrade_to_pipeline("exact error message please"));
        // "verbatim" + "specific" = 2 patterns
        assert!(should_auto_upgrade_to_pipeline("verbatim specific text"));
        println!("[PASS] Keyword patterns trigger auto-upgrade to Pipeline");
    }

    #[test]
    fn test_no_auto_upgrade_generic_query() {
        // Generic queries → no upgrade (stay on MultiSpace)
        assert!(!should_auto_upgrade_to_pipeline("what is authentication"));
        assert!(!should_auto_upgrade_to_pipeline("how does the system work"));
        assert!(!should_auto_upgrade_to_pipeline("find database queries"));
        println!("[PASS] Generic queries do not auto-upgrade");
    }

    #[test]
    fn test_no_auto_upgrade_single_keyword() {
        // Single keyword pattern (below threshold) → no upgrade
        assert!(!should_auto_upgrade_to_pipeline("exact match")); // Only 1 pattern
        println!("[PASS] Single keyword pattern does not auto-upgrade");
    }

    #[test]
    fn test_auto_upgrade_error_patterns() {
        // Error messages often need precision
        // "error:" + "specific" = 2 patterns
        assert!(should_auto_upgrade_to_pipeline("specific error: connection failed"));
        println!("[PASS] Error patterns can trigger auto-upgrade");
    }
}
