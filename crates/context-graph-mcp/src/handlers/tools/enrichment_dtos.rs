//! DTOs for Autonomous Multi-Embedder Enrichment.
//!
//! Per Constitution v6.4: Each of the 13 embedders looks at a query from its unique angle.
//! E1 (semantic) is the foundation, but other embedders find what E1 misses:
//! - E5 (causal): "the bug that caused the crash"
//! - E7 (code): structural code patterns
//! - E10 (intent): same goal, different words
//! - E11 (entity): "Diesel" = database ORM for Rust
//!
//! This module provides DTOs for the enrichment pipeline that autonomously:
//! 1. Detects query types (causal, code, entity, intent, keyword)
//! 2. Selects relevant enhancer embedders
//! 3. Runs E1 foundation + parallel enhancer searches
//! 4. Combines via Weighted RRF (per ARCH-21)
//! 5. Reports agreement metrics and blind spots
//!
//! # FAIL FAST Principle
//! All validation errors are immediate and detailed. No silent fallbacks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::embedder_dtos::EmbedderId;

// =============================================================================
// ENRICHMENT MODE
// =============================================================================

/// Enrichment mode controls how much multi-embedder analysis is performed.
///
/// Per PRD v6.3: The system auto-detects query type and selects embedders.
/// EnrichmentMode controls the depth of this analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnrichmentMode {
    /// E1-only search (legacy behavior, no enrichment).
    /// Use when: Maximum speed required, simple queries.
    Off,

    /// E1 + 1-2 enhancers based on query type, basic agreement metrics.
    /// Use when: Balanced speed/insight tradeoff.
    /// Target latency: <500ms
    #[default]
    Light,

    /// All relevant embedders, full metrics, blind spot detection.
    /// Use when: Maximum insight required, complex queries.
    /// Target latency: <800ms
    Full,
}

impl EnrichmentMode {
    /// Parse from string. Returns None if invalid.
    ///
    /// # FAIL FAST: Returns None for unknown values.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "off" | "none" | "disabled" => Some(EnrichmentMode::Off),
            "light" | "default" => Some(EnrichmentMode::Light),
            "full" | "complete" | "max" => Some(EnrichmentMode::Full),
            _ => None,
        }
    }

    /// Check if this mode enables any enrichment.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, EnrichmentMode::Off)
    }

    /// Check if this mode enables blind spot detection.
    pub fn enables_blind_spots(&self) -> bool {
        matches!(self, EnrichmentMode::Full)
    }

    /// Get the maximum number of enhancer embedders to use.
    pub fn max_enhancers(&self) -> usize {
        match self {
            EnrichmentMode::Off => 0,
            EnrichmentMode::Light => 2,
            EnrichmentMode::Full => 6, // All non-E1 semantic enhancers
        }
    }
}

// =============================================================================
// QUERY TYPE DETECTION
// =============================================================================

/// Detected query type that maps to specific enhancer embedders.
///
/// Per Constitution retrieval section:
/// - use_E5: "Causal queries (why, what caused)"
/// - use_E7: "Code queries (implementations, functions)"
/// - use_E10: "Intent queries (same goal, similar purpose)"
/// - use_E11: "Entity queries (specific named things)"
/// - use_E6_E13: "Keyword queries (exact terms, jargon)"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QueryType {
    /// Causal queries: "why", "caused", "because", "led to"
    /// Enhancers: E5 (causal), E8 (graph for causal chains)
    Causal,

    /// Code queries: function names, imports, code patterns
    /// Enhancers: E7 (code), E6 (keyword for exact matches)
    Code,

    /// Entity queries: capitalized names, known entities
    /// Enhancers: E11 (entity via KEPLER), E6 (keyword)
    Entity,

    /// Intent queries: "goal", "purpose", "trying to", "accomplish"
    /// Enhancers: E10 (intent/multimodal)
    Intent,

    /// Keyword queries: quoted terms, technical jargon, exact phrases
    /// Enhancers: E6 (sparse keyword), E13 (SPLADE expansion)
    Keyword,

    /// Temporal queries: "before", "after", "yesterday", "last week"
    /// Note: E2-E4 are POST-RETRIEVAL ONLY per ARCH-25, not used in fusion
    Temporal,

    /// General semantic query with no specific type detected.
    /// Uses E1 foundation only (or minimal enhancers in Light mode).
    General,
}

impl QueryType {
    /// Get the primary enhancer embedders for this query type.
    pub fn primary_enhancers(&self) -> Vec<EmbedderId> {
        match self {
            QueryType::Causal => vec![EmbedderId::E5],
            QueryType::Code => vec![EmbedderId::E7],
            QueryType::Entity => vec![EmbedderId::E11],
            QueryType::Intent => vec![EmbedderId::E10],
            QueryType::Keyword => vec![EmbedderId::E6, EmbedderId::E13],
            QueryType::Temporal => vec![], // E2-E4 are post-retrieval only
            QueryType::General => vec![],
        }
    }

    /// Get secondary enhancer embedders (for Full mode).
    pub fn secondary_enhancers(&self) -> Vec<EmbedderId> {
        match self {
            QueryType::Causal => vec![EmbedderId::E8], // Graph for causal chains
            QueryType::Code => vec![EmbedderId::E6],   // Keywords for function names
            QueryType::Entity => vec![EmbedderId::E6], // Keywords for entity names
            QueryType::Intent => vec![EmbedderId::E5], // Causal intent often overlaps
            QueryType::Keyword => vec![],
            QueryType::Temporal => vec![],
            QueryType::General => vec![EmbedderId::E5, EmbedderId::E10], // Best general enhancers
        }
    }

    /// Human-readable description of what this query type represents.
    pub fn description(&self) -> &'static str {
        match self {
            QueryType::Causal => "Causal/explanatory query (why, what caused)",
            QueryType::Code => "Code/implementation query (functions, patterns)",
            QueryType::Entity => "Entity/factual query (named things, concepts)",
            QueryType::Intent => "Intent/goal query (purpose, trying to accomplish)",
            QueryType::Keyword => "Keyword/exact match query (specific terms)",
            QueryType::Temporal => "Temporal query (time-based, sequence)",
            QueryType::General => "General semantic query (no specific type)",
        }
    }
}

// =============================================================================
// ENRICHMENT CONFIGURATION
// =============================================================================

/// Configuration for the enrichment pipeline.
///
/// Built by the query type detector, consumed by the enrichment pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnrichmentConfig {
    /// The enrichment mode (Off, Light, Full).
    pub mode: EnrichmentMode,

    /// Detected query types (may be multiple).
    pub detected_types: Vec<QueryType>,

    /// Selected enhancer embedders based on detected types and mode.
    /// Always excludes E1 (foundation is always used).
    /// Always excludes E12/E13 (pipeline stages only per ARCH-13).
    pub selected_embedders: Vec<EmbedderId>,

    /// Whether to compute blind spot detection.
    pub detect_blind_spots: bool,
}

impl EnrichmentConfig {
    /// Create a new config with Off mode (no enrichment).
    pub fn off() -> Self {
        Self {
            mode: EnrichmentMode::Off,
            detected_types: vec![QueryType::General],
            selected_embedders: vec![],
            detect_blind_spots: false,
        }
    }

    /// Create a new config for Light mode with given detected types.
    pub fn light(detected_types: Vec<QueryType>) -> Self {
        let mut embedders = Vec::new();
        for qt in &detected_types {
            embedders.extend(qt.primary_enhancers());
        }
        // Deduplicate and limit to 2
        embedders.sort_by_key(|e| e.to_index());
        embedders.dedup();
        embedders.truncate(2);

        Self {
            mode: EnrichmentMode::Light,
            detected_types,
            selected_embedders: embedders,
            detect_blind_spots: false,
        }
    }

    /// Create a new config for Full mode with given detected types.
    pub fn full(detected_types: Vec<QueryType>) -> Self {
        let mut embedders = Vec::new();
        for qt in &detected_types {
            embedders.extend(qt.primary_enhancers());
            embedders.extend(qt.secondary_enhancers());
        }
        // Deduplicate and limit to 6
        embedders.sort_by_key(|e| e.to_index());
        embedders.dedup();
        embedders.truncate(6);

        Self {
            mode: EnrichmentMode::Full,
            detected_types,
            selected_embedders: embedders,
            detect_blind_spots: true,
        }
    }

    /// Get the number of selected enhancers.
    pub fn enhancer_count(&self) -> usize {
        self.selected_embedders.len()
    }
}

// =============================================================================
// AGREEMENT METRICS
// =============================================================================

/// Metrics showing which embedders agree on a result.
///
/// Per Constitution topic system: weighted_agreement >= 2.5 indicates topic formation.
/// This applies the same principle to search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgreementMetrics {
    /// Which embedders found this result with similarity above threshold.
    pub embedders_agree: Vec<EmbedderId>,

    /// Raw agreement score (0.0-1.0): fraction of queried embedders that agree.
    pub agreement_score: f32,

    /// Constitution-weighted agreement using topic category weights:
    /// - SEMANTIC (E1,E5,E6,E7,E10,E12,E13): 1.0
    /// - RELATIONAL (E8,E11): 0.5
    /// - TEMPORAL (E2-E4): 0.0
    /// - STRUCTURAL (E9): 0.5
    pub weighted_agreement: f32,

    /// Number of embedders that found this result.
    pub embedder_count: usize,
}

impl AgreementMetrics {
    /// Create new agreement metrics from a list of embedders that found the result.
    pub fn from_embedders(embedders_agree: Vec<EmbedderId>, total_queried: usize) -> Self {
        let embedder_count = embedders_agree.len();
        let agreement_score = if total_queried > 0 {
            embedder_count as f32 / total_queried as f32
        } else {
            0.0
        };

        // Compute weighted agreement per Constitution topic system
        let weighted_agreement = embedders_agree
            .iter()
            .map(|e| e.topic_weight())
            .sum();

        Self {
            embedders_agree,
            agreement_score,
            weighted_agreement,
            embedder_count,
        }
    }

    /// Check if this result has strong agreement (meets topic threshold).
    pub fn is_strong_agreement(&self) -> bool {
        self.weighted_agreement >= 2.5
    }
}

// =============================================================================
// BLIND SPOT ALERT
// =============================================================================

/// Alert when an enhancer finds something E1 (semantic) missed.
///
/// This is the core value of multi-embedder enrichment:
/// E1 might miss "Diesel" when searching for "database" because Diesel
/// doesn't contain the word "database". E11 (entity) knows Diesel IS
/// a database ORM and surfaces it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlindSpotAlert {
    /// The memory that was found by enhancers but missed/low-scored by E1.
    pub node_id: Uuid,

    /// Which embedders found this result with high similarity.
    pub found_by: Vec<EmbedderId>,

    /// Whether E1 completely missed this (similarity < 0.1).
    pub missed_by_e1: bool,

    /// E1's similarity score for this result (low if blind spot).
    pub e1_score: f32,

    /// Enhancer similarity scores.
    pub enhancer_scores: HashMap<String, f32>,

    /// Human-readable explanation of why this is a blind spot.
    pub explanation: String,
}

impl BlindSpotAlert {
    /// Create a new blind spot alert.
    pub fn new(
        node_id: Uuid,
        found_by: Vec<EmbedderId>,
        e1_score: f32,
        enhancer_scores: HashMap<String, f32>,
    ) -> Self {
        let missed_by_e1 = e1_score < 0.1;
        let top_enhancer = found_by.first().map(|e| e.name()).unwrap_or("unknown");
        let top_score = enhancer_scores.values().cloned().fold(0.0f32, f32::max);

        let explanation = if missed_by_e1 {
            format!(
                "{} found this (score {:.2}) that E1 completely missed (score {:.2})",
                top_enhancer, top_score, e1_score
            )
        } else {
            format!(
                "{} found this (score {:.2}) that E1 ranked low (score {:.2})",
                top_enhancer, top_score, e1_score
            )
        };

        Self {
            node_id,
            found_by,
            missed_by_e1,
            e1_score,
            enhancer_scores,
            explanation,
        }
    }

    /// Check if this is a critical blind spot (E1 completely missed).
    pub fn is_critical(&self) -> bool {
        self.missed_by_e1
    }
}

// =============================================================================
// SCORING BREAKDOWN
// =============================================================================

/// Detailed breakdown of how a result's final score was computed.
///
/// Per ARCH-21: Uses Weighted RRF, not weighted sum.
/// Per ARCH-17: E10 uses multiplicative boost, not linear blending.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoringBreakdown {
    /// E1 (semantic) similarity score - the foundation.
    pub e1_score: f32,

    /// Scores from each enhancer embedder used.
    /// Key format: "e5", "e7", "e10", "e11", etc.
    pub enhancer_scores: HashMap<String, f32>,

    /// Final score after Weighted RRF fusion.
    pub rrf_final: f32,

    /// E10 multiplicative boost applied (if any).
    /// Per ARCH-28: E10 uses multiplicative boost: E1 * (1 + boost)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub e10_boost_applied: Option<f32>,

    /// RRF rank contributions from each embedder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rrf_contributions: Option<HashMap<String, f32>>,
}

impl ScoringBreakdown {
    /// Create a new scoring breakdown for E1-only search.
    pub fn e1_only(e1_score: f32) -> Self {
        Self {
            e1_score,
            enhancer_scores: HashMap::new(),
            rrf_final: e1_score,
            e10_boost_applied: None,
            rrf_contributions: None,
        }
    }

    /// Create a full scoring breakdown with RRF fusion.
    pub fn with_rrf(
        e1_score: f32,
        enhancer_scores: HashMap<String, f32>,
        rrf_final: f32,
        e10_boost: Option<f32>,
        rrf_contributions: Option<HashMap<String, f32>>,
    ) -> Self {
        Self {
            e1_score,
            enhancer_scores,
            rrf_final,
            e10_boost_applied: e10_boost,
            rrf_contributions,
        }
    }

    /// Check if any enhancers scored higher than E1.
    pub fn has_enhancer_advantage(&self) -> bool {
        self.enhancer_scores.values().any(|&s| s > self.e1_score)
    }

    /// Get the highest scoring enhancer.
    pub fn top_enhancer(&self) -> Option<(&str, f32)> {
        self.enhancer_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.as_str(), *v))
    }
}

// =============================================================================
// ENRICHED SEARCH RESULT
// =============================================================================

/// A single search result enriched with multi-embedder insights.
///
/// This is the return type for enriched search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnrichedSearchResult {
    /// Memory/fingerprint ID.
    pub node_id: Uuid,

    /// Content text (if includeContent=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Detailed scoring breakdown.
    pub scoring: ScoringBreakdown,

    /// Agreement metrics showing which embedders found this result.
    pub agreement: AgreementMetrics,

    /// Blind spot alert if this was found by enhancers but missed by E1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blind_spot: Option<BlindSpotAlert>,

    /// Source metadata (file path, chunk info, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,

    /// Which embedder contributed most to this result's ranking.
    pub dominant_embedder: String,
}

impl EnrichedSearchResult {
    /// Create a new enriched result.
    pub fn new(
        node_id: Uuid,
        scoring: ScoringBreakdown,
        agreement: AgreementMetrics,
    ) -> Self {
        let dominant_embedder = if let Some((name, _)) = scoring.top_enhancer() {
            if scoring.has_enhancer_advantage() {
                name.to_uppercase()
            } else {
                "E1".to_string()
            }
        } else {
            "E1".to_string()
        };

        Self {
            node_id,
            content: None,
            scoring,
            agreement,
            blind_spot: None,
            source: None,
            dominant_embedder,
        }
    }

    /// Add content to the result.
    pub fn with_content(mut self, content: String) -> Self {
        self.content = Some(content);
        self
    }

    /// Add blind spot alert.
    pub fn with_blind_spot(mut self, alert: BlindSpotAlert) -> Self {
        self.blind_spot = Some(alert);
        self
    }

    /// Add source metadata.
    pub fn with_source(mut self, source: serde_json::Value) -> Self {
        self.source = Some(source);
        self
    }

    /// Check if this result was primarily found by an enhancer (not E1).
    pub fn is_enhancer_discovery(&self) -> bool {
        self.dominant_embedder != "E1"
    }
}

// =============================================================================
// ENRICHMENT SUMMARY
// =============================================================================

/// Summary of the enrichment process for a search operation.
///
/// Provides high-level insights about how the enrichment improved results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnrichmentSummary {
    /// The enrichment mode used.
    pub mode: EnrichmentMode,

    /// Detected query types.
    pub detected_types: Vec<QueryType>,

    /// Which embedders were used (always includes E1).
    pub embedders_used: Vec<EmbedderId>,

    /// Number of blind spots found (results E1 missed).
    pub blind_spots_found: usize,

    /// Total unique discoveries by enhancers.
    /// These are results that ranked significantly higher with enhancers.
    pub unique_discoveries: usize,

    /// Total time for enrichment pipeline (ms).
    pub enrichment_time_ms: u64,

    /// Breakdown of time spent in each phase.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_breakdown: Option<TimeBreakdown>,

    /// Whether E9 HDC fallback was triggered due to weak E1 results.
    /// Trigger condition: E1 top score < 0.4 (per remaining_embedder_integration_plan.md).
    #[serde(default)]
    pub hdc_fallback_used: bool,
}

/// Timing breakdown for the enrichment pipeline phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeBreakdown {
    /// E1 foundation search time (ms).
    pub e1_search_ms: u64,

    /// Parallel enhancer searches time (ms).
    pub enhancer_search_ms: u64,

    /// RRF fusion time (ms).
    pub rrf_fusion_ms: u64,

    /// Agreement calculation time (ms).
    pub agreement_ms: u64,

    /// Blind spot detection time (ms).
    pub blind_spot_ms: u64,
}

impl EnrichmentSummary {
    /// Create a summary for Off mode (no enrichment).
    pub fn off(search_time_ms: u64) -> Self {
        Self {
            mode: EnrichmentMode::Off,
            detected_types: vec![QueryType::General],
            embedders_used: vec![EmbedderId::E1],
            blind_spots_found: 0,
            unique_discoveries: 0,
            enrichment_time_ms: search_time_ms,
            time_breakdown: None,
            hdc_fallback_used: false,
        }
    }

    /// Create a full summary with all fields.
    pub fn new(
        mode: EnrichmentMode,
        detected_types: Vec<QueryType>,
        embedders_used: Vec<EmbedderId>,
        blind_spots_found: usize,
        unique_discoveries: usize,
        enrichment_time_ms: u64,
        time_breakdown: Option<TimeBreakdown>,
    ) -> Self {
        Self {
            mode,
            detected_types,
            embedders_used,
            blind_spots_found,
            unique_discoveries,
            enrichment_time_ms,
            time_breakdown,
            hdc_fallback_used: false,
        }
    }

    /// Create a full summary with HDC fallback flag.
    pub fn new_with_fallback(
        mode: EnrichmentMode,
        detected_types: Vec<QueryType>,
        embedders_used: Vec<EmbedderId>,
        blind_spots_found: usize,
        unique_discoveries: usize,
        enrichment_time_ms: u64,
        time_breakdown: Option<TimeBreakdown>,
        hdc_fallback_used: bool,
    ) -> Self {
        Self {
            mode,
            detected_types,
            embedders_used,
            blind_spots_found,
            unique_discoveries,
            enrichment_time_ms,
            time_breakdown,
            hdc_fallback_used,
        }
    }

    /// Check if enrichment found any blind spots.
    pub fn has_blind_spots(&self) -> bool {
        self.blind_spots_found > 0
    }

    /// Check if enrichment improved results significantly.
    pub fn improved_results(&self) -> bool {
        self.unique_discoveries > 0 || self.blind_spots_found > 0
    }
}

// =============================================================================
// ENRICHED SEARCH RESPONSE
// =============================================================================

/// Full response for an enriched search operation.
///
/// This is the top-level response type returned by search_graph with enrichMode enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnrichedSearchResponse {
    /// Search results with enrichment data.
    pub results: Vec<EnrichedSearchResult>,

    /// Total number of results.
    pub count: usize,

    /// Enrichment summary.
    pub enrichment: EnrichmentSummary,

    /// The original query.
    pub query: String,

    /// Search strategy used (e1_only, multi_space, pipeline).
    pub search_strategy: String,
}

impl EnrichedSearchResponse {
    /// Create a new enriched search response.
    pub fn new(
        results: Vec<EnrichedSearchResult>,
        enrichment: EnrichmentSummary,
        query: String,
        search_strategy: String,
    ) -> Self {
        let count = results.len();
        Self {
            results,
            count,
            enrichment,
            query,
            search_strategy,
        }
    }
}

// =============================================================================
// EMBEDDERID EXTENSIONS
// =============================================================================

impl EmbedderId {
    /// Get the Constitution topic weight for this embedder.
    ///
    /// Per Constitution topics.categories:
    /// - SEMANTIC (E1,E5,E6,E7,E10,E12,E13): 1.0
    /// - RELATIONAL (E8,E11): 0.5
    /// - STRUCTURAL (E9): 0.5
    /// - TEMPORAL (E2,E3,E4): 0.0
    pub fn topic_weight(&self) -> f32 {
        match self {
            // SEMANTIC: weight 1.0
            EmbedderId::E1 => 1.0,
            EmbedderId::E5 => 1.0,
            EmbedderId::E6 => 1.0,
            EmbedderId::E7 => 1.0,
            EmbedderId::E10 => 1.0,
            EmbedderId::E12 => 1.0,
            EmbedderId::E13 => 1.0,
            // RELATIONAL: weight 0.5
            EmbedderId::E8 => 0.5,
            EmbedderId::E11 => 0.5,
            // STRUCTURAL: weight 0.5
            EmbedderId::E9 => 0.5,
            // TEMPORAL: weight 0.0 (never for topics)
            EmbedderId::E2 => 0.0,
            EmbedderId::E3 => 0.0,
            EmbedderId::E4 => 0.0,
        }
    }

    /// Get short lowercase key for JSON (e.g., "e5", "e11").
    pub fn json_key(&self) -> &'static str {
        match self {
            EmbedderId::E1 => "e1",
            EmbedderId::E2 => "e2",
            EmbedderId::E3 => "e3",
            EmbedderId::E4 => "e4",
            EmbedderId::E5 => "e5",
            EmbedderId::E6 => "e6",
            EmbedderId::E7 => "e7",
            EmbedderId::E8 => "e8",
            EmbedderId::E9 => "e9",
            EmbedderId::E10 => "e10",
            EmbedderId::E11 => "e11",
            EmbedderId::E12 => "e12",
            EmbedderId::E13 => "e13",
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enrichment_mode_parsing() {
        assert_eq!(EnrichmentMode::from_str("off"), Some(EnrichmentMode::Off));
        assert_eq!(EnrichmentMode::from_str("light"), Some(EnrichmentMode::Light));
        assert_eq!(EnrichmentMode::from_str("full"), Some(EnrichmentMode::Full));
        assert_eq!(EnrichmentMode::from_str("FULL"), Some(EnrichmentMode::Full));
        assert_eq!(EnrichmentMode::from_str("invalid"), None);
    }

    #[test]
    fn test_enrichment_mode_properties() {
        assert!(!EnrichmentMode::Off.is_enabled());
        assert!(EnrichmentMode::Light.is_enabled());
        assert!(EnrichmentMode::Full.is_enabled());

        assert!(!EnrichmentMode::Off.enables_blind_spots());
        assert!(!EnrichmentMode::Light.enables_blind_spots());
        assert!(EnrichmentMode::Full.enables_blind_spots());

        assert_eq!(EnrichmentMode::Off.max_enhancers(), 0);
        assert_eq!(EnrichmentMode::Light.max_enhancers(), 2);
        assert_eq!(EnrichmentMode::Full.max_enhancers(), 6);
    }

    #[test]
    fn test_query_type_enhancers() {
        let causal = QueryType::Causal.primary_enhancers();
        assert!(causal.contains(&EmbedderId::E5));

        let code = QueryType::Code.primary_enhancers();
        assert!(code.contains(&EmbedderId::E7));

        let entity = QueryType::Entity.primary_enhancers();
        assert!(entity.contains(&EmbedderId::E11));

        let intent = QueryType::Intent.primary_enhancers();
        assert!(intent.contains(&EmbedderId::E10));

        // Temporal should not have enhancers (post-retrieval only)
        let temporal = QueryType::Temporal.primary_enhancers();
        assert!(temporal.is_empty());
    }

    #[test]
    fn test_enrichment_config_light() {
        let config = EnrichmentConfig::light(vec![QueryType::Causal, QueryType::Code]);
        assert_eq!(config.mode, EnrichmentMode::Light);
        assert!(config.selected_embedders.len() <= 2);
        assert!(!config.detect_blind_spots);
    }

    #[test]
    fn test_enrichment_config_full() {
        let config = EnrichmentConfig::full(vec![QueryType::Causal]);
        assert_eq!(config.mode, EnrichmentMode::Full);
        assert!(config.detect_blind_spots);
        // Should include E5 (primary) and E8 (secondary)
        assert!(config.selected_embedders.contains(&EmbedderId::E5));
    }

    #[test]
    fn test_agreement_metrics() {
        let embedders = vec![EmbedderId::E1, EmbedderId::E5, EmbedderId::E11];
        let metrics = AgreementMetrics::from_embedders(embedders, 5);

        assert_eq!(metrics.embedder_count, 3);
        assert!((metrics.agreement_score - 0.6).abs() < 0.01);
        // E1: 1.0 + E5: 1.0 + E11: 0.5 = 2.5
        assert!((metrics.weighted_agreement - 2.5).abs() < 0.01);
        assert!(metrics.is_strong_agreement());
    }

    #[test]
    fn test_blind_spot_alert() {
        let mut enhancer_scores = HashMap::new();
        enhancer_scores.insert("e11".to_string(), 0.85);

        let alert = BlindSpotAlert::new(
            Uuid::new_v4(),
            vec![EmbedderId::E11],
            0.05,
            enhancer_scores,
        );

        assert!(alert.is_critical());
        assert!(alert.missed_by_e1);
        assert!(alert.explanation.contains("completely missed"));
    }

    #[test]
    fn test_scoring_breakdown() {
        let mut enhancer_scores = HashMap::new();
        enhancer_scores.insert("e5".to_string(), 0.8);
        enhancer_scores.insert("e7".to_string(), 0.6);

        let breakdown = ScoringBreakdown::with_rrf(0.7, enhancer_scores, 0.75, None, None);

        assert!(breakdown.has_enhancer_advantage()); // E5: 0.8 > E1: 0.7
        let (top, score) = breakdown.top_enhancer().unwrap();
        assert_eq!(top, "e5");
        assert!((score - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_enriched_search_result() {
        let scoring = ScoringBreakdown::e1_only(0.9);
        let agreement = AgreementMetrics::from_embedders(vec![EmbedderId::E1], 1);

        let result = EnrichedSearchResult::new(Uuid::new_v4(), scoring, agreement);

        assert_eq!(result.dominant_embedder, "E1");
        assert!(!result.is_enhancer_discovery());
    }

    #[test]
    fn test_embedder_topic_weights() {
        // SEMANTIC = 1.0
        assert!((EmbedderId::E1.topic_weight() - 1.0).abs() < 0.001);
        assert!((EmbedderId::E5.topic_weight() - 1.0).abs() < 0.001);
        assert!((EmbedderId::E7.topic_weight() - 1.0).abs() < 0.001);

        // RELATIONAL = 0.5
        assert!((EmbedderId::E8.topic_weight() - 0.5).abs() < 0.001);
        assert!((EmbedderId::E11.topic_weight() - 0.5).abs() < 0.001);

        // TEMPORAL = 0.0
        assert!((EmbedderId::E2.topic_weight() - 0.0).abs() < 0.001);
        assert!((EmbedderId::E3.topic_weight() - 0.0).abs() < 0.001);
        assert!((EmbedderId::E4.topic_weight() - 0.0).abs() < 0.001);
    }
}
