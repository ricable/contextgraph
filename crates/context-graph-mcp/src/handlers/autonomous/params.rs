//! Parameter structs for autonomous MCP handlers.
//!
//! All parameter structs follow the teleological.rs pattern with serde defaults.

use serde::Deserialize;

// ============================================================================
// Default Value Functions
// ============================================================================

pub(super) fn default_confidence_threshold() -> f32 {
    0.7
}

pub(super) fn default_max_candidates() -> usize {
    10
}

pub(super) fn default_timeframe() -> String {
    "24h".to_string()
}

pub(super) fn default_pruning_limit() -> usize {
    20
}

pub(super) fn default_min_staleness_days() -> u64 {
    30
}

pub(super) fn default_min_alignment() -> f32 {
    0.4
}

pub(super) fn default_max_memories() -> usize {
    100
}

pub(super) fn default_consolidation_strategy() -> String {
    "similarity".to_string()
}

pub(super) fn default_consolidation_similarity() -> f32 {
    0.85
}

pub(super) fn default_min_subgoal_confidence() -> f32 {
    0.6
}

pub(super) fn default_max_subgoals() -> usize {
    5
}

pub(super) fn default_history_count() -> usize {
    10
}

// ============================================================================
// Parameter Structs
// ============================================================================

/// Parameters for auto_bootstrap_north_star tool.
#[derive(Debug, Deserialize)]
pub struct AutoBootstrapParams {
    /// Optional confidence threshold for bootstrapping (default: 0.7)
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,

    /// Optional maximum number of candidates to evaluate (default: 10)
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,
}

/// Parameters for get_alignment_drift tool.
#[derive(Debug, Deserialize)]
pub struct GetAlignmentDriftParams {
    /// Optional timeframe filter: "1h", "24h", "7d", "30d" (default: "24h")
    #[serde(default = "default_timeframe")]
    pub timeframe: String,

    /// Include full drift history in response (default: false)
    #[serde(default)]
    pub include_history: bool,
}

/// Parameters for get_drift_history tool (NORTH-010, TASK-FIX-002).
#[derive(Debug, Deserialize)]
pub struct GetDriftHistoryParams {
    /// Goal UUID to retrieve history for (defaults to North Star)
    #[serde(default)]
    pub goal_id: Option<String>,

    /// Time range filter: "1h", "6h", "24h", "7d", "30d", "all" (default: "24h")
    #[serde(default = "default_timeframe")]
    pub time_range: String,

    /// Maximum entries to return (1-100, default 50)
    #[serde(default = "default_history_limit")]
    pub limit: usize,

    /// Include per-embedder breakdown (default: false)
    #[serde(default)]
    pub include_per_embedder: bool,

    /// Compute deltas between entries (default: true)
    #[serde(default = "default_true")]
    pub compute_deltas: bool,
}

pub(super) fn default_history_limit() -> usize {
    50
}

pub(super) fn default_true() -> bool {
    true
}

/// Parameters for trigger_drift_correction tool.
#[derive(Debug, Deserialize)]
pub struct TriggerDriftCorrectionParams {
    /// Force correction even if drift severity is low (default: false)
    #[serde(default)]
    pub force: bool,

    /// Target alignment to achieve (optional, uses adaptive if not set)
    pub target_alignment: Option<f32>,
}

/// Parameters for get_pruning_candidates tool.
#[derive(Debug, Deserialize)]
pub struct GetPruningCandidatesParams {
    /// Maximum number of candidates to return (default: 20)
    #[serde(default = "default_pruning_limit")]
    pub limit: usize,

    /// Minimum staleness in days for a memory to be considered (default: 30)
    #[serde(default = "default_min_staleness_days")]
    pub min_staleness_days: u64,

    /// Minimum alignment threshold (below this = candidate) (default: 0.4)
    #[serde(default = "default_min_alignment")]
    pub min_alignment: f32,
}

/// Parameters for trigger_consolidation tool.
#[derive(Debug, Deserialize)]
pub struct TriggerConsolidationParams {
    /// Maximum memories to process in one batch (default: 100)
    #[serde(default = "default_max_memories")]
    pub max_memories: usize,

    /// Consolidation strategy: "similarity", "temporal", "semantic" (default: "similarity")
    #[serde(default = "default_consolidation_strategy")]
    pub strategy: String,

    /// Minimum similarity for consolidation (default: 0.85)
    #[serde(default = "default_consolidation_similarity")]
    pub min_similarity: f32,
}

/// Parameters for discover_sub_goals tool.
#[derive(Debug, Deserialize)]
pub struct DiscoverSubGoalsParams {
    /// Minimum confidence for a discovered sub-goal (default: 0.6)
    #[serde(default = "default_min_subgoal_confidence")]
    pub min_confidence: f32,

    /// Maximum number of sub-goals to discover (default: 5)
    #[serde(default = "default_max_subgoals")]
    pub max_goals: usize,

    /// Parent goal ID to discover sub-goals for (optional, uses North Star if not set)
    pub parent_goal_id: Option<String>,
}

/// Parameters for get_autonomous_status tool.
#[derive(Debug, Deserialize)]
pub struct GetAutonomousStatusParams {
    /// Include detailed metrics per service (default: false)
    #[serde(default)]
    pub include_metrics: bool,

    /// Include recent operation history (default: false)
    #[serde(default)]
    pub include_history: bool,

    /// Number of history entries to include (default: 10)
    #[serde(default = "default_history_count")]
    pub history_count: usize,
}

// ============================================================================
// SPEC-AUTONOMOUS-001: Parameter Structs for 5 New Tools
// ============================================================================

/// Parameters for get_learner_state tool.
/// SPEC-AUTONOMOUS-001: NORTH-009, METAUTL-004
#[derive(Debug, Deserialize)]
pub struct GetLearnerStateParams {
    /// Optional domain filter (e.g., "Code", "Medical", "General")
    pub domain: Option<String>,
}

/// Parameters for observe_outcome tool.
/// SPEC-AUTONOMOUS-001: NORTH-009, METAUTL-001
#[derive(Debug, Deserialize)]
pub struct ObserveOutcomeParams {
    /// UUID of the prediction to update
    pub prediction_id: String,

    /// Actual outcome value (0.0-1.0)
    pub actual_outcome: f32,

    /// Optional context for the outcome
    pub context: Option<ObserveOutcomeContext>,
}

/// Context for observe_outcome.
#[derive(Debug, Deserialize)]
pub struct ObserveOutcomeContext {
    /// Domain of the prediction (Code, Medical, General)
    pub domain: Option<String>,

    /// Type of query (retrieval, classification, etc.)
    pub query_type: Option<String>,
}

/// Parameters for execute_prune tool.
/// SPEC-AUTONOMOUS-001: NORTH-012
#[derive(Debug, Deserialize)]
pub struct ExecutePruneParams {
    /// Array of node UUIDs to prune
    pub node_ids: Vec<String>,

    /// Reason for pruning (for audit logging)
    pub reason: String,

    /// Also prune dependent nodes and edges (default: false)
    #[serde(default)]
    pub cascade: bool,
}

/// Parameters for get_health_status tool.
/// SPEC-AUTONOMOUS-001: NORTH-020
#[derive(Debug, Deserialize)]
pub struct GetHealthStatusParams {
    /// Specific subsystem to query, or "all" (default: "all")
    #[serde(default = "default_all_subsystem")]
    pub subsystem: String,
}

pub(super) fn default_all_subsystem() -> String {
    "all".to_string()
}

/// Parameters for trigger_healing tool.
/// SPEC-AUTONOMOUS-001: NORTH-020
#[derive(Debug, Deserialize)]
pub struct TriggerHealingParams {
    /// Subsystem to heal (utl, gwt, dream, storage)
    pub subsystem: String,

    /// Healing severity (affects action aggressiveness)
    #[serde(default = "default_medium_severity")]
    pub severity: String,
}

pub(super) fn default_medium_severity() -> String {
    "medium".to_string()
}
