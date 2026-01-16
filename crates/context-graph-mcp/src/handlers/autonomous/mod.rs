//! NORTH Autonomous System MCP Handlers
//!
//! TASK-AUTONOMOUS-MCP: MCP tool handlers for autonomous topic-based management.
//! SPEC-AUTONOMOUS-001: Added 5 new tools (learner, health, execute_prune).
//! TASK-P0-001: REMOVED auto_bootstrap_north_star per ARCH-03 (goals emerge from topic clustering).
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.
//!
//! ## Tools Implemented (12 tools after TASK-P0-001 removal)
//!
//! 1. `get_alignment_drift` - Get drift state and history using DriftDetector
//! 2. `trigger_drift_correction` - Manually trigger correction using DriftCorrector
//! 3. `get_pruning_candidates` - Get memories for potential pruning using PruningService
//! 4. `trigger_consolidation` - Trigger memory consolidation using ConsolidationService
//! 5. `discover_sub_goals` - Discover potential sub-goals using SubGoalDiscovery
//! 6. `get_autonomous_status` - Get comprehensive status from all services
//! 7. `get_learner_state` - Get Meta-UTL learner state (NORTH-009)
//! 8. `observe_outcome` - Record learning outcome (NORTH-009)
//! 9. `execute_prune` - Execute pruning on candidates (NORTH-012)
//! 10. `get_health_status` - Get system-wide health (NORTH-020)
//! 11. `trigger_healing` - Trigger self-healing (NORTH-020)
//! 12. `get_drift_history` - Get drift history (TASK-FIX-002/NORTH-010)
//!
//! ## FAIL FAST Policy
//!
//! - NO MOCK DATA - all calls go to real services
//! - NO FALLBACKS - errors propagate with full context
//! - All errors include operation context for debugging
//!
//! ## Module Organization
//!
//! - `params`: Parameter structs and default value functions
//! - `error_codes`: Autonomous-specific error codes
//! - `bootstrap`: Bootstrap handler implementation
//! - `drift`: Drift detection and correction handlers
//! - `maintenance`: Pruning, consolidation, and execute_prune handlers
//! - `discovery`: Sub-goal discovery handler
//! - `status`: Autonomous status handler
//! - `learner`: Meta-UTL learner state and outcome handlers (SPEC-AUTONOMOUS-001)
//! - `health`: System health and healing handlers (SPEC-AUTONOMOUS-001)

mod bootstrap;
mod discovery;
mod drift;
mod error_codes;
mod health;
mod learner;
mod maintenance;
mod params;
mod prediction_history;
mod status;

// TASK-004: Re-export PredictionHistory for use in Handlers struct
pub use prediction_history::{PredictionEntry, PredictionHistory};

#[cfg(test)]
mod tests;

// Re-export all parameter structs for backwards compatibility
// TASK-P0-001: Removed AutoBootstrapParams (ARCH-03)
#[allow(unused_imports)]
pub use params::{
    DiscoverSubGoalsParams, ExecutePruneParams, GetAlignmentDriftParams,
    GetAutonomousStatusParams, GetDriftHistoryParams, GetHealthStatusParams, GetLearnerStateParams,
    GetPruningCandidatesParams, ObserveOutcomeParams, TriggerConsolidationParams,
    TriggerDriftCorrectionParams, TriggerHealingParams,
};

// Re-export error codes module for backwards compatibility
#[allow(unused_imports)]
pub use error_codes::autonomous_error_codes;
