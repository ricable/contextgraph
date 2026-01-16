//! Tests for autonomous MCP handlers.
//!
//! TASK-P0-001: Removed test_auto_bootstrap_params_defaults per ARCH-03.
//! Goals now emerge autonomously from topic clustering (constitution v6.0.0).

use super::params::*;

// REMOVED: test_auto_bootstrap_params_defaults per TASK-P0-001 (ARCH-03)
// AutoBootstrapParams struct has been removed as goals emerge from topic clustering.

#[test]
fn test_get_alignment_drift_params_defaults() {
    let json = serde_json::json!({});
    let params: GetAlignmentDriftParams = serde_json::from_value(json).unwrap();
    assert_eq!(params.timeframe, "24h");
    assert!(!params.include_history);
    println!("[VERIFIED] GetAlignmentDriftParams defaults work correctly");
}

#[test]
fn test_trigger_drift_correction_params_defaults() {
    let json = serde_json::json!({});
    let params: TriggerDriftCorrectionParams = serde_json::from_value(json).unwrap();
    assert!(!params.force);
    assert!(params.target_alignment.is_none());
    println!("[VERIFIED] TriggerDriftCorrectionParams defaults work correctly");
}

#[test]
fn test_get_pruning_candidates_params_defaults() {
    let json = serde_json::json!({});
    let params: GetPruningCandidatesParams = serde_json::from_value(json).unwrap();
    assert_eq!(params.limit, 20);
    assert_eq!(params.min_staleness_days, 30);
    assert!((params.min_alignment - 0.4).abs() < f32::EPSILON);
    println!("[VERIFIED] GetPruningCandidatesParams defaults work correctly");
}

#[test]
fn test_trigger_consolidation_params_defaults() {
    let json = serde_json::json!({});
    let params: TriggerConsolidationParams = serde_json::from_value(json).unwrap();
    assert_eq!(params.max_memories, 100);
    assert_eq!(params.strategy, "similarity");
    assert!((params.min_similarity - 0.85).abs() < f32::EPSILON);
    println!("[VERIFIED] TriggerConsolidationParams defaults work correctly");
}

#[test]
fn test_discover_sub_goals_params_defaults() {
    let json = serde_json::json!({});
    let params: DiscoverSubGoalsParams = serde_json::from_value(json).unwrap();
    assert!((params.min_confidence - 0.6).abs() < f32::EPSILON);
    assert_eq!(params.max_goals, 5);
    assert!(params.parent_goal_id.is_none());
    println!("[VERIFIED] DiscoverSubGoalsParams defaults work correctly");
}

#[test]
fn test_get_autonomous_status_params_defaults() {
    let json = serde_json::json!({});
    let params: GetAutonomousStatusParams = serde_json::from_value(json).unwrap();
    assert!(!params.include_metrics);
    assert!(!params.include_history);
    assert_eq!(params.history_count, 10);
    println!("[VERIFIED] GetAutonomousStatusParams defaults work correctly");
}

#[test]
fn test_autonomous_error_codes_values() {
    use super::error_codes::autonomous_error_codes;

    // Ensure error codes are in the correct range (-32110 to -32119)
    assert_eq!(autonomous_error_codes::BOOTSTRAP_ERROR, -32110);
    assert_eq!(autonomous_error_codes::DRIFT_DETECTOR_ERROR, -32111);
    assert_eq!(autonomous_error_codes::DRIFT_CORRECTOR_ERROR, -32112);
    assert_eq!(autonomous_error_codes::PRUNING_ERROR, -32113);
    assert_eq!(autonomous_error_codes::CONSOLIDATION_ERROR, -32114);
    assert_eq!(autonomous_error_codes::SUBGOAL_DISCOVERY_ERROR, -32115);
    assert_eq!(autonomous_error_codes::STATUS_AGGREGATION_ERROR, -32116);
    assert_eq!(autonomous_error_codes::NO_NORTH_STAR_FOR_AUTONOMOUS, -32117);
    println!("[VERIFIED] Autonomous error codes are in correct range");
}
