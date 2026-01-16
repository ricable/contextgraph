//! Autonomous MCP tools tests (12 tools after TASK-P0-001):
//!
//! TASK-P0-001: REMOVED auto_bootstrap_north_star per ARCH-03.
//! Goals now emerge autonomously from topic clustering (constitution v6.0.0).
//!
//! Current tools:
//! - get_alignment_drift
//! - trigger_drift_correction
//! - get_pruning_candidates
//! - trigger_consolidation
//! - discover_sub_goals
//! - get_autonomous_status
//! - get_learner_state
//! - observe_outcome
//! - execute_prune
//! - get_health_status
//! - trigger_healing
//! - get_drift_history

use serde_json::json;

use crate::handlers::tests::create_test_handlers;
use super::helpers::{make_tool_call, assert_success};

// REMOVED: auto_bootstrap_north_star tests per TASK-P0-001 (ARCH-03)
// The tool has been removed. Goals now emerge from topic clustering.

// -------------------------------------------------------------------------
// get_alignment_drift
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_alignment_drift_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_alignment_drift", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_alignment_drift");
}

#[tokio::test]
async fn test_get_alignment_drift_with_timeframe() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "get_alignment_drift",
        json!({
            "timeframe": "7d"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_alignment_drift");
}

#[tokio::test]
async fn test_get_alignment_drift_with_history() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "get_alignment_drift",
        json!({
            "include_history": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_alignment_drift");
}

// -------------------------------------------------------------------------
// trigger_drift_correction
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_trigger_drift_correction_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("trigger_drift_correction", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_drift_correction");
}

#[tokio::test]
async fn test_trigger_drift_correction_with_force() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "trigger_drift_correction",
        json!({
            "force": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_drift_correction");
}

#[tokio::test]
async fn test_trigger_drift_correction_with_target() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "trigger_drift_correction",
        json!({
            "target_alignment": 0.9
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_drift_correction");
}

// -------------------------------------------------------------------------
// get_pruning_candidates
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_pruning_candidates_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_pruning_candidates", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_pruning_candidates");
}

#[tokio::test]
async fn test_get_pruning_candidates_with_params() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "get_pruning_candidates",
        json!({
            "limit": 10,
            "min_staleness_days": 7,
            "min_alignment": 0.3
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_pruning_candidates");
}

// -------------------------------------------------------------------------
// trigger_consolidation
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_trigger_consolidation_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("trigger_consolidation", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_consolidation");
}

#[tokio::test]
async fn test_trigger_consolidation_similarity() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "trigger_consolidation",
        json!({
            "strategy": "similarity"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_consolidation");
}

#[tokio::test]
async fn test_trigger_consolidation_temporal() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "trigger_consolidation",
        json!({
            "strategy": "temporal"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_consolidation");
}

#[tokio::test]
async fn test_trigger_consolidation_semantic() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "trigger_consolidation",
        json!({
            "strategy": "semantic"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_consolidation");
}

// -------------------------------------------------------------------------
// discover_sub_goals
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_discover_sub_goals_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("discover_sub_goals", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "discover_sub_goals");
}

#[tokio::test]
async fn test_discover_sub_goals_with_params() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "discover_sub_goals",
        json!({
            "min_confidence": 0.7,
            "max_goals": 3
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "discover_sub_goals");
}

// -------------------------------------------------------------------------
// get_autonomous_status
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_autonomous_status_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_autonomous_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_autonomous_status");
}

#[tokio::test]
async fn test_get_autonomous_status_with_metrics() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "get_autonomous_status",
        json!({
            "include_metrics": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_autonomous_status");
}

#[tokio::test]
async fn test_get_autonomous_status_with_history() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "get_autonomous_status",
        json!({
            "include_history": true,
            "history_count": 5
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_autonomous_status");
}
