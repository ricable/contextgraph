//! GWT/Consciousness MCP tools tests (7 tools):
//! - get_consciousness_state
//! - get_kuramoto_sync
//! - get_workspace_status
//! - get_ego_state
//! - trigger_workspace_broadcast
//! - adjust_coupling
//! - get_coherence_state (TASK-34)

use serde_json::json;
use uuid::Uuid;

use crate::handlers::tests::create_test_handlers_with_warm_gwt;
use super::helpers::{make_tool_call, assert_success, assert_tool_error, get_tool_data};
use super::synthetic_data;

// -------------------------------------------------------------------------
// get_consciousness_state
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_consciousness_state_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_consciousness_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_consciousness_state");

    let data = get_tool_data(&response);
    // Verify consciousness equation components: C = I × R × D
    assert!(
        data.get("consciousness_level").is_some() || data.get("C").is_some(),
        "Must have consciousness level"
    );
}

#[tokio::test]
async fn test_get_consciousness_state_with_session() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "get_consciousness_state",
        json!({
            "session_id": "test-session-123"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_consciousness_state");
}

#[tokio::test]
async fn test_consciousness_level_in_range() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_consciousness_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_consciousness_state");

    let data = get_tool_data(&response);
    if let Some(c) = data
        .get("consciousness_level")
        .or(data.get("C"))
        .and_then(|v| v.as_f64())
    {
        assert!(
            (synthetic_data::consciousness::C_MIN..=synthetic_data::consciousness::C_MAX)
                .contains(&c),
            "Consciousness level {} must be in [0,1]",
            c
        );
    }
}

// -------------------------------------------------------------------------
// get_kuramoto_sync
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_kuramoto_sync_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    assert!(data.get("r").is_some(), "Must have order parameter r");
    assert!(data.get("psi").is_some() || data.get("mean_phase").is_some(), "Must have mean phase");
}

#[tokio::test]
async fn test_kuramoto_order_parameter_in_range() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    let r = data["r"].as_f64().expect("r must be f64");

    assert!(
        (synthetic_data::kuramoto::ORDER_PARAM_MIN..=synthetic_data::kuramoto::ORDER_PARAM_MAX)
            .contains(&r),
        "Order parameter r={} must be in [0,1]",
        r
    );
}

#[tokio::test]
async fn test_kuramoto_warm_state_synchronized() {
    // Warm GWT should have synchronized Kuramoto (r ≈ 1.0)
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    let r = data["r"].as_f64().expect("r must be f64");

    assert!(
        r >= synthetic_data::kuramoto::SYNC_THRESHOLD,
        "Warm GWT should have r >= {} (synchronized), got {}",
        synthetic_data::kuramoto::SYNC_THRESHOLD,
        r
    );
}

#[tokio::test]
async fn test_kuramoto_13_oscillators() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    if let Some(phases) = data.get("phases").and_then(|v| v.as_array()) {
        assert_eq!(
            phases.len(),
            synthetic_data::kuramoto::NUM_OSCILLATORS,
            "Must have exactly 13 oscillator phases"
        );
    }
}

// -------------------------------------------------------------------------
// get_workspace_status
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_workspace_status_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_workspace_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_workspace_status");

    let data = get_tool_data(&response);
    // Should have workspace-related fields
    assert!(
        data.get("active_memory").is_some()
            || data.get("broadcast_state").is_some()
            || data.get("state").is_some(),
        "Must have workspace state info"
    );
}

// -------------------------------------------------------------------------
// get_ego_state
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_ego_state_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_ego_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_ego_state");

    let data = get_tool_data(&response);
    assert!(
        data.get("purpose_vector").is_some(),
        "Must have purpose_vector (13D)"
    );
}

#[tokio::test]
async fn test_ego_state_purpose_vector_13d() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_ego_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_ego_state");

    let data = get_tool_data(&response);
    if let Some(pv) = data.get("purpose_vector").and_then(|v| v.as_array()) {
        assert_eq!(
            pv.len(),
            13,
            "Purpose vector must be 13-dimensional, got {}",
            pv.len()
        );
    }
}

#[tokio::test]
async fn test_ego_state_warm_nonzero_purpose() {
    // Warm GWT should have non-zero purpose vector
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_ego_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_ego_state");

    let data = get_tool_data(&response);
    if let Some(pv) = data.get("purpose_vector").and_then(|v| v.as_array()) {
        let sum: f64 = pv.iter().filter_map(|v| v.as_f64()).sum();
        assert!(
            sum > 0.0,
            "Warm GWT should have non-zero purpose vector sum"
        );
    }
}

// -------------------------------------------------------------------------
// trigger_workspace_broadcast
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_trigger_workspace_broadcast_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let memory_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "trigger_workspace_broadcast",
        json!({
            "memory_id": memory_id
        }),
    );

    let response = handlers.dispatch(request).await;
    // May fail if memory doesn't exist, but should not be JSON-RPC error
    assert!(
        response.error.is_none(),
        "Should not be JSON-RPC error"
    );
}

#[tokio::test]
async fn test_trigger_workspace_broadcast_with_params() {
    let handlers = create_test_handlers_with_warm_gwt();
    let memory_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "trigger_workspace_broadcast",
        json!({
            "memory_id": memory_id,
            "importance": 0.9,
            "alignment": 0.8,
            "force": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_trigger_workspace_broadcast_missing_memory_id() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("trigger_workspace_broadcast", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "trigger_workspace_broadcast");
}

// -------------------------------------------------------------------------
// adjust_coupling
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_adjust_coupling_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "adjust_coupling",
        json!({
            "new_K": 2.0
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_coupling");

    let data = get_tool_data(&response);
    assert!(
        data.get("old_K").is_some() || data.get("new_K").is_some(),
        "Must return coupling values"
    );
}

#[tokio::test]
async fn test_adjust_coupling_boundary_min() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "adjust_coupling",
        json!({
            "new_K": synthetic_data::kuramoto::COUPLING_MIN
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_coupling");
}

#[tokio::test]
async fn test_adjust_coupling_boundary_max() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "adjust_coupling",
        json!({
            "new_K": synthetic_data::kuramoto::COUPLING_MAX
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_coupling");
}

#[tokio::test]
async fn test_adjust_coupling_missing_new_k() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("adjust_coupling", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "adjust_coupling");
}

// -------------------------------------------------------------------------
// get_coherence_state (TASK-34)
// -------------------------------------------------------------------------

/// TASK-34: Basic get_coherence_state returns coherence level.
#[tokio::test]
async fn test_get_coherence_state_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    assert!(
        data.get("order_parameter").is_some(),
        "Must have order_parameter (Kuramoto r)"
    );
    assert!(
        data.get("coherence_level").is_some(),
        "Must have coherence_level classification"
    );
}

/// TASK-34: Verify order_parameter is in valid range [0, 1].
#[tokio::test]
async fn test_get_coherence_state_order_parameter_range() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    let r = data["order_parameter"]
        .as_f64()
        .expect("order_parameter must be f64");

    assert!(
        (synthetic_data::kuramoto::ORDER_PARAM_MIN..=synthetic_data::kuramoto::ORDER_PARAM_MAX)
            .contains(&r),
        "Order parameter r={} must be in [0,1]",
        r
    );
}

/// TASK-34: Verify warm GWT has High coherence (r > 0.8).
#[tokio::test]
async fn test_get_coherence_state_warm_gwt_high_coherence() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    let r = data["order_parameter"]
        .as_f64()
        .expect("order_parameter must be f64");
    let coherence_level = data["coherence_level"]
        .as_str()
        .expect("coherence_level must be string");

    // Warm GWT should have synchronized Kuramoto (r > 0.9)
    assert!(
        r >= 0.9,
        "Warm GWT should have r >= 0.9 (synchronized), got {}",
        r
    );

    // For r > 0.8, coherence_level should be "High"
    assert_eq!(
        coherence_level, "High",
        "With r={} (> 0.8), coherence_level should be 'High', got '{}'",
        r, coherence_level
    );
}

/// TASK-34: Verify coherence level classification is valid.
#[tokio::test]
async fn test_get_coherence_state_valid_coherence_level() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    let coherence_level = data["coherence_level"]
        .as_str()
        .expect("coherence_level must be string");

    // Valid levels: High, Medium, Low
    assert!(
        ["High", "Medium", "Low"].contains(&coherence_level),
        "coherence_level '{}' must be one of High/Medium/Low",
        coherence_level
    );
}

/// TASK-34: Verify broadcasting status is boolean.
#[tokio::test]
async fn test_get_coherence_state_has_broadcasting_status() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    assert!(
        data.get("is_broadcasting").is_some(),
        "Must have is_broadcasting field"
    );
    assert!(
        data["is_broadcasting"].is_boolean(),
        "is_broadcasting must be boolean"
    );
}

/// TASK-34: Verify conflict status is boolean.
#[tokio::test]
async fn test_get_coherence_state_has_conflict_status() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    assert!(
        data.get("has_conflict").is_some(),
        "Must have has_conflict field"
    );
    assert!(
        data["has_conflict"].is_boolean(),
        "has_conflict must be boolean"
    );
}

/// TASK-34: Verify thresholds are returned.
#[tokio::test]
async fn test_get_coherence_state_returns_thresholds() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    let thresholds = data.get("thresholds").expect("Must have thresholds");

    // Verify threshold values match constitution-mandated values
    assert_eq!(
        thresholds.get("high").and_then(|v| v.as_f64()),
        Some(0.8),
        "thresholds.high must be 0.8"
    );
    assert_eq!(
        thresholds.get("medium").and_then(|v| v.as_f64()),
        Some(0.5),
        "thresholds.medium must be 0.5"
    );
    assert_eq!(
        thresholds.get("low").and_then(|v| v.as_f64()),
        Some(0.0),
        "thresholds.low must be 0.0"
    );
}

/// TASK-34: Verify include_phases=false returns null phases.
#[tokio::test]
async fn test_get_coherence_state_no_phases_by_default() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_coherence_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    // phases should be null when not requested
    let phases = data.get("phases");
    assert!(
        phases.is_none() || phases.unwrap().is_null(),
        "phases should be null when include_phases is not specified"
    );
}

/// TASK-34: Verify include_phases=true returns 13 oscillator phases.
#[tokio::test]
async fn test_get_coherence_state_with_phases() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "get_coherence_state",
        json!({
            "include_phases": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    let phases = data
        .get("phases")
        .and_then(|v| v.as_array())
        .expect("phases must be array when include_phases=true");

    assert_eq!(
        phases.len(),
        synthetic_data::kuramoto::NUM_OSCILLATORS,
        "Must have exactly 13 oscillator phases"
    );
}

/// TASK-34: Verify include_phases=false explicitly returns null phases.
#[tokio::test]
async fn test_get_coherence_state_exclude_phases_explicitly() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "get_coherence_state",
        json!({
            "include_phases": false
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_coherence_state");

    let data = get_tool_data(&response);
    let phases = data.get("phases");
    assert!(
        phases.is_none() || phases.unwrap().is_null(),
        "phases should be null when include_phases=false"
    );
}
