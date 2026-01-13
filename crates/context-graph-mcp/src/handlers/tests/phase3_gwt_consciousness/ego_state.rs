//! Scenario 4: Ego State Verification Tests
//!
//! Tests get_ego_state tool:
//! - Valid identity and purpose vector
//! - 13-element purpose vector verification
//! - Warm state non-zero values

#![allow(clippy::type_complexity)] // Complex types needed for test infrastructure
#![allow(clippy::absurd_extreme_comparisons)] // Intentional FSV checks for documentation
#![allow(unused_comparisons)] // Intentional FSV checks for documentation

use serde_json::json;

use crate::handlers::tests::{create_test_handlers_with_warm_gwt, extract_mcp_tool_data};
use crate::protocol::{JsonRpcId, JsonRpcRequest};
use crate::tools::tool_names;

/// FSV Test: get_ego_state returns valid identity and purpose vector.
///
/// Source of Truth: SelfEgoProvider
/// Expected: Response contains purpose_vector (13D), identity_coherence, identity_status
#[tokio::test]
async fn test_get_ego_state_returns_valid_data() {
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_ego_state
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Response is successful
    assert!(
        response.error.is_none(),
        "[FSV] Expected success, got error: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: CRITICAL - Verify purpose_vector has exactly 13 elements
    let pv = data
        .get("purpose_vector")
        .and_then(|v| v.as_array())
        .expect("purpose_vector must exist");
    assert_eq!(
        pv.len(),
        13,
        "[FSV] CRITICAL: Purpose vector must have 13 elements (one per embedder), got {}",
        pv.len()
    );

    // FSV: Verify all purpose vector elements are floats in [-1, 1]
    // (Purpose alignments are cosine similarities)
    for (i, val) in pv.iter().enumerate() {
        let v = val.as_f64().expect("purpose_vector elements must be f64");
        assert!(
            (-1.0..=1.0).contains(&v),
            "[FSV] Purpose vector[{}] must be in [-1, 1], got {}",
            i,
            v
        );
    }

    // FSV: Verify identity_coherence is in [0, 1]
    let identity_coherence = data
        .get("identity_coherence")
        .and_then(|v| v.as_f64())
        .expect("identity_coherence must exist");
    assert!(
        (0.0..=1.0).contains(&identity_coherence),
        "[FSV] identity_coherence must be in [0, 1], got {}",
        identity_coherence
    );

    // FSV: Verify identity_status is valid
    let status = data
        .get("identity_status")
        .and_then(|v| v.as_str())
        .expect("identity_status must exist");
    let valid_statuses = ["Healthy", "Warning", "Degraded", "Critical"];
    // Status might be Debug formatted (e.g., "Healthy" or "IdentityStatus::Healthy")
    let status_valid = valid_statuses.iter().any(|s| status.contains(s));
    assert!(
        status_valid,
        "[FSV] Invalid identity_status: {}, expected one containing {:?}",
        status,
        valid_statuses
    );

    // FSV: Verify coherence_with_actions is in [0, 1]
    let coherence_with_actions = data
        .get("coherence_with_actions")
        .and_then(|v| v.as_f64())
        .expect("coherence_with_actions must exist");
    assert!(
        (0.0..=1.0).contains(&coherence_with_actions),
        "[FSV] coherence_with_actions must be in [0, 1], got {}",
        coherence_with_actions
    );

    // FSV: Verify trajectory_length is non-negative
    let trajectory_length = data
        .get("trajectory_length")
        .and_then(|v| v.as_u64())
        .expect("trajectory_length must exist");
    assert!(
        trajectory_length >= 0,
        "[FSV] trajectory_length must be non-negative"
    );

    // FSV: Verify thresholds are present
    let thresholds = data.get("thresholds").expect("thresholds must exist");
    assert_eq!(
        thresholds.get("healthy").and_then(|v| v.as_f64()),
        Some(0.9),
        "[FSV] thresholds.healthy must be 0.9"
    );
    assert_eq!(
        thresholds.get("warning").and_then(|v| v.as_f64()),
        Some(0.7),
        "[FSV] thresholds.warning must be 0.7"
    );

    println!("[FSV] Phase 3 - get_ego_state verification PASSED");
    println!(
        "[FSV]   purpose_vector.len={}, identity_coherence={:.4}, status={}",
        pv.len(),
        identity_coherence,
        status
    );
    println!(
        "[FSV]   trajectory_length={}, coherence_with_actions={:.4}",
        trajectory_length, coherence_with_actions
    );
}

/// FSV Test: get_ego_state with WARM state has non-zero purpose vector.
#[tokio::test]
async fn test_get_ego_state_warm_has_non_zero_purpose_vector() {
    // Warm GWT state includes a pre-initialized purpose vector
    let handlers = create_test_handlers_with_warm_gwt();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let data = extract_mcp_tool_data(&response.result.unwrap());
    let pv = data
        .get("purpose_vector")
        .and_then(|v| v.as_array())
        .expect("purpose_vector must exist");

    // FSV: At least some elements should be non-zero in warm state
    let non_zero_count = pv
        .iter()
        .filter(|v| {
            let val = v.as_f64().unwrap_or(0.0);
            val.abs() > 0.001
        })
        .count();

    assert!(
        non_zero_count > 0,
        "[FSV] WARM state should have non-zero purpose vector elements, got {} non-zero",
        non_zero_count
    );

    println!("[FSV] Phase 3 - get_ego_state WARM state verification PASSED");
    println!("[FSV]   Non-zero purpose vector elements: {}/13", non_zero_count);
}

// =============================================================================
// TASK-IDENTITY-P0-007: Identity Continuity MCP Tool Exposure Tests
// =============================================================================

/// FSV Test: get_ego_state includes identity_continuity object.
///
/// TASK-IDENTITY-P0-007: Verify the enhanced get_ego_state response includes
/// the identity_continuity object with:
/// - ic: float 0.0-1.0
/// - status: Healthy|Warning|Degraded|Critical
/// - in_crisis: bool
/// - history_len: int
/// - last_detection: null or CrisisDetectionResult
///
/// Source of Truth: GwtSystemProvider.identity_*() async methods
#[tokio::test]
async fn test_get_ego_state_includes_identity_continuity() {
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_ego_state
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Response is successful
    assert!(
        response.error.is_none(),
        "[FSV] Expected success, got error: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: CRITICAL - Verify identity_continuity object exists
    let identity_continuity = data
        .get("identity_continuity")
        .expect("[FSV] TASK-IDENTITY-P0-007: identity_continuity object must exist in get_ego_state response");

    // FSV: Verify ic field is in [0, 1]
    let ic = identity_continuity
        .get("ic")
        .and_then(|v| v.as_f64())
        .expect("[FSV] identity_continuity.ic must exist and be a number");
    assert!(
        (0.0..=1.0).contains(&ic),
        "[FSV] identity_continuity.ic must be in [0, 1], got {}",
        ic
    );

    // FSV: Verify status field is a valid status string
    let status = identity_continuity
        .get("status")
        .and_then(|v| v.as_str())
        .expect("[FSV] identity_continuity.status must exist and be a string");
    let valid_statuses = ["Healthy", "Warning", "Degraded", "Critical"];
    let status_valid = valid_statuses.iter().any(|s| status.contains(s));
    assert!(
        status_valid,
        "[FSV] identity_continuity.status must contain one of {:?}, got {}",
        valid_statuses, status
    );

    // FSV: Verify in_crisis field is a boolean
    let in_crisis = identity_continuity
        .get("in_crisis")
        .and_then(|v| v.as_bool())
        .expect("[FSV] identity_continuity.in_crisis must exist and be a boolean");
    // Just verify it's parseable - actual value depends on state
    let _ = in_crisis;

    // FSV: Verify history_len field is a non-negative integer
    let history_len = identity_continuity
        .get("history_len")
        .and_then(|v| v.as_u64())
        .expect("[FSV] identity_continuity.history_len must exist and be a number");
    // History length is non-negative (obviously)
    let _ = history_len;

    // FSV: Verify last_detection field exists (can be null)
    let _last_detection = identity_continuity
        .get("last_detection")
        .expect("[FSV] identity_continuity.last_detection field must exist (can be null)");
    // Note: last_detection can be null initially before any detect_crisis() call

    println!("[FSV] TASK-IDENTITY-P0-007 - get_ego_state identity_continuity verification PASSED");
    println!(
        "[FSV]   ic={:.4}, status={}, in_crisis={}, history_len={}",
        ic, status, in_crisis, history_len
    );
}

/// FSV Test: identity_continuity last_detection contains valid CrisisDetectionResult when present.
///
/// TASK-IDENTITY-P0-007: When last_detection is not null, verify it contains:
/// - identity_coherence: float
/// - previous_status: string
/// - current_status: string
/// - status_changed: bool
/// - entering_crisis: bool
/// - entering_critical: bool
/// - recovering: bool
/// - time_since_last_event_ms: int or null
/// - can_emit_event: bool
#[tokio::test]
async fn test_get_ego_state_last_detection_structure_when_present() {
    let handlers = create_test_handlers_with_warm_gwt();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let data = extract_mcp_tool_data(&response.result.unwrap());
    let identity_continuity = data
        .get("identity_continuity")
        .expect("identity_continuity must exist");

    let last_detection = identity_continuity.get("last_detection");

    // If last_detection is null, that's valid - no crisis detection has occurred yet
    if last_detection.map(|v| v.is_null()).unwrap_or(true) {
        println!("[FSV] TASK-IDENTITY-P0-007 - last_detection is null (no detection yet) - VALID");
        return;
    }

    // If last_detection is present and not null, verify its structure
    let det = last_detection.unwrap();

    // Verify all required fields exist
    let ic_det = det
        .get("identity_coherence")
        .and_then(|v| v.as_f64())
        .expect("[FSV] last_detection.identity_coherence must exist");
    assert!(
        (0.0..=1.0).contains(&ic_det),
        "[FSV] last_detection.identity_coherence must be in [0, 1]"
    );

    let _prev_status = det
        .get("previous_status")
        .and_then(|v| v.as_str())
        .expect("[FSV] last_detection.previous_status must exist");

    let _curr_status = det
        .get("current_status")
        .and_then(|v| v.as_str())
        .expect("[FSV] last_detection.current_status must exist");

    let _status_changed = det
        .get("status_changed")
        .and_then(|v| v.as_bool())
        .expect("[FSV] last_detection.status_changed must exist");

    let _entering_crisis = det
        .get("entering_crisis")
        .and_then(|v| v.as_bool())
        .expect("[FSV] last_detection.entering_crisis must exist");

    let _entering_critical = det
        .get("entering_critical")
        .and_then(|v| v.as_bool())
        .expect("[FSV] last_detection.entering_critical must exist");

    let _recovering = det
        .get("recovering")
        .and_then(|v| v.as_bool())
        .expect("[FSV] last_detection.recovering must exist");

    // time_since_last_event_ms can be null or a number
    let _time_since = det
        .get("time_since_last_event_ms")
        .expect("[FSV] last_detection.time_since_last_event_ms field must exist");

    let _can_emit = det
        .get("can_emit_event")
        .and_then(|v| v.as_bool())
        .expect("[FSV] last_detection.can_emit_event must exist");

    println!("[FSV] TASK-IDENTITY-P0-007 - last_detection structure verification PASSED");
    println!("[FSV]   last_detection.identity_coherence={:.4}", ic_det);
}
