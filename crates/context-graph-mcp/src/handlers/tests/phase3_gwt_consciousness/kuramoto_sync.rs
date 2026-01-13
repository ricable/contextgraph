//! Scenario 2: Kuramoto Synchronization Tests
//!
//! Tests get_kuramoto_sync and adjust_coupling tools:
//! - 13 oscillators verification
//! - Coupling constant modification
//! - State persistence

use serde_json::json;

use crate::handlers::tests::{create_test_handlers_with_warm_gwt, extract_mcp_tool_data};
use crate::protocol::{JsonRpcId, JsonRpcRequest};
use crate::tools::tool_names;

/// FSV Test: get_kuramoto_sync returns valid oscillator network state.
///
/// Source of Truth: KuramotoProvider
/// Expected: Response contains r, phases[13], natural_freqs[13], coupling, thresholds.
#[tokio::test]
async fn test_get_kuramoto_sync_returns_13_oscillators() {
    // SETUP: Create handlers with WARM GWT state
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_kuramoto_sync
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_KURAMOTO_SYNC,
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

    // FSV: Verify r (order parameter) is in [0, 1]
    let r = data.get("r").and_then(|v| v.as_f64()).expect("r must exist");
    assert!(
        (0.0..=1.0).contains(&r),
        "[FSV] r must be in [0, 1], got {}",
        r
    );

    // FSV: Verify synchronized network has high r
    assert!(
        r > 0.99,
        "[FSV] Synchronized network should have r > 0.99, got {}",
        r
    );

    // FSV: Verify psi (mean phase) is in [0, 2*PI] (allowing some tolerance)
    let psi = data.get("psi").and_then(|v| v.as_f64()).expect("psi must exist");
    assert!(
        (0.0..=2.0 * std::f64::consts::PI + 0.01).contains(&psi),
        "[FSV] psi must be in [0, 2*PI], got {}",
        psi
    );

    // FSV: CRITICAL - Verify phases array has exactly 13 oscillators (one per embedder)
    let phases = data
        .get("phases")
        .and_then(|v| v.as_array())
        .expect("phases must exist");
    assert_eq!(
        phases.len(),
        13,
        "[FSV] CRITICAL: Must have 13 oscillator phases (one per embedder), got {}",
        phases.len()
    );

    // FSV: Verify natural frequencies array has exactly 13 elements
    let natural_freqs = data
        .get("natural_freqs")
        .and_then(|v| v.as_array())
        .expect("natural_freqs must exist");
    assert_eq!(
        natural_freqs.len(),
        13,
        "[FSV] Must have 13 natural frequencies, got {}",
        natural_freqs.len()
    );

    // FSV: Verify all natural frequencies are positive
    for (i, freq) in natural_freqs.iter().enumerate() {
        let freq_val = freq.as_f64().expect("freq must be f64");
        assert!(
            freq_val > 0.0,
            "[FSV] Frequency[{}] must be positive, got {}",
            i,
            freq_val
        );
    }

    // FSV: Verify coupling strength K is present
    let coupling = data
        .get("coupling")
        .and_then(|v| v.as_f64())
        .expect("coupling must exist");
    assert!(
        coupling >= 0.0,
        "[FSV] Coupling K must be non-negative, got {}",
        coupling
    );

    // FSV: Verify thresholds are constitution-mandated values
    let thresholds = data.get("thresholds").expect("thresholds must exist");
    assert_eq!(
        thresholds.get("conscious").and_then(|v| v.as_f64()),
        Some(0.8),
        "[FSV] thresholds.conscious must be 0.8"
    );
    assert_eq!(
        thresholds.get("fragmented").and_then(|v| v.as_f64()),
        Some(0.5),
        "[FSV] thresholds.fragmented must be 0.5"
    );
    assert_eq!(
        thresholds.get("hypersync").and_then(|v| v.as_f64()),
        Some(0.95),
        "[FSV] thresholds.hypersync must be 0.95"
    );

    // FSV: Verify embedding labels are present (13 labels)
    let labels = data
        .get("embedding_labels")
        .and_then(|v| v.as_array())
        .expect("embedding_labels must exist");
    assert_eq!(
        labels.len(),
        13,
        "[FSV] Must have 13 embedding labels, got {}",
        labels.len()
    );

    // FSV: Verify state is valid
    let state = data
        .get("state")
        .and_then(|v| v.as_str())
        .expect("state must exist");
    let valid_states = ["DORMANT", "FRAGMENTED", "EMERGING", "CONSCIOUS", "HYPERSYNC"];
    assert!(
        valid_states.contains(&state),
        "[FSV] Invalid state: {}",
        state
    );

    println!("[FSV] Phase 3 - get_kuramoto_sync verification PASSED");
    println!(
        "[FSV]   r={:.4}, psi={:.4}, phases.len={}, natural_freqs.len={}, coupling={}",
        r,
        psi,
        phases.len(),
        natural_freqs.len(),
        coupling
    );
    println!("[FSV]   KURAMOTO 13 OSCILLATORS: VERIFIED");
}

/// FSV Test: adjust_coupling modifies Kuramoto coupling constant K.
///
/// Source of Truth: KuramotoProvider::set_coupling_strength()
/// Expected: old_K, new_K in response, K changes persist
#[tokio::test]
async fn test_adjust_coupling_modifies_k() {
    // SETUP: Create handlers with WARM GWT state
    let handlers = create_test_handlers_with_warm_gwt();

    // STEP 1: Get initial coupling via get_kuramoto_sync
    let sync_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_KURAMOTO_SYNC,
            "arguments": {}
        })),
    };
    let sync_response = handlers.dispatch(sync_request).await;
    assert!(sync_response.error.is_none(), "Initial sync should succeed");
    let sync_data = extract_mcp_tool_data(&sync_response.result.unwrap());
    let initial_k = sync_data
        .get("coupling")
        .and_then(|v| v.as_f64())
        .expect("coupling must exist");

    // STEP 2: Adjust coupling to new value
    let new_k_target = 3.5;
    let adjust_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::ADJUST_COUPLING,
            "arguments": {
                "new_K": new_k_target
            }
        })),
    };
    let adjust_response = handlers.dispatch(adjust_request).await;

    // VERIFY: Adjustment succeeded
    assert!(
        adjust_response.error.is_none(),
        "[FSV] adjust_coupling should succeed: {:?}",
        adjust_response.error
    );
    let adjust_data = extract_mcp_tool_data(&adjust_response.result.unwrap());

    // FSV: Verify old_K matches initial
    let old_k = adjust_data
        .get("old_K")
        .and_then(|v| v.as_f64())
        .expect("old_K must exist");
    assert!(
        (old_k - initial_k).abs() < 0.01,
        "[FSV] old_K ({}) should match initial_k ({})",
        old_k,
        initial_k
    );

    // FSV: Verify new_K is as requested (may be clamped to [0, 10])
    let new_k = adjust_data
        .get("new_K")
        .and_then(|v| v.as_f64())
        .expect("new_K must exist");
    assert!(
        (new_k - new_k_target).abs() < 0.01,
        "[FSV] new_K ({}) should be {} (requested)",
        new_k,
        new_k_target
    );

    // STEP 3: Verify change persists via another get_kuramoto_sync
    let verify_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_KURAMOTO_SYNC,
            "arguments": {}
        })),
    };
    let verify_response = handlers.dispatch(verify_request).await;
    assert!(verify_response.error.is_none(), "Verify sync should succeed");
    let verify_data = extract_mcp_tool_data(&verify_response.result.unwrap());
    let persisted_k = verify_data
        .get("coupling")
        .and_then(|v| v.as_f64())
        .expect("coupling must exist");

    // FSV: Verify persisted K matches new K
    assert!(
        (persisted_k - new_k).abs() < 0.01,
        "[FSV] Persisted K ({}) should match new_K ({})",
        persisted_k,
        new_k
    );

    println!("[FSV] Phase 3 - adjust_coupling verification PASSED");
    println!(
        "[FSV]   old_K={}, new_K={}, persisted_K={}",
        old_k, new_k, persisted_k
    );
    println!("[FSV]   STATE CHANGES PERSIST: VERIFIED");
}

/// FSV Test: adjust_coupling clamps K to [0, 10] range.
#[tokio::test]
async fn test_adjust_coupling_clamps_k() {
    let handlers = create_test_handlers_with_warm_gwt();

    // Test upper bound clamping
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::ADJUST_COUPLING,
            "arguments": {
                "new_K": 100.0  // Above max
            }
        })),
    };
    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should succeed with clamping");
    let data = extract_mcp_tool_data(&response.result.unwrap());
    let new_k = data.get("new_K").and_then(|v| v.as_f64()).unwrap();
    assert!(
        new_k <= 10.0,
        "[FSV] K should be clamped to max 10, got {}",
        new_k
    );

    println!("[FSV] Phase 3 - adjust_coupling clamping verification PASSED");
}
