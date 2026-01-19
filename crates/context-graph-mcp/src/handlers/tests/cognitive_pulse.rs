//! CognitivePulse integration tests for MCP responses (M05-T28).
//!
//! Tests verify that MCP tool error responses include the `_cognitive_pulse`
//! field with live UTL metrics.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

// =========================================================================
// TC-M05-T28-005: Error responses also include pulse where possible
// =========================================================================

#[tokio::test]
async fn test_error_responses_include_pulse() {
    let handlers = create_test_handlers();

    // Call inject_context without required 'content' parameter
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "inject_context",
            "arguments": {}  // Missing 'content'
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response
        .result
        .expect("Should return tool error (not JSON-RPC error)");

    // Verify isError is true
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .expect("isError should be a boolean");
    assert!(is_error, "Response should indicate error");

    // Verify _cognitive_pulse is still present
    let pulse = result.get("_cognitive_pulse");
    assert!(
        pulse.is_some(),
        "Error response should still include _cognitive_pulse. Got: {:?}",
        result
    );
}
