//! Helper functions for exhaustive MCP tool tests.

use serde_json::json;

use crate::handlers::tests::{extract_mcp_tool_data, make_request};
use crate::protocol::JsonRpcId;

/// Make a tools/call request with given tool name and arguments.
pub fn make_tool_call(tool_name: &str, arguments: serde_json::Value) -> crate::protocol::JsonRpcRequest {
    make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": tool_name,
            "arguments": arguments
        })),
    )
}

/// Assert that a response is successful (no error, isError=false).
pub fn assert_success(response: &crate::protocol::JsonRpcResponse, tool_name: &str) {
    assert!(
        response.error.is_none(),
        "{} should not return JSON-RPC error",
        tool_name
    );
    let result = response
        .result
        .as_ref()
        .unwrap_or_else(|| panic!("{} must return a result", tool_name));
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(!is_error, "{} should have isError=false", tool_name);
}

/// Assert that a response indicates a tool error (isError=true).
pub fn assert_tool_error(response: &crate::protocol::JsonRpcResponse, tool_name: &str) {
    assert!(
        response.error.is_none(),
        "{} tool errors should use isError flag, not JSON-RPC error",
        tool_name
    );
    let result = response
        .result
        .as_ref()
        .unwrap_or_else(|| panic!("{} must return a result with isError", tool_name));
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(is_error, "{} should have isError=true for errors", tool_name);
}

/// Extract the data from a successful MCP tool response.
pub fn get_tool_data(response: &crate::protocol::JsonRpcResponse) -> serde_json::Value {
    let result = response.result.as_ref().expect("Must have result");
    extract_mcp_tool_data(result)
}
