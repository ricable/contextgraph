//! Initialize Handshake Tests (MCP 2024-11-05 compliance)

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

#[tokio::test]
async fn test_initialize_returns_protocol_version_2024_11_05() {
    let handlers = create_test_handlers();
    let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Initialize should not return an error"
    );
    let result = response.result.expect("Initialize must return a result");

    // MCP REQUIREMENT: protocolVersion MUST be "2024-11-05"
    let protocol_version = result
        .get("protocolVersion")
        .expect("Response must contain protocolVersion")
        .as_str()
        .expect("protocolVersion must be a string");
    assert_eq!(
        protocol_version, "2024-11-05",
        "Protocol version must be 2024-11-05"
    );
}

#[tokio::test]
async fn test_initialize_returns_capabilities_with_tools() {
    let handlers = create_test_handlers();
    let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Initialize must return a result");

    // MCP REQUIREMENT: capabilities object MUST exist
    let capabilities = result
        .get("capabilities")
        .expect("Response must contain capabilities object");

    // MCP REQUIREMENT: capabilities.tools MUST exist with listChanged
    let tools = capabilities
        .get("tools")
        .expect("capabilities must contain tools object");
    let list_changed = tools
        .get("listChanged")
        .expect("tools must contain listChanged")
        .as_bool()
        .expect("listChanged must be a boolean");
    assert!(list_changed, "listChanged should be true");
}

#[tokio::test]
async fn test_initialize_returns_server_info() {
    let handlers = create_test_handlers();
    let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Initialize must return a result");

    // MCP REQUIREMENT: serverInfo object MUST exist with name and version
    let server_info = result
        .get("serverInfo")
        .expect("Response must contain serverInfo object");

    let name = server_info
        .get("name")
        .expect("serverInfo must contain name")
        .as_str()
        .expect("name must be a string");
    assert_eq!(
        name, "context-graph-mcp",
        "Server name must be context-graph-mcp"
    );

    let version = server_info
        .get("version")
        .expect("serverInfo must contain version")
        .as_str()
        .expect("version must be a string");
    assert!(!version.is_empty(), "Version must not be empty");
}

#[tokio::test]
async fn test_initialize_has_cognitive_pulse_extension() {
    let handlers = create_test_handlers();
    let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;

    // Context Graph Extension: cognitive_pulse header
    let pulse = response
        .cognitive_pulse
        .expect("Initialize should include cognitive pulse");

    // Verify entropy is in valid range [0.0, 1.0]
    assert!(
        pulse.entropy >= 0.0 && pulse.entropy <= 1.0,
        "Entropy must be in [0.0, 1.0], got {}",
        pulse.entropy
    );

    // Verify coherence is in valid range [0.0, 1.0]
    assert!(
        pulse.coherence >= 0.0 && pulse.coherence <= 1.0,
        "Coherence must be in [0.0, 1.0], got {}",
        pulse.coherence
    );
}

#[tokio::test]
async fn test_initialized_notification_no_response_needed() {
    let handlers = create_test_handlers();
    // Notifications have no ID
    let request = make_request("notifications/initialized", None, None);

    let response = handlers.dispatch(request).await;

    // JSON-RPC 2.0: Notifications should return "no response" indicator
    // In this implementation, we return a response with no id, result, or error
    assert!(
        response.id.is_none(),
        "Notification response should have no ID"
    );
}

#[tokio::test]
async fn test_id_echoed_correctly_number() {
    let handlers = create_test_handlers();
    let request = make_request("initialize", Some(JsonRpcId::Number(42)), None);

    let response = handlers.dispatch(request).await;

    // JSON-RPC 2.0 REQUIREMENT: ID must be echoed back exactly
    let response_id = response.id.expect("Response must include ID");
    assert_eq!(
        response_id,
        JsonRpcId::Number(42),
        "Response ID must match request ID"
    );
}

#[tokio::test]
async fn test_id_echoed_correctly_string() {
    let handlers = create_test_handlers();
    let request = make_request(
        "initialize",
        Some(JsonRpcId::String("request-abc-123".to_string())),
        None,
    );

    let response = handlers.dispatch(request).await;

    let response_id = response.id.expect("Response must include ID");
    assert_eq!(
        response_id,
        JsonRpcId::String("request-abc-123".to_string()),
        "Response ID must match request ID"
    );
}

#[tokio::test]
async fn test_id_echoed_on_error() {
    let handlers = create_test_handlers();
    let request = make_request("unknown/method", Some(JsonRpcId::Number(999)), None);

    let response = handlers.dispatch(request).await;

    // ID must be echoed even on error responses
    let response_id = response.id.expect("Error response must include ID");
    assert_eq!(
        response_id,
        JsonRpcId::Number(999),
        "Error response ID must match request ID"
    );
}

#[tokio::test]
async fn test_response_jsonrpc_version_is_2_0() {
    let handlers = create_test_handlers();
    let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;

    // JSON-RPC 2.0 REQUIREMENT: jsonrpc field must be "2.0"
    assert_eq!(response.jsonrpc, "2.0", "JSON-RPC version must be 2.0");
}
