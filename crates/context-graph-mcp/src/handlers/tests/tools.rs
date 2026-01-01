//! Tools List and Tools Call Tests (MCP 2024-11-05 compliance)

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

// =========================================================================
// Tools List Tests
// =========================================================================

#[tokio::test]
async fn test_tools_list_returns_all_5_tools() {
    let handlers = create_test_handlers();
    let request = make_request("tools/list", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "tools/list should not return an error"
    );
    let result = response.result.expect("tools/list must return a result");

    // MCP REQUIREMENT: tools array MUST exist
    let tools = result
        .get("tools")
        .expect("Response must contain tools array")
        .as_array()
        .expect("tools must be an array");

    // Verify exactly 5 tools returned
    assert_eq!(
        tools.len(),
        5,
        "Must return exactly 5 tools, got {}",
        tools.len()
    );
}

#[tokio::test]
async fn test_tools_list_each_tool_has_required_fields() {
    let handlers = create_test_handlers();
    let request = make_request("tools/list", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("tools/list must return a result");
    let tools = result.get("tools").unwrap().as_array().unwrap();

    for tool in tools {
        // MCP REQUIREMENT: each tool MUST have name (string)
        let name = tool
            .get("name")
            .expect("Tool must have name field")
            .as_str()
            .expect("Tool name must be a string");
        assert!(!name.is_empty(), "Tool name must not be empty");

        // MCP REQUIREMENT: each tool MUST have description (string)
        let description = tool
            .get("description")
            .expect("Tool must have description field")
            .as_str()
            .expect("Tool description must be a string");
        assert!(
            !description.is_empty(),
            "Tool description must not be empty"
        );

        // MCP REQUIREMENT: each tool MUST have inputSchema (JSON Schema object)
        let input_schema = tool
            .get("inputSchema")
            .expect("Tool must have inputSchema field");
        assert!(
            input_schema.is_object(),
            "inputSchema must be a JSON object"
        );

        // Verify inputSchema is valid JSON Schema (has type field)
        let schema_type = input_schema
            .get("type")
            .expect("inputSchema must have a type field")
            .as_str()
            .expect("inputSchema type must be a string");
        assert_eq!(schema_type, "object", "inputSchema type must be 'object'");
    }
}

#[tokio::test]
async fn test_tools_list_contains_expected_tool_names() {
    let handlers = create_test_handlers();
    let request = make_request("tools/list", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("tools/list must return a result");
    let tools = result.get("tools").unwrap().as_array().unwrap();

    let tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
        .collect();

    // Verify all expected tools are present
    assert!(
        tool_names.contains(&"inject_context"),
        "Missing inject_context tool"
    );
    assert!(
        tool_names.contains(&"store_memory"),
        "Missing store_memory tool"
    );
    assert!(
        tool_names.contains(&"get_memetic_status"),
        "Missing get_memetic_status tool"
    );
    assert!(
        tool_names.contains(&"get_graph_manifest"),
        "Missing get_graph_manifest tool"
    );
    assert!(
        tool_names.contains(&"search_graph"),
        "Missing search_graph tool"
    );
}

// =========================================================================
// inject_context Tool Tests
// =========================================================================

#[tokio::test]
async fn test_tools_call_inject_context_valid() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "inject_context",
        "arguments": {
            "content": "Test content for injection",
            "rationale": "Testing MCP protocol compliance"
        }
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Valid inject_context should not return an error"
    );
    let result = response.result.expect("tools/call must return a result");

    // MCP REQUIREMENT: content array MUST exist
    let content = result
        .get("content")
        .expect("Response must contain content array")
        .as_array()
        .expect("content must be an array");

    // MCP REQUIREMENT: content items must have type and text
    assert!(!content.is_empty(), "Content array must not be empty");
    let first_item = &content[0];
    let item_type = first_item
        .get("type")
        .expect("Content item must have type")
        .as_str()
        .expect("type must be a string");
    assert_eq!(item_type, "text", "Content item type must be 'text'");

    let text = first_item
        .get("text")
        .expect("Content item must have text")
        .as_str()
        .expect("text must be a string");
    assert!(!text.is_empty(), "Content text must not be empty");

    // MCP REQUIREMENT: isError MUST be false for successful calls
    let is_error = result
        .get("isError")
        .expect("Response must contain isError")
        .as_bool()
        .expect("isError must be a boolean");
    assert!(!is_error, "isError must be false for successful tool calls");

    // Verify nodeId is present in the response text
    let parsed_text: serde_json::Value =
        serde_json::from_str(text).expect("Content text should be valid JSON");
    assert!(
        parsed_text.get("nodeId").is_some(),
        "Response must contain nodeId"
    );
}

#[tokio::test]
async fn test_tools_call_inject_context_returns_utl_metrics() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "inject_context",
        "arguments": {
            "content": "Learning about MCP protocols",
            "rationale": "Understanding API standards"
        }
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("tools/call must return a result");
    let content = result.get("content").unwrap().as_array().unwrap();
    let text = content[0].get("text").unwrap().as_str().unwrap();
    let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify UTL metrics are present
    let utl = parsed_text
        .get("utl")
        .expect("Response must contain utl object");
    assert!(
        utl.get("learningScore").is_some(),
        "utl must contain learningScore"
    );
    assert!(utl.get("entropy").is_some(), "utl must contain entropy");
    assert!(utl.get("coherence").is_some(), "utl must contain coherence");
    assert!(utl.get("surprise").is_some(), "utl must contain surprise");
}

// =========================================================================
// get_memetic_status Tool Tests
// =========================================================================

#[tokio::test]
async fn test_tools_call_get_memetic_status() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "get_memetic_status",
        "arguments": {}
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "get_memetic_status should not return an error"
    );
    let result = response.result.expect("tools/call must return a result");

    // Verify MCP format
    let content = result.get("content").unwrap().as_array().unwrap();
    let text = content[0].get("text").unwrap().as_str().unwrap();
    let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify expected fields
    assert!(
        parsed_text.get("phase").is_some(),
        "Response must contain phase"
    );
    assert!(
        parsed_text.get("nodeCount").is_some(),
        "Response must contain nodeCount"
    );
    assert!(
        parsed_text.get("utl").is_some(),
        "Response must contain utl"
    );
    assert!(
        parsed_text.get("layers").is_some(),
        "Response must contain layers"
    );
}

// =========================================================================
// store_memory Tool Tests
// =========================================================================

#[tokio::test]
async fn test_tools_call_store_memory_valid() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "store_memory",
        "arguments": {
            "content": "Memory content to store",
            "importance": 0.8
        }
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Valid store_memory should not return an error"
    );
    let result = response.result.expect("tools/call must return a result");

    let is_error = result.get("isError").unwrap().as_bool().unwrap();
    assert!(!is_error, "isError must be false for successful store");

    let content = result.get("content").unwrap().as_array().unwrap();
    let text = content[0].get("text").unwrap().as_str().unwrap();
    let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

    assert!(
        parsed_text.get("nodeId").is_some(),
        "Response must contain nodeId"
    );
}

// =========================================================================
// search_graph Tool Tests
// =========================================================================

#[tokio::test]
async fn test_tools_call_search_graph_valid() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "search_graph",
        "arguments": {
            "query": "test search query",
            "topK": 5
        }
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Valid search_graph should not return an error"
    );
    let result = response.result.expect("tools/call must return a result");

    // Verify MCP format
    let is_error = result.get("isError").unwrap().as_bool().unwrap();
    assert!(!is_error, "isError must be false for successful search");

    let content = result.get("content").unwrap().as_array().unwrap();
    let text = content[0].get("text").unwrap().as_str().unwrap();
    let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify search results structure
    assert!(
        parsed_text.get("results").is_some(),
        "Response must contain results"
    );
    assert!(
        parsed_text.get("count").is_some(),
        "Response must contain count"
    );
}

#[tokio::test]
async fn test_tools_call_search_graph_missing_query() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "search_graph",
        "arguments": {
            "topK": 5
        }
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "Tool errors use isError flag");
    let result = response.result.expect("Tool error must return a result");
    let is_error = result.get("isError").unwrap().as_bool().unwrap();
    assert!(is_error, "Missing query should set isError to true");
}

#[tokio::test]
async fn test_tool_error_sets_is_error_true() {
    let handlers = create_test_handlers();
    // inject_context without required 'content' parameter
    let params = json!({
        "name": "inject_context",
        "arguments": {
            "rationale": "Missing content"
        }
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    // Tool errors return success with isError: true (MCP format)
    assert!(
        response.error.is_none(),
        "Tool errors use isError flag, not JSON-RPC error"
    );
    let result = response
        .result
        .expect("Tool error must return a result with isError");

    let is_error = result
        .get("isError")
        .expect("Response must contain isError")
        .as_bool()
        .expect("isError must be a boolean");
    assert!(is_error, "isError must be true for tool errors");

    // Verify error message is in content
    let content = result
        .get("content")
        .expect("Response must contain content")
        .as_array()
        .expect("content must be an array");
    assert!(!content.is_empty(), "Error content must not be empty");
}
