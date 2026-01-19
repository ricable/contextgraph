#![allow(clippy::field_reassign_with_default)]

//! TCP Transport Integration Tests
//!
//! TASK-INTEG-020: Full integration tests for TCP transport.
//!
//! These tests verify:
//! 1. TCP server binds and accepts connections
//! 2. NDJSON message framing works correctly
//! 3. JSON-RPC requests are processed over TCP
//! 4. Multiple concurrent clients are handled
//! 5. Connection limits are enforced
//! 6. Error responses are correctly formatted
//!
//! # Running These Tests
//!
//! These tests require a running TCP server on a free port.
//! They use real network I/O and should be run with:
//! ```bash
//! cargo test -p context-graph-mcp tcp_transport_integration --test-threads=1
//! ```

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::time::timeout;

use serde_json::json;

use crate::protocol::{error_codes, JsonRpcRequest, JsonRpcResponse};

/// Helper to connect to a TCP server with timeout.
async fn connect_with_timeout(
    addr: &SocketAddr,
    timeout_secs: u64,
) -> tokio::io::Result<TcpStream> {
    timeout(Duration::from_secs(timeout_secs), TcpStream::connect(addr))
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "Connection timeout"))?
}

/// Helper to send a JSON-RPC request and receive response.
async fn send_request(
    stream: &mut TcpStream,
    request: &JsonRpcRequest,
) -> Result<JsonRpcResponse, Box<dyn std::error::Error>> {
    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    // Send request with newline (NDJSON)
    let request_json = serde_json::to_string(request)?;
    writer.write_all(request_json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;

    // Read response line
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    // Parse response
    let response: JsonRpcResponse = serde_json::from_str(line.trim())?;
    Ok(response)
}

// ============================================================================
// TCP Transport Unit Tests (Don't require running server)
// ============================================================================

/// Test error code values are correct.
#[test]
fn test_tcp_error_codes_values() {
    assert_eq!(error_codes::TCP_BIND_FAILED, -32110);
    assert_eq!(error_codes::TCP_CONNECTION_ERROR, -32111);
    assert_eq!(error_codes::TCP_MAX_CONNECTIONS_REACHED, -32112);
    assert_eq!(error_codes::TCP_FRAME_ERROR, -32113);
    assert_eq!(error_codes::TCP_CLIENT_TIMEOUT, -32114);
}

/// Test error codes are in the reserved range.
#[test]
fn test_tcp_error_codes_in_range() {
    let codes = [
        error_codes::TCP_BIND_FAILED,
        error_codes::TCP_CONNECTION_ERROR,
        error_codes::TCP_MAX_CONNECTIONS_REACHED,
        error_codes::TCP_FRAME_ERROR,
        error_codes::TCP_CLIENT_TIMEOUT,
    ];

    for code in codes {
        assert!(
            (-32119..=-32110).contains(&code),
            "Error code {} should be in range -32119 to -32110",
            code
        );
    }
}

/// Test JsonRpcRequest serialization for TCP.
#[test]
fn test_jsonrpc_request_serialization() {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(crate::protocol::JsonRpcId::Number(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {}
        })),
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("\"jsonrpc\":\"2.0\""));
    assert!(json.contains("\"method\":\"initialize\""));
}

/// Test JsonRpcResponse error construction.
#[test]
fn test_jsonrpc_error_response() {
    use crate::protocol::JsonRpcId;

    let response = JsonRpcResponse::error(
        Some(JsonRpcId::Number(1)),
        error_codes::TCP_CONNECTION_ERROR,
        "Connection lost",
    );

    assert!(response.error.is_some());
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::TCP_CONNECTION_ERROR);
    assert_eq!(error.message, "Connection lost");
}

/// Test NDJSON format (newline-delimited JSON).
#[test]
fn test_ndjson_format() {
    let request1 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(crate::protocol::JsonRpcId::Number(1)),
        method: "tools/list".to_string(),
        params: None,
    };

    let request2 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(crate::protocol::JsonRpcId::Number(2)),
        method: "memory/search".to_string(),
        params: Some(json!({"query": "test"})),
    };

    // NDJSON: each message on its own line
    let json1 = serde_json::to_string(&request1).unwrap();
    let json2 = serde_json::to_string(&request2).unwrap();
    let ndjson = format!("{}\n{}\n", json1, json2);

    // Verify each line is valid JSON
    for line in ndjson.lines() {
        let parsed: JsonRpcRequest = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.jsonrpc, "2.0");
    }
}

/// Test transport mode configuration defaults.
#[test]
fn test_transport_config_defaults() {
    use context_graph_core::config::Config;

    let config = Config::default_config();

    assert_eq!(config.mcp.transport, "stdio");
    assert_eq!(config.mcp.bind_address, "127.0.0.1");
    assert_eq!(config.mcp.tcp_port, 3100);
    assert_eq!(config.mcp.max_connections, 32);
}

/// Test transport mode validation.
#[test]
fn test_transport_validation() {
    use context_graph_core::config::McpConfig;

    // Valid stdio
    let mut config = McpConfig::default();
    config.transport = "stdio".to_string();
    assert!(config.validate().is_ok());

    // Valid tcp
    config.transport = "tcp".to_string();
    assert!(config.validate().is_ok());

    // Invalid transport
    config.transport = "websocket".to_string();
    assert!(config.validate().is_err());
}

/// Test TCP port validation.
#[test]
fn test_tcp_port_validation() {
    use context_graph_core::config::McpConfig;

    let mut config = McpConfig::default();
    config.transport = "tcp".to_string();

    // Valid port
    config.tcp_port = 3100;
    assert!(config.validate().is_ok());

    // Port 0 is invalid for TCP
    config.tcp_port = 0;
    assert!(config.validate().is_err());

    // Maximum port
    config.tcp_port = 65535;
    assert!(config.validate().is_ok());
}

/// Test max_connections validation.
#[test]
fn test_max_connections_validation() {
    use context_graph_core::config::McpConfig;

    let mut config = McpConfig::default();
    config.transport = "tcp".to_string();

    // Valid max_connections
    config.max_connections = 32;
    assert!(config.validate().is_ok());

    // Zero is invalid for TCP
    config.max_connections = 0;
    assert!(config.validate().is_err());

    // Large value is valid
    config.max_connections = 1000;
    assert!(config.validate().is_ok());
}

/// Test bind address validation.
#[test]
fn test_bind_address_validation() {
    use context_graph_core::config::McpConfig;

    let mut config = McpConfig::default();
    config.transport = "tcp".to_string();

    // Valid addresses
    config.bind_address = "127.0.0.1".to_string();
    assert!(config.validate().is_ok());

    config.bind_address = "0.0.0.0".to_string();
    assert!(config.validate().is_ok());

    // Empty address is invalid for TCP
    config.bind_address = "".to_string();
    assert!(config.validate().is_err());

    // Whitespace-only is invalid
    config.bind_address = "   ".to_string();
    assert!(config.validate().is_err());
}

// ============================================================================
// Integration Tests (Require running server - marked ignore by default)
// ============================================================================

/// Test TCP server accepts connection and responds to initialize.
///
/// This test requires a running TCP server.
/// Run with: cargo test -p context-graph-mcp test_tcp_server_initialize -- --ignored
#[tokio::test]
#[ignore = "Requires running TCP server on 127.0.0.1:3100"]
async fn test_tcp_server_initialize() {
    let addr: SocketAddr = "127.0.0.1:3100".parse().unwrap();

    let mut stream = connect_with_timeout(&addr, 5)
        .await
        .expect("Failed to connect");

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(crate::protocol::JsonRpcId::Number(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "0.1.0"
            }
        })),
    };

    let response = send_request(&mut stream, &request)
        .await
        .expect("Failed to send request");

    assert!(response.result.is_some(), "Initialize should return result");
    assert!(response.error.is_none(), "Initialize should not error");

    if let Some(result) = response.result {
        assert!(
            result.get("protocolVersion").is_some(),
            "Result should have protocolVersion"
        );
    }
}

/// Test TCP server responds to tools/list.
///
/// This test requires a running TCP server.
#[tokio::test]
#[ignore = "Requires running TCP server on 127.0.0.1:3100"]
async fn test_tcp_server_tools_list() {
    let addr: SocketAddr = "127.0.0.1:3100".parse().unwrap();

    let mut stream = connect_with_timeout(&addr, 5)
        .await
        .expect("Failed to connect");

    // First initialize
    let init_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(crate::protocol::JsonRpcId::Number(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1.0"}
        })),
    };
    let _ = send_request(&mut stream, &init_request).await;

    // Then list tools
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(crate::protocol::JsonRpcId::Number(2)),
        method: "tools/list".to_string(),
        params: None,
    };

    let response = send_request(&mut stream, &request)
        .await
        .expect("Failed to send request");

    assert!(response.result.is_some(), "tools/list should return result");
    assert!(response.error.is_none(), "tools/list should not error");

    if let Some(result) = response.result {
        assert!(
            result.get("tools").is_some(),
            "Result should have tools array"
        );
    }
}

/// Test TCP server rejects invalid JSON.
///
/// Per constitution AP-007: FAIL FAST on parse errors.
#[tokio::test]
#[ignore = "Requires running TCP server on 127.0.0.1:3100"]
async fn test_tcp_server_rejects_invalid_json() {
    let addr: SocketAddr = "127.0.0.1:3100".parse().unwrap();

    let mut stream = connect_with_timeout(&addr, 5)
        .await
        .expect("Failed to connect");
    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    // Send invalid JSON
    writer.write_all(b"not valid json\n").await.unwrap();
    writer.flush().await.unwrap();

    // Should receive error response
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();

    let response: JsonRpcResponse = serde_json::from_str(line.trim()).unwrap();
    assert!(
        response.error.is_some(),
        "Should return error for invalid JSON"
    );

    let error = response.error.unwrap();
    assert_eq!(
        error.code,
        error_codes::PARSE_ERROR,
        "Should be parse error"
    );
}

/// Test multiple concurrent TCP connections.
#[tokio::test]
#[ignore = "Requires running TCP server on 127.0.0.1:3100"]
async fn test_tcp_multiple_connections() {
    let addr: SocketAddr = "127.0.0.1:3100".parse().unwrap();

    // Open multiple connections
    let mut handles = Vec::new();
    for i in 0..3 {
        let handle = tokio::spawn(async move {
            let mut stream = connect_with_timeout(&addr, 5)
                .await
                .expect("Failed to connect");

            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(crate::protocol::JsonRpcId::Number(i)),
                method: "initialize".to_string(),
                params: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": format!("client-{}", i), "version": "0.1.0"}
                })),
            };

            let response = send_request(&mut stream, &request).await.unwrap();
            assert!(response.result.is_some());
            i
        });
        handles.push(handle);
    }

    // Wait for all connections to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result < 3);
    }
}
