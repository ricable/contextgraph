//! MCP Client for CLI commands.
//!
//! Connects to running MCP server via TCP to use warm-loaded embedding models.
//! ELIMINATES StubMultiArrayProvider - all embeddings go through MCP server.
//!
//! # Task 14: Connect CLI to MCP Server
//!
//! This module solves the architectural bug where CLI commands used
//! StubMultiArrayProvider (zeroed embeddings) instead of connecting to
//! the MCP server which has warm-loaded GPU models.
//!
//! # Constitution Compliance
//!
//! - ARCH-01: TeleologicalArray is atomic (MCP server handles all 13 embeddings)
//! - ARCH-06: All memory ops through MCP tools
//! - ARCH-08: CUDA GPU required (MCP server uses GPU, not CLI)
//! - AP-06: No direct DB access - MCP tools only
//! - AP-07: No CPU fallback in production

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tracing::{debug, error, info, warn};

// =============================================================================
// Constants (AP-12: No magic numbers)
// =============================================================================

/// Default MCP server hostname.
const DEFAULT_MCP_HOST: &str = "127.0.0.1";

/// Default MCP server TCP port.
const DEFAULT_MCP_PORT: u16 = 3100;

/// Connection timeout in milliseconds (5 seconds).
const CONNECTION_TIMEOUT_MS: u64 = 5000;

/// Request timeout in milliseconds (30 seconds).
const REQUEST_TIMEOUT_MS: u64 = 30000;

// =============================================================================
// JSON-RPC Types
// =============================================================================

/// JSON-RPC 2.0 request structure.
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: &'static str,
    params: serde_json::Value,
}

/// JSON-RPC 2.0 response structure.
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error structure.
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

// =============================================================================
// MCP Client Error Types
// =============================================================================

/// MCP client errors with specific exit codes.
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    /// MCP server not running or unreachable.
    #[error("MCP server not running on {host}:{port}: {source}")]
    ServerNotRunning {
        host: String,
        port: u16,
        source: std::io::Error,
    },

    /// Connection timeout exceeded.
    #[error("Connection timeout after {timeout_ms}ms to {host}:{port}")]
    ConnectionTimeout {
        host: String,
        port: u16,
        timeout_ms: u64,
    },

    /// Request timeout exceeded.
    #[error("Request timeout after {timeout_ms}ms")]
    RequestTimeout { timeout_ms: u64 },

    /// MCP server returned an error.
    #[error("MCP error {code}: {message}")]
    McpError { code: i32, message: String },

    /// IO error during communication.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Response missing expected result.
    #[error("No result in MCP response")]
    NoResult,
}

// =============================================================================
// MCP Client
// =============================================================================

/// MCP TCP client for CLI commands.
///
/// Connects to the running MCP server to leverage warm-loaded GPU models
/// instead of using local stub embeddings.
///
/// # Environment Variables
///
/// - `CONTEXT_GRAPH_MCP_HOST`: MCP server hostname (default: 127.0.0.1)
/// - `CONTEXT_GRAPH_MCP_PORT`: MCP server TCP port (default: 3100)
///
/// # Example
///
/// ```rust,ignore
/// let client = McpClient::new();
/// let result = client.store_memory("Test content", 0.5, "text", None).await?;
/// ```
pub struct McpClient {
    host: String,
    port: u16,
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl McpClient {
    /// Create a new MCP client.
    ///
    /// Reads configuration from environment variables:
    /// - `CONTEXT_GRAPH_MCP_HOST` (default: 127.0.0.1)
    /// - `CONTEXT_GRAPH_MCP_PORT` (default: 3100)
    pub fn new() -> Self {
        let host = std::env::var("CONTEXT_GRAPH_MCP_HOST")
            .unwrap_or_else(|_| DEFAULT_MCP_HOST.to_string());
        let port = std::env::var("CONTEXT_GRAPH_MCP_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(DEFAULT_MCP_PORT);

        debug!("MCP client configured: {}:{}", host, port);

        Self { host, port }
    }

    /// Create a new MCP client with explicit host and port.
    pub fn with_address(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }

    /// Check if the MCP server is reachable.
    ///
    /// Attempts a quick TCP connection to verify the server is running.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if server is reachable
    /// - `Ok(false)` if server is not running
    /// - `Err(...)` if connection check fails unexpectedly
    pub async fn is_server_running(&self) -> Result<bool, McpClientError> {
        let addr = format!("{}:{}", self.host, self.port);

        match tokio::time::timeout(
            std::time::Duration::from_millis(1000),
            TcpStream::connect(&addr),
        )
        .await
        {
            Ok(Ok(_stream)) => {
                debug!("MCP server is reachable at {}", addr);
                Ok(true)
            }
            Ok(Err(_)) | Err(_) => {
                debug!("MCP server not reachable at {}", addr);
                Ok(false)
            }
        }
    }

    /// Call the `store_memory` MCP tool.
    ///
    /// Uses warm-loaded models on MCP server to generate real embeddings.
    ///
    /// # Arguments
    ///
    /// - `content`: Memory content to store
    /// - `importance`: Importance score [0.0, 1.0]
    /// - `modality`: Content type (text, code, etc.)
    /// - `tags`: Optional tags for categorization
    ///
    /// # Returns
    ///
    /// The MCP tool result as JSON value.
    pub async fn store_memory(
        &self,
        content: &str,
        importance: f64,
        modality: &str,
        tags: Option<Vec<String>>,
    ) -> Result<serde_json::Value, McpClientError> {
        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": content,
                "importance": importance,
                "modality": modality,
                "tags": tags.unwrap_or_default()
            }
        });

        info!(
            content_len = content.len(),
            importance,
            modality,
            "Calling MCP store_memory"
        );

        self.call_tool(params).await
    }

    /// Call the `inject_context` MCP tool.
    ///
    /// Uses warm-loaded models on MCP server for embedding and UTL processing.
    ///
    /// # Arguments
    ///
    /// - `content`: Context content to inject
    /// - `rationale`: Reason for storing this context
    /// - `importance`: Importance score [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// The MCP tool result as JSON value.
    pub async fn inject_context(
        &self,
        content: &str,
        rationale: &str,
        importance: f64,
    ) -> Result<serde_json::Value, McpClientError> {
        let params = json!({
            "name": "inject_context",
            "arguments": {
                "content": content,
                "rationale": rationale,
                "importance": importance
            }
        });

        info!(
            content_len = content.len(),
            importance,
            "Calling MCP inject_context"
        );

        self.call_tool(params).await
    }

    /// Call the `search_graph` MCP tool.
    ///
    /// Searches the knowledge graph using semantic similarity.
    ///
    /// # Arguments
    ///
    /// - `query`: Search query text
    /// - `top_k`: Maximum number of results to return (default: 10)
    ///
    /// # Returns
    ///
    /// The MCP tool result as JSON value containing matching memories.
    pub async fn search_graph(
        &self,
        query: &str,
        top_k: Option<u32>,
    ) -> Result<serde_json::Value, McpClientError> {
        let params = json!({
            "name": "search_graph",
            "arguments": {
                "query": query,
                "topK": top_k.unwrap_or(10)
            }
        });

        info!(query_len = query.len(), top_k, "Calling MCP search_graph");

        self.call_tool(params).await
    }

    /// Call the `get_memetic_status` MCP tool.
    ///
    /// Gets current system status with UTL metrics.
    ///
    /// # Returns
    ///
    /// The MCP tool result as JSON value containing system status.
    pub async fn get_memetic_status(&self) -> Result<serde_json::Value, McpClientError> {
        let params = json!({
            "name": "get_memetic_status",
            "arguments": {}
        });

        info!("Calling MCP get_memetic_status");

        self.call_tool(params).await
    }

    /// Internal method to call an MCP tool.
    ///
    /// Establishes TCP connection, sends JSON-RPC request, and reads response.
    async fn call_tool(
        &self,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, McpClientError> {
        let addr = format!("{}:{}", self.host, self.port);
        debug!("Connecting to MCP server at {}", addr);

        // Connect with timeout
        let stream = tokio::time::timeout(
            std::time::Duration::from_millis(CONNECTION_TIMEOUT_MS),
            TcpStream::connect(&addr),
        )
        .await
        .map_err(|_| McpClientError::ConnectionTimeout {
            host: self.host.clone(),
            port: self.port,
            timeout_ms: CONNECTION_TIMEOUT_MS,
        })?
        .map_err(|e| McpClientError::ServerNotRunning {
            host: self.host.clone(),
            port: self.port,
            source: e,
        })?;

        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);

        // Send tools/call request
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params,
        };

        let request_json = serde_json::to_string(&request)?;
        debug!("Sending: {}", request_json);

        writer.write_all(request_json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;

        // Read response with timeout
        let mut response_line = String::new();
        let bytes_read = tokio::time::timeout(
            std::time::Duration::from_millis(REQUEST_TIMEOUT_MS),
            reader.read_line(&mut response_line),
        )
        .await
        .map_err(|_| McpClientError::RequestTimeout {
            timeout_ms: REQUEST_TIMEOUT_MS,
        })??;

        if bytes_read == 0 {
            warn!("MCP server closed connection before responding");
            return Err(McpClientError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Server closed connection",
            )));
        }

        debug!("Received: {}", response_line.trim());

        let response: JsonRpcResponse = serde_json::from_str(&response_line)?;

        if let Some(error) = response.error {
            error!("MCP error {}: {}", error.code, error.message);
            return Err(McpClientError::McpError {
                code: error.code,
                message: error.message,
            });
        }

        response.result.ok_or(McpClientError::NoResult)
    }

    /// Get the server address string.
    pub fn server_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::test_utils::GLOBAL_IDENTITY_LOCK;

    #[test]
    fn test_mcp_client_new_defaults() {
        let _lock = GLOBAL_IDENTITY_LOCK.lock();

        // Clear environment to test defaults
        std::env::remove_var("CONTEXT_GRAPH_MCP_HOST");
        std::env::remove_var("CONTEXT_GRAPH_MCP_PORT");

        let client = McpClient::new();
        assert_eq!(client.host, DEFAULT_MCP_HOST);
        assert_eq!(client.port, DEFAULT_MCP_PORT);
        assert_eq!(client.server_address(), "127.0.0.1:3100");
    }

    #[test]
    fn test_mcp_client_from_env() {
        let _lock = GLOBAL_IDENTITY_LOCK.lock();

        // Clear first to ensure known state
        std::env::remove_var("CONTEXT_GRAPH_MCP_HOST");
        std::env::remove_var("CONTEXT_GRAPH_MCP_PORT");

        // Set test values
        std::env::set_var("CONTEXT_GRAPH_MCP_HOST", "192.168.1.100");
        std::env::set_var("CONTEXT_GRAPH_MCP_PORT", "9000");

        let client = McpClient::new();
        assert_eq!(client.host, "192.168.1.100");
        assert_eq!(client.port, 9000);

        // Cleanup
        std::env::remove_var("CONTEXT_GRAPH_MCP_HOST");
        std::env::remove_var("CONTEXT_GRAPH_MCP_PORT");
    }

    #[test]
    fn test_mcp_client_with_address() {
        let client = McpClient::with_address("localhost", 8080);
        assert_eq!(client.host, "localhost");
        assert_eq!(client.port, 8080);
    }

    #[test]
    fn test_mcp_client_invalid_port_env() {
        let _lock = GLOBAL_IDENTITY_LOCK.lock();

        // Clear first
        std::env::remove_var("CONTEXT_GRAPH_MCP_HOST");
        std::env::remove_var("CONTEXT_GRAPH_MCP_PORT");

        std::env::set_var("CONTEXT_GRAPH_MCP_PORT", "not-a-number");

        let client = McpClient::new();
        // Should fall back to default
        assert_eq!(client.port, DEFAULT_MCP_PORT);

        std::env::remove_var("CONTEXT_GRAPH_MCP_PORT");
    }

    #[test]
    fn test_mcp_client_error_display() {
        let server_not_running = McpClientError::ServerNotRunning {
            host: "127.0.0.1".to_string(),
            port: 3100,
            source: std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused"),
        };
        let msg = format!("{}", server_not_running);
        assert!(msg.contains("MCP server not running"));
        assert!(msg.contains("127.0.0.1:3100"));

        let timeout = McpClientError::ConnectionTimeout {
            host: "localhost".to_string(),
            port: 9000,
            timeout_ms: 5000,
        };
        let msg = format!("{}", timeout);
        assert!(msg.contains("Connection timeout"));
        assert!(msg.contains("5000ms"));

        let mcp_error = McpClientError::McpError {
            code: -32600,
            message: "Invalid request".to_string(),
        };
        let msg = format!("{}", mcp_error);
        assert!(msg.contains("MCP error -32600"));
        assert!(msg.contains("Invalid request"));
    }

    #[tokio::test]
    async fn test_is_server_running_not_running() {
        // Test with a port that should not have anything running
        let client = McpClient::with_address("127.0.0.1", 59999);
        let result = client.is_server_running().await;
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }
}
