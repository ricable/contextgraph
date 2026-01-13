//! SSE (Server-Sent Events) transport types for MCP.
//!
//! TASK-32: Defines types and configuration for SSE transport.
//! TASK-33: Will add handler implementation using these types.
//!
//! ## MCP SSE Protocol (per modelcontextprotocol.io)
//!
//! SSE enables real-time server-to-client streaming of JSON-RPC 2.0 messages.
//! Event format: `data: {json}\nid: {event-id}\n\n`
//!
//! ## Constitution Reference
//!
//! constitution.yaml mcp.transport: [stdio, sse]
//! Performance budget: MCP Response <100ms

use std::time::Duration;
use serde::{Deserialize, Serialize};

// CRITICAL: Reuse existing JsonRpcError from protocol.rs - DO NOT DUPLICATE
pub use crate::protocol::JsonRpcError;

// ============================================================================
// SSE CONFIGURATION
// ============================================================================

/// SSE transport configuration.
///
/// Default values per MCP spec and constitution.yaml performance budgets.
///
/// # Example
///
/// ```rust
/// use context_graph_mcp::transport::SseConfig;
///
/// let config = SseConfig::default();
/// assert_eq!(config.keepalive_interval.as_secs(), 15);
/// assert_eq!(config.max_connection_duration.as_secs(), 3600);
/// assert_eq!(config.buffer_size, 100);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseConfig {
    /// Keep-alive ping interval.
    ///
    /// Sends a Ping event at this interval to prevent connection timeout.
    /// Default: 15 seconds (reasonable for most proxy/load balancer timeouts).
    pub keepalive_interval: Duration,

    /// Maximum connection duration.
    ///
    /// Connections are forcibly closed after this duration to prevent
    /// resource exhaustion. Clients should reconnect if needed.
    /// Default: 1 hour (3600 seconds).
    pub max_connection_duration: Duration,

    /// Event buffer size.
    ///
    /// Number of events to buffer before applying backpressure.
    /// If the client falls behind, older events may be dropped.
    /// Default: 100 events.
    pub buffer_size: usize,
}

impl Default for SseConfig {
    fn default() -> Self {
        Self {
            keepalive_interval: Duration::from_secs(15),
            max_connection_duration: Duration::from_secs(3600),
            buffer_size: 100,
        }
    }
}

impl SseConfig {
    /// Create config with custom keep-alive interval.
    ///
    /// # Panics
    ///
    /// Panics if interval is zero.
    #[must_use]
    pub fn with_keepalive(mut self, interval: Duration) -> Self {
        assert!(!interval.is_zero(), "keepalive_interval cannot be zero");
        self.keepalive_interval = interval;
        self
    }

    /// Create config with custom max connection duration.
    ///
    /// # Panics
    ///
    /// Panics if duration is zero.
    #[must_use]
    pub fn with_max_duration(mut self, duration: Duration) -> Self {
        assert!(!duration.is_zero(), "max_connection_duration cannot be zero");
        self.max_connection_duration = duration;
        self
    }

    /// Create config with custom buffer size.
    ///
    /// # Panics
    ///
    /// Panics if size is zero.
    #[must_use]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        assert!(size > 0, "buffer_size cannot be zero");
        self.buffer_size = size;
        self
    }

    /// Validate configuration.
    ///
    /// Returns error if any value is invalid.
    pub fn validate(&self) -> Result<(), SseConfigError> {
        if self.keepalive_interval.is_zero() {
            return Err(SseConfigError::InvalidKeepalive);
        }
        if self.max_connection_duration.is_zero() {
            return Err(SseConfigError::InvalidMaxDuration);
        }
        if self.buffer_size == 0 {
            return Err(SseConfigError::InvalidBufferSize);
        }
        if self.keepalive_interval >= self.max_connection_duration {
            return Err(SseConfigError::KeepaliveExceedsMaxDuration);
        }
        Ok(())
    }
}

/// SSE configuration validation error.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum SseConfigError {
    #[error("keepalive_interval cannot be zero")]
    InvalidKeepalive,

    #[error("max_connection_duration cannot be zero")]
    InvalidMaxDuration,

    #[error("buffer_size cannot be zero")]
    InvalidBufferSize,

    #[error("keepalive_interval must be less than max_connection_duration")]
    KeepaliveExceedsMaxDuration,
}

// ============================================================================
// SSE EVENT TYPES
// ============================================================================

/// SSE event types for MCP communication.
///
/// Each variant corresponds to a JSON-RPC 2.0 message type plus a Ping
/// for keep-alive. Events are serialized to JSON and sent via SSE.
///
/// ## Wire Format
///
/// ```text
/// event: message
/// data: {"type":"Response","id":1,"result":{...}}
/// id: evt-12345
///
/// ```
///
/// ## JSON-RPC 2.0 Compliance
///
/// - `Response`: Has `id` (matches request) and `result`
/// - `Error`: Has `id` (matches request or null) and `error`
/// - `Notification`: No `id`, has `method` and `params`
/// - `Ping`: Context Graph extension, not in JSON-RPC spec
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum McpSseEvent {
    /// JSON-RPC success response.
    ///
    /// Sent when a request completes successfully.
    Response {
        /// Request ID (matches the original request).
        id: serde_json::Value,
        /// Result payload.
        result: serde_json::Value,
    },

    /// JSON-RPC error response.
    ///
    /// Sent when a request fails.
    Error {
        /// Request ID (matches the original request, or null for parse errors).
        id: serde_json::Value,
        /// Error details.
        error: JsonRpcError,
    },

    /// JSON-RPC notification (server-initiated).
    ///
    /// Sent for events that don't expect a response.
    Notification {
        /// Notification method name.
        method: String,
        /// Notification parameters.
        #[serde(default)]
        params: serde_json::Value,
    },

    /// Keep-alive ping.
    ///
    /// Context Graph extension. Sent at `keepalive_interval` to prevent
    /// connection timeout. Clients should ignore this event.
    Ping {
        /// Unix timestamp (seconds since epoch).
        timestamp: u64,
    },
}

impl McpSseEvent {
    /// Create a success response event.
    #[must_use]
    pub fn response(id: impl Into<serde_json::Value>, result: serde_json::Value) -> Self {
        Self::Response {
            id: id.into(),
            result,
        }
    }

    /// Create an error response event.
    #[must_use]
    pub fn error(id: impl Into<serde_json::Value>, error: JsonRpcError) -> Self {
        Self::Error {
            id: id.into(),
            error,
        }
    }

    /// Create a notification event.
    #[must_use]
    pub fn notification(method: impl Into<String>, params: serde_json::Value) -> Self {
        Self::Notification {
            method: method.into(),
            params,
        }
    }

    /// Create a ping event with current timestamp.
    #[must_use]
    pub fn ping() -> Self {
        Self::Ping {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before UNIX epoch")
                .as_secs(),
        }
    }

    /// Create a ping event with specific timestamp.
    #[must_use]
    pub fn ping_at(timestamp: u64) -> Self {
        Self::Ping { timestamp }
    }

    /// Check if this is a ping event.
    #[must_use]
    pub fn is_ping(&self) -> bool {
        matches!(self, Self::Ping { .. })
    }

    /// Serialize to JSON string.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails (should never happen for valid events).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SseConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sse_config_defaults() {
        let config = SseConfig::default();

        assert_eq!(config.keepalive_interval, Duration::from_secs(15));
        assert_eq!(config.max_connection_duration, Duration::from_secs(3600));
        assert_eq!(config.buffer_size, 100);
    }

    #[test]
    fn test_sse_config_validate_valid() {
        let config = SseConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sse_config_validate_keepalive_zero() {
        let config = SseConfig {
            keepalive_interval: Duration::ZERO,
            ..Default::default()
        };
        assert_eq!(config.validate(), Err(SseConfigError::InvalidKeepalive));
    }

    #[test]
    fn test_sse_config_validate_max_duration_zero() {
        let config = SseConfig {
            max_connection_duration: Duration::ZERO,
            ..Default::default()
        };
        assert_eq!(config.validate(), Err(SseConfigError::InvalidMaxDuration));
    }

    #[test]
    fn test_sse_config_validate_buffer_zero() {
        let config = SseConfig {
            buffer_size: 0,
            ..Default::default()
        };
        assert_eq!(config.validate(), Err(SseConfigError::InvalidBufferSize));
    }

    #[test]
    fn test_sse_config_validate_keepalive_exceeds_max() {
        let config = SseConfig {
            keepalive_interval: Duration::from_secs(3600),
            max_connection_duration: Duration::from_secs(60),
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(SseConfigError::KeepaliveExceedsMaxDuration)
        );
    }

    #[test]
    fn test_sse_config_builder_pattern() {
        let config = SseConfig::default()
            .with_keepalive(Duration::from_secs(30))
            .with_max_duration(Duration::from_secs(7200))
            .with_buffer_size(200);

        assert_eq!(config.keepalive_interval, Duration::from_secs(30));
        assert_eq!(config.max_connection_duration, Duration::from_secs(7200));
        assert_eq!(config.buffer_size, 200);
    }

    #[test]
    #[should_panic(expected = "keepalive_interval cannot be zero")]
    fn test_sse_config_with_keepalive_zero_panics() {
        let _ = SseConfig::default().with_keepalive(Duration::ZERO);
    }

    #[test]
    #[should_panic(expected = "max_connection_duration cannot be zero")]
    fn test_sse_config_with_max_duration_zero_panics() {
        let _ = SseConfig::default().with_max_duration(Duration::ZERO);
    }

    #[test]
    #[should_panic(expected = "buffer_size cannot be zero")]
    fn test_sse_config_with_buffer_size_zero_panics() {
        let _ = SseConfig::default().with_buffer_size(0);
    }

    // -------------------------------------------------------------------------
    // McpSseEvent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mcp_sse_event_response() {
        let event = McpSseEvent::response(1, serde_json::json!({"status": "ok"}));

        let json = event.to_json().unwrap();
        assert!(json.contains(r#""type":"Response""#));
        assert!(json.contains(r#""id":1"#));
        assert!(json.contains(r#""status":"ok""#));
    }

    #[test]
    fn test_mcp_sse_event_error() {
        let error = JsonRpcError {
            code: -32600,
            message: "Invalid Request".to_string(),
            data: None,
        };
        let event = McpSseEvent::error(1, error);

        let json = event.to_json().unwrap();
        assert!(json.contains(r#""type":"Error""#));
        assert!(json.contains(r#""code":-32600"#));
    }

    #[test]
    fn test_mcp_sse_event_notification() {
        let event = McpSseEvent::notification(
            "consciousness/state_changed",
            serde_json::json!({"level": 0.85}),
        );

        let json = event.to_json().unwrap();
        assert!(json.contains(r#""type":"Notification""#));
        assert!(json.contains(r#""method":"consciousness/state_changed""#));
    }

    #[test]
    fn test_mcp_sse_event_ping() {
        let event = McpSseEvent::ping_at(1704067200);

        let json = event.to_json().unwrap();
        assert!(json.contains(r#""type":"Ping""#));
        assert!(json.contains(r#""timestamp":1704067200"#));
    }

    #[test]
    fn test_mcp_sse_event_is_ping() {
        assert!(McpSseEvent::ping().is_ping());
        assert!(!McpSseEvent::response(1, serde_json::json!(null)).is_ping());
    }

    #[test]
    fn test_mcp_sse_event_roundtrip() {
        let original = McpSseEvent::response(
            "req-123",
            serde_json::json!({"memories": []}),
        );

        let json = original.to_json().unwrap();
        let parsed: McpSseEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_mcp_sse_event_notification_default_params() {
        // Test that missing params deserializes to null (serde default)
        let json = r#"{"type":"Notification","method":"test"}"#;
        let event: McpSseEvent = serde_json::from_str(json).unwrap();

        match event {
            McpSseEvent::Notification { method, params } => {
                assert_eq!(method, "test");
                assert!(params.is_null());
            }
            _ => panic!("Expected Notification"),
        }
    }

    #[test]
    fn test_mcp_sse_event_error_roundtrip() {
        let error = JsonRpcError {
            code: -32700,
            message: "Parse error".to_string(),
            data: Some(serde_json::json!({"details": "unexpected token"})),
        };
        let original = McpSseEvent::error(serde_json::Value::Null, error);

        let json = original.to_json().unwrap();
        let parsed: McpSseEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_mcp_sse_event_response_with_string_id() {
        let event = McpSseEvent::response("abc-123", serde_json::json!({"data": [1, 2, 3]}));

        let json = event.to_json().unwrap();
        assert!(json.contains(r#""id":"abc-123""#));
    }

    #[test]
    fn test_sse_config_validate_keepalive_equals_max() {
        // Edge case: keepalive == max_duration should fail
        let config = SseConfig {
            keepalive_interval: Duration::from_secs(100),
            max_connection_duration: Duration::from_secs(100),
            buffer_size: 50,
        };
        assert_eq!(
            config.validate(),
            Err(SseConfigError::KeepaliveExceedsMaxDuration)
        );
    }

    #[test]
    fn test_mcp_sse_event_ping_has_valid_timestamp() {
        let event = McpSseEvent::ping();
        if let McpSseEvent::Ping { timestamp } = event {
            // Should be a recent timestamp (after year 2020)
            assert!(timestamp > 1577836800); // 2020-01-01
        } else {
            panic!("Expected Ping event");
        }
    }
}
