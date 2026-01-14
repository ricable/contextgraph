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
// SSE HANDLER (TASK-33)
// ============================================================================

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
    Router,
};
use tokio::sync::broadcast;

/// Counter for generating unique event IDs.
static EVENT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique event ID for SSE reconnection support.
///
/// Event IDs follow the format `evt-{counter}` where counter is a
/// monotonically increasing atomic value.
///
/// # Example
///
/// ```rust,ignore
/// let id1 = next_event_id(); // "evt-1"
/// let id2 = next_event_id(); // "evt-2"
/// ```
#[inline]
pub fn next_event_id() -> String {
    format!("evt-{}", EVENT_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// SSE application state.
///
/// Shared across all SSE connections. Contains the broadcast channel
/// for event distribution and configuration.
///
/// # Thread Safety
///
/// - `event_tx`: Broadcast sender is Clone + Send + Sync
/// - `config`: SseConfig is Clone (no interior mutability)
/// - `connection_count`: AtomicU64 for observability
///
/// # Example
///
/// ```rust
/// use context_graph_mcp::transport::{SseAppState, SseConfig};
///
/// let state = SseAppState::new(SseConfig::default()).unwrap();
/// assert_eq!(state.active_connections(), 0);
/// ```
#[derive(Clone)]
pub struct SseAppState {
    /// Broadcast sender for SSE events.
    ///
    /// Multiple connections subscribe to this channel.
    /// Buffer size is `config.buffer_size`.
    pub event_tx: broadcast::Sender<McpSseEvent>,

    /// SSE configuration (validated on creation).
    pub config: SseConfig,

    /// Active connection count (for observability).
    connection_count: std::sync::Arc<AtomicU64>,
}

impl SseAppState {
    /// Create a new SSE application state.
    ///
    /// # Errors
    ///
    /// Returns `SseConfigError` if configuration is invalid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_mcp::transport::{SseAppState, SseConfig};
    ///
    /// let state = SseAppState::new(SseConfig::default()).unwrap();
    /// ```
    pub fn new(config: SseConfig) -> Result<Self, SseConfigError> {
        config.validate()?;

        let (event_tx, _) = broadcast::channel(config.buffer_size);

        Ok(Self {
            event_tx,
            config,
            connection_count: std::sync::Arc::new(AtomicU64::new(0)),
        })
    }

    /// Get the current number of active SSE connections.
    #[must_use]
    pub fn active_connections(&self) -> u64 {
        self.connection_count.load(Ordering::Relaxed)
    }

    /// Broadcast an event to all connected clients.
    ///
    /// Returns the number of receivers that received the event.
    /// Returns 0 if no clients are connected.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_mcp::transport::{SseAppState, SseConfig, McpSseEvent};
    ///
    /// let state = SseAppState::new(SseConfig::default()).unwrap();
    /// let event = McpSseEvent::notification("test", serde_json::json!({}));
    /// let receivers = state.broadcast(event);
    /// println!("Sent to {} clients", receivers);
    /// ```
    pub fn broadcast(&self, event: McpSseEvent) -> usize {
        // send() returns Err if no receivers, which is fine
        self.event_tx.send(event).unwrap_or(0)
    }

    /// Subscribe to the event stream.
    ///
    /// Returns a broadcast receiver. Each call creates a new subscription.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<McpSseEvent> {
        self.event_tx.subscribe()
    }

    /// Increment connection count (called on connect).
    pub fn increment_connections(&self) -> u64 {
        self.connection_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrement connection count (called on disconnect).
    ///
    /// Returns the new connection count after decrementing.
    /// If the count is already 0, it remains 0 (no underflow).
    pub fn decrement_connections(&self) -> u64 {
        // Use fetch_update to prevent underflow below 0
        let result = self.connection_count.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| {
                if current > 0 {
                    Some(current - 1)
                } else {
                    None // Don't update if already 0
                }
            }
        );

        // Return the new value (either decremented or unchanged 0)
        match result {
            Ok(old_val) => old_val - 1, // Successfully decremented, return new value
            Err(val) => val,             // Was already 0, return 0
        }
    }
}

impl std::fmt::Debug for SseAppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SseAppState")
            .field("config", &self.config)
            .field("active_connections", &self.active_connections())
            .field("buffer_size", &self.config.buffer_size)
            .finish()
    }
}

/// SSE handler error.
///
/// Errors that can occur during SSE streaming.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SseHandlerError {
    /// Broadcast channel was closed unexpectedly.
    #[error("broadcast channel closed")]
    ChannelClosed,

    /// Failed to serialize an event to JSON.
    #[error("event serialization failed: {0}")]
    Serialization(String),

    /// Connection duration exceeded maximum allowed.
    #[error("connection duration exceeded")]
    MaxDurationExceeded,
}

/// Create SSE router for MCP transport.
///
/// Returns a router with a single `/events` endpoint that streams
/// MCP events to connected clients.
///
/// # Example
///
/// ```rust,no_run
/// use context_graph_mcp::transport::{create_sse_router, SseAppState, SseConfig};
///
/// #[tokio::main]
/// async fn main() {
///     let state = SseAppState::new(SseConfig::default()).unwrap();
///     let router = create_sse_router(state);
///
///     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
///     axum::serve(listener, router).await.unwrap();
/// }
/// ```
pub fn create_sse_router(state: SseAppState) -> Router {
    Router::new()
        .route("/events", get(sse_handler))
        .with_state(state)
}

/// SSE endpoint handler.
///
/// Streams MCP events to the client with automatic keep-alive pings.
///
/// # Behavior
///
/// 1. Subscribes to the broadcast channel
/// 2. Sends events as they arrive (JSON serialized)
/// 3. Sends Ping events at `keepalive_interval`
/// 4. Closes connection after `max_connection_duration`
///
/// # Wire Format
///
/// ```text
/// event: message
/// data: {"type":"Response","id":1,"result":{...}}
/// id: evt-12345
///
/// event: message
/// data: {"type":"Ping","timestamp":1704067200}
/// id: evt-12346
///
/// ```
///
/// # Constitution Reference
///
/// - `mcp.transport: [stdio, sse]` - SSE is a required transport
/// - `perf.latency.mcp: <100ms` - Events must be delivered quickly
pub async fn sse_handler(
    State(state): State<SseAppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let config = state.config.clone();
    let mut rx = state.subscribe();

    // Track connection
    let conn_count = state.increment_connections();
    tracing::info!(connections = conn_count, "SSE client connected");

    // Clone state for decrement on drop
    let state_for_drop = state.clone();

    // Create the event stream
    let stream = async_stream::stream! {
        use tokio::time::{interval, Instant};

        let mut keepalive_timer = interval(config.keepalive_interval);
        let connection_start = Instant::now();
        let max_duration = config.max_connection_duration;

        loop {
            // Check if max duration exceeded
            if connection_start.elapsed() >= max_duration {
                tracing::info!("SSE connection max duration exceeded, closing");
                break;
            }

            tokio::select! {
                // Keepalive ping
                _ = keepalive_timer.tick() => {
                    let ping_event = McpSseEvent::ping();
                    match ping_event.to_json() {
                        Ok(json) => {
                            let event_id = next_event_id();
                            yield Ok(Event::default()
                                .event("message")
                                .data(json)
                                .id(event_id));
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Failed to serialize ping event");
                        }
                    }
                }

                // Broadcast events
                result = rx.recv() => {
                    match result {
                        Ok(mcp_event) => {
                            match mcp_event.to_json() {
                                Ok(json) => {
                                    let event_id = next_event_id();
                                    yield Ok(Event::default()
                                        .event("message")
                                        .data(json)
                                        .id(event_id));
                                }
                                Err(e) => {
                                    tracing::error!(error = %e, "Failed to serialize event");
                                }
                            }
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            tracing::info!("SSE broadcast channel closed, ending stream");
                            break;
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            // Client fell behind, skip missed events
                            tracing::warn!(missed = n, "SSE client lagged, skipped events");
                            // Continue receiving
                        }
                    }
                }
            }
        }

        // Decrement connection count on stream end
        let remaining = state_for_drop.decrement_connections();
        tracing::info!(connections = remaining, "SSE client disconnected");
    };

    // Apply axum's SSE wrapper with keep-alive
    // Note: We handle our own pings above, but axum's KeepAlive provides
    // an additional layer of protection against proxy timeouts.
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(config.keepalive_interval)
            .text("keep-alive")
    )
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

    // -------------------------------------------------------------------------
    // TASK-33: SseAppState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sse_app_state_creation() {
        let state = SseAppState::new(SseConfig::default()).unwrap();
        assert_eq!(state.active_connections(), 0);
    }

    #[test]
    fn test_sse_app_state_invalid_config() {
        let invalid_config = SseConfig {
            keepalive_interval: Duration::ZERO,
            ..Default::default()
        };
        let result = SseAppState::new(invalid_config);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SseConfigError::InvalidKeepalive);
    }

    #[test]
    fn test_sse_app_state_broadcast_no_receivers() {
        let state = SseAppState::new(SseConfig::default()).unwrap();
        let event = McpSseEvent::ping_at(12345);

        // No receivers connected, should return 0
        let receivers = state.broadcast(event);
        assert_eq!(receivers, 0);
    }

    #[test]
    fn test_sse_app_state_broadcast_with_receiver() {
        let state = SseAppState::new(SseConfig::default()).unwrap();
        let mut rx = state.subscribe();

        let event = McpSseEvent::notification("test", serde_json::json!({"foo": "bar"}));
        let receivers = state.broadcast(event.clone());

        assert_eq!(receivers, 1);

        // Receiver should get the event
        let received = rx.try_recv().unwrap();
        assert_eq!(received, event);
    }

    #[test]
    fn test_sse_app_state_multiple_subscribers() {
        let state = SseAppState::new(SseConfig::default()).unwrap();
        let mut rx1 = state.subscribe();
        let mut rx2 = state.subscribe();
        let mut rx3 = state.subscribe();

        let event = McpSseEvent::response(1, serde_json::json!({"result": "ok"}));
        let receivers = state.broadcast(event.clone());

        assert_eq!(receivers, 3);

        // All receivers should get the same event
        assert_eq!(rx1.try_recv().unwrap(), event);
        assert_eq!(rx2.try_recv().unwrap(), event);
        assert_eq!(rx3.try_recv().unwrap(), event);
    }

    #[test]
    fn test_sse_app_state_connection_counting() {
        let state = SseAppState::new(SseConfig::default()).unwrap();

        assert_eq!(state.active_connections(), 0);

        let count1 = state.increment_connections();
        assert_eq!(count1, 1);
        assert_eq!(state.active_connections(), 1);

        let count2 = state.increment_connections();
        assert_eq!(count2, 2);
        assert_eq!(state.active_connections(), 2);

        let count3 = state.decrement_connections();
        assert_eq!(count3, 1);
        assert_eq!(state.active_connections(), 1);
    }

    #[test]
    fn test_sse_app_state_decrement_no_underflow() {
        let state = SseAppState::new(SseConfig::default()).unwrap();

        // Start at 0
        assert_eq!(state.active_connections(), 0);

        // Decrement when already at 0 should NOT underflow
        let count = state.decrement_connections();
        assert_eq!(count, 0);
        assert_eq!(state.active_connections(), 0);

        // Another decrement still stays at 0
        let count2 = state.decrement_connections();
        assert_eq!(count2, 0);
        assert_eq!(state.active_connections(), 0);

        // Now increment and decrement properly
        state.increment_connections();
        assert_eq!(state.active_connections(), 1);
        let count3 = state.decrement_connections();
        assert_eq!(count3, 0);
        assert_eq!(state.active_connections(), 0);
    }

    #[test]
    fn test_next_event_id_unique() {
        let id1 = next_event_id();
        let id2 = next_event_id();
        let id3 = next_event_id();

        assert!(id1.starts_with("evt-"));
        assert!(id2.starts_with("evt-"));
        assert!(id3.starts_with("evt-"));

        // All IDs should be unique
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_create_sse_router() {
        let state = SseAppState::new(SseConfig::default()).unwrap();
        let _router = create_sse_router(state);
        // Router creation should not panic
    }

    #[test]
    fn test_sse_app_state_debug_impl() {
        let state = SseAppState::new(SseConfig::default()).unwrap();
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("SseAppState"));
        assert!(debug_str.contains("active_connections"));
    }

    #[test]
    fn test_sse_handler_error_display() {
        let err = SseHandlerError::ChannelClosed;
        assert_eq!(err.to_string(), "broadcast channel closed");

        let err2 = SseHandlerError::Serialization("bad json".to_string());
        assert!(err2.to_string().contains("bad json"));

        let err3 = SseHandlerError::MaxDurationExceeded;
        assert!(err3.to_string().contains("duration exceeded"));
    }

    #[test]
    fn test_sse_app_state_clone() {
        let state = SseAppState::new(SseConfig::default()).unwrap();
        state.increment_connections();

        let state_clone = state.clone();

        // Both should see same connection count (Arc shared)
        assert_eq!(state.active_connections(), state_clone.active_connections());

        // Both should share the same broadcast channel
        let mut rx = state.subscribe();
        let event = McpSseEvent::ping_at(999);
        state_clone.broadcast(event.clone());

        assert_eq!(rx.try_recv().unwrap(), event);
    }

    // -------------------------------------------------------------------------
    // TASK-33: Integration Tests (require tokio runtime)
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_sse_handler_integration() {
        use axum::{
            body::Body,
            http::{Request, StatusCode},
        };
        use tower::ServiceExt;

        let state = SseAppState::new(
            SseConfig::default()
                .with_keepalive(Duration::from_millis(50))
                .with_max_duration(Duration::from_millis(200))
        ).unwrap();

        // Spawn a task to broadcast events
        let state_clone = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            state_clone.broadcast(McpSseEvent::notification("test", serde_json::json!({})));
        });

        let router = create_sse_router(state);

        let request = Request::builder()
            .uri("/events")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Check content-type is text/event-stream
        let content_type = response.headers().get("content-type").unwrap();
        assert!(content_type.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_sse_state_is_cloneable_and_sharable() {
        let state = SseAppState::new(SseConfig::default()).unwrap();

        // Test that state can be cloned and used across tasks
        let state1 = state.clone();
        let state2 = state.clone();

        let handle1 = tokio::spawn(async move {
            state1.increment_connections();
            tokio::time::sleep(Duration::from_millis(10)).await;
            state1.active_connections()
        });

        let handle2 = tokio::spawn(async move {
            state2.increment_connections();
            tokio::time::sleep(Duration::from_millis(10)).await;
            state2.active_connections()
        });

        let (count1, count2) = tokio::join!(handle1, handle2);

        // Both increments should be visible
        assert!(count1.unwrap() >= 1);
        assert!(count2.unwrap() >= 1);
        // Final count should be 2 (both increments applied)
        assert_eq!(state.active_connections(), 2);
    }

    #[tokio::test]
    async fn test_sse_broadcast_to_multiple_async_receivers() {
        let state = SseAppState::new(
            SseConfig::default().with_buffer_size(10)
        ).unwrap();

        let mut rx1 = state.subscribe();
        let mut rx2 = state.subscribe();

        // Broadcast from a separate task
        let state_clone = state.clone();
        tokio::spawn(async move {
            for i in 0..5 {
                let event = McpSseEvent::notification(
                    "counter",
                    serde_json::json!({"value": i})
                );
                state_clone.broadcast(event);
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        // Both receivers should get all events
        let mut count1 = 0;
        let mut count2 = 0;

        for _ in 0..5 {
            tokio::select! {
                Ok(_) = rx1.recv() => { count1 += 1; }
                _ = tokio::time::sleep(Duration::from_millis(100)) => { break; }
            }
        }

        for _ in 0..5 {
            tokio::select! {
                Ok(_) = rx2.recv() => { count2 += 1; }
                _ = tokio::time::sleep(Duration::from_millis(100)) => { break; }
            }
        }

        // Both should have received events
        assert!(count1 > 0, "rx1 should receive events");
        assert!(count2 > 0, "rx2 should receive events");
    }
}
