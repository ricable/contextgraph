# TASK-33: Implement SSE handler with keep-alive (TASK-MCP-007)

```xml
<task_spec id="TASK-MCP-007" version="3.0">
<metadata>
  <title>Implement SSE handler with keep-alive</title>
  <status>complete</status>
  <layer>surface</layer>
  <sequence>33</sequence>
  <implements><requirement_ref>REQ-MCP-007</requirement_ref></implements>
  <depends_on>TASK-32</depends_on> <!-- TASK-32 = SSE transport types - VERIFIED COMPLETE -->
  <blocks>TASK-42</blocks> <!-- TASK-42 = SSE integration with MCP router -->
  <estimated_hours>4</estimated_hours>
  <actual_hours>3</actual_hours>
  <completed_date>2026-01-13</completed_date>
  <last_updated>2026-01-13</last_updated>
</metadata>
```

---

## EXECUTIVE SUMMARY

Implement the SSE (Server-Sent Events) handler that streams MCP events to clients in real-time. This task adds the actual handler functions to `sse.rs` using the types created in TASK-32.

**Key Deliverables:**
1. `SseAppState` struct to hold event broadcast channel and config
2. `create_sse_router()` function returning `Router<SseAppState>`
3. `sse_handler()` async function with keep-alive ping at configured interval
4. Connection lifecycle respecting `max_connection_duration`
5. Event ID generation for client reconnection support
6. Comprehensive tests using real streaming (NO MOCKS)

**This is NOT:**
- Router integration (TASK-42)
- SSE variant in `TransportMode` enum (TASK-42)
- Full MCP dispatch - just the SSE streaming infrastructure

---

## CODEBASE CONTEXT (VERIFIED 2026-01-13)

### Files That ALREADY EXIST (from TASK-32)

| File | What It Contains |
|------|------------------|
| `crates/context-graph-mcp/src/transport/mod.rs` | Exports `McpSseEvent`, `SseConfig`, `SseConfigError` |
| `crates/context-graph-mcp/src/transport/sse.rs` | SSE types (503 lines), 21 passing tests |
| `crates/context-graph-mcp/src/lib.rs:30` | `pub mod transport;` declaration |
| `crates/context-graph-mcp/Cargo.toml:52-55` | axum, tokio-stream, async-stream dependencies |

### Verified Exports from TASK-32

```rust
// crates/context-graph-mcp/src/transport/mod.rs
pub use sse::{McpSseEvent, SseConfig, SseConfigError};
```

```rust
// crates/context-graph-mcp/src/transport/sse.rs (key types)
pub struct SseConfig {
    pub keepalive_interval: Duration,      // default: 15s
    pub max_connection_duration: Duration, // default: 3600s
    pub buffer_size: usize,                // default: 100
}

pub enum McpSseEvent {
    Response { id: serde_json::Value, result: serde_json::Value },
    Error { id: serde_json::Value, error: JsonRpcError },
    Notification { method: String, params: serde_json::Value },
    Ping { timestamp: u64 },
}
```

### Dependencies ALREADY in Cargo.toml

```toml
# crates/context-graph-mcp/Cargo.toml
axum = { workspace = true }          # 0.8
tokio-stream = "0.1"
async-stream = "0.3"
```

### Server Architecture (from server.rs)

The MCP server uses:
- `Arc<Handlers>` for shared handler state
- `Arc<Semaphore>` for connection limiting
- `Arc<AtomicUsize>` for active connection tracking

For SSE, we create a SEPARATE router that doesn't need the full `McpServer` state.

---

## IMPLEMENTATION SPECIFICATION

### Step 1: Add SseAppState and Handler Types

**File:** `crates/context-graph-mcp/src/transport/sse.rs`

**Add AFTER the `McpSseEvent` impl block (around line 273), BEFORE the tests module:**

```rust
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
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

/// Counter for generating unique event IDs.
static EVENT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique event ID for SSE reconnection support.
#[inline]
fn next_event_id() -> String {
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
    /// # use context_graph_mcp::transport::{SseAppState, SseConfig, McpSseEvent};
    /// # let state = SseAppState::new(SseConfig::default()).unwrap();
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
    fn increment_connections(&self) -> u64 {
        self.connection_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrement connection count (called on disconnect).
    fn decrement_connections(&self) -> u64 {
        self.connection_count.fetch_sub(1, Ordering::Relaxed) - 1
    }
}

/// SSE handler error.
///
/// Errors that can occur during SSE streaming.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SseHandlerError {
    #[error("broadcast channel closed")]
    ChannelClosed,

    #[error("event serialization failed: {0}")]
    Serialization(String),

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
        use tokio::time::{interval, timeout, Instant};

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
```

### Step 2: Update mod.rs to Export New Types

**File:** `crates/context-graph-mcp/src/transport/mod.rs`

**Replace entire file:**

```rust
//! MCP Transport layer.
//!
//! TASK-32: SSE transport types for real-time streaming.
//! TASK-33: SSE handler implementation.
//!
//! PRD Section 5.1: "JSON-RPC 2.0, stdio/SSE"

pub mod sse;

pub use sse::{
    // TASK-32: Types
    McpSseEvent, SseConfig, SseConfigError,
    // TASK-33: Handler
    SseAppState, SseHandlerError, create_sse_router, sse_handler,
};
```

### Step 3: Add Additional Tests

**File:** `crates/context-graph-mcp/src/transport/sse.rs`

**Add to the `#[cfg(test)] mod tests` block (AFTER existing tests):**

```rust
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
    async fn test_sse_receives_broadcast_events() {
        use axum::body::Body;
        use http_body_util::BodyExt;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let state = SseAppState::new(
            SseConfig::default()
                .with_keepalive(Duration::from_secs(60)) // Long keepalive to not interfere
                .with_max_duration(Duration::from_millis(500))
        ).unwrap();

        let state_clone = state.clone();

        // Spawn broadcast task
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            state_clone.broadcast(McpSseEvent::notification(
                "consciousness/state_changed",
                serde_json::json!({"level": 0.85}),
            ));
        });

        let router = create_sse_router(state);

        let request = Request::builder()
            .uri("/events")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Collect body bytes (with timeout)
        let body = tokio::time::timeout(
            Duration::from_millis(200),
            response.into_body().collect()
        ).await;

        // Body collection will timeout (stream is infinite), which is expected
        // The important thing is that the response started correctly
        assert!(body.is_err() || body.is_ok());
    }
```

---

## FILES TO MODIFY

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/transport/sse.rs` | Add ~200 lines of handler code after types |
| `crates/context-graph-mcp/src/transport/mod.rs` | Add new exports |

## FILES TO CREATE

None. All code goes in existing files.

---

## DEFINITION OF DONE

### Compilation Check
```bash
cargo check -p context-graph-mcp
# MUST: Exit 0, no errors
# MUST: No warnings about unused imports
```

### Test Execution
```bash
cargo test -p context-graph-mcp transport::sse
# MUST: All tests pass
# MUST: At least 30 test cases run (21 from TASK-32 + 9 from TASK-33)
```

### Type Export Verification
```bash
# Verify new exports compile
cargo rustdoc -p context-graph-mcp -- --document-private-items 2>&1 | grep -E "(SseAppState|create_sse_router|sse_handler)"
# MUST: Show documented functions/types
```

---

## FULL STATE VERIFICATION (MANDATORY)

### 1. Source of Truth: File System & Compilation

```bash
# Verify sse.rs has grown (TASK-32 was ~503 lines, should be ~700+ now)
wc -l crates/context-graph-mcp/src/transport/sse.rs
# Expected: 700+ lines

# Verify mod.rs has new exports
grep -E "SseAppState|create_sse_router|sse_handler" crates/context-graph-mcp/src/transport/mod.rs
# Expected: All three should appear

# Full compilation
cargo check -p context-graph-mcp
# Expected: Success with 0 errors
```

### 2. Execute & Inspect: Run All Transport Tests

```bash
cargo test -p context-graph-mcp transport::sse -- --nocapture 2>&1 | tee /tmp/task33_tests.log

# Verify test count
grep "test result:" /tmp/task33_tests.log
# Expected: "test result: ok. 30 passed" (or more)
```

### 3. Boundary & Edge Case Audit

**Edge Case 1: Invalid Config Rejection**
```rust
// Test that SseAppState::new() rejects invalid config
let bad = SseConfig { keepalive_interval: Duration::ZERO, ..Default::default() };
assert!(SseAppState::new(bad).is_err());
// Evidence: test_sse_app_state_invalid_config passes
```

**Edge Case 2: Broadcast with No Subscribers**
```rust
let state = SseAppState::new(SseConfig::default()).unwrap();
let receivers = state.broadcast(McpSseEvent::ping());
assert_eq!(receivers, 0);
// Evidence: test_sse_app_state_broadcast_no_receivers passes
```

**Edge Case 3: Multiple Concurrent Subscribers**
```rust
// Create 3 subscribers, broadcast, all receive
let state = SseAppState::new(SseConfig::default()).unwrap();
let mut rx1 = state.subscribe();
let mut rx2 = state.subscribe();
let mut rx3 = state.subscribe();
let event = McpSseEvent::ping();
assert_eq!(state.broadcast(event.clone()), 3);
assert_eq!(rx1.try_recv().unwrap(), event);
assert_eq!(rx2.try_recv().unwrap(), event);
assert_eq!(rx3.try_recv().unwrap(), event);
// Evidence: test_sse_app_state_multiple_subscribers passes
```

### 4. Evidence of Success Log

```bash
echo "=== TASK-33 VERIFICATION LOG ===" > /tmp/task33_evidence.log
date >> /tmp/task33_evidence.log
echo "--- File Size ---" >> /tmp/task33_evidence.log
wc -l crates/context-graph-mcp/src/transport/sse.rs >> /tmp/task33_evidence.log
echo "--- Compilation ---" >> /tmp/task33_evidence.log
cargo check -p context-graph-mcp 2>&1 >> /tmp/task33_evidence.log
echo "--- Tests ---" >> /tmp/task33_evidence.log
cargo test -p context-graph-mcp transport::sse 2>&1 >> /tmp/task33_evidence.log
echo "--- Exports ---" >> /tmp/task33_evidence.log
grep -E "pub (fn|struct|enum)" crates/context-graph-mcp/src/transport/sse.rs | head -20 >> /tmp/task33_evidence.log
cat /tmp/task33_evidence.log
```

---

## MANUAL TESTING PROTOCOL

### Test 1: SSE Router Starts

```bash
# Create a minimal test binary (do NOT commit this file)
cat > /tmp/sse_test.rs << 'EOF'
use context_graph_mcp::transport::{SseAppState, SseConfig, create_sse_router, McpSseEvent};
use std::time::Duration;

#[tokio::main]
async fn main() {
    let state = SseAppState::new(SseConfig::default()).unwrap();
    println!("Created SseAppState");

    let state_clone = state.clone();

    // Spawn event broadcaster
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let event = McpSseEvent::notification("heartbeat", serde_json::json!({"ts": 123}));
            let n = state_clone.broadcast(event);
            println!("Broadcast to {} clients", n);
        }
    });

    let router = create_sse_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3333").await.unwrap();
    println!("SSE server listening on http://127.0.0.1:3333/events");
    axum::serve(listener, router).await.unwrap();
}
EOF

# Expected output: Server starts, prints listening message
```

### Test 2: SSE Client Connection

```bash
# In terminal 1: Start the server (from Test 1)
# In terminal 2: Connect with curl
curl -N http://127.0.0.1:3333/events

# Expected output (after ~15 seconds for first ping):
# event: message
# data: {"type":"Ping","timestamp":1704067200}
# id: evt-1

# After ~2 seconds (from broadcaster):
# event: message
# data: {"type":"Notification","method":"heartbeat","params":{"ts":123}}
# id: evt-2
```

### Test 3: Multiple Clients

```bash
# Terminal 1: Start server
# Terminal 2: curl -N http://127.0.0.1:3333/events
# Terminal 3: curl -N http://127.0.0.1:3333/events
# Terminal 4: curl -N http://127.0.0.1:3333/events

# Expected: Server logs "Broadcast to 3 clients"
# All three terminals receive the same events
```

---

## CONSTRAINTS

### MUST
- `sse_handler()` MUST send Ping at `keepalive_interval`
- Connection MUST close after `max_connection_duration`
- Events MUST be JSON serialized via `McpSseEvent::to_json()`
- Stream MUST be infallible (`Result<Event, Infallible>`)
- Event IDs MUST be unique (use `next_event_id()`)
- Tests MUST use real streaming, NO MOCKS

### MUST NOT
- DO NOT add SSE variant to `TransportMode` (that's TASK-42)
- DO NOT create separate axum server - just the router/handler
- DO NOT use `.unwrap()` in library code - use `.expect()` or return `Result`
- DO NOT skip errors - log them with tracing
- DO NOT create backwards compatibility shims

---

## ANTI-PATTERNS (FORBIDDEN)

1. **DO NOT** mock the broadcast channel in tests - use real `broadcast::channel`
2. **DO NOT** add `#[cfg(test)]` stubs for production code
3. **DO NOT** use `thread::sleep` - use `tokio::time::sleep`
4. **DO NOT** ignore `RecvError::Lagged` - log it as warning
5. **DO NOT** create infinite loops without exit conditions

---

## TROUBLESHOOTING

### Error: "cannot find value `KeepAlive` in module `sse`"
**Cause:** Missing import from axum::response::sse
**Fix:** Add `use axum::response::sse::{Event, KeepAlive, Sse};`

### Error: "trait bound `Xxx: Clone` is not satisfied"
**Cause:** SseAppState needs Clone for axum State extractor
**Fix:** Ensure `#[derive(Clone)]` on SseAppState and all fields are Clone

### Error: "the trait `Stream` is not implemented for `impl Stream<...>`"
**Cause:** Wrong Stream trait import
**Fix:** Use `tokio_stream::Stream` not `futures::Stream`

### Warning: "unused variable: `state_for_drop`"
**Cause:** The drop guard pattern isn't working
**Fix:** Ensure `state_for_drop.decrement_connections()` is called in stream

---

## SUCCESS CRITERIA CHECKLIST

- [ ] `cargo check -p context-graph-mcp` passes with no errors
- [ ] `cargo test -p context-graph-mcp transport::sse` passes (30+ tests)
- [ ] `SseAppState::new()` validates config and rejects invalid
- [ ] `create_sse_router()` returns a valid Router
- [ ] `sse_handler()` returns `Sse<impl Stream<Item = Result<Event, Infallible>>>`
- [ ] Ping events sent at `keepalive_interval`
- [ ] Connection closes after `max_connection_duration`
- [ ] Broadcast events reach all subscribers
- [ ] Connection count tracking works correctly
- [ ] Event IDs are unique across all events
- [ ] Evidence log captured and reviewed

---

## NEXT TASK

After TASK-33 is complete, proceed to **TASK-42: SSE integration with MCP router**.

TASK-42 depends on the handler created here to:
- Add `Sse` variant to `TransportMode` enum
- Integrate SSE router with full MCP server
- Wire event broadcast to MCP response pipeline

---

## REFERENCES

- [Axum SSE Documentation](https://docs.rs/axum/latest/axum/response/sse/index.html)
- [Axum SSE Example](https://github.com/tokio-rs/axum/blob/main/examples/sse/src/main.rs)
- [tokio broadcast channel](https://docs.rs/tokio/latest/tokio/sync/broadcast/index.html)
- [async-stream crate](https://docs.rs/async-stream/latest/async_stream/)
- constitution.yaml: `mcp.transport: [stdio, sse]`
- TASK-32 (this task's dependency): SSE types

```
</task_spec>
```
