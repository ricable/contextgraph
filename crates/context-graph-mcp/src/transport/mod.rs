//! MCP Transport layer.
//!
//! TASK-32: SSE transport types for real-time streaming.
//! TASK-33: SSE handler implementation.
//!
//! PRD Section 5.1: "JSON-RPC 2.0, stdio/SSE"

pub mod sse;

// Note: McpSseEvent is re-exported for external use even if not used within this crate.
// It's part of the public API documented in sse.rs doctests.
#[allow(unused_imports)]
pub use sse::{create_sse_router, McpSseEvent, SseAppState, SseConfig};
