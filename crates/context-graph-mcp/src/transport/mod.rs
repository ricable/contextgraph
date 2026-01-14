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
    SseAppState, SseHandlerError, create_sse_router, sse_handler, next_event_id,
};
