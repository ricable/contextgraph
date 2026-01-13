//! MCP Transport layer.
//!
//! TASK-32: SSE transport types for real-time streaming.
//! TASK-33: SSE handler implementation (future).
//!
//! PRD Section 5.1: "JSON-RPC 2.0, stdio/SSE"

pub mod sse;

pub use sse::{McpSseEvent, SseConfig, SseConfigError};
