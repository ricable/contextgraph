//! MCP Protocol Compliance Unit Tests
//!
//! Tests verify compliance with MCP protocol version 2024-11-05
//! Reference: https://spec.modelcontextprotocol.io/specification/2024-11-05/

mod error_codes;
mod initialize;
mod memory;
mod meta_cognitive;
mod tools;
mod utl;

use std::sync::Arc;

use context_graph_core::stubs::{InMemoryStore, StubUtlProcessor};
use context_graph_core::traits::{MemoryStore, UtlProcessor};

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

/// Create test handlers with real stub implementations (no mocks).
pub(crate) fn create_test_handlers() -> Handlers {
    let memory_store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    Handlers::new(memory_store, utl_processor)
}

/// Create a JSON-RPC request for testing.
pub(crate) fn make_request(
    method: &str,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id,
        method: method.to_string(),
        params,
    }
}
