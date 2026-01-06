//! MCP Protocol Compliance Unit Tests
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore.
//!
//! Tests verify compliance with MCP protocol version 2024-11-05
//! Reference: https://spec.modelcontextprotocol.io/specification/2024-11-05/

mod cognitive_pulse;
mod error_codes;
mod full_state_verification;
mod full_state_verification_search;
mod initialize;
mod memory;
mod meta_cognitive;
mod search;
mod tools_list;
mod tools_call;
mod utl;

use std::sync::Arc;

use context_graph_core::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor};
use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor};

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

/// Create test handlers with real stub implementations (no mocks).
///
/// TASK-S001: Uses TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
/// NO legacy MemoryStore support.
pub(crate) fn create_test_handlers() -> Handlers {
    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> = Arc::new(StubMultiArrayProvider::new());
    Handlers::new(teleological_store, utl_processor, multi_array_provider)
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
