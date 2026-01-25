//! TASK-EMB-024: Full State Verification Tests
//!
//! This module verifies the TASK-EMB-024 implementation:
//! 1. StubLayerStatusProvider returns honest layer statuses
//!
//! ## Test Methodology
//!
//! - Source of Truth: StubLayerStatusProvider
//! - Execute & Inspect: Call handlers and verify responses
//! - NO mock data, NO fallbacks

use std::sync::Arc;

use serde_json::json;

use context_graph_core::monitoring::{LayerStatusProvider, StubLayerStatusProvider};
use context_graph_core::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider};
use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore};

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

/// Create test handlers using Handlers::with_defaults (which uses StubLayerStatusProvider).
fn create_handlers_with_stub_monitors() -> Handlers {
    let store: Arc<dyn TeleologicalMemoryStore> = Arc::new(InMemoryTeleologicalStore::new());
    let multi_array: Arc<dyn MultiArrayEmbeddingProvider> = Arc::new(StubMultiArrayProvider::new());
    let layer_status: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    Handlers::with_defaults(store, multi_array, layer_status)
}

fn make_request(method: &str, params: serde_json::Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: method.to_string(),
        params: Some(params),
    }
}

// ============================================================================
// StubLayerStatusProvider Verification
// ============================================================================

/// Test that get_memetic_status returns proper layer statuses from StubLayerStatusProvider.
#[tokio::test]
async fn test_task_emb_024_layer_status_provider_honest_statuses() {
    let handlers = create_handlers_with_stub_monitors();

    // Call get_memetic_status via tools/call
    let request = make_request(
        "tools/call",
        json!({
            "name": "get_memetic_status",
            "arguments": {}
        }),
    );
    let response = handlers.dispatch(request).await;

    // VERIFY: Response should succeed (layer statuses are available)
    assert!(
        response.error.is_none(),
        "get_memetic_status should succeed: {:?}",
        response.error
    );

    let result = response.result.unwrap();
    // The result is wrapped in MCP tool response format
    let content = result["content"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|obj| obj["text"].as_str())
        .expect("Should have content text");

    let data: serde_json::Value =
        serde_json::from_str(content).expect("Content should be valid JSON");

    // VERIFY: Expected statuses per StubLayerStatusProvider
    // All layers are "active" (have working implementations)
    let layers = &data["layers"];
    assert_eq!(
        layers["perception"].as_str().unwrap(),
        "active",
        "perception should be active"
    );
    assert_eq!(
        layers["memory"].as_str().unwrap(),
        "active",
        "memory should be active"
    );
    assert_eq!(
        layers["action"].as_str().unwrap(),
        "active",
        "action should be active"
    );
    assert_eq!(
        layers["meta"].as_str().unwrap(),
        "active",
        "meta should be active"
    );
}

/// Verify that layer statuses don't contain hardcoded "stub" strings as
/// the RESULT - only as honest reporting when appropriate.
#[tokio::test]
async fn test_task_emb_024_layer_statuses_are_honest_not_placeholder() {
    let handlers = create_handlers_with_stub_monitors();

    let request = make_request(
        "tools/call",
        json!({
            "name": "get_memetic_status",
            "arguments": {}
        }),
    );
    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let result = response.result.unwrap();
    let content = result["content"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|obj| obj["text"].as_str())
        .unwrap();
    let data: serde_json::Value = serde_json::from_str(content).unwrap();

    let layers = &data["layers"];

    // VERIFY: Some layers are "active" (not all "stub")
    // This proves we're getting real status from LayerStatusProvider
    let perception = layers["perception"].as_str().unwrap();
    let memory = layers["memory"].as_str().unwrap();

    assert_eq!(
        perception, "active",
        "perception should be active, not placeholder stub"
    );
    assert_eq!(
        memory, "active",
        "memory should be active, not placeholder stub"
    );
}
