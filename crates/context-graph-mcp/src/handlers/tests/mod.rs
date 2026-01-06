//! MCP Protocol Compliance Unit Tests
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! TASK-S003: Added GoalAlignmentCalculator and GoalHierarchy for purpose operations.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore.
//!
//! Tests verify compliance with MCP protocol version 2024-11-05
//! Reference: https://spec.modelcontextprotocol.io/specification/2024-11-05/

mod cognitive_pulse;
mod error_codes;
mod full_state_verification;
mod full_state_verification_johari;
mod full_state_verification_search;
mod full_state_verification_purpose;
mod initialize;
mod manual_fsv_purpose;
mod memory;
mod meta_cognitive;
mod purpose;
mod search;
mod tools_list;
mod tools_call;
mod utl;

use std::sync::Arc;

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::purpose::{GoalHierarchy, GoalId, GoalLevel, GoalNode};
use context_graph_core::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor};
use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor};

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

/// Create test handlers with real stub implementations (no mocks).
///
/// TASK-S001: Uses TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
/// TASK-S003: Uses DefaultAlignmentCalculator and test GoalHierarchy.
/// NO legacy MemoryStore support.
pub(crate) fn create_test_handlers() -> Handlers {
    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> = Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> = Arc::new(DefaultAlignmentCalculator::new());
    let goal_hierarchy = create_test_hierarchy();
    Handlers::new(teleological_store, utl_processor, multi_array_provider, alignment_calculator, goal_hierarchy)
}

/// Create test handlers WITHOUT a North Star goal (for testing error cases).
pub(crate) fn create_test_handlers_no_north_star() -> Handlers {
    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> = Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator: Arc<dyn GoalAlignmentCalculator> = Arc::new(DefaultAlignmentCalculator::new());
    let goal_hierarchy = GoalHierarchy::new(); // Empty hierarchy
    Handlers::new(teleological_store, utl_processor, multi_array_provider, alignment_calculator, goal_hierarchy)
}

/// Create a test goal hierarchy with North Star and sub-goals.
///
/// Hierarchy:
/// - NorthStar: "Build the best ML learning system"
///   - Strategic: "Improve retrieval accuracy"
///     - Tactical: "Implement semantic search"
///       - Immediate: "Add vector similarity"
///   - Strategic: "Enhance user experience"
pub(crate) fn create_test_hierarchy() -> GoalHierarchy {
    let mut hierarchy = GoalHierarchy::new();

    // Create embedding that varies by dimension for distinctiveness
    let ns_embedding: Vec<f32> = (0..1024)
        .map(|i| (i as f32 / 1024.0).sin() * 0.8)
        .collect();

    // North Star
    hierarchy
        .add_goal(GoalNode::north_star(
            "ns_ml_system",
            "Build the best ML learning system",
            ns_embedding.clone(),
            vec!["ml".into(), "learning".into(), "system".into()],
        ))
        .expect("Failed to add North Star");

    // Strategic goal 1
    hierarchy
        .add_goal(GoalNode::child(
            "s1_retrieval",
            "Improve retrieval accuracy",
            GoalLevel::Strategic,
            GoalId::new("ns_ml_system"),
            ns_embedding.clone(),
            0.8,
            vec!["retrieval".into(), "accuracy".into()],
        ))
        .expect("Failed to add strategic goal 1");

    // Strategic goal 2
    hierarchy
        .add_goal(GoalNode::child(
            "s2_ux",
            "Enhance user experience",
            GoalLevel::Strategic,
            GoalId::new("ns_ml_system"),
            ns_embedding.clone(),
            0.7,
            vec!["ux".into(), "user".into()],
        ))
        .expect("Failed to add strategic goal 2");

    // Tactical goal
    hierarchy
        .add_goal(GoalNode::child(
            "t1_semantic",
            "Implement semantic search",
            GoalLevel::Tactical,
            GoalId::new("s1_retrieval"),
            ns_embedding.clone(),
            0.6,
            vec!["semantic".into(), "search".into()],
        ))
        .expect("Failed to add tactical goal");

    // Immediate goal
    hierarchy
        .add_goal(GoalNode::child(
            "i1_vector",
            "Add vector similarity",
            GoalLevel::Immediate,
            GoalId::new("t1_semantic"),
            ns_embedding,
            0.5,
            vec!["vector".into(), "similarity".into()],
        ))
        .expect("Failed to add immediate goal");

    hierarchy
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
