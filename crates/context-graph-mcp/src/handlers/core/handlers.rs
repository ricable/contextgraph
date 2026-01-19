//! Handlers struct definition and constructors.
//!
//! PRD v6 Section 10 - Handlers for all 14 MCP tools.
//!
//! TASK-INTEG-TOPIC: Added clustering dependencies for topic tools integration.

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::json;
use tracing::info;

use context_graph_core::clustering::{MultiSpaceClusterManager, TopicStabilityTracker};
use context_graph_core::monitoring::LayerStatusProvider;
use context_graph_core::traits::{
    MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor,
};

use crate::protocol::{JsonRpcId, JsonRpcResponse};

/// Request handlers for MCP protocol.
///
/// PRD v6 Section 10 - Supports all 14 MCP tools:
/// - Core: inject_context, store_memory, get_memetic_status, search_graph, trigger_consolidation
/// - Topic: get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts
/// - Curation: merge_concepts, forget_concept, boost_importance
/// - Dream: trigger_dream, get_dream_status
pub struct Handlers {
    /// Teleological memory store - stores TeleologicalFingerprint with 13 embeddings.
    pub(in crate::handlers) teleological_store: Arc<dyn TeleologicalMemoryStore>,

    /// UTL processor for computing learning metrics.
    pub(in crate::handlers) utl_processor: Arc<dyn UtlProcessor>,

    /// Multi-array embedding provider - generates all 13 embeddings per content.
    pub(in crate::handlers) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,

    /// Layer status provider for get_memetic_status.
    pub(in crate::handlers) layer_status_provider: Arc<dyn LayerStatusProvider>,

    /// Multi-space cluster manager for topic detection and clustering.
    /// TASK-INTEG-TOPIC: Added for topic tools integration.
    pub(in crate::handlers) cluster_manager: Arc<RwLock<MultiSpaceClusterManager>>,

    /// Topic stability tracker for portfolio-level stability metrics.
    /// TASK-INTEG-TOPIC: Added for topic tools integration.
    pub(in crate::handlers) stability_tracker: Arc<RwLock<TopicStabilityTracker>>,
}

impl Handlers {
    /// Create handlers with all dependencies explicitly provided.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint
    /// * `utl_processor` - UTL processor for learning metrics
    /// * `multi_array_provider` - 13-embedding generator
    /// * `layer_status_provider` - Provider for layer status information
    /// * `cluster_manager` - Multi-space cluster manager for topic detection
    /// * `stability_tracker` - Topic stability tracker for portfolio metrics
    #[allow(dead_code)]
    pub fn with_all(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
        cluster_manager: Arc<RwLock<MultiSpaceClusterManager>>,
        stability_tracker: Arc<RwLock<TopicStabilityTracker>>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            layer_status_provider,
            cluster_manager,
            stability_tracker,
        }
    }

    /// Create handlers with default clustering components.
    ///
    /// This is a convenience constructor that creates default cluster manager
    /// and stability tracker. Use `with_all` for full control over dependencies.
    ///
    /// TASK-INTEG-TOPIC: Added for backwards compatibility during integration.
    pub fn with_defaults(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        layer_status_provider: Arc<dyn LayerStatusProvider>,
    ) -> Self {
        // Create default cluster manager
        let cluster_manager = MultiSpaceClusterManager::with_defaults()
            .expect("Default cluster manager should always succeed");

        // Create default stability tracker
        let stability_tracker = TopicStabilityTracker::new();

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            layer_status_provider,
            cluster_manager: Arc::new(RwLock::new(cluster_manager)),
            stability_tracker: Arc::new(RwLock::new(stability_tracker)),
        }
    }

    /// Handle MCP initialize request.
    ///
    /// Returns server capabilities per MCP protocol.
    pub async fn handle_initialize(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("MCP initialize request received");

        let capabilities = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": false
                }
            },
            "serverInfo": {
                "name": "context-graph",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        JsonRpcResponse::success(id, capabilities)
    }

    /// Handle MCP initialized notification.
    ///
    /// This is a notification (no response expected), but we return
    /// an empty success for consistency in dispatch.
    pub fn handle_initialized_notification(&self) -> JsonRpcResponse {
        info!("MCP initialized notification received");
        JsonRpcResponse::success(None, json!({}))
    }

    /// Handle MCP shutdown request.
    ///
    /// Performs graceful shutdown of handlers.
    pub async fn handle_shutdown(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("MCP shutdown request received");
        JsonRpcResponse::success(id, json!({}))
    }
}
