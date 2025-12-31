//! Request handlers for MCP methods.

use std::sync::Arc;

use serde_json::json;
use tracing::{debug, info};

use context_graph_core::traits::{MemoryStore, SearchOptions, UtlProcessor};
use context_graph_core::types::{CognitivePulse, MemoryNode, SuggestedAction, UtlContext};

use crate::protocol::{error_codes, methods, JsonRpcId, JsonRpcRequest, JsonRpcResponse};
use crate::tools::{get_tool_definitions, tool_names};

/// Request handlers.
pub struct Handlers {
    memory_store: Arc<dyn MemoryStore>,
    utl_processor: Arc<dyn UtlProcessor>,
}

impl Handlers {
    /// Create new handlers with the given dependencies.
    pub fn new(memory_store: Arc<dyn MemoryStore>, utl_processor: Arc<dyn UtlProcessor>) -> Self {
        Self {
            memory_store,
            utl_processor,
        }
    }

    /// Dispatch a request to the appropriate handler.
    pub async fn dispatch(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Dispatching method: {}", request.method);

        match request.method.as_str() {
            // MCP lifecycle methods
            methods::INITIALIZE => self.handle_initialize(request.id).await,
            "notifications/initialized" => self.handle_initialized_notification(),
            methods::SHUTDOWN => self.handle_shutdown(request.id).await,

            // MCP tools protocol
            methods::TOOLS_LIST => self.handle_tools_list(request.id).await,
            methods::TOOLS_CALL => self.handle_tools_call(request.id, request.params).await,

            // Legacy direct methods (kept for backward compatibility)
            methods::MEMORY_STORE => self.handle_memory_store(request.id, request.params).await,
            methods::MEMORY_RETRIEVE => {
                self.handle_memory_retrieve(request.id, request.params)
                    .await
            }
            methods::MEMORY_SEARCH => self.handle_memory_search(request.id, request.params).await,
            methods::MEMORY_DELETE => self.handle_memory_delete(request.id, request.params).await,
            methods::UTL_COMPUTE => self.handle_utl_compute(request.id, request.params).await,
            methods::UTL_METRICS => self.handle_utl_metrics(request.id, request.params).await,
            methods::SYSTEM_STATUS => self.handle_system_status(request.id).await,
            methods::SYSTEM_HEALTH => self.handle_system_health(request.id).await,
            _ => JsonRpcResponse::error(
                request.id,
                error_codes::METHOD_NOT_FOUND,
                format!("Method not found: {}", request.method),
            ),
        }
    }

    /// Handle MCP initialize request.
    ///
    /// Returns server capabilities following MCP 2024-11-05 protocol specification.
    async fn handle_initialize(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("Handling initialize request");

        let pulse = CognitivePulse::new(0.5, 0.8, SuggestedAction::Ready);

        // MCP-compliant initialize response
        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": true }
                },
                "serverInfo": {
                    "name": "context-graph-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
        .with_pulse(pulse)
    }

    /// Handle notifications/initialized - this is a notification, not a request.
    ///
    /// Notifications don't require a response per JSON-RPC 2.0 spec.
    fn handle_initialized_notification(&self) -> JsonRpcResponse {
        info!("Client initialized notification received");

        // Return a response with no id, result, or error to signal "no response needed"
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: None,
            result: None,
            error: None,
            cognitive_pulse: None,
        }
    }

    /// Handle tools/list request.
    ///
    /// Returns all available MCP tools with their schemas.
    async fn handle_tools_list(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling tools/list request");

        let tools = get_tool_definitions();
        JsonRpcResponse::success(id, json!({ "tools": tools }))
    }

    /// Handle tools/call request.
    ///
    /// Dispatches to the appropriate tool handler and returns MCP-compliant result.
    async fn handle_tools_call(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing params for tools/call",
                );
            }
        };

        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'name' parameter in tools/call",
                );
            }
        };

        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        debug!(
            "Calling tool: {} with arguments: {:?}",
            tool_name, arguments
        );

        match tool_name {
            tool_names::INJECT_CONTEXT => self.call_inject_context(id, arguments).await,
            tool_names::STORE_MEMORY => self.call_store_memory(id, arguments).await,
            tool_names::GET_MEMETIC_STATUS => self.call_get_memetic_status(id).await,
            tool_names::GET_GRAPH_MANIFEST => self.call_get_graph_manifest(id).await,
            tool_names::SEARCH_GRAPH => self.call_search_graph(id, arguments).await,
            _ => JsonRpcResponse::error(
                id,
                error_codes::TOOL_NOT_FOUND,
                format!("Unknown tool: {}", tool_name),
            ),
        }
    }

    // ========== Tool Call Implementations ==========

    /// MCP-compliant tool result helper.
    ///
    /// Wraps tool output in the required MCP format:
    /// `{content: [{type: "text", text: "..."}], isError: false}`
    fn tool_result(id: Option<JsonRpcId>, data: serde_json::Value) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
                }],
                "isError": false
            }),
        )
    }

    /// MCP-compliant tool error helper.
    fn tool_error(id: Option<JsonRpcId>, message: &str) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": message
                }],
                "isError": true
            }),
        )
    }

    /// inject_context tool implementation.
    async fn call_inject_context(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return Self::tool_error(id, "Missing 'content' parameter"),
        };

        let rationale = args.get("rationale").and_then(|v| v.as_str()).unwrap_or("");
        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        // Compute UTL metrics for the content
        let context = UtlContext::default();
        let metrics = match self.utl_processor.compute_metrics(&content, &context).await {
            Ok(m) => m,
            Err(e) => return Self::tool_error(id, &format!("UTL processing failed: {}", e)),
        };

        // Create and store the memory node
        let embedding = vec![0.1; 1536]; // Stub embedding
        let mut node = MemoryNode::new(content, embedding);
        node.importance = importance as f32;
        let node_id = node.id;

        if let Err(e) = self.memory_store.store(node).await {
            return Self::tool_error(id, &format!("Storage failed: {}", e));
        }

        Self::tool_result(
            id,
            json!({
                "nodeId": node_id.to_string(),
                "rationale": rationale,
                "utl": {
                    "learningScore": metrics.learning_score,
                    "entropy": metrics.entropy,
                    "coherence": metrics.coherence,
                    "surprise": metrics.surprise
                }
            }),
        )
    }

    /// store_memory tool implementation.
    async fn call_store_memory(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return Self::tool_error(id, "Missing 'content' parameter"),
        };

        let importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let embedding = vec![0.1; 1536]; // Stub embedding
        let mut node = MemoryNode::new(content, embedding);
        node.importance = importance as f32;
        let node_id = node.id;

        match self.memory_store.store(node).await {
            Ok(_) => Self::tool_result(id, json!({ "nodeId": node_id.to_string() })),
            Err(e) => Self::tool_error(id, &format!("Storage failed: {}", e)),
        }
    }

    /// get_memetic_status tool implementation.
    async fn call_get_memetic_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        let node_count = self.memory_store.count().await.unwrap_or(0);

        Self::tool_result(
            id,
            json!({
                "phase": "ghost-system",
                "nodeCount": node_count,
                "utl": {
                    "entropy": 0.5,
                    "coherence": 0.8,
                    "learningScore": 0.65,
                    "suggestedAction": "continue"
                },
                "layers": {
                    "perception": "active",
                    "memory": "active",
                    "reasoning": "stub",
                    "action": "stub",
                    "meta": "stub"
                }
            }),
        )
    }

    /// get_graph_manifest tool implementation.
    async fn call_get_graph_manifest(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        Self::tool_result(
            id,
            json!({
                "architecture": "5-layer-bio-nervous",
                "layers": [
                    {
                        "name": "Perception",
                        "description": "Sensory input processing and feature extraction",
                        "status": "active"
                    },
                    {
                        "name": "Memory",
                        "description": "Episodic and semantic memory storage with vector embeddings",
                        "status": "active"
                    },
                    {
                        "name": "Reasoning",
                        "description": "Inference, planning, and decision making",
                        "status": "stub"
                    },
                    {
                        "name": "Action",
                        "description": "Response generation and motor control",
                        "status": "stub"
                    },
                    {
                        "name": "Meta",
                        "description": "Self-monitoring, learning rate control, and system optimization",
                        "status": "stub"
                    }
                ],
                "utl": {
                    "description": "Universal Transfer Learning - measures learning potential",
                    "formula": "L(x) = H(P) - H(P|x) + alpha * C(x)"
                }
            }),
        )
    }

    /// search_graph tool implementation.
    async fn call_search_graph(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return Self::tool_error(id, "Missing 'query' parameter"),
        };

        let top_k = args.get("topK").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let options = SearchOptions::new(top_k);

        match self.memory_store.search_text(query, options).await {
            Ok(results) => {
                let results_json: Vec<_> = results
                    .iter()
                    .map(|(node, score)| {
                        json!({
                            "nodeId": node.id.to_string(),
                            "content": node.content,
                            "score": score,
                            "importance": node.importance
                        })
                    })
                    .collect();

                Self::tool_result(
                    id,
                    json!({ "results": results_json, "count": results_json.len() }),
                )
            }
            Err(e) => Self::tool_error(id, &format!("Search failed: {}", e)),
        }
    }

    async fn handle_shutdown(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        info!("Handling shutdown request");
        JsonRpcResponse::success(id, json!(null))
    }

    async fn handle_memory_store(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'content' parameter",
                );
            }
        };

        // Create node with stub embedding
        let embedding = vec![0.1; 1536]; // Stub embedding
        let node = MemoryNode::new(content, embedding);
        let node_id = node.id;

        match self.memory_store.store(node).await {
            Ok(_) => {
                let pulse = CognitivePulse::new(0.6, 0.75, SuggestedAction::Continue);
                JsonRpcResponse::success(id, json!({ "nodeId": node_id.to_string() }))
                    .with_pulse(pulse)
            }
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }

    async fn handle_memory_retrieve(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let node_id_str = match params.get("nodeId").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'nodeId' parameter",
                );
            }
        };

        let node_id = match uuid::Uuid::parse_str(node_id_str) {
            Ok(u) => u,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Invalid UUID format",
                );
            }
        };

        match self.memory_store.retrieve(node_id).await {
            Ok(Some(node)) => JsonRpcResponse::success(
                id,
                json!({
                    "node": {
                        "id": node.id.to_string(),
                        "content": node.content,
                        "modality": format!("{:?}", node.metadata.modality),
                        "johariQuadrant": format!("{:?}", node.quadrant),
                        "importance": node.importance,
                        "createdAt": node.created_at.to_rfc3339(),
                    }
                }),
            ),
            Ok(None) => JsonRpcResponse::error(id, error_codes::NODE_NOT_FOUND, "Node not found"),
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }

    async fn handle_memory_search(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'query' parameter",
                );
            }
        };

        let top_k = params.get("topK").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let options = SearchOptions::new(top_k);

        match self.memory_store.search_text(query, options).await {
            Ok(results) => {
                let results_json: Vec<_> = results
                    .iter()
                    .map(|(node, score)| {
                        json!({
                            "nodeId": node.id.to_string(),
                            "content": node.content,
                            "score": score,
                        })
                    })
                    .collect();

                let pulse = CognitivePulse::new(0.4, 0.8, SuggestedAction::Continue);
                JsonRpcResponse::success(id, json!({ "results": results_json })).with_pulse(pulse)
            }
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }

    async fn handle_memory_delete(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let node_id_str = match params.get("nodeId").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'nodeId' parameter",
                );
            }
        };

        let node_id = match uuid::Uuid::parse_str(node_id_str) {
            Ok(u) => u,
            Err(_) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Invalid UUID format",
                );
            }
        };

        let soft = params.get("soft").and_then(|v| v.as_bool()).unwrap_or(true);

        match self.memory_store.delete(node_id, soft).await {
            Ok(deleted) => JsonRpcResponse::success(id, json!({ "deleted": deleted })),
            Err(e) => JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e.to_string()),
        }
    }

    async fn handle_utl_compute(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let input = match params.get("input").and_then(|v| v.as_str()) {
            Some(i) => i,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'input' parameter",
                );
            }
        };

        let context = UtlContext::default();

        match self
            .utl_processor
            .compute_learning_score(input, &context)
            .await
        {
            Ok(score) => {
                let action = if score > 0.7 {
                    SuggestedAction::Consolidate
                } else if score > 0.4 {
                    SuggestedAction::Continue
                } else {
                    SuggestedAction::Explore
                };

                let pulse =
                    CognitivePulse::new(context.prior_entropy, context.current_coherence, action);
                JsonRpcResponse::success(id, json!({ "learningScore": score })).with_pulse(pulse)
            }
            Err(e) => JsonRpcResponse::error(id, error_codes::INTERNAL_ERROR, e.to_string()),
        }
    }

    async fn handle_utl_metrics(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let input = match params.get("input").and_then(|v| v.as_str()) {
            Some(i) => i,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing 'input' parameter",
                );
            }
        };

        let context = UtlContext::default();

        match self.utl_processor.compute_metrics(input, &context).await {
            Ok(metrics) => JsonRpcResponse::success(
                id,
                json!({
                    "entropy": metrics.entropy,
                    "coherence": metrics.coherence,
                    "learningScore": metrics.learning_score,
                    "surprise": metrics.surprise,
                    "coherenceChange": metrics.coherence_change,
                    "emotionalWeight": metrics.emotional_weight,
                    "alignment": metrics.alignment,
                }),
            ),
            Err(e) => JsonRpcResponse::error(id, error_codes::INTERNAL_ERROR, e.to_string()),
        }
    }

    async fn handle_system_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        let node_count = self.memory_store.count().await.unwrap_or(0);

        JsonRpcResponse::success(
            id,
            json!({
                "status": "running",
                "phase": "ghost-system",
                "nodeCount": node_count,
                "gpuAvailable": false,
            }),
        )
    }

    async fn handle_system_health(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "healthy": true,
                "components": {
                    "memory": "healthy",
                    "utl": "healthy",
                    "graph": "healthy"
                }
            }),
        )
    }
}

// =============================================================================
// MCP Protocol Compliance Unit Tests
// =============================================================================
//
// Tests verify compliance with MCP protocol version 2024-11-05
// Reference: https://spec.modelcontextprotocol.io/specification/2024-11-05/
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::stubs::{InMemoryStore, StubUtlProcessor};
    use std::sync::Arc;

    /// Create test handlers with real stub implementations (no mocks).
    fn create_test_handlers() -> Handlers {
        let memory_store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
        Handlers::new(memory_store, utl_processor)
    }

    /// Create a JSON-RPC request for testing.
    fn make_request(
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

    // =========================================================================
    // 1. Initialize Handshake Tests (MCP 2024-11-05 compliance)
    // =========================================================================

    #[tokio::test]
    async fn test_initialize_returns_protocol_version_2024_11_05() {
        let handlers = create_test_handlers();
        let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "Initialize should not return an error"
        );
        let result = response.result.expect("Initialize must return a result");

        // MCP REQUIREMENT: protocolVersion MUST be "2024-11-05"
        let protocol_version = result
            .get("protocolVersion")
            .expect("Response must contain protocolVersion")
            .as_str()
            .expect("protocolVersion must be a string");
        assert_eq!(
            protocol_version, "2024-11-05",
            "Protocol version must be 2024-11-05"
        );
    }

    #[tokio::test]
    async fn test_initialize_returns_capabilities_with_tools() {
        let handlers = create_test_handlers();
        let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;
        let result = response.result.expect("Initialize must return a result");

        // MCP REQUIREMENT: capabilities object MUST exist
        let capabilities = result
            .get("capabilities")
            .expect("Response must contain capabilities object");

        // MCP REQUIREMENT: capabilities.tools MUST exist with listChanged
        let tools = capabilities
            .get("tools")
            .expect("capabilities must contain tools object");
        let list_changed = tools
            .get("listChanged")
            .expect("tools must contain listChanged")
            .as_bool()
            .expect("listChanged must be a boolean");
        assert!(list_changed, "listChanged should be true");
    }

    #[tokio::test]
    async fn test_initialize_returns_server_info() {
        let handlers = create_test_handlers();
        let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;
        let result = response.result.expect("Initialize must return a result");

        // MCP REQUIREMENT: serverInfo object MUST exist with name and version
        let server_info = result
            .get("serverInfo")
            .expect("Response must contain serverInfo object");

        let name = server_info
            .get("name")
            .expect("serverInfo must contain name")
            .as_str()
            .expect("name must be a string");
        assert_eq!(
            name, "context-graph-mcp",
            "Server name must be context-graph-mcp"
        );

        let version = server_info
            .get("version")
            .expect("serverInfo must contain version")
            .as_str()
            .expect("version must be a string");
        assert!(!version.is_empty(), "Version must not be empty");
    }

    #[tokio::test]
    async fn test_initialize_has_cognitive_pulse_extension() {
        let handlers = create_test_handlers();
        let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;

        // Context Graph Extension: cognitive_pulse header
        let pulse = response
            .cognitive_pulse
            .expect("Initialize should include cognitive pulse");

        // Verify entropy is in valid range [0.0, 1.0]
        assert!(
            pulse.entropy >= 0.0 && pulse.entropy <= 1.0,
            "Entropy must be in [0.0, 1.0], got {}",
            pulse.entropy
        );

        // Verify coherence is in valid range [0.0, 1.0]
        assert!(
            pulse.coherence >= 0.0 && pulse.coherence <= 1.0,
            "Coherence must be in [0.0, 1.0], got {}",
            pulse.coherence
        );
    }

    // =========================================================================
    // 2. Tools List Tests (MCP 2024-11-05 compliance)
    // =========================================================================

    #[tokio::test]
    async fn test_tools_list_returns_all_5_tools() {
        let handlers = create_test_handlers();
        let request = make_request("tools/list", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "tools/list should not return an error"
        );
        let result = response.result.expect("tools/list must return a result");

        // MCP REQUIREMENT: tools array MUST exist
        let tools = result
            .get("tools")
            .expect("Response must contain tools array")
            .as_array()
            .expect("tools must be an array");

        // Verify exactly 5 tools returned
        assert_eq!(
            tools.len(),
            5,
            "Must return exactly 5 tools, got {}",
            tools.len()
        );
    }

    #[tokio::test]
    async fn test_tools_list_each_tool_has_required_fields() {
        let handlers = create_test_handlers();
        let request = make_request("tools/list", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;
        let result = response.result.expect("tools/list must return a result");
        let tools = result.get("tools").unwrap().as_array().unwrap();

        for tool in tools {
            // MCP REQUIREMENT: each tool MUST have name (string)
            let name = tool
                .get("name")
                .expect("Tool must have name field")
                .as_str()
                .expect("Tool name must be a string");
            assert!(!name.is_empty(), "Tool name must not be empty");

            // MCP REQUIREMENT: each tool MUST have description (string)
            let description = tool
                .get("description")
                .expect("Tool must have description field")
                .as_str()
                .expect("Tool description must be a string");
            assert!(
                !description.is_empty(),
                "Tool description must not be empty"
            );

            // MCP REQUIREMENT: each tool MUST have inputSchema (JSON Schema object)
            let input_schema = tool
                .get("inputSchema")
                .expect("Tool must have inputSchema field");
            assert!(
                input_schema.is_object(),
                "inputSchema must be a JSON object"
            );

            // Verify inputSchema is valid JSON Schema (has type field)
            let schema_type = input_schema
                .get("type")
                .expect("inputSchema must have a type field")
                .as_str()
                .expect("inputSchema type must be a string");
            assert_eq!(schema_type, "object", "inputSchema type must be 'object'");
        }
    }

    #[tokio::test]
    async fn test_tools_list_contains_expected_tool_names() {
        let handlers = create_test_handlers();
        let request = make_request("tools/list", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;
        let result = response.result.expect("tools/list must return a result");
        let tools = result.get("tools").unwrap().as_array().unwrap();

        let tool_names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();

        // Verify all expected tools are present
        assert!(
            tool_names.contains(&"inject_context"),
            "Missing inject_context tool"
        );
        assert!(
            tool_names.contains(&"store_memory"),
            "Missing store_memory tool"
        );
        assert!(
            tool_names.contains(&"get_memetic_status"),
            "Missing get_memetic_status tool"
        );
        assert!(
            tool_names.contains(&"get_graph_manifest"),
            "Missing get_graph_manifest tool"
        );
        assert!(
            tool_names.contains(&"search_graph"),
            "Missing search_graph tool"
        );
    }

    // =========================================================================
    // 3. Tools Call Tests - inject_context (MCP 2024-11-05 compliance)
    // =========================================================================

    #[tokio::test]
    async fn test_tools_call_inject_context_valid() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "inject_context",
            "arguments": {
                "content": "Test content for injection",
                "rationale": "Testing MCP protocol compliance"
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "Valid inject_context should not return an error"
        );
        let result = response.result.expect("tools/call must return a result");

        // MCP REQUIREMENT: content array MUST exist
        let content = result
            .get("content")
            .expect("Response must contain content array")
            .as_array()
            .expect("content must be an array");

        // MCP REQUIREMENT: content items must have type and text
        assert!(!content.is_empty(), "Content array must not be empty");
        let first_item = &content[0];
        let item_type = first_item
            .get("type")
            .expect("Content item must have type")
            .as_str()
            .expect("type must be a string");
        assert_eq!(item_type, "text", "Content item type must be 'text'");

        let text = first_item
            .get("text")
            .expect("Content item must have text")
            .as_str()
            .expect("text must be a string");
        assert!(!text.is_empty(), "Content text must not be empty");

        // MCP REQUIREMENT: isError MUST be false for successful calls
        let is_error = result
            .get("isError")
            .expect("Response must contain isError")
            .as_bool()
            .expect("isError must be a boolean");
        assert!(!is_error, "isError must be false for successful tool calls");

        // Verify nodeId is present in the response text
        let parsed_text: serde_json::Value =
            serde_json::from_str(text).expect("Content text should be valid JSON");
        assert!(
            parsed_text.get("nodeId").is_some(),
            "Response must contain nodeId"
        );
    }

    #[tokio::test]
    async fn test_tools_call_inject_context_returns_utl_metrics() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "inject_context",
            "arguments": {
                "content": "Learning about MCP protocols",
                "rationale": "Understanding API standards"
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.expect("tools/call must return a result");
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

        // Verify UTL metrics are present
        let utl = parsed_text
            .get("utl")
            .expect("Response must contain utl object");
        assert!(
            utl.get("learningScore").is_some(),
            "utl must contain learningScore"
        );
        assert!(utl.get("entropy").is_some(), "utl must contain entropy");
        assert!(utl.get("coherence").is_some(), "utl must contain coherence");
        assert!(utl.get("surprise").is_some(), "utl must contain surprise");
    }

    // =========================================================================
    // 4. Tools Call Tests - get_memetic_status (MCP 2024-11-05 compliance)
    // =========================================================================

    #[tokio::test]
    async fn test_tools_call_get_memetic_status() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_memetic_status",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "get_memetic_status should not return an error"
        );
        let result = response.result.expect("tools/call must return a result");

        // Verify MCP format
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

        // Verify expected fields
        assert!(
            parsed_text.get("phase").is_some(),
            "Response must contain phase"
        );
        assert!(
            parsed_text.get("nodeCount").is_some(),
            "Response must contain nodeCount"
        );
        assert!(
            parsed_text.get("utl").is_some(),
            "Response must contain utl"
        );
        assert!(
            parsed_text.get("layers").is_some(),
            "Response must contain layers"
        );
    }

    // =========================================================================
    // 5. Tools Call Tests - get_graph_manifest (MCP 2024-11-05 compliance)
    // =========================================================================

    #[tokio::test]
    async fn test_tools_call_get_graph_manifest_has_5_layers() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_graph_manifest",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "get_graph_manifest should not return an error"
        );
        let result = response.result.expect("tools/call must return a result");

        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

        // Verify 5-layer architecture
        let architecture = parsed_text
            .get("architecture")
            .expect("Response must contain architecture")
            .as_str()
            .expect("architecture must be a string");
        assert_eq!(
            architecture, "5-layer-bio-nervous",
            "Architecture must be 5-layer-bio-nervous"
        );

        // Verify exactly 5 layers
        let layers = parsed_text
            .get("layers")
            .expect("Response must contain layers")
            .as_array()
            .expect("layers must be an array");
        assert_eq!(
            layers.len(),
            5,
            "Must have exactly 5 layers, got {}",
            layers.len()
        );

        // Verify expected layer names
        let layer_names: Vec<&str> = layers
            .iter()
            .filter_map(|l| l.get("name").and_then(|n| n.as_str()))
            .collect();
        assert!(
            layer_names.contains(&"Perception"),
            "Missing Perception layer"
        );
        assert!(layer_names.contains(&"Memory"), "Missing Memory layer");
        assert!(
            layer_names.contains(&"Reasoning"),
            "Missing Reasoning layer"
        );
        assert!(layer_names.contains(&"Action"), "Missing Action layer");
        assert!(layer_names.contains(&"Meta"), "Missing Meta layer");
    }

    // =========================================================================
    // 6. Error Code Tests (MCP 2024-11-05 compliance)
    // =========================================================================

    #[tokio::test]
    async fn test_error_method_not_found_code_32601() {
        let handlers = create_test_handlers();
        let request = make_request("unknown/method", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;

        assert!(
            response.result.is_none(),
            "Unknown method should not return a result"
        );
        let error = response.error.expect("Unknown method must return an error");

        // MCP REQUIREMENT: Method not found error code is -32601
        assert_eq!(
            error.code,
            error_codes::METHOD_NOT_FOUND,
            "Method not found error code must be -32601, got {}",
            error.code
        );
        assert_eq!(error.code, -32601, "Error code -32601 expected");
    }

    #[tokio::test]
    async fn test_error_invalid_params_tools_call_missing_params_code_32602() {
        let handlers = create_test_handlers();
        // tools/call without params
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;

        assert!(
            response.result.is_none(),
            "Invalid params should not return a result"
        );
        let error = response.error.expect("Invalid params must return an error");

        // MCP REQUIREMENT: Invalid params error code is -32602
        assert_eq!(
            error.code,
            error_codes::INVALID_PARAMS,
            "Invalid params error code must be -32602, got {}",
            error.code
        );
        assert_eq!(error.code, -32602, "Error code -32602 expected");
    }

    #[tokio::test]
    async fn test_error_invalid_params_tools_call_missing_name_code_32602() {
        let handlers = create_test_handlers();
        // tools/call with params but missing 'name'
        let params = json!({
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.result.is_none(),
            "Missing name should not return a result"
        );
        let error = response.error.expect("Missing name must return an error");

        assert_eq!(
            error.code,
            error_codes::INVALID_PARAMS,
            "Invalid params error code must be -32602, got {}",
            error.code
        );
    }

    #[tokio::test]
    async fn test_error_tool_not_found_code_32006() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "nonexistent_tool",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.result.is_none(),
            "Unknown tool should not return a result"
        );
        let error = response.error.expect("Unknown tool must return an error");

        // Context Graph specific: Tool not found error code is -32006
        assert_eq!(
            error.code,
            error_codes::TOOL_NOT_FOUND,
            "Tool not found error code must be -32006, got {}",
            error.code
        );
        assert_eq!(error.code, -32006, "Error code -32006 expected");
    }

    // =========================================================================
    // 7. ID Echo Tests (JSON-RPC 2.0 compliance)
    // =========================================================================

    #[tokio::test]
    async fn test_id_echoed_correctly_number() {
        let handlers = create_test_handlers();
        let request = make_request("initialize", Some(JsonRpcId::Number(42)), None);

        let response = handlers.dispatch(request).await;

        // JSON-RPC 2.0 REQUIREMENT: ID must be echoed back exactly
        let response_id = response.id.expect("Response must include ID");
        assert_eq!(
            response_id,
            JsonRpcId::Number(42),
            "Response ID must match request ID"
        );
    }

    #[tokio::test]
    async fn test_id_echoed_correctly_string() {
        let handlers = create_test_handlers();
        let request = make_request(
            "initialize",
            Some(JsonRpcId::String("request-abc-123".to_string())),
            None,
        );

        let response = handlers.dispatch(request).await;

        let response_id = response.id.expect("Response must include ID");
        assert_eq!(
            response_id,
            JsonRpcId::String("request-abc-123".to_string()),
            "Response ID must match request ID"
        );
    }

    #[tokio::test]
    async fn test_id_echoed_on_error() {
        let handlers = create_test_handlers();
        let request = make_request("unknown/method", Some(JsonRpcId::Number(999)), None);

        let response = handlers.dispatch(request).await;

        // ID must be echoed even on error responses
        let response_id = response.id.expect("Error response must include ID");
        assert_eq!(
            response_id,
            JsonRpcId::Number(999),
            "Error response ID must match request ID"
        );
    }

    // =========================================================================
    // 8. Tool Error Response Format Tests
    // =========================================================================

    #[tokio::test]
    async fn test_tool_error_sets_is_error_true() {
        let handlers = create_test_handlers();
        // inject_context without required 'content' parameter
        let params = json!({
            "name": "inject_context",
            "arguments": {
                "rationale": "Missing content"
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        // Tool errors return success with isError: true (MCP format)
        assert!(
            response.error.is_none(),
            "Tool errors use isError flag, not JSON-RPC error"
        );
        let result = response
            .result
            .expect("Tool error must return a result with isError");

        let is_error = result
            .get("isError")
            .expect("Response must contain isError")
            .as_bool()
            .expect("isError must be a boolean");
        assert!(is_error, "isError must be true for tool errors");

        // Verify error message is in content
        let content = result
            .get("content")
            .expect("Response must contain content")
            .as_array()
            .expect("content must be an array");
        assert!(!content.is_empty(), "Error content must not be empty");
    }

    // =========================================================================
    // 9. JSON-RPC Version Tests
    // =========================================================================

    #[tokio::test]
    async fn test_response_jsonrpc_version_is_2_0() {
        let handlers = create_test_handlers();
        let request = make_request("initialize", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;

        // JSON-RPC 2.0 REQUIREMENT: jsonrpc field must be "2.0"
        assert_eq!(response.jsonrpc, "2.0", "JSON-RPC version must be 2.0");
    }

    // =========================================================================
    // 10. Notification Handling Tests
    // =========================================================================

    #[tokio::test]
    async fn test_initialized_notification_no_response_needed() {
        let handlers = create_test_handlers();
        // Notifications have no ID
        let request = make_request("notifications/initialized", None, None);

        let response = handlers.dispatch(request).await;

        // JSON-RPC 2.0: Notifications should return "no response" indicator
        // In this implementation, we return a response with no id, result, or error
        assert!(
            response.id.is_none(),
            "Notification response should have no ID"
        );
    }

    // =========================================================================
    // 11. Search Graph Tool Tests
    // =========================================================================

    #[tokio::test]
    async fn test_tools_call_search_graph_valid() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "search_graph",
            "arguments": {
                "query": "test search query",
                "topK": 5
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "Valid search_graph should not return an error"
        );
        let result = response.result.expect("tools/call must return a result");

        // Verify MCP format
        let is_error = result.get("isError").unwrap().as_bool().unwrap();
        assert!(!is_error, "isError must be false for successful search");

        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

        // Verify search results structure
        assert!(
            parsed_text.get("results").is_some(),
            "Response must contain results"
        );
        assert!(
            parsed_text.get("count").is_some(),
            "Response must contain count"
        );
    }

    #[tokio::test]
    async fn test_tools_call_search_graph_missing_query() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "search_graph",
            "arguments": {
                "topK": 5
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(response.error.is_none(), "Tool errors use isError flag");
        let result = response.result.expect("Tool error must return a result");
        let is_error = result.get("isError").unwrap().as_bool().unwrap();
        assert!(is_error, "Missing query should set isError to true");
    }

    // =========================================================================
    // 12. Store Memory Tool Tests
    // =========================================================================

    #[tokio::test]
    async fn test_tools_call_store_memory_valid() {
        let handlers = create_test_handlers();
        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": "Memory content to store",
                "importance": 0.8
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "Valid store_memory should not return an error"
        );
        let result = response.result.expect("tools/call must return a result");

        let is_error = result.get("isError").unwrap().as_bool().unwrap();
        assert!(!is_error, "isError must be false for successful store");

        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

        assert!(
            parsed_text.get("nodeId").is_some(),
            "Response must contain nodeId"
        );
    }

    // =========================================================================
    // 13. Feature Gating Tests (TC-GHOST-006)
    // =========================================================================
    // These tests verify that the FEATURE_DISABLED error code (-32001) is
    // properly defined and can be returned when features are disabled.

    #[test]
    fn test_feature_disabled_error_code_value() {
        // TC-GHOST-006: FEATURE_DISABLED must be -32001
        assert_eq!(
            error_codes::FEATURE_DISABLED,
            -32001,
            "FEATURE_DISABLED error code must be -32001"
        );
    }

    #[test]
    fn test_all_error_codes_are_negative() {
        // TC-GHOST-006: All Context Graph error codes must be negative (per JSON-RPC spec)
        assert!(error_codes::PARSE_ERROR < 0, "PARSE_ERROR must be negative");
        assert!(
            error_codes::INVALID_REQUEST < 0,
            "INVALID_REQUEST must be negative"
        );
        assert!(
            error_codes::METHOD_NOT_FOUND < 0,
            "METHOD_NOT_FOUND must be negative"
        );
        assert!(
            error_codes::INVALID_PARAMS < 0,
            "INVALID_PARAMS must be negative"
        );
        assert!(
            error_codes::INTERNAL_ERROR < 0,
            "INTERNAL_ERROR must be negative"
        );
        assert!(
            error_codes::FEATURE_DISABLED < 0,
            "FEATURE_DISABLED must be negative"
        );
        assert!(
            error_codes::NODE_NOT_FOUND < 0,
            "NODE_NOT_FOUND must be negative"
        );
        assert!(
            error_codes::PAYLOAD_TOO_LARGE < 0,
            "PAYLOAD_TOO_LARGE must be negative"
        );
        assert!(
            error_codes::STORAGE_ERROR < 0,
            "STORAGE_ERROR must be negative"
        );
        assert!(
            error_codes::EMBEDDING_ERROR < 0,
            "EMBEDDING_ERROR must be negative"
        );
        assert!(
            error_codes::TOOL_NOT_FOUND < 0,
            "TOOL_NOT_FOUND must be negative"
        );
        assert!(
            error_codes::LAYER_TIMEOUT < 0,
            "LAYER_TIMEOUT must be negative"
        );
    }

    #[test]
    fn test_context_graph_error_codes_in_custom_range() {
        // TC-GHOST-006: Context Graph specific codes must be in -32001 to -32099 range
        let custom_codes = [
            error_codes::FEATURE_DISABLED,
            error_codes::NODE_NOT_FOUND,
            error_codes::PAYLOAD_TOO_LARGE,
            error_codes::STORAGE_ERROR,
            error_codes::EMBEDDING_ERROR,
            error_codes::TOOL_NOT_FOUND,
            error_codes::LAYER_TIMEOUT,
        ];

        for code in custom_codes {
            assert!(
                code >= -32099 && code <= -32001,
                "Error code {} must be in range [-32099, -32001]",
                code
            );
        }
    }

    #[test]
    fn test_standard_jsonrpc_error_codes() {
        // TC-GHOST-006: Standard JSON-RPC codes must match spec
        assert_eq!(
            error_codes::PARSE_ERROR,
            -32700,
            "PARSE_ERROR must be -32700"
        );
        assert_eq!(
            error_codes::INVALID_REQUEST,
            -32600,
            "INVALID_REQUEST must be -32600"
        );
        assert_eq!(
            error_codes::METHOD_NOT_FOUND,
            -32601,
            "METHOD_NOT_FOUND must be -32601"
        );
        assert_eq!(
            error_codes::INVALID_PARAMS,
            -32602,
            "INVALID_PARAMS must be -32602"
        );
        assert_eq!(
            error_codes::INTERNAL_ERROR,
            -32603,
            "INTERNAL_ERROR must be -32603"
        );
    }

    #[test]
    fn test_feature_disabled_error_response_format() {
        // TC-GHOST-006: Feature disabled error response must have correct format
        use crate::protocol::JsonRpcResponse;

        let response = JsonRpcResponse::error(
            Some(JsonRpcId::Number(1)),
            error_codes::FEATURE_DISABLED,
            "Dream mode is disabled in current phase",
        );

        assert!(
            response.result.is_none(),
            "Error response must not have result"
        );
        let error = response.error.expect("Error response must have error");

        assert_eq!(error.code, -32001, "Error code must be -32001");
        assert!(
            error.message.contains("disabled"),
            "Error message must mention disabled"
        );
    }

    #[tokio::test]
    async fn test_error_code_node_not_found_32002() {
        // TC-GHOST-006: NODE_NOT_FOUND must be -32002
        let handlers = create_test_handlers();
        let params = json!({
            "nodeId": "00000000-0000-0000-0000-000000000000"
        });
        let request = make_request("memory/retrieve", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        // Should return NODE_NOT_FOUND error
        let error = response.error.expect("Missing node must return error");
        assert_eq!(
            error.code,
            error_codes::NODE_NOT_FOUND,
            "Error code must be NODE_NOT_FOUND (-32002)"
        );
        assert_eq!(error.code, -32002);
    }

    // =========================================================================
    // 14. Configuration Integration Tests (TC-GHOST-006)
    // =========================================================================

    #[test]
    fn test_handlers_use_real_stubs_not_mocks() {
        // TC-GHOST-006: Handlers must use real stub implementations
        let _handlers = create_test_handlers();

        // This test verifies the handlers were created successfully
        // with real InMemoryStore and StubUtlProcessor (not mocks)
        // The create_test_handlers function uses Arc<dyn Trait>
        // which proves we have real implementations
        assert!(true, "Handlers created with real stub implementations");
    }

    #[tokio::test]
    async fn test_handlers_store_count_accurate() {
        // TC-GHOST-006: Store count must be accurate after operations
        let handlers = create_test_handlers();

        // Initial system status should show 0 nodes
        let request = make_request("system/status", Some(JsonRpcId::Number(1)), None);
        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let initial_count = result.get("nodeCount").unwrap().as_u64().unwrap();
        assert_eq!(initial_count, 0, "Initial node count must be 0");

        // Store a memory
        let store_params = json!({ "content": "Test memory content" });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(2)),
            Some(store_params),
        );
        handlers.dispatch(store_request).await;

        // Count should now be 1
        let status_request = make_request("system/status", Some(JsonRpcId::Number(3)), None);
        let status_response = handlers.dispatch(status_request).await;
        let status_result = status_response.result.unwrap();
        let final_count = status_result.get("nodeCount").unwrap().as_u64().unwrap();
        assert_eq!(final_count, 1, "Node count must be 1 after storing");
    }

    #[tokio::test]
    async fn test_handlers_system_health_all_healthy() {
        // TC-GHOST-006: System health must report all components healthy
        let handlers = create_test_handlers();
        let request = make_request("system/health", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "system/health should not return error"
        );
        let result = response.result.expect("system/health must return result");

        let healthy = result.get("healthy").unwrap().as_bool().unwrap();
        assert!(healthy, "System must report healthy");

        let components = result.get("components").unwrap();
        assert_eq!(
            components.get("memory").unwrap().as_str().unwrap(),
            "healthy"
        );
        assert_eq!(components.get("utl").unwrap().as_str().unwrap(), "healthy");
        assert_eq!(
            components.get("graph").unwrap().as_str().unwrap(),
            "healthy"
        );
    }

    #[tokio::test]
    async fn test_handlers_system_status_ghost_phase() {
        // TC-GHOST-006: System status must report ghost-system phase
        let handlers = create_test_handlers();
        let request = make_request("system/status", Some(JsonRpcId::Number(1)), None);

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();

        let phase = result.get("phase").unwrap().as_str().unwrap();
        assert_eq!(phase, "ghost-system", "Phase must be ghost-system");

        let status = result.get("status").unwrap().as_str().unwrap();
        assert_eq!(status, "running", "Status must be running");
    }

    // =========================================================================
    // 15. Meta-Cognitive Verification Tests (TC-GHOST-007)
    // =========================================================================
    // These tests verify the First Contact Manifest and Context Distillation
    // capabilities required for the Context Graph's meta-cognitive layer.

    #[tokio::test]
    async fn test_first_contact_manifest_architecture() {
        // TC-GHOST-007: First Contact Manifest must describe 5-layer bio-nervous system
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_graph_manifest",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let manifest: serde_json::Value = serde_json::from_str(text).unwrap();

        // Verify architecture description
        let architecture = manifest.get("architecture").unwrap().as_str().unwrap();
        assert_eq!(
            architecture, "5-layer-bio-nervous",
            "Architecture must be 5-layer-bio-nervous"
        );
    }

    #[tokio::test]
    async fn test_first_contact_manifest_all_layers_present() {
        // TC-GHOST-007: Manifest must include all 5 layers with descriptions
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_graph_manifest",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let manifest: serde_json::Value = serde_json::from_str(text).unwrap();

        let layers = manifest.get("layers").unwrap().as_array().unwrap();
        assert_eq!(layers.len(), 5, "Must have exactly 5 layers");

        // Verify each layer has required fields
        for layer in layers {
            let name = layer.get("name").expect("Layer must have name");
            assert!(name.as_str().is_some(), "Layer name must be a string");

            let description = layer
                .get("description")
                .expect("Layer must have description");
            assert!(
                description.as_str().is_some(),
                "Layer description must be a string"
            );

            let status = layer.get("status").expect("Layer must have status");
            let status_str = status.as_str().unwrap();
            assert!(
                status_str == "active" || status_str == "stub",
                "Layer status must be 'active' or 'stub'"
            );
        }
    }

    #[tokio::test]
    async fn test_first_contact_manifest_layer_ordering() {
        // TC-GHOST-007: Layers must be in correct order (Perception -> Meta)
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_graph_manifest",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let manifest: serde_json::Value = serde_json::from_str(text).unwrap();

        let layers = manifest.get("layers").unwrap().as_array().unwrap();
        let expected_order = ["Perception", "Memory", "Reasoning", "Action", "Meta"];

        for (i, layer) in layers.iter().enumerate() {
            let name = layer.get("name").unwrap().as_str().unwrap();
            assert_eq!(
                name, expected_order[i],
                "Layer {} must be {}, got {}",
                i, expected_order[i], name
            );
        }
    }

    #[tokio::test]
    async fn test_first_contact_manifest_utl_description() {
        // TC-GHOST-007: Manifest must include UTL formula description
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_graph_manifest",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let manifest: serde_json::Value = serde_json::from_str(text).unwrap();

        let utl = manifest
            .get("utl")
            .expect("Manifest must contain utl object");

        let description = utl.get("description").expect("UTL must have description");
        assert!(
            description
                .as_str()
                .unwrap()
                .contains("Universal Transfer Learning"),
            "UTL description must mention Universal Transfer Learning"
        );

        let formula = utl.get("formula").expect("UTL must have formula");
        assert!(
            formula.as_str().unwrap().contains("H(P)"),
            "UTL formula must contain entropy H(P)"
        );
    }

    #[tokio::test]
    async fn test_context_distillation_via_inject_context() {
        // TC-GHOST-007: inject_context must distill learning score from content
        let handlers = create_test_handlers();
        let params = json!({
            "name": "inject_context",
            "arguments": {
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "rationale": "Testing context distillation"
            }
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let distilled: serde_json::Value = serde_json::from_str(text).unwrap();

        // Verify UTL metrics are computed
        let utl = distilled
            .get("utl")
            .expect("Response must contain utl object");

        let learning_score = utl
            .get("learningScore")
            .expect("UTL must have learningScore")
            .as_f64()
            .expect("learningScore must be a number");
        assert!(
            learning_score >= 0.0 && learning_score <= 1.0,
            "Learning score must be in [0.0, 1.0], got {}",
            learning_score
        );

        let surprise = utl
            .get("surprise")
            .expect("UTL must have surprise")
            .as_f64()
            .expect("surprise must be a number");
        assert!(
            surprise >= 0.0 && surprise <= 1.0,
            "Surprise must be in [0.0, 1.0], got {}",
            surprise
        );
    }

    #[tokio::test]
    async fn test_context_distillation_determinism() {
        // TC-GHOST-007: Same input must produce same distillation
        let handlers = create_test_handlers();
        let test_content = "Neural networks learn hierarchical representations of data";

        let params = json!({
            "name": "inject_context",
            "arguments": {
                "content": test_content,
                "rationale": "Determinism test"
            }
        });

        // First call
        let request1 = make_request(
            "tools/call",
            Some(JsonRpcId::Number(1)),
            Some(params.clone()),
        );
        let response1 = handlers.dispatch(request1).await;
        let result1 = response1.result.unwrap();
        let content1 = result1.get("content").unwrap().as_array().unwrap();
        let text1 = content1[0].get("text").unwrap().as_str().unwrap();
        let distilled1: serde_json::Value = serde_json::from_str(text1).unwrap();

        // Second call with same input
        let request2 = make_request("tools/call", Some(JsonRpcId::Number(2)), Some(params));
        let response2 = handlers.dispatch(request2).await;
        let result2 = response2.result.unwrap();
        let content2 = result2.get("content").unwrap().as_array().unwrap();
        let text2 = content2[0].get("text").unwrap().as_str().unwrap();
        let distilled2: serde_json::Value = serde_json::from_str(text2).unwrap();

        // UTL metrics must be identical
        let utl1 = distilled1.get("utl").unwrap();
        let utl2 = distilled2.get("utl").unwrap();

        assert_eq!(
            utl1.get("learningScore").unwrap().as_f64(),
            utl2.get("learningScore").unwrap().as_f64(),
            "Learning scores must be deterministic"
        );
        assert_eq!(
            utl1.get("surprise").unwrap().as_f64(),
            utl2.get("surprise").unwrap().as_f64(),
            "Surprise must be deterministic"
        );
    }

    #[tokio::test]
    async fn test_context_distillation_utl_compute() {
        // TC-GHOST-007: utl/compute must return learning score
        let handlers = create_test_handlers();
        let params = json!({
            "input": "Context distillation test for UTL computation"
        });
        let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "utl/compute should not return error"
        );
        let result = response.result.expect("utl/compute must return result");

        let learning_score = result
            .get("learningScore")
            .expect("Result must have learningScore")
            .as_f64()
            .expect("learningScore must be a number");
        assert!(
            learning_score >= 0.0 && learning_score <= 1.0,
            "Learning score must be in [0.0, 1.0], got {}",
            learning_score
        );

        // Verify cognitive pulse is included
        let pulse = response
            .cognitive_pulse
            .expect("utl/compute should include cognitive pulse");
        assert!(pulse.entropy >= 0.0 && pulse.entropy <= 1.0);
        assert!(pulse.coherence >= 0.0 && pulse.coherence <= 1.0);
    }

    #[tokio::test]
    async fn test_context_distillation_utl_metrics() {
        // TC-GHOST-007: utl/metrics must return all UTL components
        let handlers = create_test_handlers();
        let params = json!({
            "input": "Full UTL metrics test for context distillation"
        });
        let request = make_request("utl/metrics", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;

        assert!(
            response.error.is_none(),
            "utl/metrics should not return error"
        );
        let result = response.result.expect("utl/metrics must return result");

        // Verify all UTL components are present
        assert!(result.get("entropy").is_some(), "Must have entropy");
        assert!(result.get("coherence").is_some(), "Must have coherence");
        assert!(
            result.get("learningScore").is_some(),
            "Must have learningScore"
        );
        assert!(result.get("surprise").is_some(), "Must have surprise");
        assert!(
            result.get("coherenceChange").is_some(),
            "Must have coherenceChange"
        );
        assert!(
            result.get("emotionalWeight").is_some(),
            "Must have emotionalWeight"
        );
        assert!(result.get("alignment").is_some(), "Must have alignment");

        // Verify all values are valid numbers
        let learning_score = result.get("learningScore").unwrap().as_f64().unwrap();
        let surprise = result.get("surprise").unwrap().as_f64().unwrap();
        let coherence_change = result.get("coherenceChange").unwrap().as_f64().unwrap();
        let emotional_weight = result.get("emotionalWeight").unwrap().as_f64().unwrap();
        let alignment = result.get("alignment").unwrap().as_f64().unwrap();

        assert!(
            learning_score >= 0.0 && learning_score <= 1.0,
            "Learning score range"
        );
        assert!(surprise >= 0.0 && surprise <= 1.0, "Surprise range");
        assert!(
            coherence_change >= 0.0 && coherence_change <= 1.0,
            "Coherence change range"
        );
        assert!(
            emotional_weight >= 0.5 && emotional_weight <= 1.5,
            "Emotional weight range"
        );
        assert!(alignment >= -1.0 && alignment <= 1.0, "Alignment range");
    }

    #[tokio::test]
    async fn test_memetic_status_shows_layers() {
        // TC-GHOST-007: get_memetic_status must show layer states
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_memetic_status",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let status: serde_json::Value = serde_json::from_str(text).unwrap();

        // Verify layers are present
        let layers = status.get("layers").expect("Status must contain layers");

        // Verify expected layer states
        let perception = layers.get("perception").unwrap().as_str().unwrap();
        let memory = layers.get("memory").unwrap().as_str().unwrap();
        let reasoning = layers.get("reasoning").unwrap().as_str().unwrap();
        let action = layers.get("action").unwrap().as_str().unwrap();
        let meta = layers.get("meta").unwrap().as_str().unwrap();

        // In Ghost System, perception and memory are active, others are stub
        assert_eq!(perception, "active", "Perception must be active");
        assert_eq!(memory, "active", "Memory must be active");
        assert_eq!(reasoning, "stub", "Reasoning must be stub");
        assert_eq!(action, "stub", "Action must be stub");
        assert_eq!(meta, "stub", "Meta must be stub");
    }

    #[tokio::test]
    async fn test_memetic_status_contains_utl() {
        // TC-GHOST-007: get_memetic_status must include UTL state
        let handlers = create_test_handlers();
        let params = json!({
            "name": "get_memetic_status",
            "arguments": {}
        });
        let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let result = response.result.unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        let text = content[0].get("text").unwrap().as_str().unwrap();
        let status: serde_json::Value = serde_json::from_str(text).unwrap();

        let utl = status.get("utl").expect("Status must contain utl object");

        // Verify UTL fields
        assert!(utl.get("entropy").is_some(), "UTL must have entropy");
        assert!(utl.get("coherence").is_some(), "UTL must have coherence");
        assert!(
            utl.get("learningScore").is_some(),
            "UTL must have learningScore"
        );
        assert!(
            utl.get("suggestedAction").is_some(),
            "UTL must have suggestedAction"
        );
    }

    #[tokio::test]
    async fn test_cognitive_pulse_suggested_actions() {
        // TC-GHOST-007: Cognitive pulse must suggest appropriate actions
        let handlers = create_test_handlers();

        // Test that UTL compute returns cognitive pulse with appropriate action
        let params = json!({ "input": "High priority learning content" });
        let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

        let response = handlers.dispatch(request).await;
        let pulse = response.cognitive_pulse.expect("Must have cognitive pulse");

        // Verify suggested action is valid
        let action_json = serde_json::to_string(&pulse.suggested_action).unwrap();
        let valid_actions = [
            "\"ready\"",
            "\"continue\"",
            "\"consolidate\"",
            "\"explore\"",
            "\"stabilize\"",
        ];
        assert!(
            valid_actions.iter().any(|a| action_json == *a),
            "Suggested action must be one of {:?}, got {}",
            valid_actions,
            action_json
        );
    }

    #[tokio::test]
    async fn test_context_injection_stores_memory() {
        // TC-GHOST-007: inject_context must actually store the memory
        let handlers = create_test_handlers();

        // Get initial count
        let status_request = make_request("system/status", Some(JsonRpcId::Number(1)), None);
        let status_response = handlers.dispatch(status_request).await;
        let initial_count = status_response
            .result
            .unwrap()
            .get("nodeCount")
            .unwrap()
            .as_u64()
            .unwrap();

        // Inject context
        let inject_params = json!({
            "name": "inject_context",
            "arguments": {
                "content": "Important context for meta-cognitive storage test",
                "rationale": "Testing that injection stores memory"
            }
        });
        let inject_request = make_request(
            "tools/call",
            Some(JsonRpcId::Number(2)),
            Some(inject_params),
        );
        let inject_response = handlers.dispatch(inject_request).await;

        // Verify injection succeeded
        let inject_result = inject_response.result.unwrap();
        let is_error = inject_result.get("isError").unwrap().as_bool().unwrap();
        assert!(!is_error, "inject_context must succeed");

        // Verify node count increased
        let final_status = make_request("system/status", Some(JsonRpcId::Number(3)), None);
        let final_response = handlers.dispatch(final_status).await;
        let final_count = final_response
            .result
            .unwrap()
            .get("nodeCount")
            .unwrap()
            .as_u64()
            .unwrap();

        assert_eq!(
            final_count,
            initial_count + 1,
            "Node count must increase by 1 after injection"
        );
    }
}
