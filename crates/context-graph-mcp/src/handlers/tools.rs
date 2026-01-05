//! MCP tool call handlers.
//!
//! All tool responses include `_cognitive_pulse` with live UTL metrics.
//! This provides real-time cognitive state in every MCP response.
//!
//! # Constitution Reference
//!
//! Johari quadrant to action mapping (constitution.yaml:159-163):
//! - Open: delta_s < 0.5, delta_c > 0.5 -> DirectRecall
//! - Blind: delta_s > 0.5, delta_c < 0.5 -> TriggerDream
//! - Hidden: delta_s < 0.5, delta_c < 0.5 -> GetNeighborhood
//! - Unknown: delta_s > 0.5, delta_c > 0.5 -> EpistemicAction

use serde_json::json;
use tracing::{debug, warn};

use context_graph_core::traits::SearchOptions;
use context_graph_core::types::{MemoryNode, UtlContext};

use crate::middleware::CognitivePulse;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};
use crate::tools::{get_tool_definitions, tool_names};

use super::Handlers;

impl Handlers {
    /// Handle tools/list request.
    ///
    /// Returns all available MCP tools with their schemas.
    pub(super) async fn handle_tools_list(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling tools/list request");

        let tools = get_tool_definitions();
        JsonRpcResponse::success(id, json!({ "tools": tools }))
    }

    /// Handle tools/call request.
    ///
    /// Dispatches to the appropriate tool handler and returns MCP-compliant result.
    pub(super) async fn handle_tools_call(
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
            tool_names::UTL_STATUS => self.call_utl_status(id).await,
            _ => JsonRpcResponse::error(
                id,
                error_codes::TOOL_NOT_FOUND,
                format!("Unknown tool: {}", tool_name),
            ),
        }
    }

    // ========== Tool Call Implementations ==========

    /// MCP-compliant tool result helper WITH CognitivePulse injection.
    ///
    /// Wraps tool output in the required MCP format with live UTL metrics:
    /// ```json
    /// {
    ///   "content": [{"type": "text", "text": "..."}],
    ///   "isError": false,
    ///   "_cognitive_pulse": {
    ///     "entropy": 0.42,
    ///     "coherence": 0.78,
    ///     "learning_score": 0.55,
    ///     "quadrant": "Open",
    ///     "suggested_action": "DirectRecall"
    ///   }
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// CognitivePulse computation targets < 1ms. Warning logged if exceeded.
    ///
    /// # Error Handling
    ///
    /// FAIL FAST: If CognitivePulse computation fails, the ENTIRE tool call
    /// fails with a detailed error. NO fallbacks, NO default values.
    pub(super) fn tool_result_with_pulse(
        &self,
        id: Option<JsonRpcId>,
        data: serde_json::Value,
    ) -> JsonRpcResponse {
        // Compute CognitivePulse - FAIL FAST if unavailable
        let pulse = match CognitivePulse::from_processor(self.utl_processor.as_ref()) {
            Ok(p) => p,
            Err(e) => {
                // FAIL FAST - no fallbacks
                tracing::error!(
                    error = %e,
                    "CognitivePulse computation FAILED - tool call rejected"
                );
                return JsonRpcResponse::success(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("UTL pulse computation failed: {}", e)
                        }],
                        "isError": true
                    }),
                );
            }
        };

        JsonRpcResponse::success(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
                }],
                "isError": false,
                "_cognitive_pulse": pulse
            }),
        )
    }

    /// MCP-compliant tool error helper WITH CognitivePulse injection.
    ///
    /// Even error responses include the cognitive pulse to maintain
    /// consistent system state visibility.
    ///
    /// # Error Handling
    ///
    /// If pulse computation fails during error response, logs warning
    /// but still returns the original error (pulse failure is secondary).
    pub(super) fn tool_error_with_pulse(&self, id: Option<JsonRpcId>, message: &str) -> JsonRpcResponse {
        // Try to compute pulse, but don't fail the error response if it fails
        let pulse_result = CognitivePulse::from_processor(self.utl_processor.as_ref());

        match pulse_result {
            Ok(pulse) => JsonRpcResponse::success(
                id,
                json!({
                    "content": [{
                        "type": "text",
                        "text": message
                    }],
                    "isError": true,
                    "_cognitive_pulse": pulse
                }),
            ),
            Err(e) => {
                warn!(
                    error = %e,
                    original_error = message,
                    "CognitivePulse computation failed for error response"
                );
                // Still return the original error, just without pulse
                JsonRpcResponse::success(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("{} (pulse unavailable: {})", message, e)
                        }],
                        "isError": true
                    }),
                )
            }
        }
    }

    /// inject_context tool implementation.
    ///
    /// Injects context into the memory graph with UTL metrics computation.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_inject_context(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return self.tool_error_with_pulse(id, "Missing 'content' parameter"),
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
            Err(e) => return self.tool_error_with_pulse(id, &format!("UTL processing failed: {}", e)),
        };

        // Create and store the memory node
        let embedding = vec![0.1; 1536]; // Stub embedding
        let mut node = MemoryNode::new(content, embedding);
        node.importance = importance as f32;
        let node_id = node.id;

        if let Err(e) = self.memory_store.store(node).await {
            return self.tool_error_with_pulse(id, &format!("Storage failed: {}", e));
        }

        self.tool_result_with_pulse(
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
    ///
    /// Stores content in the memory graph.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_store_memory(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return self.tool_error_with_pulse(id, "Missing 'content' parameter"),
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
            Ok(_) => self.tool_result_with_pulse(id, json!({ "nodeId": node_id.to_string() })),
            Err(e) => self.tool_error_with_pulse(id, &format!("Storage failed: {}", e)),
        }
    }

    /// get_memetic_status tool implementation.
    ///
    /// Returns comprehensive system status including:
    /// - Node count from memory store
    /// - Live UTL metrics from UtlProcessor (NOT hardcoded)
    /// - 5-layer bio-nervous system status
    /// - `_cognitive_pulse` with live system state
    ///
    /// # Constitution References
    /// - UTL formula: constitution.yaml:152
    /// - Johari quadrant actions: constitution.yaml:159-163
    pub(super) async fn call_get_memetic_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        let node_count = self.memory_store.count().await.unwrap_or(0);

        // Get LIVE UTL status from the processor
        let utl_status = self.utl_processor.get_status();

        // Extract values with explicit defaults on parse failure
        let lifecycle_phase = utl_status
            .get("lifecycle_phase")
            .and_then(|v| v.as_str())
            .unwrap_or("Infancy");

        let entropy = utl_status
            .get("entropy")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.0);

        let coherence = utl_status
            .get("coherence")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.0);

        let learning_score = utl_status
            .get("learning_score")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.0);

        let johari_quadrant = utl_status
            .get("johari_quadrant")
            .and_then(|v| v.as_str())
            .unwrap_or("Hidden");

        let consolidation_phase = utl_status
            .get("consolidation_phase")
            .and_then(|v| v.as_str())
            .unwrap_or("Wake");

        // Map Johari quadrant to suggested action per constitution.yaml:159-163
        let suggested_action = match johari_quadrant {
            "Open" => "direct_recall",
            "Blind" => "trigger_dream",
            "Hidden" => "get_neighborhood",
            "Unknown" => "epistemic_action",
            _ => "continue",
        };

        self.tool_result_with_pulse(
            id,
            json!({
                "phase": lifecycle_phase,
                "nodeCount": node_count,
                "utl": {
                    "entropy": entropy,
                    "coherence": coherence,
                    "learningScore": learning_score,
                    "johariQuadrant": johari_quadrant,
                    "consolidationPhase": consolidation_phase,
                    "suggestedAction": suggested_action
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
    ///
    /// Returns the 5-layer bio-nervous architecture manifest.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_get_graph_manifest(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        self.tool_result_with_pulse(
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
    ///
    /// Searches the memory graph for matching content.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_search_graph(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return self.tool_error_with_pulse(id, "Missing 'query' parameter"),
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

                self.tool_result_with_pulse(
                    id,
                    json!({ "results": results_json, "count": results_json.len() }),
                )
            }
            Err(e) => self.tool_error_with_pulse(id, &format!("Search failed: {}", e)),
        }
    }

    /// utl_status tool implementation.
    ///
    /// Returns current UTL system state including lifecycle phase, entropy,
    /// coherence, learning score, Johari quadrant, and consolidation phase.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_utl_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling utl_status tool call");

        // Get status from UTL processor (returns serde_json::Value)
        let status = self.utl_processor.get_status();

        self.tool_result_with_pulse(id, status)
    }
}
