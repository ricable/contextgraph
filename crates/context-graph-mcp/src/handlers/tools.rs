//! MCP tool call handlers.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore.
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
use sha2::{Digest, Sha256};
use tracing::{debug, error, warn};

use context_graph_core::traits::TeleologicalSearchOptions;
use context_graph_core::types::UtlContext;
use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, TeleologicalFingerprint, NUM_EMBEDDERS,
};

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
                error!(
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
    /// TASK-S001: Updated to use TeleologicalMemoryStore with 13-embedding fingerprint.
    ///
    /// Injects context into the memory graph with UTL metrics computation.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_inject_context(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c.to_string(),
            Some(_) => return self.tool_error_with_pulse(id, "Content cannot be empty"),
            None => return self.tool_error_with_pulse(id, "Missing 'content' parameter"),
        };

        let rationale = args.get("rationale").and_then(|v| v.as_str()).unwrap_or("");
        let _importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        // Compute UTL metrics for the content
        let context = UtlContext::default();
        let metrics = match self.utl_processor.compute_metrics(&content, &context).await {
            Ok(m) => m,
            Err(e) => {
                error!(error = %e, "inject_context: UTL processing FAILED");
                return self.tool_error_with_pulse(id, &format!("UTL processing failed: {}", e));
            }
        };

        // Generate all 13 embeddings using MultiArrayEmbeddingProvider
        let embedding_output = match self.multi_array_provider.embed_all(&content).await {
            Ok(output) => output,
            Err(e) => {
                error!(error = %e, "inject_context: Multi-array embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Embedding failed: {}", e));
            }
        };

        // Compute content hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let content_hash: [u8; 32] = hasher.finalize().into();

        // Create TeleologicalFingerprint with default purpose (will be computed later)
        let fingerprint = TeleologicalFingerprint::new(
            embedding_output.fingerprint,
            PurposeVector::default(),
            JohariFingerprint::zeroed(),
            content_hash,
        );
        let fingerprint_id = fingerprint.id;

        // Store in TeleologicalMemoryStore
        if let Err(e) = self.teleological_store.store(fingerprint).await {
            error!(error = %e, "inject_context: Storage FAILED");
            return self.tool_error_with_pulse(id, &format!("Storage failed: {}", e));
        }

        self.tool_result_with_pulse(
            id,
            json!({
                "fingerprintId": fingerprint_id.to_string(),
                "rationale": rationale,
                "embedderCount": NUM_EMBEDDERS,
                "embeddingLatencyMs": embedding_output.total_latency.as_millis(),
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
    /// TASK-S001: Updated to use TeleologicalMemoryStore with 13-embedding fingerprint.
    ///
    /// Stores content in the memory graph.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_store_memory(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) if !c.is_empty() => c.to_string(),
            Some(_) => return self.tool_error_with_pulse(id, "Content cannot be empty"),
            None => return self.tool_error_with_pulse(id, "Missing 'content' parameter"),
        };

        let _importance = args
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        // Generate all 13 embeddings using MultiArrayEmbeddingProvider
        let embedding_output = match self.multi_array_provider.embed_all(&content).await {
            Ok(output) => output,
            Err(e) => {
                error!(error = %e, "store_memory: Multi-array embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Embedding failed: {}", e));
            }
        };

        // Compute content hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let content_hash: [u8; 32] = hasher.finalize().into();

        // Create TeleologicalFingerprint
        let fingerprint = TeleologicalFingerprint::new(
            embedding_output.fingerprint,
            PurposeVector::default(),
            JohariFingerprint::zeroed(),
            content_hash,
        );
        let fingerprint_id = fingerprint.id;

        match self.teleological_store.store(fingerprint).await {
            Ok(_) => self.tool_result_with_pulse(
                id,
                json!({
                    "fingerprintId": fingerprint_id.to_string(),
                    "embedderCount": NUM_EMBEDDERS,
                    "embeddingLatencyMs": embedding_output.total_latency.as_millis()
                }),
            ),
            Err(e) => {
                error!(error = %e, "store_memory: Storage FAILED");
                self.tool_error_with_pulse(id, &format!("Storage failed: {}", e))
            }
        }
    }

    /// get_memetic_status tool implementation.
    ///
    /// TASK-S001: Updated to use TeleologicalMemoryStore count.
    ///
    /// Returns comprehensive system status including:
    /// - Fingerprint count from TeleologicalMemoryStore
    /// - Live UTL metrics from UtlProcessor (NOT hardcoded)
    /// - 5-layer bio-nervous system status
    /// - `_cognitive_pulse` with live system state
    ///
    /// # Constitution References
    /// - UTL formula: constitution.yaml:152
    /// - Johari quadrant actions: constitution.yaml:159-163
    pub(super) async fn call_get_memetic_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        let fingerprint_count = match self.teleological_store.count().await {
            Ok(count) => count,
            Err(e) => {
                error!(error = %e, "get_memetic_status: TeleologicalStore.count() FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Failed to get fingerprint count: {}", e),
                );
            }
        };

        // Get LIVE UTL status from the processor
        let utl_status = self.utl_processor.get_status();

        // FAIL-FAST: UTL processor MUST return all required fields.
        // Per constitution AP-007: No stubs or fallbacks in production code paths.
        // If the UTL processor doesn't have these fields, the system is broken.
        let lifecycle_phase = match utl_status.get("lifecycle_phase").and_then(|v| v.as_str()) {
            Some(phase) => phase,
            None => {
                error!("get_memetic_status: UTL processor missing 'lifecycle_phase' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'lifecycle_phase'. \
                     This indicates a broken UTL system that must be fixed.".to_string(),
                );
            }
        };

        let entropy = match utl_status.get("entropy").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!("get_memetic_status: UTL processor missing 'entropy' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'entropy'. \
                     This indicates a broken UTL system that must be fixed.".to_string(),
                );
            }
        };

        let coherence = match utl_status.get("coherence").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!("get_memetic_status: UTL processor missing 'coherence' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'coherence'. \
                     This indicates a broken UTL system that must be fixed.".to_string(),
                );
            }
        };

        let learning_score = match utl_status.get("learning_score").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!("get_memetic_status: UTL processor missing 'learning_score' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'learning_score'. \
                     This indicates a broken UTL system that must be fixed.".to_string(),
                );
            }
        };

        let johari_quadrant = match utl_status.get("johari_quadrant").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                error!("get_memetic_status: UTL processor missing 'johari_quadrant' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'johari_quadrant'. \
                     This indicates a broken UTL system that must be fixed.".to_string(),
                );
            }
        };

        let consolidation_phase = match utl_status.get("consolidation_phase").and_then(|v| v.as_str()) {
            Some(phase) => phase,
            None => {
                error!("get_memetic_status: UTL processor missing 'consolidation_phase' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'consolidation_phase'. \
                     This indicates a broken UTL system that must be fixed.".to_string(),
                );
            }
        };

        // Map Johari quadrant to suggested action per constitution.yaml:159-163
        let suggested_action = match johari_quadrant {
            "Open" => "direct_recall",
            "Blind" => "trigger_dream",
            "Hidden" => "get_neighborhood",
            "Unknown" => "epistemic_action",
            _ => "continue",
        };

        // Get quadrant counts from teleological store
        let quadrant_counts = match self.teleological_store.count_by_quadrant().await {
            Ok(counts) => counts,
            Err(e) => {
                error!(error = %e, "get_memetic_status: count_by_quadrant() FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Failed to get quadrant counts: {}", e),
                );
            }
        };

        // TASK-EMB-024: Get REAL layer statuses from LayerStatusProvider
        let perception_status = self.layer_status_provider.perception_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: perception_status FAILED");
                "error".to_string()
            });
        let memory_status = self.layer_status_provider.memory_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: memory_status FAILED");
                "error".to_string()
            });
        let reasoning_status = self.layer_status_provider.reasoning_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: reasoning_status FAILED");
                "error".to_string()
            });
        let action_status = self.layer_status_provider.action_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: action_status FAILED");
                "error".to_string()
            });
        let meta_status = self.layer_status_provider.meta_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: meta_status FAILED");
                "error".to_string()
            });

        self.tool_result_with_pulse(
            id,
            json!({
                "phase": lifecycle_phase,
                "fingerprintCount": fingerprint_count,
                "embedderCount": NUM_EMBEDDERS,
                "storageBackend": self.teleological_store.backend_type().to_string(),
                "storageSizeBytes": self.teleological_store.storage_size_bytes(),
                "quadrantCounts": {
                    "open": quadrant_counts[0],
                    "hidden": quadrant_counts[1],
                    "blind": quadrant_counts[2],
                    "unknown": quadrant_counts[3]
                },
                "utl": {
                    "entropy": entropy,
                    "coherence": coherence,
                    "learningScore": learning_score,
                    "johariQuadrant": johari_quadrant,
                    "consolidationPhase": consolidation_phase,
                    "suggestedAction": suggested_action
                },
                "layers": {
                    "perception": perception_status,
                    "memory": memory_status,
                    "reasoning": reasoning_status,
                    "action": action_status,
                    "meta": meta_status
                }
            }),
        )
    }

    /// get_graph_manifest tool implementation.
    ///
    /// Returns the 5-layer bio-nervous architecture manifest.
    /// Response includes `_cognitive_pulse` with live system state.
    ///
    /// TASK-EMB-024: Layer statuses now come from LayerStatusProvider.
    pub(super) async fn call_get_graph_manifest(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        // TASK-EMB-024: Get REAL layer statuses from LayerStatusProvider
        let perception_status = self.layer_status_provider.perception_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: perception_status FAILED");
                "error".to_string()
            });
        let memory_status = self.layer_status_provider.memory_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: memory_status FAILED");
                "error".to_string()
            });
        let reasoning_status = self.layer_status_provider.reasoning_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: reasoning_status FAILED");
                "error".to_string()
            });
        let action_status = self.layer_status_provider.action_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: action_status FAILED");
                "error".to_string()
            });
        let meta_status = self.layer_status_provider.meta_status().await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_graph_manifest: meta_status FAILED");
                "error".to_string()
            });

        self.tool_result_with_pulse(
            id,
            json!({
                "architecture": "5-layer-bio-nervous",
                "fingerprintType": "TeleologicalFingerprint",
                "embedderCount": NUM_EMBEDDERS,
                "layers": [
                    {
                        "name": "Perception",
                        "description": "Sensory input processing and feature extraction",
                        "status": perception_status
                    },
                    {
                        "name": "Memory",
                        "description": "Teleological memory with 13-embedding semantic fingerprints",
                        "status": memory_status
                    },
                    {
                        "name": "Reasoning",
                        "description": "Inference, planning, and decision making",
                        "status": reasoning_status
                    },
                    {
                        "name": "Action",
                        "description": "Response generation and motor control",
                        "status": action_status
                    },
                    {
                        "name": "Meta",
                        "description": "Self-monitoring, learning rate control, and system optimization",
                        "status": meta_status
                    }
                ],
                "utl": {
                    "description": "Universal Transfer Learning - measures learning potential",
                    "formula": "L(x) = H(P) - H(P|x) + alpha * C(x)"
                },
                "teleological": {
                    "description": "Purpose-aware retrieval with North Star alignment",
                    "purposeVectorDimension": NUM_EMBEDDERS,
                    "johariQuadrants": ["Open", "Hidden", "Blind", "Unknown"]
                }
            }),
        )
    }

    /// search_graph tool implementation.
    ///
    /// TASK-S001: Updated to use TeleologicalMemoryStore search_semantic.
    ///
    /// Searches the memory graph for matching content.
    /// Response includes `_cognitive_pulse` with live system state.
    pub(super) async fn call_search_graph(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            Some(_) => return self.tool_error_with_pulse(id, "Query cannot be empty"),
            None => return self.tool_error_with_pulse(id, "Missing 'query' parameter"),
        };

        let top_k = args.get("topK").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let options = TeleologicalSearchOptions::quick(top_k);

        // Generate query embedding
        let query_embedding = match self.multi_array_provider.embed_all(query).await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "search_graph: Query embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Query embedding failed: {}", e));
            }
        };

        match self.teleological_store.search_semantic(&query_embedding, options).await {
            Ok(results) => {
                let results_json: Vec<_> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "fingerprintId": r.fingerprint.id.to_string(),
                            "similarity": r.similarity,
                            "purposeAlignment": r.purpose_alignment,
                            "dominantEmbedder": r.dominant_embedder(),
                            "thetaToNorthStar": r.fingerprint.theta_to_north_star
                        })
                    })
                    .collect();

                self.tool_result_with_pulse(
                    id,
                    json!({ "results": results_json, "count": results_json.len() }),
                )
            }
            Err(e) => {
                error!(error = %e, "search_graph: Search FAILED");
                self.tool_error_with_pulse(id, &format!("Search failed: {}", e))
            }
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
