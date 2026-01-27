//! Causal discovery tool implementations.
//!
//! Provides MCP tools for triggering and monitoring the LLM-based
//! causal discovery agent (Qwen2.5/Hermes 2 Pro).
//!
//! # Tools
//!
//! - `trigger_causal_discovery` - Manually trigger causal analysis on memories
//! - `get_causal_discovery_status` - Get agent status and statistics
//!
//! # Architecture
//!
//! Supports two modes:
//! - `mode: "pairs"` - Pair analysis using GraphDiscoveryService (existing)
//! - `mode: "extract"` - Multi-relationship extraction per memory (NEW)
//!
//! The "extract" mode uses the CausalHintProvider's `extract_all_relationships`
//! method to find ALL cause-effect relationships in individual memories,
//! generating E5 dual embeddings for each.

use serde_json::json;
use tracing::{debug, error, info, warn};

use context_graph_graph_agent::MemoryForGraphAnalysis;

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

impl Handlers {
    /// trigger_causal_discovery tool implementation.
    ///
    /// Manually triggers causal analysis on memories using the LLM.
    /// Discovered relationships are stored with E5 asymmetric embeddings.
    ///
    /// # Parameters
    ///
    /// - `mode`: Analysis mode ("pairs" for pair analysis, "extract" for multi-relationship extraction, default: "extract")
    /// - `maxMemories`: Maximum memories to analyze (1-200, default: 50)
    /// - `minConfidence`: Minimum LLM confidence to accept (0.5-1.0, default: 0.7)
    /// - `sessionScope`: Scope of memories ("current", "all", "recent", default: "all")
    /// - `dryRun`: Analyze but don't store relationships (default: false)
    ///
    /// # Modes
    ///
    /// - `"extract"`: Uses multi-relationship extraction to find ALL cause-effect
    ///   relationships in each memory. Best for dense causal content.
    /// - `"pairs"`: Uses pair analysis via GraphDiscoveryService. Best for finding
    ///   relationships BETWEEN memories.
    pub(crate) async fn call_trigger_causal_discovery(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse mode parameter (default: "extract")
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("extract");

        // Validate mode
        if !matches!(mode, "extract" | "pairs") {
            return self.tool_error(id, "mode must be 'extract' or 'pairs'");
        }

        // Parse parameters with defaults per schema
        let max_memories = args
            .get("maxMemories")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;
        let min_confidence = args
            .get("minConfidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;
        let session_scope = args
            .get("sessionScope")
            .and_then(|v| v.as_str())
            .unwrap_or("all");
        let dry_run = args
            .get("dryRun")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Validate parameters
        if max_memories < 1 || max_memories > 200 {
            return self.tool_error(id, "maxMemories must be between 1 and 200");
        }
        if !(0.5..=1.0).contains(&min_confidence) {
            return self.tool_error(id, "minConfidence must be between 0.5 and 1.0");
        }
        if !matches!(session_scope, "current" | "all" | "recent") {
            return self.tool_error(id, "sessionScope must be 'current', 'all', or 'recent'");
        }

        info!(
            mode = mode,
            max_memories = max_memories,
            min_confidence = min_confidence,
            session_scope = session_scope,
            dry_run = dry_run,
            "trigger_causal_discovery: Starting"
        );

        // Route to appropriate mode handler
        if mode == "extract" {
            return self
                .trigger_causal_discovery_extract(id, max_memories, min_confidence, dry_run)
                .await;
        }

        // Get graph discovery service - GUARANTEED available (NO FALLBACKS)
        let service = self.graph_discovery_service();

        // For now, we'll get recent fingerprints from the store and run discovery
        // In the future, this could be optimized with a dedicated candidate finder

        // Get list of indexed files as a proxy for available content
        let indexed_files = match self.teleological_store.list_indexed_files().await {
            Ok(files) => files,
            Err(e) => {
                warn!(error = %e, "trigger_causal_discovery: Could not list indexed files");
                Vec::new()
            }
        };

        // Collect memory IDs from indexed files (up to max_memories * 2 to have enough pairs)
        let mut memory_ids: Vec<uuid::Uuid> = Vec::new();
        for file in indexed_files.iter().take(max_memories * 2) {
            if let Ok(ids) = self
                .teleological_store
                .get_fingerprints_for_file(&file.file_path)
                .await
            {
                memory_ids.extend(ids);
                if memory_ids.len() >= max_memories * 2 {
                    break;
                }
            }
        }

        debug!(
            memory_ids_count = memory_ids.len(),
            "trigger_causal_discovery: Collected memory IDs"
        );

        // If we don't have enough memories, return early
        if memory_ids.len() < 2 {
            info!("trigger_causal_discovery: Not enough memories for analysis (need at least 2)");
            return self.tool_result(
                id,
                json!({
                    "status": "completed",
                    "pairsAnalyzed": 0,
                    "relationshipsFound": 0,
                    "message": "Not enough memories for causal analysis (need at least 2)",
                    "sessionScope": session_scope,
                    "dryRun": dry_run
                }),
            );
        }

        // Limit to max_memories * 2 memories (to create max_memories pairs)
        memory_ids.truncate(max_memories * 2);

        // Fetch memory data for analysis
        let mut memories_for_analysis: Vec<MemoryForGraphAnalysis> = Vec::new();
        let mut fetch_errors = 0;

        for uuid in &memory_ids {
            // Get fingerprint
            let fingerprint = match self.teleological_store.retrieve(*uuid).await {
                Ok(Some(fp)) => fp,
                Ok(None) => {
                    fetch_errors += 1;
                    continue;
                }
                Err(_) => {
                    fetch_errors += 1;
                    continue;
                }
            };

            // Get content
            let content = match self.teleological_store.get_content(*uuid).await {
                Ok(Some(c)) => c,
                Ok(None) | Err(_) => {
                    fetch_errors += 1;
                    continue;
                }
            };

            // Get source metadata (optional)
            let source_metadata = self
                .teleological_store
                .get_source_metadata(*uuid)
                .await
                .ok()
                .flatten();

            memories_for_analysis.push(MemoryForGraphAnalysis {
                id: *uuid,
                content,
                created_at: fingerprint.created_at,
                session_id: source_metadata.as_ref().and_then(|m| m.session_id.clone()),
                e1_embedding: fingerprint.semantic.e1_semantic.to_vec(),
                source_type: source_metadata.as_ref().map(|m| format!("{}", m.source_type)),
                file_path: source_metadata.and_then(|m| m.file_path),
            });

            if memories_for_analysis.len() >= max_memories * 2 {
                break;
            }
        }

        if memories_for_analysis.len() < 2 {
            info!(
                fetch_errors = fetch_errors,
                "trigger_causal_discovery: Not enough valid memories after fetch"
            );
            return self.tool_result(
                id,
                json!({
                    "status": "completed",
                    "pairsAnalyzed": 0,
                    "relationshipsFound": 0,
                    "message": format!("Not enough valid memories (got {}, need 2+). Fetch errors: {}", memories_for_analysis.len(), fetch_errors),
                    "sessionScope": session_scope,
                    "dryRun": dry_run
                }),
            );
        }

        info!(
            valid_memories = memories_for_analysis.len(),
            fetch_errors = fetch_errors,
            "trigger_causal_discovery: Running discovery cycle"
        );

        // If dry run, just return what would be analyzed
        if dry_run {
            return self.tool_result(
                id,
                json!({
                    "status": "dry_run",
                    "mode": "pairs",
                    "memoriesAvailable": memories_for_analysis.len(),
                    "potentialPairs": memories_for_analysis.len() * (memories_for_analysis.len() - 1) / 2,
                    "maxPairsToAnalyze": max_memories.min(memories_for_analysis.len() * (memories_for_analysis.len() - 1) / 2),
                    "minConfidence": min_confidence,
                    "sessionScope": session_scope,
                    "dryRun": true,
                    "message": "Dry run completed - no relationships stored"
                }),
            );
        }

        // Run the discovery cycle
        let result = match service.run_discovery_cycle(&memories_for_analysis).await {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "trigger_causal_discovery: Discovery cycle failed");
                return self.tool_error(id, &format!("Discovery failed: {}", e));
            }
        };

        info!(
            candidates_found = result.candidates_found,
            relationships_confirmed = result.relationships_confirmed,
            relationships_rejected = result.relationships_rejected,
            duration_ms = result.duration.as_millis(),
            "trigger_causal_discovery: Complete"
        );

        // Calculate average confidence from activator stats
        let stats = service.activator_stats();
        let average_confidence = if result.relationships_confirmed > 0 {
            // Estimate from confirmed vs rejected ratio
            min_confidence + (1.0 - min_confidence) * 0.5
        } else {
            0.0
        };

        self.tool_result(
            id,
            json!({
                "status": "completed",
                "pairsAnalyzed": result.candidates_found,
                "relationshipsFound": result.relationships_confirmed,
                "relationshipsRejected": result.relationships_rejected,
                "averageConfidence": average_confidence,
                "edgesCreated": stats.edges_created,
                "embeddingsGenerated": stats.embeddings_generated,
                "durationMs": result.duration.as_millis(),
                "errors": result.errors,
                "sessionScope": session_scope,
                "dryRun": false
            }),
        )
    }

    /// get_causal_discovery_status tool implementation.
    ///
    /// Returns status and statistics of the causal discovery agent.
    ///
    /// # Parameters
    ///
    /// - `includeLastResult`: Include detailed results from last cycle (default: true)
    /// - `includeGraphStats`: Include causal graph statistics (default: true)
    pub(crate) async fn call_get_causal_discovery_status(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        let include_last_result = args
            .get("includeLastResult")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let include_graph_stats = args
            .get("includeGraphStats")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        info!(
            include_last_result = include_last_result,
            include_graph_stats = include_graph_stats,
            "get_causal_discovery_status: Querying"
        );

        // Get service status
        let service = self.graph_discovery_service();
        let service_status = service.status();
        let is_running = service.is_running();

        // Get LLM availability from causal hint provider
        let llm_available = self
            .causal_hint_provider
            .as_ref()
            .map(|p| p.is_available())
            .unwrap_or(false);

        // Build base response
        let status_str = match service_status {
            context_graph_graph_agent::ServiceStatus::Stopped => "stopped",
            context_graph_graph_agent::ServiceStatus::Starting => "starting",
            context_graph_graph_agent::ServiceStatus::Running => "running",
            context_graph_graph_agent::ServiceStatus::Stopping => "stopping",
        };

        let mut response = json!({
            "agentStatus": if is_running { "running" } else { status_str },
            "llmAvailable": llm_available,
            "modelName": "Hermes-2-Pro-Mistral-7B",
            "estimatedVramMb": 6000,
            "pairsAnalyzedTotal": service.scanner_analyzed_count()
        });

        // Include last cycle result if requested
        if include_last_result {
            if let Some(last_result) = service.last_result() {
                response["lastCycle"] = json!({
                    "startedAt": last_result.started_at.to_rfc3339(),
                    "completedAt": last_result.completed_at.to_rfc3339(),
                    "durationMs": last_result.duration.as_millis(),
                    "candidatesFound": last_result.candidates_found,
                    "relationshipsConfirmed": last_result.relationships_confirmed,
                    "relationshipsRejected": last_result.relationships_rejected,
                    "embeddingsGenerated": last_result.embeddings_generated,
                    "edgesCreated": last_result.edges_created,
                    "errors": last_result.errors
                });
            } else {
                response["lastCycle"] = json!(null);
            }
        }

        // Include graph stats if requested
        if include_graph_stats {
            let activator_stats = service.activator_stats();
            let graph = service.graph();
            let graph_read = graph.read();
            let edge_count = graph_read.edge_count();

            response["graphStats"] = json!({
                "totalEdges": edge_count,
                "totalEmbeddingsGenerated": activator_stats.embeddings_generated,
                "totalEdgesCreated": activator_stats.edges_created,
                "totalProcessed": activator_stats.processed,
                "skippedLowConfidence": activator_stats.skipped_low_confidence,
                "skippedExisting": activator_stats.skipped_existing,
                "errors": activator_stats.errors
            });
        }

        info!(
            agent_status = status_str,
            llm_available = llm_available,
            "get_causal_discovery_status: Returning status"
        );

        self.tool_result(id, response)
    }

    /// Multi-relationship extraction mode for trigger_causal_discovery.
    ///
    /// Extracts ALL cause-effect relationships from each memory using the
    /// LLM's multi-relationship extraction, generates E5 dual embeddings,
    /// and stores each relationship.
    async fn trigger_causal_discovery_extract(
        &self,
        id: Option<JsonRpcId>,
        max_memories: usize,
        min_confidence: f32,
        dry_run: bool,
    ) -> JsonRpcResponse {
        // Check if causal hint provider is available
        let provider = match &self.causal_hint_provider {
            Some(p) if p.is_available() => p,
            _ => {
                return self.tool_error(
                    id,
                    "Causal hint provider not available - LLM not loaded",
                );
            }
        };

        // Get list of indexed files as source of memories
        let indexed_files = match self.teleological_store.list_indexed_files().await {
            Ok(files) => files,
            Err(e) => {
                warn!(error = %e, "trigger_causal_discovery_extract: Could not list indexed files");
                Vec::new()
            }
        };

        // Collect memory IDs from indexed files
        let mut memory_ids: Vec<uuid::Uuid> = Vec::new();
        for file in indexed_files.iter().take(max_memories * 2) {
            if let Ok(ids) = self
                .teleological_store
                .get_fingerprints_for_file(&file.file_path)
                .await
            {
                memory_ids.extend(ids);
                if memory_ids.len() >= max_memories {
                    break;
                }
            }
        }

        if memory_ids.is_empty() {
            return self.tool_result(
                id,
                json!({
                    "status": "completed",
                    "mode": "extract",
                    "memoriesAnalyzed": 0,
                    "relationshipsFound": 0,
                    "message": "No memories found for analysis",
                    "dryRun": dry_run
                }),
            );
        }

        memory_ids.truncate(max_memories);

        info!(
            memory_count = memory_ids.len(),
            dry_run = dry_run,
            "trigger_causal_discovery_extract: Starting extraction"
        );

        // If dry run, return what would be analyzed
        if dry_run {
            return self.tool_result(
                id,
                json!({
                    "status": "dry_run",
                    "mode": "extract",
                    "memoriesAvailable": memory_ids.len(),
                    "minConfidence": min_confidence,
                    "dryRun": true,
                    "message": "Dry run completed - no relationships extracted"
                }),
            );
        }

        let mut memories_analyzed = 0;
        let mut total_relationships = 0;
        let mut errors = 0;
        let start_time = std::time::Instant::now();

        for memory_id in &memory_ids {
            // Get content for this memory
            let content = match self.teleological_store.get_content(*memory_id).await {
                Ok(Some(c)) => c,
                _ => {
                    errors += 1;
                    continue;
                }
            };

            // Extract all causal relationships from this memory
            let extracted = provider.extract_all_relationships(&content).await;
            memories_analyzed += 1;

            if extracted.is_empty() {
                debug!(
                    memory_id = %memory_id,
                    "trigger_causal_discovery_extract: No relationships found"
                );
                continue;
            }

            // Filter by confidence and store each relationship
            for relationship in extracted {
                if relationship.confidence < min_confidence {
                    continue;
                }

                // Generate E5 dual embeddings for the explanation
                let e5_result = self
                    .multi_array_provider
                    .embed_e5_dual(&relationship.explanation)
                    .await;

                let (e5_cause, e5_effect) = match e5_result {
                    Ok(dual) => dual,
                    Err(e) => {
                        warn!(
                            memory_id = %memory_id,
                            error = %e,
                            "trigger_causal_discovery_extract: Failed to generate E5 embeddings"
                        );
                        errors += 1;
                        continue;
                    }
                };

                // Generate E1 semantic embedding
                let e1_result = self
                    .multi_array_provider
                    .embed_e1_only(&relationship.explanation)
                    .await;

                let e1_semantic = match e1_result {
                    Ok(emb) => emb,
                    Err(e) => {
                        warn!(
                            memory_id = %memory_id,
                            error = %e,
                            "trigger_causal_discovery_extract: Failed to generate E1 embedding"
                        );
                        errors += 1;
                        continue;
                    }
                };

                // Create and store the causal relationship
                let causal_rel = context_graph_core::types::CausalRelationship::new(
                    relationship.cause.clone(),
                    relationship.effect.clone(),
                    relationship.explanation.clone(),
                    e5_cause,
                    e5_effect,
                    e1_semantic,
                    content.clone(),
                    *memory_id,
                    relationship.confidence,
                    relationship.mechanism_type.as_str().to_string(),
                );

                match self
                    .teleological_store
                    .store_causal_relationship(&causal_rel)
                    .await
                {
                    Ok(causal_id) => {
                        debug!(
                            causal_id = %causal_id,
                            source_id = %memory_id,
                            cause = %causal_rel.cause_statement,
                            "trigger_causal_discovery_extract: Stored relationship"
                        );
                        total_relationships += 1;
                    }
                    Err(e) => {
                        warn!(
                            memory_id = %memory_id,
                            error = %e,
                            "trigger_causal_discovery_extract: Failed to store relationship"
                        );
                        errors += 1;
                    }
                }
            }
        }

        let duration = start_time.elapsed();

        info!(
            memories_analyzed = memories_analyzed,
            relationships_found = total_relationships,
            errors = errors,
            duration_ms = duration.as_millis(),
            "trigger_causal_discovery_extract: Complete"
        );

        self.tool_result(
            id,
            json!({
                "status": "completed",
                "mode": "extract",
                "memoriesAnalyzed": memories_analyzed,
                "relationshipsFound": total_relationships,
                "minConfidence": min_confidence,
                "errors": errors,
                "durationMs": duration.as_millis(),
                "dryRun": false
            }),
        )
    }
}
