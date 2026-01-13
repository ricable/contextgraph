//! Purpose drift check handler.
//!
//! TASK-INTEG-002: Implements the `purpose/drift_check` MCP method for
//! per-embedder drift analysis using TeleologicalDriftDetector.

#![allow(clippy::result_large_err)] // JsonRpcResponse has large error variants by design

use serde_json::json;
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

use context_graph_core::autonomous::drift::{DriftResult, TeleologicalDriftDetector};
use context_graph_core::teleological::{SearchStrategy, TeleologicalComparator};

use crate::handlers::Handlers;
use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::drift_response::{build_drift_response, handle_drift_error};

impl Handlers {
    /// Handle purpose/drift_check request.
    ///
    /// TASK-INTEG-002: Integrates TeleologicalDriftDetector (TASK-LOGIC-010) for
    /// per-embedder drift analysis with 5-level DriftLevel classification.
    ///
    /// # Request Parameters
    /// - `fingerprint_ids` (required): Array of fingerprint UUIDs to check
    /// - `goal_id` (optional): Goal to check drift against (default: North Star)
    /// - `strategy` (optional): Comparison strategy (default: "cosine")
    ///
    /// # Response
    /// - `overall_drift`: Overall drift level, similarity, score, has_drifted flag
    /// - `per_embedder_drift`: Array of 13 embedder-specific drift results
    /// - `most_drifted_embedders`: Top 5 most drifted embedders sorted worst-first
    /// - `recommendations`: Action recommendations based on drift levels
    /// - `analyzed_count`: Number of memories analyzed
    /// - `timestamp`: RFC3339 formatted timestamp
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid fingerprint IDs or parameters
    /// - GOAL_NOT_FOUND (-32020): Specified goal_id not found in hierarchy
    /// - ALIGNMENT_COMPUTATION_ERROR (-32022): Drift check failed
    #[instrument(skip(self, params), fields(method = "purpose/drift_check"))]
    pub(in crate::handlers) async fn handle_purpose_drift_check(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                error!("purpose/drift_check: Missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters - fingerprint_ids required",
                );
            }
        };

        // FAIL FAST: Extract fingerprint_ids (required, cannot be empty)
        let fingerprint_ids = match self.parse_fingerprint_ids(&id, &params) {
            Ok(ids) => ids,
            Err(response) => return response,
        };

        // Parse comparison strategy (default: Cosine)
        let strategy = match self.parse_drift_strategy(&id, &params) {
            Ok(s) => s,
            Err(response) => return response,
        };

        // Get goal fingerprint (North Star by default, or specified goal_id)
        let goal_fingerprint = match self.get_goal_fingerprint(&id, &params) {
            Ok(Some(fp)) => fp,
            Ok(None) => return self.build_no_north_star_response(id),
            Err(response) => return response,
        };

        let check_start = std::time::Instant::now();

        // Collect all fingerprints - FAIL FAST on any error
        let memories = match self.collect_memories(&id, &fingerprint_ids).await {
            Ok(m) => m,
            Err(response) => return response,
        };

        // Create TeleologicalDriftDetector and execute drift check
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);

        let drift_result: DriftResult =
            match detector.check_drift(&memories, &goal_fingerprint, strategy) {
                Ok(result) => result,
                Err(e) => return handle_drift_error(&id, e),
            };

        let check_time_ms = check_start.elapsed().as_millis();

        info!(
            overall_level = ?drift_result.overall_drift.drift_level,
            analyzed_count = drift_result.analyzed_count,
            check_time_ms = check_time_ms,
            "purpose/drift_check: Completed with per-embedder analysis"
        );

        JsonRpcResponse::success(id, build_drift_response(&drift_result, check_time_ms))
    }

    /// Parse fingerprint_ids from params.
    fn parse_fingerprint_ids(
        &self,
        id: &Option<JsonRpcId>,
        params: &serde_json::Value,
    ) -> Result<Vec<Uuid>, JsonRpcResponse> {
        match params.get("fingerprint_ids").and_then(|v| v.as_array()) {
            Some(arr) => {
                if arr.is_empty() {
                    error!("purpose/drift_check: Empty fingerprint_ids array");
                    return Err(JsonRpcResponse::error(
                        id.clone(),
                        error_codes::INVALID_PARAMS,
                        "fingerprint_ids array cannot be empty - FAIL FAST",
                    ));
                }
                let mut ids = Vec::with_capacity(arr.len());
                for (i, v) in arr.iter().enumerate() {
                    match v.as_str().and_then(|s| Uuid::parse_str(s).ok()) {
                        Some(uuid) => ids.push(uuid),
                        None => {
                            error!(index = i, "purpose/drift_check: Invalid UUID format");
                            return Err(JsonRpcResponse::error(
                                id.clone(),
                                error_codes::INVALID_PARAMS,
                                format!("Invalid UUID at fingerprint_ids[{}]", i),
                            ));
                        }
                    }
                }
                Ok(ids)
            }
            None => {
                error!("purpose/drift_check: Missing 'fingerprint_ids' parameter");
                Err(JsonRpcResponse::error(
                    id.clone(),
                    error_codes::INVALID_PARAMS,
                    "Missing required 'fingerprint_ids' parameter (array of UUIDs)",
                ))
            }
        }
    }

    /// Parse drift strategy from params.
    fn parse_drift_strategy(
        &self,
        id: &Option<JsonRpcId>,
        params: &serde_json::Value,
    ) -> Result<SearchStrategy, JsonRpcResponse> {
        match params
            .get("strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("cosine")
        {
            "cosine" => Ok(SearchStrategy::Cosine),
            "euclidean" => Ok(SearchStrategy::Euclidean),
            "synergy" | "synergy_weighted" => Ok(SearchStrategy::SynergyWeighted),
            "group" | "hierarchical" => Ok(SearchStrategy::GroupHierarchical),
            "cross_correlation" => Ok(SearchStrategy::CrossCorrelationDominant),
            other => {
                error!(strategy = other, "purpose/drift_check: Invalid strategy");
                Err(JsonRpcResponse::error(
                    id.clone(),
                    error_codes::INVALID_PARAMS,
                    format!(
                        "Invalid strategy '{}'. Valid: cosine, euclidean, synergy, group, cross_correlation",
                        other
                    ),
                ))
            }
        }
    }

    /// Get goal fingerprint from params or North Star.
    fn get_goal_fingerprint(
        &self,
        id: &Option<JsonRpcId>,
        params: &serde_json::Value,
    ) -> Result<Option<context_graph_core::types::fingerprint::TeleologicalArray>, JsonRpcResponse>
    {
        let hierarchy = self.goal_hierarchy.read();

        if let Some(goal_id_str) = params.get("goal_id").and_then(|v| v.as_str()) {
            let goal_id = match Uuid::parse_str(goal_id_str) {
                Ok(uuid) => uuid,
                Err(_) => {
                    error!(goal_id = goal_id_str, "purpose/drift_check: Invalid goal UUID");
                    return Err(JsonRpcResponse::error(
                        id.clone(),
                        error_codes::INVALID_PARAMS,
                        format!("Invalid goal_id UUID format: {}", goal_id_str),
                    ));
                }
            };
            match hierarchy.get(&goal_id) {
                Some(g) => Ok(Some(g.teleological_array.clone())),
                None => {
                    error!(goal_id = %goal_id, "purpose/drift_check: Goal not found");
                    Err(JsonRpcResponse::error(
                        id.clone(),
                        error_codes::GOAL_NOT_FOUND,
                        format!("Goal {} not found in hierarchy", goal_id),
                    ))
                }
            }
        } else if hierarchy.has_north_star() {
            Ok(Some(
                hierarchy.north_star().unwrap().teleological_array.clone(),
            ))
        } else {
            Ok(None) // No North Star configured
        }
    }

    /// Build response when no North Star is configured.
    fn build_no_north_star_response(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("purpose/drift_check: No North Star configured, returning no-drift response");
        JsonRpcResponse::success(
            id,
            json!({
                "overall_drift": {
                    "level": "None",
                    "similarity": 1.0,
                    "drift_score": 0.0,
                    "has_drifted": false,
                    "message": "No North Star configured - drift measurement not applicable"
                },
                "per_embedder_drift": [],
                "most_drifted_embedders": [],
                "recommendations": [{
                    "action": "store_memories",
                    "priority": "medium",
                    "reason": "Store memories and use auto_bootstrap_north_star to discover emergent purpose patterns"
                }],
                "analyzed_count": 0,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "autonomous_mode": true
            }),
        )
    }

    /// Collect memories from fingerprint IDs.
    async fn collect_memories(
        &self,
        id: &Option<JsonRpcId>,
        fingerprint_ids: &[Uuid],
    ) -> Result<Vec<context_graph_core::types::fingerprint::SemanticFingerprint>, JsonRpcResponse>
    {
        let mut memories = Vec::with_capacity(fingerprint_ids.len());
        for fp_id in fingerprint_ids {
            let fingerprint = match self.teleological_store.retrieve(*fp_id).await {
                Ok(Some(fp)) => fp,
                Ok(None) => {
                    error!(fingerprint_id = %fp_id, "purpose/drift_check: Fingerprint not found");
                    return Err(JsonRpcResponse::error(
                        id.clone(),
                        error_codes::FINGERPRINT_NOT_FOUND,
                        format!("Fingerprint {} not found - FAIL FAST", fp_id),
                    ));
                }
                Err(e) => {
                    error!(fingerprint_id = %fp_id, error = %e, "purpose/drift_check: Storage error");
                    return Err(JsonRpcResponse::error(
                        id.clone(),
                        error_codes::ALIGNMENT_COMPUTATION_ERROR,
                        format!("Storage error retrieving {}: {} - FAIL FAST", fp_id, e),
                    ));
                }
            };
            memories.push(fingerprint.semantic);
        }
        Ok(memories)
    }

}
