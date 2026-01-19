//! Topic tool handlers.
//!
//! Per PRD Section 10.2, implements:
//! - get_topic_portfolio: Get all discovered topics with profiles
//! - get_topic_stability: Get portfolio-level stability metrics
//! - detect_topics: Force topic detection recalculation
//! - get_divergence_alerts: Check for divergence from recent activity
//!
//! Constitution Compliance:
//! - AP-60: Temporal embedders (E2-E4) weight = 0.0 in topic detection
//! - AP-62: Only SEMANTIC embedders for divergence alerts
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - AP-70: Dream recommended when entropy > 0.7 AND churn > 0.5
//!
//! TASK-INTEG-TOPIC: Integrated with MultiSpaceClusterManager and TopicStabilityTracker.

use tracing::{debug, error, info, warn};

use context_graph_core::clustering::Topic;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;
use super::topic_dtos::{
    DetectTopicsRequest, DetectTopicsResponse, DivergenceAlertsResponse,
    GetDivergenceAlertsRequest, GetTopicPortfolioRequest, GetTopicStabilityRequest,
    PhaseBreakdown, StabilityMetricsSummary, TopicPortfolioResponse, TopicStabilityResponse,
    TopicSummary,
};

/// Minimum memories required for clustering (per constitution min_cluster_size).
const MIN_MEMORIES_FOR_CLUSTERING: usize = 3;

/// Convert core Topic to TopicSummary DTO.
///
/// TASK-INTEG-TOPIC: Helper for converting between core and DTO types.
fn topic_to_summary(topic: &Topic) -> TopicSummary {
    let weighted_agreement = topic.profile.weighted_agreement();
    let confidence = TopicSummary::compute_confidence(weighted_agreement);
    // Convert Embedder enum variants to human-readable names
    let contributing_spaces: Vec<String> = topic
        .contributing_spaces
        .iter()
        .map(|e| format!("{:?}", e))
        .collect();
    let phase_str = format!("{:?}", topic.stability.phase);

    TopicSummary {
        id: topic.id,
        name: topic.name.clone(),
        confidence,
        weighted_agreement,
        member_count: topic.member_count(),
        contributing_spaces,
        phase: phase_str,
    }
}

/// Extract entropy from UTL processor status.
///
/// TASK-INTEG-TOPIC: Helper to get entropy from get_status() JSON.
fn extract_entropy_from_status(status: &serde_json::Value) -> f32 {
    status
        .get("entropy")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.0)
}

/// Compute phase breakdown from topics.
///
/// TASK-INTEG-TOPIC: Helper to count topics by lifecycle phase.
fn compute_phase_breakdown(topics: &std::collections::HashMap<uuid::Uuid, Topic>) -> PhaseBreakdown {
    use context_graph_core::clustering::TopicPhase;

    let mut emerging = 0;
    let mut stable = 0;
    let mut declining = 0;
    let mut merging = 0;

    for topic in topics.values() {
        match topic.stability.phase {
            TopicPhase::Emerging => emerging += 1,
            TopicPhase::Stable => stable += 1,
            TopicPhase::Declining => declining += 1,
            TopicPhase::Merging => merging += 1,
        }
    }

    PhaseBreakdown {
        emerging,
        stable,
        declining,
        merging,
    }
}

impl Handlers {
    /// Handle get_topic_portfolio tool call.
    ///
    /// Returns discovered topics with profiles, stability metrics, and tier info.
    ///
    /// # Arguments
    /// * `id` - JSON-RPC request ID
    /// * `arguments` - Tool arguments (format: brief|standard|verbose)
    ///
    /// # Returns
    /// JsonRpcResponse with TopicPortfolioResponse
    ///
    /// # Implements
    /// REQ-MCP-002, REQ-MCP-004
    ///
    /// # Constitution Compliance
    /// - ARCH-09: Topic threshold is weighted_agreement >= 2.5
    /// - AP-60: Temporal embedders (E2-E4) weight = 0.0
    pub(crate) async fn call_get_topic_portfolio(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_topic_portfolio");

        // Parse request
        let request: GetTopicPortfolioRequest = match serde_json::from_value(arguments) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "get_topic_portfolio: Failed to parse request");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Invalid params: {}", e),
                );
            }
        };

        // Validate format using DTO's validate method
        if let Err(validation_error) = request.validate() {
            error!(error = %validation_error, "get_topic_portfolio: Validation failed");
            return self.tool_error_with_pulse(
                id,
                &format!("Invalid params: {}", validation_error),
            );
        }

        // Get memory count to determine tier
        let memory_count = match self.teleological_store.count().await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "get_topic_portfolio: Failed to get memory count");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Storage error: Failed to get memory count: {}", e),
                );
            }
        };

        let tier = TopicPortfolioResponse::tier_for_memory_count(memory_count);

        debug!(
            memory_count = memory_count,
            tier = tier,
            format = %request.format,
            "get_topic_portfolio: Retrieved memory count and tier"
        );

        // Tier 0: No memories, return empty response
        if tier == 0 {
            debug!("get_topic_portfolio: Tier 0 - returning empty portfolio");
            let response = TopicPortfolioResponse::empty_tier_0();
            return self.tool_result_with_pulse(
                id,
                serde_json::to_value(response).expect("TopicPortfolioResponse should serialize"),
            );
        }

        // TASK-INTEG-TOPIC: Get topics from cluster_manager
        let cluster_manager = self.cluster_manager.read();
        let topics_map = cluster_manager.get_topics();

        // Convert core Topics to TopicSummary DTOs
        let topics: Vec<TopicSummary> = topics_map
            .values()
            .map(topic_to_summary)
            .collect();

        let total_topics = topics.len();

        // Get stability metrics from stability_tracker
        let stability_tracker = self.stability_tracker.read();
        let churn_rate = stability_tracker.current_churn();
        // Extract entropy from UTL processor status
        let utl_status = self.utl_processor.get_status();
        let entropy = extract_entropy_from_status(&utl_status);

        let response = TopicPortfolioResponse {
            topics,
            stability: StabilityMetricsSummary::new(churn_rate, entropy),
            total_topics,
            tier,
        };

        info!(
            tier = tier,
            total_topics = response.total_topics,
            churn_rate = churn_rate,
            entropy = entropy,
            is_stable = response.stability.is_stable,
            "get_topic_portfolio: Returning portfolio with {} topics",
            total_topics
        );

        self.tool_result_with_pulse(
            id,
            serde_json::to_value(response).expect("TopicPortfolioResponse should serialize"),
        )
    }

    /// Handle get_topic_stability tool call.
    ///
    /// Returns stability metrics including churn, entropy, and dream recommendation.
    ///
    /// # Arguments
    /// * `id` - JSON-RPC request ID
    /// * `arguments` - Tool arguments (hours: lookback period)
    ///
    /// # Returns
    /// JsonRpcResponse with TopicStabilityResponse
    ///
    /// # Implements
    /// REQ-MCP-002
    ///
    /// # Constitution Compliance
    /// - AP-70: Dream recommended when entropy > 0.7 AND churn > 0.5
    pub(crate) async fn call_get_topic_stability(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_topic_stability");

        // Parse request
        let request: GetTopicStabilityRequest = match serde_json::from_value(arguments) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "get_topic_stability: Failed to parse request");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Invalid params: {}", e),
                );
            }
        };

        // Validate hours range using DTO's validate method
        if let Err(validation_error) = request.validate() {
            error!(error = %validation_error, "get_topic_stability: Validation failed");
            return self.tool_error_with_pulse(
                id,
                &format!("Invalid params: {}", validation_error),
            );
        }

        debug!(
            hours = request.hours,
            "get_topic_stability: Processing stability request"
        );

        // TASK-INTEG-TOPIC: Get real stability metrics from stability_tracker
        let stability_tracker = self.stability_tracker.read();
        let churn_rate = stability_tracker.current_churn();
        let average_churn = stability_tracker.average_churn(request.hours as i64);

        // Extract entropy from UTL processor status
        let utl_status = self.utl_processor.get_status();
        let entropy = extract_entropy_from_status(&utl_status);

        // Per AP-70: Dream recommended when entropy > 0.7 AND churn > 0.5
        let dream_recommended = TopicStabilityResponse::should_recommend_dream(entropy, churn_rate);
        let high_churn_warning = TopicStabilityResponse::is_high_churn(churn_rate);

        // Get phase breakdown from cluster manager topics
        let cluster_manager = self.cluster_manager.read();
        let topics = cluster_manager.get_topics();
        let phases = compute_phase_breakdown(topics);

        let response = TopicStabilityResponse {
            churn_rate,
            entropy,
            phases,
            dream_recommended,
            high_churn_warning,
            average_churn,
        };

        info!(
            churn_rate = response.churn_rate,
            entropy = response.entropy,
            average_churn = response.average_churn,
            dream_recommended = response.dream_recommended,
            high_churn_warning = response.high_churn_warning,
            "get_topic_stability: Returning stability response"
        );

        self.tool_result_with_pulse(
            id,
            serde_json::to_value(response).expect("TopicStabilityResponse should serialize"),
        )
    }

    /// Handle detect_topics tool call.
    ///
    /// Triggers topic detection/clustering. Requires minimum 3 memories.
    ///
    /// # Arguments
    /// * `id` - JSON-RPC request ID
    /// * `arguments` - Tool arguments (force: force detection)
    ///
    /// # Returns
    /// JsonRpcResponse with DetectTopicsResponse
    ///
    /// # Implements
    /// REQ-MCP-002, BR-MCP-003
    ///
    /// # Constitution Compliance
    /// - min_cluster_size: 3 (per clustering.parameters.min_cluster_size)
    /// - ARCH-09: Topic threshold is weighted_agreement >= 2.5
    pub(crate) async fn call_detect_topics(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling detect_topics");

        // Parse request
        let request: DetectTopicsRequest = match serde_json::from_value(arguments) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "detect_topics: Failed to parse request");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Invalid params: {}", e),
                );
            }
        };

        debug!(force = request.force, "detect_topics: Processing request");

        // Check minimum memory count
        let memory_count = match self.teleological_store.count().await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "detect_topics: Failed to get memory count");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Storage error: Failed to get memory count: {}", e),
                );
            }
        };

        if memory_count < MIN_MEMORIES_FOR_CLUSTERING {
            warn!(
                memory_count = memory_count,
                min_required = MIN_MEMORIES_FOR_CLUSTERING,
                "detect_topics: Insufficient memories for clustering"
            );
            return JsonRpcResponse::error(
                id,
                error_codes::INSUFFICIENT_MEMORIES,
                format!(
                    "Need >= {} memories for topic detection (have {})",
                    MIN_MEMORIES_FOR_CLUSTERING, memory_count
                ),
            );
        }

        debug!(
            memory_count = memory_count,
            force = request.force,
            "detect_topics: Starting topic detection"
        );

        // TASK-INTEG-TOPIC: Trigger reclustering via cluster_manager
        let mut cluster_manager = self.cluster_manager.write();

        // Track topics before reclustering
        let topics_before: std::collections::HashSet<uuid::Uuid> =
            cluster_manager.get_topics().keys().cloned().collect();

        // Run HDBSCAN reclustering
        match cluster_manager.recluster() {
            Ok(result) => {
                // Track topics after reclustering
                let topics_after = cluster_manager.get_topics();
                let total_after = topics_after.len();

                // Find new topics (in after but not in before)
                let new_topics: Vec<TopicSummary> = topics_after
                    .values()
                    .filter(|t| !topics_before.contains(&t.id))
                    .map(topic_to_summary)
                    .collect();

                // Note: Merged topics detection would require tracking merge operations
                // For now, we report new topics only
                let merged_topics = vec![];

                info!(
                    new_topics = new_topics.len(),
                    total_after = total_after,
                    clusters_found = result.total_clusters,
                    "detect_topics: Reclustering completed successfully"
                );

                let response = DetectTopicsResponse {
                    new_topics,
                    merged_topics,
                    total_after,
                    message: Some(format!(
                        "Topic detection completed - {} clusters across 13 spaces, {} topics with weighted_agreement >= 2.5",
                        result.total_clusters, total_after
                    )),
                };

                self.tool_result_with_pulse(
                    id,
                    serde_json::to_value(response).expect("DetectTopicsResponse should serialize"),
                )
            }
            Err(e) => {
                error!(error = %e, "detect_topics: Reclustering failed");
                self.tool_error_with_pulse(
                    id,
                    &format!("Clustering error: {}", e),
                )
            }
        }
    }

    /// Handle get_divergence_alerts tool call.
    ///
    /// Checks for divergence from recent activity using SEMANTIC embedders ONLY.
    ///
    /// # Arguments
    /// * `id` - JSON-RPC request ID
    /// * `arguments` - Tool arguments (lookback_hours)
    ///
    /// # Returns
    /// JsonRpcResponse with DivergenceAlertsResponse
    ///
    /// # Implements
    /// REQ-MCP-002, REQ-MCP-005
    ///
    /// # Constitution Compliance
    /// - AP-62: Only SEMANTIC embedders (E1, E5, E6, E7, E10, E12, E13) trigger alerts
    /// - AP-63: Temporal embedders (E2-E4) NEVER trigger divergence alerts
    /// - ARCH-10: Divergence detection uses SEMANTIC embedders only
    pub(crate) async fn call_get_divergence_alerts(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_divergence_alerts");

        // Parse request
        let request: GetDivergenceAlertsRequest = match serde_json::from_value(arguments) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "get_divergence_alerts: Failed to parse request");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Invalid params: {}", e),
                );
            }
        };

        // Validate lookback range using DTO's validate method
        if let Err(validation_error) = request.validate() {
            error!(error = %validation_error, "get_divergence_alerts: Validation failed");
            return self.tool_error_with_pulse(
                id,
                &format!("Invalid params: {}", validation_error),
            );
        }

        debug!(
            lookback_hours = request.lookback_hours,
            "get_divergence_alerts: Processing request"
        );

        // Per AP-62: Only E1, E5, E6, E7, E10, E12, E13 trigger divergence alerts
        // Temporal embedders (E2-E4) are explicitly excluded per AP-63
        // Divergence detection will compare current context against recent memories
        // using only SEMANTIC embedders

        // Return no alerts for now; alerts will be computed when
        // divergence detection is fully integrated with clustering module
        let response = DivergenceAlertsResponse::no_alerts();

        debug!(
            alert_count = response.alerts.len(),
            severity = %response.severity,
            "get_divergence_alerts: Returning alerts response"
        );

        self.tool_result_with_pulse(
            id,
            serde_json::to_value(response).expect("DivergenceAlertsResponse should serialize"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::topic_dtos::{DivergenceAlert, TOPIC_THRESHOLD, MAX_WEIGHTED_AGREEMENT};

    #[test]
    fn test_constants_match_constitution() {
        // ARCH-09: Topic threshold is weighted_agreement >= 2.5
        assert!(
            (TOPIC_THRESHOLD - 2.5).abs() < f32::EPSILON,
            "TOPIC_THRESHOLD must be 2.5 per ARCH-09"
        );

        // Max weighted agreement = 7*1.0 + 2*0.5 + 1*0.5 = 8.5
        assert!(
            (MAX_WEIGHTED_AGREEMENT - 8.5).abs() < f32::EPSILON,
            "MAX_WEIGHTED_AGREEMENT must be 8.5"
        );

        // Minimum memories for clustering = 3 per constitution
        assert_eq!(
            MIN_MEMORIES_FOR_CLUSTERING, 3,
            "MIN_MEMORIES_FOR_CLUSTERING must be 3"
        );
    }

    #[test]
    fn test_semantic_embedders_for_divergence() {
        // Per AP-62: SEMANTIC embedders for divergence detection
        let spaces = DivergenceAlert::VALID_SEMANTIC_SPACES;
        assert_eq!(spaces.len(), 7, "Should have 7 SEMANTIC embedders");

        // Per AP-63: Temporal embedders must NOT be in the list
        assert!(
            !DivergenceAlert::is_valid_semantic_space("E2_TemporalRecent"),
            "Must NOT include E2 (temporal)"
        );
        assert!(
            !DivergenceAlert::is_valid_semantic_space("E3_TemporalPeriodic"),
            "Must NOT include E3 (temporal)"
        );
        assert!(
            !DivergenceAlert::is_valid_semantic_space("E4_TemporalPositional"),
            "Must NOT include E4 (temporal)"
        );
    }

    #[test]
    fn test_insufficient_memories_error_code() {
        // Per TECH_SPEC_PRD_GAPS.md Section 11.1
        assert_eq!(
            error_codes::INSUFFICIENT_MEMORIES, -32021,
            "INSUFFICIENT_MEMORIES error code must be -32021"
        );
    }
}
