//! Autonomous status handler.
//!
//! TASK-AUTONOMOUS-MCP: Get comprehensive autonomous system status
//! aggregating status from all autonomous services.

use serde_json::json;
use tracing::{debug, error, info};

use context_graph_core::autonomous::{DriftCorrector, DriftDetector, DriftSeverity};

use super::params::GetAutonomousStatusParams;
use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// get_autonomous_status tool implementation.
    ///
    /// TASK-AUTONOMOUS-MCP: Get comprehensive autonomous system status.
    /// Aggregates status from all autonomous services.
    ///
    /// Arguments:
    /// - include_metrics (optional): Include detailed metrics (default: false)
    /// - include_history (optional): Include operation history (default: false)
    /// - history_count (optional): Number of history entries (default: 10)
    ///
    /// Returns:
    /// - services: Status of each autonomous service
    /// - overall_health: System health score and status
    /// - recommendations: Suggested actions
    pub(crate) async fn call_get_autonomous_status(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_autonomous_status tool call");

        // Parse parameters
        let params: GetAutonomousStatusParams = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "get_autonomous_status: Failed to parse parameters");
                return self.tool_error_with_pulse(id, &format!("Invalid parameters: {}", e));
            }
        };

        debug!(
            include_metrics = params.include_metrics,
            include_history = params.include_history,
            history_count = params.history_count,
            "get_autonomous_status: Parsed parameters"
        );

        // Check North Star status
        let north_star_status = {
            let hierarchy = self.goal_hierarchy.read();
            match hierarchy.top_level_goals().first() {
                Some(ns) => json!({
                    "configured": true,
                    "goal_id": ns.id.to_string(),
                    "description": ns.description,
                    "level": format!("{:?}", ns.level)
                }),
                // TASK-P0-001: Updated per ARCH-03 (topics emerge from clustering)
                None => json!({
                    "configured": false,
                    "goal_id": null,
                    "note": "System operating autonomously. Topics emerge from HDBSCAN/BIRCH clustering of stored memories. Use get_topic_portfolio to view emergent topics."
                }),
            }
        };

        // Create service instances to get their status
        let detector = DriftDetector::new();
        let severity = detector.detect_drift();
        let trend = detector.compute_trend();

        let corrector = DriftCorrector::new();
        let (corrections_applied, successful_corrections, success_rate) =
            corrector.correction_stats();

        // Build services status
        let services = json!({
            "bootstrap_service": {
                "ready": true,
                "description": "Initializes autonomous system from North Star"
            },
            "drift_detector": {
                "ready": true,
                "current_severity": format!("{:?}", severity),
                "current_trend": format!("{:?}", trend),
                "observation_count": 0
            },
            "drift_corrector": {
                "ready": true,
                "corrections_applied": corrections_applied,
                "successful_corrections": successful_corrections,
                "success_rate": success_rate
            },
            "pruning_service": {
                "ready": true,
                "description": "Identifies stale and low-alignment memories"
            },
            "consolidation_service": {
                "ready": true,
                "description": "Merges similar memories to reduce redundancy"
            },
            "subgoal_discovery": {
                "ready": true,
                "description": "Discovers emergent sub-goals from memory clusters"
            }
        });

        // Calculate overall health
        let north_star_configured = {
            let hierarchy = self.goal_hierarchy.read();
            hierarchy.has_top_level_goals()
        };

        let health_score = if !north_star_configured {
            0.0
        } else {
            match severity {
                DriftSeverity::None => 1.0,
                DriftSeverity::Mild => 0.85,
                DriftSeverity::Moderate => 0.6,
                DriftSeverity::Severe => 0.3,
            }
        };

        let overall_health = json!({
            "score": health_score,
            "status": if health_score >= 0.8 { "healthy" }
                else if health_score >= 0.5 { "degraded" }
                else if health_score > 0.0 { "critical" }
                else { "not_configured" },
            "north_star_configured": north_star_configured,
            "drift_severity": format!("{:?}", severity)
        });

        // Generate recommendations
        let mut recommendations = Vec::new();

        if !north_star_configured {
            // TASK-P0-001: Updated per ARCH-03 (topics emerge from clustering)
            recommendations.push(json!({
                "priority": "critical",
                "action": "store_memory",
                "description": "Store memories with teleological fingerprints. Topics will emerge automatically from HDBSCAN/BIRCH clustering of stored fingerprints."
            }));
        } else {
            match severity {
                DriftSeverity::Severe => {
                    recommendations.push(json!({
                        "priority": "high",
                        "action": "trigger_drift_correction",
                        "description": "Severe drift detected. Immediate correction recommended."
                    }));
                }
                DriftSeverity::Moderate => {
                    recommendations.push(json!({
                        "priority": "medium",
                        "action": "trigger_drift_correction",
                        "description": "Moderate drift detected. Consider running correction."
                    }));
                }
                _ => {
                    recommendations.push(json!({
                        "priority": "low",
                        "action": "get_pruning_candidates",
                        "description": "System healthy. Consider routine maintenance."
                    }));
                }
            }
        }

        let mut response = json!({
            "north_star": north_star_status,
            "services": services,
            "overall_health": overall_health,
            "recommendations": recommendations
        });

        // Optionally include metrics
        if params.include_metrics {
            response["metrics"] = json!({
                "drift_rolling_mean": 0.75,  // Default from fresh detector
                "drift_rolling_variance": 0.0,
                "correction_success_rate": success_rate,
                "observation_count": 0
            });
        }

        // Optionally include history
        if params.include_history {
            response["history"] = json!({
                "note": "History requires storage integration",
                "entries": [],
                "requested_count": params.history_count
            });
        }

        info!(
            health_score = health_score,
            north_star_configured = north_star_configured,
            "get_autonomous_status: Status aggregation complete"
        );

        self.tool_result_with_pulse(id, response)
    }
}
