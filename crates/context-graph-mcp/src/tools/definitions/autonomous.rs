//! Autonomous system tool definitions.
//! TASK-AUTONOMOUS-MCP: Drift detection, correction, pruning, consolidation, sub-goals, status.
//! SPEC-AUTONOMOUS-001: Added 5 missing tools (get_learner_state, observe_outcome, execute_prune,
//!                      get_health_status, trigger_healing).
//! TASK-P0-001: REMOVED auto_bootstrap_north_star per ARCH-03 (goals emerge from topic clustering)

use serde_json::json;
use crate::tools::types::ToolDefinition;

/// Returns Autonomous tool definitions (12 tools after TASK-P0-001 removal).
/// SPEC-AUTONOMOUS-001: Added 5 new tools per constitution NORTH-009, NORTH-012, NORTH-020.
/// TASK-FIX-002/NORTH-010: Added get_drift_history tool.
/// TASK-P0-001: Removed auto_bootstrap_north_star - goals emerge autonomously from clustering.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // REMOVED: auto_bootstrap_north_star per TASK-P0-001 (ARCH-03)
        // Goals emerge autonomously from topic clustering, no manual bootstrap needed.
        // Use get_topic_portfolio and get_topic_stability for topic-based goal discovery.

        // get_alignment_drift - Get drift state and history
        ToolDefinition::new(
            "get_alignment_drift",
            "Get the current alignment drift state including severity, trend, and recommendations. \
             Drift measures how far the system has deviated from the North Star goal alignment. \
             High drift indicates memories are becoming misaligned with the primary purpose.",
            json!({
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "enum": ["1h", "24h", "7d", "30d"],
                        "default": "24h",
                        "description": "Timeframe to analyze for drift"
                    },
                    "include_history": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include full drift history in response"
                    }
                },
                "required": []
            }),
        ),

        // get_drift_history - Get historical drift measurements (NORTH-010, TASK-FIX-002)
        ToolDefinition::new(
            "get_drift_history",
            "Get historical drift measurements over time. Returns timestamped drift entries \
             with per-embedder similarity scores, overall drift deltas, and trend analysis. \
             Use this to understand drift patterns and predict future alignment issues. \
             Requires prior calls to get_alignment_drift with include_history=true or \
             check_drift_with_history to populate history data.",
            json!({
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "Goal UUID to retrieve history for (defaults to North Star if not provided)"
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["1h", "6h", "24h", "7d", "30d", "all"],
                        "default": "24h",
                        "description": "Time range filter for history entries"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                        "description": "Maximum number of history entries to return"
                    },
                    "include_per_embedder": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include per-embedder breakdown for each history entry"
                    },
                    "compute_deltas": {
                        "type": "boolean",
                        "default": true,
                        "description": "Compute drift deltas between consecutive entries"
                    }
                },
                "required": []
            }),
        ),

        // trigger_drift_correction - Manually trigger drift correction
        ToolDefinition::new(
            "trigger_drift_correction",
            "Manually trigger a drift correction cycle. Applies correction strategies based on \
             current drift severity: threshold adjustment, weight rebalancing, goal reinforcement, \
             or emergency intervention for severe drift.",
            json!({
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force correction even if drift severity is low"
                    },
                    "target_alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Target alignment to achieve (optional, uses adaptive if not set)"
                    }
                },
                "required": []
            }),
        ),

        // get_pruning_candidates - Get memories eligible for pruning
        ToolDefinition::new(
            "get_pruning_candidates",
            "Identify memories that are candidates for pruning based on staleness, low alignment, \
             redundancy, or orphaned status. Returns a ranked list with reasons and recommendations. \
             Use this for routine memory hygiene and to identify unused/outdated content.",
            json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 20,
                        "description": "Maximum number of candidates to return"
                    },
                    "min_staleness_days": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 30,
                        "description": "Minimum age in days for staleness consideration"
                    },
                    "min_alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                        "description": "Memories below this alignment are candidates for low-alignment pruning"
                    }
                },
                "required": []
            }),
        ),

        // trigger_consolidation - Trigger memory consolidation
        ToolDefinition::new(
            "trigger_consolidation",
            "Trigger memory consolidation to merge similar memories and reduce redundancy. \
             Uses similarity-based, temporal, or semantic strategies to identify merge candidates. \
             Helps optimize memory storage and improve retrieval efficiency.",
            json!({
                "type": "object",
                "properties": {
                    "max_memories": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 100,
                        "description": "Maximum memories to process in one batch"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["similarity", "temporal", "semantic"],
                        "default": "similarity",
                        "description": "Consolidation strategy to use"
                    },
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.85,
                        "description": "Minimum similarity threshold for consolidation candidates"
                    }
                },
                "required": []
            }),
        ),

        // discover_sub_goals - Discover potential sub-goals
        ToolDefinition::new(
            "discover_sub_goals",
            "Discover potential sub-goals from memory clusters. Analyzes stored memories to find \
             emergent themes and patterns that could become strategic or tactical goals. \
             Helps evolve the goal hierarchy based on actual content.",
            json!({
                "type": "object",
                "properties": {
                    "min_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.6,
                        "description": "Minimum confidence for a discovered sub-goal"
                    },
                    "max_goals": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum number of sub-goals to discover"
                    },
                    "parent_goal_id": {
                        "type": "string",
                        "description": "Parent goal ID to discover sub-goals for (defaults to North Star)"
                    }
                },
                "required": []
            }),
        ),

        // get_autonomous_status - Get comprehensive autonomous system status
        ToolDefinition::new(
            "get_autonomous_status",
            "Get comprehensive status of the autonomous North Star system including all services: \
             drift detection, correction, pruning, consolidation, and sub-goal discovery. \
             Returns health scores, recommendations, and optional detailed metrics.",
            json!({
                "type": "object",
                "properties": {
                    "include_metrics": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include detailed per-service metrics"
                    },
                    "include_history": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include recent operation history"
                    },
                    "history_count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Number of history entries to include"
                    }
                },
                "required": []
            }),
        ),

        // ========== SPEC-AUTONOMOUS-001: 5 NEW TOOLS (NORTH-009, NORTH-012, NORTH-020) ==========

        // get_learner_state - Get Meta-UTL learner state (NORTH-009, METAUTL-004)
        ToolDefinition::new(
            "get_learner_state",
            "Get Meta-UTL learner state including accuracy, prediction count, domain-specific stats, \
             and lambda weights. Per NORTH-009: Monitors learning accuracy and lambda evolution. \
             Per METAUTL-004: Domain-specific accuracy tracking required.",
            json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter (e.g., 'Code', 'Medical', 'General')"
                    }
                },
                "required": []
            }),
        ),

        // observe_outcome - Record learning outcome for Meta-UTL (NORTH-009, METAUTL-001)
        ToolDefinition::new(
            "observe_outcome",
            "Record actual outcome for a Meta-UTL prediction. Per NORTH-009: Enables self-correction \
             through outcome observation. Per METAUTL-001: prediction_error > 0.2 triggers lambda adjustment. \
             Prediction history TTL is 24 hours.",
            json!({
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "description": "UUID of the prediction to update"
                    },
                    "actual_outcome": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Actual outcome value (0.0-1.0)"
                    },
                    "context": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "Domain of the prediction (Code, Medical, General)"
                            },
                            "query_type": {
                                "type": "string",
                                "description": "Type of query (retrieval, classification, etc.)"
                            }
                        },
                        "description": "Optional context for the outcome"
                    }
                },
                "required": ["prediction_id", "actual_outcome"]
            }),
        ),

        // execute_prune - Execute graph pruning on candidates (NORTH-012)
        ToolDefinition::new(
            "execute_prune",
            "Execute pruning on identified candidate nodes. Per NORTH-012: Completes the pruning workflow \
             started by get_pruning_candidates. Uses soft delete with 30-day recovery per SEC-06. \
             SELF_EGO_NODE is protected and cannot be pruned.",
            json!({
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 0,
                        "description": "Array of node UUIDs to prune"
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["staleness", "low_alignment", "redundancy", "orphan"],
                        "description": "Reason for pruning (for audit logging)"
                    },
                    "cascade": {
                        "type": "boolean",
                        "default": false,
                        "description": "Also prune dependent nodes and edges"
                    }
                },
                "required": ["node_ids", "reason"]
            }),
        ),

        // get_health_status - Get system-wide health status (NORTH-020)
        ToolDefinition::new(
            "get_health_status",
            "Get health status for all major subsystems: UTL, GWT, Dream, Storage. \
             Per NORTH-020: Provides unified health view to identify degradation before cascading failures. \
             Returns overall_status (healthy/degraded/critical) and per-subsystem metrics.",
            json!({
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "enum": ["utl", "gwt", "dream", "storage", "all"],
                        "default": "all",
                        "description": "Specific subsystem to query, or 'all' for complete health"
                    }
                },
                "required": []
            }),
        ),

        // trigger_healing - Trigger self-healing protocol (NORTH-020)
        ToolDefinition::new(
            "trigger_healing",
            "Trigger self-healing protocol for a degraded subsystem. Per NORTH-020: Autonomous recovery \
             without manual intervention. Healing actions vary by subsystem and severity: \
             UTL resets lambda weights, GWT resets Kuramoto phases, Dream aborts/resets scheduler, \
             Storage compacts RocksDB and clears caches.",
            json!({
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "enum": ["utl", "gwt", "dream", "storage"],
                        "description": "Subsystem to heal"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "default": "medium",
                        "description": "Healing severity (affects action aggressiveness)"
                    }
                },
                "required": ["subsystem"]
            }),
        ),
    ]
}
