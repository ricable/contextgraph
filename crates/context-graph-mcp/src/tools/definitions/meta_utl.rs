//! Tool definitions for Meta-UTL self-correction MCP tools.
//!
//! TASK-METAUTL-P0-005: Defines schemas for status, recalibration, and log tools.

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns Meta-UTL tool definitions (3 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // get_meta_learning_status - Read current self-correction state
        ToolDefinition::new(
            "get_meta_learning_status",
            "Get current Meta-UTL self-correction status including accuracy, lambda weights, \
             and escalation state. Returns comprehensive system state including rolling accuracy, \
             consecutive low accuracy cycle count, current/base lambda weights, escalation status, \
             and recent event count. Optionally includes accuracy history and per-embedder breakdown.",
            json!({
                "type": "object",
                "properties": {
                    "include_accuracy_history": {
                        "type": "boolean",
                        "description": "Include rolling accuracy history values",
                        "default": false
                    },
                    "include_embedder_breakdown": {
                        "type": "boolean",
                        "description": "Include per-embedder (E1-E13) accuracy breakdown",
                        "default": false
                    }
                },
                "required": []
            }),
        ),
        // trigger_lambda_recalibration - Manually trigger lambda optimization
        ToolDefinition::new(
            "trigger_lambda_recalibration",
            "Manually trigger lambda weight recalibration using gradient adjustment or Bayesian \
             optimization. Gradient adjustment modifies lambda_s and lambda_c based on prediction \
             error and ACh-modulated learning rate. Bayesian optimization uses a Gaussian Process \
             surrogate model with Expected Improvement acquisition. Supports dry-run mode for \
             testing without state mutation.",
            json!({
                "type": "object",
                "properties": {
                    "force_bayesian": {
                        "type": "boolean",
                        "description": "Force Bayesian optimization instead of gradient adjustment",
                        "default": false
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["code", "medical", "legal", "creative", "research", "general"],
                        "description": "Target domain for calibration (optional)"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Compute adjustment but don't apply it",
                        "default": false
                    }
                },
                "required": []
            }),
        ),
        // get_meta_learning_log - Query historical meta-learning events
        ToolDefinition::new(
            "get_meta_learning_log",
            "Query meta-learning event log with optional filters. Returns events matching \
             specified time range, event type, and domain filters. Supports pagination via \
             limit/offset. Event types include lambda_adjustment, bayesian_escalation, \
             accuracy_alert, accuracy_recovery, and weight_clamped.",
            json!({
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "Start time (ISO 8601 format, e.g., 2024-01-01T00:00:00Z)"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time (ISO 8601 format)"
                    },
                    "event_type": {
                        "type": "string",
                        "enum": ["lambda_adjustment", "bayesian_escalation", "accuracy_alert", "accuracy_recovery", "weight_clamped"],
                        "description": "Filter by event type"
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["code", "medical", "legal", "creative", "research", "general"],
                        "description": "Filter by domain"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100,
                        "description": "Maximum events to return"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Number of events to skip (pagination)"
                    }
                },
                "required": []
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definitions_count() {
        let defs = definitions();
        assert_eq!(defs.len(), 3);
    }

    #[test]
    fn test_get_meta_learning_status_schema() {
        let defs = definitions();
        let status = defs.iter().find(|d| d.name == "get_meta_learning_status");
        assert!(status.is_some());
        let status = status.unwrap();
        assert!(status.input_schema.get("properties").is_some());
        let props = status.input_schema.get("properties").unwrap();
        assert!(props.get("include_accuracy_history").is_some());
        assert!(props.get("include_embedder_breakdown").is_some());
    }

    #[test]
    fn test_trigger_recalibration_schema() {
        let defs = definitions();
        let recal = defs
            .iter()
            .find(|d| d.name == "trigger_lambda_recalibration");
        assert!(recal.is_some());
        let recal = recal.unwrap();
        let props = recal.input_schema.get("properties").unwrap();
        assert!(props.get("force_bayesian").is_some());
        assert!(props.get("domain").is_some());
        assert!(props.get("dry_run").is_some());
    }

    #[test]
    fn test_get_meta_learning_log_schema() {
        let defs = definitions();
        let log = defs.iter().find(|d| d.name == "get_meta_learning_log");
        assert!(log.is_some());
        let log = log.unwrap();
        let props = log.input_schema.get("properties").unwrap();
        assert!(props.get("start_time").is_some());
        assert!(props.get("end_time").is_some());
        assert!(props.get("event_type").is_some());
        assert!(props.get("domain").is_some());
        assert!(props.get("limit").is_some());
        assert!(props.get("offset").is_some());
    }

    #[test]
    fn test_schema_serialization() {
        let defs = definitions();
        let json = serde_json::to_string(&defs);
        assert!(json.is_ok());
        let json = json.unwrap();
        assert!(json.contains("get_meta_learning_status"));
        assert!(json.contains("trigger_lambda_recalibration"));
        assert!(json.contains("get_meta_learning_log"));
    }
}
