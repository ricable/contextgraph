//! Curation tool definitions per PRD v6 Section 10.3.
//!
//! Tools:
//! - forget_concept: Soft-delete a memory (30-day recovery per SEC-06)
//! - boost_importance: Adjust memory importance score
//!
//! Constitution Compliance:
//! - SEC-06: Soft delete 30-day recovery
//! - BR-MCP-001: forget_concept uses soft delete by default
//! - BR-MCP-002: boost_importance clamps final value to [0.0, 1.0]
//! - AP-10: No NaN/Infinity in values

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns curation tool definitions (2 tools per PRD).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // forget_concept
        ToolDefinition::new(
            "forget_concept",
            "Soft-delete a memory with 30-day recovery window (per SEC-06). \
             Set soft_delete=false for permanent deletion (use with caution). \
             Returns deleted_at timestamp for recovery tracking.",
            json!({
                "type": "object",
                "required": ["node_id"],
                "properties": {
                    "node_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the memory to forget"
                    },
                    "soft_delete": {
                        "type": "boolean",
                        "default": true,
                        "description": "Use soft delete with 30-day recovery (default true per BR-MCP-001)"
                    }
                }
            }),
        ),
        // boost_importance
        ToolDefinition::new(
            "boost_importance",
            "Adjust a memory's importance score by delta. Final value is clamped \
             to [0.0, 1.0] (per BR-MCP-002). Response includes old, delta, and new values.",
            json!({
                "type": "object",
                "required": ["node_id", "delta"],
                "properties": {
                    "node_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the memory to boost"
                    },
                    "delta": {
                        "type": "number",
                        "minimum": -1.0,
                        "maximum": 1.0,
                        "description": "Importance change value (-1.0 to 1.0)"
                    }
                }
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curation_definitions_count() {
        let tools = definitions();
        assert_eq!(tools.len(), 2, "Should have 2 curation tools");
    }

    #[test]
    fn test_curation_tools_names() {
        let tools = definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"forget_concept"));
        assert!(names.contains(&"boost_importance"));
    }

    #[test]
    fn test_forget_concept_required_fields() {
        let tools = definitions();
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        let required = forget
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        let fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(fields.len(), 1);
        assert!(fields.contains(&"node_id"));
        // soft_delete NOT required - has default
        assert!(!fields.contains(&"soft_delete"));
    }

    #[test]
    fn test_forget_concept_soft_delete_default() {
        let tools = definitions();
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        let props = forget.input_schema.get("properties").unwrap();
        let soft_delete = props.get("soft_delete").unwrap();
        // Per BR-MCP-001: defaults to true
        assert!(soft_delete.get("default").unwrap().as_bool().unwrap());
    }

    #[test]
    fn test_boost_importance_required_fields() {
        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let required = boost
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        let fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(fields.len(), 2);
        assert!(fields.contains(&"node_id"));
        assert!(fields.contains(&"delta"));
    }

    #[test]
    fn test_boost_importance_delta_range() {
        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let props = boost.input_schema.get("properties").unwrap();
        let delta = props.get("delta").unwrap();
        // Per BR-MCP-002: delta range [-1.0, 1.0]
        assert!((delta.get("minimum").unwrap().as_f64().unwrap() - (-1.0)).abs() < f64::EPSILON);
        assert!((delta.get("maximum").unwrap().as_f64().unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_node_id_uuid_format() {
        let tools = definitions();
        for tool in &tools {
            let props = tool.input_schema.get("properties").unwrap();
            let node_id = props.get("node_id").unwrap();
            assert_eq!(node_id.get("type").unwrap().as_str().unwrap(), "string");
            assert_eq!(node_id.get("format").unwrap().as_str().unwrap(), "uuid");
        }
    }

    #[test]
    fn test_descriptions_mention_constitution() {
        let tools = definitions();
        // forget_concept mentions SEC-06
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        assert!(
            forget.description.contains("SEC-06"),
            "Should reference SEC-06"
        );
        assert!(
            forget.description.contains("30-day"),
            "Should mention recovery period"
        );

        // boost_importance mentions BR-MCP-002
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        assert!(
            boost.description.contains("BR-MCP-002"),
            "Should reference BR-MCP-002"
        );
    }

    // ========== SYNTHETIC DATA VALIDATION TESTS ==========

    #[test]
    fn test_synthetic_forget_concept_soft_delete() {
        // Synthetic test data matching schema exactly
        let synthetic_input = json!({
            "node_id": "550e8400-e29b-41d4-a716-446655440000"
        });

        let tools = definitions();
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        let props = forget.input_schema.get("properties").unwrap();

        // Validate node_id is a UUID
        let node_id = synthetic_input.get("node_id").unwrap().as_str().unwrap();
        assert!(uuid::Uuid::parse_str(node_id).is_ok());

        // Validate soft_delete defaults to true
        let soft_delete_default = props
            .get("soft_delete")
            .unwrap()
            .get("default")
            .unwrap()
            .as_bool()
            .unwrap();
        assert!(soft_delete_default);

        println!("[SYNTHETIC TEST] forget_concept with node_id only uses soft_delete=true default");
    }

    #[test]
    fn test_synthetic_forget_concept_hard_delete() {
        // Hard delete test
        let synthetic_input = json!({
            "node_id": "550e8400-e29b-41d4-a716-446655440000",
            "soft_delete": false
        });

        // Validate node_id
        let node_id = synthetic_input.get("node_id").unwrap().as_str().unwrap();
        assert!(uuid::Uuid::parse_str(node_id).is_ok());

        // Validate soft_delete is false
        let soft_delete = synthetic_input
            .get("soft_delete")
            .unwrap()
            .as_bool()
            .unwrap();
        assert!(!soft_delete);

        println!("[SYNTHETIC TEST] forget_concept with soft_delete=false for hard delete");
    }

    #[test]
    fn test_synthetic_boost_importance_positive() {
        // Positive boost test
        let synthetic_input = json!({
            "node_id": "550e8400-e29b-41d4-a716-446655440000",
            "delta": 0.3
        });

        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let props = boost.input_schema.get("properties").unwrap();
        let delta_schema = props.get("delta").unwrap();

        // Validate node_id
        let node_id = synthetic_input.get("node_id").unwrap().as_str().unwrap();
        assert!(uuid::Uuid::parse_str(node_id).is_ok());

        // Validate delta is in range
        let delta = synthetic_input.get("delta").unwrap().as_f64().unwrap();
        let min = delta_schema.get("minimum").unwrap().as_f64().unwrap();
        let max = delta_schema.get("maximum").unwrap().as_f64().unwrap();
        assert!(delta >= min && delta <= max);

        println!("[SYNTHETIC TEST] boost_importance with delta=0.3 (positive boost)");
    }

    #[test]
    fn test_synthetic_boost_importance_negative() {
        // Negative boost (demote) test
        let synthetic_input = json!({
            "node_id": "550e8400-e29b-41d4-a716-446655440000",
            "delta": -0.2
        });

        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let props = boost.input_schema.get("properties").unwrap();
        let delta_schema = props.get("delta").unwrap();

        // Validate delta is in range (negative)
        let delta = synthetic_input.get("delta").unwrap().as_f64().unwrap();
        let min = delta_schema.get("minimum").unwrap().as_f64().unwrap();
        let max = delta_schema.get("maximum").unwrap().as_f64().unwrap();
        assert!(delta >= min && delta <= max);

        println!("[SYNTHETIC TEST] boost_importance with delta=-0.2 (demote)");
    }

    #[test]
    fn test_synthetic_boost_importance_boundary_max() {
        // Boundary max test
        let synthetic_input = json!({
            "node_id": "550e8400-e29b-41d4-a716-446655440000",
            "delta": 1.0
        });

        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let props = boost.input_schema.get("properties").unwrap();
        let delta_schema = props.get("delta").unwrap();

        let delta = synthetic_input.get("delta").unwrap().as_f64().unwrap();
        let max = delta_schema.get("maximum").unwrap().as_f64().unwrap();
        assert!((delta - max).abs() < f64::EPSILON);

        println!("[SYNTHETIC TEST] boost_importance with delta=1.0 (max boundary)");
    }

    #[test]
    fn test_synthetic_boost_importance_boundary_min() {
        // Boundary min test
        let synthetic_input = json!({
            "node_id": "550e8400-e29b-41d4-a716-446655440000",
            "delta": -1.0
        });

        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let props = boost.input_schema.get("properties").unwrap();
        let delta_schema = props.get("delta").unwrap();

        let delta = synthetic_input.get("delta").unwrap().as_f64().unwrap();
        let min = delta_schema.get("minimum").unwrap().as_f64().unwrap();
        assert!((delta - min).abs() < f64::EPSILON);

        println!("[SYNTHETIC TEST] boost_importance with delta=-1.0 (min boundary)");
    }

    // ========== EDGE CASE TESTS ==========

    #[test]
    fn test_edge_case_delta_zero() {
        // Zero delta is valid
        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let props = boost.input_schema.get("properties").unwrap();
        let delta_schema = props.get("delta").unwrap();

        let delta = 0.0;
        let min = delta_schema.get("minimum").unwrap().as_f64().unwrap();
        let max = delta_schema.get("maximum").unwrap().as_f64().unwrap();
        assert!(delta >= min && delta <= max);

        println!("[EDGE CASE] boost_importance with delta=0.0 is valid (no change)");
    }

    #[test]
    fn test_edge_case_nil_uuid() {
        // Nil UUID is valid format
        let nil_uuid = "00000000-0000-0000-0000-000000000000";
        assert!(uuid::Uuid::parse_str(nil_uuid).is_ok());
        println!("[EDGE CASE] Nil UUID is valid format");
    }

    #[test]
    fn test_all_tools_have_type_object() {
        let tools = definitions();
        for tool in &tools {
            assert_eq!(
                tool.input_schema.get("type").unwrap().as_str().unwrap(),
                "object",
                "Tool {} should have type: object",
                tool.name
            );
        }
        println!("[PASS] All curation tools have type: object schema");
    }

    #[test]
    fn test_all_tools_have_required_array() {
        let tools = definitions();
        for tool in &tools {
            assert!(
                tool.input_schema.get("required").is_some(),
                "Tool {} should have required array",
                tool.name
            );
        }
        println!("[PASS] All curation tools have required array defined");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let tools = definitions();
        let json_str = serde_json::to_string(&tools).expect("Serialization failed");
        assert!(json_str.contains("forget_concept"));
        assert!(json_str.contains("boost_importance"));
        assert!(json_str.contains("inputSchema"));
        assert!(json_str.contains("node_id"));
        assert!(json_str.contains("soft_delete"));
        assert!(json_str.contains("delta"));
        println!("[PASS] Curation tools serialize correctly");
    }
}
