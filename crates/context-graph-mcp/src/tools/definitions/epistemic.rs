//! Epistemic action tool definitions (TASK-MCP-001).
//!
//! Implements epistemic_action tool for GWT workspace uncertainty/knowledge updates.
//! Constitution: utl.johari.Unknown → EpistemicAction
//! PRD: Section 1.8, Section 5.2 Line 527

use serde_json::json;
use crate::tools::types::ToolDefinition;

/// Returns epistemic tool definitions.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "epistemic_action",
            "Perform an epistemic action on the GWT workspace to update uncertainty \
             and knowledge states. Actions: assert (add belief), retract (remove belief), \
             query (check status), hypothesize (tentative belief), verify (confirm/deny). \
             Used when Johari quadrant is Unknown (high entropy + high coherence).",
            json!({
                "type": "object",
                "required": ["action_type", "target", "rationale"],
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["assert", "retract", "query", "hypothesize", "verify"],
                        "description": "Type of epistemic action: assert=add belief, retract=remove belief, query=check status, hypothesize=tentative belief, verify=confirm/deny"
                    },
                    "target": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 4096,
                        "description": "Target concept or proposition (1-4096 chars)"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "Confidence level [0.0, 1.0], default 0.5"
                    },
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024,
                        "description": "Rationale for action (required per PRD 0.3)"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context for the action",
                        "properties": {
                            "source_nodes": {
                                "type": "array",
                                "items": { "type": "string", "format": "uuid" },
                                "description": "UUIDs of related source nodes"
                            },
                            "uncertainty_type": {
                                "type": "string",
                                "enum": ["epistemic", "aleatory", "mixed"],
                                "description": "epistemic=knowledge gap, aleatory=inherent randomness, mixed=both"
                            }
                        }
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
    fn test_epistemic_action_definition_exists() {
        let tools = definitions();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "epistemic_action");
    }

    #[test]
    fn test_epistemic_action_schema_required_fields() {
        let tools = definitions();
        let schema = &tools[0].input_schema;
        let required = schema.get("required").unwrap().as_array().unwrap();

        assert!(required.iter().any(|v| v.as_str() == Some("action_type")));
        assert!(required.iter().any(|v| v.as_str() == Some("target")));
        assert!(required.iter().any(|v| v.as_str() == Some("rationale")));
    }

    #[test]
    fn test_epistemic_action_type_enum_values() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let action_type = props.get("action_type").unwrap();
        let enum_values = action_type.get("enum").unwrap().as_array().unwrap();

        let values: Vec<&str> = enum_values.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(values.contains(&"assert"));
        assert!(values.contains(&"retract"));
        assert!(values.contains(&"query"));
        assert!(values.contains(&"hypothesize"));
        assert!(values.contains(&"verify"));
    }

    #[test]
    fn test_target_length_constraints() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let target = props.get("target").unwrap();

        assert_eq!(target.get("minLength").unwrap().as_u64().unwrap(), 1);
        assert_eq!(target.get("maxLength").unwrap().as_u64().unwrap(), 4096);
    }

    #[test]
    fn test_rationale_length_constraints() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let rationale = props.get("rationale").unwrap();

        assert_eq!(rationale.get("minLength").unwrap().as_u64().unwrap(), 1);
        assert_eq!(rationale.get("maxLength").unwrap().as_u64().unwrap(), 1024);
    }

    #[test]
    fn test_confidence_bounds() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let confidence = props.get("confidence").unwrap();

        assert_eq!(confidence.get("minimum").unwrap().as_f64().unwrap(), 0.0);
        assert_eq!(confidence.get("maximum").unwrap().as_f64().unwrap(), 1.0);
        assert_eq!(confidence.get("default").unwrap().as_f64().unwrap(), 0.5);
    }

    #[test]
    fn test_context_uncertainty_types() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let context = props.get("context").unwrap();
        let context_props = context.get("properties").unwrap();
        let uncertainty = context_props.get("uncertainty_type").unwrap();
        let enum_values = uncertainty.get("enum").unwrap().as_array().unwrap();

        let values: Vec<&str> = enum_values.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(values.contains(&"epistemic"));
        assert!(values.contains(&"aleatory"));
        assert!(values.contains(&"mixed"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let tools = definitions();
        let json_str = serde_json::to_string(&tools).expect("Serialization failed");
        assert!(json_str.contains("epistemic_action"));
        assert!(json_str.contains("inputSchema"));
    }

    // ========== MANUAL TESTING WITH SYNTHETIC DATA (TASK-27 Section 7) ==========

    #[test]
    fn test_synthetic_valid_input() {
        // Synthetic test data from TASK-27 Section 7.1
        let synthetic_input = json!({
            "action_type": "hypothesize",
            "target": "The system should consolidate memories when IC < 0.5",
            "confidence": 0.75,
            "rationale": "Identity crisis threshold per constitution.yaml gwt.self_ego_node.thresholds.critical",
            "context": {
                "source_nodes": ["550e8400-e29b-41d4-a716-446655440000"],
                "uncertainty_type": "epistemic"
            }
        });

        let tools = definitions();
        let schema = &tools[0].input_schema;

        // Verify schema structure matches expected
        let props = schema.get("properties").unwrap();

        // action_type validation
        let action_type = props.get("action_type").unwrap();
        let action_enum = action_type.get("enum").unwrap().as_array().unwrap();
        let input_action = synthetic_input.get("action_type").unwrap().as_str().unwrap();
        let valid_actions: Vec<&str> = action_enum.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(valid_actions.contains(&input_action), "action_type 'hypothesize' should be valid");

        // target length validation
        let target = props.get("target").unwrap();
        let target_min = target.get("minLength").unwrap().as_u64().unwrap();
        let target_max = target.get("maxLength").unwrap().as_u64().unwrap();
        let input_target_len = synthetic_input.get("target").unwrap().as_str().unwrap().len() as u64;
        assert!(input_target_len >= target_min, "target length {} should be >= {}", input_target_len, target_min);
        assert!(input_target_len <= target_max, "target length {} should be <= {}", input_target_len, target_max);

        // confidence bounds validation
        let confidence = props.get("confidence").unwrap();
        let conf_min = confidence.get("minimum").unwrap().as_f64().unwrap();
        let conf_max = confidence.get("maximum").unwrap().as_f64().unwrap();
        let input_conf = synthetic_input.get("confidence").unwrap().as_f64().unwrap();
        assert!(input_conf >= conf_min && input_conf <= conf_max,
            "confidence {} should be in [{}, {}]", input_conf, conf_min, conf_max);

        // rationale length validation
        let rationale = props.get("rationale").unwrap();
        let rat_min = rationale.get("minLength").unwrap().as_u64().unwrap();
        let rat_max = rationale.get("maxLength").unwrap().as_u64().unwrap();
        let input_rat_len = synthetic_input.get("rationale").unwrap().as_str().unwrap().len() as u64;
        assert!(input_rat_len >= rat_min && input_rat_len <= rat_max,
            "rationale length {} should be in [{}, {}]", input_rat_len, rat_min, rat_max);

        // context.uncertainty_type validation
        let context = props.get("context").unwrap();
        let context_props = context.get("properties").unwrap();
        let uncertainty = context_props.get("uncertainty_type").unwrap();
        let uncertainty_enum = uncertainty.get("enum").unwrap().as_array().unwrap();
        let input_uncertainty = synthetic_input.get("context").unwrap()
            .get("uncertainty_type").unwrap().as_str().unwrap();
        let valid_uncertainties: Vec<&str> = uncertainty_enum.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(valid_uncertainties.contains(&input_uncertainty),
            "uncertainty_type 'epistemic' should be valid");
    }

    #[test]
    fn test_edge_case_maximum_target_length() {
        // Edge Case 2: Maximum length target (4096 chars)
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let target = props.get("target").unwrap();
        let max_len = target.get("maxLength").unwrap().as_u64().unwrap();

        // Schema should accept exactly 4096 chars
        assert_eq!(max_len, 4096);

        // Test that a 4096-char string would be valid
        let long_target = "x".repeat(4096);
        assert_eq!(long_target.len(), 4096);
    }

    #[test]
    fn test_edge_case_minimum_values() {
        // Edge Case: Minimum valid values
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();

        // target minLength = 1
        let target = props.get("target").unwrap();
        assert_eq!(target.get("minLength").unwrap().as_u64().unwrap(), 1);

        // rationale minLength = 1
        let rationale = props.get("rationale").unwrap();
        assert_eq!(rationale.get("minLength").unwrap().as_u64().unwrap(), 1);

        // confidence minimum = 0.0
        let confidence = props.get("confidence").unwrap();
        assert_eq!(confidence.get("minimum").unwrap().as_f64().unwrap(), 0.0);
    }

    #[test]
    fn test_edge_case_required_fields() {
        // Edge Case 3: Missing required fields - verify schema enforces them
        let tools = definitions();
        let schema = &tools[0].input_schema;
        let required = schema.get("required").unwrap().as_array().unwrap();

        // All 3 required fields must be present
        let required_fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(required_fields.len(), 3);
        assert!(required_fields.contains(&"action_type"));
        assert!(required_fields.contains(&"target"));
        assert!(required_fields.contains(&"rationale"));

        // confidence and context are NOT required
        assert!(!required_fields.contains(&"confidence"));
        assert!(!required_fields.contains(&"context"));
    }

    #[test]
    fn test_verify_epistemic_in_all_tools() {
        // Per TASK-27 Section 7.2: Verify tool appears in definitions
        let tools = crate::tools::get_tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"epistemic_action"), "epistemic_action missing from tool list");
    }

    #[test]
    fn test_all_action_type_values() {
        // Verify all 5 action types are present
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let action_type = props.get("action_type").unwrap();
        let enum_values = action_type.get("enum").unwrap().as_array().unwrap();

        let values: Vec<&str> = enum_values.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(values.len(), 5);

        // Constitution.yaml johari.Unknown -> EpistemicAction
        // These are the 5 epistemic operations:
        assert!(values.contains(&"assert"), "assert: add belief");
        assert!(values.contains(&"retract"), "retract: remove belief");
        assert!(values.contains(&"query"), "query: check status");
        assert!(values.contains(&"hypothesize"), "hypothesize: tentative belief");
        assert!(values.contains(&"verify"), "verify: confirm/deny");
    }

    #[test]
    fn test_source_nodes_uuid_format() {
        // Verify source_nodes expects UUID format
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let context = props.get("context").unwrap();
        let context_props = context.get("properties").unwrap();
        let source_nodes = context_props.get("source_nodes").unwrap();
        let items = source_nodes.get("items").unwrap();

        assert_eq!(items.get("type").unwrap().as_str().unwrap(), "string");
        assert_eq!(items.get("format").unwrap().as_str().unwrap(), "uuid");
    }
}
