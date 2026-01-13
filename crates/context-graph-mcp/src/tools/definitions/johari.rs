//! Johari classification tool definitions (TASK-MCP-005).
//!
//! Implements get_johari_classification tool for UTL Johari Window analysis.
//! Constitution: utl.johari (lines 154-157)
//!
//! ## Johari Quadrant Mapping
//! - Open (ΔS<0.5, ΔC>0.5) → DirectRecall
//! - Blind (ΔS>0.5, ΔC<0.5) → TriggerDream
//! - Hidden (ΔS<0.5, ΔC<0.5) → GetNeighborhood
//! - Unknown (ΔS>0.5, ΔC>0.5) → EpistemicAction

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns johari tool definitions.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![ToolDefinition::new(
        "get_johari_classification",
        "Classify into a Johari Window quadrant based on surprise (delta_s) \
         and coherence (delta_c) metrics. Returns quadrant, metrics, suggested action, \
         and explanation. Supports two modes: (1) Direct mode - provide delta_s and delta_c \
         values, (2) Memory mode - provide memory_id to classify from stored fingerprint. \
         Constitution: utl.johari lines 154-157.",
        json!({
            "type": "object",
            "properties": {
                "delta_s": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Surprise metric [0.0, 1.0]. Required if memory_id not provided."
                },
                "delta_c": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Coherence metric [0.0, 1.0]. Required if memory_id not provided."
                },
                "memory_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "UUID of memory to classify from stored JohariFingerprint. Mutually exclusive with delta_s/delta_c."
                },
                "embedder_index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 12,
                    "default": 0,
                    "description": "Embedder index (0-12) when using memory_id. Default: 0 (E1 semantic)."
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Classification threshold for quadrant boundaries. Default: 0.5"
                }
            },
            "oneOf": [
                {
                    "required": ["delta_s", "delta_c"],
                    "description": "Direct mode: provide both delta_s and delta_c"
                },
                {
                    "required": ["memory_id"],
                    "description": "Memory mode: provide memory_id to lookup from storage"
                }
            ]
        }),
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_johari_classification_definition_exists() {
        let tools = definitions();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_johari_classification");
    }

    #[test]
    fn test_johari_classification_schema_has_properties() {
        let tools = definitions();
        let schema = &tools[0].input_schema;
        let props = schema.get("properties").expect("properties should exist");

        // Check all expected properties exist
        assert!(props.get("delta_s").is_some(), "delta_s property missing");
        assert!(props.get("delta_c").is_some(), "delta_c property missing");
        assert!(props.get("memory_id").is_some(), "memory_id property missing");
        assert!(
            props.get("embedder_index").is_some(),
            "embedder_index property missing"
        );
        assert!(props.get("threshold").is_some(), "threshold property missing");
    }

    #[test]
    fn test_delta_s_bounds() {
        let tools = definitions();
        let props = tools[0]
            .input_schema
            .get("properties")
            .unwrap();
        let delta_s = props.get("delta_s").unwrap();

        assert_eq!(delta_s.get("minimum").unwrap().as_f64().unwrap(), 0.0);
        assert_eq!(delta_s.get("maximum").unwrap().as_f64().unwrap(), 1.0);
        assert_eq!(delta_s.get("type").unwrap().as_str().unwrap(), "number");
    }

    #[test]
    fn test_delta_c_bounds() {
        let tools = definitions();
        let props = tools[0]
            .input_schema
            .get("properties")
            .unwrap();
        let delta_c = props.get("delta_c").unwrap();

        assert_eq!(delta_c.get("minimum").unwrap().as_f64().unwrap(), 0.0);
        assert_eq!(delta_c.get("maximum").unwrap().as_f64().unwrap(), 1.0);
        assert_eq!(delta_c.get("type").unwrap().as_str().unwrap(), "number");
    }

    #[test]
    fn test_embedder_index_bounds() {
        let tools = definitions();
        let props = tools[0]
            .input_schema
            .get("properties")
            .unwrap();
        let embedder_index = props.get("embedder_index").unwrap();

        assert_eq!(embedder_index.get("minimum").unwrap().as_i64().unwrap(), 0);
        assert_eq!(embedder_index.get("maximum").unwrap().as_i64().unwrap(), 12);
        assert_eq!(embedder_index.get("default").unwrap().as_i64().unwrap(), 0);
    }

    #[test]
    fn test_threshold_bounds_and_default() {
        let tools = definitions();
        let props = tools[0]
            .input_schema
            .get("properties")
            .unwrap();
        let threshold = props.get("threshold").unwrap();

        assert_eq!(threshold.get("minimum").unwrap().as_f64().unwrap(), 0.0);
        assert_eq!(threshold.get("maximum").unwrap().as_f64().unwrap(), 1.0);
        assert_eq!(threshold.get("default").unwrap().as_f64().unwrap(), 0.5);
    }

    #[test]
    fn test_memory_id_uuid_format() {
        let tools = definitions();
        let props = tools[0]
            .input_schema
            .get("properties")
            .unwrap();
        let memory_id = props.get("memory_id").unwrap();

        assert_eq!(memory_id.get("type").unwrap().as_str().unwrap(), "string");
        assert_eq!(memory_id.get("format").unwrap().as_str().unwrap(), "uuid");
    }

    #[test]
    fn test_schema_has_oneof_constraint() {
        let tools = definitions();
        let schema = &tools[0].input_schema;
        let one_of = schema.get("oneOf").expect("oneOf should exist");
        let one_of_array = one_of.as_array().unwrap();

        // Two modes: direct (delta_s + delta_c) or memory (memory_id)
        assert_eq!(one_of_array.len(), 2);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let tools = definitions();
        let json_str = serde_json::to_string(&tools).expect("Serialization failed");
        assert!(json_str.contains("get_johari_classification"));
        assert!(json_str.contains("inputSchema"));
        assert!(json_str.contains("delta_s"));
        assert!(json_str.contains("delta_c"));
        assert!(json_str.contains("memory_id"));
    }

    #[test]
    fn test_tool_description_mentions_constitution() {
        let tools = definitions();
        let desc = &tools[0].description;
        assert!(
            desc.contains("Constitution"),
            "Description should reference constitution"
        );
        assert!(
            desc.contains("utl.johari"),
            "Description should reference utl.johari section"
        );
    }
}
