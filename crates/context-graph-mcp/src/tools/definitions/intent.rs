//! Intent tool definitions (search_by_intent, find_contextual_matches).
//!
//! E10 Priority 1 Enhancement: Leverage asymmetric E10 embeddings for intent-aware retrieval.
//!
//! Constitution Compliance:
//! - ARCH-15: Uses asymmetric E10 with separate intent/context encodings
//! - E10 ENHANCES E1 semantic search (not replaces) via blendWithSemantic parameter
//! - Direction modifiers: intent→context=1.2, context→intent=0.8

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns intent tool definitions (2 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // search_by_intent - Find memories with similar intent
        ToolDefinition::new(
            "search_by_intent",
            "Find memories that share similar intent or purpose using asymmetric E10 similarity. \
             Useful for \"what work had the same goal?\" queries. ENHANCES E1 semantic search \
             with intent awareness via blendWithSemantic parameter. Uses 1.2x intent→context boost. \
             Default blend of 0.3 means 70% E1 semantic + 30% E10 intent.",
            json!({
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The intent or goal to search for. Describe what you're trying to accomplish."
                    },
                    "topK": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50, default: 10).",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "minScore": {
                        "type": "number",
                        "description": "Minimum similarity score threshold (0-1, default: 0.2). Results below this are filtered.",
                        "default": 0.2,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "blendWithSemantic": {
                        "type": "number",
                        "description": "Blend weight for E10 intent vs E1 semantic (0-1, default: 0.3). \
                                        0.0 = pure E1 semantic, 1.0 = pure E10 intent. \
                                        Default 0.3 means 70% E1 + 30% E10.",
                        "default": 0.3,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "includeContent": {
                        "type": "boolean",
                        "description": "Include full content text in results (default: false).",
                        "default": false
                    }
                },
                "additionalProperties": false
            }),
        ),
        // find_contextual_matches - Find memories relevant to a context
        ToolDefinition::new(
            "find_contextual_matches",
            "Find memories relevant to a given context or situation using E10 context embeddings. \
             Use for \"what's relevant to this situation?\" queries. ENHANCES E1 semantic search \
             with contextual awareness. Uses 0.8x context→intent dampening per asymmetric pattern. \
             Default blend of 0.3 means 70% E1 semantic + 30% E10 context.",
            json!({
                "type": "object",
                "required": ["context"],
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The context or situation to find relevant memories for. Describe the current situation."
                    },
                    "topK": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50, default: 10).",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "minScore": {
                        "type": "number",
                        "description": "Minimum similarity score threshold (0-1, default: 0.2). Results below this are filtered.",
                        "default": 0.2,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "blendWithSemantic": {
                        "type": "number",
                        "description": "Blend weight for E10 context vs E1 semantic (0-1, default: 0.3). \
                                        0.0 = pure E1 semantic, 1.0 = pure E10 context. \
                                        Default 0.3 means 70% E1 + 30% E10.",
                        "default": 0.3,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "includeContent": {
                        "type": "boolean",
                        "description": "Include full content text in results (default: false).",
                        "default": false
                    }
                },
                "additionalProperties": false
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_tool_count() {
        assert_eq!(definitions().len(), 2);
    }

    #[test]
    fn test_search_by_intent_schema() {
        let tools = definitions();
        let search = tools.iter().find(|t| t.name == "search_by_intent").unwrap();

        // Check required fields
        let required = search
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(required.contains(&json!("query")));

        // Check properties
        let props = search
            .input_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(props.contains_key("query"));
        assert!(props.contains_key("topK"));
        assert!(props.contains_key("minScore"));
        assert!(props.contains_key("blendWithSemantic"));
        assert!(props.contains_key("includeContent"));
    }

    #[test]
    fn test_search_by_intent_defaults() {
        let tools = definitions();
        let search = tools.iter().find(|t| t.name == "search_by_intent").unwrap();

        let props = search
            .input_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();

        // Verify defaults
        assert_eq!(props["topK"]["default"], 10);
        assert_eq!(props["minScore"]["default"], 0.2);
        assert_eq!(props["blendWithSemantic"]["default"], 0.3);
        assert_eq!(props["includeContent"]["default"], false);
    }

    #[test]
    fn test_find_contextual_matches_schema() {
        let tools = definitions();
        let find = tools
            .iter()
            .find(|t| t.name == "find_contextual_matches")
            .unwrap();

        // Check required fields
        let required = find
            .input_schema
            .get("required")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(required.contains(&json!("context")));

        // Check properties
        let props = find
            .input_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(props.contains_key("context"));
        assert!(props.contains_key("topK"));
        assert!(props.contains_key("minScore"));
        assert!(props.contains_key("blendWithSemantic"));
        assert!(props.contains_key("includeContent"));
    }

    #[test]
    fn test_find_contextual_matches_defaults() {
        let tools = definitions();
        let find = tools
            .iter()
            .find(|t| t.name == "find_contextual_matches")
            .unwrap();

        let props = find
            .input_schema
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();

        // Verify defaults
        assert_eq!(props["topK"]["default"], 10);
        assert_eq!(props["minScore"]["default"], 0.2);
        assert_eq!(props["blendWithSemantic"]["default"], 0.3);
        assert_eq!(props["includeContent"]["default"], false);
    }

    #[test]
    fn test_tool_descriptions_mention_e10() {
        let tools = definitions();

        for tool in &tools {
            // Both tools should reference E10 or intent/context
            assert!(
                tool.description.contains("E10") || tool.description.contains("intent"),
                "Tool {} should mention E10 or intent",
                tool.name
            );
        }
    }

    #[test]
    fn test_tool_descriptions_mention_enhances() {
        let tools = definitions();

        for tool in &tools {
            // Both tools should mention ENHANCES (E10 enhances E1, doesn't replace)
            assert!(
                tool.description.contains("ENHANCES"),
                "Tool {} should mention ENHANCES (E10 enhances E1)",
                tool.name
            );
        }
    }

    #[test]
    fn test_blend_with_semantic_bounds() {
        let tools = definitions();

        for tool in &tools {
            let props = tool
                .input_schema
                .get("properties")
                .unwrap()
                .as_object()
                .unwrap();

            let blend = &props["blendWithSemantic"];
            assert_eq!(blend["minimum"], 0);
            assert_eq!(blend["maximum"], 1);
        }
    }

    #[test]
    fn test_direction_modifiers_documented() {
        let tools = definitions();

        // search_by_intent should mention 1.2x boost
        let search = tools.iter().find(|t| t.name == "search_by_intent").unwrap();
        assert!(
            search.description.contains("1.2"),
            "search_by_intent should document 1.2x intent→context boost"
        );

        // find_contextual_matches should mention 0.8x dampening
        let find = tools
            .iter()
            .find(|t| t.name == "find_contextual_matches")
            .unwrap();
        assert!(
            find.description.contains("0.8"),
            "find_contextual_matches should document 0.8x context→intent dampening"
        );
    }
}
