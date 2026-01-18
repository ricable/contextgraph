//! Tool definitions per PRD v6 Section 10.
//!
//! 12 tools exposed:
//! - Core (5): inject_context, store_memory, get_memetic_status, search_graph, trigger_consolidation
//! - Curation (3): merge_concepts, forget_concept, boost_importance
//! - Topic (4): get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts

pub(crate) mod core;
pub(crate) mod curation;
pub(crate) mod merge;
pub(crate) mod topic;

use crate::tools::types::ToolDefinition;

/// Get all tool definitions for the `tools/list` response.
///
/// Per PRD v6, 12 tools are exposed:
/// - Core: inject_context, store_memory, get_memetic_status, search_graph, trigger_consolidation
/// - Curation: merge_concepts, forget_concept, boost_importance
/// - Topic: get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    let mut tools = Vec::with_capacity(12);

    // Core tools (5)
    tools.extend(core::definitions());

    // Merge tool (1 - part of curation)
    tools.extend(merge::definitions());

    // Curation tools (2)
    tools.extend(curation::definitions());

    // Topic tools (4)
    tools.extend(topic::definitions());

    tools
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_tool_count() {
        let tools = get_tool_definitions();
        assert_eq!(tools.len(), 12, "PRD v6 requires exactly 12 tools");
    }

    #[test]
    fn test_all_tool_names_present() {
        let tools = get_tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        // Core tools (5)
        assert!(names.contains(&"inject_context"));
        assert!(names.contains(&"store_memory"));
        assert!(names.contains(&"get_memetic_status"));
        assert!(names.contains(&"search_graph"));
        assert!(names.contains(&"trigger_consolidation"));

        // Curation tools (3)
        assert!(names.contains(&"merge_concepts"));
        assert!(names.contains(&"forget_concept"));
        assert!(names.contains(&"boost_importance"));

        // Topic tools (4)
        assert!(names.contains(&"get_topic_portfolio"));
        assert!(names.contains(&"get_topic_stability"));
        assert!(names.contains(&"detect_topics"));
        assert!(names.contains(&"get_divergence_alerts"));
    }

    #[test]
    fn test_no_duplicate_tools() {
        let tools = get_tool_definitions();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        let len_before = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), len_before, "No duplicate tool names allowed");
    }

    #[test]
    fn test_all_tools_have_descriptions() {
        let tools = get_tool_definitions();
        for tool in &tools {
            assert!(
                !tool.description.is_empty(),
                "Tool {} missing description",
                tool.name
            );
        }
    }

    #[test]
    fn test_all_tools_have_schemas() {
        let tools = get_tool_definitions();
        for tool in &tools {
            assert!(
                tool.input_schema.get("type").is_some(),
                "Tool {} missing input_schema type",
                tool.name
            );
        }
    }

    #[test]
    fn test_tool_definitions_json_serialization() {
        let tools = get_tool_definitions();
        let json_str = serde_json::to_string_pretty(&tools).expect("Failed to serialize tools");

        // Verify key elements are present in serialized output
        assert!(json_str.contains("inject_context"));
        assert!(json_str.contains("store_memory"));
        assert!(json_str.contains("get_memetic_status"));
        assert!(json_str.contains("search_graph"));
        assert!(json_str.contains("trigger_consolidation"));
        assert!(json_str.contains("merge_concepts"));
        assert!(json_str.contains("forget_concept"));
        assert!(json_str.contains("boost_importance"));
        assert!(json_str.contains("get_topic_portfolio"));
        assert!(json_str.contains("get_topic_stability"));
        assert!(json_str.contains("detect_topics"));
        assert!(json_str.contains("get_divergence_alerts"));
        assert!(json_str.contains("inputSchema"));

        println!("Tool definitions JSON serialization verified");
    }

    #[test]
    fn test_core_tools_count() {
        let core_tools = core::definitions();
        assert_eq!(core_tools.len(), 5, "Should have 5 core tools");
    }

    #[test]
    fn test_merge_tools_count() {
        let merge_tools = merge::definitions();
        assert_eq!(merge_tools.len(), 1, "Should have 1 merge tool");
    }

    #[test]
    fn test_curation_tools_count() {
        let curation_tools = curation::definitions();
        assert_eq!(curation_tools.len(), 2, "Should have 2 curation tools");
    }

    #[test]
    fn test_topic_tools_count() {
        let topic_tools = topic::definitions();
        assert_eq!(topic_tools.len(), 4, "Should have 4 topic tools");
    }

    #[test]
    fn test_combined_count_matches_total() {
        let core_count = core::definitions().len();
        let merge_count = merge::definitions().len();
        let curation_count = curation::definitions().len();
        let topic_count = topic::definitions().len();

        let expected_total = core_count + merge_count + curation_count + topic_count;
        let actual_total = get_tool_definitions().len();

        assert_eq!(
            actual_total, expected_total,
            "Total tools ({}) should equal sum of submodules ({})",
            actual_total, expected_total
        );
        assert_eq!(actual_total, 12, "Should have exactly 12 tools total");
    }
}
