//! MCP tool definitions following the MCP 2024-11-05 protocol specification.
//!
//! This module defines the tools available through the MCP server's `tools/list`
//! and `tools/call` endpoints.

use serde::{Deserialize, Serialize};
use serde_json::json;

/// MCP tool definition following the protocol specification.
///
/// Each tool has a name, description, and JSON Schema for input validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name
    pub name: String,

    /// Human-readable description of what the tool does
    pub description: String,

    /// JSON Schema defining the tool's input parameters
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Get all tool definitions for the `tools/list` response.
///
/// Returns the complete list of MCP tools exposed by the Context Graph server.
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // inject_context - primary context injection tool
        ToolDefinition::new(
            "inject_context",
            "Inject context into the knowledge graph with UTL processing. \
             Analyzes content for learning potential and stores with computed metrics.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to inject into the knowledge graph"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this context is relevant and should be stored"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "default": "text",
                        "description": "The type/modality of the content"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    }
                },
                "required": ["content", "rationale"]
            }),
        ),

        // store_memory - store a memory node directly
        ToolDefinition::new(
            "store_memory",
            "Store a memory node directly in the knowledge graph without UTL processing.",
            json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "default": "text",
                        "description": "The type/modality of the content"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional tags for categorization"
                    }
                },
                "required": ["content"]
            }),
        ),

        // get_memetic_status - get UTL metrics and system state
        ToolDefinition::new(
            "get_memetic_status",
            "Get current system status with LIVE UTL metrics from the UtlProcessor: \
             entropy (novelty), coherence (understanding), learning score (magnitude), \
             Johari quadrant classification, consolidation phase, and suggested action. \
             Also returns node count and 5-layer bio-nervous system status.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // get_graph_manifest - describe the 5-layer architecture
        ToolDefinition::new(
            "get_graph_manifest",
            "Get the 5-layer bio-nervous system architecture description and current layer statuses.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // search_graph - semantic search
        ToolDefinition::new(
            "search_graph",
            "Search the knowledge graph using semantic similarity. \
             Returns nodes matching the query with relevance scores.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "topK": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Maximum number of results to return"
                    },
                    "minSimilarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.0,
                        "description": "Minimum similarity threshold [0.0, 1.0]"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
                        "description": "Filter results by modality"
                    }
                },
                "required": ["query"]
            }),
        ),
        // utl_status - query UTL system state
        ToolDefinition::new(
            "utl_status",
            "Query current UTL (Unified Theory of Learning) system state including lifecycle phase, \
             entropy, coherence, learning score, Johari quadrant, and consolidation phase.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // get_consciousness_state - GWT consciousness state (TASK-GWT-001)
        ToolDefinition::new(
            "get_consciousness_state",
            "Get current consciousness state including Kuramoto sync (r), consciousness level (C), \
             meta-cognitive score, differentiation, workspace status, and identity coherence. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for consciousness tracking (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // get_kuramoto_sync - Kuramoto oscillator network synchronization (TASK-GWT-001)
        ToolDefinition::new(
            "get_kuramoto_sync",
            "Get Kuramoto oscillator network synchronization state including order parameter (r), \
             mean phase (psi), all 13 oscillator phases, natural frequencies, and coupling strength. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // get_workspace_status - Global Workspace status (TASK-GWT-001)
        ToolDefinition::new(
            "get_workspace_status",
            "Get Global Workspace status including active memory, competing candidates, \
             broadcast state, and coherence threshold. Returns WTA selection details. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // get_ego_state - Self-Ego Node state (TASK-GWT-001)
        ToolDefinition::new(
            "get_ego_state",
            "Get Self-Ego Node state including purpose vector (13D), identity continuity, \
             coherence with actions, and trajectory length. Used for identity monitoring. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional, uses default if not provided)"
                    }
                },
                "required": []
            }),
        ),

        // trigger_workspace_broadcast - Trigger WTA selection (TASK-GWT-001)
        ToolDefinition::new(
            "trigger_workspace_broadcast",
            "Trigger winner-take-all workspace broadcast with a specific memory. \
             Forces memory into workspace competition. Requires write lock on workspace. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of memory to broadcast into workspace"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "Importance score for the memory [0.0, 1.0]"
                    },
                    "alignment": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "North star alignment score [0.0, 1.0]"
                    },
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force broadcast even if below coherence threshold"
                    }
                },
                "required": ["memory_id"]
            }),
        ),

        // adjust_coupling - Adjust Kuramoto coupling strength (TASK-GWT-001)
        ToolDefinition::new(
            "adjust_coupling",
            "Adjust Kuramoto oscillator network coupling strength K. \
             Higher K leads to faster synchronization. K is clamped to [0, 10]. \
             Returns old and new K values plus predicted order parameter r. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "new_K": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "New coupling strength K (clamped to [0, 10])"
                    }
                },
                "required": ["new_K"]
            }),
        ),
    ]
}

/// Tool names as constants for dispatch matching.
pub mod tool_names {
    pub const INJECT_CONTEXT: &str = "inject_context";
    pub const STORE_MEMORY: &str = "store_memory";
    pub const GET_MEMETIC_STATUS: &str = "get_memetic_status";
    pub const GET_GRAPH_MANIFEST: &str = "get_graph_manifest";
    pub const SEARCH_GRAPH: &str = "search_graph";
    pub const UTL_STATUS: &str = "utl_status";
    /// TASK-GWT-001: Get consciousness state from GWT/Kuramoto system
    pub const GET_CONSCIOUSNESS_STATE: &str = "get_consciousness_state";
    /// TASK-GWT-001: Get Kuramoto oscillator network synchronization state
    pub const GET_KURAMOTO_SYNC: &str = "get_kuramoto_sync";
    /// TASK-GWT-001: Get Global Workspace status (active memory, competing, broadcast)
    pub const GET_WORKSPACE_STATUS: &str = "get_workspace_status";
    /// TASK-GWT-001: Get Self-Ego Node state (purpose vector, identity continuity)
    pub const GET_EGO_STATE: &str = "get_ego_state";
    /// TASK-GWT-001: Trigger workspace broadcast with a memory
    pub const TRIGGER_WORKSPACE_BROADCAST: &str = "trigger_workspace_broadcast";
    /// TASK-GWT-001: Adjust Kuramoto coupling strength K
    pub const ADJUST_COUPLING: &str = "adjust_coupling";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tool_definitions() {
        let tools = get_tool_definitions();
        // 6 original + 6 GWT tools = 12 total
        assert_eq!(tools.len(), 12);

        let tool_names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        // Original 6 tools
        assert!(tool_names.contains(&"inject_context"));
        assert!(tool_names.contains(&"store_memory"));
        assert!(tool_names.contains(&"get_memetic_status"));
        assert!(tool_names.contains(&"get_graph_manifest"));
        assert!(tool_names.contains(&"search_graph"));
        assert!(tool_names.contains(&"utl_status"));
        // GWT tools (TASK-GWT-001)
        assert!(tool_names.contains(&"get_consciousness_state"));
        assert!(tool_names.contains(&"get_kuramoto_sync"));
        assert!(tool_names.contains(&"get_workspace_status"));
        assert!(tool_names.contains(&"get_ego_state"));
        assert!(tool_names.contains(&"trigger_workspace_broadcast"));
        assert!(tool_names.contains(&"adjust_coupling"));
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tools = get_tool_definitions();
        let json = serde_json::to_string(&tools).unwrap();
        assert!(json.contains("inject_context"));
        assert!(json.contains("inputSchema"));
    }

    #[test]
    fn test_inject_context_schema() {
        let tools = get_tool_definitions();
        let inject = tools.iter().find(|t| t.name == "inject_context").unwrap();

        let schema = &inject.input_schema;
        let required = schema.get("required").unwrap().as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("content")));
        assert!(required.iter().any(|v| v.as_str() == Some("rationale")));
    }
}
