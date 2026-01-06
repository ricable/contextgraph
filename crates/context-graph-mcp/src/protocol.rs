//! MCP JSON-RPC protocol types.

use serde::{Deserialize, Serialize};

use context_graph_core::types::CognitivePulse;

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<JsonRpcId>,
    pub method: String,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<JsonRpcId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    /// Cognitive Pulse header (Context Graph extension).
    #[serde(rename = "X-Cognitive-Pulse", skip_serializing_if = "Option::is_none")]
    pub cognitive_pulse: Option<CognitivePulse>,
}

/// JSON-RPC ID (can be string, number, or null).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum JsonRpcId {
    String(String),
    Number(i64),
}

/// JSON-RPC error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: Option<JsonRpcId>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
            cognitive_pulse: None,
        }
    }

    /// Create an error response.
    pub fn error(id: Option<JsonRpcId>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
            cognitive_pulse: None,
        }
    }

    /// Add cognitive pulse header.
    pub fn with_pulse(mut self, pulse: CognitivePulse) -> Self {
        self.cognitive_pulse = Some(pulse);
        self
    }
}

/// Standard JSON-RPC error codes.
///
/// TASK-S001: Added teleological-specific error codes for TeleologicalMemoryStore operations.
#[allow(dead_code)]
pub mod error_codes {
    // Standard JSON-RPC 2.0 error codes
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;

    // Context Graph specific error codes (-32001 to -32099)
    pub const FEATURE_DISABLED: i32 = -32001;
    pub const NODE_NOT_FOUND: i32 = -32002;
    pub const PAYLOAD_TOO_LARGE: i32 = -32003;
    pub const STORAGE_ERROR: i32 = -32004;
    pub const EMBEDDING_ERROR: i32 = -32005;
    pub const TOOL_NOT_FOUND: i32 = -32006;
    pub const LAYER_TIMEOUT: i32 = -32007;

    // Teleological-specific error codes (-32010 to -32019) - TASK-S001
    /// TeleologicalFingerprint not found by UUID
    pub const FINGERPRINT_NOT_FOUND: i32 = -32010;
    /// Multi-array embedding provider not ready (13 embedders)
    pub const EMBEDDER_NOT_READY: i32 = -32011;
    /// Purpose vector computation failed
    pub const PURPOSE_COMPUTATION_ERROR: i32 = -32012;
    /// Johari quadrant classification failed
    pub const JOHARI_CLASSIFICATION_ERROR: i32 = -32013;
    /// Sparse search (SPLADE E13) failed
    pub const SPARSE_SEARCH_ERROR: i32 = -32014;
    /// Semantic search (13-embedding) failed
    pub const SEMANTIC_SEARCH_ERROR: i32 = -32015;
    /// Purpose alignment search failed
    pub const PURPOSE_SEARCH_ERROR: i32 = -32016;
    /// Checkpoint/restore operation failed
    pub const CHECKPOINT_ERROR: i32 = -32017;
    /// Batch operation failed
    pub const BATCH_OPERATION_ERROR: i32 = -32018;

    // Goal/alignment specific error codes (-32020 to -32029) - TASK-S003
    /// Goal not found in hierarchy
    pub const GOAL_NOT_FOUND: i32 = -32020;
    /// North Star goal not configured in hierarchy
    pub const NORTH_STAR_NOT_CONFIGURED: i32 = -32021;
    /// Alignment computation failed
    pub const ALIGNMENT_COMPUTATION_ERROR: i32 = -32022;
    /// Goal hierarchy operation failed
    pub const GOAL_HIERARCHY_ERROR: i32 = -32023;

    // Johari-specific error codes (-32030 to -32039) - TASK-S004
    /// Invalid embedder index (must be 0-12)
    pub const JOHARI_INVALID_EMBEDDER_INDEX: i32 = -32030;
    /// Invalid quadrant string (must be open/hidden/blind/unknown)
    pub const JOHARI_INVALID_QUADRANT: i32 = -32031;
    /// Soft classification weights don't sum to 1.0
    pub const JOHARI_INVALID_SOFT_CLASSIFICATION: i32 = -32032;
    /// Transition validation failed
    pub const JOHARI_TRANSITION_ERROR: i32 = -32033;
    /// Batch transition failed (all-or-nothing)
    pub const JOHARI_BATCH_ERROR: i32 = -32034;
}

/// MCP method names.
#[allow(dead_code)]
pub mod methods {
    // MCP lifecycle methods
    pub const INITIALIZE: &str = "initialize";
    pub const SHUTDOWN: &str = "shutdown";

    // MCP tools protocol methods
    pub const TOOLS_LIST: &str = "tools/list";
    pub const TOOLS_CALL: &str = "tools/call";

    // Memory operations
    pub const MEMORY_STORE: &str = "memory/store";
    pub const MEMORY_RETRIEVE: &str = "memory/retrieve";
    pub const MEMORY_SEARCH: &str = "memory/search";
    pub const MEMORY_DELETE: &str = "memory/delete";

    // Search operations (TASK-S002)
    pub const SEARCH_MULTI: &str = "search/multi";
    pub const SEARCH_SINGLE_SPACE: &str = "search/single_space";
    pub const SEARCH_BY_PURPOSE: &str = "search/by_purpose";
    pub const SEARCH_WEIGHT_PROFILES: &str = "search/weight_profiles";

    // Graph operations
    pub const GRAPH_CONNECT: &str = "graph/connect";
    pub const GRAPH_TRAVERSE: &str = "graph/traverse";

    // UTL operations
    pub const UTL_COMPUTE: &str = "utl/compute";
    pub const UTL_METRICS: &str = "utl/metrics";

    // System operations
    pub const SYSTEM_STATUS: &str = "system/status";
    pub const SYSTEM_HEALTH: &str = "system/health";

    // Purpose/goal operations (TASK-S003)
    /// Query memories by 13D purpose vector similarity
    pub const PURPOSE_QUERY: &str = "purpose/query";
    /// Check alignment to North Star goal with threshold classification
    pub const PURPOSE_NORTH_STAR_ALIGNMENT: &str = "purpose/north_star_alignment";
    /// Navigate goal hierarchy (get_children, get_ancestors, get_subtree, get_aligned_memories)
    pub const GOAL_HIERARCHY_QUERY: &str = "goal/hierarchy_query";
    /// Find memories aligned to a specific goal
    pub const GOAL_ALIGNED_MEMORIES: &str = "goal/aligned_memories";
    /// Detect alignment drift in memories
    pub const PURPOSE_DRIFT_CHECK: &str = "purpose/drift_check";
    /// Update the North Star goal
    pub const NORTH_STAR_UPDATE: &str = "purpose/north_star_update";

    // Johari operations (TASK-S004)
    /// Get per-embedder Johari quadrant distribution
    pub const JOHARI_GET_DISTRIBUTION: &str = "johari/get_distribution";
    /// Find memories by quadrant for specific embedder
    pub const JOHARI_FIND_BY_QUADRANT: &str = "johari/find_by_quadrant";
    /// Execute single Johari transition
    pub const JOHARI_TRANSITION: &str = "johari/transition";
    /// Execute batch Johari transitions (atomic)
    pub const JOHARI_TRANSITION_BATCH: &str = "johari/transition_batch";
    /// Cross-space Johari analysis (blind spots, opportunities)
    pub const JOHARI_CROSS_SPACE_ANALYSIS: &str = "johari/cross_space_analysis";
    /// Get transition probability matrix
    pub const JOHARI_TRANSITION_PROBABILITIES: &str = "johari/transition_probabilities";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_request() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(JsonRpcId::Number(1)));
    }

    #[test]
    fn test_success_response() {
        let resp = JsonRpcResponse::success(
            Some(JsonRpcId::Number(1)),
            serde_json::json!({"status": "ok"}),
        );
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_error_response() {
        let resp = JsonRpcResponse::error(
            Some(JsonRpcId::String("req-123".to_string())),
            error_codes::METHOD_NOT_FOUND,
            "Method not found",
        );
        assert!(resp.result.is_none());
        assert_eq!(
            resp.error.as_ref().unwrap().code,
            error_codes::METHOD_NOT_FOUND
        );
    }
}
