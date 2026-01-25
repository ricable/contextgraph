//! MCP tool result helpers.

use serde_json::json;

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

impl Handlers {
    /// MCP-compliant tool result helper.
    ///
    /// Wraps tool output in the required MCP format:
    /// ```json
    /// {
    ///   "content": [{"type": "text", "text": "..."}],
    ///   "isError": false
    /// }
    /// ```
    pub(crate) fn tool_result(
        &self,
        id: Option<JsonRpcId>,
        data: serde_json::Value,
    ) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&data).unwrap_or_else(|_| "{}".to_string())
                }],
                "isError": false
            }),
        )
    }

    /// MCP-compliant tool error helper.
    ///
    /// Returns an error response in MCP format:
    /// ```json
    /// {
    ///   "content": [{"type": "text", "text": "error message"}],
    ///   "isError": true
    /// }
    /// ```
    pub(crate) fn tool_error(&self, id: Option<JsonRpcId>, message: &str) -> JsonRpcResponse {
        JsonRpcResponse::success(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": message
                }],
                "isError": true
            }),
        )
    }
}
