//! MCP tool result and request-parsing helpers.

use serde::de::DeserializeOwned;
use serde_json::json;

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::super::Handlers;
use super::validate::{Validate, ValidateInto};

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

    /// Parse JSON args into a typed DTO and run `validate() -> Result<(), String>`.
    ///
    /// Eliminates the repeated parse+validate boilerplate across all handler
    /// methods whose DTOs implement [`Validate`].
    ///
    /// Returns `Ok(request)` on success, or an MCP error `JsonRpcResponse`
    /// on parse/validation failure.
    pub(crate) fn parse_request<T: DeserializeOwned + Validate>(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
        tool_name: &str,
    ) -> Result<T, JsonRpcResponse> {
        let request: T = serde_json::from_value(args).map_err(|e| {
            tracing::error!("[{}] Invalid request: {}", tool_name, e);
            self.tool_error(id.clone(), &format!("Invalid request: {}", e))
        })?;

        request.validate().map_err(|e| {
            tracing::error!("[{}] Validation failed: {}", tool_name, e);
            self.tool_error(id.clone(), &format!("Invalid request: {}", e))
        })?;

        Ok(request)
    }

    /// Embed a query string using all 13 embedders and return the fingerprint.
    ///
    /// Eliminates the repeated embed+error-handling boilerplate across ~25 search
    /// handlers. Returns the `SemanticFingerprint` on success, or an MCP error
    /// `JsonRpcResponse` on embedding failure.
    pub(crate) async fn embed_query(
        &self,
        id: Option<JsonRpcId>,
        query: &str,
        tool_name: &str,
    ) -> Result<context_graph_core::types::fingerprint::SemanticFingerprint, JsonRpcResponse> {
        self.multi_array_provider
            .embed_all(query)
            .await
            .map(|output| output.fingerprint)
            .map_err(|e| {
                tracing::error!("[{}] Embedding failed: {}", tool_name, e);
                self.tool_error(id, &format!("Embedding failed: {}", e))
            })
    }

    /// Parse JSON args into a typed DTO and run `validate() -> Result<Output, String>`.
    ///
    /// Eliminates the repeated parse+validate boilerplate across all handler
    /// methods whose DTOs implement [`ValidateInto`] (i.e. validation produces
    /// a parsed value such as `Uuid`, `Vec<Uuid>`, or `(Uuid, Uuid)`).
    ///
    /// Returns `Ok((request, validated_output))` on success, or an MCP error
    /// `JsonRpcResponse` on parse/validation failure.
    pub(crate) fn parse_request_validated<T: DeserializeOwned + ValidateInto>(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
        tool_name: &str,
    ) -> Result<(T, T::Output), JsonRpcResponse> {
        let request: T = serde_json::from_value(args).map_err(|e| {
            tracing::error!("[{}] Invalid request: {}", tool_name, e);
            self.tool_error(id.clone(), &format!("Invalid request: {}", e))
        })?;

        let output = request.validate().map_err(|e| {
            tracing::error!("[{}] Validation failed: {}", tool_name, e);
            self.tool_error(id.clone(), &format!("Invalid request: {}", e))
        })?;

        Ok((request, output))
    }
}
