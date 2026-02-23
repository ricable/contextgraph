//! RVF tool definitions for MCP server.
//!
//! Provides tools for interacting with RVF cognitive containers:
//! - cg_rvf_store: Store vectors in RVF format
//! - cg_rvf_search: Search vectors with progressive recall
//! - cg_rvf_derive: Create COW branch from parent store
//! - cg_rvf_status: Get RVF store status and metrics

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns RVF tool definitions (4 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // cg_rvf_store - Store vectors in RVF format
        ToolDefinition::new(
            "cg_rvf_store",
            "Store vectors in RVF (RuVector Format) cognitive container with dual-write support. \
             Optionally writes to both usearch HNSW and RVF for progressive recall capability.",
            json!({
                "type": "object",
                "properties": {
                    "vectors": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": { "type": "number" }
                        },
                        "description": "Array of vectors to store"
                    },
                    "ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional array of IDs for the vectors"
                    },
                    "metadata": {
                        "type": "array",
                        "items": { "type": "object" },
                        "description": "Optional metadata for each vector"
                    },
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "RVF namespace"
                    },
                    "dual_write": {
                        "type": "boolean",
                        "default": false,
                        "description": "Enable dual-write to both usearch and RVF"
                    }
                },
                "required": ["vectors"],
                "additionalProperties": false
            }),
        ),
        // cg_rvf_search - Search vectors with progressive recall
        ToolDefinition::new(
            "cg_rvf_search",
            "Search vectors in RVF store with progressive recall (Layer A: 70%, B: 85%, C: 95%). \
             Uses SONA confidence scoring to automatically escalate layers if needed.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Query vector"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "Number of results to return"
                    },
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "RVF namespace"
                    },
                    "prefer_rvf": {
                        "type": "boolean",
                        "default": true,
                        "description": "Prefer RVF over usearch fallback"
                    },
                    "initial_layer": {
                        "type": "string",
                        "enum": ["layer_a", "layer_b", "layer_c"],
                        "default": "layer_a",
                        "description": "Initial progressive recall layer"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        ),
        // cg_rvf_derive - Create COW branch
        ToolDefinition::new(
            "cg_rvf_derive",
            "Create a COW (Copy-on-Write) branch from an existing RVF store. \
             Supports include/exclude filters for multi-tenant isolation.",
            json!({
                "type": "object",
                "properties": {
                    "parent_id": {
                        "type": "string",
                        "description": "Parent RVF store ID"
                    },
                    "tenant_id": {
                        "type": "string",
                        "description": "Tenant ID for the derived branch"
                    },
                    "filter_type": {
                        "type": "string",
                        "enum": ["include", "exclude"],
                        "default": "include",
                        "description": "Filter type: include only specified IDs, or exclude"
                    },
                    "filter_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "IDs to include/exclude based on filter_type"
                    }
                },
                "required": ["parent_id", "tenant_id"],
                "additionalProperties": false
            }),
        ),
        // cg_rvf_status - Get RVF status
        ToolDefinition::new(
            "cg_rvf_status",
            "Get RVF cognitive container status including vector count, segment statistics, \
             SONA learning state, and progressive recall layer.",
            json!({
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "RVF namespace"
                    }
                },
                "required": [],
                "additionalProperties": false
            }),
        ),
    ]
}
