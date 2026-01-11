# TASK-MCP-P2-001: Add MCP Tool Naming Aliases

## Task Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-MCP-P2-001 |
| **Title** | Implement MCP Tool Naming Aliases for Backwards Compatibility |
| **Status** | Ready |
| **Priority** | P2 |
| **Estimated Effort** | 2-3 hours |
| **Parent Spec** | SPEC-MCP-001 |
| **Created** | 2025-01-11 |
| **Gap Reference** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md - Refinement 2 |

---

## 1. Objective

Add alias resolution to the MCP tool registry to enable backwards-compatible invocation using PRD-specified tool names.

**Aliases to implement:**
| PRD Name | Canonical Name |
|----------|----------------|
| `discover_goals` | `discover_sub_goals` |
| `consolidate_memories` | `trigger_consolidation` |

---

## 2. Input Context Files

The following files must be read and understood before implementation:

| File Path | Purpose | Critical Sections |
|-----------|---------|-------------------|
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools.rs` | Tool dispatch logic | `handle_tools_call` method |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/autonomous.rs` | Autonomous tool handlers | `call_discover_sub_goals`, `call_trigger_consolidation` |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools.rs` | Tool definitions | `ToolDefinition` struct, `get_tool_definitions()` |
| `/home/cabdru/contextgraph/crates/context-graph-mcp/src/protocol.rs` | Error codes | `error_codes::TOOL_NOT_FOUND` |

---

## 3. Files to Modify

### 3.1 Primary Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `crates/context-graph-mcp/src/handlers/tools.rs` | Modify | Add alias resolution before tool dispatch |
| `crates/context-graph-mcp/src/tools.rs` | Modify | Add `aliases` field to `ToolDefinition`, update definitions |

### 3.2 New Files

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/aliases.rs` | Alias registry module (optional - can be inline) |

### 3.3 Test Files

| File | Change Type | Description |
|------|-------------|-------------|
| `crates/context-graph-mcp/src/handlers/tools.rs` | Modify | Add tests in inline `#[cfg(test)]` module |
| `crates/context-graph-mcp/tests/mcp_alias_tests.rs` | Create | Integration tests for alias invocation |

---

## 4. Implementation Steps

### Step 1: Add Alias Resolution Module

**Location**: `crates/context-graph-mcp/src/handlers/tools.rs` (inline) or new file

```rust
//! Tool alias resolution for backwards compatibility with PRD names.
//!
//! TASK-MCP-P2-001: Maps PRD-specified names to canonical implementation names.

use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Static alias map: PRD name -> canonical name
/// Add new aliases here as needed.
pub static TOOL_ALIASES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    // Gap Analysis Refinement 2 aliases
    m.insert("discover_goals", "discover_sub_goals");
    m.insert("consolidate_memories", "trigger_consolidation");
    m
});

/// Resolve a tool name to its canonical form.
///
/// If `name` is an alias, returns the canonical name.
/// Otherwise, returns `name` unchanged.
///
/// # Performance
/// O(1) average case via HashMap lookup.
#[inline]
pub fn resolve_tool_alias(name: &str) -> &str {
    TOOL_ALIASES.get(name).copied().unwrap_or(name)
}
```

### Step 2: Integrate Alias Resolution into Tool Dispatch

**Location**: `crates/context-graph-mcp/src/handlers/tools.rs`

**Before** (current code):
```rust
match tool_name {
    "discover_sub_goals" => self.call_discover_sub_goals(id, arguments).await,
    "trigger_consolidation" => self.call_trigger_consolidation(id, arguments).await,
    // ... other tools ...
}
```

**After** (with alias resolution):
```rust
use tracing::debug;

// Resolve alias to canonical name
let canonical_name = resolve_tool_alias(tool_name);

if canonical_name != tool_name {
    debug!(
        alias = %tool_name,
        canonical = %canonical_name,
        "TASK-MCP-P2-001: Resolved tool alias"
    );
}

// Dispatch using canonical name
match canonical_name {
    "discover_sub_goals" => self.call_discover_sub_goals(id, arguments).await,
    "trigger_consolidation" => self.call_trigger_consolidation(id, arguments).await,
    // ... other tools ...
}
```

### Step 3: Enhance ToolDefinition Struct

**Location**: `crates/context-graph-mcp/src/tools.rs`

```rust
/// Tool definition for MCP tools/list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,

    /// Optional aliases for backwards compatibility.
    /// TASK-MCP-P2-001: Added for PRD name support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
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
            aliases: None,
        }
    }

    /// Add aliases to this tool definition.
    /// TASK-MCP-P2-001: Builder pattern for alias assignment.
    pub fn with_aliases(mut self, aliases: Vec<impl Into<String>>) -> Self {
        self.aliases = Some(aliases.into_iter().map(|a| a.into()).collect());
        self
    }
}
```

### Step 4: Update Tool Definitions

**Location**: `crates/context-graph-mcp/src/tools.rs` in `get_tool_definitions()`

```rust
// discover_sub_goals with alias
ToolDefinition::new(
    "discover_sub_goals",
    "Discover potential sub-goals from memory clusters using GoalDiscoveryPipeline. \
     Analyzes stored memories via K-means clustering to find emergent themes. \
     ARCH-03 compliant: works without North Star for autonomous goal discovery. \
     \n\nAlias: `discover_goals` (PRD compatibility)",
    json!({
        "type": "object",
        "properties": {
            "min_confidence": {
                "type": "number",
                "description": "Minimum confidence/coherence for discovered goals (default: 0.6)"
            },
            "max_goals": {
                "type": "integer",
                "description": "Maximum number of goals to discover (default: 5)"
            },
            "parent_goal_id": {
                "type": "string",
                "description": "Parent goal UUID (default: North Star if exists)"
            },
            "memory_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Specific memory UUIDs to analyze"
            },
            "algorithm": {
                "type": "string",
                "enum": ["kmeans", "hdbscan", "spectral"],
                "description": "Clustering algorithm (default: kmeans)"
            }
        }
    })
).with_aliases(vec!["discover_goals"]),

// trigger_consolidation with alias
ToolDefinition::new(
    "trigger_consolidation",
    "Trigger memory consolidation to merge similar memories and reduce redundancy. \
     Uses similarity-based, temporal, or semantic strategies. \
     Analyzes stored fingerprints to find consolidation candidates. \
     \n\nAlias: `consolidate_memories` (PRD compatibility)",
    json!({
        "type": "object",
        "properties": {
            "max_memories": {
                "type": "integer",
                "description": "Maximum memories to process (default: 100)"
            },
            "strategy": {
                "type": "string",
                "enum": ["similarity", "temporal", "semantic"],
                "description": "Consolidation strategy (default: similarity)"
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum similarity threshold for merge (default: 0.85)"
            }
        }
    })
).with_aliases(vec!["consolidate_memories"]),
```

### Step 5: Add Unit Tests

**Location**: `crates/context-graph-mcp/src/handlers/tools.rs`

```rust
#[cfg(test)]
mod alias_tests {
    use super::*;

    #[test]
    fn test_resolve_discover_goals_alias() {
        assert_eq!(resolve_tool_alias("discover_goals"), "discover_sub_goals");
    }

    #[test]
    fn test_resolve_consolidate_memories_alias() {
        assert_eq!(resolve_tool_alias("consolidate_memories"), "trigger_consolidation");
    }

    #[test]
    fn test_canonical_names_unchanged() {
        assert_eq!(resolve_tool_alias("discover_sub_goals"), "discover_sub_goals");
        assert_eq!(resolve_tool_alias("trigger_consolidation"), "trigger_consolidation");
        assert_eq!(resolve_tool_alias("store_memory"), "store_memory");
    }

    #[test]
    fn test_unknown_names_pass_through() {
        assert_eq!(resolve_tool_alias("unknown_tool"), "unknown_tool");
        assert_eq!(resolve_tool_alias(""), "");
    }

    #[test]
    fn test_case_sensitivity() {
        // Aliases are case-sensitive
        assert_eq!(resolve_tool_alias("Discover_Goals"), "Discover_Goals"); // Not resolved
        assert_eq!(resolve_tool_alias("CONSOLIDATE_MEMORIES"), "CONSOLIDATE_MEMORIES"); // Not resolved
    }

    #[test]
    fn test_no_alias_collisions() {
        // Ensure no alias matches a canonical tool name
        let canonical_names = vec![
            "store_memory",
            "retrieve_memory",
            "search_memories",
            "compute_teleological_vector",
            "compute_alignment",
            "get_alignment_drift",
            "trigger_drift_correction",
            "get_pruning_candidates",
            "trigger_consolidation",
            "discover_sub_goals",
            "get_autonomous_status",
            "auto_bootstrap_north_star",
            "set_north_star",
            "get_north_star",
            // ... add all canonical names
        ];

        for (alias, _) in TOOL_ALIASES.iter() {
            assert!(
                !canonical_names.contains(alias),
                "Alias '{}' collides with canonical tool name",
                alias
            );
        }
    }
}
```

### Step 6: Add Integration Tests

**Location**: `crates/context-graph-mcp/tests/mcp_alias_tests.rs`

```rust
//! Integration tests for MCP tool alias resolution.
//! TASK-MCP-P2-001: Verify end-to-end alias invocation.

use context_graph_mcp::protocol::{JsonRpcRequest, JsonRpcResponse};
use serde_json::json;

#[tokio::test]
async fn test_discover_goals_alias_invocation() {
    let server = TestMcpServer::new().await;

    // Call using PRD alias
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "discover_goals",
            "arguments": {
                "min_confidence": 0.7,
                "max_goals": 3
            }
        })),
        id: Some(1.into()),
    };

    let response = server.handle(request).await;

    // Should succeed (not return TOOL_NOT_FOUND)
    assert!(response.error.is_none(), "Alias should resolve successfully");
}

#[tokio::test]
async fn test_consolidate_memories_alias_invocation() {
    let server = TestMcpServer::new().await;

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "consolidate_memories",
            "arguments": {
                "strategy": "similarity"
            }
        })),
        id: Some(1.into()),
    };

    let response = server.handle(request).await;

    assert!(response.error.is_none(), "Alias should resolve successfully");
}

#[tokio::test]
async fn test_tools_list_includes_aliases() {
    let server = TestMcpServer::new().await;

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        method: "tools/list".to_string(),
        params: None,
        id: Some(1.into()),
    };

    let response = server.handle(request).await;
    let tools = response.result.unwrap();
    let tools_array = tools["tools"].as_array().unwrap();

    // Find discover_sub_goals
    let discover_tool = tools_array
        .iter()
        .find(|t| t["name"] == "discover_sub_goals")
        .expect("discover_sub_goals should exist");

    assert_eq!(
        discover_tool["aliases"],
        json!(["discover_goals"]),
        "discover_sub_goals should list discover_goals as alias"
    );

    // Find trigger_consolidation
    let consolidation_tool = tools_array
        .iter()
        .find(|t| t["name"] == "trigger_consolidation")
        .expect("trigger_consolidation should exist");

    assert_eq!(
        consolidation_tool["aliases"],
        json!(["consolidate_memories"]),
        "trigger_consolidation should list consolidate_memories as alias"
    );
}
```

---

## 5. Definition of Done

### 5.1 Code Requirements

- [ ] `TOOL_ALIASES` static map created with both alias mappings
- [ ] `resolve_tool_alias()` function implemented and inlined
- [ ] `handle_tools_call` calls `resolve_tool_alias` before dispatch
- [ ] Debug logging added for alias resolution
- [ ] `ToolDefinition` struct includes optional `aliases` field
- [ ] `with_aliases()` builder method implemented
- [ ] `discover_sub_goals` definition includes `["discover_goals"]` alias
- [ ] `trigger_consolidation` definition includes `["consolidate_memories"]` alias

### 5.2 Test Requirements

- [ ] Unit test: `discover_goals` resolves to `discover_sub_goals`
- [ ] Unit test: `consolidate_memories` resolves to `trigger_consolidation`
- [ ] Unit test: Canonical names pass through unchanged
- [ ] Unit test: Unknown names pass through unchanged
- [ ] Unit test: Case sensitivity verified
- [ ] Unit test: No alias collisions with canonical names
- [ ] Integration test: Alias invocation succeeds end-to-end
- [ ] Integration test: `tools/list` includes alias metadata

### 5.3 Documentation Requirements

- [ ] Inline code comments reference TASK-MCP-P2-001
- [ ] Tool descriptions updated to mention aliases
- [ ] No new documentation files needed (changes are inline)

### 5.4 Build Requirements

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes all new and existing tests
- [ ] `cargo clippy` reports no new warnings
- [ ] `cargo fmt --check` passes

---

## 6. Validation Criteria

### 6.1 Manual Validation

```bash
# Start MCP server
cargo run --bin context-graph-mcp

# Test alias invocation (using MCP client or curl)
# Both should succeed and invoke the same handler:

# PRD alias
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"discover_goals","arguments":{}},"id":1}' | nc localhost 3000

# Canonical name
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"discover_sub_goals","arguments":{}},"id":2}' | nc localhost 3000
```

### 6.2 Automated Validation

```bash
# Run all tests
cargo test --package context-graph-mcp

# Run specific alias tests
cargo test --package context-graph-mcp alias_tests

# Check for regressions
cargo test --package context-graph-mcp -- --ignored
```

### 6.3 Performance Validation

```bash
# Benchmark alias resolution (should be <1us)
cargo bench --package context-graph-mcp alias_resolution
```

---

## 7. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Alias collides with future tool name | High | Low | Validate at compile time; reserve alias names |
| Performance regression from HashMap lookup | Medium | Very Low | Use `once_cell::Lazy` for single initialization |
| Breaking change if alias behavior differs | High | Very Low | Aliases route to exact same handler code |

---

## 8. Rollback Plan

If issues are discovered post-merge:

1. Remove alias resolution call from `handle_tools_call`
2. Revert `ToolDefinition` changes
3. Remove `TOOL_ALIASES` map
4. Run test suite to confirm rollback

Estimated rollback time: 15 minutes

---

## 9. Related Tasks

| Task ID | Relationship | Status |
|---------|--------------|--------|
| TASK-AUTONOMOUS-MCP | Parent implementation | Complete |
| TASK-INTEG-002 | Related integration | Complete |
| SPEC-MCP-001 | Parent specification | Draft |

---

## 10. Appendix: Full Alias Mapping Table

| PRD Name | Canonical Name | Handler Function | Category |
|----------|----------------|------------------|----------|
| `discover_goals` | `discover_sub_goals` | `call_discover_sub_goals` | Autonomous |
| `consolidate_memories` | `trigger_consolidation` | `call_trigger_consolidation` | Autonomous |

Future aliases can be added to this table and the `TOOL_ALIASES` map as needed.
