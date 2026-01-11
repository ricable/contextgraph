# SPEC-MCP-001: MCP Tool Naming Aliases

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-MCP-001 |
| **Title** | MCP Tool Naming Aliases for Backwards Compatibility |
| **Status** | Draft |
| **Priority** | P2 (Minor Refinement) |
| **Owner** | ContextGraph Team |
| **Created** | 2025-01-11 |
| **Last Updated** | 2025-01-11 |
| **Related Specs** | SPEC-MCP-TOOLS-001 |
| **Gap Reference** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md - Refinement 2 |

---

## 1. Overview

### 1.1 Problem Statement

The MCP tool implementation has naming inconsistencies between the PRD specification and actual implementation. This creates integration friction for clients that reference the PRD-specified names but encounter different names at runtime.

| PRD Name | Actual Name | Category |
|----------|-------------|----------|
| `discover_goals` | `discover_sub_goals` | Autonomous Tools |
| `consolidate_memories` | `trigger_consolidation` | Autonomous Tools |

### 1.2 Scope

This specification defines the addition of **tool name aliases** to the MCP tool registry, enabling clients to invoke tools using either:
1. The **canonical name** (current implementation)
2. The **PRD alias** (backwards compatibility)

### 1.3 Success Criteria

- Both `discover_goals` and `discover_sub_goals` invoke the same handler
- Both `consolidate_memories` and `trigger_consolidation` invoke the same handler
- `tools/list` returns both canonical names AND aliases with clear documentation
- Zero breaking changes to existing integrations
- Test coverage for alias resolution

---

## 2. Requirements

### 2.1 Functional Requirements

#### FR-MCP-001: Alias Resolution in Tool Dispatch
| ID | Description | Priority |
|----|-------------|----------|
| FR-001 | The `tools/call` handler MUST resolve tool aliases before dispatching | Must-Have |
| FR-002 | Alias resolution MUST map PRD names to canonical implementation names | Must-Have |
| FR-003 | Alias resolution MUST be case-sensitive | Must-Have |
| FR-004 | Unknown tool names MUST return error code `-32004` (TOOL_NOT_FOUND) | Must-Have |

#### FR-MCP-002: Tool Listing with Aliases
| ID | Description | Priority |
|----|-------------|----------|
| FR-010 | `tools/list` MUST include alias information in tool metadata | Should-Have |
| FR-011 | Each tool definition MAY include an `aliases` field (array of strings) | Should-Have |
| FR-012 | Aliases SHOULD be documented in tool descriptions | Should-Have |

#### FR-MCP-003: Logging and Observability
| ID | Description | Priority |
|----|-------------|----------|
| FR-020 | Alias resolution MUST log when an alias is used (debug level) | Should-Have |
| FR-021 | Metrics SHOULD track alias usage vs canonical name usage | Nice-to-Have |

### 2.2 Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-001 | Performance | Alias resolution overhead < 1us | p99 latency |
| NFR-002 | Maintainability | Alias mappings centralized in single location | Code review |
| NFR-003 | Testability | All aliases covered by unit tests | Test coverage |

---

## 3. Technical Design

### 3.1 Alias Registry Structure

```rust
/// Tool alias registry for backwards compatibility
/// Maps PRD names to canonical implementation names
pub mod tool_aliases {
    use std::collections::HashMap;
    use once_cell::sync::Lazy;

    /// Static alias map: PRD name -> canonical name
    pub static ALIAS_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
        let mut m = HashMap::new();
        // PRD Refinement 2 aliases
        m.insert("discover_goals", "discover_sub_goals");
        m.insert("consolidate_memories", "trigger_consolidation");
        m
    });

    /// Resolve a tool name, returning canonical name if alias exists
    #[inline]
    pub fn resolve(name: &str) -> &str {
        ALIAS_MAP.get(name).copied().unwrap_or(name)
    }
}
```

### 3.2 Integration Point

The alias resolution integrates into `handle_tools_call` in `handlers/tools.rs`:

```rust
pub(super) async fn handle_tools_call(
    &self,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcResponse {
    // ... parameter extraction ...

    let tool_name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                "Missing 'name' parameter in tools/call",
            );
        }
    };

    // Resolve alias to canonical name
    let canonical_name = tool_aliases::resolve(tool_name);

    if canonical_name != tool_name {
        debug!(
            alias = tool_name,
            canonical = canonical_name,
            "Resolved tool alias to canonical name"
        );
    }

    // Dispatch using canonical_name instead of tool_name
    match canonical_name {
        // ... existing dispatch ...
    }
}
```

### 3.3 Tool Definition Enhancement

```rust
/// Enhanced tool definition with optional aliases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,

    /// Optional aliases for backwards compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
}
```

### 3.4 Affected Tool Definitions

#### 3.4.1 discover_sub_goals (Canonical)

```rust
ToolDefinition::new(
    "discover_sub_goals",
    "Discover potential sub-goals from memory clusters. Analyzes stored memories to find \
     emergent themes and patterns that could become strategic or tactical goals. \
     Helps evolve the goal hierarchy based on actual content. \
     \n\nAlias: discover_goals (PRD compatibility)",
    // ... schema unchanged ...
).with_aliases(vec!["discover_goals"])
```

#### 3.4.2 trigger_consolidation (Canonical)

```rust
ToolDefinition::new(
    "trigger_consolidation",
    "Trigger memory consolidation to merge similar memories and reduce redundancy. \
     Uses similarity-based, temporal, or semantic strategies to identify merge candidates. \
     Helps optimize memory storage and improve retrieval efficiency. \
     \n\nAlias: consolidate_memories (PRD compatibility)",
    // ... schema unchanged ...
).with_aliases(vec!["consolidate_memories"])
```

---

## 4. Acceptance Criteria

### 4.1 Gherkin Scenarios

```gherkin
Feature: MCP Tool Naming Aliases

  Background:
    Given the MCP server is running
    And the tool registry is initialized

  Scenario: Invoke tool using canonical name
    When I call tools/call with name "discover_sub_goals"
    Then the discover_sub_goals handler is invoked
    And no alias resolution log is emitted

  Scenario: Invoke tool using PRD alias
    When I call tools/call with name "discover_goals"
    Then the discover_sub_goals handler is invoked
    And a debug log is emitted: "Resolved tool alias to canonical name"

  Scenario: Invoke consolidation using PRD alias
    When I call tools/call with name "consolidate_memories"
    Then the trigger_consolidation handler is invoked
    And a debug log is emitted: "Resolved tool alias to canonical name"

  Scenario: Tool list includes alias information
    When I call tools/list
    Then the response includes "discover_sub_goals" with aliases ["discover_goals"]
    And the response includes "trigger_consolidation" with aliases ["consolidate_memories"]

  Scenario: Unknown tool returns error
    When I call tools/call with name "nonexistent_tool"
    Then the response contains error code -32004
    And the error message contains "Unknown tool: nonexistent_tool"
```

### 4.2 Test Coverage Requirements

| Test Type | Scope | Coverage Target |
|-----------|-------|-----------------|
| Unit | `tool_aliases::resolve()` | 100% |
| Unit | `ToolDefinition::with_aliases()` | 100% |
| Integration | `handle_tools_call` alias dispatch | All aliases |
| Integration | `tools/list` alias serialization | All aliases |

---

## 5. Edge Cases

### 5.1 Alias Collision Prevention

**Scenario**: An alias matches an existing canonical tool name.

**Prevention**: The alias registry MUST be validated at compile time or server startup to ensure:
1. No alias matches any canonical tool name
2. No two tools share the same alias
3. Aliases do not form circular references

```rust
#[cfg(test)]
mod alias_validation {
    use super::*;

    #[test]
    fn test_no_alias_collisions_with_canonical() {
        let canonical_names: HashSet<_> = get_tool_definitions()
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        for (alias, _) in tool_aliases::ALIAS_MAP.iter() {
            assert!(
                !canonical_names.contains(alias),
                "Alias '{}' collides with canonical tool name",
                alias
            );
        }
    }
}
```

### 5.2 Case Sensitivity

Tool names and aliases are **case-sensitive**. The following are distinct:
- `discover_goals` (valid alias)
- `Discover_Goals` (unknown tool)
- `DISCOVER_GOALS` (unknown tool)

---

## 6. Migration Notes

### 6.1 Backwards Compatibility

This change is **fully backwards compatible**:
- Existing clients using canonical names continue to work unchanged
- New clients can use PRD aliases immediately
- No API version bump required

### 6.2 Deprecation Strategy

Aliases should be considered **permanent backwards compatibility shims**. If deprecation is desired in the future:

1. Add deprecation notice to tool descriptions
2. Emit deprecation warning log on alias use
3. Track alias usage metrics for impact analysis
4. Provide 6-month minimum deprecation window

---

## 7. Dependencies

### 7.1 Implementation Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| `once_cell` | Lazy static initialization for alias map | 1.x |
| `tracing` | Debug logging for alias resolution | existing |

### 7.2 Specification Dependencies

| Spec ID | Relationship |
|---------|--------------|
| SPEC-MCP-TOOLS-001 | Base tool definitions |
| PRD-CONSCIOUSNESS-001 | Source of PRD tool names |

---

## 8. Validation Checklist

- [ ] All aliases map to existing canonical tool names
- [ ] No alias collides with a canonical tool name
- [ ] `tools/list` output includes alias metadata
- [ ] Debug logging enabled for alias resolution
- [ ] Unit tests cover all alias mappings
- [ ] Integration tests verify end-to-end alias invocation
- [ ] Performance benchmark confirms <1us overhead

---

## 9. References

- **Gap Analysis**: `/docs/MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md` - Refinement 2
- **Tool Definitions**: `/crates/context-graph-mcp/src/tools.rs`
- **Tool Handlers**: `/crates/context-graph-mcp/src/handlers/tools.rs`
- **Autonomous Handlers**: `/crates/context-graph-mcp/src/handlers/autonomous.rs`
