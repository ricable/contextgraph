# Task 08: Add Tool Definitions for PRD v6 Tools

## Metadata
- **Task ID**: TASK-GAP-008
- **Phase**: 2 (MCP Infrastructure)
- **Priority**: High
- **Complexity**: Low
- **Dependencies**: task06 (TASK-GAP-006), task07 (TASK-GAP-007) - handlers ALREADY implemented
- **Branch**: multistar

## Current State Assessment (2026-01-18)

### ALREADY COMPLETED (by task07)
| Component | Status | File |
|-----------|--------|------|
| dispatch.rs routing | DONE | `src/handlers/tools/dispatch.rs` - All 12 tools routed |
| tool_names constants | DONE | `src/tools/names.rs` - All 12 constants defined |
| topic_tools handlers | DONE | `src/handlers/tools/topic_tools.rs` (377 lines) |
| topic_dtos | DONE | `src/handlers/tools/topic_dtos.rs` (1000+ lines) |
| curation_tools handlers | DONE | `src/handlers/tools/curation_tools.rs` (278 lines) |
| curation_dtos | DONE | `src/handlers/tools/curation_dtos.rs` (680 lines) |
| Compilation | PASSING | `cargo check -p context-graph-mcp` succeeds |

### REMAINING WORK
The **ONLY** remaining work is adding ToolDefinition schemas for 6 new tools:
- 4 Topic tools: `get_topic_portfolio`, `get_topic_stability`, `detect_topics`, `get_divergence_alerts`
- 2 Curation tools: `forget_concept`, `boost_importance`

**Current `tools/list` returns 6 tools, needs to return 12.**

## Objective

Add tool schema definitions for the 6 new MCP tools so they appear in the `tools/list` response. The dispatch routing and handlers are already implemented.

## Input Context Files (MUST READ)

```bash
# 1. Current definitions mod - see how tools are aggregated
/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/mod.rs

# 2. Core definitions pattern - use this as template
/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/core.rs

# 3. Merge definitions pattern - another example
/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/merge.rs

# 4. Topic DTOs - contains validation constants and schema details
/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs

# 5. Curation DTOs - contains validation constants and schema details
/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs

# 6. Constitution reference
/home/cabdru/contextgraph/docs2/constitution.yaml
```

## Files to Create/Modify

### Create: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/topic.rs`
**New file** containing topic tool definitions.

### Create: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/curation.rs`
**New file** containing curation tool definitions.

### Modify: `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/mod.rs`
Update to include new modules and aggregate all 12 tools.

## Implementation Steps

### Step 1: Create topic.rs

Create `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/topic.rs`:

```rust
//! Topic tool definitions per PRD v6 Section 10.2.
//!
//! Tools:
//! - get_topic_portfolio: Get all discovered topics with profiles
//! - get_topic_stability: Get portfolio-level stability metrics
//! - detect_topics: Force topic detection recalculation
//! - get_divergence_alerts: Check for divergence from recent activity
//!
//! Constitution Compliance:
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - AP-60: Temporal embedders (E2-E4) weight = 0.0 in topic detection
//! - AP-70: Dream recommended when entropy > 0.7 AND churn > 0.5

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns topic tool definitions (4 tools per PRD).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // get_topic_portfolio
        ToolDefinition::new(
            "get_topic_portfolio",
            "Get all discovered topics with profiles, stability metrics, and tier info. \
             Topics emerge from weighted multi-space clustering (threshold >= 2.5). \
             Temporal embedders (E2-E4) are excluded from topic detection.",
            json!({
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["brief", "standard", "verbose"],
                        "default": "standard",
                        "description": "Output format: brief (names only), standard (with spaces), verbose (full profiles)"
                    }
                }
            }),
        ),

        // get_topic_stability
        ToolDefinition::new(
            "get_topic_stability",
            "Get portfolio-level stability metrics including churn rate, entropy, and phase breakdown. \
             Dream consolidation is recommended when entropy > 0.7 AND churn > 0.5 (per AP-70).",
            json!({
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 168,
                        "default": 6,
                        "description": "Lookback period in hours for computing averages"
                    }
                }
            }),
        ),

        // detect_topics
        ToolDefinition::new(
            "detect_topics",
            "Force topic detection recalculation using HDBSCAN clustering. \
             Requires minimum 3 memories (per clustering.parameters.min_cluster_size). \
             Topics require weighted_agreement >= 2.5 to be recognized.",
            json!({
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "default": false,
                        "description": "Force detection even if recently computed"
                    }
                }
            }),
        ),

        // get_divergence_alerts
        ToolDefinition::new(
            "get_divergence_alerts",
            "Check for divergence from recent activity using SEMANTIC embedders only \
             (E1, E5, E6, E7, E10, E12, E13 per AP-62). Temporal embedders (E2-E4) are \
             excluded from divergence detection per AP-63.",
            json!({
                "type": "object",
                "properties": {
                    "lookback_hours": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 48,
                        "default": 2,
                        "description": "Hours to look back for recent activity comparison"
                    }
                }
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_definitions_count() {
        let tools = definitions();
        assert_eq!(tools.len(), 4, "Should have 4 topic tools");
    }

    #[test]
    fn test_topic_tools_names() {
        let tools = definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"get_topic_portfolio"));
        assert!(names.contains(&"get_topic_stability"));
        assert!(names.contains(&"detect_topics"));
        assert!(names.contains(&"get_divergence_alerts"));
    }

    #[test]
    fn test_get_topic_portfolio_format_enum() {
        let tools = definitions();
        let portfolio = tools.iter().find(|t| t.name == "get_topic_portfolio").unwrap();
        let props = portfolio.input_schema.get("properties").unwrap();
        let format = props.get("format").unwrap();
        let enum_vals = format.get("enum").unwrap().as_array().unwrap();
        assert_eq!(enum_vals.len(), 3);
        let values: Vec<&str> = enum_vals.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(values.contains(&"brief"));
        assert!(values.contains(&"standard"));
        assert!(values.contains(&"verbose"));
    }

    #[test]
    fn test_get_topic_stability_hours_range() {
        let tools = definitions();
        let stability = tools.iter().find(|t| t.name == "get_topic_stability").unwrap();
        let props = stability.input_schema.get("properties").unwrap();
        let hours = props.get("hours").unwrap();
        assert_eq!(hours.get("minimum").unwrap().as_u64().unwrap(), 1);
        assert_eq!(hours.get("maximum").unwrap().as_u64().unwrap(), 168);
        assert_eq!(hours.get("default").unwrap().as_u64().unwrap(), 6);
    }

    #[test]
    fn test_detect_topics_force_default() {
        let tools = definitions();
        let detect = tools.iter().find(|t| t.name == "detect_topics").unwrap();
        let props = detect.input_schema.get("properties").unwrap();
        let force = props.get("force").unwrap();
        assert_eq!(force.get("default").unwrap().as_bool().unwrap(), false);
    }

    #[test]
    fn test_get_divergence_alerts_lookback_range() {
        let tools = definitions();
        let alerts = tools.iter().find(|t| t.name == "get_divergence_alerts").unwrap();
        let props = alerts.input_schema.get("properties").unwrap();
        let lookback = props.get("lookback_hours").unwrap();
        assert_eq!(lookback.get("minimum").unwrap().as_u64().unwrap(), 1);
        assert_eq!(lookback.get("maximum").unwrap().as_u64().unwrap(), 48);
        assert_eq!(lookback.get("default").unwrap().as_u64().unwrap(), 2);
    }

    #[test]
    fn test_descriptions_mention_constitution() {
        let tools = definitions();
        // get_topic_portfolio mentions threshold
        let portfolio = tools.iter().find(|t| t.name == "get_topic_portfolio").unwrap();
        assert!(portfolio.description.contains("2.5"), "Should mention threshold");

        // get_topic_stability mentions AP-70
        let stability = tools.iter().find(|t| t.name == "get_topic_stability").unwrap();
        assert!(stability.description.contains("AP-70"), "Should reference AP-70");

        // get_divergence_alerts mentions AP-62
        let alerts = tools.iter().find(|t| t.name == "get_divergence_alerts").unwrap();
        assert!(alerts.description.contains("AP-62"), "Should reference AP-62");
    }
}
```

### Step 2: Create curation.rs

Create `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/curation.rs`:

```rust
//! Curation tool definitions per PRD v6 Section 10.3.
//!
//! Tools:
//! - forget_concept: Soft-delete a memory (30-day recovery per SEC-06)
//! - boost_importance: Adjust memory importance score
//!
//! Constitution Compliance:
//! - SEC-06: Soft delete 30-day recovery
//! - BR-MCP-001: forget_concept uses soft delete by default
//! - BR-MCP-002: boost_importance clamps final value to [0.0, 1.0]
//! - AP-10: No NaN/Infinity in values

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns curation tool definitions (2 tools per PRD).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // forget_concept
        ToolDefinition::new(
            "forget_concept",
            "Soft-delete a memory with 30-day recovery window (per SEC-06). \
             Set soft_delete=false for permanent deletion (use with caution). \
             Returns deleted_at timestamp for recovery tracking.",
            json!({
                "type": "object",
                "required": ["node_id"],
                "properties": {
                    "node_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the memory to forget"
                    },
                    "soft_delete": {
                        "type": "boolean",
                        "default": true,
                        "description": "Use soft delete with 30-day recovery (default true per BR-MCP-001)"
                    }
                }
            }),
        ),

        // boost_importance
        ToolDefinition::new(
            "boost_importance",
            "Adjust a memory's importance score by delta. Final value is clamped \
             to [0.0, 1.0] (per BR-MCP-002). Response includes old, delta, and new values.",
            json!({
                "type": "object",
                "required": ["node_id", "delta"],
                "properties": {
                    "node_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "UUID of the memory to boost"
                    },
                    "delta": {
                        "type": "number",
                        "minimum": -1.0,
                        "maximum": 1.0,
                        "description": "Importance change value (-1.0 to 1.0)"
                    }
                }
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curation_definitions_count() {
        let tools = definitions();
        assert_eq!(tools.len(), 2, "Should have 2 curation tools");
    }

    #[test]
    fn test_curation_tools_names() {
        let tools = definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"forget_concept"));
        assert!(names.contains(&"boost_importance"));
    }

    #[test]
    fn test_forget_concept_required_fields() {
        let tools = definitions();
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        let required = forget.input_schema.get("required").unwrap().as_array().unwrap();
        let fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(fields.len(), 1);
        assert!(fields.contains(&"node_id"));
        // soft_delete NOT required - has default
        assert!(!fields.contains(&"soft_delete"));
    }

    #[test]
    fn test_forget_concept_soft_delete_default() {
        let tools = definitions();
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        let props = forget.input_schema.get("properties").unwrap();
        let soft_delete = props.get("soft_delete").unwrap();
        // Per BR-MCP-001: defaults to true
        assert_eq!(soft_delete.get("default").unwrap().as_bool().unwrap(), true);
    }

    #[test]
    fn test_boost_importance_required_fields() {
        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let required = boost.input_schema.get("required").unwrap().as_array().unwrap();
        let fields: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(fields.len(), 2);
        assert!(fields.contains(&"node_id"));
        assert!(fields.contains(&"delta"));
    }

    #[test]
    fn test_boost_importance_delta_range() {
        let tools = definitions();
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        let props = boost.input_schema.get("properties").unwrap();
        let delta = props.get("delta").unwrap();
        // Per BR-MCP-002: delta range [-1.0, 1.0]
        assert_eq!(delta.get("minimum").unwrap().as_f64().unwrap(), -1.0);
        assert_eq!(delta.get("maximum").unwrap().as_f64().unwrap(), 1.0);
    }

    #[test]
    fn test_node_id_uuid_format() {
        let tools = definitions();
        for tool in &tools {
            let props = tool.input_schema.get("properties").unwrap();
            let node_id = props.get("node_id").unwrap();
            assert_eq!(node_id.get("type").unwrap().as_str().unwrap(), "string");
            assert_eq!(node_id.get("format").unwrap().as_str().unwrap(), "uuid");
        }
    }

    #[test]
    fn test_descriptions_mention_constitution() {
        let tools = definitions();
        // forget_concept mentions SEC-06
        let forget = tools.iter().find(|t| t.name == "forget_concept").unwrap();
        assert!(forget.description.contains("SEC-06"), "Should reference SEC-06");
        assert!(forget.description.contains("30-day"), "Should mention recovery period");

        // boost_importance mentions BR-MCP-002
        let boost = tools.iter().find(|t| t.name == "boost_importance").unwrap();
        assert!(boost.description.contains("BR-MCP-002"), "Should reference BR-MCP-002");
    }
}
```

### Step 3: Update mod.rs

Modify `/home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/mod.rs`:

```rust
//! Tool definitions per PRD v6 Section 10.
//!
//! 12 tools exposed:
//! - Core (5): inject_context, store_memory, get_memetic_status, search_graph, trigger_consolidation
//! - Curation (3): merge_concepts, forget_concept, boost_importance
//! - Topic (4): get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts

pub(crate) mod core;
pub(crate) mod curation;
pub mod merge;
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
            assert!(!tool.description.is_empty(), "Tool {} missing description", tool.name);
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
}
```

## Definition of Done

- [x] `topic.rs` created with 4 tool definitions
- [x] `curation.rs` created with 2 tool definitions
- [x] `mod.rs` updated to aggregate all 12 tools
- [x] `cargo check -p context-graph-mcp` passes
- [x] `cargo clippy -p context-graph-mcp -- -D warnings` passes
- [x] `cargo test -p context-graph-mcp definitions` passes (67 tests)
- [x] All 12 tools appear in `tools/list` response

## Completion Status: DONE (2026-01-18)

### Files Created/Modified
1. **CREATED**: `src/tools/definitions/topic.rs` (380 lines) - 4 topic tools with tests
2. **CREATED**: `src/tools/definitions/curation.rs` (292 lines) - 2 curation tools with tests
3. **MODIFIED**: `src/tools/definitions/mod.rs` - Updated to aggregate all 12 tools

### Verification Results
- `cargo check -p context-graph-mcp`: PASSED
- `cargo clippy -p context-graph-mcp -- -D warnings`: PASSED (no warnings)
- `cargo test -p context-graph-mcp definitions`: 67 passed, 0 failed, 1 ignored
- Tool count test: 12 tools verified
- Dispatch routes: 12 tools routed in dispatch.rs
- Edge case tests: 19 passed
- Synthetic data validation tests: 12 passed

### Code Review (code-simplifier)
- Overall assessment: **GOOD**
- Constitution compliance: **EXCELLENT**
- Minor fix applied: Changed `pub mod merge;` to `pub(crate) mod merge;` for consistency

## Verification Commands

```bash
cd /home/cabdru/contextgraph

# 1. Verify compilation
cargo check -p context-graph-mcp

# 2. Verify no clippy warnings
cargo clippy -p context-graph-mcp -- -D warnings

# 3. Run definition tests
cargo test -p context-graph-mcp definitions -- --nocapture

# 4. Count tool definitions (expected: 12)
cargo test -p context-graph-mcp test_total_tool_count -- --nocapture

# 5. Verify all tool names present
cargo test -p context-graph-mcp test_all_tool_names_present -- --nocapture

# 6. Verify dispatch has all 12 tools (already passing)
grep "tool_names::" crates/context-graph-mcp/src/handlers/tools/dispatch.rs | wc -l
# Expected: 12
```

## Full State Verification (FSV)

After completing the logic, perform the following verification:

### Source of Truth
The source of truth is `get_tool_definitions()` in `mod.rs`, which returns a `Vec<ToolDefinition>` that is serialized to JSON for the `tools/list` MCP response.

### Execute & Inspect
```bash
# Run the test that counts tools
cargo test -p context-graph-mcp test_total_tool_count -- --nocapture 2>&1 | grep -E "(running|test|PASSED|FAILED)"

# Run test that verifies all names
cargo test -p context-graph-mcp test_all_tool_names_present -- --nocapture 2>&1 | grep -E "(running|test|PASSED|FAILED)"
```

### Boundary & Edge Case Audit

**Edge Case 1: Empty properties object**
```rust
// detect_topics has optional-only parameters
// Verify it handles empty JSON: {}
let request = serde_json::json!({});
// Expected: force defaults to false
```

**Edge Case 2: Maximum values**
```rust
// get_topic_stability hours at max (168)
let request = serde_json::json!({"hours": 168});
// Expected: valid, should not error

// get_divergence_alerts lookback at max (48)
let request = serde_json::json!({"lookback_hours": 48});
// Expected: valid, should not error
```

**Edge Case 3: Invalid enum value**
```rust
// get_topic_portfolio with invalid format
let request = serde_json::json!({"format": "invalid"});
// Expected: validation error with message listing valid formats
```

### Evidence of Success

After implementation, provide:
1. Output of `cargo test -p context-graph-mcp definitions` showing all tests pass
2. Output of `cargo test -p context-graph-mcp test_total_tool_count` showing 12 tools
3. The actual JSON structure returned by `get_tool_definitions()` (log or print in test)

## Manual Testing Protocol

### Test 1: tools/list Response Contains All 12 Tools
```bash
# If MCP server is running, call tools/list and verify response
# Or use test to verify:
cargo test -p context-graph-mcp test_all_tool_names_present -- --nocapture
```

**Expected Output**: Test passes, showing all 12 tool names.

### Test 2: Schema Validation for Topic Tools
```bash
cargo test -p context-graph-mcp topic::tests -- --nocapture
```

**Expected Output**: All 7 topic tool tests pass, validating format enums, hour ranges, defaults.

### Test 3: Schema Validation for Curation Tools
```bash
cargo test -p context-graph-mcp curation::tests -- --nocapture
```

**Expected Output**: All 7 curation tool tests pass, validating required fields, defaults, ranges.

### Test 4: No Duplicate Tools
```bash
cargo test -p context-graph-mcp test_no_duplicate_tools -- --nocapture
```

**Expected Output**: Test passes confirming no duplicate tool names.

## Synthetic Data Validation

### Topic Tool Synthetic Inputs

```json
// get_topic_portfolio - valid
{"format": "verbose"}

// get_topic_portfolio - invalid (should fail validation in handler)
{"format": "detailed"}

// get_topic_stability - valid
{"hours": 24}

// get_topic_stability - boundary max
{"hours": 168}

// detect_topics - valid with force
{"force": true}

// detect_topics - empty (uses defaults)
{}

// get_divergence_alerts - valid
{"lookback_hours": 6}
```

### Curation Tool Synthetic Inputs

```json
// forget_concept - soft delete (default)
{"node_id": "550e8400-e29b-41d4-a716-446655440000"}

// forget_concept - hard delete
{"node_id": "550e8400-e29b-41d4-a716-446655440000", "soft_delete": false}

// boost_importance - positive boost
{"node_id": "550e8400-e29b-41d4-a716-446655440000", "delta": 0.3}

// boost_importance - negative boost (demote)
{"node_id": "550e8400-e29b-41d4-a716-446655440000", "delta": -0.2}

// boost_importance - boundary max
{"node_id": "550e8400-e29b-41d4-a716-446655440000", "delta": 1.0}
```

## Constitution Compliance Checklist

- [x] ARCH-09: Topic threshold 2.5 mentioned in `get_topic_portfolio` description
- [x] AP-60: Temporal embedder exclusion mentioned in `get_topic_portfolio` description
- [x] AP-62: SEMANTIC-only for divergence mentioned in `get_divergence_alerts` description
- [x] AP-63: Temporal exclusion from divergence mentioned in description
- [x] AP-70: Dream trigger conditions mentioned in `get_topic_stability` description
- [x] SEC-06: 30-day recovery mentioned in `forget_concept` description
- [x] BR-MCP-001: soft_delete default=true in `forget_concept` schema
- [x] BR-MCP-002: Importance clamping mentioned in `boost_importance` description
- [x] AP-10: Delta range [-1.0, 1.0] in `boost_importance` schema (prevents NaN/Infinity)

## Error Handling Requirements

All tools must fail fast with clear error messages:

1. **Invalid format enum**: Return error listing valid formats
2. **Hours out of range**: Return error with valid range
3. **Missing required field**: Return error naming missing field
4. **Invalid UUID format**: Return error indicating invalid UUID
5. **Delta out of range**: Return error with valid range

No fallbacks. No defaults for required fields. Explicit errors for all invalid inputs.

## Notes

- Handlers are ALREADY implemented in `topic_tools.rs` and `curation_tools.rs`
- Dispatch routing is ALREADY implemented in `dispatch.rs`
- This task ONLY adds schema definitions for tool discovery via `tools/list`
- Use existing patterns from `core.rs` and `merge.rs` for consistency
