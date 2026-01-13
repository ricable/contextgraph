# TASK-27: Implement epistemic_action tool schema

## CRITICAL: READ THIS FIRST

**Task ID**: TASK-MCP-001 (sequence 27)
**Status**: COMPLETE (2026-01-13)
**Dependencies**: NONE (but TASK-19 through TASK-26 must be complete before Phase 4)
**Blocks**: TASK-28 (handler implementation) - NOW UNBLOCKED

---

## 1. Context & Purpose

The `epistemic_action` MCP tool allows clients to perform epistemic actions on the GWT workspace to update uncertainty and knowledge states. This task creates the **type-safe Rust schema structs** for input/output validation.

### Why This Tool Exists
Per constitution.yaml `utl.johari`:
- **Unknown quadrant** (ΔS>0.5, ΔC>0.5) → suggests `EpistemicAction`
- The cognitive pulse (PRD Section 1.3) recommends `epistemic_action` when entropy >0.7 and coherence >0.5

This is **NOT using schemars**. The existing codebase pattern uses `serde_json::json!()` macros to define schemas inline within `ToolDefinition::new()` calls.

---

## 2. Current Codebase State (Verified 2026-01-13)

### Completed Prerequisites
- TASK-17: parking_lot::RwLock added (commit da4ccc3)
- TASK-18: HashMap pre-allocation (commit da4ccc3)
- TASK-19: IdentityCritical variant added (commit 06d8578)
- TASK-20: TriggerConfig implemented (commit b224df5)
- TASK-21: TriggerManager IC checking (commit b224df5)
- TASK-22: GpuMonitor trait (commit 4d455c3)

### Existing Directory Structure
```
crates/context-graph-mcp/src/
├── tools/
│   ├── mod.rs           # Re-exports definitions::get_tool_definitions()
│   ├── types.rs         # ToolDefinition struct
│   ├── names.rs         # Tool name constants
│   ├── aliases.rs       # Tool aliases
│   └── definitions/
│       ├── mod.rs       # Aggregates all definitions
│       ├── core.rs      # inject_context, store_memory, search_graph, etc.
│       ├── gwt.rs       # get_consciousness_state, get_kuramoto_sync, etc.
│       ├── utl.rs       # gwt/compute_delta_sc
│       ├── atc.rs       # threshold tools
│       ├── dream.rs     # trigger_dream, get_dream_status, etc.
│       ├── neuromod.rs  # neuromodulation tools
│       ├── steering.rs  # get_steering_feedback
│       ├── causal.rs    # omni_infer
│       ├── teleological.rs  # search_teleological, etc.
│       ├── autonomous.rs    # auto_bootstrap_north_star, etc.
│       └── meta_utl.rs      # meta-learning tools
└── handlers/
    ├── mod.rs
    ├── johari/          # Johari quadrant handlers
    ├── memory/          # Memory handlers
    ├── utl/             # UTL handlers
    └── ...
```

### Schema Pattern Used in Codebase
**IMPORTANT**: The codebase does NOT use `schemars::JsonSchema`. Schemas are defined inline using `serde_json::json!()` macro:

```rust
// Example from crates/context-graph-mcp/src/tools/definitions/core.rs
ToolDefinition::new(
    "inject_context",
    "Description here...",
    json!({
        "type": "object",
        "properties": { ... },
        "required": ["content", "rationale"]
    }),
)
```

---

## 3. Exact Implementation Requirements

### 3.1 Files to CREATE

#### File: `crates/context-graph-mcp/src/tools/definitions/epistemic.rs`
```rust
//! Epistemic action tool definitions (TASK-MCP-001).
//!
//! Implements epistemic_action tool for GWT workspace uncertainty/knowledge updates.
//! Constitution: utl.johari.Unknown → EpistemicAction
//! PRD: Section 1.8, Section 5.2 Line 527

use serde_json::json;
use crate::tools::types::ToolDefinition;

/// Returns epistemic tool definitions.
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "epistemic_action",
            "Perform an epistemic action on the GWT workspace to update uncertainty \
             and knowledge states. Actions: assert (add belief), retract (remove belief), \
             query (check status), hypothesize (tentative belief), verify (confirm/deny). \
             Used when Johari quadrant is Unknown (high entropy + high coherence).",
            json!({
                "type": "object",
                "required": ["action_type", "target", "rationale"],
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["assert", "retract", "query", "hypothesize", "verify"],
                        "description": "Type of epistemic action: assert=add belief, retract=remove belief, query=check status, hypothesize=tentative belief, verify=confirm/deny"
                    },
                    "target": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 4096,
                        "description": "Target concept or proposition (1-4096 chars)"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "Confidence level [0.0, 1.0], default 0.5"
                    },
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024,
                        "description": "Rationale for action (required per PRD 0.3)"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context for the action",
                        "properties": {
                            "source_nodes": {
                                "type": "array",
                                "items": { "type": "string", "format": "uuid" },
                                "description": "UUIDs of related source nodes"
                            },
                            "uncertainty_type": {
                                "type": "string",
                                "enum": ["epistemic", "aleatory", "mixed"],
                                "description": "epistemic=knowledge gap, aleatory=inherent randomness, mixed=both"
                            }
                        }
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
    fn test_epistemic_action_definition_exists() {
        let tools = definitions();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "epistemic_action");
    }

    #[test]
    fn test_epistemic_action_schema_required_fields() {
        let tools = definitions();
        let schema = &tools[0].input_schema;
        let required = schema.get("required").unwrap().as_array().unwrap();

        assert!(required.iter().any(|v| v.as_str() == Some("action_type")));
        assert!(required.iter().any(|v| v.as_str() == Some("target")));
        assert!(required.iter().any(|v| v.as_str() == Some("rationale")));
    }

    #[test]
    fn test_epistemic_action_type_enum_values() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let action_type = props.get("action_type").unwrap();
        let enum_values = action_type.get("enum").unwrap().as_array().unwrap();

        let values: Vec<&str> = enum_values.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(values.contains(&"assert"));
        assert!(values.contains(&"retract"));
        assert!(values.contains(&"query"));
        assert!(values.contains(&"hypothesize"));
        assert!(values.contains(&"verify"));
    }

    #[test]
    fn test_target_length_constraints() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let target = props.get("target").unwrap();

        assert_eq!(target.get("minLength").unwrap().as_u64().unwrap(), 1);
        assert_eq!(target.get("maxLength").unwrap().as_u64().unwrap(), 4096);
    }

    #[test]
    fn test_rationale_length_constraints() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let rationale = props.get("rationale").unwrap();

        assert_eq!(rationale.get("minLength").unwrap().as_u64().unwrap(), 1);
        assert_eq!(rationale.get("maxLength").unwrap().as_u64().unwrap(), 1024);
    }

    #[test]
    fn test_confidence_bounds() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let confidence = props.get("confidence").unwrap();

        assert_eq!(confidence.get("minimum").unwrap().as_f64().unwrap(), 0.0);
        assert_eq!(confidence.get("maximum").unwrap().as_f64().unwrap(), 1.0);
        assert_eq!(confidence.get("default").unwrap().as_f64().unwrap(), 0.5);
    }

    #[test]
    fn test_context_uncertainty_types() {
        let tools = definitions();
        let props = tools[0].input_schema.get("properties").unwrap();
        let context = props.get("context").unwrap();
        let context_props = context.get("properties").unwrap();
        let uncertainty = context_props.get("uncertainty_type").unwrap();
        let enum_values = uncertainty.get("enum").unwrap().as_array().unwrap();

        let values: Vec<&str> = enum_values.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(values.contains(&"epistemic"));
        assert!(values.contains(&"aleatory"));
        assert!(values.contains(&"mixed"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let tools = definitions();
        let json_str = serde_json::to_string(&tools).expect("Serialization failed");
        assert!(json_str.contains("epistemic_action"));
        assert!(json_str.contains("inputSchema"));
    }
}
```

### 3.2 Files to MODIFY

#### File: `crates/context-graph-mcp/src/tools/definitions/mod.rs`
Add the epistemic module:

```rust
// Add this line with other module declarations:
pub mod epistemic;

// In get_tool_definitions() function, add:
tools.extend(epistemic::definitions());
```

#### File: `crates/context-graph-mcp/src/tools/names.rs`
Add the tool name constant:

```rust
// Add in GWT EXTENSION TOOLS section or create new EPISTEMIC section:
pub const EPISTEMIC_ACTION: &str = "epistemic_action";
```

#### File: `crates/context-graph-mcp/src/tools/mod.rs`
Update the test assertion for tool count:

```rust
// Update from:
assert_eq!(tools.len(), 39);
// To:
assert_eq!(tools.len(), 40); // Added epistemic_action
```

---

## 4. Constraints (MUST NOT Violate)

| Constraint | Requirement |
|------------|-------------|
| target length | 1-4096 chars (per PRD Section 26) |
| rationale length | 1-1024 chars (REQUIRED per PRD 0.3) |
| confidence range | [0.0, 1.0], default 0.5 |
| action_type | MUST be one of: assert, retract, query, hypothesize, verify |
| uncertainty_type | MUST be one of: epistemic, aleatory, mixed |
| source_nodes | UUIDs only (format: uuid) |
| NO schemars | Use serde_json::json!() pattern like existing tools |

---

## 5. Verification Commands

```bash
# Step 1: Check compilation
cargo check -p context-graph-mcp

# Step 2: Run schema tests
cargo test -p context-graph-mcp definitions::epistemic

# Step 3: Verify tool count
cargo test -p context-graph-mcp test_get_tool_definitions

# Step 4: Full test suite
cargo test -p context-graph-mcp --lib
```

---

## 6. Full State Verification Protocol

After implementing the schema, you MUST perform these verification steps:

### 6.1 Source of Truth
The tool definition is stored in memory as part of `get_tool_definitions()` return value. Verify by:

```bash
# Run the tool definitions test
cargo test -p context-graph-mcp test_get_tool_definitions -- --nocapture
```

**Expected output**: Test passes, tool count = 40

### 6.2 Execute & Inspect

```bash
# Check that the tool appears in the list
cargo test -p context-graph-mcp test_epistemic_action_definition_exists -- --nocapture
```

**Expected output**:
```
test definitions::epistemic::tests::test_epistemic_action_definition_exists ... ok
```

### 6.3 Boundary & Edge Case Audit

Run these 3 edge case tests and verify state before/after:

**Edge Case 1: Empty target string (should fail validation later)**
```rust
// In handler tests (TASK-28), verify empty target rejected
let input = json!({"action_type": "assert", "target": "", "rationale": "test"});
// Schema allows minLength:1, so empty string should be caught by validation middleware
```

**Edge Case 2: Maximum length target (4096 chars)**
```rust
// Verify schema accepts 4096-char target
let long_target = "x".repeat(4096);
// Schema maxLength:4096 should accept this
```

**Edge Case 3: Missing required fields**
```rust
// Verify required fields enforced
let input = json!({"action_type": "assert"}); // Missing target and rationale
// Schema "required": ["action_type", "target", "rationale"] should catch this
```

### 6.4 Evidence of Success

After running tests, provide log output showing:
1. All 7+ epistemic tests pass
2. Tool count test passes with count = 40
3. Serialization roundtrip test passes

Example evidence format:
```
running 7 tests
test definitions::epistemic::tests::test_epistemic_action_definition_exists ... ok
test definitions::epistemic::tests::test_epistemic_action_schema_required_fields ... ok
test definitions::epistemic::tests::test_epistemic_action_type_enum_values ... ok
test definitions::epistemic::tests::test_target_length_constraints ... ok
test definitions::epistemic::tests::test_rationale_length_constraints ... ok
test definitions::epistemic::tests::test_confidence_bounds ... ok
test definitions::epistemic::tests::test_context_uncertainty_types ... ok
test definitions::epistemic::tests::test_serialization_roundtrip ... ok

test result: ok. 8 passed; 0 failed; 0 ignored
```

---

## 7. Manual Testing Protocol

### 7.1 Synthetic Test Data

Use this synthetic input to verify schema structure:

```json
{
  "action_type": "hypothesize",
  "target": "The system should consolidate memories when IC < 0.5",
  "confidence": 0.75,
  "rationale": "Identity crisis threshold per constitution.yaml gwt.self_ego_node.thresholds.critical",
  "context": {
    "source_nodes": ["550e8400-e29b-41d4-a716-446655440000"],
    "uncertainty_type": "epistemic"
  }
}
```

**Expected**: Schema validates all fields pass constraints:
- action_type: "hypothesize" ∈ ["assert", "retract", "query", "hypothesize", "verify"] ✓
- target: 59 chars ∈ [1, 4096] ✓
- confidence: 0.75 ∈ [0.0, 1.0] ✓
- rationale: 74 chars ∈ [1, 1024] ✓
- source_nodes: valid UUID format ✓
- uncertainty_type: "epistemic" ∈ ["epistemic", "aleatory", "mixed"] ✓

### 7.2 Verify Tool Registration

After implementation, verify the tool appears in definitions:

```rust
// Add this test or run manually
#[test]
fn verify_epistemic_in_all_tools() {
    let tools = crate::tools::get_tool_definitions();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"epistemic_action"), "epistemic_action missing from tool list");
}
```

---

## 8. Files Summary

| Action | File Path |
|--------|-----------|
| CREATE | `crates/context-graph-mcp/src/tools/definitions/epistemic.rs` |
| MODIFY | `crates/context-graph-mcp/src/tools/definitions/mod.rs` |
| MODIFY | `crates/context-graph-mcp/src/tools/names.rs` |
| MODIFY | `crates/context-graph-mcp/src/tools/mod.rs` |

---

## 9. Definition of Done Checklist

- [ ] `epistemic.rs` file created with `definitions()` function
- [ ] `mod.rs` includes `pub mod epistemic` and extends tool list
- [ ] `names.rs` includes `EPISTEMIC_ACTION` constant
- [ ] Tool count test updated to 40
- [ ] `cargo check -p context-graph-mcp` passes
- [ ] All epistemic tests pass
- [ ] Tool count test passes
- [ ] Manual verification with synthetic data documented
- [ ] Evidence log captured showing all tests passing

---

## 10. Common Pitfalls to Avoid

1. **DO NOT use schemars** - The codebase uses `serde_json::json!()` macros
2. **DO NOT forget to update tool count** in `mod.rs` test assertion
3. **DO NOT make rationale optional** - It's required per PRD 0.3
4. **DO NOT use `unwrap()` in library code** - Use `expect()` with context
5. **DO NOT add handlers** - That's TASK-28's scope

---

## 11. Related Files Reference

For understanding the pattern, read these existing files:
- `crates/context-graph-mcp/src/tools/definitions/core.rs` (inject_context pattern)
- `crates/context-graph-mcp/src/tools/definitions/gwt.rs` (GWT tools pattern)
- `crates/context-graph-mcp/src/tools/types.rs` (ToolDefinition struct)

---

## 12. COMPLETION EVIDENCE (2026-01-13)

### Files Created/Modified
| Action | File Path |
|--------|-----------|
| ✅ CREATED | `crates/context-graph-mcp/src/tools/definitions/epistemic.rs` |
| ✅ MODIFIED | `crates/context-graph-mcp/src/tools/definitions/mod.rs` |
| ✅ MODIFIED | `crates/context-graph-mcp/src/tools/names.rs` |
| ✅ MODIFIED | `crates/context-graph-mcp/src/tools/mod.rs` |
| ✅ MODIFIED | `crates/context-graph-mcp/src/handlers/tests/tools_list.rs` |

### Test Results
```
running 15 tests
test tools::definitions::epistemic::tests::test_all_action_type_values ... ok
test tools::definitions::epistemic::tests::test_confidence_bounds ... ok
test tools::definitions::epistemic::tests::test_context_uncertainty_types ... ok
test tools::definitions::epistemic::tests::test_edge_case_maximum_target_length ... ok
test tools::definitions::epistemic::tests::test_edge_case_minimum_values ... ok
test tools::definitions::epistemic::tests::test_edge_case_required_fields ... ok
test tools::definitions::epistemic::tests::test_epistemic_action_definition_exists ... ok
test tools::definitions::epistemic::tests::test_epistemic_action_schema_required_fields ... ok
test tools::definitions::epistemic::tests::test_epistemic_action_type_enum_values ... ok
test tools::definitions::epistemic::tests::test_rationale_length_constraints ... ok
test tools::definitions::epistemic::tests::test_source_nodes_uuid_format ... ok
test tools::definitions::epistemic::tests::test_synthetic_valid_input ... ok
test tools::definitions::epistemic::tests::test_serialization_roundtrip ... ok
test tools::definitions::epistemic::tests::test_target_length_constraints ... ok
test tools::definitions::epistemic::tests::test_verify_epistemic_in_all_tools ... ok

test result: ok. 15 passed; 0 failed; 0 ignored
```

### Definition of Done Checklist
- [x] `epistemic.rs` file created with `definitions()` function
- [x] `mod.rs` includes `pub mod epistemic` and extends tool list
- [x] `names.rs` includes `EPISTEMIC_ACTION` constant
- [x] Tool count test updated to 40
- [x] `cargo check -p context-graph-mcp` passes
- [x] All epistemic tests pass
- [x] Tool count test passes
- [x] Manual verification with synthetic data documented
- [x] Evidence log captured showing all tests passing
- [x] Code-simplifier review: PASSED

### Memory File
Completion record stored in: `TASK-27_epistemic_action_schema_completion.md`

---

*Document Version: 2.1.0 | Updated: 2026-01-13 | Status: COMPLETE*
