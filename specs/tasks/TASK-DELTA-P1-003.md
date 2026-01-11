# TASK-DELTA-P1-003: MCP Handler Registration for compute_delta_sc

## Metadata

| Field | Value |
|-------|-------|
| **ID** | TASK-DELTA-P1-003 |
| **Version** | 1.0 |
| **Status** | ready |
| **Layer** | surface |
| **Sequence** | 3 of 4 |
| **Priority** | P1 |
| **Estimated Complexity** | medium |
| **Estimated Duration** | 2-3 hours |
| **Implements** | REQ-UTL-001, REQ-UTL-020 |
| **Depends On** | TASK-DELTA-P1-001, TASK-DELTA-P1-002 |
| **Spec Ref** | SPEC-UTL-001 |
| **Gap Ref** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md GAP 1 |

---

## Context

This task wires the `DeltaScComputer` from TASK-DELTA-P1-002 to the MCP protocol layer, registering `gwt/compute_delta_sc` as a discoverable tool.

**Why This Third**: Following Inside-Out, Bottom-Up:
1. Types (TASK-DELTA-P1-001) and logic (TASK-DELTA-P1-002) must exist first
2. Surface layer consumes the logic layer
3. MCP registration makes the tool externally accessible

**Gap Being Addressed**:
> GAP 1: UTL compute_delta_sc MCP Tool Missing
> "compute_delta_sc" must be in gwt_tools list per constitution.yaml

---

## Input Context Files

| Purpose | File |
|---------|------|
| Request/Response types | `crates/context-graph-mcp/src/types/delta_sc.rs` |
| DeltaScComputer | `crates/context-graph-mcp/src/services/delta_sc_computer.rs` |
| Existing handler pattern | `crates/context-graph-mcp/src/handlers/utl.rs` |
| Protocol types | `crates/context-graph-mcp/src/protocol.rs` |
| Error codes | `crates/context-graph-mcp/src/protocol.rs#error_codes` |
| Handlers struct | `crates/context-graph-mcp/src/handlers/mod.rs` |
| Tool registration | `crates/context-graph-mcp/src/handlers/tools.rs` |

---

## Prerequisites

| Check | Verification |
|-------|--------------|
| TASK-DELTA-P1-001 complete | Types compile |
| TASK-DELTA-P1-002 complete | DeltaScComputer tests pass |
| Handler pattern understood | Read existing utl.rs handlers |
| Error codes defined | Check protocol.rs for error codes |

---

## Scope

### In Scope

- Add `handle_gwt_compute_delta_sc()` method to `Handlers` struct
- Add `DeltaScComputer` field to `Handlers` struct
- Register tool in `tools/list` response
- Wire request parsing and response formatting
- Add appropriate error handling per SPEC-UTL-001
- Add tracing spans for observability

### Out of Scope

- Integration tests (TASK-DELTA-P1-004)
- Performance benchmarking (TASK-DELTA-P1-004)
- Claude Code hook integration (future task)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-mcp/src/handlers/gwt.rs (new or existing)

use tracing::{debug, error, warn, instrument};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};
use crate::services::DeltaScComputer;
use crate::types::delta_sc::{ComputeDeltaScRequest, ComputeDeltaScResponse};

use super::Handlers;

impl Handlers {
    /// Handle gwt/compute_delta_sc request.
    ///
    /// Computes Delta-S (entropy) and Delta-C (coherence) for a vertex update.
    /// Required by constitution.yaml gwt_tools.
    ///
    /// # Parameters
    /// - `vertex_id`: UUID of the vertex being updated
    /// - `old_fingerprint`: Previous teleological fingerprint (13 embedders)
    /// - `new_fingerprint`: New teleological fingerprint (13 embedders)
    /// - `include_diagnostics`: Optional, include computation details (default: false)
    /// - `johari_threshold`: Optional, override Johari threshold (default: 0.5)
    ///
    /// # Returns
    /// - `delta_s_per_embedder`: Per-embedder entropy change [0, 1]
    /// - `delta_s_aggregate`: Weighted average entropy
    /// - `delta_c`: Coherence change [0, 1]
    /// - `johari_quadrants`: Per-embedder Johari classification
    /// - `johari_aggregate`: Overall Johari classification
    /// - `utl_learning_potential`: Delta-S * Delta-C
    /// - `diagnostics`: Optional computation details
    ///
    /// # Error Codes
    /// - -32602 (INVALID_PARAMS): Missing or invalid parameters
    /// - -32603 (INTERNAL_ERROR): Computation failure
    /// - -32801 (INVALID_FINGERPRINT): Incomplete fingerprint
    /// - -32802 (COMPUTATION_ERROR): Entropy/coherence calculation failed
    #[instrument(skip(self, params), fields(method = "gwt/compute_delta_sc"))]
    pub(super) async fn handle_gwt_compute_delta_sc(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        debug!("gwt/compute_delta_sc: starting");

        // 1. Parse and validate parameters
        let params = match params {
            Some(p) => p,
            None => {
                warn!("gwt/compute_delta_sc: missing parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing parameters",
                );
            }
        };

        let request: ComputeDeltaScRequest = match serde_json::from_value(params) {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "gwt/compute_delta_sc: invalid parameters");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid parameters: {}", e),
                );
            }
        };

        // 2. Validate Johari threshold if provided
        if let Some(threshold) = request.johari_threshold {
            if !(0.35..=0.65).contains(&threshold) {
                warn!(threshold, "gwt/compute_delta_sc: invalid johari_threshold");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("johari_threshold must be in [0.35, 0.65], got {}", threshold),
                );
            }
        }

        // 3. Compute Delta-S and Delta-C
        match self.delta_sc_computer.compute(&request).await {
            Ok(response) => {
                debug!(
                    delta_s = %response.delta_s_aggregate,
                    delta_c = %response.delta_c,
                    johari = ?response.johari_aggregate,
                    "gwt/compute_delta_sc: completed"
                );
                JsonRpcResponse::success(id, serde_json::to_value(response).unwrap())
            }
            Err(e) => {
                error!(error = %e, "gwt/compute_delta_sc: computation failed");
                JsonRpcResponse::error(
                    id,
                    error_codes::COMPUTATION_ERROR,
                    format!("Computation failed: {}", e),
                )
            }
        }
    }
}
```

```rust
// File: crates/context-graph-mcp/src/handlers/mod.rs (additions)

use crate::services::DeltaScComputer;

/// MCP request handlers.
pub struct Handlers {
    // ... existing fields ...

    /// Delta-S/Delta-C computer for UTL learning.
    pub(crate) delta_sc_computer: DeltaScComputer,
}

impl Handlers {
    pub fn new(/* existing params */) -> Self {
        Self {
            // ... existing fields ...
            delta_sc_computer: DeltaScComputer::new(),
        }
    }
}
```

```rust
// File: crates/context-graph-mcp/src/handlers/tools.rs (additions)

// In tools_list response, add:
{
    "name": "gwt/compute_delta_sc",
    "description": "Compute Delta-S (entropy) and Delta-C (coherence) for vertex update per UTL equation",
    "inputSchema": {
        "type": "object",
        "properties": {
            "vertex_id": {
                "type": "string",
                "format": "uuid",
                "description": "Vertex identifier"
            },
            "old_fingerprint": {
                "type": "object",
                "description": "Previous teleological fingerprint (13 embedders)"
            },
            "new_fingerprint": {
                "type": "object",
                "description": "New teleological fingerprint (13 embedders)"
            },
            "include_diagnostics": {
                "type": "boolean",
                "default": false,
                "description": "Include detailed computation diagnostics"
            },
            "johari_threshold": {
                "type": "number",
                "minimum": 0.35,
                "maximum": 0.65,
                "description": "Override Johari classification threshold (default: 0.5)"
            }
        },
        "required": ["vertex_id", "old_fingerprint", "new_fingerprint"]
    }
}
```

### Constraints

- Tool name MUST be `gwt/compute_delta_sc` per constitution.yaml gwt_tools
- Error codes MUST match existing protocol.rs patterns
- Tracing spans MUST be added for observability
- Response MUST serialize to JSON matching SPEC-UTL-001 schema
- Handler MUST be async for consistency with other handlers
- Johari threshold validation MUST enforce [0.35, 0.65] range

### Verification

```bash
# Handler compiles
cargo check -p context-graph-mcp

# Tool appears in tools/list
cargo test -p context-graph-mcp test_tools_list_includes_compute_delta_sc

# Request parsing works
cargo test -p context-graph-mcp test_compute_delta_sc_request_parsing

# Error handling correct
cargo test -p context-graph-mcp test_compute_delta_sc_error_handling
```

---

## Pseudo Code

```rust
// Routing addition in dispatch():
async fn dispatch(&self, method: &str, id: Option<JsonRpcId>, params: Option<Value>) -> JsonRpcResponse {
    match method {
        // ... existing handlers ...
        "gwt/compute_delta_sc" => self.handle_gwt_compute_delta_sc(id, params).await,
        // ...
    }
}

// Tool registration in handle_tools_list():
fn handle_tools_list(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
    let tools = vec![
        // ... existing tools ...
        json!({
            "name": "gwt/compute_delta_sc",
            "description": "Compute Delta-S and Delta-C for vertex update",
            "inputSchema": { /* schema from definition of done */ }
        }),
        // ...
    ];
    JsonRpcResponse::success(id, json!({ "tools": tools }))
}
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/gwt.rs` | GWT handlers (if not exists) |

---

## Files to Modify

| Path | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Add delta_sc_computer field, import gwt module |
| `crates/context-graph-mcp/src/handlers/tools.rs` | Register tool in tools/list |
| `crates/context-graph-mcp/src/protocol.rs` | Add error codes if needed (INVALID_FINGERPRINT, COMPUTATION_ERROR) |
| `crates/context-graph-mcp/src/server.rs` | Add routing for gwt/compute_delta_sc |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Tool discoverable via tools/list | Integration test |
| Valid request returns success | Unit test with mock fingerprints |
| Missing params returns -32602 | Error handling test |
| Invalid UUID returns -32602 | Error handling test |
| Computation error returns -32802 | Error handling test with broken input |
| Tracing spans present | Log output inspection |

---

## Test Commands

```bash
# Handler compiles
cargo check -p context-graph-mcp

# Unit tests
cargo test -p context-graph-mcp gwt_compute_delta_sc -- --nocapture

# Integration test (if MCP server running)
# echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | nc localhost 3000

# Full test suite
cargo test -p context-graph-mcp --lib
```

---

## Notes

- Follow existing handler patterns from `utl.rs` and `memory.rs`
- The tool is part of GWT subsystem per constitution.yaml gwt_tools
- CognitivePulse can optionally be attached to response (future enhancement)
- Consider adding rate limiting if compute_delta_sc is expensive
