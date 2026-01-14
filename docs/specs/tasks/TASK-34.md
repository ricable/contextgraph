# TASK-34: Implement get_coherence_state MCP Tool

```xml
<task_spec id="TASK-34" version="2.0">
<metadata>
  <title>Implement get_coherence_state MCP Tool</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>34</sequence>
  <implements><requirement_ref>REQ-MCP-008</requirement_ref></implements>
  <depends_on>TASK-12</depends_on>
  <estimated_hours>3</estimated_hours>
  <original_id>TASK-MCP-008</original_id>
</metadata>
```

---

## CRITICAL: READ THIS SECTION FIRST

**You are implementing a NEW MCP tool that exposes GWT workspace coherence metrics.**

### What This Task IS:
- Create a new MCP tool named `get_coherence_state`
- The tool returns Kuramoto order parameter, coherence level enum, broadcasting status, and conflict status
- It queries the existing `KuramotoProvider` trait (already implemented in TASK-12)

### What This Task IS NOT:
- NOT implementing new Kuramoto logic (that exists in TASK-10, TASK-11, TASK-12)
- NOT duplicating `get_kuramoto_sync` (that returns raw oscillator data; this returns high-level coherence state)
- NOT modifying core GWT logic

### Key Distinction from Existing Tools:
| Tool | Purpose |
|------|---------|
| `get_kuramoto_sync` | Returns raw oscillator data: all 13 phases, frequencies, coupling, r, psi |
| `get_consciousness_state` | Returns full consciousness state: C(t), workspace, identity, component analysis |
| **`get_coherence_state`** (NEW) | Returns HIGH-LEVEL coherence summary: order_parameter, coherence_level enum, is_broadcasting, has_conflict |

---

## CURRENT PROJECT STATE (Verified 2026-01-13)

### Completed Dependencies:
- **TASK-10**: `KURAMOTO_N = 13` constant exists at `crates/context-graph-core/src/layers/coherence/constants.rs:17`
- **TASK-11**: `KuramotoNetwork` implemented at `crates/context-graph-core/src/layers/coherence/network.rs`
- **TASK-12**: `KuramotoStepper` wired to MCP lifecycle at `crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs`

### Files That ALREADY EXIST (DO NOT CREATE):
```
crates/context-graph-mcp/src/tools/names.rs           # Tool name constants
crates/context-graph-mcp/src/tools/definitions/gwt.rs # Existing GWT tool definitions
crates/context-graph-mcp/src/handlers/tools/dispatch.rs # Tool dispatch
crates/context-graph-mcp/src/handlers/tools/gwt_consciousness.rs # Existing GWT handlers
crates/context-graph-mcp/src/handlers/gwt_traits.rs   # GWT provider traits
crates/context-graph-mcp/src/handlers/gwt_providers.rs # GWT provider implementations
```

### Key Existing Types/Traits (USE THESE):
- `KuramotoProvider` trait: `crates/context-graph-mcp/src/handlers/gwt_traits.rs`
  - `order_parameter(&self) -> (f64, f64)` - Returns (r, psi)
  - `phases(&self) -> [f32; 13]` - All 13 oscillator phases
  - `synchronization(&self) -> f64` - Same as r but f64
- `ConsciousnessState`: `crates/context-graph-core/src/gwt/state_machine.rs`
  - `from_level(r: f32) -> Self` - Classifies r into DORMANT/FRAGMENTED/EMERGING/CONSCIOUS/HYPERSYNC
- `WorkspaceProvider` trait: `crates/context-graph-mcp/src/handlers/gwt_traits.rs`
  - `is_broadcasting(&self) -> impl Future<Output = bool>` - Async method
  - `has_conflict(&self) -> impl Future<Output = bool>` - Async method

---

## EXACT IMPLEMENTATION STEPS

### Step 1: Add Tool Name Constant

**File**: `crates/context-graph-mcp/src/tools/names.rs`
**Location**: After line 128 (after GET_JOHARI_CLASSIFICATION)

**ADD**:
```rust
// ========== COHERENCE STATE TOOL (TASK-34) ==========

/// TASK-34: Get high-level coherence state from Kuramoto/GWT system
pub const GET_COHERENCE_STATE: &str = "get_coherence_state";
```

### Step 2: Add Tool Definition

**File**: `crates/context-graph-mcp/src/tools/definitions/gwt.rs`
**Location**: After the `adjust_coupling` definition (around line 143), inside the `definitions()` function

**ADD** (as 7th element in the Vec):
```rust
        // get_coherence_state - High-level coherence summary (TASK-34)
        ToolDefinition::new(
            "get_coherence_state",
            "Get high-level GWT workspace coherence state. Returns Kuramoto order parameter, \
             coherence level classification (High/Medium/Low), workspace broadcasting status, \
             and conflict detection status. Use get_kuramoto_sync for detailed oscillator data. \
             Requires GWT providers to be initialized via with_gwt() constructor.",
            json!({
                "type": "object",
                "properties": {
                    "include_phases": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include all 13 oscillator phases in response (optional)"
                    }
                },
                "required": []
            }),
        ),
```

**ALSO UPDATE** the function docstring from "Returns GWT tool definitions (6 tools)" to "(7 tools)".

### Step 3: Create Handler Implementation

**File**: `crates/context-graph-mcp/src/handlers/tools/gwt_consciousness.rs`
**Location**: After the `call_get_ego_state` method (around line 389), inside `impl Handlers`

**ADD**:
```rust
    /// get_coherence_state tool implementation.
    ///
    /// TASK-34: Returns high-level GWT workspace coherence state.
    /// Unlike get_kuramoto_sync (raw data) or get_consciousness_state (full state),
    /// this returns a focused coherence summary for quick status checks.
    ///
    /// FAIL FAST on missing GWT components - no stubs or fallbacks.
    ///
    /// Returns:
    /// - order_parameter: Kuramoto r in [0, 1]
    /// - coherence_level: High (r > 0.8) / Medium (0.5 <= r <= 0.8) / Low (r < 0.5)
    /// - is_broadcasting: Whether workspace is currently broadcasting
    /// - has_conflict: Whether there's a workspace conflict (two r > 0.8)
    /// - phases: Optional 13 oscillator phases (if include_phases = true)
    pub(crate) async fn call_get_coherence_state(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_coherence_state tool call");

        // Parse include_phases argument
        let include_phases = arguments
            .get("include_phases")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // FAIL FAST: Check required Kuramoto provider
        let kuramoto = match &self.kuramoto_network {
            Some(k) => k,
            None => {
                error!("get_coherence_state: Kuramoto network not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::GWT_NOT_INITIALIZED,
                    "Kuramoto network not initialized - use with_gwt() constructor",
                );
            }
        };

        // FAIL FAST: Check required workspace provider
        let workspace = match &self.workspace_provider {
            Some(w) => w,
            None => {
                error!("get_coherence_state: Workspace provider not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::GWT_NOT_INITIALIZED,
                    "Workspace provider not initialized - use with_gwt() constructor",
                );
            }
        };

        // Get Kuramoto order parameter (r, psi)
        let (r, _psi) = {
            let kuramoto_guard = kuramoto.read();
            kuramoto_guard.order_parameter()
        };

        // Classify coherence level based on r thresholds (constitution.yaml gwt.kuramoto.thresholds)
        let coherence_level = if r > 0.8 {
            "High"
        } else if r >= 0.5 {
            "Medium"
        } else {
            "Low"
        };

        // Get workspace status (async methods per TASK-07)
        let workspace_guard = workspace.read().await;
        let is_broadcasting = workspace_guard.is_broadcasting().await;
        let has_conflict = workspace_guard.has_conflict().await;
        drop(workspace_guard);

        // Optionally get phases
        let phases_json = if include_phases {
            let kuramoto_guard = kuramoto.read();
            let phases = kuramoto_guard.phases();
            Some(serde_json::json!(phases.to_vec()))
        } else {
            None
        };

        self.tool_result_with_pulse(
            id,
            json!({
                "order_parameter": r,
                "coherence_level": coherence_level,
                "is_broadcasting": is_broadcasting,
                "has_conflict": has_conflict,
                "phases": phases_json,
                "thresholds": {
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.0
                }
            }),
        )
    }
```

### Step 4: Add Dispatch Entry

**File**: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs`
**Location**: After line 161 (after `GET_JOHARI_CLASSIFICATION` dispatch), before the `_ =>` catch-all

**ADD**:
```rust
            // TASK-34: Coherence state summary
            tool_names::GET_COHERENCE_STATE => {
                self.call_get_coherence_state(id, arguments).await
            }
```

---

## VERIFICATION COMMANDS

### Compile Check:
```bash
cargo check -p context-graph-mcp 2>&1 | head -50
```

### Run Tests:
```bash
cargo test -p context-graph-mcp get_coherence --nocapture 2>&1 | head -100
cargo test -p context-graph-mcp tools_list --nocapture 2>&1 | head -50
```

### Verify Tool Registration:
```bash
cargo test -p context-graph-mcp test_tools_list_contains_all_tools --nocapture 2>&1
```

---

## FULL STATE VERIFICATION (MANDATORY)

After implementing the logic, you MUST perform Full State Verification.

### 1. Define Source of Truth

The source of truth for this tool is:
- **Kuramoto network state**: `Handlers.kuramoto_network: Arc<parking_lot::RwLock<dyn KuramotoProvider>>`
- **Workspace state**: `Handlers.workspace_provider: Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>`

### 2. Execute & Inspect

Create a test that:
1. Creates `Handlers` with `with_default_gwt()`
2. Starts the Kuramoto stepper
3. Calls `get_coherence_state` tool
4. Reads back from `kuramoto_network` and `workspace_provider` directly
5. Verifies the tool output matches the actual state

### 3. Boundary & Edge Case Audit

Manually simulate these 3 edge cases:

**Edge Case 1: GWT Not Initialized**
```rust
#[tokio::test]
async fn test_get_coherence_state_no_gwt() {
    println!("\n=== EDGE CASE 1: GWT Not Initialized ===");

    // Create handlers WITHOUT GWT (no with_gwt() call)
    let handlers = Handlers::new(/* minimal deps */);

    println!("STATE BEFORE: kuramoto_network = None, workspace_provider = None");

    let response = handlers.call_get_coherence_state(Some(JsonRpcId::Num(1)), json!({})).await;

    println!("STATE AFTER: response = {:?}", response);

    // MUST return error code GWT_NOT_INITIALIZED
    assert!(response.error.is_some());
    assert_eq!(response.error.unwrap().code, error_codes::GWT_NOT_INITIALIZED);

    println!("EVIDENCE: Tool correctly fails fast with GWT_NOT_INITIALIZED");
}
```

**Edge Case 2: Empty Phases (include_phases = false)**
```rust
#[tokio::test]
async fn test_get_coherence_state_no_phases() {
    println!("\n=== EDGE CASE 2: include_phases = false ===");

    let handlers = create_handlers_with_gwt();

    println!("STATE BEFORE: include_phases = false");

    let response = handlers.call_get_coherence_state(
        Some(JsonRpcId::Num(1)),
        json!({ "include_phases": false })
    ).await;

    let result = response.result.unwrap();
    println!("STATE AFTER: phases = {:?}", result.get("phases"));

    // phases should be null when include_phases = false
    assert!(result.get("phases").unwrap().is_null());

    println!("EVIDENCE: Phases correctly omitted when include_phases = false");
}
```

**Edge Case 3: High Coherence (r > 0.8)**
```rust
#[tokio::test]
async fn test_get_coherence_state_high_coherence() {
    println!("\n=== EDGE CASE 3: High Coherence ===");

    // Create handlers with synchronized network (r ≈ 1.0)
    let handlers = create_handlers_with_synchronized_gwt();

    let r_before = {
        let net = handlers.kuramoto_network.as_ref().unwrap().read();
        net.order_parameter().0
    };
    println!("STATE BEFORE: r = {}", r_before);

    let response = handlers.call_get_coherence_state(
        Some(JsonRpcId::Num(1)),
        json!({})
    ).await;

    let result = response.result.unwrap();
    let coherence_level = result.get("coherence_level").unwrap().as_str().unwrap();
    let order_param = result.get("order_parameter").unwrap().as_f64().unwrap();

    println!("STATE AFTER: order_parameter = {}, coherence_level = {}", order_param, coherence_level);

    // With synchronized network, r should be > 0.8 and level should be "High"
    if order_param > 0.8 {
        assert_eq!(coherence_level, "High");
        println!("EVIDENCE: High coherence correctly classified");
    } else {
        println!("NOTE: Network not synchronized enough for High, got r = {}", order_param);
    }
}
```

### 4. Evidence of Success

Your test MUST print a log showing:
```
[FSV] get_coherence_state verification:
  Tool registered: YES (in tools_list)
  Dispatch working: YES (no unknown_tool error)
  order_parameter: 0.XXXX (matches kuramoto_network.read().order_parameter().0)
  coherence_level: High/Medium/Low (matches threshold logic)
  is_broadcasting: true/false (matches workspace_provider.read().is_broadcasting())
  has_conflict: true/false (matches workspace_provider.read().has_conflict())
  phases: [13 floats] or null (matches include_phases argument)
```

---

## MANUAL VERIFICATION CHECKLIST

After implementation, verify EACH of these:

- [ ] `cargo check -p context-graph-mcp` passes with no errors
- [ ] `GET_COHERENCE_STATE` constant exists in `tools/names.rs`
- [ ] Tool definition exists in `tools/definitions/gwt.rs` (7 tools total)
- [ ] Handler method `call_get_coherence_state` exists in `handlers/tools/gwt_consciousness.rs`
- [ ] Dispatch entry exists in `handlers/tools/dispatch.rs`
- [ ] Tool appears in `tools/list` response
- [ ] Tool returns error code `GWT_NOT_INITIALIZED` when GWT not configured
- [ ] `order_parameter` is a valid float in [0, 1]
- [ ] `coherence_level` is exactly one of: "High", "Medium", "Low"
- [ ] `is_broadcasting` is a boolean
- [ ] `has_conflict` is a boolean
- [ ] `phases` is null when `include_phases` is false/omitted
- [ ] `phases` is array of 13 floats when `include_phases` is true

---

## FORBIDDEN PATTERNS (DO NOT DO)

1. **DO NOT** create a new file `handlers/coherence.rs` - add to existing `gwt_consciousness.rs`
2. **DO NOT** create `tools/definitions/coherence.rs` - add to existing `gwt.rs`
3. **DO NOT** create `tools/schemas/coherence.rs` - the task spec was WRONG, this project doesn't use a schemas directory
4. **DO NOT** return mock/stub data - query real providers
5. **DO NOT** catch and swallow errors - FAIL FAST with descriptive error codes
6. **DO NOT** create backwards compatibility shims - if something breaks, fix it properly
7. **DO NOT** use `unwrap()` - use proper error handling with `?` or match statements

---

## CONSTITUTION COMPLIANCE

This tool implements:
- `constitution.yaml gwt.kuramoto.thresholds`: coherent ≥ 0.8, fragmented < 0.5
- `constitution.yaml mcp.core_tools.gwt`: Lists `get_coherence_state` as required
- `constitution.yaml perf.latency.mcp`: <100ms response time

---

## ESTIMATED EFFORT

| Activity | Time |
|----------|------|
| Add tool name constant | 5 min |
| Add tool definition | 10 min |
| Implement handler | 45 min |
| Add dispatch entry | 5 min |
| Write tests | 60 min |
| Full State Verification | 30 min |
| Manual verification | 15 min |
| **Total** | **~2.5-3 hours** |

---

## SUCCESS CRITERIA

Task is COMPLETE when:
1. `cargo test -p context-graph-mcp` passes (all tests green)
2. Tool appears in `tools/list` response
3. Tool returns valid coherence state when GWT is initialized
4. Tool returns `GWT_NOT_INITIALIZED` error when GWT is not initialized
5. All 3 edge case tests pass with printed evidence
6. FSV log shows actual data matching source of truth

```xml
</task_spec>
```
