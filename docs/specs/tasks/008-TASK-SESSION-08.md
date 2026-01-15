# TASK-SESSION-08: CLI Dream Trigger Integration

```xml
<task_spec id="TASK-SESSION-08" version="2.0">
<metadata>
  <title>CLI Dream Trigger Integration</title>
  <status>completed</status>
  <layer>logic</layer>
  <sequence>8</sequence>
  <implements>
    <requirement_ref>REQ-SESSION-08</requirement_ref>
    <constitution_ref>AP-26</constitution_ref>
    <constitution_ref>AP-38</constitution_ref>
    <constitution_ref>AP-42</constitution_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SESSION-07</task_ref>
  </depends_on>
  <estimated_hours>1.5</estimated_hours>
  <last_audit>2026-01-15</last_audit>
</metadata>
```

## Current Codebase State (Verified 2026-01-15)

### CRITICAL DISCOVERY: MCP Tools Already Exist

The following infrastructure is **already implemented and tested**:

| Component | Location | Status |
|-----------|----------|--------|
| `trigger_dream` MCP tool | `crates/context-graph-mcp/src/handlers/dream/handlers.rs:61` | ✅ IMPLEMENTED |
| `trigger_mental_check` MCP tool | `crates/context-graph-mcp/src/handlers/dream/handlers.rs:692` | ✅ IMPLEMENTED |
| TriggerManager | `crates/context-graph-core/src/dream/triggers.rs` | ✅ IMPLEMENTED |
| `is_ic_crisis(ic: f32) -> bool` | `crates/context-graph-core/src/gwt/session_identity/manager.rs:175` | ✅ IMPLEMENTED |
| `classify_ic(ic: f32) -> &'static str` | `crates/context-graph-core/src/gwt/session_identity/manager.rs:158` | ✅ IMPLEMENTED |
| `ExtendedTriggerReason::IdentityCritical` | `crates/context-graph-core/src/dream/types.rs` | ✅ IMPLEMENTED |

### What Actually Needs Implementation

This task creates a **CLI command** that:
1. Checks IC after restore (TASK-SESSION-12 will call this)
2. Calls `is_ic_crisis()` from session_identity
3. If crisis detected, calls `trigger_dream` MCP tool
4. Outputs to stderr per constitution

**The task is NOT about creating a new dream_trigger module - it's about CLI integration.**

## Objective

Implement CLI command `consciousness check-identity --auto-dream` that:
1. Reads current IC from IdentityCache
2. Uses `is_ic_crisis()` to check threshold
3. Fires `trigger_dream` MCP tool if IC < 0.5
4. Returns structured JSON output for hooks integration

## Input Context Files

```xml
<input_context_files>
  <file purpose="is_ic_crisis function">crates/context-graph-core/src/gwt/session_identity/manager.rs</file>
  <file purpose="MCP tool handlers">crates/context-graph-mcp/src/handlers/dream/handlers.rs</file>
  <file purpose="TriggerManager">crates/context-graph-core/src/dream/triggers.rs</file>
  <file purpose="IdentityCache">crates/context-graph-core/src/gwt/session_identity/cache.rs</file>
  <file purpose="CLI structure">crates/context-graph-cli/src/main.rs</file>
</input_context_files>
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-cli/src/commands/consciousness/check_identity.rs` | CLI command implementation |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-cli/src/commands/consciousness/mod.rs` | Add check_identity subcommand |
| `crates/context-graph-cli/src/commands/mod.rs` | Ensure consciousness module exported |

## Implementation Steps

### Step 1: Create CheckIdentityArgs

Location: `crates/context-graph-cli/src/commands/consciousness/check_identity.rs`

```rust
//! consciousness check-identity CLI command
//!
//! TASK-SESSION-08: Implements AP-26/AP-38 auto-dream trigger.
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.
//!
//! # Constitution Reference
//! - AP-26: IC<0.5 MUST trigger dream - no silent failures
//! - AP-38: IC<0.5 MUST auto-trigger dream
//! - AP-42: entropy>0.7 MUST wire to TriggerManager

use clap::Args;
use serde::Serialize;
use tracing::{error, info, warn};

use context_graph_core::gwt::session_identity::{classify_ic, is_ic_crisis, IdentityCache};

/// Arguments for `consciousness check-identity` command.
#[derive(Args, Debug)]
pub struct CheckIdentityArgs {
    /// Auto-trigger dream consolidation if IC < 0.5.
    /// Per AP-26 and AP-38: IC crisis MUST trigger dream automatically.
    #[arg(long, default_value = "false")]
    pub auto_dream: bool,

    /// Output format (json for hooks, human for interactive).
    #[arg(long, default_value = "json")]
    pub format: OutputFormat,

    /// Entropy value to check for mental_check trigger (optional).
    /// Per AP-42: entropy > 0.7 triggers mental_check.
    #[arg(long)]
    pub entropy: Option<f64>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    Json,
    Human,
}

/// Response from check-identity command.
#[derive(Debug, Serialize)]
pub struct CheckIdentityResponse {
    /// Current IC value (from cache).
    pub ic: f32,
    /// IC classification per IDENTITY-002.
    pub status: &'static str,
    /// Whether IC < 0.5 (identity crisis).
    pub is_crisis: bool,
    /// Whether dream was triggered (only true if --auto-dream and is_crisis).
    pub dream_triggered: bool,
    /// Trigger rationale (if triggered).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_rationale: Option<String>,
    /// Error message (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl CheckIdentityResponse {
    pub fn healthy(ic: f32) -> Self {
        Self {
            ic,
            status: classify_ic(ic),
            is_crisis: false,
            dream_triggered: false,
            trigger_rationale: None,
            error: None,
        }
    }

    pub fn crisis_no_trigger(ic: f32) -> Self {
        Self {
            ic,
            status: classify_ic(ic),
            is_crisis: true,
            dream_triggered: false,
            trigger_rationale: None,
            error: None,
        }
    }

    pub fn crisis_triggered(ic: f32, rationale: String) -> Self {
        Self {
            ic,
            status: classify_ic(ic),
            is_crisis: true,
            dream_triggered: true,
            trigger_rationale: Some(rationale),
            error: None,
        }
    }

    pub fn error(ic: f32, msg: String) -> Self {
        Self {
            ic,
            status: classify_ic(ic),
            is_crisis: is_ic_crisis(ic),
            dream_triggered: false,
            trigger_rationale: None,
            error: Some(msg),
        }
    }
}
```

### Step 2: Implement check_identity_command

Add to the same file after the structs:

```rust
use context_graph_mcp::handlers::Handlers;
use context_graph_mcp::protocol::JsonRpcRequest;

/// Execute the check-identity command.
///
/// # Fail Fast Policy
/// - If IdentityCache is empty: Error immediately
/// - If MCP call fails: Propagate error with details
/// - Never return default IC values
///
/// # Returns
/// Exit code per AP-26:
/// - 0: Success (no crisis, or crisis + dream triggered)
/// - 1: Error (MCP failure, cache empty)
/// - 2: Corruption detected (reserved, not used here)
pub async fn check_identity_command(args: CheckIdentityArgs, handlers: &Handlers) -> i32 {
    // STEP 1: Read IC from IdentityCache
    let cached = IdentityCache::get_cached();

    let ic = match cached {
        Some(data) => data.ic,
        None => {
            // FAIL FAST: Cache empty means restore_identity was not called
            error!("check-identity: IdentityCache is empty - call 'session restore-identity' first");
            let response = CheckIdentityResponse {
                ic: 0.0,
                status: "Unknown",
                is_crisis: false,
                dream_triggered: false,
                trigger_rationale: None,
                error: Some("IdentityCache empty - restore identity first".to_string()),
            };
            output_response(&response, args.format);
            return 1; // Error exit code
        }
    };

    // STEP 2: Check for IC crisis
    let is_crisis = is_ic_crisis(ic);

    if !is_crisis {
        // Healthy or warning - no action needed
        info!("check-identity: IC={:.3} status={} - no crisis", ic, classify_ic(ic));
        let response = CheckIdentityResponse::healthy(ic);
        output_response(&response, args.format);
        return 0;
    }

    // STEP 3: IC crisis detected
    warn!("check-identity: IC CRISIS detected IC={:.3} < 0.5", ic);

    if !args.auto_dream {
        // Crisis but --auto-dream not set
        eprintln!("IC crisis ({:.2}), --auto-dream not set, dream NOT triggered", ic);
        let response = CheckIdentityResponse::crisis_no_trigger(ic);
        output_response(&response, args.format);
        return 0; // Success but no action
    }

    // STEP 4: Auto-dream enabled, trigger via MCP
    let rationale = format!("IC crisis: {:.3} (threshold: 0.5)", ic);
    eprintln!("IC crisis ({:.2}), dream triggered via MCP", ic);

    // Call trigger_dream MCP tool
    let request_args = serde_json::json!({
        "rationale": rationale,
        "phase": "full_cycle",
        "force": false
    });

    let mcp_result = handlers
        .call_trigger_dream(None, request_args)
        .await;

    // STEP 5: Check MCP result
    if mcp_result.is_error() {
        error!("check-identity: trigger_dream MCP call failed: {:?}", mcp_result);
        let response = CheckIdentityResponse::error(
            ic,
            format!("MCP trigger_dream failed: {:?}", mcp_result),
        );
        output_response(&response, args.format);
        return 1; // Error exit code
    }

    info!("check-identity: Dream triggered successfully, rationale='{}'", rationale);
    let response = CheckIdentityResponse::crisis_triggered(ic, rationale);
    output_response(&response, args.format);
    0 // Success
}

/// Output response in requested format.
fn output_response(response: &CheckIdentityResponse, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            // JSON to stdout for hooks integration
            println!("{}", serde_json::to_string(response).unwrap());
        }
        OutputFormat::Human => {
            println!("Identity Continuity Check");
            println!("========================");
            println!("IC Value:        {:.3}", response.ic);
            println!("Status:          {}", response.status);
            println!("Crisis:          {}", if response.is_crisis { "YES" } else { "No" });
            println!("Dream Triggered: {}", if response.dream_triggered { "YES" } else { "No" });
            if let Some(ref rationale) = response.trigger_rationale {
                println!("Rationale:       {}", rationale);
            }
            if let Some(ref error) = response.error {
                eprintln!("Error:           {}", error);
            }
        }
    }
}
```

### Step 3: Add Entropy Check (AP-42)

Add after the IC check, before returning:

```rust
/// Check entropy and trigger mental_check if needed (AP-42).
async fn check_entropy_trigger(
    handlers: &Handlers,
    entropy: f64,
) -> Option<String> {
    // AP-42: entropy > 0.7 triggers mental_check
    if entropy <= 0.7 {
        return None;
    }

    let rationale = format!("High entropy: {:.3} (threshold: 0.7)", entropy);
    eprintln!("High entropy ({:.2}), mental_check triggered via MCP", entropy);

    let request_args = serde_json::json!({
        "entropy": entropy,
        "force": false,
        "phase": "full_cycle"
    });

    let mcp_result = handlers
        .call_trigger_mental_check(None, request_args)
        .await;

    if mcp_result.is_error() {
        error!("check-identity: trigger_mental_check failed: {:?}", mcp_result);
        return None;
    }

    Some(rationale)
}
```

### Step 4: Update mod.rs

File: `crates/context-graph-cli/src/commands/consciousness/mod.rs`

Add:
```rust
mod check_identity;
pub use check_identity::{check_identity_command, CheckIdentityArgs, CheckIdentityResponse};
```

## Constitution Requirements

| Requirement | Implementation | Verification |
|-------------|----------------|--------------|
| AP-26 | Exit code 1 on error, never silent failures | Test with empty cache |
| AP-38 | `--auto-dream` triggers `trigger_dream` MCP | Test with IC=0.4 |
| AP-42 | `--entropy` > 0.7 triggers `trigger_mental_check` | Test with entropy=0.8 |
| IDENTITY-002 | Uses `classify_ic()` for status | Verify output status field |

## Definition of Done

### Acceptance Criteria

- [ ] `consciousness check-identity` returns IC from IdentityCache
- [ ] `--auto-dream` triggers dream when IC < 0.5
- [ ] Output goes to stderr for crisis message
- [ ] JSON output to stdout for hooks integration
- [ ] Exit code 0 on success, 1 on error
- [ ] Calls `trigger_dream` MCP tool (NOT TriggerManager directly)
- [ ] Logs all operations with tracing
- [ ] `--entropy` triggers `trigger_mental_check` when > 0.7

### Constraints

- MUST call MCP `trigger_dream` tool (not TriggerManager directly)
- Crisis message MUST go to stderr
- JSON response MUST go to stdout
- Exit codes per AP-26
- NO mock data in tests - use real IdentityCache data

## Full State Verification Protocol

### 1. Source of Truth Definition

The source of truth is:
1. **IdentityCache contents** - actual IC value from last restore_identity
2. **MCP tool response** - actual trigger_dream result
3. **stderr output** - human-readable crisis message
4. **stdout output** - JSON response for hooks

### 2. Execute & Inspect Commands

```bash
# Pre-requisite: Restore identity first (sets IdentityCache)
cargo run -p context-graph-cli -- session restore-identity

# Test 1: Check identity (no crisis expected with healthy IC)
cargo run -p context-graph-cli -- consciousness check-identity --format json 2>/dev/null | jq .

# Expected output for healthy IC:
# {
#   "ic": 0.95,
#   "status": "Healthy",
#   "is_crisis": false,
#   "dream_triggered": false
# }

# Test 2: Force low IC in cache (for testing only - requires test harness)
# Then verify crisis detection and dream trigger
cargo run -p context-graph-cli -- consciousness check-identity --auto-dream --format json 2>&1

# Expected stderr for IC < 0.5:
# IC crisis (0.40), dream triggered via MCP

# Expected stdout JSON:
# {
#   "ic": 0.40,
#   "status": "Degraded",
#   "is_crisis": true,
#   "dream_triggered": true,
#   "trigger_rationale": "IC crisis: 0.400 (threshold: 0.5)"
# }
```

### 3. Boundary & Edge Case Audit

| Test Case | IC Value | --auto-dream | Expected Behavior |
|-----------|----------|--------------|-------------------|
| TC-08-01 | 0.95 | false | status="Healthy", is_crisis=false |
| TC-08-02 | 0.75 | false | status="Good", is_crisis=false |
| TC-08-03 | 0.55 | false | status="Warning", is_crisis=false |
| TC-08-04 | 0.50 | false | status="Warning", is_crisis=false (boundary) |
| TC-08-05 | 0.499 | false | status="Degraded", is_crisis=true, dream_triggered=false |
| TC-08-06 | 0.499 | true | status="Degraded", is_crisis=true, dream_triggered=true |
| TC-08-07 | 0.0 | true | status="Degraded", is_crisis=true, dream_triggered=true |
| TC-08-08 | -0.1 | true | status="Degraded", is_crisis=true (clamped) |
| TC-08-09 | NaN | N/A | FAIL FAST - error in cache read |
| TC-08-10 | Cache empty | N/A | Exit code 1, error="IdentityCache empty" |

### 4. Evidence of Success

After implementation, verify:

```bash
# 1. Build succeeds
cargo build -p context-graph-cli

# 2. Command is available
cargo run -p context-graph-cli -- consciousness check-identity --help

# 3. Run unit tests
cargo test -p context-graph-cli check_identity -- --nocapture

# 4. Integration test with real MCP
# (requires MCP server running)
./scripts/test_check_identity_integration.sh
```

## Test Cases

### TC-SESSION-08-01: Normal IC Check

```rust
#[tokio::test]
async fn tc_session_08_01_normal_ic_check() {
    // SETUP: Set known IC in cache
    use context_graph_core::gwt::session_identity::update_cache;
    update_cache(0.85, 0.9, "Good synchronization");

    // EXECUTE: Check identity
    let args = CheckIdentityArgs {
        auto_dream: false,
        format: OutputFormat::Json,
        entropy: None,
    };
    let handlers = create_test_handlers().await;
    let exit_code = check_identity_command(args, &handlers).await;

    // VERIFY
    assert_eq!(exit_code, 0);
    // Capture stdout and verify JSON
}
```

### TC-SESSION-08-02: IC Crisis with Auto-Dream

```rust
#[tokio::test]
async fn tc_session_08_02_ic_crisis_auto_dream() {
    // SETUP: Set crisis IC in cache
    update_cache(0.45, 0.6, "Fragmented");

    // EXECUTE
    let args = CheckIdentityArgs {
        auto_dream: true,
        format: OutputFormat::Json,
        entropy: None,
    };
    let handlers = create_test_handlers_with_mcp().await;
    let exit_code = check_identity_command(args, &handlers).await;

    // VERIFY
    assert_eq!(exit_code, 0);
    // Verify MCP trigger_dream was called
    // Verify stderr contains "IC crisis"
}
```

### TC-SESSION-08-03: Empty Cache Fails Fast

```rust
#[tokio::test]
async fn tc_session_08_03_empty_cache_fails_fast() {
    // SETUP: Clear cache
    context_graph_core::gwt::session_identity::clear_cache();

    // EXECUTE
    let args = CheckIdentityArgs {
        auto_dream: true,
        format: OutputFormat::Json,
        entropy: None,
    };
    let handlers = create_test_handlers().await;
    let exit_code = check_identity_command(args, &handlers).await;

    // VERIFY: FAIL FAST
    assert_eq!(exit_code, 1);
    // Verify error message in output
}
```

### TC-SESSION-08-04: Entropy Trigger (AP-42)

```rust
#[tokio::test]
async fn tc_session_08_04_entropy_trigger() {
    // SETUP
    update_cache(0.85, 0.9, "Good synchronization");

    // EXECUTE with high entropy
    let args = CheckIdentityArgs {
        auto_dream: false,
        format: OutputFormat::Json,
        entropy: Some(0.75),
    };
    let handlers = create_test_handlers_with_mcp().await;
    let exit_code = check_identity_command(args, &handlers).await;

    // VERIFY
    assert_eq!(exit_code, 0);
    // Verify trigger_mental_check was called
}
```

## Manual Testing Procedure

### Prerequisites

```bash
# 1. Build the project
cargo build -p context-graph-cli

# 2. Start MCP server (in separate terminal)
cargo run -p context-graph-mcp -- serve --port 3000
```

### Test Procedure

```bash
# Step 1: Restore identity to populate cache
cargo run -p context-graph-cli -- session restore-identity

# Step 2: Check identity (should be healthy from fresh restore)
OUTPUT=$(cargo run -p context-graph-cli -- consciousness check-identity --format json 2>/dev/null)
echo "$OUTPUT" | jq .

# Verify:
# - ic field is a number
# - status is one of: Healthy, Good, Warning, Degraded
# - is_crisis is boolean
# - No error field

# Step 3: Test with --auto-dream (safe - won't trigger if IC healthy)
cargo run -p context-graph-cli -- consciousness check-identity --auto-dream --format human

# Step 4: Test entropy trigger
cargo run -p context-graph-cli -- consciousness check-identity --entropy 0.8 --format json

# Verify: No errors, entropy check logged

# Step 5: Verify exit codes
cargo run -p context-graph-cli -- consciousness check-identity; echo "Exit: $?"
# Should be 0

# Step 6: Test with invalid state (clear cache manually if possible)
# Should exit with code 1
```

### Expected Outputs

**Healthy IC (json):**
```json
{
  "ic": 0.92,
  "status": "Healthy",
  "is_crisis": false,
  "dream_triggered": false
}
```

**IC Crisis with auto-dream (stderr + json):**
```
stderr: IC crisis (0.40), dream triggered via MCP
stdout: {"ic":0.40,"status":"Degraded","is_crisis":true,"dream_triggered":true,"trigger_rationale":"IC crisis: 0.400 (threshold: 0.5)"}
```

**Empty cache (json with error):**
```json
{
  "ic": 0.0,
  "status": "Unknown",
  "is_crisis": false,
  "dream_triggered": false,
  "error": "IdentityCache empty - restore identity first"
}
```

## Verification Commands

```bash
# Build
cargo build -p context-graph-cli

# Run specific tests
cargo test -p context-graph-cli tc_session_08 -- --nocapture

# Verify command exists
cargo run -p context-graph-cli -- consciousness --help

# Integration test
cargo run -p context-graph-cli -- consciousness check-identity --help
```

## Exit Conditions

- **Success**: Command exits 0, returns valid JSON, triggers dream on crisis
- **Error (exit 1)**: Cache empty, MCP failure
- **Corruption (exit 2)**: Reserved - not used in this command

## Failure Modes (Fail Fast)

| Failure | Exit Code | Error Message | Resolution |
|---------|-----------|---------------|------------|
| Empty cache | 1 | "IdentityCache empty" | Call restore-identity first |
| MCP timeout | 1 | "MCP trigger_dream timeout" | Check MCP server |
| Invalid IC (NaN) | 1 | "Invalid IC value in cache" | Investigate cache corruption |
| TriggerManager not init | 1 | "TriggerManager not initialized" | Configure Handlers properly |

## Next Task

After completion, proceed to **009-TASK-SESSION-09** (format_brief() Performance).

TASK-SESSION-09 depends on:
- IdentityCache from TASK-SESSION-02
- No direct dependency on this task

```xml
</task_spec>
```
