# Task Specification: MCP Tool Exposure

**Task ID:** TASK-IDENTITY-P0-007
**Version:** 1.0.0
**Status:** Ready
**Layer:** Surface
**Sequence:** 7
**Estimated Complexity:** Low

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-009 |
| Depends On | TASK-IDENTITY-P0-006 |
| Blocks | None (terminal task) |
| Priority | P0 - Critical |

---

## Context

The identity continuity loop is now fully implemented:
- TASK-IDENTITY-P0-001 through P0-005: Core types and logic
- TASK-IDENTITY-P0-006: GWT attention wiring via IdentityContinuityListener

This final task exposes the enhanced identity state via the existing `get_ego_state` MCP tool. The tool must now include:

1. Full IC computation state from IdentityContinuityListener
2. Crisis detection status
3. Purpose vector history length
4. Time since last crisis event

Per REQ-IDENTITY-009: "IC state MUST be accessible via get_ego_state MCP tool"

The existing `get_ego_state` implementation (tools.rs lines 1355-1418) returns:
- purpose_vector
- identity_coherence
- coherence_with_actions
- identity_status
- trajectory_length

This task enhances it with:
- crisis detection state
- previous status (for transition detection)
- time since last crisis event
- entering_critical flag
- recovering flag

---

## Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/tools.rs` | Existing get_ego_state implementation |
| `crates/context-graph-mcp/src/tools.rs` | Tool definitions |
| `specs/tasks/TASK-IDENTITY-P0-006.md` | IdentityContinuityListener |
| `crates/context-graph-core/src/gwt/ego_node.rs` | IdentityContinuityMonitor, CrisisDetectionResult |

---

## Prerequisites

- [x] TASK-IDENTITY-P0-006 completed (IdentityContinuityListener integrated)
- [x] get_ego_state tool exists and returns basic identity state
- [x] GwtSystem.identity_listener() available

---

## Scope

### In Scope

1. Enhance `call_get_ego_state()` to include crisis detection state
2. Update tool definition description for expanded response
3. Add identity continuity fields to response JSON
4. Add integration tests for enhanced response

### Out of Scope

- New MCP tools (only enhancing existing)
- Breaking changes to existing response fields
- Persistence of IC history (future task)

---

## Definition of Done

### Enhanced Response Schema

```rust
// File: crates/context-graph-mcp/src/handlers/tools.rs

/// get_ego_state enhanced response schema:
///
/// {
///   "purpose_vector": [f32; 13],
///   "identity_coherence": f32,
///   "coherence_with_actions": f32,
///   "identity_status": "Healthy" | "Warning" | "Degraded" | "Critical",
///   "trajectory_length": usize,
///   "thresholds": {
///     "healthy": 0.9,
///     "warning": 0.7,
///     "degraded": 0.5,
///     "critical": 0.0
///   },
///   // NEW FIELDS (TASK-IDENTITY-P0-007):
///   "identity_continuity": {
///     "current_ic": f32,
///     "previous_status": "Healthy" | "Warning" | "Degraded" | "Critical",
///     "status_changed": bool,
///     "entering_crisis": bool,
///     "entering_critical": bool,
///     "recovering": bool,
///     "time_since_last_event_ms": Option<u64>,
///     "can_emit_event": bool,
///     "is_first_vector": bool
///   }
/// }
```

### Exact Implementation

```rust
// File: crates/context-graph-mcp/src/handlers/tools.rs

/// get_ego_state tool implementation.
///
/// TASK-GWT-001: Returns Self-Ego Node state including purpose vector,
/// identity continuity, coherence with actions, and trajectory length.
///
/// TASK-IDENTITY-P0-007: Enhanced with full identity continuity state
/// from IdentityContinuityListener including crisis detection.
///
/// FAIL FAST on missing self-ego provider - no stubs or fallbacks.
///
/// Returns:
/// - purpose_vector: 13D purpose alignment vector
/// - identity_coherence: IC = cos(PV_t, PV_{t-1}) x r(t)
/// - coherence_with_actions: Alignment between actions and purpose
/// - identity_status: Healthy/Warning/Degraded/Critical
/// - trajectory_length: Number of purpose snapshots stored
/// - identity_continuity: Full IC state with crisis detection (NEW)
pub(super) async fn call_get_ego_state(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
    debug!("Handling get_ego_state tool call");

    // FAIL FAST: Check self-ego provider
    let self_ego = match &self.self_ego {
        Some(s) => s,
        None => {
            error!("get_ego_state: Self-ego provider not initialized");
            return JsonRpcResponse::error(
                id,
                error_codes::GWT_NOT_INITIALIZED,
                "Self-ego provider not initialized - use with_gwt() constructor",
            );
        }
    };

    // TASK-IDENTITY-P0-007: Get identity continuity listener from GWT system
    let identity_listener = match &self.gwt_system {
        Some(gwt) => Some(gwt.identity_listener()),
        None => None,
    };

    // Acquire read lock (tokio RwLock)
    let ego = self_ego.read().await;

    // Get purpose vector
    let purpose_vector = ego.purpose_vector();

    // Get identity coherence
    let identity_coherence = ego.identity_coherence();

    // Get coherence with actions
    let coherence_with_actions = ego.coherence_with_actions();

    // Get identity status
    let identity_status = ego.identity_status();

    // Get trajectory length
    let trajectory_length = ego.trajectory_length();

    // TASK-IDENTITY-P0-007: Get full identity continuity state
    let identity_continuity = if let Some(listener) = identity_listener {
        let ic = listener.identity_coherence().await;
        let status = listener.identity_status().await;
        let is_in_crisis = listener.is_in_crisis().await;

        // Get crisis detection state from the listener's monitor
        // Note: This requires the listener to expose detection state
        let detection_state = listener.last_detection().await;

        Some(json!({
            "current_ic": ic,
            "previous_status": detection_state.as_ref()
                .map(|d| format!("{:?}", d.previous_status))
                .unwrap_or_else(|| format!("{:?}", IdentityStatus::Healthy)),
            "status_changed": detection_state.as_ref()
                .map(|d| d.status_changed)
                .unwrap_or(false),
            "entering_crisis": detection_state.as_ref()
                .map(|d| d.entering_crisis)
                .unwrap_or(false),
            "entering_critical": detection_state.as_ref()
                .map(|d| d.entering_critical)
                .unwrap_or(false),
            "recovering": detection_state.as_ref()
                .map(|d| d.recovering)
                .unwrap_or(false),
            "time_since_last_event_ms": detection_state.as_ref()
                .and_then(|d| d.time_since_last_event)
                .map(|d| d.as_millis() as u64),
            "can_emit_event": detection_state.as_ref()
                .map(|d| d.can_emit_event)
                .unwrap_or(true),
            "is_in_crisis": is_in_crisis
        }))
    } else {
        // GWT not initialized - return None for identity_continuity
        None
    };

    let mut response = json!({
        "purpose_vector": purpose_vector.to_vec(),
        "identity_coherence": identity_coherence,
        "coherence_with_actions": coherence_with_actions,
        "identity_status": format!("{:?}", identity_status),
        "trajectory_length": trajectory_length,
        "thresholds": {
            "healthy": 0.9,
            "warning": 0.7,
            "degraded": 0.5,
            "critical": 0.0
        }
    });

    // Add identity_continuity if available
    if let Some(ic_state) = identity_continuity {
        response["identity_continuity"] = ic_state;
    }

    self.tool_result_with_pulse(id, response)
}
```

### IdentityContinuityListener Extension

```rust
// File: crates/context-graph-core/src/gwt/listeners.rs

impl IdentityContinuityListener {
    /// Get the last crisis detection result
    ///
    /// Returns None if no detection has been performed yet
    pub async fn last_detection(&self) -> Option<CrisisDetectionResult> {
        self.monitor.read().await.last_detection()
    }
}

// Add to IdentityContinuityMonitor:
impl IdentityContinuityMonitor {
    /// Get the last crisis detection result
    pub fn last_detection(&self) -> Option<CrisisDetectionResult> {
        self.last_detection.clone()
    }
}

// Add field to IdentityContinuityMonitor struct:
/// Last crisis detection result
last_detection: Option<CrisisDetectionResult>,
```

### Constraints

1. MUST NOT break existing get_ego_state response fields
2. MUST include identity_continuity only when GWT is initialized
3. MUST return None for identity_continuity when listener unavailable
4. MUST serialize status enums as strings for JSON compatibility
5. Time values MUST be in milliseconds for consistency
6. NO panics if GWT system is not initialized

### Verification Commands

```bash
# Build
cargo build -p context-graph-mcp

# Run enhanced get_ego_state tests
cargo test -p context-graph-mcp get_ego_state

# Run identity continuity integration tests
cargo test -p context-graph-mcp identity_continuity_mcp

# Clippy
cargo clippy -p context-graph-mcp -- -D warnings
```

---

## Pseudo Code

```rust
// In handlers/tools.rs

pub(super) async fn call_get_ego_state(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
    debug!("Handling get_ego_state tool call");

    // Check providers
    let self_ego = match &self.self_ego {
        Some(s) => s,
        None => {
            return JsonRpcResponse::error(id, GWT_NOT_INITIALIZED, "...");
        }
    };

    let identity_listener = self.gwt_system.as_ref().map(|g| g.identity_listener());

    // Get basic ego state (existing)
    let ego = self_ego.read().await;
    let purpose_vector = ego.purpose_vector();
    let identity_coherence = ego.identity_coherence();
    let coherence_with_actions = ego.coherence_with_actions();
    let identity_status = ego.identity_status();
    let trajectory_length = ego.trajectory_length();

    // NEW: Get identity continuity state
    let identity_continuity = if let Some(listener) = identity_listener {
        let detection = listener.last_detection().await;
        Some(json!({
            "current_ic": listener.identity_coherence().await,
            "previous_status": detection.map(|d| format!("{:?}", d.previous_status)),
            "status_changed": detection.map(|d| d.status_changed).unwrap_or(false),
            "entering_crisis": detection.map(|d| d.entering_crisis).unwrap_or(false),
            "entering_critical": detection.map(|d| d.entering_critical).unwrap_or(false),
            "recovering": detection.map(|d| d.recovering).unwrap_or(false),
            "time_since_last_event_ms": detection.and_then(|d| d.time_since_last_event).map(|d| d.as_millis()),
            "can_emit_event": detection.map(|d| d.can_emit_event).unwrap_or(true),
            "is_in_crisis": listener.is_in_crisis().await
        }))
    } else {
        None
    };

    // Build response
    let mut response = json!({
        "purpose_vector": purpose_vector.to_vec(),
        "identity_coherence": identity_coherence,
        "coherence_with_actions": coherence_with_actions,
        "identity_status": format!("{:?}", identity_status),
        "trajectory_length": trajectory_length,
        "thresholds": {
            "healthy": 0.9,
            "warning": 0.7,
            "degraded": 0.5,
            "critical": 0.0
        }
    });

    if let Some(ic) = identity_continuity {
        response["identity_continuity"] = ic;
    }

    self.tool_result_with_pulse(id, response)
}
```

---

## Files to Create

None - all modifications go to existing files.

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-mcp/src/handlers/tools.rs` | Enhance call_get_ego_state() |
| `crates/context-graph-mcp/src/tools.rs` | Update get_ego_state tool description |
| `crates/context-graph-core/src/gwt/listeners.rs` | Add last_detection() to IdentityContinuityListener |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add last_detection field and method to monitor |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Existing response fields unchanged | Unit test comparing old vs new |
| identity_continuity included when GWT initialized | Integration test |
| identity_continuity is null when GWT not initialized | Integration test |
| current_ic matches listener.identity_coherence() | Unit test |
| status_changed reflects actual transitions | Unit test with state changes |
| entering_critical only true for -> Critical | Unit test |
| Time values in milliseconds | Unit test |
| No panics from missing GWT | Chaos test |

---

## Test Cases

```rust
#[cfg(test)]
mod get_ego_state_enhanced_tests {
    use super::*;

    #[tokio::test]
    async fn test_response_includes_identity_continuity_when_gwt_initialized() {
        let handlers = create_handlers_with_gwt();

        let response = handlers.call_get_ego_state(Some(JsonRpcId::Number(1))).await;
        let result: serde_json::Value = serde_json::from_str(
            &response.result.unwrap()["content"][0]["text"].as_str().unwrap()
        ).unwrap();

        // Existing fields still present
        assert!(result.get("purpose_vector").is_some());
        assert!(result.get("identity_coherence").is_some());
        assert!(result.get("identity_status").is_some());
        assert!(result.get("trajectory_length").is_some());
        assert!(result.get("thresholds").is_some());

        // New identity_continuity field present
        let ic = result.get("identity_continuity").unwrap();
        assert!(ic.get("current_ic").is_some());
        assert!(ic.get("status_changed").is_some());
        assert!(ic.get("entering_crisis").is_some());
        assert!(ic.get("entering_critical").is_some());
        assert!(ic.get("recovering").is_some());
        assert!(ic.get("can_emit_event").is_some());
        assert!(ic.get("is_in_crisis").is_some());
    }

    #[tokio::test]
    async fn test_response_excludes_identity_continuity_when_gwt_not_initialized() {
        let handlers = create_handlers_without_gwt();

        let response = handlers.call_get_ego_state(Some(JsonRpcId::Number(1))).await;
        let result: serde_json::Value = serde_json::from_str(
            &response.result.unwrap()["content"][0]["text"].as_str().unwrap()
        ).unwrap();

        // Existing fields still present
        assert!(result.get("purpose_vector").is_some());

        // identity_continuity should NOT be present
        assert!(result.get("identity_continuity").is_none());
    }

    #[tokio::test]
    async fn test_backward_compatibility_existing_fields() {
        let handlers = create_handlers_with_gwt();

        let response = handlers.call_get_ego_state(Some(JsonRpcId::Number(1))).await;
        let result: serde_json::Value = serde_json::from_str(
            &response.result.unwrap()["content"][0]["text"].as_str().unwrap()
        ).unwrap();

        // Verify exact field types
        assert!(result["purpose_vector"].is_array());
        assert!(result["identity_coherence"].is_f64());
        assert!(result["coherence_with_actions"].is_f64());
        assert!(result["identity_status"].is_string());
        assert!(result["trajectory_length"].is_u64());

        // Verify thresholds structure
        let thresholds = &result["thresholds"];
        assert_eq!(thresholds["healthy"], 0.9);
        assert_eq!(thresholds["warning"], 0.7);
        assert_eq!(thresholds["degraded"], 0.5);
        assert_eq!(thresholds["critical"], 0.0);
    }

    #[tokio::test]
    async fn test_time_since_last_event_in_milliseconds() {
        let handlers = create_handlers_with_gwt_and_crisis();

        // Trigger a crisis event
        trigger_crisis(&handlers).await;

        // Wait a known amount of time
        tokio::time::sleep(Duration::from_millis(100)).await;

        let response = handlers.call_get_ego_state(Some(JsonRpcId::Number(1))).await;
        let result: serde_json::Value = serde_json::from_str(
            &response.result.unwrap()["content"][0]["text"].as_str().unwrap()
        ).unwrap();

        let ic = &result["identity_continuity"];
        let time_ms = ic["time_since_last_event_ms"].as_u64().unwrap();

        // Should be at least 100ms
        assert!(time_ms >= 100);
    }

    #[tokio::test]
    async fn test_entering_critical_flag() {
        let handlers = create_handlers_with_gwt();

        // Setup: Healthy state
        set_identity_state(&handlers, IdentityStatus::Healthy, 0.95).await;

        // Transition to Critical
        set_identity_state(&handlers, IdentityStatus::Critical, 0.30).await;

        let response = handlers.call_get_ego_state(Some(JsonRpcId::Number(1))).await;
        let result: serde_json::Value = serde_json::from_str(
            &response.result.unwrap()["content"][0]["text"].as_str().unwrap()
        ).unwrap();

        let ic = &result["identity_continuity"];
        assert_eq!(ic["entering_critical"], true);
        assert_eq!(ic["status_changed"], true);
    }

    #[tokio::test]
    async fn test_recovering_flag() {
        let handlers = create_handlers_with_gwt();

        // Setup: Critical state
        set_identity_state(&handlers, IdentityStatus::Critical, 0.30).await;

        // Recover to Degraded
        set_identity_state(&handlers, IdentityStatus::Degraded, 0.55).await;

        let response = handlers.call_get_ego_state(Some(JsonRpcId::Number(1))).await;
        let result: serde_json::Value = serde_json::from_str(
            &response.result.unwrap()["content"][0]["text"].as_str().unwrap()
        ).unwrap();

        let ic = &result["identity_continuity"];
        assert_eq!(ic["recovering"], true);
    }
}
```

---

## Tool Description Update

Update the tool description in `tools.rs`:

```rust
// File: crates/context-graph-mcp/src/tools.rs

// get_ego_state - Self-Ego Node state (TASK-GWT-001, TASK-IDENTITY-P0-007)
ToolDefinition::new(
    "get_ego_state",
    "Get Self-Ego Node state including purpose vector (13D), identity continuity, \
     coherence with actions, and trajectory length. TASK-IDENTITY-P0-007: Enhanced with \
     full identity continuity monitoring state including crisis detection, status transitions, \
     and event cooldown tracking. Used for identity monitoring and crisis detection. \
     Requires GWT providers to be initialized via with_gwt() constructor.",
    json!({
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID (optional, uses default if not provided)"
            }
        },
        "required": []
    }),
),
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
