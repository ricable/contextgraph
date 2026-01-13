# TASK-26: Wire IC monitor to emit IdentityCritical events (COMPLETED)

```xml
<task_spec id="TASK-26" version="3.0">
<metadata>
  <title>Wire IC monitor to emit IdentityCritical events</title>
  <original_id>TASK-DREAM-005</original_id>
  <status>completed</status>
  <completed_at>2026-01-13</completed_at>
  <layer>integration</layer>
  <sequence>26</sequence>
  <implements><requirement_ref>REQ-DREAM-005</requirement_ref></implements>
  <depends_on>TASK-25 (KuramotoStepper MCP integration)</depends_on>
</metadata>

<context>
## CRITICAL NOTICE FOR AI AGENTS

**THIS TASK IS COMPLETE.** All functionality was implemented incrementally across
multiple IDENTITY tasks (P0-004 through P0-007). The IC monitor to IdentityCritical
event pipeline is fully wired and tested.

### Constitution Requirements Satisfied
- **IDENTITY-007**: "IC < 0.5 → auto-trigger dream" - IMPLEMENTED
- **AP-26**: "IC<0.5 MUST trigger dream - no silent failures" - IMPLEMENTED
- **AP-37**: "IdentityContinuityMonitor MUST exist" - IMPLEMENTED
- **AP-38**: "IC<0.5 MUST auto-trigger dream" - IMPLEMENTED

### Architecture Implemented
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IC → Dream Event Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WorkspaceEvent::MemoryEnters                                               │
│         │                                                                   │
│         ▼                                                                   │
│  IdentityContinuityListener.on_event()                                      │
│         │                                                                   │
│         ├─→ Extract purpose_vector from TeleologicalFingerprint             │
│         │                                                                   │
│         ▼                                                                   │
│  IdentityContinuityMonitor.compute_continuity(pv, kuramoto_r)               │
│         │                                                                   │
│         ├─→ IC = cos(PV_t, PV_{t-1}) × r(t)                                 │
│         │                                                                   │
│         ▼                                                                   │
│  IdentityContinuityMonitor.detect_crisis()                                  │
│         │                                                                   │
│         ├─→ Track status transitions (Healthy→Warning→Critical)             │
│         │                                                                   │
│         ▼                                                                   │
│  CrisisProtocol.execute(detection)                                          │
│         │                                                                   │
│         ├─→ Record snapshot if Warning/Degraded                             │
│         ├─→ Emit IdentityCritical if Critical + cooldown allows             │
│         │                                                                   │
│         ▼                                                                   │
│  WorkspaceEventBroadcaster.broadcast(IdentityCritical)                      │
│         │                                                                   │
│         ▼                                                                   │
│  DreamEventListener.on_event(IdentityCritical)                              │
│         │                                                                   │
│         ├─→ Update TriggerManager with IC value                             │
│         ├─→ Check if IC < 0.5 threshold                                     │
│         │                                                                   │
│         ▼                                                                   │
│  DreamConsolidationCallback invoked (triggers dream cycle)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
</context>

<current_codebase_state verified="2026-01-13">
## Verified File Locations (ACTUAL PATHS)

### IdentityContinuityMonitor Implementation
```
crates/context-graph-core/src/gwt/ego_node/monitor.rs (396 lines)
├── IdentityContinuityMonitor struct (lines 90-113)
│   ├── history: PurposeVectorHistory
│   ├── last_result: Option<IdentityContinuity>
│   ├── crisis_threshold: f32
│   ├── previous_status: IdentityStatus
│   ├── last_event_time: Option<Instant>
│   └── last_detection: Option<CrisisDetectionResult>
│
├── CrisisDetectionResult struct (lines 40-59)
│   ├── identity_coherence: f32
│   ├── previous_status: IdentityStatus
│   ├── current_status: IdentityStatus
│   ├── status_changed: bool
│   ├── entering_crisis: bool
│   ├── entering_critical: bool
│   ├── recovering: bool
│   ├── time_since_last_event: Option<Duration>
│   └── can_emit_event: bool
│
└── Methods:
    ├── new() → Self
    ├── compute_continuity(pv, r, context) → IdentityContinuity
    ├── detect_crisis() → CrisisDetectionResult
    ├── mark_event_emitted()
    ├── last_detection() → Option<CrisisDetectionResult>
    └── is_in_crisis() → bool
```

### IdentityContinuityListener Implementation
```
crates/context-graph-core/src/gwt/listeners/identity.rs (264 lines)
├── IdentityContinuityListener struct (lines 43-50)
│   ├── monitor: Arc<RwLock<IdentityContinuityMonitor>>
│   ├── protocol: Arc<CrisisProtocol>
│   └── broadcaster: Arc<WorkspaceEventBroadcaster>
│
├── WorkspaceEventListener impl (lines 154-178)
│   └── on_event(&self, event: &WorkspaceEvent)
│       └── Spawns async task to process_event()
│
└── IdentityContinuityListenerInner.process_event() (lines 192-256)
    └── Handles WorkspaceEvent::MemoryEnters:
        1. Extract purpose_vector from fingerprint
        2. Compute IC via monitor.compute_continuity()
        3. Detect crisis via monitor.detect_crisis()
        4. Execute CrisisProtocol if not Healthy
        5. Broadcast IdentityCritical event if triggered
```

### DreamEventListener (IC Consumer)
```
crates/context-graph-core/src/gwt/listeners/dream.rs (782 lines)
├── DreamEventListener struct (lines 74-85)
│   ├── dream_queue: Arc<RwLock<Vec<Uuid>>>
│   ├── trigger_manager: Option<Arc<Mutex<TriggerManager>>>
│   └── consolidation_callback: Option<DreamConsolidationCallback>
│
├── on_event() handles WorkspaceEvent::IdentityCritical (lines 256-269)
│   └── Calls handle_identity_critical()
│
└── handle_identity_critical() (lines 179-225)
    1. Log IC event
    2. Update TriggerManager with IC value
    3. Check if triggers fire (IC < 0.5)
    4. Invoke consolidation callback if triggered
```

### WorkspaceEvent::IdentityCritical
```
crates/context-graph-core/src/gwt/workspace/events.rs
└── WorkspaceEvent::IdentityCritical {
        identity_coherence: f32,      // IC value [0.0, 1.0]
        previous_status: String,       // e.g., "Healthy"
        current_status: String,        // e.g., "Critical"
        reason: String,                // Human-readable cause
        timestamp: DateTime<Utc>,
    }
```

### CrisisProtocol (Event Emitter)
```
crates/context-graph-core/src/gwt/ego_node/crisis.rs
├── CrisisProtocol struct
│   └── ego_node: Arc<RwLock<SelfEgoNode>>
│
└── execute(detection, monitor) → CrisisProtocolResult
    ├── If Warning/Degraded: Record snapshot
    └── If Critical + can_emit: Create IdentityCrisisEvent
        └── IdentityCrisisEvent.to_workspace_event() → WorkspaceEvent::IdentityCritical
```

### TriggerManager (IC Threshold Checker)
```
crates/context-graph-core/src/dream/triggers.rs
├── TriggerManager struct
│   ├── config: TriggerConfig
│   ├── current_ic: f32
│   ├── last_triggered: Option<Instant>
│   └── trigger_source: Option<ExtendedTriggerReason>
│
├── update_identity_coherence(ic: f32) - Updates stored IC
├── check_triggers() → Option<ExtendedTriggerReason>
│   └── Returns IdentityCritical{ic_value} if ic < 0.5
└── mark_triggered(reason) - Starts cooldown
```
</current_codebase_state>

<completion_evidence>
## Full State Verification Evidence

### FSV 1: IC Computation Produces Correct Values
```
Source of Truth: IdentityContinuityMonitor.compute_continuity()

Test: context-graph-core tests for identity_continuity
Verification:
  - Input: PV_t = [1.0, 0.0, ...], PV_{t-1} = [1.0, 0.0, ...], r = 0.8
  - Expected: IC = cos(identical) × r = 1.0 × 0.8 = 0.8
  - Formula: IC = cos(PV_t, PV_{t-1}) × r(t)

Evidence: tests/gwt/ego_node/monitor_tests.rs
```

### FSV 2: Crisis Detection Tracks Status Transitions
```
Source of Truth: CrisisDetectionResult fields

Test: monitor.detect_crisis() after IC drops
Verification:
  - Scenario 1: Healthy (IC=0.95) → Warning (IC=0.65)
    - status_changed: true
    - entering_crisis: true
    - entering_critical: false
    - previous_status: Healthy
    - current_status: Warning

  - Scenario 2: Warning (IC=0.65) → Critical (IC=0.35)
    - status_changed: true
    - entering_crisis: false (already in crisis)
    - entering_critical: true
    - previous_status: Warning
    - current_status: Critical

Evidence: monitor.rs detect_crisis() lines 284-333
```

### FSV 3: IdentityCritical Event Emitted on Crisis
```
Source of Truth: WorkspaceEventBroadcaster.broadcast() call

Code Path:
1. IdentityContinuityListener.on_event(MemoryEnters)
2. → process_event() (line 192)
3. → monitor.compute_continuity() (line 217)
4. → monitor.detect_crisis() (line 231)
5. → protocol.execute(detection) (line 235)
6. → if protocol_result.event_emitted (line 238)
7. → broadcaster.broadcast(crisis_event.to_workspace_event()) (line 241)

Evidence: identity.rs lines 238-249
```

### FSV 4: DreamEventListener Receives and Processes IdentityCritical
```
Source of Truth: DreamEventListener.on_event() match arm

Code Path:
1. WorkspaceEvent::IdentityCritical received (line 256)
2. → handle_identity_critical() called (line 264)
3. → trigger_manager.lock() (line 198)
4. → manager.update_identity_coherence(ic) (line 201)
5. → manager.check_triggers() (line 204)
6. → if Some(trigger_reason) (line 207)
7. → callback(trigger_reason) (line 216)

Evidence: dream.rs lines 256-269, 179-225
```

### FSV 5: TriggerManager Threshold Check
```
Source of Truth: TriggerManager.check_triggers()

Configuration: TriggerConfig.ic_threshold = 0.5 (constitution default)
Logic: if current_ic < ic_threshold → return Some(IdentityCritical{ic_value})

Evidence: triggers.rs check_triggers() implementation
```

### FSV 6: End-to-End Dream Trigger on IC < 0.5
```
Source of Truth: DreamConsolidationCallback invocation

Test: test_ic_crisis_triggers_dream_consolidation (dream.rs line 408)
Steps:
1. Create TriggerManager with IC threshold 0.5
2. Create DreamEventListener with callback
3. Send IdentityCritical event with IC=0.3
4. Verify callback was invoked with IC=0.3

Result: PASS - callback_called = true, stored_ic = 0.3

Evidence: dream.rs test lines 408-464
```
</completion_evidence>

<edge_case_audit>
| Edge Case | Expected | Actual | Verified |
|-----------|----------|--------|----------|
| First PV (no previous) | IC=1.0, Healthy | IC=1.0, Healthy | PASS |
| Identical consecutive PVs | IC = r(t) | cos=1.0, IC=r | PASS |
| Orthogonal PVs | IC = 0 × r = 0 | IC=0 | PASS |
| IC exactly at 0.5 | No trigger (< not ≤) | No trigger | PASS |
| IC = 0.4999 (just below) | Trigger | Triggers | PASS |
| IC = 0.0 (minimum) | Trigger | Triggers | PASS |
| Rapid IC drops (cooldown) | Only first triggers | First only | PASS |
| No TriggerManager wired | Log only, no crash | Logs, no crash | PASS |
| No callback wired | No crash | No crash | PASS |
| Zero-vector PV | IC = 1.0 (assume continuity) | IC=1.0 | PASS |
| Recovery (Critical→Healthy) | recovering=true | recovering=true | PASS |
</edge_case_audit>

<verification_commands>
```bash
# Run all identity continuity tests
cargo test -p context-graph-core identity -- --nocapture

# Run monitor tests specifically
cargo test -p context-graph-core monitor -- --nocapture

# Run dream listener tests (includes IC triggering)
cargo test -p context-graph-core dream -- --nocapture

# Run specific FSV test for IC crisis triggering
cargo test -p context-graph-core test_ic_crisis_triggers_dream_consolidation -- --nocapture

# Verify IC threshold boundary tests
cargo test -p context-graph-core test_ic_at_threshold_no_trigger -- --nocapture
cargo test -p context-graph-core test_ic_just_below_threshold_triggers -- --nocapture

# Check compilation
cargo check -p context-graph-core

# Run all workspace tests
cargo test -p context-graph-core workspace -- --nocapture
```
</verification_commands>

<success_checklist>
- [x] IdentityContinuityMonitor exists (gwt/ego_node/monitor.rs)
- [x] IC calculation uses cosine similarity × order parameter
- [x] CrisisDetectionResult tracks all status transitions
- [x] Cooldown mechanism prevents event spam
- [x] IdentityContinuityListener wired to workspace events
- [x] CrisisProtocol emits IdentityCritical when IC < threshold
- [x] WorkspaceEvent::IdentityCritical has all required fields
- [x] DreamEventListener consumes IdentityCritical events
- [x] TriggerManager checks IC threshold (0.5 per constitution)
- [x] DreamConsolidationCallback invoked on trigger
- [x] All edge cases tested and passing
- [x] 50+ tests covering IC computation, crisis detection, event emission
</success_checklist>

<notes>
## Why This Task Was Already Complete

The original TASK-26 specification described creating an `IdentityContinuityMonitor`
in `gwt/monitors/` which was NEVER the correct location. The actual implementation
was done incrementally across these tasks:

| Task | What It Implemented |
|------|---------------------|
| TASK-IDENTITY-P0-001 | IdentityContinuity struct with IC formula |
| TASK-IDENTITY-P0-002 | PurposeVectorHistory for tracking PV sequence |
| TASK-IDENTITY-P0-003 | IdentityContinuityMonitor in gwt/ego_node/ |
| TASK-IDENTITY-P0-004 | CrisisDetectionResult with status transitions |
| TASK-IDENTITY-P0-005 | CrisisProtocol for event emission |
| TASK-IDENTITY-P0-006 | IdentityContinuityListener (workspace wiring) |
| TASK-IDENTITY-P0-007 | MCP tool exposure via last_detection() |
| TASK-24 (DREAM-003) | DreamEventListener.handle_identity_critical() |
| TASK-20/21 | TriggerManager with IC threshold checking |

## Key Architecture Differences from Original Spec

| Original Spec | Actual Implementation |
|---------------|----------------------|
| `gwt/monitors/identity_continuity.rs` | `gwt/ego_node/monitor.rs` |
| `prev_pv: Option<Vec<f32>>` | `history: PurposeVectorHistory` (1000 snapshots) |
| `event_tx: mpsc::Sender` | `WorkspaceEventBroadcaster` (async broadcast) |
| `pub async fn update()` | `compute_continuity()` + `detect_crisis()` |
| Direct event send | CrisisProtocol with cooldown |

## Why the Architecture Changed

1. **PurposeVectorHistory**: Constitution requires up to 1000 identity snapshots
   for trajectory analysis, not just previous PV

2. **CrisisProtocol**: Centralizes crisis handling logic including:
   - Snapshot recording for Warning/Degraded states
   - Event cooldown to prevent spam
   - Clean separation from monitoring

3. **WorkspaceEventBroadcaster**: Allows multiple listeners to receive events
   (DreamEventListener, NeuromodulationEventListener, etc.)

4. **ego_node location**: IC monitoring is fundamentally about self-identity,
   which belongs in the SelfEgoNode domain, not a generic "monitors" module
</notes>
</task_spec>
```

---

## Execution Verification for AI Agents

If you are an AI agent verifying this task is complete, run:

```bash
# 1. Verify IdentityContinuityMonitor exists
grep -n "pub struct IdentityContinuityMonitor" crates/context-graph-core/src/gwt/ego_node/monitor.rs

# 2. Verify IdentityContinuityListener exists and handles MemoryEnters
grep -n "WorkspaceEvent::MemoryEnters" crates/context-graph-core/src/gwt/listeners/identity.rs

# 3. Verify DreamEventListener handles IdentityCritical
grep -n "WorkspaceEvent::IdentityCritical" crates/context-graph-core/src/gwt/listeners/dream.rs

# 4. Run IC crisis triggering test
cargo test -p context-graph-core test_ic_crisis_triggers_dream_consolidation -- --nocapture

# 5. Verify IC threshold boundary
cargo test -p context-graph-core test_ic_just_below_threshold_triggers -- --nocapture

# 6. Run all identity tests (expect 20+ pass)
cargo test -p context-graph-core identity 2>&1 | grep -E "(test result|passed|failed)"

# 7. Check compilation
cargo check -p context-graph-core
```

Expected output:
- Step 1: Line ~90 shown
- Step 2: Line ~194 shown
- Step 3: Line ~256 shown
- Step 4: Test PASSES with callback invoked
- Step 5: Test PASSES (IC=0.4999 triggers)
- Step 6: `test result: ok. N passed; 0 failed` (N >= 20)
- Step 7: No errors

**If all checks pass, TASK-26 is COMPLETE. Proceed to TASK-27.**

---

## Manual Verification Checklist

For humans or AI agents who want to manually verify the implementation:

### 1. IC Computation Verification
```bash
# Open monitor.rs and verify compute_continuity() implements:
# IC = cos(PV_t, PV_{t-1}) × r(t)
cat -n crates/context-graph-core/src/gwt/ego_node/monitor.rs | head -210 | tail -30
```

### 2. Event Emission Verification
```bash
# Verify CrisisProtocol emits event when IC < threshold
grep -A 20 "if protocol_result.event_emitted" crates/context-graph-core/src/gwt/listeners/identity.rs
```

### 3. Dream Trigger Verification
```bash
# Verify DreamEventListener processes IdentityCritical
grep -A 15 "WorkspaceEvent::IdentityCritical" crates/context-graph-core/src/gwt/listeners/dream.rs
```

### 4. Database/File Physical Verification
Not applicable - this task wires in-memory event handling, no database storage.

### 5. Constitution Compliance Check
```bash
# Verify IC threshold is 0.5 per constitution
grep -r "ic_threshold\|IC_CRISIS_THRESHOLD\|0.5" crates/context-graph-core/src/dream/
grep -r "ic_threshold\|IC_CRISIS_THRESHOLD\|0.5" crates/context-graph-core/src/gwt/
```

---

## Traceability

| Requirement | Implementation | Verified |
|-------------|----------------|----------|
| REQ-DREAM-005 | IC monitor emits IdentityCritical | YES |
| IDENTITY-007 | IC < 0.5 → auto-trigger dream | YES |
| AP-26 | No silent failures on IC crisis | YES |
| AP-37 | IdentityContinuityMonitor exists | YES |
| AP-38 | IC<0.5 auto-triggers dream | YES |
| GWT-006 | Kuramoto wired to lifecycle | YES (TASK-25) |
