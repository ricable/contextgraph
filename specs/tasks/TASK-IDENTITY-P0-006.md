# Task Specification: GWT Attention Wiring

**Task ID:** TASK-IDENTITY-P0-006
**Version:** 3.0.0
**Status:** BLOCKED - Prerequisites Not Implemented
**Layer:** Surface
**Sequence:** 6
**Estimated Complexity:** Medium

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-008, IDENTITY-006 |
| Depends On | TASK-IDENTITY-P0-003 (COMPLETED), TASK-IDENTITY-P0-004 (NOT IMPLEMENTED), TASK-IDENTITY-P0-005 (NOT IMPLEMENTED) |
| Blocks | TASK-IDENTITY-P0-007 |
| Priority | P0 - Critical |

---

## CRITICAL: Implementation Policies

### NO Backwards Compatibility
- **FAIL FAST**: Invalid input MUST return `CoreError` immediately
- **NO silent fallbacks**: Do not default to safe values
- **Robust error logging**: All errors MUST include context (file, line, function, values)
- **NO workarounds**: If something fails, it errors out with clear diagnostics

### NO Mock Data in Tests
- ALL tests MUST use real data structures
- Tests MUST verify actual integration with GWT system
- NO `#[cfg(test)]` mock implementations that hide broken functionality
- Tests must prove the system works end-to-end

---

## Codebase Audit (2026-01-12)

### CRITICAL: Dependency Status

| Task | Expected Status | Actual Status | Files |
|------|-----------------|---------------|-------|
| TASK-IDENTITY-P0-003 | COMPLETED | **COMPLETED** | `ego_node/monitor.rs` |
| TASK-IDENTITY-P0-004 | COMPLETED | **NOT IMPLEMENTED** | CrisisDetectionResult missing |
| TASK-IDENTITY-P0-005 | COMPLETED | **NOT IMPLEMENTED** | CrisisProtocol missing |

### What EXISTS (Verified):

| Component | Location | Status |
|-----------|----------|--------|
| `IdentityContinuityMonitor` | `gwt/ego_node/monitor.rs` | EXISTS - Has `compute_continuity()`, `is_in_crisis()`, `identity_coherence()` |
| `IdentityContinuity` | `gwt/ego_node/identity_continuity.rs` | EXISTS - IC calculation result struct |
| `IdentityStatus` | `gwt/ego_node/types.rs` | EXISTS - Healthy/Warning/Degraded/Critical enum |
| `SelfEgoNode` | `gwt/ego_node/self_ego_node.rs` | EXISTS - System identity node |
| `cosine_similarity_13d` | `gwt/ego_node/cosine.rs` | EXISTS - Public function |
| `PurposeVectorHistory` | `gwt/ego_node/purpose_vector_history.rs` | EXISTS - Ring buffer |
| `WorkspaceEvent` | `gwt/workspace/events.rs` | EXISTS - Has MemoryEnters, IdentityCritical |
| `WorkspaceEventBroadcaster` | `gwt/workspace/events.rs` | EXISTS - Broadcasts to listeners |
| `WorkspaceEventListener` trait | `gwt/workspace/events.rs` | EXISTS - `on_event(&self, event: &WorkspaceEvent)` |
| `DreamEventListener` | `gwt/listeners/dream.rs` | EXISTS - Queues exiting memories |
| `NeuromodulationEventListener` | `gwt/listeners/neuromod.rs` | EXISTS - DA boost on entry |
| `MetaCognitiveEventListener` | `gwt/listeners/meta_cognitive.rs` | EXISTS - Epistemic action trigger |
| `IC_CRISIS_THRESHOLD` | `gwt/ego_node/types.rs` | EXISTS - 0.7 |
| `IC_CRITICAL_THRESHOLD` | `gwt/ego_node/types.rs` | EXISTS - 0.5 |
| `TeleologicalFingerprint` | `types/fingerprint/teleological/types.rs` | EXISTS - Has `purpose_vector: PurposeVector` |
| `PurposeVector` | `types/fingerprint/purpose.rs` | EXISTS - `alignments: [f32; 13]` |
| `KuramotoNetwork` | `layers/coherence/network.rs` | EXISTS - Has `order_parameter()` |

### What DOES NOT EXIST (Missing from P0-004 and P0-005):

| Component | Expected Location | Status |
|-----------|-------------------|--------|
| `CrisisDetectionResult` | `gwt/ego_node/` | **DOES NOT EXIST** |
| `detect_crisis()` method | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `previous_status` field | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `last_event_time` field | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `mark_event_emitted()` | `IdentityContinuityMonitor` | **DOES NOT EXIST** |
| `CRISIS_EVENT_COOLDOWN` | `gwt/ego_node/types.rs` | **DOES NOT EXIST** |
| `CrisisProtocol` | `gwt/ego_node/` | **DOES NOT EXIST** |
| `CrisisProtocolResult` | `gwt/ego_node/` | **DOES NOT EXIST** |
| `CrisisAction` enum | `gwt/ego_node/` | **DOES NOT EXIST** |
| `IdentityCrisisEvent` | `gwt/ego_node/` | **DOES NOT EXIST** |
| `IdentityContinuityListener` | `gwt/listeners/` | **DOES NOT EXIST** |

### WorkspaceEvent::MemoryEnters Current State

```rust
// File: crates/context-graph-core/src/gwt/workspace/events.rs (lines 11-17)
WorkspaceEvent::MemoryEnters {
    id: Uuid,
    order_parameter: f32,
    timestamp: DateTime<Utc>,
}
// MISSING: fingerprint: Option<TeleologicalFingerprint>
```

### WorkspaceEvent::IdentityCritical Current State

```rust
// File: crates/context-graph-core/src/gwt/workspace/events.rs (lines 36-40)
IdentityCritical {
    identity_coherence: f32,
    reason: String,
    timestamp: DateTime<Utc>,
}
// NOTE: Missing previous_status and current_status fields
```

---

## Context

This task wires identity continuity monitoring to the GWT workspace attention mechanism. When memories enter the Global Workspace (broadcast), the system:

1. Extracts purpose vector from entering memory's teleological fingerprint
2. Computes identity continuity via `IdentityContinuityMonitor`
3. Detects crisis states via `detect_crisis()` (from P0-004)
4. Executes crisis protocol if needed (from P0-005)
5. Emits `WorkspaceEvent::IdentityCritical` if IC < 0.5

**Per constitution.yaml (GWT section lines 365-392):**
- IC = cos(PV_t, PV_{t-1}) x r(t)
- IC < 0.7: Warning/Degraded (record snapshot)
- IC < 0.5: Critical (emit IdentityCritical event, trigger dream)

---

## Prerequisites - MUST BE COMPLETED FIRST

### 1. TASK-IDENTITY-P0-004 (Crisis Detection) - NOT IMPLEMENTED

**Must add to `IdentityContinuityMonitor`:**

```rust
// Add to ego_node/types.rs
pub const CRISIS_EVENT_COOLDOWN: Duration = Duration::from_secs(30);

// Add struct to ego_node/monitor.rs
#[derive(Debug, Clone, PartialEq)]
pub struct CrisisDetectionResult {
    pub identity_coherence: f32,
    pub previous_status: IdentityStatus,
    pub current_status: IdentityStatus,
    pub status_changed: bool,
    pub entering_crisis: bool,      // Healthy -> Warning/Degraded/Critical
    pub entering_critical: bool,    // Any -> Critical
    pub recovering: bool,           // Lower -> Higher status
    pub time_since_last_event: Option<Duration>,
    pub can_emit_event: bool,
}

// Add fields to IdentityContinuityMonitor
previous_status: IdentityStatus,
last_event_time: Option<Instant>,

// Add methods to IdentityContinuityMonitor
pub fn detect_crisis(&mut self) -> CrisisDetectionResult;
pub fn mark_event_emitted(&mut self);
pub fn previous_status(&self) -> IdentityStatus;
```

### 2. TASK-IDENTITY-P0-005 (Crisis Protocol) - NOT IMPLEMENTED

**Must add:**

```rust
// File: crates/context-graph-core/src/gwt/ego_node/crisis_protocol.rs

#[derive(Debug, Clone)]
pub struct CrisisProtocolResult {
    pub detection: CrisisDetectionResult,
    pub snapshot_recorded: bool,
    pub snapshot_context: Option<String>,
    pub event: Option<IdentityCrisisEvent>,
    pub event_emitted: bool,
    pub actions: Vec<CrisisAction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrisisAction {
    SnapshotRecorded { context: String },
    EventGenerated { event_type: String },
    EventSkippedCooldown { remaining: Duration },
    IntrospectionTriggered,
}

#[derive(Debug, Clone)]
pub struct IdentityCrisisEvent {
    pub identity_coherence: f32,
    pub previous_status: IdentityStatus,
    pub current_status: IdentityStatus,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

pub struct CrisisProtocol {
    ego_node: Arc<RwLock<SelfEgoNode>>,
}

impl CrisisProtocol {
    pub fn new(ego_node: Arc<RwLock<SelfEgoNode>>) -> Self;
    pub async fn execute(
        &self,
        detection: CrisisDetectionResult,
        monitor: &mut IdentityContinuityMonitor,
    ) -> CoreResult<CrisisProtocolResult>;
}

impl IdentityCrisisEvent {
    pub fn from_detection(detection: &CrisisDetectionResult, reason: impl Into<String>) -> Self;
    pub fn to_workspace_event(&self) -> WorkspaceEvent;
}
```

---

## This Task's Scope (P0-006)

### In Scope

1. **Update `WorkspaceEvent::MemoryEnters`** to include fingerprint field
2. **Create `IdentityContinuityListener`** implementing `WorkspaceEventListener`
3. **Subscribe to `MemoryEnters` events**
4. **Extract purpose vector from fingerprint**
5. **Compute IC via monitor**
6. **Execute crisis protocol if not Healthy**
7. **Emit IdentityCritical if critical and cooldown allows**
8. **Register listener with GwtSystem**
9. **Add getter methods to GwtSystem**

### Out of Scope

- MCP tool exposure (TASK-IDENTITY-P0-007)
- Dream system triggering (handled by Dream subsystem listening to IdentityCritical)
- IC persistence history (future task)

---

## Definition of Done

### 1. Update WorkspaceEvent::MemoryEnters

```rust
// File: crates/context-graph-core/src/gwt/workspace/events.rs

/// Events fired by workspace state changes
#[derive(Debug, Clone)]
pub enum WorkspaceEvent {
    /// Memory entered workspace (r crossed 0.8 upward)
    MemoryEnters {
        id: Uuid,
        order_parameter: f32,
        timestamp: DateTime<Utc>,
        /// TASK-IDENTITY-P0-006: Fingerprint for IC computation
        fingerprint: Option<TeleologicalFingerprint>,
    },
    // ... other variants unchanged
}
```

**Migration:** Update ALL callers of `MemoryEnters` to include `fingerprint` field.

### 2. Create IdentityContinuityListener

```rust
// File: crates/context-graph-core/src/gwt/listeners/identity.rs

use crate::gwt::ego_node::{
    CrisisProtocol, IdentityContinuityMonitor, IdentityStatus,
};
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventBroadcaster, WorkspaceEventListener};
use crate::gwt::ego_node::SelfEgoNode;
use crate::layers::coherence::KuramotoNetwork;
use crate::error::CoreResult;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Listener for identity continuity monitoring on workspace events
///
/// Subscribes to MemoryEnters events and computes identity continuity
/// for each memory that enters the Global Workspace.
pub struct IdentityContinuityListener {
    /// Identity continuity monitor
    monitor: Arc<RwLock<IdentityContinuityMonitor>>,
    /// Crisis protocol executor
    protocol: Arc<CrisisProtocol>,
    /// Reference to workspace broadcaster for emitting events
    broadcaster: Arc<RwLock<WorkspaceEventBroadcaster>>,
    /// Reference to Kuramoto network for r(t) (fallback)
    kuramoto: Arc<RwLock<KuramotoNetwork>>,
}

impl IdentityContinuityListener {
    /// Create new listener with all dependencies
    pub fn new(
        ego_node: Arc<RwLock<SelfEgoNode>>,
        broadcaster: Arc<RwLock<WorkspaceEventBroadcaster>>,
        kuramoto: Arc<RwLock<KuramotoNetwork>>,
    ) -> Self {
        let monitor = Arc::new(RwLock::new(IdentityContinuityMonitor::new()));
        let protocol = Arc::new(CrisisProtocol::new(ego_node));

        Self {
            monitor,
            protocol,
            broadcaster,
            kuramoto,
        }
    }

    /// Process a workspace event
    async fn process_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        match event {
            WorkspaceEvent::MemoryEnters {
                id,
                fingerprint,
                order_parameter,
                ..
            } => {
                // Extract purpose vector from fingerprint
                let pv = match fingerprint {
                    Some(fp) => fp.purpose_vector.alignments,
                    None => {
                        tracing::debug!(
                            memory_id = %id,
                            "Memory entered without fingerprint, skipping IC computation"
                        );
                        return Ok(());
                    }
                };

                // Use provided order_parameter as Kuramoto r
                let kuramoto_r = *order_parameter;

                // Compute identity continuity
                let mut monitor = self.monitor.write().await;
                let ic_result = monitor.compute_continuity(
                    &pv,
                    kuramoto_r,
                    format!("MemoryEnters:{}", id),
                );

                // Detect crisis
                let detection = monitor.detect_crisis();

                // Execute crisis protocol if not Healthy
                if detection.current_status != IdentityStatus::Healthy {
                    let protocol_result = self.protocol.execute(
                        detection.clone(),
                        &mut monitor,
                    ).await?;

                    // Emit event if critical and cooldown allows
                    if protocol_result.event_emitted {
                        if let Some(crisis_event) = protocol_result.event {
                            let ws_event = crisis_event.to_workspace_event();
                            let broadcaster = self.broadcaster.write().await;
                            broadcaster.broadcast(ws_event).await;

                            tracing::warn!(
                                ic = %ic_result.identity_coherence,
                                status = ?ic_result.status,
                                "Identity crisis event emitted"
                            );
                        }
                    }
                }

                tracing::trace!(
                    ic = %ic_result.identity_coherence,
                    status = ?ic_result.status,
                    "Identity continuity computed"
                );

                Ok(())
            }
            _ => Ok(()), // Ignore other event types
        }
    }

    /// Get current identity coherence
    pub async fn identity_coherence(&self) -> f32 {
        self.monitor
            .read()
            .await
            .identity_coherence()
            .unwrap_or(0.0)
    }

    /// Get current identity status
    pub async fn identity_status(&self) -> IdentityStatus {
        self.monitor
            .read()
            .await
            .current_status()
            .unwrap_or(IdentityStatus::Critical)
    }

    /// Check if in crisis
    pub async fn is_in_crisis(&self) -> bool {
        self.monitor.read().await.is_in_crisis()
    }
}

impl WorkspaceEventListener for IdentityContinuityListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        // Clone Arc for spawned task
        let monitor = Arc::clone(&self.monitor);
        let protocol = Arc::clone(&self.protocol);
        let broadcaster = Arc::clone(&self.broadcaster);
        let event = event.clone();

        // Process async in background
        tokio::spawn(async move {
            // Recreate self for process_event call
            let listener = IdentityContinuityListenerInner {
                monitor,
                protocol,
                broadcaster,
            };
            if let Err(e) = listener.process_event(&event).await {
                tracing::error!(error = %e, "Failed to process workspace event for IC");
            }
        });
    }
}

/// Inner struct for async processing (avoids self-reference issues)
struct IdentityContinuityListenerInner {
    monitor: Arc<RwLock<IdentityContinuityMonitor>>,
    protocol: Arc<CrisisProtocol>,
    broadcaster: Arc<RwLock<WorkspaceEventBroadcaster>>,
}

impl IdentityContinuityListenerInner {
    async fn process_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        // Same implementation as above
        // ...
    }
}
```

### 3. Update listeners/mod.rs

```rust
// File: crates/context-graph-core/src/gwt/listeners/mod.rs

mod dream;
mod identity;  // NEW
mod meta_cognitive;
mod neuromod;

pub use dream::DreamEventListener;
pub use identity::IdentityContinuityListener;  // NEW
pub use meta_cognitive::MetaCognitiveEventListener;
pub use neuromod::NeuromodulationEventListener;
```

### 4. Update GwtSystem

```rust
// File: crates/context-graph-core/src/gwt/system.rs

impl GwtSystem {
    pub async fn new(/* existing args */) -> CoreResult<Self> {
        // ... existing initialization ...

        // TASK-IDENTITY-P0-006: Create and register identity listener
        let identity_listener = Arc::new(IdentityContinuityListener::new(
            ego_node.clone(),
            broadcaster.clone(),
            kuramoto.clone(),
        ));

        // Register with broadcaster
        broadcaster.write().await.register_listener(
            Box::new(IdentityContinuityListenerWrapper::new(identity_listener.clone()))
        ).await;

        Self {
            // ... existing fields ...
            identity_listener,
        }
    }

    /// Get identity continuity listener
    pub fn identity_listener(&self) -> Arc<IdentityContinuityListener> {
        self.identity_listener.clone()
    }

    /// Get current identity coherence (sync wrapper)
    pub fn identity_coherence(&self) -> f32 {
        tokio::runtime::Handle::current()
            .block_on(self.identity_listener.identity_coherence())
    }

    /// Get current identity status (sync wrapper)
    pub fn identity_status(&self) -> IdentityStatus {
        tokio::runtime::Handle::current()
            .block_on(self.identity_listener.identity_status())
    }
}
```

---

## Constraints

1. Listener MUST be async-safe (spawn task for processing)
2. MUST use `Arc<RwLock>` for all shared state
3. MUST extract purpose vector from `fingerprint.purpose_vector.alignments`
4. MUST handle missing fingerprint gracefully (skip IC computation)
5. MUST emit IdentityCritical via broadcaster, not directly
6. Processing time MUST be < 5ms (per NFR-IDENTITY-002)
7. NO panics from any event type or input combination
8. FAIL FAST on errors with detailed logging

---

## Full State Verification (FSV) Requirements

### 1. Source of Truth

| Artifact | Location | Verification Method |
|----------|----------|---------------------|
| `IdentityContinuityListener` struct | `gwt/listeners/identity.rs` | `grep "pub struct IdentityContinuityListener"` |
| Listener registered | `GwtSystem::new()` | `grep "identity_listener" system.rs` |
| WorkspaceEvent fingerprint field | `gwt/workspace/events.rs` | `grep "fingerprint:" events.rs` |

### 2. Execute & Inspect

After implementation, verify with SEPARATE commands:

```bash
# Step 1: Build
cargo build -p context-graph-core 2>&1 | tee /tmp/p006_build.log
echo "BUILD_EXIT_CODE: $?"

# Step 2: Run tests
cargo test -p context-graph-core identity_continuity_listener 2>&1 | tee /tmp/p006_test.log
echo "TEST_EXIT_CODE: $?"

# Step 3: Clippy
cargo clippy -p context-graph-core -- -D warnings 2>&1 | tee /tmp/p006_clippy.log
echo "CLIPPY_EXIT_CODE: $?"

# Step 4: Verify structs exist
grep -n "pub struct IdentityContinuityListener" crates/context-graph-core/src/gwt/listeners/identity.rs
grep -n "fingerprint:" crates/context-graph-core/src/gwt/workspace/events.rs

# Step 5: Verify exports
grep -n "IdentityContinuityListener" crates/context-graph-core/src/gwt/listeners/mod.rs
```

### 3. Boundary & Edge Case Audit (Minimum 3)

| Edge Case | Input | Expected Output | Test Name |
|-----------|-------|-----------------|-----------|
| Missing fingerprint | `MemoryEnters { fingerprint: None }` | Skip IC, no error | `test_listener_handles_missing_fingerprint` |
| First vector | First MemoryEnters with fingerprint | IC=1.0, Healthy | `test_listener_first_vector_healthy` |
| Critical state | Orthogonal PV, r=0.3 | IC<0.5, IdentityCritical emitted | `test_listener_emits_critical_event` |
| Cooldown active | Rapid crisis events | Second event NOT emitted | `test_cooldown_prevents_spam` |
| Zero Kuramoto r | Valid PV, order_parameter=0.0 | IC=0.0, Critical | `test_zero_order_param_critical` |

### 4. Evidence of Success

After running tests, manually verify:

```bash
# Check listener was registered and received events
cargo test -p context-graph-core gwt_identity_integration -- --nocapture 2>&1 | grep -E "(IdentityContinuityListener|identity_coherence|IdentityCritical)"

# Check that IdentityCritical event was broadcast
cargo test -p context-graph-core test_listener_emits_critical_event -- --nocapture 2>&1 | grep -E "(EVIDENCE|broadcast|emitted)"
```

**Expected evidence in test output:**
```
[DEBUG] IdentityContinuityListener: Processing MemoryEnters event
[DEBUG] Identity continuity computed: IC=0.30, status=Critical
[WARN] Identity crisis event emitted: IC=0.30, status=Critical
EVIDENCE: IdentityCritical event in broadcaster queue
```

---

## Test Cases (NO MOCK DATA)

```rust
#[cfg(test)]
mod identity_continuity_listener_tests {
    use super::*;
    use crate::types::fingerprint::teleological::TeleologicalFingerprint;
    use uuid::Uuid;

    fn create_test_fingerprint(pv: [f32; 13]) -> TeleologicalFingerprint {
        // Use REAL TeleologicalFingerprint construction
        TeleologicalFingerprint::builder()
            .with_purpose_vector(PurposeVector::from_alignments(pv))
            .build()
            .expect("Failed to build test fingerprint")
    }

    fn create_memory_enters_event(
        fp: Option<TeleologicalFingerprint>,
        r: f32,
    ) -> WorkspaceEvent {
        WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            fingerprint: fp,
            order_parameter: r,
            timestamp: Utc::now(),
        }
    }

    async fn setup_listener() -> (
        Arc<IdentityContinuityListener>,
        Arc<RwLock<WorkspaceEventBroadcaster>>,
    ) {
        let ego = Arc::new(RwLock::new(SelfEgoNode::new()));
        let broadcaster = Arc::new(RwLock::new(WorkspaceEventBroadcaster::new()));
        let kuramoto = Arc::new(RwLock::new(KuramotoNetwork::new(13, 0.5)));

        let listener = Arc::new(IdentityContinuityListener::new(
            ego,
            broadcaster.clone(),
            kuramoto,
        ));

        (listener, broadcaster)
    }

    #[tokio::test]
    async fn test_listener_first_vector_healthy() {
        println!("=== FSV: First vector returns Healthy ===");

        let (listener, _) = setup_listener().await;

        // BEFORE
        let before_ic = listener.identity_coherence().await;
        println!("BEFORE: IC = {}", before_ic);

        // EXECUTE
        let fp = create_test_fingerprint([0.8; 13]);
        let event = create_memory_enters_event(Some(fp), 0.9);
        listener.on_event(&event);

        // Wait for async processing
        tokio::time::sleep(Duration::from_millis(50)).await;

        // AFTER
        let after_ic = listener.identity_coherence().await;
        let status = listener.identity_status().await;
        println!("AFTER: IC = {}, status = {:?}", after_ic, status);

        assert!((after_ic - 1.0).abs() < 1e-6, "First vector should have IC=1.0");
        assert_eq!(status, IdentityStatus::Healthy);

        println!("EVIDENCE: First vector correctly returns Healthy");
    }

    #[tokio::test]
    async fn test_listener_handles_missing_fingerprint() {
        println!("=== FSV: Missing fingerprint handled gracefully ===");

        let (listener, _) = setup_listener().await;

        // EXECUTE
        let event = create_memory_enters_event(None, 0.9);
        listener.on_event(&event); // Should NOT panic

        tokio::time::sleep(Duration::from_millis(50)).await;

        // IC should still be default (no computation happened)
        let ic = listener.identity_coherence().await;
        println!("EVIDENCE: No panic, IC = {}", ic);
    }

    #[tokio::test]
    async fn test_listener_emits_critical_event() {
        println!("=== FSV: Critical state emits IdentityCritical ===");

        let (listener, broadcaster) = setup_listener().await;

        // Register test listener to capture broadcast events
        let captured = Arc::new(RwLock::new(Vec::new()));
        let captured_clone = captured.clone();

        struct CaptureListener {
            events: Arc<RwLock<Vec<WorkspaceEvent>>>,
        }
        impl WorkspaceEventListener for CaptureListener {
            fn on_event(&self, event: &WorkspaceEvent) {
                let events = self.events.clone();
                let event = event.clone();
                tokio::spawn(async move {
                    events.write().await.push(event);
                });
            }
        }

        broadcaster.write().await.register_listener(
            Box::new(CaptureListener { events: captured_clone })
        ).await;

        // First event (Healthy baseline)
        let fp1 = create_test_fingerprint([1.0; 13]);
        let event1 = create_memory_enters_event(Some(fp1), 0.95);
        listener.on_event(&event1);
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Second event with drastically different PV and low r
        let mut fp2_pv = [0.0; 13];
        fp2_pv[0] = 1.0; // Orthogonal
        let fp2 = create_test_fingerprint(fp2_pv);
        let event2 = create_memory_enters_event(Some(fp2), 0.3);
        listener.on_event(&event2);
        tokio::time::sleep(Duration::from_millis(100)).await;

        // VERIFY
        let status = listener.identity_status().await;
        let is_crisis = listener.is_in_crisis().await;

        println!("AFTER: status = {:?}, is_in_crisis = {}", status, is_crisis);

        assert_eq!(status, IdentityStatus::Critical);
        assert!(is_crisis);

        // Check broadcaster received IdentityCritical
        let events = captured.read().await;
        let has_critical = events.iter().any(|e| {
            matches!(e, WorkspaceEvent::IdentityCritical { .. })
        });

        assert!(has_critical, "IdentityCritical event should have been broadcast");
        println!("EVIDENCE: IdentityCritical event found in broadcaster");
    }

    #[tokio::test]
    async fn test_listener_ignores_other_events() {
        println!("=== FSV: Non-MemoryEnters events ignored ===");

        let (listener, _) = setup_listener().await;

        // Send non-MemoryEnters event
        let event = WorkspaceEvent::MemoryExits {
            id: Uuid::new_v4(),
            order_parameter: 0.5,
            timestamp: Utc::now(),
        };

        listener.on_event(&event); // Should NOT panic
        tokio::time::sleep(Duration::from_millis(50)).await;

        println!("EVIDENCE: Non-MemoryEnters event processed without error");
    }

    #[tokio::test]
    async fn test_cooldown_prevents_spam() {
        println!("=== FSV: Cooldown prevents rapid event emission ===");

        let (listener, broadcaster) = setup_listener().await;

        let event_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = event_count.clone();

        struct CountListener {
            count: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl WorkspaceEventListener for CountListener {
            fn on_event(&self, event: &WorkspaceEvent) {
                if matches!(event, WorkspaceEvent::IdentityCritical { .. }) {
                    self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }

        broadcaster.write().await.register_listener(
            Box::new(CountListener { count: count_clone })
        ).await;

        // First critical event
        let fp1 = create_test_fingerprint([1.0; 13]);
        listener.on_event(&create_memory_enters_event(Some(fp1), 0.95));
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut fp2_pv = [0.0; 13];
        fp2_pv[0] = 1.0;
        let fp2 = create_test_fingerprint(fp2_pv);
        listener.on_event(&create_memory_enters_event(Some(fp2), 0.2));
        tokio::time::sleep(Duration::from_millis(100)).await;

        let count1 = event_count.load(std::sync::atomic::Ordering::SeqCst);

        // Immediately send another critical event (should be blocked by cooldown)
        let mut fp3_pv = [0.0; 13];
        fp3_pv[1] = 1.0;
        let fp3 = create_test_fingerprint(fp3_pv);
        listener.on_event(&create_memory_enters_event(Some(fp3), 0.15));
        tokio::time::sleep(Duration::from_millis(100)).await;

        let count2 = event_count.load(std::sync::atomic::Ordering::SeqCst);

        println!("BEFORE second critical: {} events", count1);
        println!("AFTER second critical: {} events", count2);

        assert_eq!(count1, count2, "Second critical event should be blocked by cooldown");
        println!("EVIDENCE: Cooldown correctly prevented rapid event emission");
    }
}
```

---

## Implementation Checklist

### Prerequisites (P0-004 + P0-005) - MUST COMPLETE FIRST

- [ ] Add `CRISIS_EVENT_COOLDOWN` constant to `types.rs`
- [ ] Add `CrisisDetectionResult` struct to monitor
- [ ] Add `previous_status`, `last_event_time` fields to monitor
- [ ] Implement `detect_crisis()` method
- [ ] Implement `mark_event_emitted()` method
- [ ] Create `crisis_protocol.rs` with all types
- [ ] Implement `CrisisProtocol::execute()`
- [ ] Implement `IdentityCrisisEvent` with conversion methods

### This Task (P0-006)

- [ ] Update `WorkspaceEvent::MemoryEnters` to include `fingerprint` field
- [ ] Update all callers of `MemoryEnters` to include fingerprint
- [ ] Create `listeners/identity.rs` file
- [ ] Define `IdentityContinuityListener` struct
- [ ] Implement `IdentityContinuityListener::new()`
- [ ] Implement `process_event()` method
- [ ] Implement `identity_coherence()`, `identity_status()`, `is_in_crisis()` getters
- [ ] Implement `WorkspaceEventListener` trait
- [ ] Create async wrapper for event processing
- [ ] Update `listeners/mod.rs` with exports
- [ ] Add `identity_listener` field to `GwtSystem`
- [ ] Register listener in `GwtSystem::new()`
- [ ] Add `identity_listener()`, `identity_coherence()`, `identity_status()` to GwtSystem
- [ ] Add unit tests (all 5 edge cases)
- [ ] Add integration test with full event flow
- [ ] Verify processing time < 5ms
- [ ] Run clippy with `-D warnings`

### Verification

- [ ] `cargo build -p context-graph-core` succeeds
- [ ] `cargo test -p context-graph-core identity_continuity_listener` passes
- [ ] `cargo clippy -p context-graph-core -- -D warnings` clean
- [ ] Manual verification of IdentityCritical in broadcaster

---

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/listeners/identity.rs` | IdentityContinuityListener |

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/workspace/events.rs` | Add `fingerprint` field to `MemoryEnters` |
| `crates/context-graph-core/src/gwt/listeners/mod.rs` | Export `IdentityContinuityListener` |
| `crates/context-graph-core/src/gwt/system.rs` | Add `identity_listener` field and methods |
| All callers of `WorkspaceEvent::MemoryEnters` | Add `fingerprint: None` or `fingerprint: Some(fp)` |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
| 1.1.0 | 2026-01-11 | Claude Opus 4.5 | Added implementation checklist |
| 2.0.0 | 2026-01-11 | Claude Opus 4.5 | Comprehensive rewrite with GwtSystem integration |
| 3.0.0 | 2026-01-12 | Claude Opus 4.5 | **AUDIT REWRITE**: Updated prerequisites to show P0-004/P0-005 NOT IMPLEMENTED, added FSV requirements, NO mock data policy, fail-fast policy, detailed codebase state, edge case audit requirements, manual verification checklist |
