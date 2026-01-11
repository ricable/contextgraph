# Task Specification: GWT Attention Wiring

**Task ID:** TASK-IDENTITY-P0-006
**Version:** 1.0.0
**Status:** Ready
**Layer:** Surface
**Sequence:** 6
**Estimated Complexity:** Medium

---

## Metadata

| Field | Value |
|-------|-------|
| Implements | REQ-IDENTITY-008 |
| Depends On | TASK-IDENTITY-P0-003, TASK-IDENTITY-P0-005 |
| Blocks | TASK-IDENTITY-P0-007 |
| Priority | P0 - Critical |

---

## Context

The identity continuity system must be wired to the Global Workspace Theory (GWT) attention mechanism. When memories enter the workspace (broadcast), the system should:

1. Compute identity continuity for the new memory's context
2. Detect crisis states
3. Execute crisis protocol if needed
4. Emit appropriate workspace events

This task creates the `IdentityContinuityListener` that subscribes to workspace events and triggers the identity monitoring loop.

Per TASK-GWT-P1-002, workspace events are already wired to subsystem listeners. This task adds the identity monitoring as another listener.

---

## Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/listeners.rs` | Existing listener infrastructure |
| `crates/context-graph-core/src/gwt/workspace.rs` | WorkspaceEvent, WorkspaceBroadcaster |
| `crates/context-graph-core/src/gwt/mod.rs` | GwtSystem |
| `specs/tasks/TASK-IDENTITY-P0-003.md` | IdentityContinuityMonitor |
| `specs/tasks/TASK-IDENTITY-P0-005.md` | CrisisProtocol |

---

## Prerequisites

- [x] TASK-IDENTITY-P0-003 completed (IdentityContinuityMonitor)
- [x] TASK-IDENTITY-P0-005 completed (CrisisProtocol)
- [x] TASK-GWT-P1-002 completed (listener infrastructure)
- [x] WorkspaceEvent::MemoryEnters exists

---

## Scope

### In Scope

1. Create `IdentityContinuityListener` implementing `WorkspaceEventListener`
2. Subscribe to `MemoryEnters` events
3. Extract purpose vector from entering memory
4. Compute IC and detect crisis
5. Execute crisis protocol if needed
6. Emit `IdentityCritical` event via broadcaster
7. Integration with GwtSystem

### Out of Scope

- MCP tool exposure (TASK-IDENTITY-P0-007)
- Dream system triggering (handled by Dream subsystem)
- Persistence of IC history (future task)

---

## Definition of Done

### Exact Signatures Required

```rust
// File: crates/context-graph-core/src/gwt/listeners.rs (or ego_node.rs)

/// Listener for identity continuity monitoring on workspace events
///
/// Subscribes to MemoryEnters events and computes identity continuity
/// for each memory that enters the Global Workspace.
///
/// # Thread Safety
/// Uses Arc<RwLock> for shared state access.
pub struct IdentityContinuityListener {
    /// Identity continuity monitor
    monitor: Arc<RwLock<IdentityContinuityMonitor>>,
    /// Crisis protocol executor
    protocol: Arc<CrisisProtocol>,
    /// Reference to workspace broadcaster for emitting events
    broadcaster: Arc<RwLock<WorkspaceBroadcaster>>,
    /// Reference to Kuramoto network for r(t)
    kuramoto: Arc<RwLock<KuramotoNetwork>>,
}

impl IdentityContinuityListener {
    /// Create new listener with all dependencies
    pub fn new(
        ego_node: Arc<RwLock<SelfEgoNode>>,
        broadcaster: Arc<RwLock<WorkspaceBroadcaster>>,
        kuramoto: Arc<RwLock<KuramotoNetwork>>,
    ) -> Self;

    /// Process a workspace event
    ///
    /// For MemoryEnters events:
    /// 1. Extract purpose vector from memory fingerprint
    /// 2. Get current Kuramoto r
    /// 3. Compute identity continuity
    /// 4. Detect crisis
    /// 5. Execute crisis protocol if needed
    /// 6. Emit IdentityCritical if critical and cooldown allows
    async fn process_event(&self, event: &WorkspaceEvent) -> CoreResult<()>;

    /// Get current identity coherence
    pub async fn identity_coherence(&self) -> f32;

    /// Get current identity status
    pub async fn identity_status(&self) -> IdentityStatus;

    /// Check if in crisis
    pub async fn is_in_crisis(&self) -> bool;
}

#[async_trait]
impl WorkspaceEventListener for IdentityContinuityListener {
    fn name(&self) -> &str {
        "IdentityContinuityListener"
    }

    fn event_types(&self) -> Vec<WorkspaceEventType> {
        vec![WorkspaceEventType::MemoryEnters]
    }

    async fn on_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        self.process_event(event).await
    }
}
```

### GwtSystem Integration

```rust
// Add to GwtSystem:
impl GwtSystem {
    /// Get identity continuity listener
    pub fn identity_listener(&self) -> Arc<IdentityContinuityListener>;

    /// Get current identity coherence
    pub fn identity_coherence(&self) -> f32;

    /// Get current identity status
    pub fn identity_status(&self) -> IdentityStatus;
}
```

### Constraints

1. Listener MUST be async to avoid blocking workspace
2. MUST use Arc<RwLock> for all shared state
3. MUST extract purpose vector from memory fingerprint
4. MUST handle missing fingerprint gracefully (skip IC)
5. MUST emit IdentityCritical via broadcaster, not directly
6. Processing time MUST be < 5ms (per NFR-IDENTITY-002)
7. NO panics from any event type

### Verification Commands

```bash
# Build
cargo build -p context-graph-core

# Run listener tests
cargo test -p context-graph-core identity_continuity_listener

# Run integration tests
cargo test -p context-graph-core gwt_identity_integration

# Clippy
cargo clippy -p context-graph-core -- -D warnings
```

---

## Pseudo Code

```rust
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct IdentityContinuityListener {
    monitor: Arc<RwLock<IdentityContinuityMonitor>>,
    protocol: Arc<CrisisProtocol>,
    broadcaster: Arc<RwLock<WorkspaceBroadcaster>>,
    kuramoto: Arc<RwLock<KuramotoNetwork>>,
}

impl IdentityContinuityListener {
    pub fn new(
        ego_node: Arc<RwLock<SelfEgoNode>>,
        broadcaster: Arc<RwLock<WorkspaceBroadcaster>>,
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
                            "Memory {} entered without fingerprint, skipping IC",
                            id
                        );
                        return Ok(());
                    }
                };

                // Get Kuramoto r (use provided or fetch)
                let kuramoto_r = *order_parameter;

                // Compute identity continuity
                let mut monitor = self.monitor.write().await;
                let ic_result = monitor.compute_continuity(&pv, kuramoto_r)?;

                // Detect crisis
                let detection = monitor.detect_crisis();

                // Execute crisis protocol if needed
                if detection.current_status != IdentityStatus::Healthy {
                    let protocol_result = self.protocol.execute(detection.clone(), &mut monitor).await?;

                    // Emit event if critical and allowed
                    if protocol_result.event_emitted {
                        if let Some(crisis_event) = protocol_result.event {
                            let ws_event = crisis_event.to_workspace_event();
                            let broadcaster = self.broadcaster.write().await;
                            broadcaster.broadcast(ws_event).await?;

                            tracing::warn!(
                                "Identity crisis event emitted: IC={:.4}, status={:?}",
                                ic_result.identity_coherence,
                                ic_result.status
                            );
                        }
                    }
                }

                tracing::trace!(
                    "Identity continuity computed: IC={:.4}, status={:?}",
                    ic_result.identity_coherence,
                    ic_result.status
                );

                Ok(())
            }
            _ => Ok(()), // Ignore other event types
        }
    }

    pub async fn identity_coherence(&self) -> f32 {
        self.monitor.read().await.identity_coherence()
    }

    pub async fn identity_status(&self) -> IdentityStatus {
        self.monitor.read().await.current_status()
    }

    pub async fn is_in_crisis(&self) -> bool {
        self.monitor.read().await.is_in_crisis()
    }
}

#[async_trait]
impl WorkspaceEventListener for IdentityContinuityListener {
    fn name(&self) -> &str {
        "IdentityContinuityListener"
    }

    fn event_types(&self) -> Vec<WorkspaceEventType> {
        vec![WorkspaceEventType::MemoryEnters]
    }

    async fn on_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        self.process_event(event).await
    }
}
```

---

## Files to Create

None - additions go to existing files.

---

## Files to Modify

| File | Changes |
|------|---------|
| `crates/context-graph-core/src/gwt/listeners.rs` | Add IdentityContinuityListener |
| `crates/context-graph-core/src/gwt/mod.rs` | Add identity_listener() to GwtSystem, register listener |

---

## Validation Criteria

| Criterion | Verification Method |
|-----------|---------------------|
| Listener receives MemoryEnters events | Integration test |
| IC computed from fingerprint PV | Unit test |
| Missing fingerprint handled gracefully | Unit test |
| Crisis protocol executed for IC < 0.7 | Integration test |
| IdentityCritical emitted for IC < 0.5 | Integration test |
| Cooldown respected | Integration test |
| Processing < 5ms | Benchmark |
| No panics from edge cases | Chaos test |

---

## Test Cases

```rust
#[cfg(test)]
mod identity_continuity_listener_tests {
    use super::*;

    fn create_test_fingerprint(pv: [f32; 13]) -> TeleologicalFingerprint {
        let purpose_vector = PurposeVector::new(pv);
        TeleologicalFingerprint {
            id: Uuid::new_v4(),
            purpose_vector,
            // ... other fields with defaults
        }
    }

    fn create_memory_enters_event(fp: Option<TeleologicalFingerprint>, r: f32) -> WorkspaceEvent {
        WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            fingerprint: fp,
            order_parameter: r,
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_listener_computes_ic() {
        let ego = Arc::new(RwLock::new(SelfEgoNode::new()));
        let broadcaster = Arc::new(RwLock::new(WorkspaceBroadcaster::new()));
        let kuramoto = Arc::new(RwLock::new(KuramotoNetwork::new()));

        let listener = IdentityContinuityListener::new(
            ego,
            broadcaster,
            kuramoto,
        );

        // First event
        let fp = create_test_fingerprint([0.8; 13]);
        let event = create_memory_enters_event(Some(fp), 0.9);
        listener.on_event(&event).await.unwrap();

        // Should be healthy (first vector)
        assert_eq!(listener.identity_coherence().await, 1.0);
        assert_eq!(listener.identity_status().await, IdentityStatus::Healthy);
    }

    #[tokio::test]
    async fn test_listener_handles_missing_fingerprint() {
        let ego = Arc::new(RwLock::new(SelfEgoNode::new()));
        let broadcaster = Arc::new(RwLock::new(WorkspaceBroadcaster::new()));
        let kuramoto = Arc::new(RwLock::new(KuramotoNetwork::new()));

        let listener = IdentityContinuityListener::new(
            ego,
            broadcaster,
            kuramoto,
        );

        // Event without fingerprint
        let event = create_memory_enters_event(None, 0.9);
        let result = listener.on_event(&event).await;

        assert!(result.is_ok()); // Should not panic
    }

    #[tokio::test]
    async fn test_listener_emits_critical_event() {
        let ego = Arc::new(RwLock::new(SelfEgoNode::new()));
        let broadcaster = Arc::new(RwLock::new(WorkspaceBroadcaster::new()));
        let kuramoto = Arc::new(RwLock::new(KuramotoNetwork::new()));

        let listener = IdentityContinuityListener::new(
            ego,
            broadcaster.clone(),
            kuramoto,
        );

        // First event (healthy)
        let fp1 = create_test_fingerprint([1.0; 13]);
        let event1 = create_memory_enters_event(Some(fp1), 0.95);
        listener.on_event(&event1).await.unwrap();

        // Second event with drastically different PV and low r
        let mut fp2_pv = [0.0; 13];
        fp2_pv[0] = 1.0; // Orthogonal
        let fp2 = create_test_fingerprint(fp2_pv);
        let event2 = create_memory_enters_event(Some(fp2), 0.3);
        listener.on_event(&event2).await.unwrap();

        // Should be in crisis
        assert!(listener.is_in_crisis().await);
        assert_eq!(listener.identity_status().await, IdentityStatus::Critical);

        // Verify event was broadcast (check broadcaster's event log)
        let bc = broadcaster.read().await;
        // ... assert IdentityCritical was in events
    }

    #[tokio::test]
    async fn test_listener_ignores_other_events() {
        let ego = Arc::new(RwLock::new(SelfEgoNode::new()));
        let broadcaster = Arc::new(RwLock::new(WorkspaceBroadcaster::new()));
        let kuramoto = Arc::new(RwLock::new(KuramotoNetwork::new()));

        let listener = IdentityContinuityListener::new(
            ego,
            broadcaster,
            kuramoto,
        );

        // Send non-MemoryEnters event
        let event = WorkspaceEvent::MemoryExits {
            id: Uuid::new_v4(),
            reason: "test".to_string(),
            timestamp: Utc::now(),
        };

        let result = listener.on_event(&event).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_event_types_subscription() {
        let ego = Arc::new(RwLock::new(SelfEgoNode::new()));
        let broadcaster = Arc::new(RwLock::new(WorkspaceBroadcaster::new()));
        let kuramoto = Arc::new(RwLock::new(KuramotoNetwork::new()));

        let listener = IdentityContinuityListener::new(
            ego,
            broadcaster,
            kuramoto,
        );

        let event_types = listener.event_types();
        assert!(event_types.contains(&WorkspaceEventType::MemoryEnters));
        assert_eq!(event_types.len(), 1);
    }
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude Opus 4.5 | Initial task specification |
