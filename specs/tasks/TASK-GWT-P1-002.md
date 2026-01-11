# TASK-GWT-P1-002: Wire Workspace Events to Subsystem Listeners

## Metadata
| Field | Value |
|-------|-------|
| **Task ID** | TASK-GWT-P1-002 |
| **Title** | Verify and Wire Workspace Event Listeners |
| **Status** | Completed |
| **Priority** | P1 |
| **Layer** | Logic (Layer 2) |
| **Parent Spec** | SPEC-GWT-001 |
| **Estimated Effort** | 3 hours |
| **Created** | 2026-01-11 |

---

## 1. Input Context Files

| File | Purpose | Key Sections |
|------|---------|--------------|
| `crates/context-graph-core/src/gwt/mod.rs` | Event broadcasting infrastructure | WorkspaceEventBroadcaster, register_listener |
| `crates/context-graph-core/src/gwt/listeners.rs` | Event listener implementations | DreamEventListener, NeuromodulationEventListener, MetaCognitiveEventListener |
| `crates/context-graph-core/src/gwt/workspace.rs` | GlobalWorkspace events | WorkspaceEvent enum |
| `docs2/constitution.yaml` | Event requirements | Lines 359-363, workspace events |

---

## 2. Problem Statement

The constitution requires that workspace events trigger specific subsystem responses:

| Event | Required Response | Subsystem |
|-------|-------------------|-----------|
| `MemoryEnters` | Boost dopamine | Neuromodulation |
| `MemoryExits` | Queue for dream replay | Dream Layer |
| `WorkspaceEmpty` | Trigger epistemic action | MetaCognitive |
| `IdentityCritical` | Trigger dream consolidation | Dream Layer |

Current state (from mod.rs tests):
- Listeners are implemented
- Tests verify individual listener behavior
- **GAP**: Need to verify listeners are registered at GwtSystem init and actually receive events

---

## 3. Definition of Done

### 3.1 Required Signatures

```rust
// In gwt/mod.rs or gwt/listeners.rs
pub trait WorkspaceEventListener: Send + Sync {
    fn on_event(&self, event: &WorkspaceEvent) -> CoreResult<()>;
    fn name(&self) -> &'static str;
}

// Event types (workspace.rs)
pub enum WorkspaceEvent {
    MemoryEnters { memory_id: Uuid, score: f32 },
    MemoryExits { memory_id: Uuid, reason: ExitReason },
    WorkspaceEmpty,
    IdentityCritical { ic_value: f32 },
}

// Broadcaster (mod.rs)
impl WorkspaceEventBroadcaster {
    pub fn register_listener(&mut self, listener: Arc<dyn WorkspaceEventListener>);
    pub async fn broadcast(&self, event: WorkspaceEvent) -> CoreResult<()>;
    pub fn listener_count(&self) -> usize;
}

// GwtSystem integration
impl GwtSystem {
    pub fn register_default_listeners(&mut self);
    pub fn get_listener_names(&self) -> Vec<&'static str>;
}
```

### 3.2 Required Tests

```rust
#[tokio::test]
async fn test_all_listeners_registered_on_init() {
    let gwt = GwtSystem::new(None).await.unwrap();
    let names = gwt.get_listener_names();
    assert!(names.contains(&"DreamEventListener"));
    assert!(names.contains(&"NeuromodulationEventListener"));
    assert!(names.contains(&"MetaCognitiveEventListener"));
    assert_eq!(names.len(), 3);
}

#[tokio::test]
async fn test_memory_enters_broadcasts_to_neuromod() {
    // Verify dopamine boost signal received
}

#[tokio::test]
async fn test_memory_exits_queues_for_dream() {
    // Verify memory queued in dream layer
}

#[tokio::test]
async fn test_workspace_empty_triggers_epistemic() {
    // Verify epistemic action triggered
}

#[tokio::test]
async fn test_identity_critical_triggers_consolidation() {
    // Verify dream consolidation started
}

#[tokio::test]
async fn test_broadcast_latency_under_5ms() {
    // Measure broadcast time to all listeners
}
```

---

## 4. Files to Create/Modify

| File | Action | Changes |
|------|--------|---------|
| `crates/context-graph-core/src/gwt/listeners.rs` | Verify/Modify | Ensure all 3 listeners implement trait correctly |
| `crates/context-graph-core/src/gwt/workspace.rs` | Verify/Modify | Ensure WorkspaceEvent has all required variants |
| `crates/context-graph-core/src/gwt/mod.rs` | Modify | Add `register_default_listeners()`, wire on init |
| `crates/context-graph-core/tests/gwt_event_tests.rs` | Create | Integration tests for event wiring |

---

## 5. Implementation Steps

### Step 1: Verify WorkspaceEvent Enum
```rust
// gwt/workspace.rs
#[derive(Debug, Clone)]
pub enum WorkspaceEvent {
    /// Memory entered the global workspace
    MemoryEnters {
        memory_id: Uuid,
        score: f32,
        timestamp: DateTime<Utc>,
    },
    /// Memory exited the workspace
    MemoryExits {
        memory_id: Uuid,
        reason: ExitReason,
        timestamp: DateTime<Utc>,
    },
    /// Workspace is empty (no competing memories)
    WorkspaceEmpty {
        duration_empty: Duration,
    },
    /// Identity continuity dropped below threshold
    IdentityCritical {
        ic_value: f32,
        ego_node_id: Uuid,
    },
}

#[derive(Debug, Clone)]
pub enum ExitReason {
    Displaced,      // Replaced by higher-scoring memory
    Timeout,        // Time in workspace expired
    Inhibited,      // Actively inhibited
    Consolidated,   // Moved to long-term storage
}
```

### Step 2: Verify Listener Implementations
```rust
// gwt/listeners.rs

pub struct DreamEventListener {
    dream_queue: Arc<Mutex<VecDeque<Uuid>>>,
}

impl WorkspaceEventListener for DreamEventListener {
    fn on_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        match event {
            WorkspaceEvent::MemoryExits { memory_id, reason, .. } => {
                // Queue for dream replay
                let mut queue = self.dream_queue.lock().unwrap();
                queue.push_back(*memory_id);
                tracing::debug!("Queued memory {} for dream replay (reason={:?})", memory_id, reason);
            }
            WorkspaceEvent::IdentityCritical { ic_value, .. } => {
                // Trigger dream consolidation
                tracing::warn!("IdentityCritical: IC={:.3}, triggering consolidation", ic_value);
                self.trigger_consolidation()?;
            }
            _ => {}
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DreamEventListener"
    }
}

pub struct NeuromodulationEventListener {
    neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
}

impl WorkspaceEventListener for NeuromodulationEventListener {
    fn on_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        match event {
            WorkspaceEvent::MemoryEnters { memory_id, score, .. } => {
                // Boost dopamine
                let mut manager = self.neuromod_manager.write().unwrap();
                manager.boost_dopamine(0.1 * score); // Score-proportional boost
                tracing::debug!("Boosted dopamine for memory {} (score={:.3})", memory_id, score);
            }
            _ => {}
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "NeuromodulationEventListener"
    }
}

pub struct MetaCognitiveEventListener {
    epistemic_trigger: Arc<AtomicBool>,
}

impl WorkspaceEventListener for MetaCognitiveEventListener {
    fn on_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        match event {
            WorkspaceEvent::WorkspaceEmpty { duration_empty } => {
                if duration_empty.num_seconds() > 5 {
                    // Trigger epistemic action (seek information)
                    self.epistemic_trigger.store(true, Ordering::SeqCst);
                    tracing::info!("Workspace empty for {}s, triggering epistemic action",
                        duration_empty.num_seconds());
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "MetaCognitiveEventListener"
    }
}
```

### Step 3: Wire Listeners in GwtSystem
```rust
// gwt/mod.rs
impl GwtSystem {
    pub async fn new(db: Option<&RocksDbHandle>) -> CoreResult<Self> {
        let mut system = Self {
            // ... initialize fields
            event_broadcaster: WorkspaceEventBroadcaster::new(),
        };

        // Register default listeners
        system.register_default_listeners();

        Ok(system)
    }

    pub fn register_default_listeners(&mut self) {
        // 1. Dream Event Listener
        let dream_listener = Arc::new(DreamEventListener::new(
            self.dream_queue.clone()
        ));
        self.event_broadcaster.register_listener(dream_listener);

        // 2. Neuromodulation Event Listener
        let neuromod_listener = Arc::new(NeuromodulationEventListener::new(
            self.neuromod_manager.clone()
        ));
        self.event_broadcaster.register_listener(neuromod_listener);

        // 3. MetaCognitive Event Listener
        let metacog_listener = Arc::new(MetaCognitiveEventListener::new(
            self.epistemic_trigger.clone()
        ));
        self.event_broadcaster.register_listener(metacog_listener);

        tracing::info!("Registered {} default event listeners",
            self.event_broadcaster.listener_count());
    }

    pub fn get_listener_names(&self) -> Vec<&'static str> {
        self.event_broadcaster.listener_names()
    }
}
```

### Step 4: Add Listener Count and Names to Broadcaster
```rust
// gwt/mod.rs or gwt/broadcaster.rs
impl WorkspaceEventBroadcaster {
    pub fn listener_count(&self) -> usize {
        self.listeners.len()
    }

    pub fn listener_names(&self) -> Vec<&'static str> {
        self.listeners.iter().map(|l| l.name()).collect()
    }
}
```

---

## 6. Validation Criteria

| Criterion | Test | Expected |
|-----------|------|----------|
| 3 listeners registered | `gwt.get_listener_names().len()` | 3 |
| Dream listener present | Names contains "DreamEventListener" | true |
| Neuromod listener present | Names contains "NeuromodulationEventListener" | true |
| MetaCog listener present | Names contains "MetaCognitiveEventListener" | true |
| MemoryEnters -> dopamine | Broadcast, check dopamine level | Increased |
| MemoryExits -> dream queue | Broadcast, check queue | Memory in queue |
| WorkspaceEmpty -> epistemic | Broadcast, check trigger | Set to true |
| IdentityCritical -> consolidation | Broadcast, check consolidation | Started |
| Broadcast latency | Time broadcast to 3 listeners | < 5ms |

### Verification Commands

```bash
# Run listener tests
cargo test --package context-graph-core listener --no-fail-fast

# Run event wiring tests
cargo test --package context-graph-core --test gwt_event_tests

# Run performance tests
cargo test --package context-graph-core broadcast_latency
```

---

## 7. Dependencies

### Upstream
- TASK-GWT-P1-001 (persistence for IdentityCritical events)

### Downstream
- TASK-GWT-P1-003 (integration tests depend on full event wiring)

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Listener panic crashes broadcast | Low | High | Catch panics, log, continue |
| Deadlock in listener | Medium | High | Timeout on listener calls |
| Event ordering issues | Low | Medium | Use sequence numbers |

---

## 9. Notes

- Existing tests in mod.rs verify individual listener behavior
- This task focuses on ensuring all listeners are wired at initialization
- Consider adding listener health checks
- Broadcast should be non-blocking (spawn tasks for each listener)
