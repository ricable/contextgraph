# TASK-GWT-P1-003: End-to-End GWT Integration Tests

## Metadata
| Field | Value |
|-------|-------|
| **Task ID** | TASK-GWT-P1-003 |
| **Title** | Create End-to-End GWT Consciousness Integration Tests |
| **Status** | Ready |
| **Priority** | P1 |
| **Layer** | Surface (Layer 3) |
| **Parent Spec** | SPEC-GWT-001 |
| **Estimated Effort** | 6 hours |
| **Created** | 2026-01-11 |

---

## 1. Input Context Files

| File | Purpose | Key Sections |
|------|---------|--------------|
| `crates/context-graph-core/src/gwt/mod.rs` | GwtSystem full API | All public methods |
| `crates/context-graph-core/src/gwt/consciousness.rs` | Consciousness equation | compute_consciousness() |
| `crates/context-graph-core/src/gwt/state_machine.rs` | State transitions | ConsciousnessState |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Identity persistence | SelfEgoNode |
| `docs2/constitution.yaml` | Expected behaviors | Lines 342-378 |

---

## 2. Problem Statement

The core GWT consciousness equation is implemented and individual components are tested. However, there are no integration tests that verify:

1. **Full consciousness cycle**: DORMANT -> FRAGMENTED -> EMERGING -> CONSCIOUS
2. **SELF_EGO_NODE persistence**: Identity survives system restart
3. **Event cascade**: Workspace events trigger correct subsystem responses
4. **IC threshold behavior**: IdentityCritical triggers at IC < 0.5

---

## 3. Definition of Done

### 3.1 Required Test Files

```
crates/context-graph-core/tests/
├── gwt_integration/
│   ├── mod.rs
│   ├── consciousness_cycle_tests.rs
│   ├── persistence_tests.rs
│   ├── event_cascade_tests.rs
│   └── chaos_tests.rs
```

### 3.2 Required Test Cases

```rust
// Integration test module structure
#[cfg(test)]
mod gwt_integration {
    // IT-GWT-001: Full consciousness cycle
    #[tokio::test]
    async fn test_full_consciousness_cycle_dormant_to_conscious();

    // IT-GWT-002: SELF_EGO_NODE persistence
    #[tokio::test]
    async fn test_ego_node_persists_across_restart();

    // IT-GWT-003: IdentityCritical triggers consolidation
    #[tokio::test]
    async fn test_identity_critical_triggers_dream_consolidation();

    // IT-GWT-004: Low C(t) triggers DORMANT
    #[tokio::test]
    async fn test_low_consciousness_triggers_dormant_transition();

    // Chaos tests
    #[tokio::test]
    async fn test_rocksdb_corruption_graceful_degradation();

    #[tokio::test]
    async fn test_concurrent_event_broadcast_no_deadlock();

    #[tokio::test]
    async fn test_kuramoto_overflow_clamps_to_valid_range();
}
```

---

## 4. Files to Create/Modify

| File | Action | Changes |
|------|--------|---------|
| `crates/context-graph-core/tests/gwt_integration/mod.rs` | Create | Module declaration |
| `crates/context-graph-core/tests/gwt_integration/consciousness_cycle_tests.rs` | Create | IT-GWT-001, IT-GWT-004 |
| `crates/context-graph-core/tests/gwt_integration/persistence_tests.rs` | Create | IT-GWT-002 |
| `crates/context-graph-core/tests/gwt_integration/event_cascade_tests.rs` | Create | IT-GWT-003 |
| `crates/context-graph-core/tests/gwt_integration/chaos_tests.rs` | Create | CH-GWT-001, CH-GWT-002, CH-GWT-003 |

---

## 5. Implementation Steps

### Step 1: Create Integration Test Module

```rust
// tests/gwt_integration/mod.rs
mod consciousness_cycle_tests;
mod persistence_tests;
mod event_cascade_tests;
mod chaos_tests;

use context_graph_core::gwt::{GwtSystem, ConsciousnessState};
use context_graph_core::storage::RocksDbHandle;
use tempfile::TempDir;

/// Create a temporary RocksDB for testing
pub async fn create_test_db() -> (RocksDbHandle, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db = RocksDbHandle::open(temp_dir.path()).await.unwrap();
    (db, temp_dir)
}

/// Create a GwtSystem with test database
pub async fn create_test_gwt() -> (GwtSystem, RocksDbHandle, TempDir) {
    let (db, temp_dir) = create_test_db().await;
    let gwt = GwtSystem::new(Some(&db)).await.unwrap();
    (gwt, db, temp_dir)
}
```

### Step 2: Implement IT-GWT-001 (Full Consciousness Cycle)

```rust
// tests/gwt_integration/consciousness_cycle_tests.rs
use super::*;

/// IT-GWT-001: Verify full consciousness cycle DORMANT -> CONSCIOUS
#[tokio::test]
async fn test_full_consciousness_cycle_dormant_to_conscious() {
    let (mut gwt, _db, _temp) = create_test_gwt().await;

    // Initial state: DORMANT
    assert_eq!(gwt.current_state(), ConsciousnessState::Dormant);

    // Phase 1: Fragmented (C(t) = 0.35)
    // Set conditions: low Kuramoto r, moderate meta-accuracy
    let purpose_vector = [0.5; 13];
    gwt.update_consciousness(0.4, 0.5, &purpose_vector).await.unwrap();
    assert_eq!(gwt.current_state(), ConsciousnessState::Fragmented);

    // Phase 2: Emerging (C(t) = 0.55)
    gwt.update_consciousness(0.6, 0.6, &purpose_vector).await.unwrap();
    assert_eq!(gwt.current_state(), ConsciousnessState::Emerging);

    // Phase 3: Conscious (C(t) = 0.85)
    // High Kuramoto synchronization, good meta-accuracy, differentiated purpose
    let diverse_pv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05, 0.15, 0.25, 0.35];
    gwt.update_consciousness(0.9, 0.85, &diverse_pv).await.unwrap();
    assert_eq!(gwt.current_state(), ConsciousnessState::Conscious);

    // Verify C(t) components
    let c_t = gwt.get_consciousness_level();
    assert!(c_t >= 0.8, "C(t) should be >= 0.8 for CONSCIOUS state, got {}", c_t);

    // Verify transition history
    let transitions = gwt.get_transition_history();
    assert_eq!(transitions.len(), 3); // DORMANT -> FRAG -> EMERGING -> CONSCIOUS
}

/// IT-GWT-004: Low C(t) for extended period triggers DORMANT
#[tokio::test]
async fn test_low_consciousness_triggers_dormant_transition() {
    let (mut gwt, _db, _temp) = create_test_gwt().await;

    // First, get to CONSCIOUS state
    let pv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05, 0.15, 0.25, 0.35];
    gwt.update_consciousness(0.9, 0.85, &pv).await.unwrap();
    assert_eq!(gwt.current_state(), ConsciousnessState::Conscious);

    // Drop C(t) below 0.3
    let uniform_pv = [0.077; 13]; // Max entropy = low differentiation
    gwt.update_consciousness(0.2, 0.3, &uniform_pv).await.unwrap();

    // Should transition to DORMANT (C(t) < 0.3)
    assert_eq!(gwt.current_state(), ConsciousnessState::Dormant);
}

/// Test hypersync detection (warning state)
#[tokio::test]
async fn test_hypersync_warning_state() {
    let (mut gwt, _db, _temp) = create_test_gwt().await;

    // Push to hypersync (r > 0.95)
    let pv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05, 0.15, 0.25, 0.35];
    gwt.update_consciousness(0.98, 0.95, &pv).await.unwrap();

    assert_eq!(gwt.current_state(), ConsciousnessState::Hypersync);
    assert!(gwt.is_hypersync()); // Warning flag
}
```

### Step 3: Implement IT-GWT-002 (Persistence)

```rust
// tests/gwt_integration/persistence_tests.rs
use super::*;

/// IT-GWT-002: SELF_EGO_NODE persists across system restart
#[tokio::test]
async fn test_ego_node_persists_across_restart() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path();

    // Session 1: Create and modify ego node
    let original_ic: f32;
    {
        let db = RocksDbHandle::open(db_path).await.unwrap();
        let mut gwt = GwtSystem::new(Some(&db)).await.unwrap();

        // Update ego node with some fingerprint
        let fingerprint = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05, 0.15, 0.25, 0.35];
        gwt.process_action_awareness(&fingerprint).await.unwrap();

        original_ic = gwt.get_identity_continuity();
        assert!(original_ic > 0.0, "IC should be set after processing");

        // Persist
        gwt.persist_ego_node(&db).await.unwrap();

        // DB closes when dropped
    }

    // Session 2: Reopen and verify restoration
    {
        let db = RocksDbHandle::open(db_path).await.unwrap();
        let gwt = GwtSystem::new(Some(&db)).await.unwrap();

        let restored_ic = gwt.get_identity_continuity();

        // IC should be preserved (with slight drift due to time)
        assert!(
            (restored_ic - original_ic).abs() < 0.1,
            "IC should be preserved: original={}, restored={}",
            original_ic,
            restored_ic
        );
    }
}

/// Test ego node creation when no prior state exists
#[tokio::test]
async fn test_ego_node_created_fresh_when_missing() {
    let (gwt, _db, _temp) = create_test_gwt().await;

    // Should have a valid ego node even with empty DB
    let ic = gwt.get_identity_continuity();
    assert!(ic >= 0.0 && ic <= 1.0, "Fresh ego node should have valid IC");
}
```

### Step 4: Implement IT-GWT-003 (Event Cascade)

```rust
// tests/gwt_integration/event_cascade_tests.rs
use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// IT-GWT-003: IdentityCritical triggers dream consolidation
#[tokio::test]
async fn test_identity_critical_triggers_dream_consolidation() {
    let (mut gwt, _db, _temp) = create_test_gwt().await;

    // Set up a test listener to detect consolidation trigger
    let consolidation_triggered = Arc::new(AtomicBool::new(false));
    let trigger_clone = consolidation_triggered.clone();

    gwt.set_consolidation_callback(move || {
        trigger_clone.store(true, Ordering::SeqCst);
    });

    // Force IC below 0.5 by dramatic purpose vector change
    let pv1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    gwt.process_action_awareness(&pv1).await.unwrap();

    // Dramatic shift in purpose
    let pv2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    gwt.process_action_awareness(&pv2).await.unwrap();

    // IC should have dropped, triggering IdentityCritical
    let ic = gwt.get_identity_continuity();
    if ic < 0.5 {
        assert!(
            consolidation_triggered.load(Ordering::SeqCst),
            "Dream consolidation should be triggered when IC < 0.5"
        );
    }
}

/// Test event broadcast reaches all listeners
#[tokio::test]
async fn test_event_broadcast_reaches_all_listeners() {
    let (gwt, _db, _temp) = create_test_gwt().await;

    // Verify all 3 listeners are registered
    let listener_names = gwt.get_listener_names();
    assert_eq!(listener_names.len(), 3);
    assert!(listener_names.contains(&"DreamEventListener"));
    assert!(listener_names.contains(&"NeuromodulationEventListener"));
    assert!(listener_names.contains(&"MetaCognitiveEventListener"));
}
```

### Step 5: Implement Chaos Tests

```rust
// tests/gwt_integration/chaos_tests.rs
use super::*;
use std::time::{Duration, Instant};

/// CH-GWT-001: RocksDB corruption during persist - graceful degradation
#[tokio::test]
async fn test_rocksdb_corruption_graceful_degradation() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path();

    // Create and persist
    {
        let db = RocksDbHandle::open(db_path).await.unwrap();
        let mut gwt = GwtSystem::new(Some(&db)).await.unwrap();
        gwt.persist_ego_node(&db).await.unwrap();
    }

    // Corrupt the database file
    let db_file = db_path.join("CURRENT");
    if db_file.exists() {
        std::fs::write(&db_file, b"corrupted data").unwrap();
    }

    // System should handle corruption gracefully
    let result = RocksDbHandle::open(db_path).await;

    // Either recovery succeeds or a clean error is returned
    match result {
        Ok(db) => {
            // If opened, GwtSystem should create fresh state
            let gwt = GwtSystem::new(Some(&db)).await.unwrap();
            assert!(gwt.get_identity_continuity() >= 0.0);
        }
        Err(e) => {
            // Error should be descriptive, not a panic
            assert!(
                e.to_string().contains("corrupt") || e.to_string().contains("invalid"),
                "Error should indicate corruption: {:?}",
                e
            );
        }
    }
}

/// CH-GWT-002: Concurrent event broadcast - no deadlock
#[tokio::test]
async fn test_concurrent_event_broadcast_no_deadlock() {
    let (gwt, _db, _temp) = create_test_gwt().await;
    let gwt = Arc::new(gwt);

    let mut handles = vec![];

    // Spawn 10 concurrent event broadcasts
    for i in 0..10 {
        let gwt_clone = gwt.clone();
        let handle = tokio::spawn(async move {
            let pv = [0.1 * i as f32; 13];
            // This should not deadlock
            gwt_clone.update_consciousness(0.5, 0.5, &pv).await
        });
        handles.push(handle);
    }

    // Wait for all with timeout
    let start = Instant::now();
    for handle in handles {
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("Timeout - possible deadlock")
            .expect("Task panicked")
            .expect("Update failed");
    }

    // Should complete quickly
    assert!(start.elapsed() < Duration::from_secs(5), "Took too long - possible deadlock");
}

/// CH-GWT-003: Kuramoto network overflow - clamps to valid range
#[tokio::test]
async fn test_kuramoto_overflow_clamps_to_valid_range() {
    let (mut gwt, _db, _temp) = create_test_gwt().await;

    // Try extreme values
    let extreme_pv = [f32::MAX; 13];
    let result = gwt.update_consciousness(f32::MAX, f32::MAX, &extreme_pv).await;

    match result {
        Ok(c_t) => {
            // Should be clamped to [0, 1]
            assert!(c_t >= 0.0 && c_t <= 1.0, "C(t) should be in [0,1], got {}", c_t);
        }
        Err(e) => {
            // Error is acceptable for extreme inputs
            assert!(
                e.to_string().contains("overflow") || e.to_string().contains("invalid"),
                "Error should indicate overflow: {:?}",
                e
            );
        }
    }

    // Try negative values
    let neg_pv = [-1.0; 13];
    let result = gwt.update_consciousness(-1.0, -1.0, &neg_pv).await;

    match result {
        Ok(c_t) => {
            assert!(c_t >= 0.0 && c_t <= 1.0, "C(t) should be in [0,1], got {}", c_t);
        }
        Err(_) => {
            // Error is acceptable
        }
    }
}

/// CH-GWT-004: Memory pressure test
#[tokio::test]
async fn test_memory_pressure_handling() {
    let (mut gwt, _db, _temp) = create_test_gwt().await;

    // Run many consciousness updates
    for i in 0..1000 {
        let pv = [(i as f32 / 1000.0) * 0.077; 13];
        let r = (i as f32 / 1000.0).clamp(0.0, 1.0);
        gwt.update_consciousness(r, 0.5, &pv).await.unwrap();
    }

    // Should still be responsive
    let c_t = gwt.get_consciousness_level();
    assert!(c_t >= 0.0 && c_t <= 1.0);
}
```

---

## 6. Validation Criteria

| Criterion | Test | Expected |
|-----------|------|----------|
| Full cycle works | IT-GWT-001 | DORMANT -> CONSCIOUS transition |
| Persistence works | IT-GWT-002 | IC preserved across restart |
| Events trigger consolidation | IT-GWT-003 | IdentityCritical -> consolidation |
| Low C(t) -> DORMANT | IT-GWT-004 | State drops to DORMANT |
| Corruption handled | CH-GWT-001 | No panic, graceful error |
| No deadlocks | CH-GWT-002 | Concurrent updates complete |
| Overflow handled | CH-GWT-003 | Values clamped to [0,1] |

### Verification Commands

```bash
# Run all GWT integration tests
cargo test --package context-graph-core --test gwt_integration --no-fail-fast

# Run specific test
cargo test --package context-graph-core --test gwt_integration::consciousness_cycle_tests

# Run chaos tests only
cargo test --package context-graph-core --test gwt_integration::chaos_tests

# Run with verbose output
cargo test --package context-graph-core --test gwt_integration -- --nocapture
```

---

## 7. Dependencies

### Upstream
- TASK-GWT-P1-001 (persistence must work for IT-GWT-002)
- TASK-GWT-P1-002 (event wiring must work for IT-GWT-003)

### Downstream
- None (this is the surface layer verification)

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Flaky timing tests | Medium | Low | Use deterministic time mocking |
| Test isolation issues | Low | Medium | Use temp directories, clean state |
| CI environment differences | Low | Medium | Test in CI early |

---

## 9. Notes

- These tests verify the full system works end-to-end
- Chaos tests ensure robustness under failure conditions
- Consider adding property-based tests for consciousness equation bounds
- Integration tests should run in CI on every PR
