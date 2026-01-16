//! Integration tests for autonomous storage with real RocksDB.
//!
//! # CRITICAL: NO MOCK DATA
//!
//! All tests use REAL RocksDB instances and REAL data structures.
//! Tests verify:
//! 1. 5 autonomous column families can be opened together
//! 2. All singleton config/state can be stored and retrieved
//! 3. Time-series data (lineage, consolidation) persists correctly
//! 4. Per-memory data operates correctly
//! 5. Data survives close/reopen cycles (persistence verification)
//!
//! TASK-P0-004: Removed drift_history and goal_activity_metrics tests after North Star removal.
//! - drift_history removed - old drift detection replaced by topic_stability.churn_rate (ARCH-10)
//! - goal_activity_metrics removed - manual goals forbidden by ARCH-03

use chrono::Utc;
use context_graph_core::autonomous::{
    AdaptiveThresholdState, AutonomousConfig, GoalId, MemoryCurationState,
};
// TASK-P0-004: Removed DriftDataPoint, GoalActivityMetrics imports (used by removed CFs)
use context_graph_storage::autonomous::{ConsolidationRecord, LineageEvent, RocksDbAutonomousStore};
use tempfile::TempDir;
use uuid::Uuid;

// =========================================================================
// PHASE 1: Basic CRUD Operations
// =========================================================================

#[test]
fn test_autonomous_config_crud_real_db() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Initially empty
    let initial = store.get_autonomous_config().expect("get failed");
    assert!(initial.is_none(), "Config should not exist initially");

    // Write config
    let auto_config = AutonomousConfig::default();
    store
        .store_autonomous_config(&auto_config)
        .expect("store failed");

    // Verify written
    let retrieved = store
        .get_autonomous_config()
        .expect("get failed")
        .expect("Config should exist after store");

    // Check that enabled field persisted correctly
    assert_eq!(
        retrieved.enabled, auto_config.enabled,
        "Enabled flag mismatch"
    );

    println!("[PASS] AutonomousConfig CRUD with real RocksDB");
}

#[test]
fn test_threshold_state_crud_real_db() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Write state
    let threshold_state = AdaptiveThresholdState::default();
    store
        .store_threshold_state(&threshold_state)
        .expect("store failed");

    // Verify written
    let retrieved = store
        .get_threshold_state()
        .expect("get failed")
        .expect("State should exist after store");

    // Check that optimal threshold persisted correctly
    assert_eq!(
        retrieved.optimal, threshold_state.optimal,
        "Optimal threshold mismatch"
    );

    println!("[PASS] AdaptiveThresholdState CRUD with real RocksDB");
}

// TASK-P0-004: Removed test_drift_history_crud_real_db - old drift detection replaced by topic_stability.churn_rate (ARCH-10)
// TASK-P0-004: Removed test_goal_activity_metrics_crud_real_db - manual goals forbidden by ARCH-03

#[test]
fn test_memory_curation_state_crud_real_db() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Create memory ID
    let memory_uuid = Uuid::new_v4();

    // Write state (Active variant)
    let state = MemoryCurationState::Active;
    store
        .store_curation_state(memory_uuid, &state)
        .expect("store failed");

    // Verify written
    let retrieved = store
        .get_curation_state(memory_uuid)
        .expect("get failed")
        .expect("State should exist after store");

    assert_eq!(retrieved, MemoryCurationState::Active, "State mismatch");

    // Update to Dormant variant
    let dormant_state = MemoryCurationState::Dormant { since: Utc::now() };
    store
        .store_curation_state(memory_uuid, &dormant_state)
        .expect("store update failed");

    // Verify update
    let retrieved = store
        .get_curation_state(memory_uuid)
        .expect("get failed")
        .expect("State should exist after update");

    match retrieved {
        MemoryCurationState::Dormant { since: _ } => {}
        _ => panic!("Expected Dormant state, got {:?}", retrieved),
    }

    println!("[PASS] MemoryCurationState CRUD with real RocksDB");
}

// =========================================================================
// PHASE 2: Persistence Verification (Source of Truth)
// =========================================================================

// TASK-P0-004: Updated test to remove drift_history and goal_activity_metrics after North Star removal.
// Only tests remaining 5 CFs: autonomous_config, adaptive_threshold_state, autonomous_lineage,
// consolidation_history, and memory_curation.
#[test]
fn test_persistence_across_close_reopen() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let memory_uuid = Uuid::new_v4();

    // Phase 1: Write data and close
    {
        let store = RocksDbAutonomousStore::open(&db_path).expect("Failed to open store");

        // Write AutonomousConfig
        store
            .store_autonomous_config(&AutonomousConfig::default())
            .expect("store config failed");

        // Write threshold state
        store
            .store_threshold_state(&AdaptiveThresholdState::default())
            .expect("store threshold failed");

        // Write lineage event (replaces drift_history for time-series testing)
        let event = LineageEvent::new("bootstrap", "Test persistence event");
        store.store_lineage_event(&event).expect("store lineage failed");

        // Write consolidation record
        let record = ConsolidationRecord::success(
            vec![],
            context_graph_core::autonomous::MemoryId::new(),
            0.95,
            0.03,
        );
        store
            .store_consolidation_record(&record)
            .expect("store consolidation failed");

        // Write memory curation
        let curation = MemoryCurationState::Active;
        store
            .store_curation_state(memory_uuid, &curation)
            .expect("store curation failed");

        // Flush to ensure persistence
        store.flush().expect("flush failed");

        // Store drops here, closing RocksDB
    }

    // Phase 2: Reopen and verify all data
    {
        let store = RocksDbAutonomousStore::open(&db_path).expect("Failed to reopen store");

        // Verify AutonomousConfig
        let config = store
            .get_autonomous_config()
            .expect("get config failed")
            .expect("Config must persist");
        assert!(config.enabled, "Config enabled flag invalid");

        // Verify threshold state
        let threshold = store
            .get_threshold_state()
            .expect("get threshold failed")
            .expect("Threshold state must persist");
        assert!(threshold.optimal > 0.0, "Threshold optimal invalid");

        // Verify lineage event
        let lineage = store.get_lineage_history(None).expect("get lineage failed");
        assert!(!lineage.is_empty(), "Lineage history must persist");
        assert_eq!(lineage[0].event_type, "bootstrap", "Lineage event type mismatch");

        // Verify consolidation record
        let consolidation = store.get_consolidation_history(None).expect("get consolidation failed");
        assert!(!consolidation.is_empty(), "Consolidation history must persist");
        assert!(consolidation[0].success, "Consolidation record success flag invalid");

        // Verify memory curation
        let curation = store
            .get_curation_state(memory_uuid)
            .expect("get curation failed")
            .expect("Curation state must persist");
        assert_eq!(curation, MemoryCurationState::Active, "State mismatch");
    }

    println!("[PASS] All data persists across close/reopen cycle");
}

// =========================================================================
// PHASE 3: Lineage and Consolidation Events
// =========================================================================

#[test]
fn test_lineage_events_real_db() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Write lineage events
    for i in 0..5 {
        let event = LineageEvent::new(
            format!("test_event_{}", i),
            format!("Event {} description", i),
        );
        store.store_lineage_event(&event).expect("store failed");
    }

    // Retrieve all lineage history
    let events = store.get_lineage_history(None).expect("get failed");
    assert_eq!(events.len(), 5, "Should have 5 lineage events");

    println!("[PASS] LineageEvent storage with real RocksDB");
}

#[test]
fn test_consolidation_records_real_db() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Write consolidation records
    for i in 0..3 {
        let record = ConsolidationRecord::success(
            vec![],                                          // source memories
            context_graph_core::autonomous::MemoryId::new(), // target memory
            0.92 + (i as f32 * 0.01),                        // similarity
            0.03 + (i as f32 * 0.01),                        // theta_diff
        );
        store
            .store_consolidation_record(&record)
            .expect("store failed");
    }

    // Retrieve all consolidation history
    let records = store.get_consolidation_history(None).expect("get failed");
    assert_eq!(records.len(), 3, "Should have 3 consolidation records");

    // Verify data integrity
    for record in &records {
        assert!(record.similarity_score > 0.9, "Invalid similarity");
        assert!(record.success, "Record should be success");
    }

    println!("[PASS] ConsolidationRecord storage with real RocksDB");
}

// =========================================================================
// PHASE 4: Health Check and Database Integrity
// =========================================================================

#[test]
fn test_health_check_reports_correct_status() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Write some data
    store
        .store_autonomous_config(&AutonomousConfig::default())
        .expect("store failed");

    // Health check should pass
    let health = store.health_check();
    assert!(health.is_ok(), "Health check should pass");

    println!("[PASS] Health check reports correct status");
}

// TASK-P0-004: Updated test to use lineage events instead of drift_history after North Star removal.
#[test]
fn test_flush_and_compact_operations() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Write data (using lineage events instead of drift_history)
    for i in 0..10 {
        let event = LineageEvent::new(
            format!("test_event_{}", i),
            format!("Test event {} for flush/compact", i),
        );
        store.store_lineage_event(&event).expect("store failed");
    }

    // Flush should succeed
    store.flush().expect("flush failed");

    // Compact should succeed (even if no-op)
    store.compact().expect("compact failed");

    // Verify data still accessible
    let history = store.get_lineage_history(None).expect("get failed");
    assert_eq!(history.len(), 10, "Data should survive flush and compact");

    println!("[PASS] Flush and compact operations work correctly");
}

// =========================================================================
// PHASE 5: Update Operations
// =========================================================================

// TASK-P0-004: Updated test to use memory curation instead of goal_activity_metrics after North Star removal.
// Goal activity metrics removed - manual goals forbidden by ARCH-03.
#[test]
fn test_update_overwrites_previous_value() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    let memory_uuid = Uuid::new_v4();

    // Initial write - Active state
    let initial_state = MemoryCurationState::Active;
    store
        .store_curation_state(memory_uuid, &initial_state)
        .expect("store failed");

    // Verify initial
    let retrieved = store
        .get_curation_state(memory_uuid)
        .expect("get failed")
        .expect("State should exist");
    assert_eq!(retrieved, MemoryCurationState::Active, "Initial state wrong");

    // Update to Dormant
    let updated_state = MemoryCurationState::Dormant { since: Utc::now() };
    store
        .store_curation_state(memory_uuid, &updated_state)
        .expect("store failed");

    // Verify update
    let retrieved = store
        .get_curation_state(memory_uuid)
        .expect("get failed")
        .expect("State should exist after update");
    match retrieved {
        MemoryCurationState::Dormant { .. } => {}
        _ => panic!("Expected Dormant state, got {:?}", retrieved),
    }

    println!("[PASS] Update overwrites previous value correctly");
}

// TASK-P0-004: Removed test_list_all_goal_metrics - manual goals forbidden by ARCH-03

#[test]
fn test_list_all_curation_states() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Write multiple curation states
    for i in 0..3 {
        let memory_uuid = Uuid::new_v4();
        let state = if i == 0 {
            MemoryCurationState::Active
        } else if i == 1 {
            MemoryCurationState::Dormant { since: Utc::now() }
        } else {
            MemoryCurationState::Archived { since: Utc::now() }
        };
        store
            .store_curation_state(memory_uuid, &state)
            .expect("store failed");
    }

    // List all states
    let all_states = store.list_all_curation_states().expect("list failed");
    assert_eq!(all_states.len(), 3, "Should have 3 curation states");

    println!("[PASS] list_all_curation_states works correctly");
}

// =========================================================================
// PHASE 6: Lineage Event Builder Tests
// =========================================================================

#[test]
fn test_lineage_event_builder_pattern() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbAutonomousStore::open(temp_dir.path()).expect("Failed to open store");

    // Test builder pattern
    let goal_id = GoalId::new();
    let event = LineageEvent::new("bootstrap", "System initialized")
        .with_goal(goal_id.clone())
        .with_metadata(r#"{"version": "1.0"}"#);

    store.store_lineage_event(&event).expect("store failed");

    // Retrieve and verify
    let events = store.get_lineage_history(None).expect("get failed");
    assert_eq!(events.len(), 1, "Should have 1 event");

    let retrieved = &events[0];
    assert_eq!(retrieved.event_type, "bootstrap");
    assert!(retrieved.goal_id.is_some(), "Goal ID should be set");
    assert!(retrieved.metadata.is_some(), "Metadata should be set");

    println!("[PASS] LineageEvent builder pattern works correctly");
}
