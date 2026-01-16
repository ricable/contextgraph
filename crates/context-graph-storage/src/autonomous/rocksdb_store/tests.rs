//! Tests for RocksDbAutonomousStore.
//!
//! TASK-P0-004: Removed drift_history and goal_activity_metrics tests after North Star removal.

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use tempfile::TempDir;

    use context_graph_core::autonomous::{
        AdaptiveThresholdState, AutonomousConfig, GoalId,
        MemoryCurationState, MemoryId,
    };
    // TASK-P0-004: Removed DriftDataPoint, GoalActivityMetrics imports (used by removed CF)

    use super::super::{ConsolidationRecord, LineageEvent, RocksDbAutonomousStore};

    fn create_test_store() -> (TempDir, RocksDbAutonomousStore) {
        let tmp = TempDir::new().unwrap();
        let store = RocksDbAutonomousStore::open(tmp.path()).unwrap();
        (tmp, store)
    }

    #[test]
    fn test_open_and_health_check() {
        let (_tmp, store) = create_test_store();
        assert!(store.health_check().is_ok());
    }

    #[test]
    fn test_autonomous_config_crud() {
        let (_tmp, store) = create_test_store();

        // Initially empty
        let config = store.get_autonomous_config().unwrap();
        assert!(config.is_none());

        // Store config
        let config = AutonomousConfig::default();
        store.store_autonomous_config(&config).unwrap();

        // Retrieve config
        let retrieved = store.get_autonomous_config().unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.enabled, config.enabled);
    }

    #[test]
    fn test_threshold_state_crud() {
        let (_tmp, store) = create_test_store();

        // Initially empty
        let state = store.get_threshold_state().unwrap();
        assert!(state.is_none());

        // Store state
        let state = AdaptiveThresholdState::default();
        store.store_threshold_state(&state).unwrap();

        // Retrieve state
        let retrieved = store.get_threshold_state().unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert!((retrieved.optimal - state.optimal).abs() < f32::EPSILON);
    }

    // TASK-P0-004: Removed test_drift_history - old drift detection replaced by topic_stability.churn_rate (ARCH-10)
    // TASK-P0-004: Removed test_goal_metrics_crud - manual goals forbidden by ARCH-03

    #[test]
    fn test_lineage_events() {
        let (_tmp, store) = create_test_store();

        // Store some events
        for i in 0..3 {
            let event = LineageEvent::new(format!("test_event_{}", i), format!("Test event {}", i));
            store.store_lineage_event(&event).unwrap();
        }

        // Retrieve all
        let history = store.get_lineage_history(None).unwrap();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_consolidation_records() {
        let (_tmp, store) = create_test_store();

        // Store a success record
        let record = ConsolidationRecord::success(
            vec![MemoryId::new(), MemoryId::new()],
            MemoryId::new(),
            0.95,
            0.03,
        );
        store.store_consolidation_record(&record).unwrap();

        // Store a failure record
        let record = ConsolidationRecord::failure(
            vec![MemoryId::new()],
            MemoryId::new(),
            0.85,
            0.08,
            "Test error",
        );
        store.store_consolidation_record(&record).unwrap();

        // Retrieve all
        let history = store.get_consolidation_history(None).unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_curation_state_crud() {
        let (_tmp, store) = create_test_store();

        let memory_id = MemoryId::new();
        let state = MemoryCurationState::Active;

        // Store state
        store.store_curation_state(memory_id.0, &state).unwrap();

        // Retrieve state
        let retrieved = store.get_curation_state(memory_id.0).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), MemoryCurationState::Active);

        // Update to different state
        let dormant_state = MemoryCurationState::Dormant { since: Utc::now() };
        store
            .store_curation_state(memory_id.0, &dormant_state)
            .unwrap();

        let retrieved = store.get_curation_state(memory_id.0).unwrap();
        assert!(retrieved.is_some());
        match retrieved.unwrap() {
            MemoryCurationState::Dormant { .. } => {}
            _ => panic!("Expected Dormant state"),
        }

        // List all
        let all_states = store.list_all_curation_states().unwrap();
        assert_eq!(all_states.len(), 1);
    }

    #[test]
    fn test_persistence_across_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        // Store data
        {
            let store = RocksDbAutonomousStore::open(&path).unwrap();
            let config = AutonomousConfig::default();
            store.store_autonomous_config(&config).unwrap();
            store.flush().unwrap();
        }

        // Reopen and verify
        {
            let store = RocksDbAutonomousStore::open(&path).unwrap();
            let config = store.get_autonomous_config().unwrap();
            assert!(config.is_some());
        }
    }

    #[test]
    fn test_flush_and_compact() {
        let (_tmp, store) = create_test_store();

        // Store some data
        store
            .store_autonomous_config(&AutonomousConfig::default())
            .unwrap();

        // Flush and compact should succeed
        assert!(store.flush().is_ok());
        assert!(store.compact().is_ok());
    }

    #[test]
    fn test_lineage_event_builder() {
        let goal_id = GoalId::new();
        let memory_id = MemoryId::new();

        let event = LineageEvent::new("bootstrap", "Initial bootstrap")
            .with_goal(goal_id.clone())
            .with_memory(memory_id.clone())
            .with_metadata(r#"{"source": "test"}"#);

        assert_eq!(event.event_type, "bootstrap");
        assert!(event.goal_id.is_some());
        assert!(event.memory_id.is_some());
        assert!(event.metadata.is_some());
    }

    #[test]
    fn test_consolidation_record_constructors() {
        let sources = vec![MemoryId::new(), MemoryId::new()];
        let target = MemoryId::new();

        let success = ConsolidationRecord::success(sources.clone(), target.clone(), 0.95, 0.02);
        assert!(success.success);
        assert!(success.error_message.is_none());

        let failure =
            ConsolidationRecord::failure(sources, target, 0.90, 0.05, "Test error message");
        assert!(!failure.success);
        assert!(failure.error_message.is_some());
    }
}
