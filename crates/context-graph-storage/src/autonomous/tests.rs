//! Comprehensive tests for autonomous storage module.
//!
//! # Test Coverage
//! - Column family definitions and options
//! - Key format functions
//! - Key parsing (including FAIL FAST panics)
//! - Edge cases and boundary conditions
//! - Integration with real data types
//!
//! TASK-P0-004: Removed drift_history and goal_activity_metrics tests after North Star removal.

use super::*;
use rocksdb::Cache;
use uuid::Uuid;

// =============================================================================
// INTEGRATION TESTS WITH REAL DATA TYPES
// =============================================================================

mod integration_tests {
    use super::*;
    use chrono::Utc;
    use context_graph_core::autonomous::{
        curation::{MemoryCurationState, MemoryId},
        thresholds::AdaptiveThresholdState,
        workflow::AutonomousConfig,
    };
    // TASK-P0-004: Removed GoalId, DriftConfig, DriftDataPoint, DriftState, GoalActivityMetrics imports

    #[test]
    fn test_autonomous_config_key_usage() {
        // Verify the singleton key is appropriate for AutonomousConfig storage
        let key = AUTONOMOUS_CONFIG_KEY;
        assert_eq!(key, b"config");

        // Create a real AutonomousConfig to verify the type is correct
        let config = AutonomousConfig::default();
        assert!(config.enabled);
        assert!(config.bootstrap.auto_init);
    }

    #[test]
    fn test_adaptive_threshold_state_key_usage() {
        // Verify the singleton key is appropriate for AdaptiveThresholdState storage
        let key = ADAPTIVE_THRESHOLD_STATE_KEY;
        assert_eq!(key, b"state");

        // Create a real AdaptiveThresholdState to verify the type is correct
        let state = AdaptiveThresholdState::default();
        assert_eq!(state.per_embedder.len(), 13);
        assert!((state.optimal - 0.75).abs() < f32::EPSILON);
    }

    // TASK-P0-004: Removed test_drift_history_with_real_data_point - old drift detection (ARCH-10)
    // TASK-P0-004: Removed test_drift_history_time_range_query_pattern - old drift detection (ARCH-10)
    // TASK-P0-004: Removed test_goal_activity_metrics_with_real_goal_id - manual goals forbidden (ARCH-03)

    #[test]
    fn test_memory_curation_with_real_memory_id() {
        // Create a real MemoryId
        let memory_id = MemoryId::new();
        let key = memory_curation_key(&memory_id.0);

        assert_eq!(key.len(), 16);

        // Parse back
        let parsed_uuid = parse_memory_curation_key(&key);
        assert_eq!(parsed_uuid, memory_id.0);

        // Create a real MemoryCurationState (note: this is what would be stored as value)
        let state = MemoryCurationState::default();
        assert_eq!(state, MemoryCurationState::Active);

        let dormant_state = MemoryCurationState::Dormant { since: Utc::now() };
        assert_ne!(dormant_state, MemoryCurationState::Active);
    }

    #[test]
    fn test_autonomous_lineage_with_real_timestamps() {
        let now = Utc::now();
        let event_id = Uuid::new_v4();
        let timestamp_ms = now.timestamp_millis();

        let key = autonomous_lineage_key(timestamp_ms, &event_id);
        assert_eq!(key.len(), 24);

        let (parsed_ts, parsed_uuid) = parse_autonomous_lineage_key(&key);
        assert_eq!(parsed_ts, timestamp_ms);
        assert_eq!(parsed_uuid, event_id);
    }

    #[test]
    fn test_consolidation_history_with_real_timestamps() {
        let now = Utc::now();
        let record_id = Uuid::new_v4();
        let timestamp_ms = now.timestamp_millis();

        let key = consolidation_history_key(timestamp_ms, &record_id);
        assert_eq!(key.len(), 24);

        let (parsed_ts, parsed_uuid) = parse_consolidation_history_key(&key);
        assert_eq!(parsed_ts, timestamp_ms);
        assert_eq!(parsed_uuid, record_id);
    }

    // TASK-P0-004: Removed test_drift_state_integration - old drift detection replaced by topic_stability.churn_rate (ARCH-10)
}

// =============================================================================
// COLUMN FAMILY DESCRIPTOR TESTS
// =============================================================================

mod cf_descriptor_tests {
    use super::*;

    #[test]
    fn test_all_descriptors_created_with_shared_cache() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);

        assert_eq!(descriptors.len(), AUTONOMOUS_CF_COUNT);

        // Verify each descriptor has a valid name
        for desc in &descriptors {
            assert!(!desc.name().is_empty());
        }
    }

    #[test]
    fn test_descriptor_names_match_constants() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);

        // TASK-P0-004: Updated to 5 CFs after removing drift_history and goal_activity_metrics
        let expected_names: [&str; 5] = [
            CF_AUTONOMOUS_CONFIG,
            CF_ADAPTIVE_THRESHOLD_STATE,
            CF_AUTONOMOUS_LINEAGE,
            CF_CONSOLIDATION_HISTORY,
            CF_MEMORY_CURATION,
        ];

        for (i, expected) in expected_names.iter().enumerate() {
            assert_eq!(
                descriptors[i].name(),
                *expected,
                "Descriptor at index {} has wrong name",
                i
            );
        }
    }

    #[test]
    fn test_cf_options_with_various_cache_sizes() {
        let sizes = [
            0,                  // Zero
            1024,               // 1KB
            1024 * 1024,        // 1MB
            256 * 1024 * 1024,  // 256MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let cache = Cache::new_lru_cache(size);
            let descriptors = get_autonomous_cf_descriptors(&cache);
            assert_eq!(
                descriptors.len(),
                AUTONOMOUS_CF_COUNT,
                "Failed with cache size {}",
                size
            );
        }
    }
}

// =============================================================================
// KEY FORMAT EDGE CASES
// =============================================================================
// TASK-P0-004: Updated tests to use autonomous_lineage and memory_curation
// instead of removed drift_history and goal_activity_metrics functions.

mod key_format_edge_cases {
    use super::*;

    #[test]
    fn test_timestamp_boundary_values() {
        let uuid = Uuid::new_v4();

        // Test all boundary timestamps using autonomous_lineage (same key format as removed drift_history)
        let timestamps = vec![i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX];

        for ts in timestamps {
            let key = autonomous_lineage_key(ts, &uuid);
            let (parsed_ts, parsed_uuid) = parse_autonomous_lineage_key(&key);
            assert_eq!(parsed_ts, ts, "Roundtrip failed for timestamp {}", ts);
            assert_eq!(parsed_uuid, uuid);
        }
    }

    #[test]
    fn test_nil_uuid_handling() {
        let nil = Uuid::nil();

        // All remaining key types should handle nil UUID
        let key1 = memory_curation_key(&nil);
        assert_eq!(parse_memory_curation_key(&key1), nil);

        let key2 = autonomous_lineage_key(0, &nil);
        let (_, parsed) = parse_autonomous_lineage_key(&key2);
        assert_eq!(parsed, nil);
    }

    #[test]
    fn test_max_uuid_handling() {
        let max = Uuid::max();

        let key = memory_curation_key(&max);
        assert_eq!(parse_memory_curation_key(&key), max);
    }

    #[test]
    fn test_specific_uuid_roundtrip() {
        // Use a well-known UUID format
        let uuid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

        let key = memory_curation_key(&uuid);
        let parsed = parse_memory_curation_key(&key);
        assert_eq!(parsed, uuid);
    }

    #[test]
    fn test_key_ordering_for_range_scans() {
        let uuid = Uuid::new_v4();

        // Create keys with sequential timestamps using autonomous_lineage
        let keys: Vec<_> = (0..10)
            .map(|i| autonomous_lineage_key(i * 1000, &uuid))
            .collect();

        // Verify lexicographic ordering matches timestamp ordering
        for i in 0..keys.len() - 1 {
            assert!(keys[i] < keys[i + 1], "Key ordering broken at index {}", i);
        }
    }

    #[test]
    fn test_prefix_extraction_consistency() {
        let timestamp_ms: i64 = 1704067200000;

        // All timestamp-prefixed keys should have consistent prefix
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();

        let key1 = autonomous_lineage_key(timestamp_ms, &uuid1);
        let key2 = autonomous_lineage_key(timestamp_ms, &uuid2);
        let key3 = consolidation_history_key(timestamp_ms, &uuid1);

        // First 8 bytes (timestamp prefix) should be identical
        assert_eq!(&key1[..8], &key2[..8]);
        assert_eq!(&key2[..8], &key3[..8]);
        assert_eq!(&key1[..8], timestamp_ms.to_be_bytes());
    }
}

// =============================================================================
// FAIL FAST PANIC TESTS
// =============================================================================
// TASK-P0-004: Removed drift_history and goal_activity_metrics panic tests

mod fail_fast_tests {
    use super::*;

    // TASK-P0-004: Removed test_drift_history_key_parse_fails_on_short_input
    // TASK-P0-004: Removed test_drift_history_key_parse_fails_on_long_input
    // TASK-P0-004: Removed test_goal_activity_metrics_key_parse_fails_on_short_input
    // TASK-P0-004: Removed test_goal_activity_metrics_key_parse_fails_on_long_input

    #[test]
    #[should_panic(expected = "STORAGE ERROR")]
    fn test_memory_curation_key_parse_fails_on_empty_input() {
        parse_memory_curation_key(&[]);
    }

    #[test]
    #[should_panic(expected = "STORAGE ERROR")]
    fn test_autonomous_lineage_key_parse_fails_on_wrong_size() {
        parse_autonomous_lineage_key(&[0u8; 10]);
    }

    #[test]
    #[should_panic(expected = "STORAGE ERROR")]
    fn test_consolidation_history_key_parse_fails_on_wrong_size() {
        parse_consolidation_history_key(&[0u8; 30]);
    }
}

// =============================================================================
// DETERMINISM TESTS
// =============================================================================
// TASK-P0-004: Updated to use autonomous_lineage instead of removed drift_history

mod determinism_tests {
    use super::*;

    #[test]
    fn test_key_generation_is_deterministic() {
        let uuid = Uuid::parse_str("12345678-1234-5678-1234-567812345678").unwrap();
        let timestamp_ms: i64 = 1704067200000;

        // Generate same key multiple times using autonomous_lineage (same format as removed drift_history)
        let keys: Vec<_> = (0..100)
            .map(|_| autonomous_lineage_key(timestamp_ms, &uuid))
            .collect();

        // All keys must be identical
        for key in &keys {
            assert_eq!(key, &keys[0]);
        }
    }

    #[test]
    fn test_singleton_keys_are_constant() {
        // Verify singleton keys don't change
        assert_eq!(AUTONOMOUS_CONFIG_KEY, b"config");
        assert_eq!(ADAPTIVE_THRESHOLD_STATE_KEY, b"state");

        // Verify they're static references
        let k1: &'static [u8] = AUTONOMOUS_CONFIG_KEY;
        let k2: &'static [u8] = ADAPTIVE_THRESHOLD_STATE_KEY;
        assert!(!k1.is_empty());
        assert!(!k2.is_empty());
    }
}

// =============================================================================
// SERIALIZATION PATTERN TESTS (for values, not keys)
// =============================================================================
// TASK-P0-004: Removed DriftDataPoint and GoalActivityMetrics serialization tests

mod serialization_pattern_tests {
    use chrono::Utc;
    use context_graph_core::autonomous::{
        curation::MemoryCurationState, thresholds::AdaptiveThresholdState,
        workflow::AutonomousConfig,
    };
    // TASK-P0-004: Removed GoalId, DriftDataPoint, GoalActivityMetrics imports

    #[test]
    fn test_autonomous_config_serialization_roundtrip() {
        let config = AutonomousConfig::default();
        let serialized = serde_json::to_vec(&config).expect("serialize");
        let deserialized: AutonomousConfig =
            serde_json::from_slice(&serialized).expect("deserialize");

        assert_eq!(deserialized.enabled, config.enabled);
        assert_eq!(deserialized.bootstrap.auto_init, config.bootstrap.auto_init);
    }

    #[test]
    fn test_adaptive_threshold_state_serialization_roundtrip() {
        let state = AdaptiveThresholdState::default();
        let serialized = serde_json::to_vec(&state).expect("serialize");
        let deserialized: AdaptiveThresholdState =
            serde_json::from_slice(&serialized).expect("deserialize");

        assert!((deserialized.optimal - state.optimal).abs() < f32::EPSILON);
        assert_eq!(deserialized.per_embedder.len(), 13);
    }

    // TASK-P0-004: Removed test_drift_data_point_serialization_roundtrip - old drift detection (ARCH-10)
    // TASK-P0-004: Removed test_goal_activity_metrics_serialization_roundtrip - manual goals forbidden (ARCH-03)

    #[test]
    fn test_memory_curation_state_serialization_roundtrip() {
        let states = vec![
            MemoryCurationState::Active,
            MemoryCurationState::Dormant { since: Utc::now() },
            MemoryCurationState::Archived { since: Utc::now() },
            MemoryCurationState::PendingDeletion {
                scheduled: Utc::now(),
            },
        ];

        for state in states {
            let serialized = serde_json::to_vec(&state).expect("serialize");
            let deserialized: MemoryCurationState =
                serde_json::from_slice(&serialized).expect("deserialize");
            assert_eq!(deserialized, state);
        }
    }
}

// =============================================================================
// CF NAMES VALIDATION
// =============================================================================

mod cf_names_validation {
    use super::*;

    #[test]
    fn test_no_cf_name_collisions_with_base_cfs() {
        use crate::column_families::cf_names;

        // Ensure no autonomous CF names collide with base CF names
        for auto_cf in AUTONOMOUS_CFS {
            assert!(
                !cf_names::ALL.contains(auto_cf),
                "Autonomous CF '{}' collides with base CF",
                auto_cf
            );
        }
    }

    #[test]
    fn test_no_cf_name_collisions_with_teleological_cfs() {
        use crate::teleological::TELEOLOGICAL_CFS;

        // Ensure no autonomous CF names collide with teleological CF names
        for auto_cf in AUTONOMOUS_CFS {
            assert!(
                !TELEOLOGICAL_CFS.contains(auto_cf),
                "Autonomous CF '{}' collides with teleological CF",
                auto_cf
            );
        }
    }

    #[test]
    fn test_cf_names_are_valid_identifiers() {
        for name in AUTONOMOUS_CFS {
            // Must start with a letter
            assert!(
                name.chars().next().unwrap().is_ascii_lowercase(),
                "CF name '{}' must start with lowercase letter",
                name
            );

            // Must contain only lowercase letters and underscores
            assert!(
                name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "CF name '{}' contains invalid characters",
                name
            );

            // Must not be empty
            assert!(!name.is_empty(), "CF name must not be empty");

            // Must not start or end with underscore
            assert!(
                !name.starts_with('_') && !name.ends_with('_'),
                "CF name '{}' should not start or end with underscore",
                name
            );

            // Must not have consecutive underscores
            assert!(
                !name.contains("__"),
                "CF name '{}' should not have consecutive underscores",
                name
            );
        }
    }
}
