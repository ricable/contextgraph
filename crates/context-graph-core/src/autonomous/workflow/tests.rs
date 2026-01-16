//! Tests for autonomous workflow types.

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::autonomous::bootstrap::{BootstrapConfig, GoalId};
    use crate::autonomous::curation::{ConsolidationConfig, MemoryId, PruningConfig};
    use crate::autonomous::drift::{DriftConfig, DriftState};
    use crate::autonomous::evolution::GoalEvolutionConfig;
    use crate::autonomous::thresholds::{AdaptiveThresholdConfig, AdaptiveThresholdState};

    use super::super::{
        AutonomousConfig, AutonomousHealth, AutonomousStatus, DailySchedule, OptimizationEvent,
        ScheduleValidationError, ScheduledCheckType,
    };

    // AutonomousConfig tests
    #[test]
    fn test_autonomous_config_default() {
        let config = AutonomousConfig::default();
        assert!(config.enabled);
        assert!(config.bootstrap.auto_init);
        assert!(config.thresholds.enabled);
        assert!(config.pruning.enabled);
        assert!(config.consolidation.enabled);
        assert!(config.drift.auto_correct);
        assert!(config.goals.auto_discover);
    }

    #[test]
    fn test_autonomous_config_disabled() {
        let config = AutonomousConfig::disabled();
        assert!(!config.enabled);
        assert!(!config.bootstrap.auto_init);
        assert!(!config.thresholds.enabled);
        assert!(!config.pruning.enabled);
        assert!(!config.consolidation.enabled);
        assert!(!config.drift.auto_correct);
        assert!(!config.goals.auto_discover);
    }

    #[test]
    fn test_autonomous_config_has_any_enabled() {
        let config = AutonomousConfig::default();
        assert!(config.has_any_enabled());

        let disabled = AutonomousConfig::disabled();
        assert!(!disabled.has_any_enabled());

        // Test with just one feature enabled
        let partial = AutonomousConfig {
            enabled: false,
            bootstrap: BootstrapConfig {
                auto_init: true,
                ..Default::default()
            },
            thresholds: AdaptiveThresholdConfig {
                enabled: false,
                ..Default::default()
            },
            pruning: PruningConfig {
                enabled: false,
                ..Default::default()
            },
            consolidation: ConsolidationConfig {
                enabled: false,
                ..Default::default()
            },
            drift: DriftConfig {
                auto_correct: false,
                ..Default::default()
            },
            goals: GoalEvolutionConfig {
                auto_discover: false,
                ..Default::default()
            },
        };
        assert!(partial.has_any_enabled());
    }

    #[test]
    fn test_autonomous_config_serialization() {
        let config = AutonomousConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: AutonomousConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.enabled, config.enabled);
        assert_eq!(deserialized.bootstrap.auto_init, config.bootstrap.auto_init);
        assert_eq!(deserialized.thresholds.enabled, config.thresholds.enabled);
        assert_eq!(deserialized.pruning.enabled, config.pruning.enabled);
        assert_eq!(
            deserialized.consolidation.enabled,
            config.consolidation.enabled
        );
        assert_eq!(deserialized.drift.auto_correct, config.drift.auto_correct);
        assert_eq!(deserialized.goals.auto_discover, config.goals.auto_discover);
    }

    #[test]
    fn test_autonomous_config_nested_values() {
        let config = AutonomousConfig::default();

        // Verify nested default values are correct
        assert_eq!(config.pruning.min_age_days, 30);
        assert!((config.consolidation.similarity_threshold - 0.92).abs() < f32::EPSILON);
        assert!((config.drift.alert_threshold - 0.05).abs() < f32::EPSILON);
        assert_eq!(config.goals.min_cluster_size, 10);
        assert!((config.thresholds.learning_rate - 0.01).abs() < f32::EPSILON);
    }

    // DailySchedule tests
    #[test]
    fn test_daily_schedule_default() {
        let schedule = DailySchedule::default();
        assert_eq!(schedule.consolidation_window, (0, 2));
        assert_eq!(schedule.drift_check_hour, 6);
        assert_eq!(schedule.stats_hour, 12);
        assert_eq!(schedule.prep_hour, 18);
    }

    #[test]
    fn test_daily_schedule_is_consolidation_hour() {
        let schedule = DailySchedule::default(); // Window: 0 to 2

        assert!(schedule.is_consolidation_hour(0));
        assert!(schedule.is_consolidation_hour(1));
        assert!(!schedule.is_consolidation_hour(2));
        assert!(!schedule.is_consolidation_hour(12));
        assert!(!schedule.is_consolidation_hour(23));
    }

    #[test]
    fn test_daily_schedule_is_consolidation_hour_wraparound() {
        let schedule = DailySchedule {
            consolidation_window: (22, 2), // 10pm to 2am
            ..Default::default()
        };

        assert!(schedule.is_consolidation_hour(22));
        assert!(schedule.is_consolidation_hour(23));
        assert!(schedule.is_consolidation_hour(0));
        assert!(schedule.is_consolidation_hour(1));
        assert!(!schedule.is_consolidation_hour(2));
        assert!(!schedule.is_consolidation_hour(12));
        assert!(!schedule.is_consolidation_hour(21));
    }

    #[test]
    fn test_daily_schedule_next_check_for_hour() {
        let schedule = DailySchedule::default();

        // Consolidation window
        assert_eq!(
            schedule.next_check_for_hour(0),
            Some(ScheduledCheckType::ConsolidationWindow)
        );
        assert_eq!(
            schedule.next_check_for_hour(1),
            Some(ScheduledCheckType::ConsolidationWindow)
        );

        // Drift check hour
        assert_eq!(
            schedule.next_check_for_hour(6),
            Some(ScheduledCheckType::DriftCheck)
        );

        // Stats hour
        assert_eq!(
            schedule.next_check_for_hour(12),
            Some(ScheduledCheckType::StatisticsCollection)
        );

        // Prep hour
        assert_eq!(
            schedule.next_check_for_hour(18),
            Some(ScheduledCheckType::IndexOptimization)
        );

        // No scheduled check
        assert_eq!(schedule.next_check_for_hour(10), None);
        assert_eq!(schedule.next_check_for_hour(15), None);
    }

    #[test]
    fn test_daily_schedule_validate_valid() {
        let schedule = DailySchedule::default();
        assert!(schedule.validate().is_ok());

        let custom = DailySchedule {
            consolidation_window: (22, 4),
            drift_check_hour: 23,
            stats_hour: 0,
            prep_hour: 23,
        };
        assert!(custom.validate().is_ok());
    }

    #[test]
    fn test_daily_schedule_validate_invalid_hours() {
        let invalid_drift = DailySchedule {
            drift_check_hour: 24,
            ..Default::default()
        };
        assert!(matches!(
            invalid_drift.validate(),
            Err(ScheduleValidationError::InvalidHour { field, value }) if field == "drift_check_hour" && value == 24
        ));

        let invalid_stats = DailySchedule {
            stats_hour: 25,
            ..Default::default()
        };
        assert!(matches!(
            invalid_stats.validate(),
            Err(ScheduleValidationError::InvalidHour { field, value }) if field == "stats_hour" && value == 25
        ));

        let invalid_prep = DailySchedule {
            prep_hour: 100,
            ..Default::default()
        };
        assert!(matches!(
            invalid_prep.validate(),
            Err(ScheduleValidationError::InvalidHour { field, value }) if field == "prep_hour" && value == 100
        ));
    }

    #[test]
    fn test_daily_schedule_validate_invalid_window() {
        let invalid_window = DailySchedule {
            consolidation_window: (24, 2),
            ..Default::default()
        };
        assert!(matches!(
            invalid_window.validate(),
            Err(ScheduleValidationError::InvalidConsolidationWindow { .. })
        ));

        let invalid_window2 = DailySchedule {
            consolidation_window: (0, 25),
            ..Default::default()
        };
        assert!(matches!(
            invalid_window2.validate(),
            Err(ScheduleValidationError::InvalidConsolidationWindow { .. })
        ));
    }

    #[test]
    fn test_daily_schedule_serialization() {
        let schedule = DailySchedule::default();
        let json = serde_json::to_string(&schedule).expect("serialize");
        let deserialized: DailySchedule = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(
            deserialized.consolidation_window,
            schedule.consolidation_window
        );
        assert_eq!(deserialized.drift_check_hour, schedule.drift_check_hour);
        assert_eq!(deserialized.stats_hour, schedule.stats_hour);
        assert_eq!(deserialized.prep_hour, schedule.prep_hour);
    }

    // ScheduleValidationError tests
    #[test]
    fn test_schedule_validation_error_display() {
        let error = ScheduleValidationError::InvalidHour {
            field: "test_field".into(),
            value: 25,
        };
        assert!(error.to_string().contains("25"));
        assert!(error.to_string().contains("test_field"));

        let error = ScheduleValidationError::InvalidConsolidationWindow { start: 24, end: 25 };
        assert!(error.to_string().contains("24"));
        assert!(error.to_string().contains("25"));
    }

    // ScheduledCheckType tests
    #[test]
    fn test_scheduled_check_type_equality() {
        assert_eq!(
            ScheduledCheckType::DriftCheck,
            ScheduledCheckType::DriftCheck
        );
        assert_ne!(
            ScheduledCheckType::DriftCheck,
            ScheduledCheckType::ConsolidationWindow
        );
        assert_ne!(
            ScheduledCheckType::ConsolidationWindow,
            ScheduledCheckType::StatisticsCollection
        );
        assert_ne!(
            ScheduledCheckType::StatisticsCollection,
            ScheduledCheckType::IndexOptimization
        );
    }

    #[test]
    fn test_scheduled_check_type_serialization() {
        let types = [
            ScheduledCheckType::DriftCheck,
            ScheduledCheckType::ConsolidationWindow,
            ScheduledCheckType::StatisticsCollection,
            ScheduledCheckType::IndexOptimization,
        ];

        for check_type in types {
            let json = serde_json::to_string(&check_type).expect("serialize");
            let deserialized: ScheduledCheckType =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, check_type);
        }
    }

    // OptimizationEvent tests
    #[test]
    fn test_optimization_event_memory_stored() {
        let memory_id = MemoryId::new();
        let event = OptimizationEvent::MemoryStored {
            memory_id: memory_id.clone(),
        };

        assert_eq!(event.event_type_name(), "memory_stored");
        assert!(!event.is_urgent());
    }

    #[test]
    fn test_optimization_event_memory_retrieved() {
        let memory_id = MemoryId::new();
        let event = OptimizationEvent::MemoryRetrieved {
            memory_id: memory_id.clone(),
            query: "test query".into(),
        };

        assert_eq!(event.event_type_name(), "memory_retrieved");
        assert!(!event.is_urgent());
    }

    // REMOVED: test_optimization_event_north_star_updated per TASK-P0-005 (ARCH-03)

    #[test]
    fn test_optimization_event_goal_added() {
        let goal_id = GoalId::new();
        let event = OptimizationEvent::GoalAdded {
            goal_id: goal_id.clone(),
        };

        assert_eq!(event.event_type_name(), "goal_added");
        assert!(!event.is_urgent());
    }

    #[test]
    fn test_optimization_event_consciousness_dropped() {
        let event = OptimizationEvent::ConsciousnessDropped { level: 0.3 };

        assert_eq!(event.event_type_name(), "consciousness_dropped");
        assert!(event.is_urgent());
    }

    #[test]
    fn test_optimization_event_scheduled_check() {
        let event = OptimizationEvent::ScheduledCheck {
            check_type: ScheduledCheckType::DriftCheck,
        };

        assert_eq!(event.event_type_name(), "scheduled_check");
        assert!(!event.is_urgent());
    }

    #[test]
    fn test_optimization_event_serialization() {
        // TASK-P0-005: Removed NorthStarUpdated from events array per ARCH-03
        let events = [
            OptimizationEvent::MemoryStored {
                memory_id: MemoryId::new(),
            },
            OptimizationEvent::MemoryRetrieved {
                memory_id: MemoryId::new(),
                query: "test".into(),
            },
            OptimizationEvent::GoalAdded {
                goal_id: GoalId::new(),
            },
            OptimizationEvent::ConsciousnessDropped { level: 0.5 },
            OptimizationEvent::ScheduledCheck {
                check_type: ScheduledCheckType::DriftCheck,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).expect("serialize");
            let _deserialized: OptimizationEvent =
                serde_json::from_str(&json).expect("deserialize");
            // Can't compare events directly due to random IDs, but serialization works
        }
    }

    // AutonomousHealth tests
    #[test]
    fn test_autonomous_health_default() {
        let health = AutonomousHealth::default();
        assert_eq!(health, AutonomousHealth::Healthy);
        assert!(health.is_healthy());
        assert!(health.can_continue());
        assert!(health.message().is_none());
    }

    #[test]
    fn test_autonomous_health_warning() {
        let health = AutonomousHealth::warning("Test warning");
        assert!(!health.is_healthy());
        assert!(health.can_continue());
        assert_eq!(health.message(), Some("Test warning"));
    }

    #[test]
    fn test_autonomous_health_recoverable_error() {
        let health = AutonomousHealth::recoverable_error("Recoverable error");
        assert!(!health.is_healthy());
        assert!(health.can_continue());
        assert_eq!(health.message(), Some("Recoverable error"));
    }

    #[test]
    fn test_autonomous_health_fatal_error() {
        let health = AutonomousHealth::fatal_error("Fatal error");
        assert!(!health.is_healthy());
        assert!(!health.can_continue());
        assert_eq!(health.message(), Some("Fatal error"));
    }

    #[test]
    fn test_autonomous_health_equality() {
        assert_eq!(AutonomousHealth::Healthy, AutonomousHealth::Healthy);
        assert_ne!(
            AutonomousHealth::Healthy,
            AutonomousHealth::Warning {
                message: "test".into()
            }
        );
        assert_eq!(
            AutonomousHealth::Warning {
                message: "test".into()
            },
            AutonomousHealth::Warning {
                message: "test".into()
            }
        );
        assert_ne!(
            AutonomousHealth::Warning {
                message: "test1".into()
            },
            AutonomousHealth::Warning {
                message: "test2".into()
            }
        );
    }

    #[test]
    fn test_autonomous_health_serialization() {
        let healths = [
            AutonomousHealth::Healthy,
            AutonomousHealth::Warning {
                message: "test warning".into(),
            },
            AutonomousHealth::Error {
                message: "recoverable".into(),
                recoverable: true,
            },
            AutonomousHealth::Error {
                message: "fatal".into(),
                recoverable: false,
            },
        ];

        for health in healths {
            let json = serde_json::to_string(&health).expect("serialize");
            let deserialized: AutonomousHealth = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, health);
        }
    }

    // AutonomousStatus tests
    #[test]
    fn test_autonomous_status_default() {
        let status = AutonomousStatus::default();
        assert!(!status.enabled);
        assert!(!status.bootstrap_complete);
        assert!(!status.strategic_goal_configured);
        assert_eq!(status.pending_prune_count, 0);
        assert_eq!(status.consolidation_queue_size, 0);
        assert!(status.next_scheduled.is_none());
        assert_eq!(status.health, AutonomousHealth::Healthy);
    }

    #[test]
    fn test_autonomous_status_initialized() {
        let status = AutonomousStatus::initialized(true);
        assert!(status.enabled);
        assert!(status.bootstrap_complete);
        assert!(status.strategic_goal_configured);
        assert!(status.health.is_healthy());
    }

    #[test]
    // TASK-P0-005: Renamed from test_autonomous_status_initialized_without_north_star
    fn test_autonomous_status_initialized_without_strategic_goal() {
        let status = AutonomousStatus::initialized(false);
        assert!(status.enabled);
        assert!(status.bootstrap_complete);
        assert!(!status.strategic_goal_configured);
    }

    #[test]
    fn test_autonomous_status_is_ready() {
        // Not ready - disabled
        let status = AutonomousStatus::default();
        assert!(!status.is_ready());

        // Ready - enabled and bootstrapped
        let status = AutonomousStatus::initialized(true);
        assert!(status.is_ready());

        // Not ready - has fatal error
        let mut status = AutonomousStatus::initialized(true);
        status.health = AutonomousHealth::fatal_error("Fatal");
        assert!(!status.is_ready());

        // Ready - has warning
        let mut status = AutonomousStatus::initialized(true);
        status.health = AutonomousHealth::warning("Warning");
        assert!(status.is_ready());

        // Ready - has recoverable error
        let mut status = AutonomousStatus::initialized(true);
        status.health = AutonomousHealth::recoverable_error("Recoverable");
        assert!(status.is_ready());
    }

    #[test]
    fn test_autonomous_status_has_pending_work() {
        let status = AutonomousStatus::default();
        assert!(!status.has_pending_work());

        // With pending prune
        let status = AutonomousStatus {
            pending_prune_count: 5,
            ..AutonomousStatus::default()
        };
        assert!(status.has_pending_work());

        // With consolidation queue
        let status = AutonomousStatus {
            consolidation_queue_size: 10,
            ..AutonomousStatus::default()
        };
        assert!(status.has_pending_work());

        // With drift requiring attention
        let mut status = AutonomousStatus::default();
        let drift_config = DriftConfig::default();
        status.drift_state.add_data_point(0.65, 5, &drift_config); // Creates moderate drift
        status.drift_state.baseline = 0.75;
        status.drift_state.add_data_point(0.65, 5, &drift_config);
        assert!(status.has_pending_work());
    }

    #[test]
    fn test_autonomous_status_summary() {
        let status = AutonomousStatus::default();
        let summary = status.summary();
        assert!(summary.contains("enabled=false"));
        assert!(summary.contains("ready=false"));
        assert!(summary.contains("healthy"));

        let status = AutonomousStatus::initialized(true);
        let summary = status.summary();
        assert!(summary.contains("enabled=true"));
        assert!(summary.contains("ready=true"));

        let status = AutonomousStatus {
            pending_prune_count: 5,
            consolidation_queue_size: 10,
            ..AutonomousStatus::default()
        };
        let summary = status.summary();
        assert!(summary.contains("prune=5"));
        assert!(summary.contains("consolidate=10"));
    }

    #[test]
    fn test_autonomous_status_summary_with_errors() {
        let status = AutonomousStatus {
            health: AutonomousHealth::warning("Warning"),
            ..AutonomousStatus::default()
        };
        let summary = status.summary();
        assert!(summary.contains("warning"));

        let status = AutonomousStatus {
            health: AutonomousHealth::recoverable_error("Error"),
            ..AutonomousStatus::default()
        };
        let summary = status.summary();
        assert!(summary.contains("recoverable"));

        let status = AutonomousStatus {
            health: AutonomousHealth::fatal_error("Fatal"),
            ..AutonomousStatus::default()
        };
        let summary = status.summary();
        assert!(summary.contains("fatal"));
    }

    #[test]
    fn test_autonomous_status_serialization() {
        let status = AutonomousStatus::initialized(true);
        let json = serde_json::to_string(&status).expect("serialize");
        let deserialized: AutonomousStatus = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.enabled, status.enabled);
        assert_eq!(deserialized.bootstrap_complete, status.bootstrap_complete);
        assert_eq!(
            deserialized.strategic_goal_configured,
            status.strategic_goal_configured
        );
        assert_eq!(deserialized.pending_prune_count, status.pending_prune_count);
        assert_eq!(
            deserialized.consolidation_queue_size,
            status.consolidation_queue_size
        );
        assert_eq!(deserialized.health, status.health);
    }

    #[test]
    fn test_autonomous_status_with_all_fields_populated() {
        let now = Utc::now();
        let next = now + chrono::Duration::hours(1);

        let status = AutonomousStatus {
            enabled: true,
            bootstrap_complete: true,
            strategic_goal_configured: true,
            drift_state: DriftState::with_baseline(0.80),
            threshold_state: AdaptiveThresholdState::default(),
            pending_prune_count: 15,
            consolidation_queue_size: 25,
            last_optimization: now,
            next_scheduled: Some(next),
            health: AutonomousHealth::warning("Test warning"),
        };

        // Verify all fields
        assert!(status.enabled);
        assert!(status.bootstrap_complete);
        assert!(status.strategic_goal_configured);
        assert!((status.drift_state.baseline - 0.80).abs() < f32::EPSILON);
        assert_eq!(status.pending_prune_count, 15);
        assert_eq!(status.consolidation_queue_size, 25);
        assert!(status.next_scheduled.is_some());
        assert!(!status.health.is_healthy());

        // Test serialization roundtrip
        let json = serde_json::to_string(&status).expect("serialize");
        let deserialized: AutonomousStatus = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.enabled, status.enabled);
        assert_eq!(deserialized.pending_prune_count, status.pending_prune_count);
        assert!(deserialized.next_scheduled.is_some());
    }

    // Integration tests
    #[test]
    fn test_config_and_status_interaction() {
        let config = AutonomousConfig::default();

        // Simulate initialization based on config
        let status = AutonomousStatus {
            enabled: config.enabled,
            bootstrap_complete: config.bootstrap.auto_init,
            ..AutonomousStatus::default()
        };

        assert!(status.enabled);
        assert!(status.bootstrap_complete);
    }

    #[test]
    fn test_schedule_generates_events() {
        let schedule = DailySchedule::default();

        // Check that each scheduled hour produces an event
        for hour in 0..24 {
            if let Some(check_type) = schedule.next_check_for_hour(hour) {
                let event = OptimizationEvent::ScheduledCheck { check_type };
                assert_eq!(event.event_type_name(), "scheduled_check");
            }
        }
    }

    #[test]
    fn test_health_transitions() {
        let mut health = AutonomousHealth::Healthy;
        assert!(health.is_healthy());
        assert!(health.can_continue());

        health = AutonomousHealth::warning("Minor issue");
        assert!(!health.is_healthy());
        assert!(health.can_continue());

        health = AutonomousHealth::recoverable_error("Temporary failure");
        assert!(!health.is_healthy());
        assert!(health.can_continue());

        health = AutonomousHealth::fatal_error("Critical failure");
        assert!(!health.is_healthy());
        assert!(!health.can_continue());
    }
}
