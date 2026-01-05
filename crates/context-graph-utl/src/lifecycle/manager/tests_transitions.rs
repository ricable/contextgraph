//! Transition tests for lifecycle manager.

#[cfg(test)]
mod tests {
    use crate::config::LifecycleConfig;
    use crate::error::UtlError;
    use crate::lifecycle::manager::LifecycleManager;
    use crate::lifecycle::stage::LifecycleStage;

    fn test_config() -> LifecycleConfig {
        LifecycleConfig::default()
    }

    #[test]
    fn test_increment() {
        let config = test_config();
        let mut manager = LifecycleManager::new(&config);

        for _ in 0..49 {
            let transitioned = manager.increment();
            assert!(!transitioned);
            assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
        }

        // At 49 interactions, still Infancy
        assert_eq!(manager.interaction_count(), 49);
    }

    #[test]
    fn test_auto_transition_to_growth() {
        let mut config = test_config();
        config.transition_hysteresis = 0; // Disable hysteresis for test
        let mut manager = LifecycleManager::new(&config);

        // Increment to 50
        for _ in 0..50 {
            manager.increment();
        }

        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
        assert_eq!(manager.interaction_count(), 50);
    }

    #[test]
    fn test_auto_transition_to_maturity() {
        let mut config = test_config();
        config.transition_hysteresis = 0;
        let mut manager = LifecycleManager::new(&config);

        manager.increment_by(500);

        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);
        assert_eq!(manager.interaction_count(), 500);
    }

    #[test]
    fn test_increment_by() {
        let mut config = test_config();
        config.transition_hysteresis = 0;
        let mut manager = LifecycleManager::new(&config);

        let transitioned = manager.increment_by(100);
        assert!(transitioned);
        assert_eq!(manager.interaction_count(), 100);
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    }

    #[test]
    fn test_manual_transition() {
        let config = test_config();
        let mut manager = LifecycleManager::new(&config);

        // Forward transition should succeed
        assert!(manager.transition_to(LifecycleStage::Growth).is_ok());
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
        assert!(manager.interaction_count() >= 50); // Count updated to stage minimum

        // Skip to Maturity should also succeed
        assert!(manager.transition_to(LifecycleStage::Maturity).is_ok());
        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);
    }

    #[test]
    fn test_invalid_backward_transition() {
        let config = test_config();
        let mut manager = LifecycleManager::with_count(&config, 100);

        assert_eq!(manager.current_stage(), LifecycleStage::Growth);

        // Backward transition should fail
        let result = manager.transition_to(LifecycleStage::Infancy);
        assert!(result.is_err());

        match result.unwrap_err() {
            UtlError::InvalidLifecycleTransition { from, to, .. } => {
                assert_eq!(from, "Growth");
                assert_eq!(to, "Infancy");
            }
            _ => panic!("Expected InvalidLifecycleTransition error"),
        }
    }

    #[test]
    fn test_same_stage_transition() {
        let config = test_config();
        let mut manager = LifecycleManager::with_count(&config, 100);

        // Transition to same stage should succeed
        assert!(manager.transition_to(LifecycleStage::Growth).is_ok());
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let mut manager = LifecycleManager::with_count(&config, 600);

        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);

        manager.reset();

        assert_eq!(manager.interaction_count(), 0);
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
    }

    #[test]
    fn test_hysteresis() {
        let mut config = test_config();
        config.transition_hysteresis = 10;
        let mut manager = LifecycleManager::with_count(&config, 45);

        // At 45, we're in Infancy
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);

        // Increment to 50 - but hysteresis prevents immediate transition
        manager.increment_by(5);
        assert_eq!(manager.interaction_count(), 50);
        // Due to hysteresis, we might still be in Infancy
        // (depends on last_transition_count)

        // Continue incrementing until past hysteresis
        manager.increment_by(10);
        // Now we should have transitioned
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    }

    #[test]
    fn test_full_lifecycle_progression() {
        let mut config = test_config();
        config.transition_hysteresis = 0;
        let mut manager = LifecycleManager::new(&config);

        // Start in Infancy
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);

        // Progress through Infancy
        for i in 1..50 {
            manager.increment();
            assert_eq!(manager.interaction_count(), i);
            assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
        }

        // Transition to Growth at 50
        manager.increment();
        assert_eq!(manager.interaction_count(), 50);
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);

        // Progress through Growth
        manager.increment_by(449);
        assert_eq!(manager.interaction_count(), 499);
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);

        // Transition to Maturity at 500
        manager.increment();
        assert_eq!(manager.interaction_count(), 500);
        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);

        // Stay in Maturity indefinitely
        manager.increment_by(1000);
        assert_eq!(manager.interaction_count(), 1500);
        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);
    }
}
