//! Tests for UTL configuration types.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::config::*;

    #[test]
    fn test_utl_config_default() {
        let config = UtlConfig::default();
        assert!(config.validate().is_ok());
        assert!(!config.debug);
    }

    #[test]
    fn test_utl_config_presets() {
        let exploration = UtlConfig::exploration_preset();
        assert!(exploration.validate().is_ok());
        assert!(exploration.surprise.entropy_weight > 0.5);

        let curation = UtlConfig::curation_preset();
        assert!(curation.validate().is_ok());
        assert!(curation.coherence.similarity_weight > 0.5);
    }

    #[test]
    fn test_surprise_config_validation() {
        let valid = SurpriseConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = SurpriseConfig {
            entropy_weight: 1.5, // Out of range
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_coherence_config_validation() {
        let valid = CoherenceConfig::default();
        assert!(valid.validate().is_ok());

        // Edge case 1: Zero neighbor count
        let invalid = CoherenceConfig {
            neighbor_count: 0, // Invalid - must be > 0
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Edge case 2: Out-of-range weight
        let invalid_weight = CoherenceConfig {
            similarity_weight: 1.5, // Invalid - must be in [0.0, 1.0]
            ..Default::default()
        };
        assert!(invalid_weight.validate().is_err());

        // Edge case 3: Negative weight
        let negative_weight = CoherenceConfig {
            graph_weight: -0.1, // Invalid - must be in [0.0, 1.0]
            ..Default::default()
        };
        assert!(negative_weight.validate().is_err());
    }

    #[test]
    fn test_emotional_config_validation() {
        let valid = EmotionalConfig::default();
        assert!(valid.validate().is_ok());
        assert_eq!(valid.clamp(0.3), 0.5); // Below min
        assert_eq!(valid.clamp(2.0), 1.5); // Above max
        assert_eq!(valid.clamp(1.0), 1.0); // In range

        let invalid = EmotionalConfig {
            min_weight: 1.0,
            max_weight: 0.5, // max < min
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_phase_config_validation() {
        let valid = PhaseConfig::default();
        assert!(valid.validate().is_ok());
        assert_eq!(valid.clamp(-0.5), 0.0); // Below min
        assert_eq!(valid.clamp(4.0), std::f32::consts::PI); // Above max

        let invalid = PhaseConfig {
            frequency_hz: 0.0, // Invalid
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_johari_config_validation() {
        let valid = JohariConfig::default();
        assert!(valid.validate().is_ok());
        assert_eq!(valid.surprise_threshold, 0.5);
        assert_eq!(valid.coherence_threshold, 0.5);

        let invalid = JohariConfig {
            boundary_width: 0.5, // Out of range
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_stage_config_validation() {
        let infancy = StageConfig::infancy();
        assert!(infancy.validate().is_ok());
        assert_eq!(infancy.lambda_novelty + infancy.lambda_consolidation, 1.0);

        let growth = StageConfig::growth();
        assert!(growth.validate().is_ok());
        assert_eq!(growth.lambda_novelty, 0.5);

        let maturity = StageConfig::maturity();
        assert!(maturity.validate().is_ok());
        assert_eq!(maturity.lambda_consolidation, 0.7);

        let invalid = StageConfig {
            lambda_novelty: 0.6,
            lambda_consolidation: 0.6, // Sum > 1
            ..StageConfig::infancy()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_lifecycle_config_validation() {
        let valid = LifecycleConfig::default();
        assert!(valid.validate().is_ok());
        assert_eq!(valid.stages.len(), 3);

        // Test stage lookup
        assert_eq!(valid.get_stage(0).unwrap().name, "Infancy");
        assert_eq!(valid.get_stage(50).unwrap().name, "Growth");
        assert_eq!(valid.get_stage(500).unwrap().name, "Maturity");
        assert_eq!(valid.get_stage(1000).unwrap().name, "Maturity");
    }

    #[test]
    fn test_lifecycle_focused_presets() {
        let infancy = LifecycleConfig::infancy_focused();
        assert!(infancy.validate().is_ok());
        assert_eq!(infancy.stages[0].lambda_novelty, 0.8);

        let maturity = LifecycleConfig::maturity_focused();
        assert!(maturity.validate().is_ok());
        assert_eq!(maturity.stages[2].lambda_consolidation, 0.8);
    }

    #[test]
    fn test_kl_config_validation() {
        let valid = KlConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = KlConfig {
            epsilon: 1e-2, // Out of range
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_utl_thresholds_validation() {
        let valid = UtlThresholds::default();
        assert!(valid.validate().is_ok());
        assert_eq!(valid.high_quality, 0.6);
        assert_eq!(valid.info_loss_tolerance, 0.15);

        // Test clamping
        assert_eq!(valid.clamp_score(0.5), 0.5);
        assert_eq!(valid.clamp_score(-0.5), 0.0);
        assert_eq!(valid.clamp_score(1.5), 1.0);
        assert_eq!(valid.clamp_score(f32::NAN), 0.0);
        assert_eq!(valid.clamp_score(f32::INFINITY), 1.0);
        assert_eq!(valid.clamp_score(f32::NEG_INFINITY), 0.0);

        // Test quality checks
        assert!(valid.is_high_quality(0.7));
        assert!(!valid.is_high_quality(0.5));
        assert!(valid.is_low_quality(0.2));
        assert!(!valid.is_low_quality(0.5));

        let invalid = UtlThresholds {
            min_score: 1.0,
            max_score: 0.0, // max < min
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_serialization() {
        let config = UtlConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: UtlConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.validate().is_ok());
    }

    #[test]
    fn test_constitution_compliance() {
        // Verify defaults match constitution.yaml specifications
        let config = UtlConfig::default();

        // Emotional weight range: [0.5, 1.5]
        assert_eq!(config.emotional.min_weight, 0.5);
        assert_eq!(config.emotional.max_weight, 1.5);

        // Phase range: [0, pi]
        assert_eq!(config.phase.min_phase, 0.0);
        assert!((config.phase.max_phase - std::f32::consts::PI).abs() < 0.001);

        // Johari thresholds
        assert_eq!(config.johari.surprise_threshold, 0.5);
        assert_eq!(config.johari.coherence_threshold, 0.5);

        // Lifecycle stages
        let stages = &config.lifecycle.stages;
        assert_eq!(stages.len(), 3);

        // Infancy: lambda_dS=0.7, lambda_dC=0.3
        assert_eq!(stages[0].lambda_novelty, 0.7);
        assert_eq!(stages[0].lambda_consolidation, 0.3);

        // Growth: lambda_dS=0.5, lambda_dC=0.5
        assert_eq!(stages[1].lambda_novelty, 0.5);
        assert_eq!(stages[1].lambda_consolidation, 0.5);

        // Maturity: lambda_dS=0.3, lambda_dC=0.7
        assert_eq!(stages[2].lambda_novelty, 0.3);
        assert_eq!(stages[2].lambda_consolidation, 0.7);

        // UTL quality target: > 0.6
        assert_eq!(config.thresholds.high_quality, 0.6);

        // Info loss tolerance: < 15%
        assert_eq!(config.thresholds.info_loss_tolerance, 0.15);

        // Compression target: > 60%
        assert_eq!(config.thresholds.compression_target, 0.6);
    }
}
