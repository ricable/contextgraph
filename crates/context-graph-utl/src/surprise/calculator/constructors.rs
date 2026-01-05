//! Constructors and Default implementation for SurpriseCalculator.

use crate::config::{KlConfig, SurpriseConfig};

use super::super::embedding_distance::EmbeddingDistanceCalculator;
use super::super::kl_divergence::KlDivergenceCalculator;
use super::types::SurpriseCalculator;

impl SurpriseCalculator {
    /// Create a new SurpriseCalculator from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The surprise configuration settings
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::SurpriseConfig;
    /// use context_graph_utl::surprise::SurpriseCalculator;
    ///
    /// let config = SurpriseConfig::default();
    /// let calculator = SurpriseCalculator::new(&config);
    /// ```
    pub fn new(config: &SurpriseConfig) -> Self {
        Self {
            entropy_weight: config.entropy_weight,
            novelty_boost: config.novelty_boost,
            repetition_decay: config.repetition_decay,
            min_threshold: config.min_threshold,
            max_value: config.max_value,
            use_ema: config.use_ema,
            ema_alpha: config.ema_alpha,
            embedding_calc: EmbeddingDistanceCalculator::from_config(config),
            kl_calc: KlDivergenceCalculator::default(),
            ema_state: None,
        }
    }

    /// Create a calculator with custom KL divergence settings.
    ///
    /// # Arguments
    ///
    /// * `config` - The surprise configuration settings
    /// * `kl_config` - The KL divergence configuration
    pub fn with_kl_config(config: &SurpriseConfig, kl_config: &KlConfig) -> Self {
        Self {
            entropy_weight: config.entropy_weight,
            novelty_boost: config.novelty_boost,
            repetition_decay: config.repetition_decay,
            min_threshold: config.min_threshold,
            max_value: config.max_value,
            use_ema: config.use_ema,
            ema_alpha: config.ema_alpha,
            embedding_calc: EmbeddingDistanceCalculator::from_config(config),
            kl_calc: KlDivergenceCalculator::from_config(config, kl_config),
            ema_state: None,
        }
    }
}

impl Default for SurpriseCalculator {
    fn default() -> Self {
        Self::new(&SurpriseConfig::default())
    }
}
