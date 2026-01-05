//! Type definitions for the SurpriseCalculator.
//!
//! Contains the main `SurpriseCalculator` struct and its field definitions.

use super::super::embedding_distance::EmbeddingDistanceCalculator;
use super::super::kl_divergence::KlDivergenceCalculator;

/// Main calculator for computing surprise (Delta-S) values.
///
/// Combines multiple signals to compute a unified surprise score:
/// - Embedding-based surprise: measures semantic novelty
/// - KL divergence: measures distribution change
///
/// The final surprise score is a weighted combination of these signals,
/// clamped to the valid [0, 1] range per AP-009.
///
/// # Example
///
/// ```
/// use context_graph_utl::config::SurpriseConfig;
/// use context_graph_utl::surprise::SurpriseCalculator;
///
/// let config = SurpriseConfig::default();
/// let calculator = SurpriseCalculator::new(&config);
///
/// let current = vec![0.1, 0.2, 0.3, 0.4];
/// let history = vec![
///     vec![0.15, 0.25, 0.35, 0.25],
///     vec![0.12, 0.22, 0.32, 0.34],
/// ];
///
/// let surprise = calculator.compute_surprise(&current, &history);
/// assert!(surprise >= 0.0 && surprise <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct SurpriseCalculator {
    /// Weight for entropy-based surprise component.
    pub(crate) entropy_weight: f32,
    /// Boost factor for novel items.
    pub(crate) novelty_boost: f32,
    /// Decay rate for repeated exposure.
    pub(crate) repetition_decay: f32,
    /// Minimum surprise threshold.
    pub(crate) min_threshold: f32,
    /// Maximum surprise value.
    pub(crate) max_value: f32,
    /// Whether to use EMA smoothing.
    pub(crate) use_ema: bool,
    /// EMA alpha (smoothing factor).
    pub(crate) ema_alpha: f32,
    /// Embedding distance calculator.
    pub(crate) embedding_calc: EmbeddingDistanceCalculator,
    /// KL divergence calculator.
    pub(crate) kl_calc: KlDivergenceCalculator,
    /// Current EMA state for smoothing.
    pub(crate) ema_state: Option<f32>,
}
