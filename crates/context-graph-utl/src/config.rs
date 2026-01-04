//! UTL configuration types.
//!
//! This module defines comprehensive configuration types for the UTL (Unified Theory of Learning)
//! computation engine, including surprise, coherence, emotional weighting, phase oscillation,
//! lifecycle stages, Johari classification, and KL divergence settings.
//!
//! # Constitution Reference
//!
//! The UTL formula is: `L = f((ΔS × ΔC) · wₑ · cos φ)`
//!
//! Where:
//! - `ΔS`: Entropy/novelty change in range `[0, 1]`
//! - `ΔC`: Coherence/understanding change in range `[0, 1]`
//! - `wₑ`: Emotional weight in range `[0.5, 1.5]`
//! - `φ`: Phase synchronization angle in range `[0, π]`

use serde::{Deserialize, Serialize};

/// Main UTL configuration containing all subsystem settings.
///
/// This is the top-level configuration struct that aggregates all UTL-related
/// configuration options. It provides sensible defaults based on the constitution.yaml
/// specifications.
///
/// # Example
///
/// ```
/// use context_graph_utl::config::UtlConfig;
///
/// let config = UtlConfig::default();
/// assert!(config.surprise.entropy_weight > 0.0);
/// assert!(config.lifecycle.stages.len() == 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UtlConfig {
    /// Surprise computation settings.
    pub surprise: SurpriseConfig,

    /// Coherence computation settings.
    pub coherence: CoherenceConfig,

    /// Emotional weight settings.
    pub emotional: EmotionalConfig,

    /// Phase oscillation settings.
    pub phase: PhaseConfig,

    /// Johari quadrant classification thresholds.
    pub johari: JohariConfig,

    /// Lifecycle stage configurations.
    pub lifecycle: LifecycleConfig,

    /// KL divergence settings.
    pub kl: KlConfig,

    /// UTL computation thresholds.
    pub thresholds: UtlThresholds,

    /// Enable debug logging for UTL computations.
    #[serde(default)]
    pub debug: bool,
}

impl UtlConfig {
    /// Create a new configuration with custom settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for high-novelty exploration.
    ///
    /// This preset emphasizes surprise detection over coherence,
    /// suitable for early-stage learning or discovery tasks.
    pub fn exploration_preset() -> Self {
        Self {
            surprise: SurpriseConfig {
                entropy_weight: 0.8,
                novelty_boost: 1.2,
                ..Default::default()
            },
            lifecycle: LifecycleConfig::infancy_focused(),
            ..Default::default()
        }
    }

    /// Create a configuration optimized for coherence-focused curation.
    ///
    /// This preset emphasizes coherence and consolidation over novelty,
    /// suitable for mature knowledge bases.
    pub fn curation_preset() -> Self {
        Self {
            coherence: CoherenceConfig {
                similarity_weight: 0.8,
                consistency_weight: 0.9,
                ..Default::default()
            },
            lifecycle: LifecycleConfig::maturity_focused(),
            ..Default::default()
        }
    }

    /// Validate the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), String> {
        self.surprise.validate()?;
        self.coherence.validate()?;
        self.emotional.validate()?;
        self.phase.validate()?;
        self.johari.validate()?;
        self.lifecycle.validate()?;
        self.kl.validate()?;
        self.thresholds.validate()?;
        Ok(())
    }
}

/// Surprise (ΔS) computation settings.
///
/// Controls how surprise/entropy is computed for knowledge items.
/// Surprise measures the novelty or unexpectedness of information.
///
/// # Constitution Reference
///
/// - `ΔS` range: `[0, 1]` representing entropy/novelty
/// - Higher values indicate more surprising/novel information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseConfig {
    /// Weight applied to entropy component.
    /// Range: `[0.0, 1.0]`
    pub entropy_weight: f32,

    /// Boost factor for novel items.
    /// Range: `[0.5, 2.0]`
    pub novelty_boost: f32,

    /// Decay rate for repeated exposure.
    /// Range: `[0.0, 1.0]`
    pub repetition_decay: f32,

    /// Minimum surprise threshold below which items are considered familiar.
    /// Range: `[0.0, 0.5]`
    pub min_threshold: f32,

    /// Maximum surprise value (for clamping).
    /// Range: `[0.5, 1.0]`
    pub max_value: f32,

    /// Number of samples for entropy estimation.
    pub sample_count: usize,

    /// Use exponential moving average for smoothing.
    pub use_ema: bool,

    /// EMA alpha (smoothing factor).
    /// Range: `[0.0, 1.0]`
    pub ema_alpha: f32,
}

impl Default for SurpriseConfig {
    fn default() -> Self {
        Self {
            entropy_weight: 0.6,
            novelty_boost: 1.0,
            repetition_decay: 0.1,
            min_threshold: 0.05,
            max_value: 1.0,
            sample_count: 100,
            use_ema: true,
            ema_alpha: 0.3,
        }
    }
}

impl SurpriseConfig {
    /// Validate the surprise configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.entropy_weight) {
            return Err(format!(
                "entropy_weight must be in [0, 1], got {}",
                self.entropy_weight
            ));
        }
        if !(0.5..=2.0).contains(&self.novelty_boost) {
            return Err(format!(
                "novelty_boost must be in [0.5, 2.0], got {}",
                self.novelty_boost
            ));
        }
        if !(0.0..=1.0).contains(&self.repetition_decay) {
            return Err(format!(
                "repetition_decay must be in [0, 1], got {}",
                self.repetition_decay
            ));
        }
        if !(0.0..=0.5).contains(&self.min_threshold) {
            return Err(format!(
                "min_threshold must be in [0, 0.5], got {}",
                self.min_threshold
            ));
        }
        if !(0.5..=1.0).contains(&self.max_value) {
            return Err(format!(
                "max_value must be in [0.5, 1.0], got {}",
                self.max_value
            ));
        }
        if self.sample_count == 0 {
            return Err("sample_count must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.ema_alpha) {
            return Err(format!(
                "ema_alpha must be in [0, 1], got {}",
                self.ema_alpha
            ));
        }
        Ok(())
    }
}

/// Coherence (ΔC) computation settings.
///
/// Controls how coherence/understanding is computed for knowledge items.
/// Coherence measures how well information fits with existing knowledge.
///
/// # Constitution Reference
///
/// - `ΔC` range: `[0, 1]` representing coherence/understanding
/// - Higher values indicate better integration with existing knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Weight for semantic similarity contribution.
    /// Range: `[0.0, 1.0]`
    pub similarity_weight: f32,

    /// Weight for consistency with existing knowledge.
    /// Range: `[0.0, 1.0]`
    pub consistency_weight: f32,

    /// Minimum coherence threshold.
    /// Range: `[0.0, 0.5]`
    pub min_threshold: f32,

    /// Decay rate for coherence over time without reinforcement.
    /// Range: `[0.0, 1.0]`
    pub decay_rate: f32,

    /// Number of neighbors to consider for coherence calculation.
    pub neighbor_count: usize,

    /// Minimum similarity for neighbor consideration.
    /// Range: `[0.0, 1.0]`
    pub neighbor_similarity_min: f32,

    /// Enable graph-based coherence computation.
    pub use_graph_coherence: bool,

    /// Weight for graph-based coherence.
    /// Range: `[0.0, 1.0]`
    pub graph_weight: f32,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 0.5,
            consistency_weight: 0.7,
            min_threshold: 0.1,
            decay_rate: 0.05,
            neighbor_count: 10,
            neighbor_similarity_min: 0.3,
            use_graph_coherence: true,
            graph_weight: 0.4,
        }
    }
}

impl CoherenceConfig {
    /// Validate the coherence configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.similarity_weight) {
            return Err(format!(
                "similarity_weight must be in [0, 1], got {}",
                self.similarity_weight
            ));
        }
        if !(0.0..=1.0).contains(&self.consistency_weight) {
            return Err(format!(
                "consistency_weight must be in [0, 1], got {}",
                self.consistency_weight
            ));
        }
        if !(0.0..=0.5).contains(&self.min_threshold) {
            return Err(format!(
                "min_threshold must be in [0, 0.5], got {}",
                self.min_threshold
            ));
        }
        if !(0.0..=1.0).contains(&self.decay_rate) {
            return Err(format!(
                "decay_rate must be in [0, 1], got {}",
                self.decay_rate
            ));
        }
        if self.neighbor_count == 0 {
            return Err("neighbor_count must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.neighbor_similarity_min) {
            return Err(format!(
                "neighbor_similarity_min must be in [0, 1], got {}",
                self.neighbor_similarity_min
            ));
        }
        if !(0.0..=1.0).contains(&self.graph_weight) {
            return Err(format!(
                "graph_weight must be in [0, 1], got {}",
                self.graph_weight
            ));
        }
        Ok(())
    }
}

/// Emotional weight (wₑ) settings.
///
/// Controls how emotional salience affects learning score computation.
/// Emotional weight modulates the overall learning signal.
///
/// # Constitution Reference
///
/// - `wₑ` range: `[0.5, 1.5]` representing emotional weight
/// - Default value is `1.0` (neutral)
/// - Values `> 1.0` amplify learning, values `< 1.0` dampen it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalConfig {
    /// Minimum emotional weight.
    /// Constitution specifies: `0.5`
    pub min_weight: f32,

    /// Maximum emotional weight.
    /// Constitution specifies: `1.5`
    pub max_weight: f32,

    /// Default/neutral emotional weight.
    pub default_weight: f32,

    /// Decay rate for emotional salience over time.
    /// Range: `[0.0, 1.0]`
    pub decay_rate: f32,

    /// Enable arousal-based modulation.
    pub arousal_modulation: bool,

    /// Arousal sensitivity factor.
    /// Range: `[0.0, 2.0]`
    pub arousal_sensitivity: f32,

    /// Enable valence-based modulation.
    pub valence_modulation: bool,

    /// Valence sensitivity factor.
    /// Range: `[0.0, 2.0]`
    pub valence_sensitivity: f32,
}

impl Default for EmotionalConfig {
    fn default() -> Self {
        Self {
            min_weight: 0.5,
            max_weight: 1.5,
            default_weight: 1.0,
            decay_rate: 0.02,
            arousal_modulation: true,
            arousal_sensitivity: 1.0,
            valence_modulation: true,
            valence_sensitivity: 1.0,
        }
    }
}

impl EmotionalConfig {
    /// Validate the emotional configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_weight < 0.0 {
            return Err(format!("min_weight must be >= 0, got {}", self.min_weight));
        }
        if self.max_weight < self.min_weight {
            return Err(format!(
                "max_weight ({}) must be >= min_weight ({})",
                self.max_weight, self.min_weight
            ));
        }
        if !(self.min_weight..=self.max_weight).contains(&self.default_weight) {
            return Err(format!(
                "default_weight must be in [{}, {}], got {}",
                self.min_weight, self.max_weight, self.default_weight
            ));
        }
        if !(0.0..=1.0).contains(&self.decay_rate) {
            return Err(format!(
                "decay_rate must be in [0, 1], got {}",
                self.decay_rate
            ));
        }
        if !(0.0..=2.0).contains(&self.arousal_sensitivity) {
            return Err(format!(
                "arousal_sensitivity must be in [0, 2], got {}",
                self.arousal_sensitivity
            ));
        }
        if !(0.0..=2.0).contains(&self.valence_sensitivity) {
            return Err(format!(
                "valence_sensitivity must be in [0, 2], got {}",
                self.valence_sensitivity
            ));
        }
        Ok(())
    }

    /// Clamp a weight value to the valid range.
    pub fn clamp(&self, weight: f32) -> f32 {
        weight.clamp(self.min_weight, self.max_weight)
    }
}

/// Phase oscillation (φ) settings.
///
/// Controls phase synchronization between different cognitive processes.
/// Phase alignment affects the coupling of learning signals.
///
/// # Constitution Reference
///
/// - `φ` range: `[0, π]` representing phase angle
/// - `cos(φ) = 1.0` when fully synchronized (φ = 0)
/// - `cos(φ) = -1.0` when anti-phase (φ = π)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Minimum phase angle (radians).
    /// Constitution specifies: `0.0`
    pub min_phase: f32,

    /// Maximum phase angle (radians).
    /// Constitution specifies: `π` (~3.14159)
    pub max_phase: f32,

    /// Default phase angle (radians).
    pub default_phase: f32,

    /// Phase oscillation frequency (Hz).
    /// Constitution reference: L4 operates at 100Hz
    pub frequency_hz: f32,

    /// Phase coupling strength.
    /// Range: `[0.0, 1.0]`
    pub coupling_strength: f32,

    /// Enable adaptive phase adjustment.
    pub adaptive: bool,

    /// Phase adaptation rate.
    /// Range: `[0.0, 1.0]`
    pub adaptation_rate: f32,

    /// Synchronization threshold for coherence.
    /// Range: `[0.0, 1.0]`
    pub sync_threshold: f32,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            min_phase: 0.0,
            max_phase: std::f32::consts::PI,
            default_phase: 0.0,
            frequency_hz: 100.0,
            coupling_strength: 0.5,
            adaptive: true,
            adaptation_rate: 0.1,
            sync_threshold: 0.8,
        }
    }
}

impl PhaseConfig {
    /// Validate the phase configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_phase < 0.0 {
            return Err(format!("min_phase must be >= 0, got {}", self.min_phase));
        }
        if self.max_phase > std::f32::consts::PI + 0.001 {
            return Err(format!("max_phase must be <= π, got {}", self.max_phase));
        }
        if self.max_phase < self.min_phase {
            return Err(format!(
                "max_phase ({}) must be >= min_phase ({})",
                self.max_phase, self.min_phase
            ));
        }
        if !(self.min_phase..=self.max_phase).contains(&self.default_phase) {
            return Err(format!(
                "default_phase must be in [{}, {}], got {}",
                self.min_phase, self.max_phase, self.default_phase
            ));
        }
        if self.frequency_hz <= 0.0 {
            return Err(format!(
                "frequency_hz must be > 0, got {}",
                self.frequency_hz
            ));
        }
        if !(0.0..=1.0).contains(&self.coupling_strength) {
            return Err(format!(
                "coupling_strength must be in [0, 1], got {}",
                self.coupling_strength
            ));
        }
        if !(0.0..=1.0).contains(&self.adaptation_rate) {
            return Err(format!(
                "adaptation_rate must be in [0, 1], got {}",
                self.adaptation_rate
            ));
        }
        if !(0.0..=1.0).contains(&self.sync_threshold) {
            return Err(format!(
                "sync_threshold must be in [0, 1], got {}",
                self.sync_threshold
            ));
        }
        Ok(())
    }

    /// Clamp a phase value to the valid range.
    pub fn clamp(&self, phase: f32) -> f32 {
        phase.clamp(self.min_phase, self.max_phase)
    }
}

/// Johari quadrant classification thresholds.
///
/// The Johari window model classifies knowledge based on surprise and coherence:
/// - **Open**: Low surprise, high coherence (well-known, directly recallable)
/// - **Blind**: High surprise, low coherence (discovery opportunity)
/// - **Hidden**: Low surprise, low coherence (private/unexplored)
/// - **Unknown**: High surprise, high coherence (frontier knowledge)
///
/// # Constitution Reference
///
/// ```text
/// Open:    ΔS < 0.5, ΔC > 0.5 → direct recall
/// Blind:   ΔS > 0.5, ΔC < 0.5 → discovery (epistemic_action/dream)
/// Hidden:  ΔS < 0.5, ΔC < 0.5 → private (get_neighborhood)
/// Unknown: ΔS > 0.5, ΔC > 0.5 → frontier
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohariConfig {
    /// Threshold for ΔS classification (surprise).
    /// Items with ΔS < threshold are "low surprise".
    pub surprise_threshold: f32,

    /// Threshold for ΔC classification (coherence).
    /// Items with ΔC > threshold are "high coherence".
    pub coherence_threshold: f32,

    /// Enable fuzzy boundaries for quadrant classification.
    pub fuzzy_boundaries: bool,

    /// Fuzzy boundary width (for smooth transitions).
    /// Range: `[0.0, 0.2]`
    pub boundary_width: f32,

    /// Weight for Open quadrant in composite scores.
    pub open_weight: f32,

    /// Weight for Blind quadrant in composite scores.
    pub blind_weight: f32,

    /// Weight for Hidden quadrant in composite scores.
    pub hidden_weight: f32,

    /// Weight for Unknown quadrant in composite scores.
    pub unknown_weight: f32,
}

impl Default for JohariConfig {
    fn default() -> Self {
        Self {
            surprise_threshold: 0.5,
            coherence_threshold: 0.5,
            fuzzy_boundaries: true,
            boundary_width: 0.1,
            open_weight: 1.0,
            blind_weight: 1.2,
            hidden_weight: 0.8,
            unknown_weight: 1.5,
        }
    }
}

impl JohariConfig {
    /// Validate the Johari configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.surprise_threshold) {
            return Err(format!(
                "surprise_threshold must be in [0, 1], got {}",
                self.surprise_threshold
            ));
        }
        if !(0.0..=1.0).contains(&self.coherence_threshold) {
            return Err(format!(
                "coherence_threshold must be in [0, 1], got {}",
                self.coherence_threshold
            ));
        }
        if !(0.0..=0.2).contains(&self.boundary_width) {
            return Err(format!(
                "boundary_width must be in [0, 0.2], got {}",
                self.boundary_width
            ));
        }
        if self.open_weight < 0.0
            || self.blind_weight < 0.0
            || self.hidden_weight < 0.0
            || self.unknown_weight < 0.0
        {
            return Err("quadrant weights must be >= 0".to_string());
        }
        Ok(())
    }
}

/// Lifecycle stage configurations.
///
/// Knowledge bases evolve through distinct lifecycle stages, each with
/// different learning dynamics as specified by the Marblestone λ weights.
///
/// # Constitution Reference
///
/// ```text
/// Infancy (n=0-50):   λ_ΔS=0.7, λ_ΔC=0.3, stance="capture-novelty"
/// Growth (n=50-500):  λ_ΔS=0.5, λ_ΔC=0.5, stance="balanced"
/// Maturity (n=500+):  λ_ΔS=0.3, λ_ΔC=0.7, stance="curation-coherence"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    /// Stage configurations in order: [Infancy, Growth, Maturity].
    pub stages: Vec<StageConfig>,

    /// Enable automatic stage transitions.
    pub auto_transition: bool,

    /// Hysteresis for stage transitions (prevents rapid switching).
    pub transition_hysteresis: u64,

    /// Enable smooth interpolation between stages.
    pub smooth_transitions: bool,

    /// Transition smoothing window (number of interactions).
    pub smoothing_window: u64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            stages: vec![
                StageConfig::infancy(),
                StageConfig::growth(),
                StageConfig::maturity(),
            ],
            auto_transition: true,
            transition_hysteresis: 10,
            smooth_transitions: true,
            smoothing_window: 25,
        }
    }
}

impl LifecycleConfig {
    /// Create lifecycle config focused on infancy stage.
    pub fn infancy_focused() -> Self {
        let mut config = Self::default();
        config.stages[0].lambda_novelty = 0.8;
        config.stages[0].lambda_consolidation = 0.2;
        config
    }

    /// Create lifecycle config focused on maturity stage.
    pub fn maturity_focused() -> Self {
        let mut config = Self::default();
        config.stages[2].lambda_novelty = 0.2;
        config.stages[2].lambda_consolidation = 0.8;
        config
    }

    /// Get stage configuration for a given interaction count.
    pub fn get_stage(&self, interaction_count: u64) -> Option<&StageConfig> {
        for stage in self.stages.iter().rev() {
            if interaction_count >= stage.min_interactions {
                return Some(stage);
            }
        }
        self.stages.first()
    }

    /// Validate the lifecycle configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.stages.is_empty() {
            return Err("At least one lifecycle stage is required".to_string());
        }
        for (i, stage) in self.stages.iter().enumerate() {
            stage
                .validate()
                .map_err(|e| format!("Stage {}: {}", i, e))?;
        }
        // Verify stages are in order
        for window in self.stages.windows(2) {
            if window[1].min_interactions <= window[0].min_interactions {
                return Err(format!(
                    "Stages must have increasing min_interactions: {} <= {}",
                    window[1].min_interactions, window[0].min_interactions
                ));
            }
        }
        Ok(())
    }
}

/// Individual lifecycle stage configuration.
///
/// Each stage defines different λ weights for balancing novelty vs. consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Stage name (e.g., "Infancy", "Growth", "Maturity").
    pub name: String,

    /// Minimum interaction count to enter this stage.
    pub min_interactions: u64,

    /// Maximum interaction count (exclusive) for this stage.
    /// Use `u64::MAX` for the final stage.
    pub max_interactions: u64,

    /// Lambda weight for novelty/surprise (λ_ΔS).
    /// Range: `[0.0, 1.0]`
    pub lambda_novelty: f32,

    /// Lambda weight for consolidation/coherence (λ_ΔC).
    /// Range: `[0.0, 1.0]`
    /// Note: lambda_novelty + lambda_consolidation should equal 1.0
    pub lambda_consolidation: f32,

    /// Surprise trigger threshold (ΔS_trig).
    /// Range: `[0.0, 1.0]`
    pub surprise_trigger: f32,

    /// Coherence trigger threshold (ΔC_trig).
    /// Range: `[0.0, 1.0]`
    pub coherence_trigger: f32,

    /// Stage stance description.
    pub stance: String,
}

impl StageConfig {
    /// Create infancy stage configuration (0-50 interactions).
    pub fn infancy() -> Self {
        Self {
            name: "Infancy".to_string(),
            min_interactions: 0,
            max_interactions: 50,
            lambda_novelty: 0.7,
            lambda_consolidation: 0.3,
            surprise_trigger: 0.9,
            coherence_trigger: 0.2,
            stance: "capture-novelty".to_string(),
        }
    }

    /// Create growth stage configuration (50-500 interactions).
    pub fn growth() -> Self {
        Self {
            name: "Growth".to_string(),
            min_interactions: 50,
            max_interactions: 500,
            lambda_novelty: 0.5,
            lambda_consolidation: 0.5,
            surprise_trigger: 0.7,
            coherence_trigger: 0.4,
            stance: "balanced".to_string(),
        }
    }

    /// Create maturity stage configuration (500+ interactions).
    pub fn maturity() -> Self {
        Self {
            name: "Maturity".to_string(),
            min_interactions: 500,
            max_interactions: u64::MAX,
            lambda_novelty: 0.3,
            lambda_consolidation: 0.7,
            surprise_trigger: 0.6,
            coherence_trigger: 0.5,
            stance: "curation-coherence".to_string(),
        }
    }

    /// Validate the stage configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Stage name cannot be empty".to_string());
        }
        if self.max_interactions <= self.min_interactions {
            return Err(format!(
                "max_interactions ({}) must be > min_interactions ({})",
                self.max_interactions, self.min_interactions
            ));
        }
        if !(0.0..=1.0).contains(&self.lambda_novelty) {
            return Err(format!(
                "lambda_novelty must be in [0, 1], got {}",
                self.lambda_novelty
            ));
        }
        if !(0.0..=1.0).contains(&self.lambda_consolidation) {
            return Err(format!(
                "lambda_consolidation must be in [0, 1], got {}",
                self.lambda_consolidation
            ));
        }
        let lambda_sum = self.lambda_novelty + self.lambda_consolidation;
        if (lambda_sum - 1.0).abs() > 0.001 {
            return Err(format!(
                "lambda_novelty + lambda_consolidation must equal 1.0, got {}",
                lambda_sum
            ));
        }
        if !(0.0..=1.0).contains(&self.surprise_trigger) {
            return Err(format!(
                "surprise_trigger must be in [0, 1], got {}",
                self.surprise_trigger
            ));
        }
        if !(0.0..=1.0).contains(&self.coherence_trigger) {
            return Err(format!(
                "coherence_trigger must be in [0, 1], got {}",
                self.coherence_trigger
            ));
        }
        Ok(())
    }
}

/// KL divergence computation settings.
///
/// Controls how KL divergence is computed for measuring information gain
/// and distribution differences in the UTL framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlConfig {
    /// Epsilon for numerical stability (prevent log(0)).
    /// Range: `[1e-10, 1e-6]`
    pub epsilon: f64,

    /// Enable symmetric KL divergence (Jensen-Shannon).
    pub symmetric: bool,

    /// Maximum KL divergence value (for clamping).
    pub max_value: f64,

    /// Smoothing factor for probability distributions.
    /// Range: `[0.0, 0.1]`
    pub smoothing: f64,

    /// Number of bins for histogram-based estimation.
    pub histogram_bins: usize,

    /// Enable adaptive binning.
    pub adaptive_binning: bool,
}

impl Default for KlConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            symmetric: false,
            max_value: 100.0,
            smoothing: 0.01,
            histogram_bins: 256,
            adaptive_binning: true,
        }
    }
}

impl KlConfig {
    /// Validate the KL configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(1e-10..=1e-6).contains(&self.epsilon) {
            return Err(format!(
                "epsilon must be in [1e-10, 1e-6], got {}",
                self.epsilon
            ));
        }
        if self.max_value <= 0.0 {
            return Err(format!("max_value must be > 0, got {}", self.max_value));
        }
        if !(0.0..=0.1).contains(&self.smoothing) {
            return Err(format!(
                "smoothing must be in [0, 0.1], got {}",
                self.smoothing
            ));
        }
        if self.histogram_bins == 0 {
            return Err("histogram_bins must be > 0".to_string());
        }
        Ok(())
    }
}

/// UTL computation thresholds.
///
/// Defines thresholds for various UTL computations and quality checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtlThresholds {
    /// Minimum valid UTL score.
    pub min_score: f32,

    /// Maximum valid UTL score.
    pub max_score: f32,

    /// Threshold for "high quality" UTL score.
    /// Constitution target: `utl_avg > 0.6`
    pub high_quality: f32,

    /// Threshold for "low quality" requiring attention.
    pub low_quality: f32,

    /// Coherence recovery timeout (seconds).
    /// Constitution target: `< 10s`
    pub coherence_recovery_secs: u64,

    /// Information loss tolerance.
    /// Constitution target: `< 15%`
    pub info_loss_tolerance: f32,

    /// Compression target ratio.
    /// Constitution target: `> 60%`
    pub compression_target: f32,

    /// Numerical tolerance for floating point comparisons.
    pub float_tolerance: f32,

    /// NaN replacement value.
    pub nan_replacement: f32,

    /// Infinity replacement value.
    pub inf_replacement: f32,
}

impl Default for UtlThresholds {
    fn default() -> Self {
        Self {
            min_score: 0.0,
            max_score: 1.0,
            high_quality: 0.6,
            low_quality: 0.3,
            coherence_recovery_secs: 10,
            info_loss_tolerance: 0.15,
            compression_target: 0.6,
            float_tolerance: 1e-6,
            nan_replacement: 0.0,
            inf_replacement: 1.0,
        }
    }
}

impl UtlThresholds {
    /// Validate the thresholds configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_score >= self.max_score {
            return Err(format!(
                "min_score ({}) must be < max_score ({})",
                self.min_score, self.max_score
            ));
        }
        if !(self.min_score..=self.max_score).contains(&self.high_quality) {
            return Err(format!(
                "high_quality must be in [{}, {}], got {}",
                self.min_score, self.max_score, self.high_quality
            ));
        }
        if !(self.min_score..=self.max_score).contains(&self.low_quality) {
            return Err(format!(
                "low_quality must be in [{}, {}], got {}",
                self.min_score, self.max_score, self.low_quality
            ));
        }
        if self.low_quality >= self.high_quality {
            return Err(format!(
                "low_quality ({}) must be < high_quality ({})",
                self.low_quality, self.high_quality
            ));
        }
        if !(0.0..=1.0).contains(&self.info_loss_tolerance) {
            return Err(format!(
                "info_loss_tolerance must be in [0, 1], got {}",
                self.info_loss_tolerance
            ));
        }
        if !(0.0..=1.0).contains(&self.compression_target) {
            return Err(format!(
                "compression_target must be in [0, 1], got {}",
                self.compression_target
            ));
        }
        if self.float_tolerance <= 0.0 {
            return Err(format!(
                "float_tolerance must be > 0, got {}",
                self.float_tolerance
            ));
        }
        Ok(())
    }

    /// Clamp a score to valid range, replacing NaN/Inf.
    pub fn clamp_score(&self, score: f32) -> f32 {
        if score.is_nan() {
            self.nan_replacement
        } else if score.is_infinite() {
            if score.is_sign_positive() {
                self.inf_replacement
            } else {
                self.min_score
            }
        } else {
            score.clamp(self.min_score, self.max_score)
        }
    }

    /// Check if a score meets the high quality threshold.
    pub fn is_high_quality(&self, score: f32) -> bool {
        score >= self.high_quality
    }

    /// Check if a score is below the low quality threshold.
    pub fn is_low_quality(&self, score: f32) -> bool {
        score <= self.low_quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let invalid = CoherenceConfig {
            neighbor_count: 0, // Invalid
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
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

        // Phase range: [0, π]
        assert_eq!(config.phase.min_phase, 0.0);
        assert!((config.phase.max_phase - std::f32::consts::PI).abs() < 0.001);

        // Johari thresholds
        assert_eq!(config.johari.surprise_threshold, 0.5);
        assert_eq!(config.johari.coherence_threshold, 0.5);

        // Lifecycle stages
        let stages = &config.lifecycle.stages;
        assert_eq!(stages.len(), 3);

        // Infancy: λ_ΔS=0.7, λ_ΔC=0.3
        assert_eq!(stages[0].lambda_novelty, 0.7);
        assert_eq!(stages[0].lambda_consolidation, 0.3);

        // Growth: λ_ΔS=0.5, λ_ΔC=0.5
        assert_eq!(stages[1].lambda_novelty, 0.5);
        assert_eq!(stages[1].lambda_consolidation, 0.5);

        // Maturity: λ_ΔS=0.3, λ_ΔC=0.7
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
