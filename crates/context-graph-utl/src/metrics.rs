//! UTL computation metrics and status reporting.
//!
//! This module provides:
//! - `StageThresholds` - Thresholds that vary by lifecycle stage
//! - `QuadrantDistribution` - Johari quadrant classification counts
//! - `UtlComputationMetrics` - ACCUMULATED statistics across computations
//! - `UtlStatus` - Complete status for MCP responses
//!
//! NOTE: `UtlComputationMetrics` is DIFFERENT from the per-computation
//! `UtlMetrics` in context-graph-core.

use serde::{Deserialize, Serialize};

use crate::johari::JohariQuadrant;
use crate::lifecycle::{LifecycleLambdaWeights, LifecycleStage};
use crate::phase::ConsolidationPhase;

// =============================================================================
// StageThresholds
// =============================================================================

/// Thresholds that vary by lifecycle stage.
///
/// These thresholds control when learning events are triggered and
/// how memories are stored/consolidated.
///
/// # Lifecycle Stage Defaults
///
/// - **Infancy**: High entropy trigger (capture novelty), low coherence trigger
/// - **Growth**: Balanced thresholds
/// - **Maturity**: Low entropy trigger, high coherence trigger (prefer consolidation)
///
/// # Example
///
/// ```
/// use context_graph_utl::metrics::StageThresholds;
/// use context_graph_utl::lifecycle::LifecycleStage;
///
/// let thresholds = StageThresholds::for_stage(LifecycleStage::Growth);
/// assert_eq!(thresholds.entropy_trigger, 0.7);
/// assert_eq!(thresholds.coherence_trigger, 0.5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageThresholds {
    /// Entropy level that triggers learning [0.0, 1.0]
    pub entropy_trigger: f32,

    /// Coherence level that triggers consolidation [0.0, 1.0]
    pub coherence_trigger: f32,

    /// Minimum importance score to store a memory [0.0, 1.0]
    pub min_importance_store: f32,

    /// Threshold for triggering memory consolidation [0.0, 1.0]
    pub consolidation_threshold: f32,
}

impl Default for StageThresholds {
    /// Returns Growth stage thresholds as default (balanced learning).
    fn default() -> Self {
        Self {
            entropy_trigger: 0.7,
            coherence_trigger: 0.5,
            min_importance_store: 0.3,
            consolidation_threshold: 0.5,
        }
    }
}

impl StageThresholds {
    /// Create thresholds for Infancy stage (novelty-focused).
    pub fn infancy() -> Self {
        Self {
            entropy_trigger: 0.9,
            coherence_trigger: 0.2,
            min_importance_store: 0.1,
            consolidation_threshold: 0.3,
        }
    }

    /// Create thresholds for Growth stage (balanced).
    pub fn growth() -> Self {
        Self::default()
    }

    /// Create thresholds for Maturity stage (coherence-focused).
    pub fn maturity() -> Self {
        Self {
            entropy_trigger: 0.5,
            coherence_trigger: 0.7,
            min_importance_store: 0.5,
            consolidation_threshold: 0.7,
        }
    }

    /// Create thresholds for a specific lifecycle stage.
    pub fn for_stage(stage: LifecycleStage) -> Self {
        match stage {
            LifecycleStage::Infancy => Self::infancy(),
            LifecycleStage::Growth => Self::growth(),
            LifecycleStage::Maturity => Self::maturity(),
        }
    }

    /// Check if entropy exceeds trigger threshold.
    #[inline]
    pub fn should_trigger_learning(&self, entropy: f32) -> bool {
        entropy >= self.entropy_trigger
    }

    /// Check if coherence exceeds trigger threshold.
    #[inline]
    pub fn should_consolidate(&self, coherence: f32) -> bool {
        coherence >= self.coherence_trigger
    }

    /// Check if importance is sufficient to store.
    #[inline]
    pub fn should_store(&self, importance: f32) -> bool {
        importance >= self.min_importance_store
    }
}

// =============================================================================
// QuadrantDistribution
// =============================================================================

/// Distribution of classifications across Johari Window quadrants.
///
/// Maps to the UTL theory's Johari-ΔS×ΔC plane:
/// - Open: Low entropy, High coherence (known to self & others)
/// - Blind: High entropy, Low coherence (unknown to self, known to others)
/// - Hidden: Medium entropy, High coherence (known to self, hidden from others)
/// - Unknown: High entropy, Unknown coherence (unknown to all)
///
/// # Example
///
/// ```
/// use context_graph_utl::metrics::QuadrantDistribution;
/// use context_graph_utl::johari::JohariQuadrant;
///
/// let mut dist = QuadrantDistribution::new();
/// dist.increment(JohariQuadrant::Open);
/// dist.increment(JohariQuadrant::Open);
/// dist.increment(JohariQuadrant::Blind);
///
/// assert_eq!(dist.total(), 3);
/// assert_eq!(dist.dominant(), JohariQuadrant::Open);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct QuadrantDistribution {
    /// Count of Open quadrant classifications (low ΔS, high ΔC)
    pub open: u32,

    /// Count of Blind quadrant classifications (high ΔS, low ΔC)
    pub blind: u32,

    /// Count of Hidden quadrant classifications (medium ΔS, high ΔC)
    pub hidden: u32,

    /// Count of Unknown quadrant classifications (high ΔS, unknown ΔC)
    pub unknown: u32,
}

impl QuadrantDistribution {
    /// Create new empty distribution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total count across all quadrants.
    ///
    /// Uses checked arithmetic to prevent overflow, returning `u32::MAX` if overflow would occur.
    pub fn total(&self) -> u32 {
        self.open
            .checked_add(self.blind)
            .and_then(|sum| sum.checked_add(self.hidden))
            .and_then(|sum| sum.checked_add(self.unknown))
            .unwrap_or(u32::MAX)
    }

    /// Get percentages for each quadrant.
    ///
    /// Returns `[open_pct, blind_pct, hidden_pct, unknown_pct]`.
    /// Returns uniform `[0.25, 0.25, 0.25, 0.25]` when empty.
    ///
    /// # Invariant
    /// Sum of returned values equals 1.0 (within floating point tolerance).
    pub fn percentages(&self) -> [f32; 4] {
        let total = self.total();
        if total == 0 {
            return [0.25, 0.25, 0.25, 0.25];
        }

        let total_f = total as f32;
        [
            self.open as f32 / total_f,
            self.blind as f32 / total_f,
            self.hidden as f32 / total_f,
            self.unknown as f32 / total_f,
        ]
    }

    /// Get the dominant (most frequent) quadrant.
    ///
    /// Returns `JohariQuadrant::Open` on tie or when empty.
    pub fn dominant(&self) -> JohariQuadrant {
        // Use explicit comparison to ensure Open wins ties (first in order)
        let max_count = self.open.max(self.blind).max(self.hidden).max(self.unknown);

        if self.open == max_count {
            JohariQuadrant::Open
        } else if self.blind == max_count {
            JohariQuadrant::Blind
        } else if self.hidden == max_count {
            JohariQuadrant::Hidden
        } else {
            JohariQuadrant::Unknown
        }
    }

    /// Increment count for a specific quadrant.
    ///
    /// Saturates at `u32::MAX` to prevent overflow.
    pub fn increment(&mut self, quadrant: JohariQuadrant) {
        match quadrant {
            JohariQuadrant::Open => self.open = self.open.saturating_add(1),
            JohariQuadrant::Blind => self.blind = self.blind.saturating_add(1),
            JohariQuadrant::Hidden => self.hidden = self.hidden.saturating_add(1),
            JohariQuadrant::Unknown => self.unknown = self.unknown.saturating_add(1),
        }
    }

    /// Get count for a specific quadrant.
    pub fn count(&self, quadrant: JohariQuadrant) -> u32 {
        match quadrant {
            JohariQuadrant::Open => self.open,
            JohariQuadrant::Blind => self.blind,
            JohariQuadrant::Hidden => self.hidden,
            JohariQuadrant::Unknown => self.unknown,
        }
    }

    /// Reset all counts to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// =============================================================================
// UtlComputationMetrics
// =============================================================================

/// Accumulated statistics from UTL computations.
///
/// **DISTINCT FROM `UtlMetrics`**: This struct tracks AGGREGATE statistics
/// across multiple computations. `UtlMetrics` (in context-graph-core) captures
/// the values for a SINGLE UTL computation.
///
/// # Example
///
/// ```
/// use context_graph_utl::metrics::UtlComputationMetrics;
/// use context_graph_utl::johari::JohariQuadrant;
///
/// let mut metrics = UtlComputationMetrics::new();
/// assert_eq!(metrics.computation_count, 0);
/// assert!(metrics.is_healthy());
///
/// metrics.record_computation(0.7, 0.5, 0.6, JohariQuadrant::Open, 1000.0);
/// assert_eq!(metrics.computation_count, 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlComputationMetrics {
    /// Total number of UTL computations performed
    pub computation_count: u64,

    /// Running average of learning magnitude L [0.0, 1.0]
    pub avg_learning_magnitude: f32,

    /// Running average of surprise (delta_s) [0.0, 1.0]
    pub avg_delta_s: f32,

    /// Running average of coherence change (delta_c) [0.0, 1.0]
    pub avg_delta_c: f32,

    /// Distribution of Johari quadrant classifications
    pub quadrant_distribution: QuadrantDistribution,

    /// Current lifecycle stage (Infancy, Growth, Maturity)
    pub lifecycle_stage: LifecycleStage,

    /// Current Marblestone lambda weights
    pub lambda_weights: LifecycleLambdaWeights,

    /// Average computation latency in microseconds
    pub avg_latency_us: f64,

    /// 99th percentile latency in microseconds
    pub p99_latency_us: u64,
}

impl Default for UtlComputationMetrics {
    fn default() -> Self {
        Self {
            computation_count: 0,
            avg_learning_magnitude: 0.0,
            avg_delta_s: 0.0,
            avg_delta_c: 0.0,
            quadrant_distribution: QuadrantDistribution::default(),
            lifecycle_stage: LifecycleStage::default(),
            lambda_weights: LifecycleLambdaWeights::default(),
            avg_latency_us: 0.0,
            p99_latency_us: 0,
        }
    }
}

impl UtlComputationMetrics {
    /// Create new empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all metrics to initial state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get the dominant quadrant (most frequent classification).
    pub fn dominant_quadrant(&self) -> JohariQuadrant {
        self.quadrant_distribution.dominant()
    }

    /// Calculate learning efficiency (magnitude per microsecond × 1000).
    ///
    /// Returns 0.0 if no latency data or if latency is zero.
    pub fn learning_efficiency(&self) -> f64 {
        if self.avg_latency_us > 0.0 && !self.avg_latency_us.is_nan() {
            (self.avg_learning_magnitude as f64) / self.avg_latency_us * 1000.0
        } else {
            0.0
        }
    }

    /// Check if metrics indicate healthy operation.
    ///
    /// Healthy when:
    /// - Average latency < 10ms (10,000 μs)
    /// - P99 latency < 50ms (50,000 μs)
    /// - Learning magnitude is a valid number
    pub fn is_healthy(&self) -> bool {
        self.avg_latency_us < 10_000.0
            && self.p99_latency_us < 50_000
            && !self.avg_learning_magnitude.is_nan()
            && !self.avg_learning_magnitude.is_infinite()
    }

    /// Update running averages with a new computation result.
    ///
    /// Uses exponential moving average with alpha = 0.1 for smooth updates.
    pub fn record_computation(
        &mut self,
        learning_magnitude: f32,
        delta_s: f32,
        delta_c: f32,
        quadrant: JohariQuadrant,
        latency_us: f64,
    ) {
        const ALPHA: f32 = 0.1;
        const ALPHA_F64: f64 = 0.1;

        self.computation_count = self.computation_count.saturating_add(1);

        // Exponential moving average for smooth updates
        if self.computation_count == 1 {
            self.avg_learning_magnitude = learning_magnitude;
            self.avg_delta_s = delta_s;
            self.avg_delta_c = delta_c;
            self.avg_latency_us = latency_us;
        } else {
            self.avg_learning_magnitude =
                ALPHA * learning_magnitude + (1.0 - ALPHA) * self.avg_learning_magnitude;
            self.avg_delta_s = ALPHA * delta_s + (1.0 - ALPHA) * self.avg_delta_s;
            self.avg_delta_c = ALPHA * delta_c + (1.0 - ALPHA) * self.avg_delta_c;
            self.avg_latency_us =
                ALPHA_F64 * latency_us + (1.0 - ALPHA_F64) * self.avg_latency_us;
        }

        self.quadrant_distribution.increment(quadrant);

        // Update p99 (simplified: track max as approximation)
        let latency_u64 = latency_us as u64;
        if latency_u64 > self.p99_latency_us {
            self.p99_latency_us = latency_u64;
        }
    }
}

// =============================================================================
// UtlStatus
// =============================================================================

/// Complete UTL system status for monitoring and MCP responses.
///
/// Used by the `utl_status` MCP tool and `UtlProcessor::get_status()`.
///
/// # Example
///
/// ```
/// use context_graph_utl::metrics::UtlStatus;
///
/// let status = UtlStatus::new();
/// assert_eq!(status.interaction_count, 0);
/// assert!(status.is_novelty_seeking()); // Default is Infancy stage
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlStatus {
    /// Current lifecycle stage (Infancy, Growth, Maturity)
    pub lifecycle_stage: LifecycleStage,

    /// Total interaction count (determines lifecycle stage)
    pub interaction_count: u64,

    /// Current thresholds for the lifecycle stage
    pub current_thresholds: StageThresholds,

    /// Current Marblestone lambda weights
    pub lambda_weights: LifecycleLambdaWeights,

    /// Current phase oscillator angle [0, π]
    pub phase_angle: f32,

    /// Current consolidation phase (NREM, REM, Wake)
    pub consolidation_phase: ConsolidationPhase,

    /// Accumulated computation metrics
    pub metrics: UtlComputationMetrics,
}

impl Default for UtlStatus {
    fn default() -> Self {
        Self {
            lifecycle_stage: LifecycleStage::default(),
            interaction_count: 0,
            current_thresholds: StageThresholds::default(),
            lambda_weights: LifecycleLambdaWeights::default(),
            phase_angle: 0.0,
            consolidation_phase: ConsolidationPhase::default(),
            metrics: UtlComputationMetrics::default(),
        }
    }
}

impl UtlStatus {
    /// Create new default status.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if system is in encoding phase (NREM).
    pub fn is_encoding(&self) -> bool {
        matches!(self.consolidation_phase, ConsolidationPhase::NREM)
    }

    /// Check if system is in consolidation/integration phase (REM).
    pub fn is_consolidating(&self) -> bool {
        matches!(self.consolidation_phase, ConsolidationPhase::REM)
    }

    /// Check if system is in active wake phase.
    pub fn is_wake(&self) -> bool {
        matches!(self.consolidation_phase, ConsolidationPhase::Wake)
    }

    /// Check if system favors novelty (Infancy stage).
    pub fn is_novelty_seeking(&self) -> bool {
        matches!(self.lifecycle_stage, LifecycleStage::Infancy)
    }

    /// Check if system favors consolidation (Maturity stage).
    pub fn is_consolidation_focused(&self) -> bool {
        matches!(self.lifecycle_stage, LifecycleStage::Maturity)
    }

    /// Check if system is in balanced Growth stage.
    pub fn is_balanced(&self) -> bool {
        matches!(self.lifecycle_stage, LifecycleStage::Growth)
    }

    /// Get a summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "UTL: stage={:?}, interactions={}, phase={:?}, avg_L={:.3}",
            self.lifecycle_stage,
            self.interaction_count,
            self.consolidation_phase,
            self.metrics.avg_learning_magnitude
        )
    }

    /// Convert to MCP response format.
    pub fn to_mcp_response(&self) -> UtlStatusResponse {
        UtlStatusResponse {
            lifecycle_phase: format!("{:?}", self.lifecycle_stage),
            interaction_count: self.interaction_count,
            entropy: self.metrics.avg_delta_s,
            coherence: self.metrics.avg_delta_c,
            learning_score: self.metrics.avg_learning_magnitude,
            johari_quadrant: format!("{:?}", self.metrics.dominant_quadrant()),
            consolidation_phase: format!("{:?}", self.consolidation_phase),
            phase_angle: self.phase_angle,
            thresholds: ThresholdsResponse::from(&self.current_thresholds),
        }
    }
}

// =============================================================================
// MCP Response Types
// =============================================================================

/// MCP response format for utl_status tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlStatusResponse {
    /// Current lifecycle phase name
    pub lifecycle_phase: String,

    /// Total interaction count
    pub interaction_count: u64,

    /// Average entropy (delta_s)
    pub entropy: f32,

    /// Average coherence (delta_c)
    pub coherence: f32,

    /// Average learning score
    pub learning_score: f32,

    /// Dominant Johari quadrant name
    pub johari_quadrant: String,

    /// Current consolidation phase name
    pub consolidation_phase: String,

    /// Phase oscillator angle
    pub phase_angle: f32,

    /// Current thresholds
    pub thresholds: ThresholdsResponse,
}

/// Thresholds in MCP response format.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ThresholdsResponse {
    pub entropy_trigger: f32,
    pub coherence_trigger: f32,
    pub min_importance_store: f32,
    pub consolidation_threshold: f32,
}

impl From<&StageThresholds> for ThresholdsResponse {
    fn from(thresholds: &StageThresholds) -> Self {
        Self {
            entropy_trigger: thresholds.entropy_trigger,
            coherence_trigger: thresholds.coherence_trigger,
            min_importance_store: thresholds.min_importance_store,
            consolidation_threshold: thresholds.consolidation_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // StageThresholds Tests
    // =============================================================================

    #[test]
    fn test_stage_thresholds_default() {
        let thresholds = StageThresholds::default();

        // Default is Growth stage thresholds
        assert_eq!(thresholds.entropy_trigger, 0.7);
        assert_eq!(thresholds.coherence_trigger, 0.5);
        assert_eq!(thresholds.min_importance_store, 0.3);
        assert_eq!(thresholds.consolidation_threshold, 0.5);
    }

    #[test]
    fn test_stage_thresholds_infancy() {
        let thresholds = StageThresholds::infancy();

        assert_eq!(thresholds.entropy_trigger, 0.9);
        assert_eq!(thresholds.coherence_trigger, 0.2);
        assert_eq!(thresholds.min_importance_store, 0.1);
        assert_eq!(thresholds.consolidation_threshold, 0.3);
    }

    #[test]
    fn test_stage_thresholds_growth() {
        let thresholds = StageThresholds::growth();

        // Growth is the default
        assert_eq!(thresholds, StageThresholds::default());
    }

    #[test]
    fn test_stage_thresholds_maturity() {
        let thresholds = StageThresholds::maturity();

        assert_eq!(thresholds.entropy_trigger, 0.5);
        assert_eq!(thresholds.coherence_trigger, 0.7);
        assert_eq!(thresholds.min_importance_store, 0.5);
        assert_eq!(thresholds.consolidation_threshold, 0.7);
    }

    #[test]
    fn test_stage_thresholds_for_stage() {
        assert_eq!(
            StageThresholds::for_stage(LifecycleStage::Infancy),
            StageThresholds::infancy()
        );
        assert_eq!(
            StageThresholds::for_stage(LifecycleStage::Growth),
            StageThresholds::growth()
        );
        assert_eq!(
            StageThresholds::for_stage(LifecycleStage::Maturity),
            StageThresholds::maturity()
        );
    }

    #[test]
    fn test_stage_thresholds_should_trigger_learning() {
        let thresholds = StageThresholds::growth(); // entropy_trigger = 0.7

        assert!(!thresholds.should_trigger_learning(0.5)); // Below threshold
        assert!(thresholds.should_trigger_learning(0.7)); // At threshold
        assert!(thresholds.should_trigger_learning(0.9)); // Above threshold
    }

    #[test]
    fn test_stage_thresholds_should_consolidate() {
        let thresholds = StageThresholds::growth(); // coherence_trigger = 0.5

        assert!(!thresholds.should_consolidate(0.3)); // Below threshold
        assert!(thresholds.should_consolidate(0.5)); // At threshold
        assert!(thresholds.should_consolidate(0.8)); // Above threshold
    }

    #[test]
    fn test_stage_thresholds_should_store() {
        let thresholds = StageThresholds::growth(); // min_importance_store = 0.3

        assert!(!thresholds.should_store(0.1)); // Below threshold
        assert!(thresholds.should_store(0.3)); // At threshold
        assert!(thresholds.should_store(0.7)); // Above threshold
    }

    #[test]
    fn test_stage_thresholds_serialization() {
        let thresholds = StageThresholds::maturity();

        let json = serde_json::to_string(&thresholds).expect("serialize");
        let parsed: StageThresholds = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(thresholds, parsed);
    }

    #[test]
    fn test_stage_thresholds_lifecycle_progression() {
        // Verify that entropy trigger decreases as lifecycle progresses
        // (system becomes less novelty-seeking)
        let infancy = StageThresholds::infancy();
        let growth = StageThresholds::growth();
        let maturity = StageThresholds::maturity();

        assert!(infancy.entropy_trigger > growth.entropy_trigger);
        assert!(growth.entropy_trigger > maturity.entropy_trigger);

        // Verify that coherence trigger increases as lifecycle progresses
        // (system becomes more consolidation-focused)
        assert!(infancy.coherence_trigger < growth.coherence_trigger);
        assert!(growth.coherence_trigger < maturity.coherence_trigger);
    }

    // =============================================================================
    // QuadrantDistribution Tests
    // =============================================================================

    #[test]
    fn test_quadrant_distribution_default() {
        let dist = QuadrantDistribution::default();

        assert_eq!(dist.open, 0);
        assert_eq!(dist.blind, 0);
        assert_eq!(dist.hidden, 0);
        assert_eq!(dist.unknown, 0);
        assert_eq!(dist.total(), 0);
    }

    #[test]
    fn test_quadrant_distribution_percentages_empty() {
        let dist = QuadrantDistribution::default();
        let pcts = dist.percentages();

        // Uniform distribution when empty
        assert_eq!(pcts, [0.25, 0.25, 0.25, 0.25]);

        // Sum must equal 1.0
        let sum: f32 = pcts.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quadrant_distribution_percentages_with_data() {
        let dist = QuadrantDistribution {
            open: 50,
            blind: 25,
            hidden: 15,
            unknown: 10,
        };

        let pcts = dist.percentages();

        assert!((pcts[0] - 0.50).abs() < 0.001); // open
        assert!((pcts[1] - 0.25).abs() < 0.001); // blind
        assert!((pcts[2] - 0.15).abs() < 0.001); // hidden
        assert!((pcts[3] - 0.10).abs() < 0.001); // unknown

        // Sum must equal 1.0
        let sum: f32 = pcts.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quadrant_distribution_dominant() {
        let dist = QuadrantDistribution {
            open: 10,
            blind: 50, // Most frequent
            hidden: 20,
            unknown: 15,
        };

        assert_eq!(dist.dominant(), JohariQuadrant::Blind);
    }

    #[test]
    fn test_quadrant_distribution_dominant_empty() {
        let dist = QuadrantDistribution::default();

        // Default to Open when empty
        assert_eq!(dist.dominant(), JohariQuadrant::Open);
    }

    #[test]
    fn test_quadrant_distribution_dominant_tie() {
        let dist = QuadrantDistribution {
            open: 25,
            blind: 25,
            hidden: 25,
            unknown: 25,
        };

        // On tie, max_by_key returns first max (Open)
        assert_eq!(dist.dominant(), JohariQuadrant::Open);
    }

    #[test]
    fn test_quadrant_distribution_increment() {
        let mut dist = QuadrantDistribution::default();

        dist.increment(JohariQuadrant::Open);
        dist.increment(JohariQuadrant::Open);
        dist.increment(JohariQuadrant::Blind);
        dist.increment(JohariQuadrant::Hidden);

        assert_eq!(dist.open, 2);
        assert_eq!(dist.blind, 1);
        assert_eq!(dist.hidden, 1);
        assert_eq!(dist.unknown, 0);
        assert_eq!(dist.total(), 4);
    }

    #[test]
    fn test_quadrant_distribution_increment_saturation() {
        let mut dist = QuadrantDistribution {
            open: u32::MAX,
            blind: 0,
            hidden: 0,
            unknown: 0,
        };

        dist.increment(JohariQuadrant::Open);

        // Should saturate, not overflow
        assert_eq!(dist.open, u32::MAX);
    }

    #[test]
    fn test_quadrant_distribution_count() {
        let dist = QuadrantDistribution {
            open: 10,
            blind: 20,
            hidden: 30,
            unknown: 40,
        };

        assert_eq!(dist.count(JohariQuadrant::Open), 10);
        assert_eq!(dist.count(JohariQuadrant::Blind), 20);
        assert_eq!(dist.count(JohariQuadrant::Hidden), 30);
        assert_eq!(dist.count(JohariQuadrant::Unknown), 40);
    }

    #[test]
    fn test_quadrant_distribution_total_overflow_protection() {
        let dist = QuadrantDistribution {
            open: u32::MAX,
            blind: u32::MAX,
            hidden: u32::MAX,
            unknown: u32::MAX,
        };

        // Should return MAX, not panic or wrap
        assert_eq!(dist.total(), u32::MAX);
    }

    #[test]
    fn test_quadrant_distribution_reset() {
        let mut dist = QuadrantDistribution {
            open: 100,
            blind: 200,
            hidden: 300,
            unknown: 400,
        };

        dist.reset();

        assert_eq!(dist.total(), 0);
    }

    #[test]
    fn test_quadrant_distribution_serialization() {
        let dist = QuadrantDistribution {
            open: 10,
            blind: 20,
            hidden: 30,
            unknown: 40,
        };

        let json = serde_json::to_string(&dist).expect("serialize");
        let parsed: QuadrantDistribution = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(dist, parsed);
    }

    // =============================================================================
    // UtlComputationMetrics Tests
    // =============================================================================

    #[test]
    fn test_computation_metrics_default() {
        let metrics = UtlComputationMetrics::default();

        assert_eq!(metrics.computation_count, 0);
        assert_eq!(metrics.avg_learning_magnitude, 0.0);
        assert_eq!(metrics.avg_delta_s, 0.0);
        assert_eq!(metrics.avg_delta_c, 0.0);
        assert_eq!(metrics.avg_latency_us, 0.0);
        assert_eq!(metrics.p99_latency_us, 0);
    }

    #[test]
    fn test_computation_metrics_is_healthy_default() {
        let metrics = UtlComputationMetrics::default();
        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_computation_metrics_is_healthy_with_good_latency() {
        let metrics = UtlComputationMetrics {
            avg_latency_us: 5000.0, // 5ms
            p99_latency_us: 20000,  // 20ms
            avg_learning_magnitude: 0.5,
            ..Default::default()
        };

        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_computation_metrics_unhealthy_high_avg_latency() {
        let metrics = UtlComputationMetrics {
            avg_latency_us: 15000.0, // 15ms > 10ms threshold
            p99_latency_us: 20000,
            avg_learning_magnitude: 0.5,
            ..Default::default()
        };

        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_computation_metrics_unhealthy_high_p99_latency() {
        let metrics = UtlComputationMetrics {
            avg_latency_us: 5000.0,
            p99_latency_us: 60000, // 60ms > 50ms threshold
            avg_learning_magnitude: 0.5,
            ..Default::default()
        };

        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_computation_metrics_unhealthy_nan() {
        let metrics = UtlComputationMetrics {
            avg_learning_magnitude: f32::NAN,
            ..Default::default()
        };

        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_computation_metrics_unhealthy_infinite() {
        let metrics = UtlComputationMetrics {
            avg_learning_magnitude: f32::INFINITY,
            ..Default::default()
        };

        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_computation_metrics_learning_efficiency() {
        let metrics = UtlComputationMetrics {
            avg_learning_magnitude: 0.8,
            avg_latency_us: 5000.0,
            ..Default::default()
        };

        let efficiency = metrics.learning_efficiency();
        // (0.8 / 5000.0) * 1000 = 0.16
        assert!((efficiency - 0.16).abs() < 0.001);
    }

    #[test]
    fn test_computation_metrics_learning_efficiency_zero_latency() {
        let metrics = UtlComputationMetrics {
            avg_learning_magnitude: 0.8,
            avg_latency_us: 0.0,
            ..Default::default()
        };

        assert_eq!(metrics.learning_efficiency(), 0.0);
    }

    #[test]
    fn test_computation_metrics_record_first_computation() {
        let mut metrics = UtlComputationMetrics::default();

        metrics.record_computation(0.7, 0.5, 0.6, JohariQuadrant::Open, 1000.0);

        assert_eq!(metrics.computation_count, 1);
        assert_eq!(metrics.avg_learning_magnitude, 0.7);
        assert_eq!(metrics.avg_delta_s, 0.5);
        assert_eq!(metrics.avg_delta_c, 0.6);
        assert_eq!(metrics.avg_latency_us, 1000.0);
        assert_eq!(metrics.quadrant_distribution.open, 1);
    }

    #[test]
    fn test_computation_metrics_record_multiple_computations() {
        let mut metrics = UtlComputationMetrics::default();

        // First computation sets baseline
        metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Open, 1000.0);

        // Second computation uses EMA
        metrics.record_computation(0.9, 0.7, 0.8, JohariQuadrant::Blind, 2000.0);

        assert_eq!(metrics.computation_count, 2);

        // EMA: 0.1 * new + 0.9 * old
        // avg_learning_magnitude = 0.1 * 0.9 + 0.9 * 0.5 = 0.09 + 0.45 = 0.54
        assert!((metrics.avg_learning_magnitude - 0.54).abs() < 0.01);

        assert_eq!(metrics.quadrant_distribution.open, 1);
        assert_eq!(metrics.quadrant_distribution.blind, 1);
    }

    #[test]
    fn test_computation_metrics_dominant_quadrant() {
        let mut metrics = UtlComputationMetrics::default();

        metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Blind, 1000.0);
        metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Blind, 1000.0);
        metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Open, 1000.0);

        assert_eq!(metrics.dominant_quadrant(), JohariQuadrant::Blind);
    }

    #[test]
    fn test_computation_metrics_serialization() {
        let metrics = UtlComputationMetrics {
            computation_count: 100,
            avg_learning_magnitude: 0.65,
            avg_delta_s: 0.4,
            avg_delta_c: 0.5,
            quadrant_distribution: QuadrantDistribution {
                open: 40,
                blind: 30,
                hidden: 20,
                unknown: 10,
            },
            lifecycle_stage: LifecycleStage::Growth,
            lambda_weights: LifecycleLambdaWeights::default(),
            avg_latency_us: 2500.0,
            p99_latency_us: 8000,
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let parsed: UtlComputationMetrics = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(metrics, parsed);
    }

    // =============================================================================
    // UtlStatus Tests
    // =============================================================================

    #[test]
    fn test_status_default() {
        let status = UtlStatus::default();

        assert_eq!(status.lifecycle_stage, LifecycleStage::Infancy);
        assert_eq!(status.interaction_count, 0);
        assert_eq!(status.phase_angle, 0.0);
    }

    #[test]
    fn test_status_is_encoding() {
        let mut status = UtlStatus::default();
        status.consolidation_phase = ConsolidationPhase::NREM;

        assert!(status.is_encoding());
        assert!(!status.is_consolidating());
        assert!(!status.is_wake());
    }

    #[test]
    fn test_status_is_consolidating() {
        let mut status = UtlStatus::default();
        status.consolidation_phase = ConsolidationPhase::REM;

        assert!(!status.is_encoding());
        assert!(status.is_consolidating());
        assert!(!status.is_wake());
    }

    #[test]
    fn test_status_is_wake() {
        let mut status = UtlStatus::default();
        status.consolidation_phase = ConsolidationPhase::Wake;

        assert!(!status.is_encoding());
        assert!(!status.is_consolidating());
        assert!(status.is_wake());
    }

    #[test]
    fn test_status_is_novelty_seeking() {
        let mut status = UtlStatus::default();
        status.lifecycle_stage = LifecycleStage::Infancy;

        assert!(status.is_novelty_seeking());
        assert!(!status.is_consolidation_focused());
        assert!(!status.is_balanced());
    }

    #[test]
    fn test_status_is_balanced() {
        let mut status = UtlStatus::default();
        status.lifecycle_stage = LifecycleStage::Growth;

        assert!(!status.is_novelty_seeking());
        assert!(!status.is_consolidation_focused());
        assert!(status.is_balanced());
    }

    #[test]
    fn test_status_is_consolidation_focused() {
        let mut status = UtlStatus::default();
        status.lifecycle_stage = LifecycleStage::Maturity;

        assert!(!status.is_novelty_seeking());
        assert!(status.is_consolidation_focused());
        assert!(!status.is_balanced());
    }

    #[test]
    fn test_status_summary() {
        let status = UtlStatus {
            lifecycle_stage: LifecycleStage::Growth,
            interaction_count: 150,
            consolidation_phase: ConsolidationPhase::REM,
            metrics: UtlComputationMetrics {
                avg_learning_magnitude: 0.654,
                ..Default::default()
            },
            ..Default::default()
        };

        let summary = status.summary();

        assert!(summary.contains("UTL:"));
        assert!(summary.contains("Growth"));
        assert!(summary.contains("150"));
        assert!(summary.contains("REM"));
        assert!(summary.contains("0.654"));
    }

    #[test]
    fn test_status_to_mcp_response() {
        let status = UtlStatus {
            lifecycle_stage: LifecycleStage::Growth,
            interaction_count: 100,
            phase_angle: 1.57,
            consolidation_phase: ConsolidationPhase::Wake,
            current_thresholds: StageThresholds {
                entropy_trigger: 0.7,
                coherence_trigger: 0.5,
                min_importance_store: 0.3,
                consolidation_threshold: 0.6,
            },
            metrics: UtlComputationMetrics {
                avg_learning_magnitude: 0.6,
                avg_delta_s: 0.4,
                avg_delta_c: 0.5,
                ..Default::default()
            },
            ..Default::default()
        };

        let response = status.to_mcp_response();

        assert_eq!(response.lifecycle_phase, "Growth");
        assert_eq!(response.interaction_count, 100);
        assert_eq!(response.entropy, 0.4);
        assert_eq!(response.coherence, 0.5);
        assert_eq!(response.learning_score, 0.6);
        assert_eq!(response.phase_angle, 1.57);
        assert_eq!(response.consolidation_phase, "Wake");
        assert_eq!(response.thresholds.entropy_trigger, 0.7);
    }

    #[test]
    fn test_status_serialization() {
        let status = UtlStatus {
            lifecycle_stage: LifecycleStage::Maturity,
            interaction_count: 500,
            phase_angle: 2.5,
            consolidation_phase: ConsolidationPhase::REM,
            ..Default::default()
        };

        let json = serde_json::to_string(&status).expect("serialize");
        let parsed: UtlStatus = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(status, parsed);
    }

    // =============================================================================
    // ThresholdsResponse Tests
    // =============================================================================

    #[test]
    fn test_thresholds_response_from() {
        let thresholds = StageThresholds {
            entropy_trigger: 0.9,
            coherence_trigger: 0.2,
            min_importance_store: 0.1,
            consolidation_threshold: 0.3,
        };

        let response = ThresholdsResponse::from(&thresholds);

        assert_eq!(response.entropy_trigger, 0.9);
        assert_eq!(response.coherence_trigger, 0.2);
        assert_eq!(response.min_importance_store, 0.1);
        assert_eq!(response.consolidation_threshold, 0.3);
    }

    #[test]
    fn test_thresholds_response_serialization() {
        let response = ThresholdsResponse {
            entropy_trigger: 0.8,
            coherence_trigger: 0.4,
            min_importance_store: 0.2,
            consolidation_threshold: 0.5,
        };

        let json = serde_json::to_string(&response).expect("serialize");
        let parsed: ThresholdsResponse = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(response, parsed);
    }

    // =============================================================================
    // UtlStatusResponse Tests
    // =============================================================================

    #[test]
    fn test_status_response_serialization() {
        let response = UtlStatusResponse {
            lifecycle_phase: "Growth".to_string(),
            interaction_count: 200,
            entropy: 0.45,
            coherence: 0.55,
            learning_score: 0.7,
            johari_quadrant: "Open".to_string(),
            consolidation_phase: "Wake".to_string(),
            phase_angle: 1.0,
            thresholds: ThresholdsResponse::default(),
        };

        let json = serde_json::to_string(&response).expect("serialize");
        let parsed: UtlStatusResponse = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(response, parsed);
    }
}
