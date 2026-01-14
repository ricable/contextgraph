//! Meta-UTL Tracker for learning about learning.
//!
//! TASK-S005: Tracks per-embedder accuracy, pending predictions, and optimized weights.
//! TASK-METAUTL-P0-001: Extended with consecutive low tracking and weight clamping.
//! TASK-L01: Lambda adjustment for dream consolidation.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::debug;
use uuid::Uuid;

use context_graph_core::dream::ConsolidationMetrics;
use context_graph_core::johari::NUM_EMBEDDERS;

use super::types::{
    AdjustmentReason, Domain, DomainAccuracyTracker, LambdaAdjustmentResult, LambdaError,
    SelfCorrectionConfig, StoredPrediction,
};

/// Meta-UTL Tracker for learning about learning
///
/// TASK-S005: Tracks per-embedder accuracy, pending predictions, and optimized weights.
/// TASK-METAUTL-P0-001: Extended with consecutive low tracking and weight clamping.
/// Uses rolling window for accuracy tracking to maintain recency bias.
#[derive(Debug)]
pub struct MetaUtlTracker {
    /// Pending predictions awaiting validation
    pub pending_predictions: HashMap<Uuid, StoredPrediction>,
    /// Per-embedder accuracy rolling window (100 samples per embedder)
    pub embedder_accuracy: [[f32; 100]; NUM_EMBEDDERS],
    /// Current index in each embedder's rolling window
    pub accuracy_indices: [usize; NUM_EMBEDDERS],
    /// Number of samples in each embedder's rolling window
    pub accuracy_counts: [usize; NUM_EMBEDDERS],
    /// Current optimized weights (sum to 1.0, clamped to [0.05, 0.9] per constitution)
    pub current_weights: [f32; NUM_EMBEDDERS],
    /// Total predictions made
    pub prediction_count: usize,
    /// Total validations completed
    pub validation_count: usize,
    /// Last weight update timestamp
    pub last_weight_update: Option<Instant>,
    /// TASK-METAUTL-P0-001: Consecutive cycles with accuracy < 0.7
    pub consecutive_low_count: usize,
    /// TASK-METAUTL-P0-001: Whether Bayesian escalation has been triggered
    pub escalation_triggered: bool,
    /// TASK-METAUTL-P0-001: Self-correction configuration
    pub config: SelfCorrectionConfig,
    /// TASK-METAUTL-P0-001: Tracks which embedders have been updated in current cycle
    cycle_embedder_updated: [bool; NUM_EMBEDDERS],
    /// TASK-METAUTL-P0-001: Number of complete accuracy recording cycles
    cycle_count: usize,
    /// TASK-METAUTL-P1-001: Per-domain accuracy tracking for lambda recalibration
    #[allow(dead_code)]
    pub domain_accuracy: HashMap<Domain, DomainAccuracyTracker>,

    /// TASK-010: Lambda_s weight (semantic/structure focus)
    /// Per constitution lifecycle: infancy=0.7, adolescence=0.5, mature=adaptive
    pub lambda_s: f32,

    /// TASK-010: Lambda_c weight (contextual focus)
    /// Per constitution lifecycle: infancy=0.3, adolescence=0.5, mature=adaptive
    pub lambda_c: f32,

    /// TASK-010: Current lifecycle stage for lambda defaults
    pub lifecycle_stage: String,

    /// TASK-L01: Timestamps of recent lambda adjustments for rate limiting
    pub adjustment_times: Vec<Instant>,
}

impl Default for MetaUtlTracker {
    fn default() -> Self {
        // Initialize with uniform weights (1/13 each)
        let initial_weight = 1.0 / NUM_EMBEDDERS as f32;
        Self {
            pending_predictions: HashMap::new(),
            embedder_accuracy: [[0.0; 100]; NUM_EMBEDDERS],
            accuracy_indices: [0; NUM_EMBEDDERS],
            accuracy_counts: [0; NUM_EMBEDDERS],
            current_weights: [initial_weight; NUM_EMBEDDERS],
            prediction_count: 0,
            validation_count: 0,
            last_weight_update: None,
            // TASK-METAUTL-P0-001: Initialize consecutive tracking
            consecutive_low_count: 0,
            escalation_triggered: false,
            config: SelfCorrectionConfig::default(),
            cycle_embedder_updated: [false; NUM_EMBEDDERS],
            cycle_count: 0,
            // TASK-METAUTL-P1-001: Per-domain accuracy tracking
            domain_accuracy: HashMap::new(),
            // TASK-010: Lambda weights - default to adolescence (0.5/0.5)
            lambda_s: 0.5,
            lambda_c: 0.5,
            lifecycle_stage: "adolescence".to_string(),
            // TASK-L01: Initialize rate limiting
            adjustment_times: Vec::new(),
        }
    }
}

impl MetaUtlTracker {
    /// Weight precision tolerance for sum normalization
    #[allow(dead_code)]
    const WEIGHT_PRECISION: f32 = 1e-6;

    /// Threshold for detecting accuracy trend changes
    #[allow(dead_code)]
    const TREND_THRESHOLD: f32 = 0.02;

    // =========================================================================
    // TASK-L01: Lambda Adjustment Constants (per SPEC-DREAM-LAMBDA-001)
    // =========================================================================

    /// Minimum lambda value to prevent division by zero downstream.
    pub const LAMBDA_MIN: f32 = 0.001;

    /// Maximum lambda value.
    pub const LAMBDA_MAX: f32 = 1.0;

    /// Rate limit: max adjustments per minute.
    const MAX_ADJUSTMENTS_PER_MINUTE: u32 = 10;

    /// Threshold above which quality triggers lambda_s increase.
    const QUALITY_THRESHOLD: f32 = 0.7;

    /// Threshold above which coherence triggers lambda_c decrease.
    const COHERENCE_THRESHOLD: f32 = 0.8;

    /// Delta for lambda adjustments.
    const LAMBDA_DELTA: f32 = 0.05;

    /// Create a new MetaUtlTracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a prediction for later validation
    pub fn store_prediction(&mut self, prediction_id: Uuid, prediction: StoredPrediction) {
        self.pending_predictions.insert(prediction_id, prediction);
        self.prediction_count += 1;
    }

    /// Get a pending prediction by ID
    #[allow(dead_code)]
    pub fn get_prediction(&self, prediction_id: &Uuid) -> Option<&StoredPrediction> {
        self.pending_predictions.get(prediction_id)
    }

    /// Remove and return a prediction (for validation)
    pub fn remove_prediction(&mut self, prediction_id: &Uuid) -> Option<StoredPrediction> {
        self.pending_predictions.remove(prediction_id)
    }

    /// Record accuracy for an embedder
    ///
    /// TASK-METAUTL-P0-001: Also tracks consecutive low accuracy cycles.
    /// A "cycle" is considered complete when all 13 embedders have been updated.
    /// Low accuracy is defined as < 0.7 (constitution).
    pub fn record_accuracy(&mut self, embedder_index: usize, accuracy: f32) {
        if embedder_index >= NUM_EMBEDDERS {
            debug!(
                embedder_index = embedder_index,
                "record_accuracy: invalid embedder index, ignoring"
            );
            return;
        }

        // Clamp accuracy to [0.0, 1.0]
        let clamped_accuracy = accuracy.clamp(0.0, 1.0);

        let idx = self.accuracy_indices[embedder_index];
        self.embedder_accuracy[embedder_index][idx] = clamped_accuracy;
        self.accuracy_indices[embedder_index] = (idx + 1) % 100;
        if self.accuracy_counts[embedder_index] < 100 {
            self.accuracy_counts[embedder_index] += 1;
        }

        // TASK-METAUTL-P0-001: Track cycle completion
        self.cycle_embedder_updated[embedder_index] = true;

        // Check if a complete cycle has occurred (all embedders updated)
        if self.cycle_embedder_updated.iter().all(|&updated| updated) {
            self.cycle_count += 1;
            // Reset cycle tracking for next cycle
            self.cycle_embedder_updated = [false; NUM_EMBEDDERS];
            // Check consecutive low accuracy at end of cycle
            self.check_consecutive_low_accuracy();
        }
    }

    /// Check if overall accuracy is low and track consecutive count.
    ///
    /// TASK-METAUTL-P0-001: Called at the END of each complete cycle (when all
    /// 13 embedders have been recorded). This ensures we count cycles, not
    /// individual record_accuracy calls.
    fn check_consecutive_low_accuracy(&mut self) {
        // Calculate overall accuracy across all embedders
        let mut total_accuracy = 0.0f32;
        let mut embedder_count = 0usize;

        for i in 0..NUM_EMBEDDERS {
            if let Some(acc) = self.get_embedder_accuracy(i) {
                total_accuracy += acc;
                embedder_count += 1;
            }
        }

        // Only check if we have data from all embedders
        if embedder_count < NUM_EMBEDDERS {
            return;
        }

        let overall_accuracy = total_accuracy / embedder_count as f32;
        let threshold = self.config.low_accuracy_threshold; // 0.7

        if overall_accuracy < threshold {
            self.consecutive_low_count += 1;
            debug!(
                overall_accuracy = overall_accuracy,
                threshold = threshold,
                consecutive_low_count = self.consecutive_low_count,
                cycle_count = self.cycle_count,
                "Meta-UTL: low accuracy cycle recorded"
            );

            // Check if escalation should be triggered
            if self.consecutive_low_count >= self.config.max_consecutive_failures
                && !self.escalation_triggered
            {
                self.escalation_triggered = true;
                tracing::warn!(
                    consecutive_low = self.consecutive_low_count,
                    threshold = self.config.max_consecutive_failures,
                    "TASK-METAUTL-P0-001: Bayesian escalation triggered"
                );
            }
        } else {
            // Reset consecutive count on recovery
            if self.consecutive_low_count > 0 {
                debug!(
                    previous_count = self.consecutive_low_count,
                    overall_accuracy = overall_accuracy,
                    "Meta-UTL: accuracy recovered, resetting consecutive low count"
                );
            }
            self.consecutive_low_count = 0;
            // Note: We don't reset escalation_triggered here - that requires explicit reset
        }
    }

    /// Get average accuracy for an embedder
    pub fn get_embedder_accuracy(&self, embedder_index: usize) -> Option<f32> {
        if embedder_index >= NUM_EMBEDDERS || self.accuracy_counts[embedder_index] == 0 {
            return None;
        }
        let count = self.accuracy_counts[embedder_index];
        let sum: f32 = self.embedder_accuracy[embedder_index][..count].iter().sum();
        Some(sum / count as f32)
    }

    /// Get accuracy trend for an embedder (recent vs older samples)
    pub fn get_accuracy_trend(&self, embedder_index: usize) -> Option<&'static str> {
        if embedder_index >= NUM_EMBEDDERS || self.accuracy_counts[embedder_index] < 10 {
            return None;
        }
        let count = self.accuracy_counts[embedder_index];
        let recent_start = count.saturating_sub(10);
        let recent_sum: f32 = self.embedder_accuracy[embedder_index][recent_start..count]
            .iter()
            .sum();
        let recent_avg = recent_sum / 10.0;

        let older_end = if count >= 20 {
            count - 10
        } else {
            count - (count / 2)
        };
        let older_start = older_end.saturating_sub(10);
        let older_sum: f32 = self.embedder_accuracy[embedder_index][older_start..older_end]
            .iter()
            .sum();
        let older_count = older_end - older_start;
        if older_count == 0 {
            return Some("stable");
        }
        let older_avg = older_sum / older_count as f32;

        if recent_avg > older_avg + Self::TREND_THRESHOLD {
            Some("improving")
        } else if recent_avg < older_avg - Self::TREND_THRESHOLD {
            Some("declining")
        } else {
            Some("stable")
        }
    }

    /// Redistribute surplus from over-max weights to below-max weights.
    /// Returns the total surplus that was redistributed.
    fn redistribute_excess_weight(&mut self, max_weight: f32) -> f32 {
        let mut total_surplus = 0.0f32;
        let mut capped_count = 0usize;

        // Find weights above max
        for &weight in self.current_weights.iter() {
            if weight > max_weight {
                total_surplus += weight - max_weight;
                capped_count += 1;
            }
        }

        if total_surplus < Self::WEIGHT_PRECISION {
            return 0.0; // No surplus to redistribute
        }

        // Count weights below max that can receive redistribution
        let below_max_count = NUM_EMBEDDERS - capped_count;
        if below_max_count == 0 {
            // All weights at or above max - just cap them all
            for weight in self.current_weights.iter_mut() {
                if *weight > max_weight {
                    debug!(
                        original_weight = *weight,
                        clamped_weight = max_weight,
                        "TASK-METAUTL-P0-001: Lambda weight clamped to maximum"
                    );
                    *weight = max_weight;
                }
            }
            return total_surplus;
        }

        // Calculate how much each below-max weight should receive
        let redistribution = total_surplus / below_max_count as f32;

        // Apply capping and redistribution
        for weight in self.current_weights.iter_mut() {
            if *weight > max_weight {
                debug!(
                    original_weight = *weight,
                    clamped_weight = max_weight,
                    "TASK-METAUTL-P0-001: Lambda weight clamped to maximum"
                );
                *weight = max_weight;
            } else {
                *weight += redistribution;
            }
        }

        total_surplus
    }

    /// Update weights based on accuracy (called every 100 validations)
    ///
    /// TASK-METAUTL-P0-001: REQ-METAUTL-006/007 compliance.
    ///
    /// Priority of constraints:
    /// 1. Sum = 1.0 (REQ-METAUTL-006, HARD)
    /// 2. Max weight ≤ 0.9 (HARD - prevents single embedder dominance)
    /// 3. Min weight ≥ 0.05 (SOFT - best effort, may be violated in extreme cases)
    ///
    /// Algorithm:
    /// 1. Normalize weights based on accuracy (sum=1.0)
    /// 2. Cap any weight above max, redistribute surplus proportionally
    /// 3. Final normalization to ensure exact sum=1.0
    pub fn update_weights(&mut self) {
        // Calculate average accuracy per embedder
        let mut accuracies = [0.0f32; NUM_EMBEDDERS];
        let mut total_accuracy = 0.0f32;

        for (i, acc) in accuracies.iter_mut().enumerate() {
            *acc = self
                .get_embedder_accuracy(i)
                .unwrap_or(1.0 / NUM_EMBEDDERS as f32);
            total_accuracy += *acc;
        }

        // Normalize to get initial weights (sum = 1.0)
        if total_accuracy > 0.0 {
            for (weight, &acc) in self.current_weights.iter_mut().zip(accuracies.iter()) {
                *weight = acc / total_accuracy;
            }
        }

        let max_weight = self.config.max_weight; // 0.9
        let mut clamping_occurred = false;

        // STEP 1: Cap weights above max and redistribute surplus
        // This is the HARD constraint for max weight
        // Loop until no more surplus needs redistribution
        loop {
            let surplus = self.redistribute_excess_weight(max_weight);
            if surplus > 0.0 {
                clamping_occurred = true;
            }
            if surplus < Self::WEIGHT_PRECISION {
                break;
            }
        }

        // STEP 2: Final normalization to ensure exact sum=1.0
        let weight_sum: f32 = self.current_weights.iter().sum();
        if (weight_sum - 1.0).abs() > Self::WEIGHT_PRECISION {
            let scale = 1.0 / weight_sum;
            for weight in self.current_weights.iter_mut() {
                *weight *= scale;
            }
        }

        // Note: min_weight is a SOFT constraint. In extreme distributions
        // where one embedder dominates, other weights may be below min_weight
        // to maintain sum=1.0. This is mathematically necessary.
        // See EC-001 test for documentation.

        self.last_weight_update = Some(Instant::now());

        tracing::info!(
            validation_count = self.validation_count,
            weights_sum = self.current_weights.iter().sum::<f32>(),
            clamping_occurred = clamping_occurred,
            "Meta-UTL weights updated"
        );
    }

    /// Increment validation count and check if weights need update
    pub fn record_validation(&mut self) {
        self.validation_count += 1;
        if self.validation_count.is_multiple_of(100) {
            self.update_weights();
        }
    }

    /// Check if Bayesian escalation is needed.
    ///
    /// TASK-METAUTL-P0-001: Returns true when accuracy has been below 0.7
    /// for 100 or more consecutive cycles (PRD: minimum_observations for statistical significance).
    #[allow(dead_code)] // API reserved for TASK-METAUTL-P0-005 integration
    pub fn needs_escalation(&self) -> bool {
        self.escalation_triggered
    }

    /// Get the current consecutive low accuracy count.
    ///
    /// TASK-METAUTL-P0-001: Returns the number of consecutive cycles
    /// with overall accuracy below 0.7.
    #[allow(dead_code)] // API reserved for TASK-METAUTL-P0-005 integration
    pub fn consecutive_low_count(&self) -> usize {
        self.consecutive_low_count
    }

    /// Reset the consecutive low accuracy counter and escalation flag.
    ///
    /// TASK-METAUTL-P0-001: Call this after taking corrective action
    /// (e.g., after Bayesian optimization completes).
    #[allow(dead_code)] // API reserved for TASK-METAUTL-P0-005 integration
    pub fn reset_consecutive_low(&mut self) {
        if self.consecutive_low_count > 0 || self.escalation_triggered {
            debug!(
                previous_count = self.consecutive_low_count,
                was_escalated = self.escalation_triggered,
                "TASK-METAUTL-P0-001: Resetting consecutive low tracking"
            );
        }
        self.consecutive_low_count = 0;
        self.escalation_triggered = false;
    }

    /// Get the self-correction configuration.
    ///
    /// TASK-METAUTL-P0-001: Provides access to constitution-defined parameters.
    #[allow(dead_code)] // API reserved for TASK-METAUTL-P0-005 integration
    pub fn config(&self) -> &SelfCorrectionConfig {
        &self.config
    }

    // ========== TASK-METAUTL-P1-001: Per-Domain Accuracy Tracking ==========

    /// Record accuracy for a specific domain.
    /// TASK-METAUTL-P1-001: Enables domain-specific lambda optimization.
    #[allow(dead_code)]
    pub fn record_domain_accuracy(&mut self, domain: Domain, accuracy: f32) {
        let tracker = self.domain_accuracy
            .entry(domain)
            .or_default();
        tracker.record(accuracy);

        // Check for consecutive low accuracy in this domain
        if let Some(avg) = tracker.average() {
            if avg < self.config.low_accuracy_threshold {
                tracker.consecutive_low_count += 1;
            } else {
                tracker.consecutive_low_count = 0;
            }
        }

        debug!(
            domain = ?domain,
            accuracy = accuracy,
            avg = tracker.average(),
            sample_count = tracker.sample_count(),
            "Recorded domain-specific accuracy"
        );
    }

    /// Get average accuracy for a specific domain.
    /// TASK-METAUTL-P1-001: Returns None if no samples recorded for domain.
    #[allow(dead_code)]
    pub fn get_domain_accuracy(&self, domain: Domain) -> Option<f32> {
        self.domain_accuracy.get(&domain).and_then(|t| t.average())
    }

    /// Get all domain accuracies as a HashMap.
    /// TASK-METAUTL-P1-001: For introspection and MCP exposure.
    #[allow(dead_code)]
    pub fn get_all_domain_accuracies(&self) -> HashMap<Domain, f32> {
        self.domain_accuracy
            .iter()
            .filter_map(|(domain, tracker)| {
                tracker.average().map(|avg| (*domain, avg))
            })
            .collect()
    }

    /// Get domain accuracy tracker for detailed inspection.
    /// TASK-METAUTL-P1-001: For accessing consecutive_low_count and other fields.
    #[allow(dead_code)]
    pub fn get_domain_tracker(&self, domain: Domain) -> Option<&DomainAccuracyTracker> {
        self.domain_accuracy.get(&domain)
    }

    // ========== TASK-010: Lambda Reset Methods for trigger_healing ==========

    /// Reset lambda weights to lifecycle stage defaults.
    ///
    /// TASK-010: Enables trigger_healing to reset UTL to known good state.
    /// Per constitution METAUTL lifecycle:
    /// - infancy: (0.7, 0.3) - exploration-focused
    /// - adolescence: (0.5, 0.5) - balanced
    /// - mature: Cannot reset - requires full optimization
    ///
    /// # Arguments
    /// * `stage` - One of "infancy", "adolescence", or "mature"
    ///
    /// # Returns
    /// * `Ok(())` - Reset successful
    /// * `Err(String)` - Stage is "mature" (cannot reset) or unknown
    ///
    /// # Example
    /// ```ignore
    /// let mut tracker = MetaUtlTracker::new();
    /// tracker.reset_lambdas_to_stage("infancy")?; // Sets lambda_s=0.7, lambda_c=0.3
    /// ```
    pub fn reset_lambdas_to_stage(&mut self, stage: &str) -> Result<(), String> {
        let (old_lambda_s, old_lambda_c) = (self.lambda_s, self.lambda_c);
        let old_stage = self.lifecycle_stage.clone();

        match stage.to_lowercase().as_str() {
            "infancy" => {
                self.lambda_s = 0.7;
                self.lambda_c = 0.3;
                self.lifecycle_stage = "infancy".to_string();
                tracing::info!(
                    old_lambda_s = old_lambda_s,
                    old_lambda_c = old_lambda_c,
                    new_lambda_s = self.lambda_s,
                    new_lambda_c = self.lambda_c,
                    old_stage = old_stage,
                    new_stage = stage,
                    "TASK-010: Lambda weights reset to infancy defaults"
                );
                Ok(())
            }
            "adolescence" => {
                self.lambda_s = 0.5;
                self.lambda_c = 0.5;
                self.lifecycle_stage = "adolescence".to_string();
                tracing::info!(
                    old_lambda_s = old_lambda_s,
                    old_lambda_c = old_lambda_c,
                    new_lambda_s = self.lambda_s,
                    new_lambda_c = self.lambda_c,
                    old_stage = old_stage,
                    new_stage = stage,
                    "TASK-010: Lambda weights reset to adolescence defaults"
                );
                Ok(())
            }
            "mature" => {
                tracing::error!(
                    "TASK-010: Cannot reset mature stage - requires full optimization"
                );
                Err("Cannot reset lambda weights for mature stage - requires full Bayesian optimization".to_string())
            }
            unknown => {
                tracing::error!(
                    stage = unknown,
                    "TASK-010: Unknown lifecycle stage"
                );
                Err(format!("Unknown lifecycle stage: '{}'. Valid stages: infancy, adolescence, mature", unknown))
            }
        }
    }

    /// Reset accuracy tracking to cold start state.
    ///
    /// TASK-010: Clears all accuracy data, enabling fresh learning.
    /// Called during high/critical severity healing.
    ///
    /// This resets:
    /// - Per-embedder accuracy windows
    /// - Domain accuracy tracking
    /// - Consecutive low count
    /// - Escalation flag
    /// - Prediction/validation counts
    pub fn reset_accuracy(&mut self) {
        let old_prediction_count = self.prediction_count;
        let old_validation_count = self.validation_count;
        let domain_count = self.domain_accuracy.len();

        // Clear per-embedder accuracy
        self.embedder_accuracy = [[0.0; 100]; NUM_EMBEDDERS];
        self.accuracy_indices = [0; NUM_EMBEDDERS];
        self.accuracy_counts = [0; NUM_EMBEDDERS];
        self.cycle_embedder_updated = [false; NUM_EMBEDDERS];
        self.cycle_count = 0;

        // Reset counts
        self.prediction_count = 0;
        self.validation_count = 0;

        // Clear domain accuracy
        self.domain_accuracy.clear();

        // Reset consecutive low tracking
        self.consecutive_low_count = 0;
        self.escalation_triggered = false;

        // Reset weights to uniform
        let initial_weight = 1.0 / NUM_EMBEDDERS as f32;
        self.current_weights = [initial_weight; NUM_EMBEDDERS];
        self.last_weight_update = None;

        tracing::info!(
            old_prediction_count = old_prediction_count,
            old_validation_count = old_validation_count,
            domain_count_cleared = domain_count,
            "TASK-010: Accuracy tracking reset to cold start"
        );
    }

    /// Get current lambda weights.
    ///
    /// TASK-010: For MCP tool exposure and introspection.
    pub fn get_lambda_weights(&self) -> (f32, f32) {
        (self.lambda_s, self.lambda_c)
    }

    /// Get current lifecycle stage.
    ///
    /// TASK-010: For MCP tool exposure and introspection.
    pub fn get_lifecycle_stage(&self) -> &str {
        &self.lifecycle_stage
    }

    // =========================================================================
    // TASK-L01: Lambda Adjustment Methods (per SPEC-DREAM-LAMBDA-001)
    // =========================================================================

    /// Check rate limit for lambda adjustments.
    ///
    /// Returns `Ok(())` if under limit, `Err(LambdaError::RateLimitExceeded)` otherwise.
    fn check_rate_limit(&mut self) -> Result<(), LambdaError> {
        let now = Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);

        // Remove old timestamps
        self.adjustment_times.retain(|&t| t > one_minute_ago);

        // Check count
        if self.adjustment_times.len() as u32 >= Self::MAX_ADJUSTMENTS_PER_MINUTE {
            return Err(LambdaError::RateLimitExceeded {
                count: self.adjustment_times.len() as u32,
            });
        }

        // Record this adjustment
        self.adjustment_times.push(now);

        Ok(())
    }

    /// Adjust lambda weights based on dream consolidation metrics.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Consolidation metrics from completed dream cycle
    ///
    /// # Returns
    ///
    /// * `Ok(LambdaAdjustmentResult)` - Successful adjustment with before/after values
    /// * `Err(LambdaError)` - Adjustment failed
    ///
    /// # Algorithm
    ///
    /// 1. Validate metrics (no NaN/Infinity)
    /// 2. Check rate limit (max 10/minute)
    /// 3. Compute adjustment:
    ///    - If quality > 0.7: increase lambda_s by 0.05
    ///    - If coherence > 0.8: decrease lambda_c by 0.05
    ///    - Otherwise: no change
    /// 4. Clamp to [LAMBDA_MIN, LAMBDA_MAX]
    /// 5. Log before/after values
    /// 6. Return adjustment record
    ///
    /// # Thread Safety
    ///
    /// This method requires `&mut self`. Callers MUST hold exclusive access.
    /// For shared access, wrap in `Arc<Mutex<MetaUtlTracker>>`.
    ///
    /// # Constitution Compliance
    ///
    /// METAUTL-003: `"dream_triggered → lambda_adjustment"`
    pub fn adjust_lambdas(
        &mut self,
        metrics: ConsolidationMetrics,
    ) -> Result<LambdaAdjustmentResult, LambdaError> {
        // 1. Validate metrics
        metrics.validate()?;

        // 2. Check rate limit
        self.check_rate_limit()?;

        // 3. Store before values
        let lambda_s_before = self.lambda_s;
        let lambda_c_before = self.lambda_c;

        // 4. Compute adjustments based on metrics
        if metrics.quality > Self::QUALITY_THRESHOLD {
            self.lambda_s += Self::LAMBDA_DELTA;
        }
        if metrics.coherence > Self::COHERENCE_THRESHOLD {
            self.lambda_c -= Self::LAMBDA_DELTA;
        }

        // 5. Clamp to valid range
        let mut clamping_occurred = false;

        if self.lambda_s > Self::LAMBDA_MAX {
            self.lambda_s = Self::LAMBDA_MAX;
            clamping_occurred = true;
            tracing::warn!(
                value = self.lambda_s,
                max = Self::LAMBDA_MAX,
                "TASK-L01: Lambda_s clamped to maximum"
            );
        }
        if self.lambda_s < Self::LAMBDA_MIN {
            self.lambda_s = Self::LAMBDA_MIN;
            clamping_occurred = true;
            tracing::warn!(
                value = self.lambda_s,
                min = Self::LAMBDA_MIN,
                "TASK-L01: Lambda_s clamped to minimum"
            );
        }
        if self.lambda_c > Self::LAMBDA_MAX {
            self.lambda_c = Self::LAMBDA_MAX;
            clamping_occurred = true;
            tracing::warn!(
                value = self.lambda_c,
                max = Self::LAMBDA_MAX,
                "TASK-L01: Lambda_c clamped to maximum"
            );
        }
        if self.lambda_c < Self::LAMBDA_MIN {
            self.lambda_c = Self::LAMBDA_MIN;
            clamping_occurred = true;
            tracing::warn!(
                value = self.lambda_c,
                min = Self::LAMBDA_MIN,
                "TASK-L01: Lambda_c clamped to minimum"
            );
        }

        // 6. Create adjustment record
        let adjustment = LambdaAdjustmentResult::new(
            lambda_s_before,
            self.lambda_s,
            lambda_c_before,
            self.lambda_c,
            clamping_occurred,
            AdjustmentReason::DreamConsolidation,
        );

        // 7. Log adjustment
        tracing::info!(
            lambda_s_before = lambda_s_before,
            lambda_s_after = self.lambda_s,
            lambda_c_before = lambda_c_before,
            lambda_c_after = self.lambda_c,
            quality = metrics.quality,
            coherence = metrics.coherence,
            clamping = clamping_occurred,
            "METAUTL-003: Lambda weights adjusted from dream consolidation"
        );

        Ok(adjustment)
    }

    /// Reset lambdas to default values for a lifecycle stage.
    ///
    /// Returns a `LambdaAdjustmentResult` with before/after values and reason `ManualReset`.
    pub fn reset_lambdas(&mut self, stage: &str) -> LambdaAdjustmentResult {
        let lambda_s_before = self.lambda_s;
        let lambda_c_before = self.lambda_c;

        // Default values per lifecycle stage
        let (new_s, new_c) = match stage.to_lowercase().as_str() {
            "infant" | "infancy" => (0.7, 0.3),
            "adolescent" | "adolescence" => (0.5, 0.5),
            "adult" | "mature" => (0.6, 0.4),
            "elder" => (0.7, 0.3),
            _ => (0.5, 0.5), // Default
        };

        self.lambda_s = new_s;
        self.lambda_c = new_c;
        self.lifecycle_stage = stage.to_string();

        tracing::info!(
            lambda_s_before = lambda_s_before,
            lambda_s_after = self.lambda_s,
            lambda_c_before = lambda_c_before,
            lambda_c_after = self.lambda_c,
            stage = stage,
            "TASK-L01: Lambda weights reset to stage defaults"
        );

        LambdaAdjustmentResult::new(
            lambda_s_before,
            self.lambda_s,
            lambda_c_before,
            self.lambda_c,
            false,
            AdjustmentReason::ManualReset,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_accuracy_recording() {
        use super::super::types::Domain;

        let mut tracker = MetaUtlTracker::new();

        // Record some accuracies for different domains
        tracker.record_domain_accuracy(Domain::Code, 0.8);
        tracker.record_domain_accuracy(Domain::Code, 0.9);
        tracker.record_domain_accuracy(Domain::Medical, 0.7);

        // Verify Code domain
        let code_acc = tracker.get_domain_accuracy(Domain::Code);
        assert!(code_acc.is_some());
        let code_avg = code_acc.unwrap();
        assert!((code_avg - 0.85).abs() < 0.01, "Code accuracy should be ~0.85");

        // Verify Medical domain
        let medical_acc = tracker.get_domain_accuracy(Domain::Medical);
        assert!(medical_acc.is_some());
        assert!((medical_acc.unwrap() - 0.7).abs() < 0.01);

        // Verify untracked domain returns None
        assert!(tracker.get_domain_accuracy(Domain::Legal).is_none());
    }

    #[test]
    fn test_get_all_domain_accuracies() {
        use super::super::types::Domain;

        let mut tracker = MetaUtlTracker::new();

        tracker.record_domain_accuracy(Domain::Code, 0.8);
        tracker.record_domain_accuracy(Domain::Research, 0.6);

        let all = tracker.get_all_domain_accuracies();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key(&Domain::Code));
        assert!(all.contains_key(&Domain::Research));
    }

    #[test]
    fn test_domain_accuracy_clamping() {
        use super::super::types::Domain;

        let mut tracker = MetaUtlTracker::new();

        // Values should be clamped to [0.0, 1.0]
        tracker.record_domain_accuracy(Domain::Code, 1.5);  // Should clamp to 1.0
        tracker.record_domain_accuracy(Domain::Code, -0.5); // Should clamp to 0.0

        let avg = tracker.get_domain_accuracy(Domain::Code).unwrap();
        assert!((avg - 0.5).abs() < 0.01, "Average of clamped 1.0 and 0.0 should be 0.5");
    }

    #[test]
    fn test_domain_consecutive_low_tracking() {
        use super::super::types::Domain;

        let mut tracker = MetaUtlTracker::new();

        // Record consistently low accuracy (below default threshold of 0.7)
        for _ in 0..5 {
            tracker.record_domain_accuracy(Domain::Code, 0.5);
        }

        let domain_tracker = tracker.get_domain_tracker(Domain::Code);
        assert!(domain_tracker.is_some());
        assert!(domain_tracker.unwrap().consecutive_low_count > 0);
    }

    // =========================================================================
    // TASK-L01: Lambda Adjustment Tests
    // =========================================================================

    /// Create valid ConsolidationMetrics for testing.
    fn valid_metrics(quality: f32, coherence: f32) -> ConsolidationMetrics {
        ConsolidationMetrics {
            quality,
            coherence,
            edges_pruned: 100,
            shortcuts_created: 10,
            duration: Duration::from_secs(30),
            success: true,
            blind_spots_found: 2,
        }
    }

    #[test]
    fn test_adjust_lambdas_high_quality() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.5;
        tracker.lambda_c = 0.5;
        let metrics = valid_metrics(0.85, 0.5); // quality > 0.7, coherence < 0.8

        let adj = tracker.adjust_lambdas(metrics).unwrap();

        assert!((adj.lambda_s_before - 0.5).abs() < f32::EPSILON);
        assert!((adj.lambda_s_after - 0.55).abs() < f32::EPSILON);
        assert!((adj.lambda_c_before - 0.5).abs() < f32::EPSILON);
        assert!((adj.lambda_c_after - 0.5).abs() < f32::EPSILON); // No change
        assert!(adj.was_adjusted());
    }

    #[test]
    fn test_adjust_lambdas_high_coherence() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.5;
        tracker.lambda_c = 0.5;
        let metrics = valid_metrics(0.5, 0.9); // quality < 0.7, coherence > 0.8

        let adj = tracker.adjust_lambdas(metrics).unwrap();

        assert!((adj.lambda_s_after - 0.5).abs() < f32::EPSILON); // No change
        assert!((adj.lambda_c_after - 0.45).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adjust_lambdas_both_high() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.5;
        tracker.lambda_c = 0.5;
        let metrics = valid_metrics(0.85, 0.9); // Both above threshold

        let adj = tracker.adjust_lambdas(metrics).unwrap();

        assert!((adj.lambda_s_after - 0.55).abs() < f32::EPSILON);
        assert!((adj.lambda_c_after - 0.45).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adjust_lambdas_below_thresholds() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.5;
        tracker.lambda_c = 0.5;
        let metrics = valid_metrics(0.5, 0.5); // Both below threshold

        let adj = tracker.adjust_lambdas(metrics).unwrap();

        assert!(!adj.was_adjusted());
        assert!((adj.lambda_s_after - 0.5).abs() < f32::EPSILON);
        assert!((adj.lambda_c_after - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adjust_lambdas_clamping_max() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.98; // Near max
        tracker.lambda_c = 0.5;
        let metrics = valid_metrics(0.85, 0.5);

        let adj = tracker.adjust_lambdas(metrics).unwrap();

        assert!((adj.lambda_s_after - MetaUtlTracker::LAMBDA_MAX).abs() < f32::EPSILON); // Clamped
        assert!(adj.clamping_occurred);
    }

    #[test]
    fn test_adjust_lambdas_clamping_min() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.5;
        tracker.lambda_c = 0.02; // Near min
        let metrics = valid_metrics(0.5, 0.9);

        let adj = tracker.adjust_lambdas(metrics).unwrap();

        assert!((adj.lambda_c_after - MetaUtlTracker::LAMBDA_MIN).abs() < f32::EPSILON); // Clamped to LAMBDA_MIN
        assert!(adj.clamping_occurred);
    }

    #[test]
    fn test_adjust_lambdas_invalid_metrics_nan() {
        let mut tracker = MetaUtlTracker::new();
        let mut metrics = valid_metrics(0.5, 0.5);
        metrics.quality = f32::NAN;

        let result = tracker.adjust_lambdas(metrics);
        assert!(matches!(result, Err(LambdaError::InvalidMetrics(_))));
    }

    #[test]
    fn test_adjust_lambdas_invalid_metrics_out_of_range() {
        let mut tracker = MetaUtlTracker::new();
        let mut metrics = valid_metrics(0.5, 0.5);
        metrics.quality = 1.5;

        let result = tracker.adjust_lambdas(metrics);
        assert!(matches!(result, Err(LambdaError::InvalidMetrics(_))));
    }

    #[test]
    fn test_adjust_lambdas_rate_limit() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.5;
        tracker.lambda_c = 0.5;

        // Call MAX_ADJUSTMENTS_PER_MINUTE times successfully
        for _ in 0..MetaUtlTracker::MAX_ADJUSTMENTS_PER_MINUTE {
            let metrics = valid_metrics(0.85, 0.5);
            tracker.adjust_lambdas(metrics).unwrap();
        }

        // Next call should fail
        let metrics = valid_metrics(0.85, 0.5);
        let result = tracker.adjust_lambdas(metrics);
        assert!(matches!(result, Err(LambdaError::RateLimitExceeded { .. })));
    }

    #[test]
    fn test_reset_lambdas() {
        let mut tracker = MetaUtlTracker::new();
        tracker.lambda_s = 0.3;
        tracker.lambda_c = 0.7;

        let adj = tracker.reset_lambdas("infancy");

        // infancy defaults: lambda_s=0.7, lambda_c=0.3
        assert!((adj.lambda_s_after - 0.7).abs() < f32::EPSILON);
        assert!((adj.lambda_c_after - 0.3).abs() < f32::EPSILON);
        assert_eq!(adj.reason, AdjustmentReason::ManualReset);
    }

    #[test]
    fn test_constants() {
        assert!((MetaUtlTracker::LAMBDA_MIN - 0.001).abs() < f32::EPSILON);
        assert!((MetaUtlTracker::LAMBDA_MAX - 1.0).abs() < f32::EPSILON);
    }
}
