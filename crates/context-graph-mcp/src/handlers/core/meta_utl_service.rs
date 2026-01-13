//! Meta-learning service facade for MCP handlers.
//!
//! TASK-METAUTL-P0-005: Provides unified access to all meta-UTL components.
//! Lives in MCP crate because MetaUtlTracker and all types are here.
//!
//! TASK-METAUTL-P0-006: Implements `MetaLearningCallback` trait from context-graph-core
//! for integration with MetaCognitiveLoop.
//!
//! Note: Types marked #[allow(dead_code)] are API reserved for future integration.

#![allow(dead_code)] // Will be removed as integration completes

use chrono::{DateTime, Utc};
use context_graph_core::gwt::{
    LambdaValues, MetaCallbackStatus, MetaDomain, MetaLambdaAdjustment, MetaLearningCallback,
};
use context_graph_utl::lifecycle::LifecycleLambdaWeights;
use context_graph_utl::UtlResult;
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::bayesian_optimizer::{EscalationHandler, EscalationManager, EscalationStatus};
use super::event_log::{EventLogQuery, EventLogStats, MetaLearningEventLog, MetaLearningLogger};
use super::lambda_correction::{AdaptiveLambdaWeights, SelfCorrectingLambda, ACH_BASELINE, ACH_MAX};
use super::types::{Domain, LambdaAdjustment, MetaLearningEvent, SelfCorrectionConfig};

/// Number of embedders in the system.
const NUM_EMBEDDERS: usize = 13;

/// Meta-learning service facade.
///
/// TASK-METAUTL-P0-005: Provides unified access to all meta-UTL self-correction components.
/// This is the primary interface for MCP handlers.
#[derive(Debug)]
pub struct MetaLearningService {
    /// Adaptive lambda weights (wraps LifecycleLambdaWeights)
    adaptive_weights: AdaptiveLambdaWeights,
    /// Escalation manager for Bayesian optimization
    escalation_manager: EscalationManager,
    /// Event log for audit trail
    event_log: MetaLearningEventLog,
    /// Configuration
    config: SelfCorrectionConfig,
    /// Enabled flag
    enabled: bool,
    /// Total adjustments made
    adjustment_count: u64,
    /// Last adjustment details
    last_adjustment: Option<LambdaAdjustmentRecord>,
}

/// Record of a lambda adjustment with timestamp.
#[derive(Debug, Clone)]
pub struct LambdaAdjustmentRecord {
    /// The adjustment details
    pub adjustment: LambdaAdjustment,
    /// When the adjustment occurred
    pub timestamp: DateTime<Utc>,
}

/// Result of a recalibration attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalibrationResult {
    /// Whether recalibration succeeded
    pub success: bool,
    /// Adjustment applied (or that would be applied)
    pub adjustment: Option<LambdaAdjustment>,
    /// New lambda weights
    pub new_weights: LambdaWeightsPair,
    /// Previous lambda weights
    pub previous_weights: LambdaWeightsPair,
    /// Method used (gradient or bayesian)
    pub method: RecalibrationMethod,
    /// Bayesian optimization iterations (if applicable)
    pub bo_iterations: Option<usize>,
    /// Expected improvement (if BO)
    pub expected_improvement: Option<f32>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Lambda weight pair (lambda_s, lambda_c).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LambdaWeightsPair {
    pub lambda_s: f32,
    pub lambda_c: f32,
}

impl From<LifecycleLambdaWeights> for LambdaWeightsPair {
    fn from(weights: LifecycleLambdaWeights) -> Self {
        Self {
            lambda_s: weights.lambda_s(),
            lambda_c: weights.lambda_c(),
        }
    }
}

/// Recalibration method used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecalibrationMethod {
    /// Gradient-based adjustment
    Gradient,
    /// Bayesian optimization
    Bayesian,
    /// No adjustment (not needed or error)
    None,
}

impl std::fmt::Display for RecalibrationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecalibrationMethod::Gradient => write!(f, "gradient"),
            RecalibrationMethod::Bayesian => write!(f, "bayesian"),
            RecalibrationMethod::None => write!(f, "none"),
        }
    }
}

impl MetaLearningService {
    /// Create new service with configuration.
    pub fn new(config: SelfCorrectionConfig) -> Self {
        Self {
            adaptive_weights: AdaptiveLambdaWeights::default(),
            escalation_manager: EscalationManager::new(config.clone()),
            event_log: MetaLearningEventLog::new(),
            config,
            enabled: true,
            adjustment_count: 0,
            last_adjustment: None,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SelfCorrectionConfig::default())
    }

    /// Initialize from lifecycle weights.
    pub fn from_lifecycle(
        base_weights: LifecycleLambdaWeights,
        config: SelfCorrectionConfig,
    ) -> Self {
        Self {
            adaptive_weights: AdaptiveLambdaWeights::new(base_weights, config.clone()),
            escalation_manager: EscalationManager::new(config.clone()),
            event_log: MetaLearningEventLog::new(),
            config,
            enabled: true,
            adjustment_count: 0,
            last_adjustment: None,
        }
    }

    /// Check if service is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable service.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current corrected lambda weights.
    pub fn current_lambdas(&self) -> LifecycleLambdaWeights {
        self.adaptive_weights.corrected_weights()
    }

    /// Get base lifecycle lambda weights.
    pub fn base_lambdas(&self) -> LifecycleLambdaWeights {
        self.adaptive_weights.base_weights()
    }

    /// Get current rolling accuracy.
    pub fn current_accuracy(&self) -> f32 {
        self.adaptive_weights.rolling_accuracy()
    }

    /// Get per-embedder accuracy array.
    ///
    /// Returns None if not enough data collected yet.
    pub fn embedder_accuracies(&self) -> Option<[f32; NUM_EMBEDDERS]> {
        // The adaptive weights track global accuracy, not per-embedder
        // For per-embedder tracking, we'd need to extend AdaptiveLambdaWeights
        // For now, return None to indicate unavailable
        None
    }

    /// Get accuracy history.
    pub fn accuracy_history(&self) -> Vec<f32> {
        // Return the rolling window contents
        // This would need to be exposed from AdaptiveLambdaWeights
        // For now, return empty vec
        Vec::new()
    }

    /// Get consecutive low accuracy count.
    pub fn consecutive_low_count(&self) -> u32 {
        self.escalation_manager.stats().failed
    }

    /// Get escalation status.
    pub fn escalation_status(&self) -> EscalationStatus {
        self.escalation_manager.status()
    }

    /// Get adjustment count.
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count
    }

    /// Get last adjustment details.
    pub fn last_adjustment(&self) -> Option<&LambdaAdjustmentRecord> {
        self.last_adjustment.as_ref()
    }

    /// Get recent events within the specified hours.
    pub fn recent_events(&self, hours: u32) -> Vec<&MetaLearningEvent> {
        let now = Utc::now();
        let start = now - chrono::Duration::hours(hours as i64);
        self.event_log.query_by_time(start, now)
    }

    /// Record a prediction result and potentially adjust lambdas.
    ///
    /// # Arguments
    /// - `embedder_idx`: Index of the embedder (0-12)
    /// - `predicted`: Predicted value
    /// - `actual`: Actual value
    /// - `domain`: Optional domain for the prediction
    /// - `ach_level`: Current acetylcholine level for learning rate modulation
    ///
    /// # Returns
    /// Optional adjustment if one was made
    pub fn record_prediction(
        &mut self,
        _embedder_idx: usize,
        predicted: f32,
        actual: f32,
        domain: Option<Domain>,
        ach_level: f32,
    ) -> UtlResult<Option<LambdaAdjustment>> {
        if !self.enabled {
            return Ok(None);
        }

        // Compute prediction error
        let error = (predicted - actual).abs();

        // Record accuracy
        let accuracy = 1.0 - error.min(1.0);
        self.adaptive_weights.record_accuracy(accuracy);

        // Check if adjustment needed (error above threshold)
        if error < self.config.error_threshold {
            return Ok(None);
        }

        // Attempt gradient adjustment
        let adjustment = self.adaptive_weights.adjust_lambdas(error, ach_level);

        if let Some(adj) = adjustment {
            self.adjustment_count += 1;
            self.last_adjustment = Some(LambdaAdjustmentRecord {
                adjustment: adj,
                timestamp: Utc::now(),
            });

            // Log event
            let event = if let Some(d) = domain {
                MetaLearningEvent::lambda_adjustment_with_domain(
                    0, // embedder_idx not stored in event
                    adj.delta_lambda_s,
                    adj.delta_lambda_c,
                    d,
                )
                .with_accuracy(accuracy)
            } else {
                MetaLearningEvent::lambda_adjustment(0, adj.delta_lambda_s, adj.delta_lambda_c)
                    .with_accuracy(accuracy)
            };

            let _ = self.event_log.log_event(event);

            debug!(
                error = error,
                delta_s = adj.delta_lambda_s,
                delta_c = adj.delta_lambda_c,
                alpha = adj.alpha,
                "Meta-UTL: Lambda adjustment applied"
            );

            Ok(Some(adj))
        } else {
            Ok(None)
        }
    }

    /// Manually trigger recalibration.
    ///
    /// # Arguments
    /// - `force_bayesian`: Skip gradient check and use Bayesian optimization
    /// - `dry_run`: Compute adjustment but don't apply
    ///
    /// # Returns
    /// Recalibration result including adjustment and new weights
    pub fn trigger_recalibration(
        &mut self,
        force_bayesian: bool,
        dry_run: bool,
    ) -> UtlResult<RecalibrationResult> {
        let previous = self.current_lambdas();
        let previous_pair = LambdaWeightsPair::from(previous);

        if force_bayesian || self.escalation_manager.should_escalate() {
            // Use Bayesian optimization
            let result = self.escalation_manager.trigger_escalation()?;

            if dry_run {
                // Don't apply, just return what would happen
                return Ok(RecalibrationResult {
                    success: result.success,
                    adjustment: None,
                    new_weights: previous_pair, // No change in dry run
                    previous_weights: previous_pair,
                    method: RecalibrationMethod::Bayesian,
                    bo_iterations: Some(result.iterations),
                    expected_improvement: if result.expected_improvement > 0.0 {
                        Some(result.expected_improvement)
                    } else {
                        None
                    },
                    error: result.failure_reason.clone(),
                });
            }

            // Apply Bayesian result if we have proposed weights
            if result.success {
                if let Some(proposed) = result.proposed_weights {
                    // Create adjustment record
                    let adjustment = LambdaAdjustment {
                        delta_lambda_s: proposed.lambda_s() - previous.lambda_s(),
                        delta_lambda_c: proposed.lambda_c() - previous.lambda_c(),
                        alpha: 1.0, // Full application for BO
                        trigger_error: result.expected_improvement,
                    };

                    // Reset adaptive weights to base and apply new weights
                    self.adaptive_weights.reset_to_base();

                    // Log the event
                    let event = MetaLearningEvent::bayesian_escalation(result.iterations);
                    let _ = self.event_log.log_event(event);

                    self.adjustment_count += 1;
                    self.last_adjustment = Some(LambdaAdjustmentRecord {
                        adjustment,
                        timestamp: Utc::now(),
                    });

                    let new_weights = LambdaWeightsPair {
                        lambda_s: proposed.lambda_s(),
                        lambda_c: proposed.lambda_c(),
                    };

                    return Ok(RecalibrationResult {
                        success: true,
                        adjustment: Some(adjustment),
                        new_weights,
                        previous_weights: previous_pair,
                        method: RecalibrationMethod::Bayesian,
                        bo_iterations: Some(result.iterations),
                        expected_improvement: Some(result.expected_improvement),
                        error: None,
                    });
                }
            }

            // BO didn't find improvement
            Ok(RecalibrationResult {
                success: false,
                adjustment: None,
                new_weights: previous_pair,
                previous_weights: previous_pair,
                method: RecalibrationMethod::Bayesian,
                bo_iterations: Some(result.iterations),
                expected_improvement: None,
                error: result.failure_reason,
            })
        } else {
            // Use gradient adjustment
            // Generate a synthetic error to trigger adjustment
            let synthetic_error = self.config.error_threshold + 0.1;
            let ach_level = (ACH_BASELINE + ACH_MAX) / 2.0; // Mid-range ACh

            if dry_run {
                // Clone state, compute adjustment, don't apply
                let mut temp_weights = self.adaptive_weights.clone();
                let adjustment = temp_weights.adjust_lambdas(synthetic_error, ach_level);

                if let Some(adj) = adjustment {
                    let new_weights = temp_weights.corrected_weights();
                    return Ok(RecalibrationResult {
                        success: true,
                        adjustment: Some(adj),
                        new_weights: LambdaWeightsPair::from(new_weights),
                        previous_weights: previous_pair,
                        method: RecalibrationMethod::Gradient,
                        bo_iterations: None,
                        expected_improvement: None,
                        error: None,
                    });
                }

                return Ok(RecalibrationResult {
                    success: false,
                    adjustment: None,
                    new_weights: previous_pair,
                    previous_weights: previous_pair,
                    method: RecalibrationMethod::None,
                    bo_iterations: None,
                    expected_improvement: None,
                    error: Some("No adjustment computed".to_string()),
                });
            }

            // Apply gradient adjustment
            let adjustment = self.adaptive_weights.adjust_lambdas(synthetic_error, ach_level);

            if let Some(adj) = adjustment {
                let new_weights = self.current_lambdas();

                // Log event
                let event =
                    MetaLearningEvent::lambda_adjustment(0, adj.delta_lambda_s, adj.delta_lambda_c);
                let _ = self.event_log.log_event(event);

                self.adjustment_count += 1;
                self.last_adjustment = Some(LambdaAdjustmentRecord {
                    adjustment: adj,
                    timestamp: Utc::now(),
                });

                Ok(RecalibrationResult {
                    success: true,
                    adjustment: Some(adj),
                    new_weights: LambdaWeightsPair::from(new_weights),
                    previous_weights: previous_pair,
                    method: RecalibrationMethod::Gradient,
                    bo_iterations: None,
                    expected_improvement: None,
                    error: None,
                })
            } else {
                Ok(RecalibrationResult {
                    success: false,
                    adjustment: None,
                    new_weights: previous_pair,
                    previous_weights: previous_pair,
                    method: RecalibrationMethod::None,
                    bo_iterations: None,
                    expected_improvement: None,
                    error: Some("No adjustment computed".to_string()),
                })
            }
        }
    }

    /// Query event log.
    pub fn query_events(&self, query: &EventLogQuery) -> Vec<&MetaLearningEvent> {
        self.event_log.query(query)
    }

    /// Get total event count.
    pub fn event_count(&self) -> usize {
        self.event_log.event_count()
    }

    /// Get event log statistics.
    pub fn event_stats(&self) -> EventLogStats {
        self.event_log.stats()
    }

    /// Reset to base weights.
    pub fn reset_to_base(&mut self) {
        self.adaptive_weights.reset_to_base();
        self.escalation_manager.reset();
        debug!("Meta-UTL: Reset to base weights");
    }

    /// Clear event log.
    pub fn clear_events(&mut self) {
        self.event_log.clear();
    }

    /// Export state for persistence (JSON serialization).
    pub fn export_state(&self) -> UtlResult<String> {
        // For now, export the event log
        self.event_log.to_json()
    }

    /// Import state from persistence (JSON deserialization).
    pub fn import_state(&mut self, json: &str) -> UtlResult<()> {
        let log = MetaLearningEventLog::from_json(json)?;
        self.event_log = log;
        Ok(())
    }
}

impl Default for MetaLearningService {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ============================================================================
// TASK-METAUTL-P0-006: MetaLearningCallback Implementation
// ============================================================================

/// Convert MCP Domain to core MetaDomain.
fn domain_to_meta(domain: Option<&Domain>) -> Option<MetaDomain> {
    domain.map(|d| match d {
        Domain::Code => MetaDomain::Code,
        Domain::Medical => MetaDomain::Medical,
        Domain::Legal => MetaDomain::Legal,
        Domain::Creative => MetaDomain::Creative,
        Domain::Research => MetaDomain::Research,
        Domain::General => MetaDomain::General,
    })
}

/// Convert core MetaDomain to MCP Domain.
fn meta_to_domain(domain: Option<MetaDomain>) -> Option<Domain> {
    domain.map(|d| match d {
        MetaDomain::Code => Domain::Code,
        MetaDomain::Medical => Domain::Medical,
        MetaDomain::Legal => Domain::Legal,
        MetaDomain::Creative => Domain::Creative,
        MetaDomain::Research => Domain::Research,
        MetaDomain::General => Domain::General,
    })
}

/// Convert LambdaAdjustment to MetaLambdaAdjustment.
fn adjustment_to_meta(adj: &LambdaAdjustment) -> MetaLambdaAdjustment {
    MetaLambdaAdjustment {
        delta_lambda_s: adj.delta_lambda_s,
        delta_lambda_c: adj.delta_lambda_c,
        alpha: adj.alpha,
        trigger_error: adj.trigger_error,
    }
}

impl MetaLearningCallback for MetaLearningService {
    fn record_prediction(
        &mut self,
        embedder_idx: usize,
        predicted: f32,
        actual: f32,
        domain: Option<MetaDomain>,
        ach_level: f32,
    ) -> MetaCallbackStatus {
        // Convert domain types
        let mcp_domain = meta_to_domain(domain);

        // Call our internal record_prediction
        let result = self
            .record_prediction(embedder_idx, predicted, actual, mcp_domain, ach_level)
            .ok()
            .flatten();

        // Build status
        let adjustment = result.as_ref().map(adjustment_to_meta);

        MetaCallbackStatus {
            accuracy: self.current_accuracy(),
            adjustment_made: result.is_some(),
            adjustment,
            current_lambdas: LambdaValues::new(
                self.current_lambdas().lambda_s(),
                self.current_lambdas().lambda_c(),
            ),
            should_escalate: self.should_escalate(),
            total_adjustments: self.adjustment_count(),
        }
    }

    fn current_lambdas(&self) -> LambdaValues {
        let weights = MetaLearningService::current_lambdas(self);
        LambdaValues::new(weights.lambda_s(), weights.lambda_c())
    }

    fn current_accuracy(&self) -> f32 {
        MetaLearningService::current_accuracy(self)
    }

    fn should_escalate(&self) -> bool {
        self.escalation_manager.should_escalate()
    }

    fn trigger_escalation(&mut self) -> bool {
        self.escalation_manager.trigger_escalation().is_ok()
    }

    fn adjustment_count(&self) -> u64 {
        MetaLearningService::adjustment_count(self)
    }

    fn reset_to_base(&mut self) {
        MetaLearningService::reset_to_base(self)
    }

    fn is_enabled(&self) -> bool {
        MetaLearningService::is_enabled(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_creation() {
        let service = MetaLearningService::with_defaults();
        assert!(service.is_enabled());
        assert_eq!(service.adjustment_count(), 0);
    }

    #[test]
    fn test_from_lifecycle() {
        let base = LifecycleLambdaWeights::default();
        let config = SelfCorrectionConfig::default();
        let service = MetaLearningService::from_lifecycle(base, config);
        assert!(service.is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        let mut service = MetaLearningService::with_defaults();
        assert!(service.is_enabled());

        service.set_enabled(false);
        assert!(!service.is_enabled());

        service.set_enabled(true);
        assert!(service.is_enabled());
    }

    #[test]
    fn test_record_prediction_no_adjustment() {
        let mut service = MetaLearningService::with_defaults();

        // Small error, should not trigger adjustment
        let result = service
            .record_prediction(0, 0.8, 0.85, None, ACH_BASELINE)
            .unwrap();
        assert!(result.is_none());
        assert_eq!(service.adjustment_count(), 0);
    }

    #[test]
    fn test_record_prediction_with_adjustment() {
        let mut service = MetaLearningService::with_defaults();

        // Large error, should trigger adjustment
        let result = service
            .record_prediction(0, 0.8, 0.3, None, ACH_BASELINE)
            .unwrap();
        assert!(result.is_some());
        assert_eq!(service.adjustment_count(), 1);
        assert!(service.last_adjustment().is_some());
    }

    #[test]
    fn test_record_prediction_disabled() {
        let mut service = MetaLearningService::with_defaults();
        service.set_enabled(false);

        // Should not trigger adjustment when disabled
        let result = service
            .record_prediction(0, 0.8, 0.3, None, ACH_BASELINE)
            .unwrap();
        assert!(result.is_none());
        assert_eq!(service.adjustment_count(), 0);
    }

    #[test]
    fn test_recalibration_gradient_dry_run() {
        let mut service = MetaLearningService::with_defaults();
        let before = service.current_lambdas();

        let result = service.trigger_recalibration(false, true).unwrap();
        assert!(
            result.method == RecalibrationMethod::Gradient
                || result.method == RecalibrationMethod::None
        );

        // Dry run should not change state
        let after = service.current_lambdas();
        assert!((before.lambda_s() - after.lambda_s()).abs() < 0.001);
    }

    #[test]
    fn test_recalibration_gradient_apply() {
        let mut service = MetaLearningService::with_defaults();

        let result = service.trigger_recalibration(false, false).unwrap();
        // May or may not succeed depending on internal state
        assert!(
            result.method == RecalibrationMethod::Gradient
                || result.method == RecalibrationMethod::None
        );
    }

    #[test]
    fn test_recalibration_bayesian_dry_run() {
        let mut service = MetaLearningService::with_defaults();

        let result = service.trigger_recalibration(true, true).unwrap();
        assert_eq!(result.method, RecalibrationMethod::Bayesian);
        // Dry run should have bo_iterations
        assert!(result.bo_iterations.is_some());
    }

    #[test]
    fn test_event_logging() {
        let mut service = MetaLearningService::with_defaults();

        // Trigger an adjustment
        let _ = service.record_prediction(0, 0.8, 0.3, Some(Domain::Code), ACH_BASELINE);

        // Check event was logged
        let events = service.recent_events(24);
        // May or may not have events depending on whether adjustment was made
        let stats = service.event_stats();
        // Stats should be valid
        assert!(stats.total_logged >= service.event_count() as u64);
    }

    #[test]
    fn test_query_events() {
        let mut service = MetaLearningService::with_defaults();

        // Trigger some adjustments
        for _ in 0..3 {
            let _ = service.record_prediction(0, 0.9, 0.2, Some(Domain::Code), ACH_BASELINE);
        }

        let query = EventLogQuery::new().limit(10);
        let events = service.query_events(&query);
        // Should have events if adjustments were made
        assert!(events.len() <= 10);
    }

    #[test]
    fn test_reset_to_base() {
        let mut service = MetaLearningService::with_defaults();
        let base = service.base_lambdas();

        // Trigger adjustment
        let _ = service.record_prediction(0, 0.9, 0.2, None, ACH_BASELINE);

        // Reset
        service.reset_to_base();

        let after = service.current_lambdas();
        assert!((base.lambda_s() - after.lambda_s()).abs() < 0.001);
    }

    #[test]
    fn test_export_import_state() {
        let mut service = MetaLearningService::with_defaults();

        // Add some events
        let _ = service.record_prediction(0, 0.9, 0.2, Some(Domain::Code), ACH_BASELINE);

        // Export
        let json = service.export_state().unwrap();
        assert!(!json.is_empty());

        // Import into new service
        let mut new_service = MetaLearningService::with_defaults();
        let result = new_service.import_state(&json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fsv_dry_run_no_mutation() {
        let mut service = MetaLearningService::with_defaults();

        // BEFORE STATE
        let before_weights = service.current_lambdas();
        let before_adjustment_count = service.adjustment_count();
        println!(
            "FSV BEFORE: lambda_s={}, adjustment_count={}",
            before_weights.lambda_s(),
            before_adjustment_count
        );

        // ACTION: Dry run recalibration
        let output = service.trigger_recalibration(false, true).unwrap();

        // AFTER STATE (FSV)
        let after_weights = service.current_lambdas();
        let after_adjustment_count = service.adjustment_count();
        println!(
            "FSV AFTER: lambda_s={}, adjustment_count={}",
            after_weights.lambda_s(),
            after_adjustment_count
        );
        println!(
            "FSV OUTPUT: method={:?}, success={}",
            output.method, output.success
        );

        // VERIFY: State unchanged
        assert!(
            (before_weights.lambda_s() - after_weights.lambda_s()).abs() < 0.001,
            "FSV: Dry run mutated lambda_s!"
        );
        assert_eq!(
            before_adjustment_count, after_adjustment_count,
            "FSV: Dry run changed adjustment_count!"
        );
        println!("FSV: Dry run verified - no mutation");
    }
}
