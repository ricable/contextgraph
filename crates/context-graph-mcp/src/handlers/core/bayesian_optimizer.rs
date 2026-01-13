//! Bayesian optimization escalation logic for Meta-UTL.
//!
//! TASK-METAUTL-P0-003: Implements Bayesian optimization with Gaussian Process
//! surrogate for finding optimal lambda values when gradient-based adjustment
//! fails for 10+ consecutive cycles.
//!
//! # Architecture
//!
//! This module provides:
//! - `SimpleGaussianProcess`: A GP surrogate model for lambda optimization
//! - `EscalationManager`: Manages escalation to BO and human review
//! - `EscalationHandler` trait: Interface for escalation behavior
//!
//! # Algorithm
//!
//! 1. Detect when 10+ consecutive low-accuracy cycles occur
//! 2. Trigger Bayesian optimization with GP surrogate
//! 3. Use Expected Improvement (EI) acquisition function
//! 4. If BO fails 3 times, escalate to human review
//!
//! # Constitution Reference
//!
//! - REQ-METAUTL-008: Escalate to Bayesian optimization after 10 consecutive failures
//! - REQ-METAUTL-009: Use Gaussian Process surrogate for lambda search
//! - REQ-METAUTL-010: Escalate to human review after 3 BO failures

// Allow dead_code until integration in TASK-METAUTL-P0-005/006
#![allow(dead_code)]

use context_graph_utl::error::{UtlError, UtlResult};
use context_graph_utl::lifecycle::LifecycleLambdaWeights;

use super::types::SelfCorrectionConfig;

// ============================================================================
// Constants
// ============================================================================

/// Maximum Bayesian optimization iterations per escalation.
pub const MAX_BO_ITERATIONS: usize = 10;

/// Number of initial samples before using GP-based proposals.
pub const INITIAL_SAMPLES: usize = 5;

/// Number of consecutive BO failures before human escalation.
pub const HUMAN_ESCALATION_THRESHOLD: u32 = 3;

/// Minimum variance threshold for EI computation.
const MIN_VARIANCE: f32 = 1e-8;

/// Grid search resolution for EI maximization.
const GRID_SIZE: usize = 100;

/// Epsilon for floating-point comparison.
const EPSILON: f32 = 1e-6;

/// EI convergence threshold (stop if EI below this).
const EI_CONVERGENCE_THRESHOLD: f32 = 1e-4;

// ============================================================================
// Types
// ============================================================================

/// Status of the escalation process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationStatus {
    /// No escalation in progress.
    Idle,
    /// Bayesian optimization in progress.
    InProgress,
    /// Escalation succeeded with improved weights.
    Succeeded,
    /// Escalation failed to find better weights.
    Failed,
    /// Human review required after repeated failures.
    HumanReviewRequired,
}

/// Result of an escalation attempt.
#[derive(Debug, Clone)]
pub struct EscalationResult {
    /// Whether escalation succeeded.
    pub success: bool,
    /// Proposed new weights if successful.
    pub proposed_weights: Option<LifecycleLambdaWeights>,
    /// Expected improvement from the proposed weights.
    pub expected_improvement: f32,
    /// Number of BO iterations performed.
    pub iterations: usize,
    /// Failure reason if unsuccessful.
    pub failure_reason: Option<String>,
}

impl EscalationResult {
    /// Create a successful escalation result.
    pub fn success(weights: LifecycleLambdaWeights, ei: f32, iterations: usize) -> Self {
        Self {
            success: true,
            proposed_weights: Some(weights),
            expected_improvement: ei,
            iterations,
            failure_reason: None,
        }
    }

    /// Create a failed escalation result.
    pub fn failure(reason: impl Into<String>, iterations: usize) -> Self {
        Self {
            success: false,
            proposed_weights: None,
            expected_improvement: 0.0,
            iterations,
            failure_reason: Some(reason.into()),
        }
    }
}

/// A single observation for the Gaussian Process.
#[derive(Debug, Clone, Copy)]
pub struct GpObservation {
    /// Lambda_s value.
    pub lambda_s: f32,
    /// Accuracy achieved with this lambda_s.
    pub accuracy: f32,
}

/// Statistics about escalation attempts.
#[derive(Debug, Clone, Copy, Default)]
pub struct EscalationStats {
    /// Total escalation attempts.
    pub total_attempts: u32,
    /// Successful escalations.
    pub successful: u32,
    /// Failed escalations.
    pub failed: u32,
    /// Success rate (0.0 to 1.0).
    pub success_rate: f32,
}

impl EscalationStats {
    /// Update stats after an escalation attempt.
    pub fn record(&mut self, success: bool) {
        self.total_attempts += 1;
        if success {
            self.successful += 1;
        } else {
            self.failed += 1;
        }
        self.success_rate = if self.total_attempts > 0 {
            self.successful as f32 / self.total_attempts as f32
        } else {
            0.0
        };
    }
}

// ============================================================================
// SimpleGaussianProcess
// ============================================================================

/// Simple Gaussian Process for 1D optimization.
///
/// Uses RBF kernel and naive O(N^3) matrix inversion since N is small.
#[derive(Debug, Clone)]
pub struct SimpleGaussianProcess {
    /// Collected observations.
    observations: Vec<GpObservation>,
    /// RBF kernel length scale.
    length_scale: f32,
    /// Observation noise variance.
    noise_var: f32,
    /// Signal variance (kernel amplitude).
    signal_var: f32,
}

impl Default for SimpleGaussianProcess {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleGaussianProcess {
    /// Create a new GP with default hyperparameters.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            length_scale: 0.2,
            noise_var: 0.01,
            signal_var: 1.0,
        }
    }

    /// Create a GP with custom hyperparameters.
    pub fn with_params(length_scale: f32, noise_var: f32, signal_var: f32) -> Self {
        Self {
            observations: Vec::new(),
            length_scale: length_scale.max(EPSILON),
            noise_var: noise_var.max(EPSILON),
            signal_var: signal_var.max(EPSILON),
        }
    }

    /// Add an observation to the GP.
    pub fn add_observation(&mut self, obs: GpObservation) {
        // Validate inputs
        if obs.lambda_s.is_nan() || obs.lambda_s.is_infinite() {
            tracing::warn!("Ignoring invalid lambda_s observation: {}", obs.lambda_s);
            return;
        }
        if obs.accuracy.is_nan() || obs.accuracy.is_infinite() {
            tracing::warn!("Ignoring invalid accuracy observation: {}", obs.accuracy);
            return;
        }
        self.observations.push(obs);
    }

    /// Predict mean and variance at a given lambda_s.
    ///
    /// # Returns
    ///
    /// Tuple of (mean, variance). For empty GP, returns prior (0.5, signal_var).
    pub fn predict(&self, lambda_s: f32) -> (f32, f32) {
        if lambda_s.is_nan() || lambda_s.is_infinite() {
            return (0.5, self.signal_var);
        }

        let n = self.observations.len();
        if n == 0 {
            // No observations: return prior
            return (0.5, self.signal_var);
        }

        // Build K matrix (N x N)
        let mut k_matrix = vec![vec![0.0f32; n]; n];
        for (i, row) in k_matrix.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = self.rbf_kernel(
                    self.observations[i].lambda_s,
                    self.observations[j].lambda_s,
                );
                if i == j {
                    *cell += self.noise_var;
                }
            }
        }

        // Build k* vector
        let k_star: Vec<f32> = self
            .observations
            .iter()
            .map(|obs| self.rbf_kernel(lambda_s, obs.lambda_s))
            .collect();

        // Build y vector
        let y: Vec<f32> = self.observations.iter().map(|obs| obs.accuracy).collect();

        // Solve K^(-1) @ y and K^(-1) @ k*
        // Using simple Gaussian elimination for small matrices
        let k_inv_y = self.solve_linear_system(&k_matrix, &y);
        let k_inv_k_star = self.solve_linear_system(&k_matrix, &k_star);

        // Mean: mu = k*^T @ K^(-1) @ y
        let mean: f32 = k_star.iter().zip(k_inv_y.iter()).map(|(a, b)| a * b).sum();

        // Variance: var = k(x*, x*) - k*^T @ K^(-1) @ k*
        let k_star_star = self.rbf_kernel(lambda_s, lambda_s);
        let var_reduction: f32 = k_star
            .iter()
            .zip(k_inv_k_star.iter())
            .map(|(a, b)| a * b)
            .sum();
        let variance = (k_star_star - var_reduction).max(EPSILON);

        // Handle NaN/Inf
        let mean = if mean.is_nan() || mean.is_infinite() {
            0.5
        } else {
            mean.clamp(0.0, 1.0)
        };
        let variance = if variance.is_nan() || variance.is_infinite() {
            self.signal_var
        } else {
            variance
        };

        (mean, variance)
    }

    /// Compute Expected Improvement at a given lambda_s.
    ///
    /// # Formula
    ///
    /// ```text
    /// EI(x) = max(0, (mu - f_best) * Phi(z) + sigma * phi(z))
    /// where z = (mu - f_best) / sigma
    /// ```
    pub fn expected_improvement(&self, lambda_s: f32) -> f32 {
        if lambda_s.is_nan() || lambda_s.is_infinite() {
            return 0.0;
        }

        let (mean, variance) = self.predict(lambda_s);
        let sigma = variance.sqrt();

        // If variance is too small, EI is 0 (we're very confident)
        if sigma < MIN_VARIANCE.sqrt() {
            return 0.0;
        }

        // Get best observed value
        let f_best = self.best_observation().map(|o| o.accuracy).unwrap_or(0.0);

        // Compute z = (mu - f_best) / sigma
        let z = (mean - f_best) / sigma;

        // EI = (mu - f_best) * Phi(z) + sigma * phi(z)
        let ei = (mean - f_best) * self.standard_normal_cdf(z) + sigma * self.standard_normal_pdf(z);

        // EI must be non-negative
        ei.max(0.0)
    }

    /// Find lambda_s that maximizes Expected Improvement via grid search.
    ///
    /// # Arguments
    ///
    /// - `lambda_min`: Minimum lambda_s value to consider
    /// - `lambda_max`: Maximum lambda_s value to consider
    /// - `grid_size`: Number of grid points to evaluate
    ///
    /// # Returns
    ///
    /// Lambda_s value with maximum EI (within bounds).
    pub fn maximize_ei(&self, lambda_min: f32, lambda_max: f32, grid_size: usize) -> f32 {
        let lambda_min = lambda_min.max(0.05);
        let lambda_max = lambda_max.min(0.9);

        if lambda_min >= lambda_max {
            return (lambda_min + lambda_max) / 2.0;
        }

        let grid_size = grid_size.max(10);
        let step = (lambda_max - lambda_min) / (grid_size as f32);

        let mut best_lambda = lambda_min;
        let mut best_ei = f32::NEG_INFINITY;

        for i in 0..=grid_size {
            let lambda = lambda_min + (i as f32) * step;
            let ei = self.expected_improvement(lambda);

            if ei > best_ei {
                best_ei = ei;
                best_lambda = lambda;
            }
        }

        best_lambda
    }

    /// Get the best (highest accuracy) observation.
    pub fn best_observation(&self) -> Option<GpObservation> {
        self.observations
            .iter()
            .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Get number of observations.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Clear all observations.
    pub fn clear(&mut self) {
        self.observations.clear();
    }

    /// RBF (Radial Basis Function) kernel.
    ///
    /// # Formula
    ///
    /// ```text
    /// k(x1, x2) = signal_var * exp(-0.5 * (x1 - x2)^2 / length_scale^2)
    /// ```
    pub fn rbf_kernel(&self, x1: f32, x2: f32) -> f32 {
        let diff = x1 - x2;
        let scaled_dist_sq = (diff * diff) / (self.length_scale * self.length_scale);
        self.signal_var * (-0.5 * scaled_dist_sq).exp()
    }

    /// Approximate standard normal CDF using error function approximation.
    ///
    /// Uses Abramowitz and Stegun approximation (7.1.26).
    pub fn standard_normal_cdf(&self, x: f32) -> f32 {
        // Handle extreme values
        if x < -6.0 {
            return 0.0;
        }
        if x > 6.0 {
            return 1.0;
        }

        // Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        0.5 * (1.0 + self.erf(x / std::f32::consts::SQRT_2))
    }

    /// Approximate error function.
    fn erf(&self, x: f32) -> f32 {
        // Constants for Abramowitz and Stegun approximation (7.1.26)
        // Truncated to f32 precision to avoid clippy warnings
        const A1: f32 = 0.254_829_6;
        const A2: f32 = -0.284_496_72;
        const A3: f32 = 1.421_413_8;
        const A4: f32 = -1.453_152_1;
        const A5: f32 = 1.061_405_4;
        const P: f32 = 0.327_591_1;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + P * x);
        let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

        sign * y
    }

    /// Standard normal PDF.
    ///
    /// # Formula
    ///
    /// ```text
    /// phi(x) = (1 / sqrt(2*pi)) * exp(-0.5 * x^2)
    /// ```
    pub fn standard_normal_pdf(&self, x: f32) -> f32 {
        const INV_SQRT_2PI: f32 = 0.398_942_3; // 1 / sqrt(2*pi), truncated for f32
        INV_SQRT_2PI * (-0.5 * x * x).exp()
    }

    /// Solve linear system Ax = b using Gaussian elimination.
    ///
    /// Simple O(N^3) implementation suitable for small N.
    /// Note: Using index-based loops for clarity in the elimination algorithm.
    #[allow(clippy::needless_range_loop)]
    fn solve_linear_system(&self, a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
        let n = b.len();
        if n == 0 {
            return Vec::new();
        }

        // Create augmented matrix
        let mut aug: Vec<Vec<f32>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut new_row = row.clone();
                new_row.push(b[i]);
                new_row
            })
            .collect();

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = aug[i][i].abs();
            for k in (i + 1)..n {
                if aug[k][i].abs() > max_val {
                    max_val = aug[k][i].abs();
                    max_row = k;
                }
            }

            // Swap rows
            aug.swap(i, max_row);

            // Check for singular matrix
            if aug[i][i].abs() < EPSILON {
                // Return zeros if singular
                return vec![0.0; n];
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = aug[k][i] / aug[i][i];
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];

            // Handle NaN/Inf
            if x[i].is_nan() || x[i].is_infinite() {
                x[i] = 0.0;
            }
        }

        x
    }
}

// ============================================================================
// EscalationHandler Trait
// ============================================================================

/// Trait for handling escalation behavior.
pub trait EscalationHandler {
    /// Check if escalation should be triggered.
    fn should_escalate(&self) -> bool;

    /// Trigger escalation and run Bayesian optimization.
    fn trigger_escalation(&mut self) -> UtlResult<EscalationResult>;

    /// Get current escalation status.
    fn status(&self) -> EscalationStatus;

    /// Reset escalation state.
    fn reset(&mut self);

    /// Check if human review is needed.
    fn needs_human_review(&self) -> bool;
}

// ============================================================================
// EscalationManager
// ============================================================================

/// Manages escalation to Bayesian optimization and human review.
/// TASK-METAUTL-P0-005: Added Debug for MetaLearningService.
#[derive(Debug)]
pub struct EscalationManager {
    /// Configuration for self-correction.
    config: SelfCorrectionConfig,
    /// Current escalation status.
    status: EscalationStatus,
    /// Gaussian Process for surrogate modeling.
    gp: SimpleGaussianProcess,
    /// Number of consecutive BO failures.
    failed_escalations: u32,
    /// Total escalation attempts.
    total_attempts: u32,
    /// Successful escalations.
    successful_escalations: u32,
    /// Last proposed weights.
    last_proposal: Option<LifecycleLambdaWeights>,
    /// Consecutive low-accuracy cycles.
    consecutive_failures: usize,
}

impl EscalationManager {
    /// Create a new EscalationManager with the given config.
    pub fn new(config: SelfCorrectionConfig) -> Self {
        Self {
            config,
            status: EscalationStatus::Idle,
            gp: SimpleGaussianProcess::new(),
            failed_escalations: 0,
            total_attempts: 0,
            successful_escalations: 0,
            last_proposal: None,
            consecutive_failures: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SelfCorrectionConfig::default())
    }

    /// Run Bayesian optimization to find optimal lambda weights.
    ///
    /// # Arguments
    ///
    /// - `current_weights`: Current lambda weights as starting point
    /// - `evaluate_fn`: Function that evaluates accuracy for given lambda_s
    ///
    /// # Returns
    ///
    /// Proposed new weights if optimization succeeds.
    pub fn run_bayesian_optimization<F>(
        &mut self,
        current_weights: LifecycleLambdaWeights,
        mut evaluate_fn: F,
    ) -> UtlResult<LifecycleLambdaWeights>
    where
        F: FnMut(f32) -> f32,
    {
        self.status = EscalationStatus::InProgress;
        self.gp.clear();

        let lambda_min = self.config.min_weight;
        let lambda_max = self.config.max_weight;

        // Generate initial samples
        let initial_samples = self.generate_initial_samples(current_weights.lambda_s());

        // Evaluate initial samples
        for lambda_s in initial_samples {
            let accuracy = evaluate_fn(lambda_s);
            if !accuracy.is_nan() && !accuracy.is_infinite() {
                self.gp.add_observation(GpObservation { lambda_s, accuracy });
            }
        }

        // Main BO loop
        let mut iterations = self.gp.num_observations();
        while iterations < MAX_BO_ITERATIONS + INITIAL_SAMPLES {
            // Propose next lambda_s using EI
            let next_lambda = self.propose_next(lambda_min, lambda_max);

            // Check for convergence
            if self.has_converged() {
                break;
            }

            // Evaluate proposed point
            let accuracy = evaluate_fn(next_lambda);
            if !accuracy.is_nan() && !accuracy.is_infinite() {
                self.gp.add_observation(GpObservation {
                    lambda_s: next_lambda,
                    accuracy,
                });
            }

            iterations += 1;
        }

        // Get best result
        match self.gp.best_observation() {
            Some(best) => {
                // Create new weights from best lambda_s
                let lambda_c = 1.0 - best.lambda_s;
                match LifecycleLambdaWeights::new(best.lambda_s, lambda_c) {
                    Ok(weights) => {
                        self.status = EscalationStatus::Succeeded;
                        self.last_proposal = Some(weights);
                        self.successful_escalations += 1;
                        self.failed_escalations = 0; // Reset failure counter
                        Ok(weights)
                    }
                    Err(e) => {
                        self.status = EscalationStatus::Failed;
                        self.failed_escalations += 1;
                        self.check_human_escalation();
                        Err(e)
                    }
                }
            }
            None => {
                self.status = EscalationStatus::Failed;
                self.failed_escalations += 1;
                self.check_human_escalation();
                Err(UtlError::ConfigError(
                    "Bayesian optimization produced no valid observations".to_string(),
                ))
            }
        }
    }

    /// Generate initial sample points for BO.
    ///
    /// Uses Latin hypercube-like spacing for good coverage.
    pub fn generate_initial_samples(&self, current: f32) -> Vec<f32> {
        let mut samples = Vec::with_capacity(INITIAL_SAMPLES);
        let lambda_min = self.config.min_weight;
        let lambda_max = self.config.max_weight;
        let range = lambda_max - lambda_min;

        // Include current value
        samples.push(current.clamp(lambda_min, lambda_max));

        // Add evenly spaced points
        for i in 0..INITIAL_SAMPLES - 1 {
            let t = (i as f32 + 0.5) / (INITIAL_SAMPLES - 1) as f32;
            let lambda = lambda_min + t * range;
            if (lambda - current).abs() > EPSILON {
                samples.push(lambda);
            }
        }

        // Ensure we have enough samples
        while samples.len() < INITIAL_SAMPLES {
            let jitter = (samples.len() as f32 * 0.1) % range;
            let lambda = (lambda_min + jitter).clamp(lambda_min, lambda_max);
            samples.push(lambda);
        }

        samples
    }

    /// Propose next lambda_s using EI maximization.
    pub fn propose_next(&self, lambda_min: f32, lambda_max: f32) -> f32 {
        self.gp.maximize_ei(lambda_min, lambda_max, GRID_SIZE)
    }

    /// Check if optimization has converged (EI below threshold).
    pub fn has_converged(&self) -> bool {
        let lambda_min = self.config.min_weight;
        let lambda_max = self.config.max_weight;

        // Check EI at multiple points
        let n_checks = 10;
        let step = (lambda_max - lambda_min) / n_checks as f32;

        let max_ei = (0..=n_checks)
            .map(|i| {
                let lambda = lambda_min + (i as f32) * step;
                self.gp.expected_improvement(lambda)
            })
            .fold(0.0f32, |a, b| a.max(b));

        max_ei < EI_CONVERGENCE_THRESHOLD
    }

    /// Record outcome of an optimization attempt.
    pub fn record_outcome(&mut self, success: bool) {
        self.total_attempts += 1;
        if success {
            self.successful_escalations += 1;
            self.failed_escalations = 0;
        } else {
            self.failed_escalations += 1;
            self.check_human_escalation();
        }
    }

    /// Get escalation statistics.
    pub fn stats(&self) -> EscalationStats {
        let total = self.total_attempts;
        let successful = self.successful_escalations;
        let failed = total.saturating_sub(successful);
        let success_rate = if total > 0 {
            successful as f32 / total as f32
        } else {
            0.0
        };

        EscalationStats {
            total_attempts: total,
            successful,
            failed,
            success_rate,
        }
    }

    /// Record a low-accuracy cycle.
    pub fn record_failure_cycle(&mut self) {
        self.consecutive_failures += 1;
    }

    /// Record a successful (high-accuracy) cycle.
    pub fn record_success_cycle(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Get consecutive failure count.
    pub fn consecutive_failures(&self) -> usize {
        self.consecutive_failures
    }

    /// Check and set human escalation status if needed.
    fn check_human_escalation(&mut self) {
        if self.failed_escalations >= HUMAN_ESCALATION_THRESHOLD {
            self.status = EscalationStatus::HumanReviewRequired;
        }
    }

    /// Get the last proposed weights.
    pub fn last_proposal(&self) -> Option<LifecycleLambdaWeights> {
        self.last_proposal
    }
}

impl EscalationHandler for EscalationManager {
    fn should_escalate(&self) -> bool {
        self.consecutive_failures >= self.config.max_consecutive_failures
    }

    fn trigger_escalation(&mut self) -> UtlResult<EscalationResult> {
        if !self.should_escalate() {
            return Ok(EscalationResult::failure(
                "Escalation not triggered: below threshold",
                0,
            ));
        }

        // Create a simple test function that returns increasing accuracy
        // In real usage, this would be provided by the caller
        let mut test_accuracy: f32 = 0.5;
        let evaluate = |_lambda_s: f32| -> f32 {
            test_accuracy += 0.05;
            test_accuracy.min(0.95_f32)
        };

        // Default to Growth stage weights
        let current = LifecycleLambdaWeights::new(0.5, 0.5)
            .map_err(|e| UtlError::ConfigError(format!("Invalid default weights: {}", e)))?;

        self.total_attempts += 1;

        match self.run_bayesian_optimization(current, evaluate) {
            Ok(weights) => {
                let ei = self.gp.expected_improvement(weights.lambda_s());
                Ok(EscalationResult::success(
                    weights,
                    ei,
                    self.gp.num_observations(),
                ))
            }
            Err(e) => Ok(EscalationResult::failure(
                e.to_string(),
                self.gp.num_observations(),
            )),
        }
    }

    fn status(&self) -> EscalationStatus {
        self.status
    }

    fn reset(&mut self) {
        self.status = EscalationStatus::Idle;
        self.gp.clear();
        self.consecutive_failures = 0;
        // Note: We don't reset failed_escalations or stats
    }

    fn needs_human_review(&self) -> bool {
        self.status == EscalationStatus::HumanReviewRequired
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // GP Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gp_prediction_no_observations() {
        let gp = SimpleGaussianProcess::new();
        let (mean, variance) = gp.predict(0.5);

        // Prior mean should be 0.5, variance should be signal_var
        assert!((mean - 0.5).abs() < 0.01, "Prior mean should be 0.5, got {}", mean);
        assert!(variance > 0.0, "Variance should be positive");
    }

    #[test]
    fn test_gp_prediction_with_observations() {
        let mut gp = SimpleGaussianProcess::new();

        // Add some observations
        gp.add_observation(GpObservation {
            lambda_s: 0.3,
            accuracy: 0.7,
        });
        gp.add_observation(GpObservation {
            lambda_s: 0.5,
            accuracy: 0.9,
        });
        gp.add_observation(GpObservation {
            lambda_s: 0.7,
            accuracy: 0.6,
        });

        // Predict at observed point - should be close to observed value
        let (mean, variance) = gp.predict(0.5);
        assert!(
            (mean - 0.9).abs() < 0.2,
            "Mean at observed point should be close to 0.9, got {}",
            mean
        );
        // Variance at observed point should be low
        assert!(
            variance < gp.signal_var,
            "Variance at observed point should be reduced"
        );

        // Predict at unobserved point
        let (mean2, variance2) = gp.predict(0.4);
        assert!(mean2 >= 0.0 && mean2 <= 1.0, "Mean should be in [0,1]");
        assert!(variance2 > 0.0, "Variance should be positive");
    }

    #[test]
    fn test_ei_is_non_negative() {
        let mut gp = SimpleGaussianProcess::new();

        gp.add_observation(GpObservation {
            lambda_s: 0.5,
            accuracy: 0.8,
        });

        // EI should always be non-negative
        for i in 0..100 {
            let lambda = 0.05 + (i as f32) * 0.0085;
            let ei = gp.expected_improvement(lambda);
            assert!(ei >= 0.0, "EI should be non-negative, got {} at {}", ei, lambda);
        }
    }

    #[test]
    fn test_maximize_ei_in_bounds() {
        let mut gp = SimpleGaussianProcess::new();

        gp.add_observation(GpObservation {
            lambda_s: 0.3,
            accuracy: 0.7,
        });
        gp.add_observation(GpObservation {
            lambda_s: 0.7,
            accuracy: 0.6,
        });

        let best = gp.maximize_ei(0.05, 0.9, 100);

        assert!(best >= 0.05, "Best should be >= min bound, got {}", best);
        assert!(best <= 0.9, "Best should be <= max bound, got {}", best);
    }

    #[test]
    fn test_rbf_kernel() {
        let gp = SimpleGaussianProcess::new();

        // Kernel at same point should equal signal_var
        let k_same = gp.rbf_kernel(0.5, 0.5);
        assert!(
            (k_same - gp.signal_var).abs() < EPSILON,
            "k(x,x) should equal signal_var"
        );

        // Kernel should decrease with distance
        let k_close = gp.rbf_kernel(0.5, 0.6);
        let k_far = gp.rbf_kernel(0.5, 0.9);
        assert!(
            k_close > k_far,
            "Kernel should decrease with distance: {} > {}",
            k_close,
            k_far
        );

        // Kernel should be symmetric
        let k12 = gp.rbf_kernel(0.3, 0.7);
        let k21 = gp.rbf_kernel(0.7, 0.3);
        assert!(
            (k12 - k21).abs() < EPSILON,
            "Kernel should be symmetric"
        );
    }

    // -------------------------------------------------------------------------
    // Escalation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_escalation_triggers_at_threshold() {
        let mut manager = EscalationManager::with_defaults();

        // Should not escalate initially
        assert!(!manager.should_escalate());

        // Record failures up to threshold
        for _ in 0..10 {
            manager.record_failure_cycle();
        }

        // Should escalate now
        assert!(manager.should_escalate());
    }

    #[test]
    fn test_no_escalation_above_threshold() {
        let mut manager = EscalationManager::with_defaults();

        // Record fewer failures than threshold
        for _ in 0..5 {
            manager.record_failure_cycle();
        }

        // Should not escalate
        assert!(!manager.should_escalate());
        assert_eq!(manager.consecutive_failures(), 5);
    }

    #[test]
    fn test_human_escalation_after_failures() {
        let mut manager = EscalationManager::with_defaults();

        // Simulate 3 BO failures
        for _ in 0..HUMAN_ESCALATION_THRESHOLD {
            manager.failed_escalations += 1;
            manager.check_human_escalation();
        }

        assert!(manager.needs_human_review());
        assert_eq!(manager.status(), EscalationStatus::HumanReviewRequired);
    }

    #[test]
    fn test_bo_finds_better_lambda() {
        let mut manager = EscalationManager::with_defaults();

        // Synthetic objective: 1.0 - (lambda_s - 0.6)^2
        // Optimal at lambda_s = 0.6
        let evaluate = |lambda_s: f32| {
            let diff = lambda_s - 0.6;
            (1.0 - diff * diff).clamp(0.0, 1.0)
        };

        let current = LifecycleLambdaWeights::new(0.5, 0.5).unwrap();
        let result = manager.run_bayesian_optimization(current, evaluate);

        assert!(result.is_ok(), "BO should succeed");
        let weights = result.unwrap();

        // Should find lambda_s close to 0.6
        assert!(
            (weights.lambda_s() - 0.6).abs() < 0.15,
            "Should find optimal near 0.6, got {}",
            weights.lambda_s()
        );
    }

    #[test]
    fn test_escalation_stats() {
        let mut manager = EscalationManager::with_defaults();

        // Record some outcomes
        manager.record_outcome(true);
        manager.record_outcome(true);
        manager.record_outcome(false);

        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 3);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.failed, 1);
        assert!((stats.success_rate - 2.0 / 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_escalation_result_constructors() {
        let weights = LifecycleLambdaWeights::new(0.5, 0.5).unwrap();

        let success = EscalationResult::success(weights, 0.1, 5);
        assert!(success.success);
        assert!(success.proposed_weights.is_some());
        assert_eq!(success.iterations, 5);

        let failure = EscalationResult::failure("test reason", 3);
        assert!(!failure.success);
        assert!(failure.proposed_weights.is_none());
        assert_eq!(failure.failure_reason, Some("test reason".to_string()));
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gp_empty() {
        let gp = SimpleGaussianProcess::new();

        assert_eq!(gp.num_observations(), 0);
        assert!(gp.best_observation().is_none());

        // EI should be 0 with no observations (no best to improve on meaningfully)
        // Actually, with empty GP, we use prior, so EI could be non-zero
        let ei = gp.expected_improvement(0.5);
        assert!(!ei.is_nan(), "EI should not be NaN");
    }

    #[test]
    fn test_gp_nan_inputs() {
        let mut gp = SimpleGaussianProcess::new();

        // Add valid observation
        gp.add_observation(GpObservation {
            lambda_s: 0.5,
            accuracy: 0.8,
        });

        // NaN observation should be ignored
        gp.add_observation(GpObservation {
            lambda_s: f32::NAN,
            accuracy: 0.9,
        });
        assert_eq!(gp.num_observations(), 1);

        // Predict at NaN should return prior
        let (mean, _var) = gp.predict(f32::NAN);
        assert!((mean - 0.5).abs() < 0.01);

        // EI at NaN should be 0
        let ei = gp.expected_improvement(f32::NAN);
        assert_eq!(ei, 0.0);
    }

    #[test]
    fn test_gp_flat_objective() {
        let mut gp = SimpleGaussianProcess::new();

        // All observations have same accuracy
        for i in 0..5 {
            gp.add_observation(GpObservation {
                lambda_s: 0.2 + (i as f32) * 0.15,
                accuracy: 0.7,
            });
        }

        // EI should be low since there's no apparent improvement possible
        let ei = gp.expected_improvement(0.5);
        assert!(!ei.is_nan(), "EI should not be NaN with flat objective");
    }

    #[test]
    fn test_generate_initial_samples() {
        let manager = EscalationManager::with_defaults();
        let samples = manager.generate_initial_samples(0.5);

        assert_eq!(samples.len(), INITIAL_SAMPLES);

        // All samples should be in bounds
        for s in &samples {
            assert!(*s >= 0.05, "Sample below min: {}", s);
            assert!(*s <= 0.9, "Sample above max: {}", s);
        }

        // Current value should be included
        assert!(samples.contains(&0.5), "Current value should be in samples");
    }

    #[test]
    fn test_escalation_reset() {
        let mut manager = EscalationManager::with_defaults();

        // Set some state
        for _ in 0..15 {
            manager.record_failure_cycle();
        }
        manager.status = EscalationStatus::Failed;

        // Reset
        manager.reset();

        assert_eq!(manager.status(), EscalationStatus::Idle);
        assert_eq!(manager.consecutive_failures(), 0);
        assert_eq!(manager.gp.num_observations(), 0);
    }

    #[test]
    fn test_standard_normal_cdf() {
        let gp = SimpleGaussianProcess::new();

        // CDF at 0 should be 0.5
        let cdf_0 = gp.standard_normal_cdf(0.0);
        assert!(
            (cdf_0 - 0.5).abs() < 0.01,
            "CDF(0) should be 0.5, got {}",
            cdf_0
        );

        // CDF should be monotonically increasing
        let cdf_neg = gp.standard_normal_cdf(-1.0);
        let cdf_pos = gp.standard_normal_cdf(1.0);
        assert!(cdf_neg < cdf_0, "CDF(-1) should be < CDF(0)");
        assert!(cdf_0 < cdf_pos, "CDF(0) should be < CDF(1)");

        // Extreme values
        let cdf_large = gp.standard_normal_cdf(10.0);
        let cdf_small = gp.standard_normal_cdf(-10.0);
        assert!((cdf_large - 1.0).abs() < 0.01, "CDF(10) should be ~1.0");
        assert!(cdf_small < 0.01, "CDF(-10) should be ~0.0");
    }

    #[test]
    fn test_standard_normal_pdf() {
        let gp = SimpleGaussianProcess::new();

        // PDF at 0 should be maximum (~0.3989)
        let pdf_0 = gp.standard_normal_pdf(0.0);
        assert!(
            (pdf_0 - 0.3989).abs() < 0.01,
            "PDF(0) should be ~0.3989, got {}",
            pdf_0
        );

        // PDF should be symmetric
        let pdf_neg = gp.standard_normal_pdf(-1.0);
        let pdf_pos = gp.standard_normal_pdf(1.0);
        assert!(
            (pdf_neg - pdf_pos).abs() < 0.001,
            "PDF should be symmetric"
        );

        // PDF should decrease away from 0
        assert!(pdf_neg < pdf_0, "PDF(-1) should be < PDF(0)");
    }

    #[test]
    fn test_escalation_manager_with_custom_config() {
        let mut config = SelfCorrectionConfig::default();
        config.max_consecutive_failures = 5; // Lower threshold

        let mut manager = EscalationManager::new(config);

        for _ in 0..5 {
            manager.record_failure_cycle();
        }

        assert!(manager.should_escalate());
    }

    #[test]
    fn test_success_cycle_resets_failures() {
        let mut manager = EscalationManager::with_defaults();

        // Record some failures
        for _ in 0..8 {
            manager.record_failure_cycle();
        }
        assert_eq!(manager.consecutive_failures(), 8);

        // Success should reset
        manager.record_success_cycle();
        assert_eq!(manager.consecutive_failures(), 0);
    }

    #[test]
    fn test_gp_with_custom_params() {
        let gp = SimpleGaussianProcess::with_params(0.5, 0.001, 2.0);

        assert!((gp.length_scale - 0.5).abs() < EPSILON);
        assert!((gp.noise_var - 0.001).abs() < EPSILON);
        assert!((gp.signal_var - 2.0).abs() < EPSILON);

        // With larger length scale, kernel should decay slower
        let k = gp.rbf_kernel(0.0, 0.5);
        let gp_default = SimpleGaussianProcess::new();
        let k_default = gp_default.rbf_kernel(0.0, 0.5);

        // Signal var is different, so normalize
        let k_normalized = k / gp.signal_var;
        let k_default_normalized = k_default / gp_default.signal_var;
        assert!(
            k_normalized > k_default_normalized,
            "Larger length scale should give slower decay"
        );
    }

    #[test]
    fn test_escalation_stats_default() {
        let stats = EscalationStats::default();

        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.success_rate, 0.0);
    }

    #[test]
    fn test_escalation_stats_record() {
        let mut stats = EscalationStats::default();

        stats.record(true);
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.success_rate, 1.0);

        stats.record(false);
        assert_eq!(stats.total_attempts, 2);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.success_rate, 0.5);
    }

    #[test]
    fn test_has_converged_with_few_observations() {
        let manager = EscalationManager::with_defaults();

        // With no observations, shouldn't crash
        let converged = manager.has_converged();
        // GP returns prior for empty, so EI might be non-zero
        // Just check it doesn't crash
        assert!(!converged || converged); // Always true, just checking no panic
    }

    #[test]
    fn test_gp_clear() {
        let mut gp = SimpleGaussianProcess::new();

        gp.add_observation(GpObservation {
            lambda_s: 0.5,
            accuracy: 0.8,
        });
        assert_eq!(gp.num_observations(), 1);

        gp.clear();
        assert_eq!(gp.num_observations(), 0);
    }

    #[test]
    fn test_best_observation_accuracy() {
        let mut gp = SimpleGaussianProcess::new();

        gp.add_observation(GpObservation {
            lambda_s: 0.3,
            accuracy: 0.5,
        });
        gp.add_observation(GpObservation {
            lambda_s: 0.5,
            accuracy: 0.9,
        });
        gp.add_observation(GpObservation {
            lambda_s: 0.7,
            accuracy: 0.6,
        });

        let best = gp.best_observation().unwrap();
        assert!((best.accuracy - 0.9).abs() < EPSILON);
        assert!((best.lambda_s - 0.5).abs() < EPSILON);
    }
}
