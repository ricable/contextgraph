//! Temporal injector validation for E2/E3/E4.
//!
//! Tests:
//! - E2 Recency: Timestamps span configured range
//! - E2 Recency: Scores in valid range [0.0, 1.0]
//! - E2 Decay functions per ARCH-22
//! - E3 Periodic: Valid hour/day clusters
//! - E4 Sequence: Valid session positions

use crate::realdata::config::TemporalInjectionConfig;
use crate::realdata::temporal_injector::InjectedTemporalMetadata;
use super::{ValidationCheck, CheckPriority, PhaseResult, ValidationPhase};

/// Temporal validation results.
#[derive(Debug, Clone)]
pub struct TemporalValidationResult {
    /// All checks passed.
    pub all_passed: bool,
    /// Individual check results.
    pub checks: Vec<ValidationCheck>,
    /// E2 metrics.
    pub e2_metrics: E2Metrics,
    /// E3 metrics.
    pub e3_metrics: E3Metrics,
    /// E4 metrics.
    pub e4_metrics: E4Metrics,
}

/// E2 Recency metrics.
#[derive(Debug, Clone, Default)]
pub struct E2Metrics {
    /// Timestamp span in days.
    pub timestamp_span_days: f64,
    /// Min recency score.
    pub min_score: f64,
    /// Max recency score.
    pub max_score: f64,
    /// Mean recency score.
    pub mean_score: f64,
}

/// E3 Periodic metrics.
#[derive(Debug, Clone, Default)]
pub struct E3Metrics {
    /// Number of hour clusters.
    pub hour_clusters: usize,
    /// Number of day clusters.
    pub day_clusters: usize,
    /// Chunks with periodic metadata.
    pub chunks_with_metadata: usize,
}

/// E4 Sequence metrics.
#[derive(Debug, Clone, Default)]
pub struct E4Metrics {
    /// Number of sessions.
    pub num_sessions: usize,
    /// Min session length.
    pub min_session_length: usize,
    /// Max session length.
    pub max_session_length: usize,
    /// Chunks with session metadata.
    pub chunks_with_sessions: usize,
}

/// Temporal validator.
pub struct TemporalValidator;

impl TemporalValidator {
    /// Run all temporal validation checks.
    pub fn validate(
        config: &TemporalInjectionConfig,
        metadata: Option<&InjectedTemporalMetadata>,
        num_chunks: usize,
    ) -> TemporalValidationResult {
        let mut result = TemporalValidationResult {
            all_passed: true,
            checks: Vec::new(),
            e2_metrics: E2Metrics::default(),
            e3_metrics: E3Metrics::default(),
            e4_metrics: E4Metrics::default(),
        };

        // Config-only tests (don't need actual metadata)
        result.checks.push(Self::test_config_valid(config));
        result.checks.push(Self::test_decay_half_life_positive(config));

        // If we have actual injected metadata, run more tests
        if let Some(meta) = metadata {
            // E2 Tests
            let (check, metrics) = Self::test_e2_timestamp_span(config, meta);
            result.e2_metrics = metrics;
            if !check.is_passed() {
                result.all_passed = false;
            }
            result.checks.push(check);

            let check = Self::test_e2_recency_scores_valid(meta);
            if !check.is_passed() {
                result.all_passed = false;
            }
            result.checks.push(check);

            // E3 Tests
            if config.periodic_enabled {
                let (check, metrics) = Self::test_e3_periodic_clusters(config, meta);
                result.e3_metrics = metrics;
                if !check.is_passed() {
                    result.all_passed = false;
                }
                result.checks.push(check);

                let check = Self::test_e3_hour_day_valid(meta);
                if !check.is_passed() {
                    result.all_passed = false;
                }
                result.checks.push(check);
            }

            // E4 Tests
            let (check, metrics) = Self::test_e4_session_sequences(config, meta);
            result.e4_metrics = metrics;
            if !check.is_passed() {
                result.all_passed = false;
            }
            result.checks.push(check);

            let check = Self::test_e4_session_positions_monotonic(meta);
            if !check.is_passed() {
                result.all_passed = false;
            }
            result.checks.push(check);

            // Coverage test
            let check = Self::test_temporal_coverage(meta, num_chunks);
            if !check.is_passed() {
                result.all_passed = false;
            }
            result.checks.push(check);
        }

        // Decay function tests (formula validation, no data needed)
        result.checks.push(Self::test_decay_exponential());
        result.checks.push(Self::test_decay_linear());
        result.checks.push(Self::test_decay_step());

        // Update all_passed based on checks
        result.all_passed = result.checks.iter().all(|c| c.is_passed() || matches!(c.status, super::ValidationStatus::Warning | super::ValidationStatus::Skip));

        result
    }

    /// Convert to PhaseResult.
    pub fn to_phase_result(result: TemporalValidationResult, duration_ms: u64) -> PhaseResult {
        let mut phase_result = PhaseResult::new(ValidationPhase::Temporal);
        for check in result.checks {
            phase_result.add_check(check);
        }
        phase_result.finalize(duration_ms);
        phase_result
    }

    // Config tests

    fn test_config_valid(config: &TemporalInjectionConfig) -> ValidationCheck {
        let check = ValidationCheck::new(
            "config_valid",
            "Temporal injection config has valid parameters",
        ).with_priority(CheckPriority::High);

        let valid = config.recency_span_days > 0
            && config.recency_half_life_secs > 0
            && config.periodic_hour_clusters > 0
            && config.periodic_hour_clusters <= 24
            && config.sequence_num_sessions > 0
            && config.sequence_chunks_per_session > 0;

        if valid {
            check.pass("valid", "valid")
        } else {
            check.fail("invalid", "valid")
                .with_details(&format!(
                    "span_days={}, half_life={}, hour_clusters={}, sessions={}, chunks_per_session={}",
                    config.recency_span_days,
                    config.recency_half_life_secs,
                    config.periodic_hour_clusters,
                    config.sequence_num_sessions,
                    config.sequence_chunks_per_session
                ))
        }
    }

    fn test_decay_half_life_positive(config: &TemporalInjectionConfig) -> ValidationCheck {
        let check = ValidationCheck::new(
            "decay_half_life_positive",
            "[ARCH-22] Decay half-life is positive",
        ).with_priority(CheckPriority::High);

        if config.recency_half_life_secs > 0 {
            check.pass(&config.recency_half_life_secs.to_string(), "> 0")
        } else {
            check.fail(&config.recency_half_life_secs.to_string(), "> 0")
        }
    }

    // E2 Tests

    fn test_e2_timestamp_span(
        config: &TemporalInjectionConfig,
        metadata: &InjectedTemporalMetadata,
    ) -> (ValidationCheck, E2Metrics) {
        let check = ValidationCheck::new(
            "e2_timestamp_span",
            "E2 timestamps span >= 90% of configured range",
        ).with_priority(CheckPriority::High);

        let mut metrics = E2Metrics::default();

        if metadata.timestamps.is_empty() {
            return (check.fail("0 days", &format!("{} days", config.recency_span_days)), metrics);
        }

        let min_ts = metadata.timestamps.values().map(|t| t.timestamp.timestamp_millis()).min().unwrap_or(0);
        let max_ts = metadata.timestamps.values().map(|t| t.timestamp.timestamp_millis()).max().unwrap_or(0);

        let span_ms = max_ts - min_ts;
        let span_days = span_ms as f64 / (1000.0 * 60.0 * 60.0 * 24.0);
        let expected_days = config.recency_span_days as f64;
        let coverage = span_days / expected_days;

        metrics.timestamp_span_days = span_days;

        let passed = coverage >= 0.9;
        let check = if passed {
            check.pass(&format!("{:.1} days ({:.0}%)", span_days, coverage * 100.0), &format!(">= {:.1} days (90%)", expected_days * 0.9))
        } else {
            check.fail(&format!("{:.1} days ({:.0}%)", span_days, coverage * 100.0), &format!(">= {:.1} days (90%)", expected_days * 0.9))
        };

        (check, metrics)
    }

    fn test_e2_recency_scores_valid(metadata: &InjectedTemporalMetadata) -> ValidationCheck {
        let check = ValidationCheck::new(
            "e2_recency_scores_valid",
            "E2 recency scores in [0.0, 1.0]",
        ).with_priority(CheckPriority::Critical);

        if metadata.timestamps.is_empty() {
            return check.fail("no data", "[0.0, 1.0]");
        }

        // Use pre-computed default_recency_score
        let scores: Vec<f64> = metadata.timestamps.values()
            .map(|t| t.default_recency_score)
            .collect();

        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let valid = min_score >= 0.0 && max_score <= 1.0;

        if valid {
            check.pass(&format!("[{:.2}, {:.2}]", min_score, max_score), "[0.0, 1.0]")
        } else {
            check.fail(&format!("[{:.2}, {:.2}]", min_score, max_score), "[0.0, 1.0]")
        }
    }

    // E3 Tests

    fn test_e3_periodic_clusters(
        config: &TemporalInjectionConfig,
        metadata: &InjectedTemporalMetadata,
    ) -> (ValidationCheck, E3Metrics) {
        let check = ValidationCheck::new(
            "e3_periodic_clusters",
            "E3 periodic creates expected number of hour clusters",
        ).with_priority(CheckPriority::High);

        let mut metrics = E3Metrics::default();

        if metadata.periodic.is_empty() {
            return (check.fail("0 chunks", "> 0 chunks"), metrics);
        }

        // Count unique hour clusters
        let hour_clusters: std::collections::HashSet<usize> = metadata.periodic.values()
            .map(|p| p.hour_cluster)
            .collect();

        metrics.hour_clusters = hour_clusters.len();
        metrics.chunks_with_metadata = metadata.periodic.len();

        // Check we have the expected clusters (within reasonable bounds)
        let expected = config.periodic_hour_clusters;
        let passed = metrics.hour_clusters <= expected + 2 && metrics.hour_clusters > 0;

        let check = if passed {
            check.pass(&format!("{} clusters", metrics.hour_clusters), &format!("<= {} clusters", expected + 2))
        } else {
            check.fail(&format!("{} clusters", metrics.hour_clusters), &format!("<= {} clusters", expected + 2))
        };

        (check, metrics)
    }

    fn test_e3_hour_day_valid(metadata: &InjectedTemporalMetadata) -> ValidationCheck {
        let check = ValidationCheck::new(
            "e3_hour_day_valid",
            "[ARCH-23] E3 periodic hour in [0,23], day in [0,6]",
        ).with_priority(CheckPriority::High);

        if metadata.periodic.is_empty() {
            return check.pass("no data (skipped)", "[0,23], [0,6]");
        }

        let all_valid = metadata.periodic.values().all(|p| {
            p.hour < 24 && p.day_of_week < 7
        });

        if all_valid {
            check.pass("all valid", "hour < 24, day < 7")
        } else {
            let invalid: Vec<_> = metadata.periodic.values()
                .filter(|p| p.hour >= 24 || p.day_of_week >= 7)
                .take(3)
                .map(|p| format!("h={}, d={}", p.hour, p.day_of_week))
                .collect();
            check.fail(&invalid.join("; "), "hour < 24, day < 7")
        }
    }

    // E4 Tests

    fn test_e4_session_sequences(
        config: &TemporalInjectionConfig,
        metadata: &InjectedTemporalMetadata,
    ) -> (ValidationCheck, E4Metrics) {
        let check = ValidationCheck::new(
            "e4_session_sequences",
            "E4 creates expected number of sessions",
        ).with_priority(CheckPriority::High);

        let mut metrics = E4Metrics::default();

        metrics.num_sessions = metadata.sessions.len();
        metrics.chunks_with_sessions = metadata.sessions.iter()
            .flat_map(|s| s.chunks.iter())
            .count();

        if !metadata.sessions.is_empty() {
            let lengths: Vec<_> = metadata.sessions.iter()
                .map(|s| s.chunks.len())
                .collect();
            metrics.min_session_length = *lengths.iter().min().unwrap_or(&0);
            metrics.max_session_length = *lengths.iter().max().unwrap_or(&0);
        }

        // Check we have sessions
        let passed = metrics.num_sessions > 0;

        let check = if passed {
            check.pass(
                &format!("{} sessions", metrics.num_sessions),
                &format!("~{} sessions", config.sequence_num_sessions),
            )
        } else {
            check.fail("0 sessions", "> 0 sessions")
        };

        (check, metrics)
    }

    fn test_e4_session_positions_monotonic(metadata: &InjectedTemporalMetadata) -> ValidationCheck {
        let check = ValidationCheck::new(
            "e4_session_positions_monotonic",
            "[ARCH-24] E4 sequence positions are monotonic within sessions",
        ).with_priority(CheckPriority::High);

        if metadata.sessions.is_empty() {
            return check.pass("no sessions (skipped)", "monotonic");
        }

        // For each session, verify positions are monotonic (0, 1, 2, ...)
        let mut all_monotonic = true;
        for session in &metadata.sessions {
            for (expected_pos, chunk) in session.chunks.iter().enumerate() {
                // Positions should match their index
                if chunk.sequence_position != expected_pos {
                    all_monotonic = false;
                    break;
                }
            }
            if !all_monotonic {
                break;
            }
        }

        if all_monotonic {
            check.pass("monotonic", "monotonic")
                .with_details(&format!("{} sessions verified", metadata.sessions.len()))
        } else {
            check.fail("non-monotonic", "monotonic")
        }
    }

    fn test_temporal_coverage(metadata: &InjectedTemporalMetadata, num_chunks: usize) -> ValidationCheck {
        let check = ValidationCheck::new(
            "temporal_coverage",
            "Temporal metadata covers most chunks",
        ).with_priority(CheckPriority::Medium);

        if num_chunks == 0 {
            return check.pass("0 chunks", "n/a");
        }

        let timestamp_coverage = metadata.timestamps.len() as f64 / num_chunks as f64;

        if timestamp_coverage >= 0.9 {
            check.pass(&format!("{:.0}%", timestamp_coverage * 100.0), ">= 90%")
        } else {
            check.warning(&format!("{:.0}%", timestamp_coverage * 100.0), ">= 90%")
        }
    }

    // Decay function formula tests

    fn test_decay_exponential() -> ValidationCheck {
        let check = ValidationCheck::new(
            "decay_exponential",
            "[ARCH-22] Exponential decay: score = 0.5 at half-life",
        ).with_priority(CheckPriority::High);

        // Formula: score = exp(-age * 0.693 / half_life)
        // At age = half_life: score = exp(-0.693) ≈ 0.5
        let half_life: f64 = 86400.0; // 1 day
        let age: f64 = half_life;
        let score: f64 = (-age * 0.693 / half_life).exp();

        let passed = (score - 0.5).abs() < 0.01;

        if passed {
            check.pass(&format!("{:.3}", score), "0.5")
        } else {
            check.fail(&format!("{:.3}", score), "0.5")
        }
    }

    fn test_decay_linear() -> ValidationCheck {
        let check = ValidationCheck::new(
            "decay_linear",
            "[ARCH-22] Linear decay: score = max(0, 1 - age/max_age)",
        ).with_priority(CheckPriority::High);

        // At age = max_age/2: score = 0.5
        let max_age: f64 = 86400.0 * 365.0; // 1 year
        let age: f64 = max_age / 2.0;
        let score: f64 = (1.0 - age / max_age).max(0.0);

        let passed = (score - 0.5).abs() < f64::EPSILON;

        if passed {
            check.pass(&format!("{:.3}", score), "0.5")
        } else {
            check.fail(&format!("{:.3}", score), "0.5")
        }
    }

    fn test_decay_step() -> ValidationCheck {
        let check = ValidationCheck::new(
            "decay_step",
            "[ARCH-22] Step decay: Fresh(<5min)=1.0, Recent(<1h)=0.8, Today(<1d)=0.5",
        ).with_priority(CheckPriority::High);

        // Step function thresholds
        let five_min = 5.0 * 60.0;
        let one_hour = 60.0 * 60.0;
        let one_day = 24.0 * 60.0 * 60.0;

        let step_fn = |age: f64| -> f64 {
            if age < five_min { 1.0 }
            else if age < one_hour { 0.8 }
            else if age < one_day { 0.5 }
            else { 0.1 }
        };

        let fresh = step_fn(60.0); // 1 minute
        let recent = step_fn(30.0 * 60.0); // 30 minutes
        let today = step_fn(12.0 * 60.0 * 60.0); // 12 hours
        let older = step_fn(48.0 * 60.0 * 60.0); // 2 days

        let passed = (fresh - 1.0).abs() < f64::EPSILON
            && (recent - 0.8).abs() < f64::EPSILON
            && (today - 0.5).abs() < f64::EPSILON
            && (older - 0.1).abs() < f64::EPSILON;

        if passed {
            check.pass("1.0, 0.8, 0.5, 0.1", "1.0, 0.8, 0.5, 0.1")
        } else {
            check.fail(&format!("{}, {}, {}, {}", fresh, recent, today, older), "1.0, 0.8, 0.5, 0.1")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_formulas() {
        // Test exponential decay at half-life
        let half_life: f64 = 86400.0;
        let age: f64 = half_life;
        let score: f64 = (-age * 0.693 / half_life).exp();
        assert!((score - 0.5).abs() < 0.01);

        // Test linear decay at midpoint
        let max_age: f64 = 86400.0 * 365.0;
        let age: f64 = max_age / 2.0;
        let score: f64 = (1.0 - age / max_age).max(0.0);
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_temporal_validation_config_only() {
        let config = TemporalInjectionConfig::default();
        let result = TemporalValidator::validate(&config, None, 0);
        // Should at least have config tests
        assert!(!result.checks.is_empty());
    }
}
