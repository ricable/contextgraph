//! Autonomous Services Threshold Management
//!
//! Domain-aware thresholds for NORTH autonomous services:
//! - ObsolescenceDetector (NORTH-017)
//! - DriftDetector (NORTH-010)
//!
//! # Architecture
//!
//! This module bridges the ATC (Adaptive Threshold Calibration) system
//! to the autonomous services, providing domain-aware thresholds that
//! adapt based on the current domain context.
//!
//! # Legacy Values (preserved in default_general)
//!
//! The following legacy constants are preserved for backwards compatibility:
//! - DEFAULT_RELEVANCE_THRESHOLD = 0.30 -> obsolescence_low
//! - MEDIUM_CONFIDENCE_THRESHOLD = 0.60 -> obsolescence_mid
//! - HIGH_CONFIDENCE_THRESHOLD = 0.80 -> obsolescence_high
//! - WARNING_SLOPE = 0.02 -> drift_slope_warning
//! - CRITICAL_SLOPE = 0.05 -> drift_slope_critical
//!
//! # Constitution Reference
//!
//! See `docs2/constitution.yaml` sections:
//! - `adaptive_thresholds.priors` for base threshold ranges
//! - `autonomous.obsolescence` for obsolescence detection
//! - `autonomous.drift` for drift detection

use crate::atc::{AdaptiveThresholdCalibration, Domain};
use crate::error::{CoreError, CoreResult};

/// Thresholds for NORTH autonomous services.
///
/// These thresholds control the behavior of the ObsolescenceDetector (NORTH-017)
/// and DriftDetector (NORTH-010) services. They can be derived from the ATC
/// system for domain-specific calibration or use legacy defaults for backward
/// compatibility.
///
/// # Invariants
///
/// The following invariants are enforced:
/// - Monotonicity: obsolescence_high > obsolescence_mid > obsolescence_low
/// - drift_slope_critical > drift_slope_warning
/// - All values must be positive and within valid ranges
///
/// # Examples
///
/// ```rust
/// use context_graph_core::autonomous::AutonomousThresholds;
/// use context_graph_core::atc::{AdaptiveThresholdCalibration, Domain};
///
/// // Use legacy defaults
/// let thresholds = AutonomousThresholds::default_general();
/// assert_eq!(thresholds.obsolescence_low, 0.30);
/// assert_eq!(thresholds.obsolescence_mid, 0.60);
/// assert_eq!(thresholds.obsolescence_high, 0.80);
///
/// // Or derive from ATC for domain-specific thresholds
/// let atc = AdaptiveThresholdCalibration::new();
/// let medical_thresholds = AutonomousThresholds::from_atc(&atc, Domain::Medical)
///     .expect("Medical domain should be available");
/// assert!(medical_thresholds.obsolescence_high > thresholds.obsolescence_high);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutonomousThresholds {
    /// Low relevance threshold for obsolescence detection.
    ///
    /// Legacy: DEFAULT_RELEVANCE_THRESHOLD = 0.30
    /// Range: [0.20, 0.50]
    ///
    /// Scores below this indicate potentially obsolete goals.
    pub obsolescence_low: f32,

    /// Medium confidence threshold for obsolescence detection.
    ///
    /// Legacy: MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    /// Range: [0.45, 0.75]
    ///
    /// Goals with scores between low and mid require monitoring.
    pub obsolescence_mid: f32,

    /// High confidence threshold for obsolescence detection.
    ///
    /// Legacy: HIGH_CONFIDENCE_THRESHOLD = 0.80
    /// Range: [0.65, 0.90]
    ///
    /// Goals with scores above this are considered actively relevant.
    pub obsolescence_high: f32,

    /// Warning drift slope threshold.
    ///
    /// Legacy: WARNING_SLOPE = 0.02
    /// Range: [0.001, 0.05]
    ///
    /// Drift slopes above this trigger monitoring alerts.
    pub drift_slope_warning: f32,

    /// Critical drift slope threshold.
    ///
    /// Legacy: CRITICAL_SLOPE = 0.05
    /// Range: [0.01, 0.10]
    ///
    /// Drift slopes above this trigger immediate intervention.
    pub drift_slope_critical: f32,
}

impl AutonomousThresholds {
    /// Create thresholds from ATC for a specific domain.
    ///
    /// This method derives domain-aware thresholds from the ATC system,
    /// which provides adaptive threshold calibration based on domain
    /// characteristics (e.g., Medical is stricter than Creative).
    ///
    /// # Arguments
    ///
    /// * `atc` - Reference to the ATC system
    /// * `domain` - The domain to get thresholds for
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ConfigError` if:
    /// - ATC is missing thresholds for the requested domain
    ///
    /// Returns `CoreError::ValidationError` if:
    /// - The derived thresholds fail monotonicity checks
    /// - Any threshold is outside its valid range
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::autonomous::AutonomousThresholds;
    /// use context_graph_core::atc::{AdaptiveThresholdCalibration, Domain};
    ///
    /// let atc = AdaptiveThresholdCalibration::new();
    ///
    /// // Medical domain has stricter thresholds
    /// let medical = AutonomousThresholds::from_atc(&atc, Domain::Medical).unwrap();
    ///
    /// // Creative domain has looser thresholds
    /// let creative = AutonomousThresholds::from_atc(&atc, Domain::Creative).unwrap();
    ///
    /// assert!(medical.obsolescence_high > creative.obsolescence_high);
    /// ```
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> CoreResult<Self> {
        let dt = atc.get_domain_thresholds(domain).ok_or_else(|| {
            CoreError::ConfigError(format!(
                "ATC missing domain thresholds for {:?}. \
                 Ensure ATC is properly initialized with all domains.",
                domain
            ))
        })?;

        // Derive autonomous thresholds from domain thresholds
        // The drift_slope_critical is computed as 2.5x the base drift_slope
        // to maintain the legacy ratio (0.05 / 0.02 = 2.5)
        let auto = Self {
            obsolescence_low: dt.theta_obsolescence_low,
            obsolescence_mid: dt.theta_obsolescence_mid,
            obsolescence_high: dt.theta_obsolescence_high,
            drift_slope_warning: dt.theta_drift_slope,
            drift_slope_critical: dt.theta_drift_slope * 2.5,
        };

        if !auto.is_valid() {
            return Err(CoreError::ValidationError {
                field: "AutonomousThresholds".to_string(),
                message: format!(
                    "Invalid thresholds derived from domain {:?}: \
                     obsolescence [low={:.3}, mid={:.3}, high={:.3}], \
                     drift [warning={:.4}, critical={:.4}]. \
                     Check monotonicity: high > mid > low, critical > warning",
                    domain,
                    auto.obsolescence_low,
                    auto.obsolescence_mid,
                    auto.obsolescence_high,
                    auto.drift_slope_warning,
                    auto.drift_slope_critical
                ),
            });
        }

        Ok(auto)
    }

    /// Create with legacy defaults for the General domain.
    ///
    /// These values MUST match the original hardcoded constants exactly
    /// to ensure backward compatibility during migration.
    ///
    /// # Legacy Constant Mapping
    ///
    /// | Field | Legacy Constant | Value |
    /// |-------|-----------------|-------|
    /// | obsolescence_low | DEFAULT_RELEVANCE_THRESHOLD | 0.30 |
    /// | obsolescence_mid | MEDIUM_CONFIDENCE_THRESHOLD | 0.60 |
    /// | obsolescence_high | HIGH_CONFIDENCE_THRESHOLD | 0.80 |
    /// | drift_slope_warning | WARNING_SLOPE | 0.02 |
    /// | drift_slope_critical | CRITICAL_SLOPE | 0.05 |
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::autonomous::AutonomousThresholds;
    ///
    /// let t = AutonomousThresholds::default_general();
    ///
    /// // These values are guaranteed to match legacy constants
    /// assert_eq!(t.obsolescence_low, 0.30);
    /// assert_eq!(t.obsolescence_mid, 0.60);
    /// assert_eq!(t.obsolescence_high, 0.80);
    /// assert_eq!(t.drift_slope_warning, 0.02);
    /// assert_eq!(t.drift_slope_critical, 0.05);
    /// ```
    #[inline]
    pub fn default_general() -> Self {
        Self {
            obsolescence_low: 0.30,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope_warning: 0.02,
            drift_slope_critical: 0.05,
        }
    }

    /// Validate thresholds against invariants.
    ///
    /// # Checks
    ///
    /// 1. Monotonicity: obsolescence_high > obsolescence_mid > obsolescence_low
    /// 2. Monotonicity: drift_slope_critical > drift_slope_warning
    /// 3. All values are positive and non-NaN
    /// 4. Obsolescence thresholds in valid ranges
    /// 5. Drift slopes in valid ranges
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::autonomous::AutonomousThresholds;
    ///
    /// let valid = AutonomousThresholds::default_general();
    /// assert!(valid.is_valid());
    ///
    /// let invalid = AutonomousThresholds {
    ///     obsolescence_low: 0.80,  // Higher than high!
    ///     obsolescence_mid: 0.60,
    ///     obsolescence_high: 0.30, // Lower than low!
    ///     drift_slope_warning: 0.02,
    ///     drift_slope_critical: 0.05,
    /// };
    /// assert!(!invalid.is_valid());
    /// ```
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // Intentional - NaN checks done separately above
    pub fn is_valid(&self) -> bool {
        // Check for NaN or negative values
        if self.obsolescence_low.is_nan()
            || self.obsolescence_mid.is_nan()
            || self.obsolescence_high.is_nan()
            || self.drift_slope_warning.is_nan()
            || self.drift_slope_critical.is_nan()
        {
            return false;
        }

        if self.obsolescence_low < 0.0
            || self.obsolescence_mid < 0.0
            || self.obsolescence_high < 0.0
            || self.drift_slope_warning < 0.0
            || self.drift_slope_critical < 0.0
        {
            return false;
        }

        // Monotonicity: high > mid > low
        if !(self.obsolescence_high > self.obsolescence_mid
            && self.obsolescence_mid > self.obsolescence_low)
        {
            return false;
        }

        // Monotonicity: critical > warning
        if !(self.drift_slope_critical > self.drift_slope_warning) {
            return false;
        }

        // Range checks for obsolescence thresholds
        // Ranges are slightly expanded from DomainThresholds to allow flexibility
        if !(0.15..=0.55).contains(&self.obsolescence_low) {
            return false;
        }
        if !(0.40..=0.80).contains(&self.obsolescence_mid) {
            return false;
        }
        if !(0.60..=0.95).contains(&self.obsolescence_high) {
            return false;
        }

        // Range checks for drift slopes
        // Warning: [0.001, 0.05] (from theta_drift_slope range [0.001, 0.01])
        // Critical: [0.0025, 0.15] (warning * 2.5, with some tolerance)
        if !(0.001..=0.05).contains(&self.drift_slope_warning) {
            return false;
        }
        if !(0.002..=0.15).contains(&self.drift_slope_critical) {
            return false;
        }

        true
    }

    /// Classify an alignment score using obsolescence thresholds.
    ///
    /// # Returns
    ///
    /// - `ObsolescenceLevel::Active` if score >= high
    /// - `ObsolescenceLevel::Monitoring` if mid <= score < high
    /// - `ObsolescenceLevel::AtRisk` if low <= score < mid
    /// - `ObsolescenceLevel::Obsolete` if score < low
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::autonomous::{AutonomousThresholds, ObsolescenceLevel};
    ///
    /// let t = AutonomousThresholds::default_general();
    ///
    /// assert_eq!(t.classify_obsolescence(0.85), ObsolescenceLevel::Active);
    /// assert_eq!(t.classify_obsolescence(0.70), ObsolescenceLevel::Monitoring);
    /// assert_eq!(t.classify_obsolescence(0.45), ObsolescenceLevel::AtRisk);
    /// assert_eq!(t.classify_obsolescence(0.20), ObsolescenceLevel::Obsolete);
    /// ```
    pub fn classify_obsolescence(&self, score: f32) -> ObsolescenceLevel {
        if score >= self.obsolescence_high {
            ObsolescenceLevel::Active
        } else if score >= self.obsolescence_mid {
            ObsolescenceLevel::Monitoring
        } else if score >= self.obsolescence_low {
            ObsolescenceLevel::AtRisk
        } else {
            ObsolescenceLevel::Obsolete
        }
    }

    /// Classify a drift slope using drift thresholds.
    ///
    /// # Returns
    ///
    /// - `DriftLevel::Normal` if slope < warning
    /// - `DriftLevel::Warning` if warning <= slope < critical
    /// - `DriftLevel::Critical` if slope >= critical
    ///
    /// # Examples
    ///
    /// ```rust
    /// use context_graph_core::autonomous::{AutonomousThresholds, DriftLevel};
    ///
    /// let t = AutonomousThresholds::default_general();
    ///
    /// assert_eq!(t.classify_drift(0.01), DriftLevel::Normal);
    /// assert_eq!(t.classify_drift(0.03), DriftLevel::Warning);
    /// assert_eq!(t.classify_drift(0.06), DriftLevel::Critical);
    /// ```
    pub fn classify_drift(&self, slope: f32) -> DriftLevel {
        if slope >= self.drift_slope_critical {
            DriftLevel::Critical
        } else if slope >= self.drift_slope_warning {
            DriftLevel::Warning
        } else {
            DriftLevel::Normal
        }
    }
}

impl Default for AutonomousThresholds {
    /// Default implementation uses `default_general()` for backward compatibility.
    fn default() -> Self {
        Self::default_general()
    }
}

/// Obsolescence classification levels.
///
/// These levels indicate how urgently a goal needs attention based on
/// its relevance score relative to the obsolescence thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObsolescenceLevel {
    /// Score >= high threshold. Goal is actively relevant.
    Active,
    /// Mid <= score < high. Goal requires periodic monitoring.
    Monitoring,
    /// Low <= score < mid. Goal is at risk of becoming obsolete.
    AtRisk,
    /// Score < low. Goal should be considered obsolete.
    Obsolete,
}

impl ObsolescenceLevel {
    /// Check if the level requires any action.
    pub fn requires_action(&self) -> bool {
        matches!(self, ObsolescenceLevel::AtRisk | ObsolescenceLevel::Obsolete)
    }

    /// Check if the level is critical (obsolete).
    pub fn is_critical(&self) -> bool {
        matches!(self, ObsolescenceLevel::Obsolete)
    }
}

/// Drift classification levels.
///
/// These levels indicate the severity of alignment drift based on
/// the measured slope relative to drift thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriftLevel {
    /// Slope < warning threshold. Drift is within acceptable bounds.
    Normal,
    /// Warning <= slope < critical. Drift requires attention.
    Warning,
    /// Slope >= critical. Drift requires immediate intervention.
    Critical,
}

impl DriftLevel {
    /// Check if the level requires any action.
    pub fn requires_action(&self) -> bool {
        matches!(self, DriftLevel::Warning | DriftLevel::Critical)
    }

    /// Check if the level is critical.
    pub fn is_critical(&self) -> bool {
        matches!(self, DriftLevel::Critical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atc::AdaptiveThresholdCalibration;

    // ========== Legacy Value Tests ==========

    #[test]
    fn test_default_general_matches_legacy_default_relevance_threshold() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(
            t.obsolescence_low, 0.30,
            "must match DEFAULT_RELEVANCE_THRESHOLD"
        );
    }

    #[test]
    fn test_default_general_matches_legacy_medium_confidence_threshold() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(
            t.obsolescence_mid, 0.60,
            "must match MEDIUM_CONFIDENCE_THRESHOLD"
        );
    }

    #[test]
    fn test_default_general_matches_legacy_high_confidence_threshold() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(
            t.obsolescence_high, 0.80,
            "must match HIGH_CONFIDENCE_THRESHOLD"
        );
    }

    #[test]
    fn test_default_general_matches_legacy_warning_slope() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.drift_slope_warning, 0.02, "must match WARNING_SLOPE");
    }

    #[test]
    fn test_default_general_matches_legacy_critical_slope() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.drift_slope_critical, 0.05, "must match CRITICAL_SLOPE");
    }

    #[test]
    fn test_default_impl_uses_default_general() {
        let default = AutonomousThresholds::default();
        let general = AutonomousThresholds::default_general();
        assert_eq!(default, general);
    }

    // ========== Validation Tests ==========

    #[test]
    fn test_default_general_is_valid() {
        let t = AutonomousThresholds::default_general();
        assert!(t.is_valid(), "default_general must always be valid");
    }

    #[test]
    fn test_is_valid_fails_on_broken_monotonicity_obsolescence() {
        let invalid = AutonomousThresholds {
            obsolescence_low: 0.50,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.40, // < mid!
            drift_slope_warning: 0.02,
            drift_slope_critical: 0.05,
        };
        assert!(
            !invalid.is_valid(),
            "should fail when high < mid"
        );
    }

    #[test]
    fn test_is_valid_fails_on_broken_monotonicity_drift() {
        let invalid = AutonomousThresholds {
            obsolescence_low: 0.30,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope_warning: 0.05,
            drift_slope_critical: 0.02, // < warning!
        };
        assert!(
            !invalid.is_valid(),
            "should fail when critical < warning"
        );
    }

    #[test]
    fn test_is_valid_fails_on_nan() {
        let invalid = AutonomousThresholds {
            obsolescence_low: f32::NAN,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope_warning: 0.02,
            drift_slope_critical: 0.05,
        };
        assert!(!invalid.is_valid(), "should fail on NaN value");
    }

    #[test]
    fn test_is_valid_fails_on_negative() {
        let invalid = AutonomousThresholds {
            obsolescence_low: -0.30,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope_warning: 0.02,
            drift_slope_critical: 0.05,
        };
        assert!(!invalid.is_valid(), "should fail on negative value");
    }

    #[test]
    fn test_is_valid_fails_out_of_range_obsolescence_low() {
        let invalid = AutonomousThresholds {
            obsolescence_low: 0.10, // Below min 0.15
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope_warning: 0.02,
            drift_slope_critical: 0.05,
        };
        assert!(
            !invalid.is_valid(),
            "should fail when obsolescence_low out of range"
        );
    }

    #[test]
    fn test_is_valid_fails_out_of_range_drift_warning() {
        let invalid = AutonomousThresholds {
            obsolescence_low: 0.30,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope_warning: 0.0005, // Below min 0.001
            drift_slope_critical: 0.05,
        };
        assert!(
            !invalid.is_valid(),
            "should fail when drift_slope_warning out of range"
        );
    }

    // ========== from_atc Tests ==========

    #[test]
    fn test_from_atc_general_domain() {
        let atc = AdaptiveThresholdCalibration::new();
        let result = AutonomousThresholds::from_atc(&atc, Domain::General);
        assert!(result.is_ok(), "General domain should be available");

        let t = result.unwrap();
        assert!(t.is_valid(), "Thresholds from ATC must be valid");

        // Verify monotonicity
        assert!(t.obsolescence_high > t.obsolescence_mid);
        assert!(t.obsolescence_mid > t.obsolescence_low);
        assert!(t.drift_slope_critical > t.drift_slope_warning);
    }

    #[test]
    fn test_from_atc_all_domains() {
        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Legal,
            Domain::Creative,
            Domain::Research,
            Domain::General,
        ] {
            let result = AutonomousThresholds::from_atc(&atc, domain);
            assert!(
                result.is_ok(),
                "Domain {:?} should be available",
                domain
            );

            let t = result.unwrap();
            assert!(
                t.is_valid(),
                "Domain {:?} thresholds must be valid",
                domain
            );
        }
    }

    #[test]
    fn test_from_atc_medical_stricter_than_creative() {
        let atc = AdaptiveThresholdCalibration::new();

        let medical = AutonomousThresholds::from_atc(&atc, Domain::Medical).unwrap();
        let creative = AutonomousThresholds::from_atc(&atc, Domain::Creative).unwrap();

        // Medical (strictness=1.0) should have higher thresholds than Creative (strictness=0.2)
        assert!(
            medical.obsolescence_high > creative.obsolescence_high,
            "Medical high {} should > Creative high {}",
            medical.obsolescence_high,
            creative.obsolescence_high
        );

        // Medical should have smaller drift slope (more sensitive)
        assert!(
            medical.drift_slope_warning < creative.drift_slope_warning,
            "Medical drift {} should < Creative drift {}",
            medical.drift_slope_warning,
            creative.drift_slope_warning
        );
    }

    #[test]
    fn test_from_atc_drift_slope_ratio() {
        let atc = AdaptiveThresholdCalibration::new();
        let t = AutonomousThresholds::from_atc(&atc, Domain::General).unwrap();

        // Critical should be 2.5x warning (legacy ratio preserved)
        let ratio = t.drift_slope_critical / t.drift_slope_warning;
        assert!(
            (ratio - 2.5).abs() < 0.01,
            "Critical/Warning ratio should be ~2.5, got {}",
            ratio
        );
    }

    // ========== Classification Tests ==========

    #[test]
    fn test_classify_obsolescence_active() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.classify_obsolescence(0.85), ObsolescenceLevel::Active);
        assert_eq!(t.classify_obsolescence(0.80), ObsolescenceLevel::Active);
        assert_eq!(t.classify_obsolescence(1.0), ObsolescenceLevel::Active);
    }

    #[test]
    fn test_classify_obsolescence_monitoring() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.classify_obsolescence(0.70), ObsolescenceLevel::Monitoring);
        assert_eq!(t.classify_obsolescence(0.60), ObsolescenceLevel::Monitoring);
        assert_eq!(t.classify_obsolescence(0.79), ObsolescenceLevel::Monitoring);
    }

    #[test]
    fn test_classify_obsolescence_at_risk() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.classify_obsolescence(0.45), ObsolescenceLevel::AtRisk);
        assert_eq!(t.classify_obsolescence(0.30), ObsolescenceLevel::AtRisk);
        assert_eq!(t.classify_obsolescence(0.59), ObsolescenceLevel::AtRisk);
    }

    #[test]
    fn test_classify_obsolescence_obsolete() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.classify_obsolescence(0.20), ObsolescenceLevel::Obsolete);
        assert_eq!(t.classify_obsolescence(0.0), ObsolescenceLevel::Obsolete);
        assert_eq!(t.classify_obsolescence(0.29), ObsolescenceLevel::Obsolete);
    }

    #[test]
    fn test_classify_drift_normal() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.classify_drift(0.01), DriftLevel::Normal);
        assert_eq!(t.classify_drift(0.0), DriftLevel::Normal);
        assert_eq!(t.classify_drift(0.019), DriftLevel::Normal);
    }

    #[test]
    fn test_classify_drift_warning() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.classify_drift(0.02), DriftLevel::Warning);
        assert_eq!(t.classify_drift(0.03), DriftLevel::Warning);
        assert_eq!(t.classify_drift(0.049), DriftLevel::Warning);
    }

    #[test]
    fn test_classify_drift_critical() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.classify_drift(0.05), DriftLevel::Critical);
        assert_eq!(t.classify_drift(0.10), DriftLevel::Critical);
        assert_eq!(t.classify_drift(1.0), DriftLevel::Critical);
    }

    // ========== Level Helper Tests ==========

    #[test]
    fn test_obsolescence_level_requires_action() {
        assert!(!ObsolescenceLevel::Active.requires_action());
        assert!(!ObsolescenceLevel::Monitoring.requires_action());
        assert!(ObsolescenceLevel::AtRisk.requires_action());
        assert!(ObsolescenceLevel::Obsolete.requires_action());
    }

    #[test]
    fn test_obsolescence_level_is_critical() {
        assert!(!ObsolescenceLevel::Active.is_critical());
        assert!(!ObsolescenceLevel::Monitoring.is_critical());
        assert!(!ObsolescenceLevel::AtRisk.is_critical());
        assert!(ObsolescenceLevel::Obsolete.is_critical());
    }

    #[test]
    fn test_drift_level_requires_action() {
        assert!(!DriftLevel::Normal.requires_action());
        assert!(DriftLevel::Warning.requires_action());
        assert!(DriftLevel::Critical.requires_action());
    }

    #[test]
    fn test_drift_level_is_critical() {
        assert!(!DriftLevel::Normal.is_critical());
        assert!(!DriftLevel::Warning.is_critical());
        assert!(DriftLevel::Critical.is_critical());
    }

    // ========== Print Test for Manual Verification ==========

    #[test]
    fn test_print_all_domain_autonomous_thresholds() {
        println!("\n=== Autonomous Thresholds by Domain (TASK-ATC-P2-007) ===\n");

        let atc = AdaptiveThresholdCalibration::new();

        // Print legacy defaults first
        let legacy = AutonomousThresholds::default_general();
        println!("Legacy Defaults:");
        println!("  obsolescence_low: {:.3} (DEFAULT_RELEVANCE_THRESHOLD)", legacy.obsolescence_low);
        println!("  obsolescence_mid: {:.3} (MEDIUM_CONFIDENCE_THRESHOLD)", legacy.obsolescence_mid);
        println!("  obsolescence_high: {:.3} (HIGH_CONFIDENCE_THRESHOLD)", legacy.obsolescence_high);
        println!("  drift_slope_warning: {:.4} (WARNING_SLOPE)", legacy.drift_slope_warning);
        println!("  drift_slope_critical: {:.4} (CRITICAL_SLOPE)", legacy.drift_slope_critical);
        println!();

        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Legal,
            Domain::Creative,
            Domain::Research,
            Domain::General,
        ] {
            let t = AutonomousThresholds::from_atc(&atc, domain).unwrap();
            println!("{:?} (strictness={:.1}):", domain, domain.strictness());
            println!("  obsolescence_low: {:.3}", t.obsolescence_low);
            println!("  obsolescence_mid: {:.3}", t.obsolescence_mid);
            println!("  obsolescence_high: {:.3}", t.obsolescence_high);
            println!("  drift_slope_warning: {:.6}", t.drift_slope_warning);
            println!("  drift_slope_critical: {:.6}", t.drift_slope_critical);
            println!("  valid: {}", t.is_valid());
            println!();
        }
    }
}
