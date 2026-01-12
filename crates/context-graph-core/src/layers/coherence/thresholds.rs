//! GWT Threshold Management
//!
//! Provides domain-aware thresholds for Global Workspace Theory operations.
//! Replaces hardcoded constants with adaptive threshold calibration (ATC).
//!
//! # Constitution Reference
//!
//! From `docs2/constitution.yaml` lines 220-236:
//! - gwt.kuramoto.thresholds: { coherent: "r≥0.8", fragmented: "r<0.5", hypersync: "r>0.95" }
//! - gwt.workspace.coherence_threshold: 0.8
//!
//! # Legacy Values (MUST preserve for backwards compatibility)
//!
//! - GW_THRESHOLD = 0.70 (broadcast gate)
//! - HYPERSYNC_THRESHOLD = 0.95 (pathological state)
//! - FRAGMENTATION_THRESHOLD = 0.50 (fragmented state)
//!
//! # ATC Domain Thresholds
//!
//! From `atc/domain.rs`:
//! - theta_gate: [0.65, 0.95] GW broadcast gate
//! - theta_hypersync: [0.90, 0.99] Hypersync detection
//! - theta_fragmentation: [0.35, 0.65] Fragmentation warning

use crate::atc::{AdaptiveThresholdCalibration, Domain};
use crate::error::{CoreError, CoreResult};

/// GWT thresholds for consciousness state determination.
///
/// These thresholds define the boundaries for consciousness states:
/// - `gate`: Order parameter threshold for GW broadcast (ignition)
/// - `hypersync`: Order parameter above which is pathological hypersync
/// - `fragmentation`: Order parameter below which is fragmented
///
/// # Constitution Reference
///
/// - theta_gate: [0.65, 0.95] GW broadcast gate
/// - theta_hypersync: [0.90, 0.99] Hypersync detection
/// - theta_fragmentation: [0.35, 0.65] Fragmentation warning
///
/// # Invariants
///
/// For valid thresholds: fragmentation < gate < hypersync
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GwtThresholds {
    /// Threshold for GW broadcast (ignition).
    /// When order_param >= gate, broadcast is triggered.
    pub gate: f32,
    /// Threshold above which is pathological hypersync.
    /// When order_param > hypersync, system is in hypersync state.
    pub hypersync: f32,
    /// Threshold below which is fragmented.
    /// When order_param < fragmentation, system is fragmented.
    pub fragmentation: f32,
}

impl GwtThresholds {
    /// Create from ATC for a specific domain.
    ///
    /// Retrieves domain-specific thresholds from the Adaptive Threshold Calibration system.
    /// Domain strictness affects threshold values:
    /// - Stricter domains (Medical, Code) have higher gates (harder to broadcast)
    /// - Looser domains (Creative) have lower gates (easier to broadcast)
    ///
    /// # Arguments
    ///
    /// * `atc` - Reference to the AdaptiveThresholdCalibration system
    /// * `domain` - The domain to retrieve thresholds for
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - ATC doesn't have the requested domain
    /// - Retrieved thresholds fail validation (out of range or violate monotonicity)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use context_graph_core::atc::{AdaptiveThresholdCalibration, Domain};
    /// use context_graph_core::layers::coherence::GwtThresholds;
    ///
    /// let atc = AdaptiveThresholdCalibration::new();
    /// let thresholds = GwtThresholds::from_atc(&atc, Domain::Code)?;
    /// assert!(thresholds.gate > 0.80); // Code domain is strict
    /// ```
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> CoreResult<Self> {
        let domain_thresholds = atc.get_domain_thresholds(domain).ok_or_else(|| {
            CoreError::ConfigError(format!(
                "ATC missing domain thresholds for {:?}. \
                Ensure AdaptiveThresholdCalibration is properly initialized.",
                domain
            ))
        })?;

        let gwt = Self {
            gate: domain_thresholds.theta_gate,
            hypersync: domain_thresholds.theta_hypersync,
            fragmentation: domain_thresholds.theta_fragmentation,
        };

        if !gwt.is_valid() {
            return Err(CoreError::ValidationError {
                field: "GwtThresholds".to_string(),
                message: format!(
                    "Invalid thresholds from ATC domain {:?}: gate={}, hypersync={}, fragmentation={}. \
                    Required: fragmentation < gate < hypersync, and values within constitution ranges.",
                    domain, gwt.gate, gwt.hypersync, gwt.fragmentation
                ),
            });
        }

        Ok(gwt)
    }

    /// Create with legacy General domain defaults.
    ///
    /// These values MUST match the old hardcoded constants for backwards compatibility:
    /// - GW_THRESHOLD = 0.70
    /// - HYPERSYNC_THRESHOLD = 0.95
    /// - FRAGMENTATION_THRESHOLD = 0.50
    ///
    /// # Important
    ///
    /// Use this method when:
    /// - No ATC is available
    /// - Domain context is unknown
    /// - Legacy behavior must be preserved
    ///
    /// For domain-aware behavior, use [`from_atc`](Self::from_atc) instead.
    #[inline]
    pub fn default_general() -> Self {
        Self {
            gate: 0.70,
            hypersync: 0.95,
            fragmentation: 0.50,
        }
    }

    /// Validate thresholds are within constitution ranges and logically consistent.
    ///
    /// # Validation Rules
    ///
    /// 1. Range checks per constitution:
    ///    - gate: [0.65, 0.95]
    ///    - hypersync: [0.90, 0.99]
    ///    - fragmentation: [0.35, 0.65]
    ///
    /// 2. Monotonicity constraint:
    ///    - fragmentation < gate < hypersync
    ///
    /// # Returns
    ///
    /// `true` if all constraints are satisfied, `false` otherwise.
    pub fn is_valid(&self) -> bool {
        // Range checks per constitution
        if !(0.65..=0.95).contains(&self.gate) {
            return false;
        }
        if !(0.90..=0.99).contains(&self.hypersync) {
            return false;
        }
        if !(0.35..=0.65).contains(&self.fragmentation) {
            return false;
        }

        // Monotonicity constraint: fragmentation < gate < hypersync
        if self.fragmentation >= self.gate || self.gate >= self.hypersync {
            return false;
        }

        true
    }

    /// Check if order parameter should trigger GW broadcast.
    ///
    /// Broadcast occurs when order_param >= gate.
    ///
    /// # Arguments
    ///
    /// * `order_param` - Kuramoto order parameter r ∈ [0, 1]
    ///
    /// # Returns
    ///
    /// `true` if broadcast should occur, `false` otherwise.
    #[inline]
    pub fn should_broadcast(&self, order_param: f32) -> bool {
        order_param >= self.gate
    }

    /// Check if in hypersync (pathological) state.
    ///
    /// Hypersync is a pathological state where synchronization is too high,
    /// potentially indicating seizure-like activity or lock-in.
    ///
    /// # Arguments
    ///
    /// * `order_param` - Kuramoto order parameter r ∈ [0, 1]
    ///
    /// # Returns
    ///
    /// `true` if in hypersync state (order_param > hypersync), `false` otherwise.
    #[inline]
    pub fn is_hypersync(&self, order_param: f32) -> bool {
        order_param > self.hypersync
    }

    /// Check if in fragmented state.
    ///
    /// Fragmentation indicates insufficient synchronization for coherent processing.
    ///
    /// # Arguments
    ///
    /// * `order_param` - Kuramoto order parameter r ∈ [0, 1]
    ///
    /// # Returns
    ///
    /// `true` if fragmented (order_param < fragmentation), `false` otherwise.
    #[inline]
    pub fn is_fragmented(&self, order_param: f32) -> bool {
        order_param < self.fragmentation
    }
}

impl Default for GwtThresholds {
    /// Returns legacy General domain defaults.
    ///
    /// Equivalent to [`default_general()`](Self::default_general).
    fn default() -> Self {
        Self::default_general()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================
    // LEGACY VALUE COMPATIBILITY TESTS
    // ========================================================

    #[test]
    fn test_default_matches_legacy_constants() {
        let t = GwtThresholds::default_general();

        // These MUST match the old hardcoded values EXACTLY
        assert_eq!(
            t.gate, 0.70,
            "gate must match GW_THRESHOLD (0.70), got {}",
            t.gate
        );
        assert_eq!(
            t.hypersync, 0.95,
            "hypersync must match HYPERSYNC_THRESHOLD (0.95), got {}",
            t.hypersync
        );
        assert_eq!(
            t.fragmentation, 0.50,
            "fragmentation must match FRAGMENTATION_THRESHOLD (0.50), got {}",
            t.fragmentation
        );

        println!("[VERIFIED] default_general() matches legacy constants:");
        println!("  gate: {} == 0.70 (GW_THRESHOLD)", t.gate);
        println!("  hypersync: {} == 0.95 (HYPERSYNC_THRESHOLD)", t.hypersync);
        println!(
            "  fragmentation: {} == 0.50 (FRAGMENTATION_THRESHOLD)",
            t.fragmentation
        );
    }

    #[test]
    fn test_default_is_valid() {
        let t = GwtThresholds::default_general();
        assert!(
            t.is_valid(),
            "default_general() must produce valid thresholds"
        );
        println!("[VERIFIED] default_general() produces valid thresholds");
    }

    #[test]
    fn test_default_trait_matches_default_general() {
        let default_trait = GwtThresholds::default();
        let default_general = GwtThresholds::default_general();

        assert_eq!(
            default_trait, default_general,
            "Default trait must match default_general()"
        );
        println!("[VERIFIED] Default trait == default_general()");
    }

    // ========================================================
    // ATC INTEGRATION TESTS
    // ========================================================

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
            let result = GwtThresholds::from_atc(&atc, domain);
            assert!(
                result.is_ok(),
                "Domain {:?} should produce valid thresholds, got error: {:?}",
                domain,
                result.err()
            );

            let t = result.unwrap();
            assert!(
                t.is_valid(),
                "Domain {:?} thresholds should be valid: gate={}, hypersync={}, frag={}",
                domain,
                t.gate,
                t.hypersync,
                t.fragmentation
            );
        }
        println!("[VERIFIED] All 6 domains produce valid GwtThresholds from ATC");
    }

    #[test]
    fn test_domain_strictness_ordering() {
        let atc = AdaptiveThresholdCalibration::new();

        let code = GwtThresholds::from_atc(&atc, Domain::Code).unwrap();
        let creative = GwtThresholds::from_atc(&atc, Domain::Creative).unwrap();
        let medical = GwtThresholds::from_atc(&atc, Domain::Medical).unwrap();
        let general = GwtThresholds::from_atc(&atc, Domain::General).unwrap();

        // Stricter domains have HIGHER gate (harder to broadcast)
        assert!(
            code.gate > creative.gate,
            "Code gate ({}) should be > Creative gate ({})",
            code.gate,
            creative.gate
        );
        assert!(
            medical.gate > general.gate,
            "Medical gate ({}) should be > General gate ({})",
            medical.gate,
            general.gate
        );

        // Medical is strictest (strictness=1.0)
        assert!(
            medical.gate >= code.gate,
            "Medical gate ({}) should be >= Code gate ({})",
            medical.gate,
            code.gate
        );

        println!("[VERIFIED] Domain strictness ordering:");
        println!(
            "  Medical({}) >= Code({}) > General({}) > Creative({})",
            medical.gate, code.gate, general.gate, creative.gate
        );
    }

    #[test]
    fn test_print_all_domain_thresholds() {
        println!("\n=== ATC Domain GWT Thresholds ===\n");

        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Medical,
            Domain::Code,
            Domain::Legal,
            Domain::General,
            Domain::Research,
            Domain::Creative,
        ] {
            let t = GwtThresholds::from_atc(&atc, domain).unwrap();
            println!(
                "{:?} (strictness={:.1}): gate={:.3}, hypersync={:.3}, frag={:.3}",
                domain,
                domain.strictness(),
                t.gate,
                t.hypersync,
                t.fragmentation
            );
        }

        println!("\nLegacy defaults: gate=0.70, hypersync=0.95, frag=0.50");
    }

    // ========================================================
    // BOUNDARY CONDITION TESTS
    // ========================================================

    #[test]
    fn test_should_broadcast_boundary() {
        let t = GwtThresholds::default_general();

        // Just below gate (0.70)
        assert!(
            !t.should_broadcast(0.69),
            "r=0.69 should NOT trigger broadcast (gate=0.70)"
        );
        assert!(
            !t.should_broadcast(0.699),
            "r=0.699 should NOT trigger broadcast (gate=0.70)"
        );

        // At gate boundary
        assert!(
            t.should_broadcast(0.70),
            "r=0.70 SHOULD trigger broadcast (gate=0.70)"
        );

        // Above gate
        assert!(
            t.should_broadcast(0.71),
            "r=0.71 SHOULD trigger broadcast (gate=0.70)"
        );
        assert!(
            t.should_broadcast(0.90),
            "r=0.90 SHOULD trigger broadcast (gate=0.70)"
        );

        println!("[VERIFIED] should_broadcast boundary at gate=0.70");
    }

    #[test]
    fn test_is_hypersync_boundary() {
        let t = GwtThresholds::default_general();

        // Below hypersync (0.95)
        assert!(
            !t.is_hypersync(0.94),
            "r=0.94 should NOT be hypersync (threshold=0.95)"
        );

        // At hypersync boundary - EXACTLY at threshold is NOT hypersync
        assert!(
            !t.is_hypersync(0.95),
            "r=0.95 should NOT be hypersync (must be > 0.95)"
        );

        // Above hypersync
        assert!(
            t.is_hypersync(0.951),
            "r=0.951 SHOULD be hypersync (threshold=0.95)"
        );
        assert!(
            t.is_hypersync(0.96),
            "r=0.96 SHOULD be hypersync (threshold=0.95)"
        );
        assert!(
            t.is_hypersync(1.0),
            "r=1.0 SHOULD be hypersync (threshold=0.95)"
        );

        println!("[VERIFIED] is_hypersync boundary at hypersync=0.95 (strictly greater than)");
    }

    #[test]
    fn test_is_fragmented_boundary() {
        let t = GwtThresholds::default_general();

        // Below fragmentation (0.50) - is fragmented
        assert!(
            t.is_fragmented(0.49),
            "r=0.49 SHOULD be fragmented (threshold=0.50)"
        );
        assert!(
            t.is_fragmented(0.30),
            "r=0.30 SHOULD be fragmented (threshold=0.50)"
        );

        // At fragmentation boundary - EXACTLY at threshold is NOT fragmented
        assert!(
            !t.is_fragmented(0.50),
            "r=0.50 should NOT be fragmented (must be < 0.50)"
        );

        // Above fragmentation
        assert!(
            !t.is_fragmented(0.51),
            "r=0.51 should NOT be fragmented (threshold=0.50)"
        );

        println!("[VERIFIED] is_fragmented boundary at fragmentation=0.50 (strictly less than)");
    }

    // ========================================================
    // VALIDATION TESTS
    // ========================================================

    #[test]
    fn test_invalid_gate_out_of_range() {
        // Gate below minimum (0.65)
        let t1 = GwtThresholds {
            gate: 0.60,
            hypersync: 0.95,
            fragmentation: 0.50,
        };
        assert!(!t1.is_valid(), "gate=0.60 below min 0.65 should fail");

        // Gate above maximum (0.95)
        let t2 = GwtThresholds {
            gate: 0.96,
            hypersync: 0.98,
            fragmentation: 0.50,
        };
        assert!(!t2.is_valid(), "gate=0.96 above max 0.95 should fail");
    }

    #[test]
    fn test_invalid_hypersync_out_of_range() {
        // Hypersync below minimum (0.90)
        let t1 = GwtThresholds {
            gate: 0.75,
            hypersync: 0.85,
            fragmentation: 0.50,
        };
        assert!(!t1.is_valid(), "hypersync=0.85 below min 0.90 should fail");

        // Hypersync above maximum (0.99)
        let t2 = GwtThresholds {
            gate: 0.75,
            hypersync: 1.0,
            fragmentation: 0.50,
        };
        assert!(!t2.is_valid(), "hypersync=1.0 above max 0.99 should fail");
    }

    #[test]
    fn test_invalid_fragmentation_out_of_range() {
        // Fragmentation below minimum (0.35)
        let t1 = GwtThresholds {
            gate: 0.75,
            hypersync: 0.95,
            fragmentation: 0.30,
        };
        assert!(
            !t1.is_valid(),
            "fragmentation=0.30 below min 0.35 should fail"
        );

        // Fragmentation above maximum (0.65)
        let t2 = GwtThresholds {
            gate: 0.75,
            hypersync: 0.95,
            fragmentation: 0.70,
        };
        assert!(
            !t2.is_valid(),
            "fragmentation=0.70 above max 0.65 should fail"
        );
    }

    #[test]
    fn test_invalid_monotonicity() {
        // fragmentation >= gate
        let t1 = GwtThresholds {
            gate: 0.70,
            hypersync: 0.95,
            fragmentation: 0.70,
        };
        assert!(!t1.is_valid(), "fragmentation >= gate should fail");

        // gate >= hypersync
        let t2 = GwtThresholds {
            gate: 0.95,
            hypersync: 0.95,
            fragmentation: 0.50,
        };
        assert!(!t2.is_valid(), "gate >= hypersync should fail");
    }

    // ========================================================
    // FULL STATE VERIFICATION (FSV) TEST
    // ========================================================

    #[test]
    fn test_fsv_threshold_verification() {
        println!("\n=== FSV: GWT Threshold Verification ===\n");

        // 1. Verify default_general matches legacy
        let default = GwtThresholds::default_general();
        println!("Default General Thresholds:");
        println!("  gate: {} (expected: 0.70)", default.gate);
        println!("  hypersync: {} (expected: 0.95)", default.hypersync);
        println!("  fragmentation: {} (expected: 0.50)", default.fragmentation);
        assert_eq!(default.gate, 0.70);
        assert_eq!(default.hypersync, 0.95);
        assert_eq!(default.fragmentation, 0.50);
        println!("  [VERIFIED] Default matches legacy constants\n");

        // 2. Verify ATC retrieval for all domains
        let atc = AdaptiveThresholdCalibration::new();
        println!("ATC Domain Thresholds:");
        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Creative,
            Domain::General,
        ] {
            let t = GwtThresholds::from_atc(&atc, domain).unwrap();
            println!(
                "  {:?}: gate={:.3}, hypersync={:.3}, frag={:.3}",
                domain, t.gate, t.hypersync, t.fragmentation
            );
            assert!(t.is_valid());
        }
        println!("  [VERIFIED] All domains produce valid thresholds\n");

        // 3. Boundary tests with state printout
        println!("Boundary Tests:");
        let t = GwtThresholds::default_general();

        let test_cases: [(f32, &str, bool, bool); 6] = [
            (0.69, "should_broadcast", t.should_broadcast(0.69), false),
            (0.70, "should_broadcast", t.should_broadcast(0.70), true),
            (0.95, "is_hypersync", t.is_hypersync(0.95), false),
            (0.96, "is_hypersync", t.is_hypersync(0.96), true),
            (0.49, "is_fragmented", t.is_fragmented(0.49), true),
            (0.50, "is_fragmented", t.is_fragmented(0.50), false),
        ];

        for (r, method, actual, expected) in test_cases {
            println!(
                "  r={:.2}, {}() = {} (expected: {})",
                r, method, actual, expected
            );
            assert_eq!(actual, expected, "Failed for r={} on {}", r, method);
        }
        println!("  [VERIFIED] All boundary conditions correct\n");

        println!("=== FSV COMPLETE: All verifications passed ===\n");
    }
}
