//! Global Workspace State and Consciousness State Machine
//!
//! Implements Global Workspace Theory (GWT) state management.

use serde::{Deserialize, Serialize};

#[allow(deprecated)]
use super::constants::{FRAGMENTATION_THRESHOLD, HYPERSYNC_THRESHOLD};
use super::thresholds::GwtThresholds;

/// Global Workspace state for GWT implementation.
///
/// The Global Workspace represents the currently "conscious" content
/// that is broadcast to all subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspace {
    /// Whether the workspace is active (ignited)
    pub active: bool,
    /// Current ignition level (r from Kuramoto)
    pub ignition_level: f32,
    /// Broadcast content when ignited
    pub broadcast_content: Option<serde_json::Value>,
    /// Current consciousness state
    pub state: ConsciousnessState,
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self {
            active: false,
            ignition_level: 0.0,
            broadcast_content: None,
            state: ConsciousnessState::Dormant,
        }
    }
}

/// Consciousness state from GWT state machine.
///
/// From constitution gwt.state_machine.states
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConsciousnessState {
    /// r < 0.3, no active workspace
    Dormant,
    /// 0.3 ≤ r < 0.5, partial sync
    Fragmented,
    /// 0.5 ≤ r < 0.8, approaching coherence
    Emerging,
    /// r ≥ 0.8, unified percept active
    Conscious,
    /// r > 0.95, possibly pathological
    Hypersync,
}

impl ConsciousnessState {
    /// Determine state from order parameter r using provided thresholds.
    ///
    /// This method uses domain-aware thresholds from the ATC system to classify
    /// consciousness states. Different domains may have different thresholds
    /// based on their strictness requirements.
    ///
    /// # State Classification
    ///
    /// - `Hypersync`: r > thresholds.hypersync (pathological over-synchronization)
    /// - `Conscious`: r >= thresholds.gate (coherent workspace active)
    /// - `Emerging`: r >= thresholds.fragmentation (approaching coherence)
    /// - `Fragmented`: r >= 0.3 (partial synchronization)
    /// - `Dormant`: r < 0.3 (no active workspace)
    ///
    /// # Arguments
    ///
    /// * `r` - Kuramoto order parameter [0, 1]
    /// * `thresholds` - GWT thresholds for state boundaries
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use context_graph_core::layers::coherence::{ConsciousnessState, GwtThresholds};
    ///
    /// let thresholds = GwtThresholds::default_general();
    /// let state = ConsciousnessState::from_order_parameter_with_thresholds(0.75, &thresholds);
    /// assert_eq!(state, ConsciousnessState::Conscious);
    /// ```
    pub fn from_order_parameter_with_thresholds(r: f32, thresholds: &GwtThresholds) -> Self {
        if r > thresholds.hypersync {
            Self::Hypersync
        } else if r >= thresholds.gate {
            Self::Conscious
        } else if r >= thresholds.fragmentation {
            Self::Emerging
        } else if r >= 0.3 {
            Self::Fragmented
        } else {
            Self::Dormant
        }
    }

    /// Determine state from order parameter r using legacy default thresholds.
    ///
    /// # Deprecation Notice
    ///
    /// This method is deprecated. Use [`from_order_parameter_with_thresholds`](Self::from_order_parameter_with_thresholds)
    /// with explicit [`GwtThresholds`] for domain-aware behavior.
    ///
    /// # Legacy Thresholds Used
    ///
    /// - hypersync: 0.95 (HYPERSYNC_THRESHOLD)
    /// - gate: 0.80 (hardcoded)
    /// - fragmentation: 0.50 (FRAGMENTATION_THRESHOLD)
    #[deprecated(
        since = "0.5.0",
        note = "Use from_order_parameter_with_thresholds with GwtThresholds instead"
    )]
    #[allow(deprecated)]
    pub fn from_order_parameter(r: f32) -> Self {
        // Note: This uses 0.8 for gate (hardcoded in original) not 0.7 (GW_THRESHOLD)
        // This is the original behavior preserved for backwards compatibility
        if r > HYPERSYNC_THRESHOLD {
            Self::Hypersync
        } else if r >= 0.8 {
            Self::Conscious
        } else if r >= FRAGMENTATION_THRESHOLD {
            Self::Emerging
        } else if r >= 0.3 {
            Self::Fragmented
        } else {
            Self::Dormant
        }
    }

    /// Check if this is a healthy state (not Dormant or Hypersync).
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Fragmented | Self::Emerging | Self::Conscious)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================
    // NEW API TESTS
    // ========================================================

    #[test]
    fn test_consciousness_state_with_default_thresholds() {
        let t = GwtThresholds::default_general();

        // Dormant: r < 0.3
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.1, &t),
            ConsciousnessState::Dormant
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.29, &t),
            ConsciousnessState::Dormant
        );

        // Fragmented: 0.3 <= r < fragmentation(0.50)
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.30, &t),
            ConsciousnessState::Fragmented
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.49, &t),
            ConsciousnessState::Fragmented
        );

        // Emerging: fragmentation(0.50) <= r < gate(0.70)
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.50, &t),
            ConsciousnessState::Emerging
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.69, &t),
            ConsciousnessState::Emerging
        );

        // Conscious: gate(0.70) <= r <= hypersync(0.95)
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.70, &t),
            ConsciousnessState::Conscious
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.85, &t),
            ConsciousnessState::Conscious
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.95, &t),
            ConsciousnessState::Conscious
        );

        // Hypersync: r > hypersync(0.95)
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.96, &t),
            ConsciousnessState::Hypersync
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter_with_thresholds(0.98, &t),
            ConsciousnessState::Hypersync
        );

        println!("[VERIFIED] Consciousness state classification with GwtThresholds");
    }

    #[test]
    fn test_consciousness_state_domain_variation() {
        use crate::atc::{AdaptiveThresholdCalibration, Domain};

        let atc = AdaptiveThresholdCalibration::new();

        // Code domain is stricter (higher gate)
        let code_t = GwtThresholds::from_atc(&atc, Domain::Code).unwrap();
        // Creative domain is looser (lower gate)
        let creative_t = GwtThresholds::from_atc(&atc, Domain::Creative).unwrap();

        // Test with r=0.80 - should be Conscious in Creative but Emerging in Code
        let r = 0.80;
        let code_state = ConsciousnessState::from_order_parameter_with_thresholds(r, &code_t);
        let creative_state =
            ConsciousnessState::from_order_parameter_with_thresholds(r, &creative_t);

        println!("r=0.80:");
        println!(
            "  Code domain (gate={:.3}): {:?}",
            code_t.gate, code_state
        );
        println!(
            "  Creative domain (gate={:.3}): {:?}",
            creative_t.gate, creative_state
        );

        // Code has higher gate, so 0.80 might still be Emerging
        // Creative has lower gate, so 0.80 should definitely be Conscious
        assert_eq!(
            creative_state,
            ConsciousnessState::Conscious,
            "r=0.80 should be Conscious in Creative domain (gate={:.3})",
            creative_t.gate
        );

        println!("[VERIFIED] Domain-aware consciousness classification");
    }

    // ========================================================
    // LEGACY API TESTS (with deprecation suppression)
    // ========================================================

    #[test]
    #[allow(deprecated)]
    fn test_consciousness_state_from_r_legacy() {
        // Test legacy API still works (with deprecation warnings suppressed)
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.1),
            ConsciousnessState::Dormant
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.35),
            ConsciousnessState::Fragmented
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.6),
            ConsciousnessState::Emerging
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.85),
            ConsciousnessState::Conscious
        );
        assert_eq!(
            ConsciousnessState::from_order_parameter(0.98),
            ConsciousnessState::Hypersync
        );
        println!("[VERIFIED] Legacy from_order_parameter() still works");
    }

    #[test]
    fn test_consciousness_state_health() {
        assert!(!ConsciousnessState::Dormant.is_healthy());
        assert!(ConsciousnessState::Fragmented.is_healthy());
        assert!(ConsciousnessState::Emerging.is_healthy());
        assert!(ConsciousnessState::Conscious.is_healthy());
        assert!(!ConsciousnessState::Hypersync.is_healthy());
        println!("[VERIFIED] Consciousness state health check");
    }

    // ========================================================
    // FSV TEST
    // ========================================================

    #[test]
    fn test_fsv_consciousness_state_classification() {
        println!("\n=== FSV: ConsciousnessState Classification ===\n");

        let t = GwtThresholds::default_general();
        println!("Using default thresholds: gate={}, hypersync={}, frag={}",
                 t.gate, t.hypersync, t.fragmentation);
        println!();

        let test_cases: [(f32, ConsciousnessState); 10] = [
            (0.10, ConsciousnessState::Dormant),
            (0.29, ConsciousnessState::Dormant),
            (0.30, ConsciousnessState::Fragmented),
            (0.49, ConsciousnessState::Fragmented),
            (0.50, ConsciousnessState::Emerging),
            (0.69, ConsciousnessState::Emerging),
            (0.70, ConsciousnessState::Conscious),
            (0.95, ConsciousnessState::Conscious),
            (0.96, ConsciousnessState::Hypersync),
            (1.00, ConsciousnessState::Hypersync),
        ];

        println!("State Boundaries:");
        for (r, expected) in &test_cases {
            let actual = ConsciousnessState::from_order_parameter_with_thresholds(*r, &t);
            println!("  r={:.2} => {:?} (expected: {:?})", r, actual, expected);
            assert_eq!(actual, *expected, "Failed for r={}", r);
        }

        println!("\n[VERIFIED] All state classifications correct");
        println!("\n=== FSV COMPLETE ===\n");
    }
}
