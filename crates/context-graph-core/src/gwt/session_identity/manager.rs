//! SessionIdentityManager - Cross-session identity continuity.
//!
//! # TASK-SESSION-06
//!
//! Implements IDENTITY-001 formula: IC = cos(PV_current, PV_previous) * r(current)
//!
//! # Constitution Reference
//! - IDENTITY-001: IC formula
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - AP-39: cosine_similarity_13d MUST be public (already exported from ego_node)
//! - AP-25: Kuramoto MUST have exactly 13 oscillators
//!
//! # FAIL FAST Policy
//! All storage errors propagate immediately with detailed logging.
//! No silent failures or default fallbacks.
//!
//! # Architecture
//! This module defines the trait in `context-graph-core`. The concrete
//! `StandaloneSessionIdentityManager` is implemented in `context-graph-storage`
//! to avoid circular dependencies (storage already depends on core).

use crate::error::CoreResult;
use crate::gwt::cosine_similarity_13d; // AP-39: Already public

use super::{SessionIdentitySnapshot, KURAMOTO_N};

/// Trait for session identity management.
///
/// Implementors MUST:
/// - Be Send + Sync for thread safety
/// - Fail fast on all errors (no silent defaults)
/// - Log all operations with tracing
///
/// # Implementation Note
/// The concrete implementation `StandaloneSessionIdentityManager` is in
/// `context-graph-storage::rocksdb_backend::session_identity_manager` to
/// avoid circular dependency (storage depends on core for SessionIdentitySnapshot).
pub trait SessionIdentityManager: Send + Sync {
    /// Capture current GWT state into a snapshot.
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier
    ///
    /// # Returns
    /// * `Ok(snapshot)` - Successfully captured state
    /// * `Err(CoreError::Internal)` - Failed to read GWT state
    ///
    /// # FAIL FAST
    /// Any state access failure returns error immediately.
    fn capture_snapshot(&self, session_id: &str) -> CoreResult<SessionIdentitySnapshot>;

    /// Restore identity from storage, compute cross-session IC.
    ///
    /// # Arguments
    /// * `target_session` - Specific session to restore, or None for latest
    ///
    /// # Returns
    /// * `Ok((snapshot, ic))` - Restored snapshot and computed IC
    /// * `Err(CoreError::StorageError)` - Storage read failed
    ///
    /// # First Session Behavior
    /// When no previous session exists, returns IC = 1.0 and creates
    /// a new snapshot from current GWT state.
    ///
    /// # Side Effects
    /// Updates IdentityCache with restored values.
    fn restore_identity(
        &self,
        target_session: Option<&str>,
    ) -> CoreResult<(SessionIdentitySnapshot, f32)>;

    /// Compute cross-session IC using IDENTITY-001 formula.
    ///
    /// Formula: `IC = cos(PV_current, PV_previous) * r(current)`
    ///
    /// # Arguments
    /// * `current` - Current session's snapshot
    /// * `previous` - Previous session's snapshot
    ///
    /// # Returns
    /// IC value in range [0.0, 1.0] (clamped)
    ///
    /// # Edge Cases
    /// - Zero purpose vector: Returns 0.0
    /// - Identical vectors: Returns r(current)
    fn compute_cross_session_ic(
        &self,
        current: &SessionIdentitySnapshot,
        previous: &SessionIdentitySnapshot,
    ) -> f32;
}

/// Compute Kuramoto order parameter r from oscillator phases.
///
/// Formula: r = |Σ exp(iθⱼ)| / N
///
/// # Arguments
/// * `phases` - Array of 13 oscillator phases in radians
///
/// # Returns
/// Order parameter r in [0.0, 1.0]
/// - r ≈ 0: No synchronization (phases random)
/// - r ≈ 1: Full synchronization (phases aligned)
///
/// # Constitution Reference
/// gwt.kuramoto.order_param: "r·e^(iψ) = (1/N)Σⱼ e^(iθⱼ)"
pub fn compute_kuramoto_r(phases: &[f64; KURAMOTO_N]) -> f32 {
    let (sum_sin, sum_cos) = phases.iter().fold((0.0_f64, 0.0_f64), |(s, c), &theta| {
        (s + theta.sin(), c + theta.cos())
    });

    let n = KURAMOTO_N as f64;
    let magnitude = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();

    // Clamp to [0, 1] to handle floating point errors
    magnitude.clamp(0.0, 1.0) as f32
}

/// Compute cross-session IC using IDENTITY-001 formula.
///
/// This is a standalone function that can be used without a manager instance.
/// Formula: `IC = cos(PV_current, PV_previous) * r(current)`
///
/// # Arguments
/// * `current` - Current session's snapshot
/// * `previous` - Previous session's snapshot
///
/// # Returns
/// IC value in range [0.0, 1.0] (clamped)
pub fn compute_ic(current: &SessionIdentitySnapshot, previous: &SessionIdentitySnapshot) -> f32 {
    // IDENTITY-001: IC = cos(PV_current, PV_previous) * r(current)

    // 1. Compute cosine similarity between purpose vectors
    let cos_sim = cosine_similarity_13d(&current.purpose_vector, &previous.purpose_vector);

    // 2. Compute Kuramoto order parameter r from current phases
    let r = compute_kuramoto_r(&current.kuramoto_phases);

    // 3. Apply formula and clamp
    (cos_sim * r).clamp(0.0, 1.0)
}

// =============================================================================
// IDENTITY-002 Classification Functions
// =============================================================================

/// Classify IC value per IDENTITY-002 constitution thresholds.
///
/// # Thresholds (IDENTITY-002)
/// - Healthy: IC >= 0.9
/// - Good: 0.7 <= IC < 0.9
/// - Warning: 0.5 <= IC < 0.7
/// - Degraded: IC < 0.5
///
/// # Returns
/// Static string label (no allocation)
#[inline]
pub fn classify_ic(ic: f32) -> &'static str {
    match ic {
        ic if ic >= 0.9 => "Healthy",
        ic if ic >= 0.7 => "Good",
        ic if ic >= 0.5 => "Warning",
        _ => "Degraded",
    }
}

/// Returns true if IC indicates identity crisis (< 0.5).
///
/// Per AP-26 and AP-38: IC < 0.5 MUST trigger automatic dream consolidation.
///
/// # Constitution Reference
/// - AP-26: IC<0.5 MUST trigger dream - no silent failures
/// - AP-38: IC<0.5 MUST auto-trigger dream
#[inline]
pub fn is_ic_crisis(ic: f32) -> bool {
    ic < 0.5
}

/// Returns true if IC is in warning range (0.5 <= IC < 0.7).
///
/// Per IDENTITY-002: This range indicates degraded identity but not crisis.
#[inline]
pub fn is_ic_warning(ic: f32) -> bool {
    ic >= 0.5 && ic < 0.7
}

/// Classify Kuramoto order parameter r.
///
/// # Thresholds (gwt.kuramoto.thresholds in constitution)
/// - Good synchronization: r >= 0.8
/// - Partial synchronization: 0.5 <= r < 0.8
/// - Fragmented: r < 0.5
///
/// # Returns
/// Static string description
#[inline]
pub fn classify_sync(r: f64) -> &'static str {
    match r {
        r if r >= 0.8 => "Good synchronization",
        r if r >= 0.5 => "Partial synchronization",
        _ => "Fragmented",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TC-SESSION-MGR-01: Identical Purpose Vectors
    // =========================================================================
    #[test]
    fn tc_session_mgr_01_identical_purpose_vectors() {
        println!("\n=== TC-SESSION-MGR-01: Identical Purpose Vectors ===");

        // SOURCE OF TRUTH: IDENTITY-001 formula
        // IC = cos(identical) * r = 1.0 * r

        let pv = [0.5_f32; KURAMOTO_N]; // Non-trivial values

        let mut current = SessionIdentitySnapshot::new("current-session");
        let mut previous = SessionIdentitySnapshot::new("previous-session");

        current.purpose_vector = pv;
        previous.purpose_vector = pv;
        current.kuramoto_phases = [0.0; KURAMOTO_N]; // Aligned phases -> r = 1.0

        println!("BEFORE:");
        println!("  current.purpose_vector: {:?}", &current.purpose_vector[..3]);
        println!(
            "  previous.purpose_vector: {:?}",
            &previous.purpose_vector[..3]
        );
        println!("  current.kuramoto_phases: all aligned at 0.0");

        // EXECUTE
        let ic = compute_ic(&current, &previous);

        println!("AFTER:");
        println!("  computed IC: {}", ic);

        // VERIFY: cos(identical) = 1.0, r = 1.0, so IC = 1.0
        assert!(
            (ic - 1.0).abs() < 0.01,
            "IC for identical vectors should be ~1.0, got {}",
            ic
        );

        println!("RESULT: PASS - Identical purpose vectors produce IC ≈ 1.0");
    }

    // =========================================================================
    // TC-SESSION-MGR-02: Orthogonal Purpose Vectors
    // =========================================================================
    #[test]
    fn tc_session_mgr_02_orthogonal_purpose_vectors() {
        println!("\n=== TC-SESSION-MGR-02: Orthogonal Purpose Vectors ===");

        // SOURCE OF TRUTH: IDENTITY-001 formula
        // IC = cos(orthogonal) * r = 0.0 * r = 0.0

        let mut current = SessionIdentitySnapshot::new("current-session");
        let mut previous = SessionIdentitySnapshot::new("previous-session");

        // Create orthogonal vectors
        current.purpose_vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        previous.purpose_vector =
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        current.kuramoto_phases = [0.0; KURAMOTO_N]; // r = 1.0 (doesn't matter since cos=0)

        println!("BEFORE:");
        println!("  current.purpose_vector[0]: {}", current.purpose_vector[0]);
        println!(
            "  previous.purpose_vector[1]: {}",
            previous.purpose_vector[1]
        );

        // EXECUTE
        let ic = compute_ic(&current, &previous);

        println!("AFTER:");
        println!("  computed IC: {}", ic);

        // VERIFY: cos(orthogonal) = 0.0, so IC ≈ 0.0
        assert!(
            ic.abs() < 0.01,
            "IC for orthogonal vectors should be ~0.0, got {}",
            ic
        );

        println!("RESULT: PASS - Orthogonal purpose vectors produce IC ≈ 0.0");
    }

    // =========================================================================
    // TC-SESSION-MGR-04: Kuramoto r Computation
    // =========================================================================
    #[test]
    fn tc_session_mgr_04_kuramoto_r_computation() {
        println!("\n=== TC-SESSION-MGR-04: Kuramoto r Computation ===");

        // SOURCE OF TRUTH: Constitution gwt.kuramoto.order_param
        // r = |Σ exp(iθⱼ)| / N

        // Case 1: Aligned phases -> r ≈ 1.0
        let aligned_phases = [0.0_f64; KURAMOTO_N];
        let r_aligned = compute_kuramoto_r(&aligned_phases);
        println!("Aligned phases (all 0): r = {:.4}", r_aligned);
        assert!(r_aligned > 0.99, "Aligned phases should give r ≈ 1.0");

        // Case 2: Evenly distributed phases -> r ≈ 0
        let mut distributed_phases = [0.0_f64; KURAMOTO_N];
        for i in 0..KURAMOTO_N {
            distributed_phases[i] = 2.0 * std::f64::consts::PI * (i as f64) / (KURAMOTO_N as f64);
        }
        let r_distributed = compute_kuramoto_r(&distributed_phases);
        println!("Distributed phases: r = {:.4}", r_distributed);
        assert!(
            r_distributed < 0.15,
            "Distributed phases should give r ≈ 0, got {}",
            r_distributed
        );

        // Case 3: Partially aligned -> 0 < r < 1
        let mut partial_phases = [0.0_f64; KURAMOTO_N];
        for i in KURAMOTO_N / 2..KURAMOTO_N {
            partial_phases[i] = std::f64::consts::PI; // Half opposite
        }
        let r_partial = compute_kuramoto_r(&partial_phases);
        println!("Partial alignment: r = {:.4}", r_partial);
        assert!(
            r_partial > 0.0 && r_partial < 1.0,
            "Partial should be between 0 and 1"
        );

        println!("RESULT: PASS - Kuramoto r computed correctly for all cases");
    }

    // =========================================================================
    // TC-SESSION-MGR-08: Zero Purpose Vector Edge Case
    // =========================================================================
    #[test]
    fn tc_session_mgr_08_zero_purpose_vector() {
        println!("\n=== TC-SESSION-MGR-08: Zero Purpose Vector Edge Case ===");

        let mut current = SessionIdentitySnapshot::new("current");
        let mut previous = SessionIdentitySnapshot::new("previous");

        // Zero vectors
        current.purpose_vector = [0.0; KURAMOTO_N];
        previous.purpose_vector = [0.0; KURAMOTO_N];
        current.kuramoto_phases = [0.0; KURAMOTO_N];

        let ic = compute_ic(&current, &previous);

        println!("IC with zero vectors: {}", ic);

        // cosine_similarity_13d returns 0.0 for zero vectors
        assert!(ic.abs() < 0.01, "IC should be 0.0 for zero vectors");

        println!("RESULT: PASS - Zero purpose vectors handled correctly");
    }

    // =========================================================================
    // TC-SESSION-MGR-05: IC Varies with Kuramoto r
    // =========================================================================
    #[test]
    fn tc_session_mgr_05_ic_varies_with_kuramoto_r() {
        println!("\n=== TC-SESSION-MGR-05: IC Varies with Kuramoto r ===");

        // Same purpose vectors, different Kuramoto synchronization
        let pv = [0.5_f32; KURAMOTO_N];

        let mut current = SessionIdentitySnapshot::new("current");
        let mut previous = SessionIdentitySnapshot::new("previous");

        current.purpose_vector = pv;
        previous.purpose_vector = pv;

        // High synchronization (r ≈ 1.0)
        current.kuramoto_phases = [0.0; KURAMOTO_N];
        let ic_high_sync = compute_ic(&current, &previous);
        println!("IC with high sync (r≈1.0): {}", ic_high_sync);

        // Lower synchronization (r < 1.0)
        let mut low_sync_phases = [0.0_f64; KURAMOTO_N];
        for i in 0..KURAMOTO_N / 2 {
            low_sync_phases[i] = std::f64::consts::PI / 4.0;
        }
        current.kuramoto_phases = low_sync_phases;
        let r_low = compute_kuramoto_r(&current.kuramoto_phases);
        let ic_low_sync = compute_ic(&current, &previous);
        println!(
            "IC with lower sync (r={:.3}): {}",
            r_low, ic_low_sync
        );

        // VERIFY: IC decreases with lower synchronization
        assert!(
            ic_high_sync > ic_low_sync,
            "IC should be higher with better synchronization"
        );

        println!("RESULT: PASS - IC correctly varies with Kuramoto r");
    }

    // =========================================================================
    // TC-SESSION-09: classify_ic() Threshold Boundaries
    // =========================================================================
    #[test]
    fn tc_session_09_classify_ic_threshold_boundaries() {
        println!("\n=== TC-SESSION-09: classify_ic() Threshold Boundaries ===");

        // SOURCE OF TRUTH: IDENTITY-002 from constitution.yaml
        // Healthy: IC >= 0.9
        // Good: 0.7 <= IC < 0.9
        // Warning: 0.5 <= IC < 0.7
        // Degraded: IC < 0.5

        let test_cases: Vec<(f32, &str)> = vec![
            // Healthy boundary
            (1.0, "Healthy"),
            (0.95, "Healthy"),
            (0.90, "Healthy"),
            // Good boundary
            (0.899, "Good"),
            (0.75, "Good"),
            (0.70, "Good"),
            // Warning boundary
            (0.699, "Warning"),
            (0.55, "Warning"),
            (0.50, "Warning"),
            // Degraded boundary
            (0.499, "Degraded"),
            (0.25, "Degraded"),
            (0.0, "Degraded"),
            // Edge case: negative (clamped)
            (-0.1, "Degraded"),
        ];

        for (ic, expected) in test_cases {
            let actual = classify_ic(ic);
            println!("classify_ic({:.3}) = '{}' (expected: '{}')", ic, actual, expected);
            assert_eq!(
                actual, expected,
                "classify_ic({}) should return '{}', got '{}'",
                ic, expected, actual
            );
        }

        println!("RESULT: PASS - All threshold boundaries correct");
    }

    #[test]
    fn tc_session_09_is_ic_crisis() {
        println!("\n=== TC-SESSION-09: is_ic_crisis() ===");

        // SOURCE OF TRUTH: AP-26, AP-38 - crisis is IC < 0.5

        // Crisis cases (IC < 0.5)
        assert!(is_ic_crisis(0.0), "0.0 should be crisis");
        assert!(is_ic_crisis(0.25), "0.25 should be crisis");
        assert!(is_ic_crisis(0.49), "0.49 should be crisis");
        assert!(is_ic_crisis(0.499), "0.499 should be crisis");

        // Non-crisis cases (IC >= 0.5)
        assert!(!is_ic_crisis(0.5), "0.5 should NOT be crisis");
        assert!(!is_ic_crisis(0.50), "0.50 should NOT be crisis");
        assert!(!is_ic_crisis(0.51), "0.51 should NOT be crisis");
        assert!(!is_ic_crisis(0.7), "0.7 should NOT be crisis");
        assert!(!is_ic_crisis(1.0), "1.0 should NOT be crisis");

        println!("RESULT: PASS - is_ic_crisis() boundary correct at 0.5");
    }

    #[test]
    fn tc_session_09_is_ic_warning() {
        println!("\n=== TC-SESSION-09: is_ic_warning() ===");

        // SOURCE OF TRUTH: IDENTITY-002 - warning is 0.5 <= IC < 0.7

        // Below warning (crisis range)
        assert!(!is_ic_warning(0.49), "0.49 should NOT be warning (is crisis)");
        assert!(!is_ic_warning(0.0), "0.0 should NOT be warning (is crisis)");

        // Warning range
        assert!(is_ic_warning(0.5), "0.5 should be warning");
        assert!(is_ic_warning(0.55), "0.55 should be warning");
        assert!(is_ic_warning(0.69), "0.69 should be warning");

        // Above warning (good or healthy range)
        assert!(!is_ic_warning(0.7), "0.7 should NOT be warning (is good)");
        assert!(!is_ic_warning(0.71), "0.71 should NOT be warning");
        assert!(!is_ic_warning(0.9), "0.9 should NOT be warning (is healthy)");
        assert!(!is_ic_warning(1.0), "1.0 should NOT be warning");

        println!("RESULT: PASS - is_ic_warning() boundaries correct at 0.5 and 0.7");
    }

    #[test]
    fn tc_session_09_classify_sync() {
        println!("\n=== TC-SESSION-09: classify_sync() ===");

        // SOURCE OF TRUTH: constitution.yaml gwt.kuramoto.thresholds
        // coherent: r >= 0.8
        // fragmented: r < 0.5

        // Good synchronization
        assert_eq!(classify_sync(1.0), "Good synchronization");
        assert_eq!(classify_sync(0.9), "Good synchronization");
        assert_eq!(classify_sync(0.8), "Good synchronization");

        // Partial synchronization
        assert_eq!(classify_sync(0.79), "Partial synchronization");
        assert_eq!(classify_sync(0.6), "Partial synchronization");
        assert_eq!(classify_sync(0.5), "Partial synchronization");

        // Fragmented
        assert_eq!(classify_sync(0.49), "Fragmented");
        assert_eq!(classify_sync(0.3), "Fragmented");
        assert_eq!(classify_sync(0.0), "Fragmented");

        println!("RESULT: PASS - classify_sync() thresholds correct");
    }
}
