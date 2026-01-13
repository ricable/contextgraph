//! Crisis Detection Tests - TASK-IDENTITY-P0-004
//!
//! Tests for crisis detection functionality including:
//! - State transition detection
//! - Cooldown mechanism
//! - Helper method behavior
//!
//! NO MOCK DATA: All tests use REAL data structures and verify actual state changes.

use crate::gwt::ego_node::{
    CrisisDetectionResult, IdentityContinuityMonitor, IdentityStatus, CRISIS_EVENT_COOLDOWN,
};
use std::time::Duration;

/// Create a uniform purpose vector for testing
fn uniform_pv(val: f32) -> [f32; 13] {
    [val; 13]
}

/// Setup monitor at a specific status by computing continuity.
///
/// Uses real compute_continuity calls to set up the monitor state.
fn setup_monitor_at_status(status: IdentityStatus) -> IdentityContinuityMonitor {
    let mut monitor = IdentityContinuityMonitor::new();

    // First vector (establishes baseline, always Healthy IC=1.0)
    monitor.compute_continuity(&uniform_pv(1.0), 1.0, "baseline");

    // Second vector to set target status based on IC thresholds
    // IC = cosine * kuramoto_r (cosine=1.0 for same PV, so IC = r)
    let kuramoto_r = match status {
        IdentityStatus::Healthy => 0.95,  // IC > 0.9
        IdentityStatus::Warning => 0.80,  // 0.7 <= IC <= 0.9
        IdentityStatus::Degraded => 0.60, // 0.5 <= IC < 0.7
        IdentityStatus::Critical => 0.30, // IC < 0.5
    };

    // Same PV (cosine=1.0), so IC = 1.0 * r = r
    monitor.compute_continuity(&uniform_pv(1.0), kuramoto_r, "target_status");

    // Call detect_crisis to update previous_status
    let _ = monitor.detect_crisis();

    monitor
}

// ============================================================================
// FSV Edge Case Tests (5 Required Cases from Task Spec)
// ============================================================================

#[test]
fn test_crisis_detection_first_call_no_history() {
    println!("=== FSV: First detection with no compute_continuity ===");

    let mut monitor = IdentityContinuityMonitor::new();

    // BEFORE
    println!("BEFORE: history_len = {}", monitor.history_len());
    assert_eq!(monitor.history_len(), 0);

    // EXECUTE
    let result = monitor.detect_crisis();

    // AFTER
    println!(
        "AFTER: IC = {}, status = {:?}",
        result.identity_coherence, result.current_status
    );
    println!("AFTER: status_changed = {}", result.status_changed);

    // First call with no history should default to Healthy
    assert!(
        (result.identity_coherence - 1.0).abs() < 1e-6,
        "Expected IC=1.0, got {}",
        result.identity_coherence
    );
    assert_eq!(result.current_status, IdentityStatus::Healthy);
    assert!(!result.status_changed); // Previous was also Healthy (default)
    assert!(result.can_emit_event); // No cooldown initially

    println!("EVIDENCE: First detection correctly returns Healthy default");
}

#[test]
fn test_crisis_detection_same_status_no_change() {
    println!("=== FSV: Same status = no transition ===");

    let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

    // BEFORE
    let prev = monitor.previous_status();
    println!("BEFORE: previous_status = {:?}", prev);

    // Stay healthy (same vector, high r)
    monitor.compute_continuity(&uniform_pv(1.0), 0.95, "stay_healthy");

    // EXECUTE
    let result = monitor.detect_crisis();

    // AFTER
    println!("AFTER: current_status = {:?}", result.current_status);
    println!("AFTER: status_changed = {}", result.status_changed);

    assert!(!result.status_changed);
    assert!(!result.entering_crisis);
    assert!(!result.entering_critical);
    assert!(!result.recovering);

    println!("EVIDENCE: No status change when staying Healthy");
}

#[test]
fn test_crisis_detection_entering_crisis_from_healthy() {
    println!("=== FSV: Healthy -> Warning = entering_crisis ===");

    let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

    // BEFORE
    println!("BEFORE: previous_status = {:?}", monitor.previous_status());
    assert_eq!(monitor.previous_status(), IdentityStatus::Healthy);

    // Drop to Warning (IC ~ 0.75)
    monitor.compute_continuity(&uniform_pv(1.0), 0.75, "drop_to_warning");

    // EXECUTE
    let result = monitor.detect_crisis();

    // AFTER
    println!("AFTER: IC = {:.4}", result.identity_coherence);
    println!(
        "AFTER: previous_status = {:?}, current_status = {:?}",
        result.previous_status, result.current_status
    );
    println!(
        "AFTER: status_changed = {}, entering_crisis = {}",
        result.status_changed, result.entering_crisis
    );

    assert!(result.status_changed);
    assert!(result.entering_crisis);
    assert!(!result.entering_critical); // Warning, not Critical
    assert!(!result.recovering);
    assert_eq!(result.previous_status, IdentityStatus::Healthy);
    assert_eq!(result.current_status, IdentityStatus::Warning);

    println!("EVIDENCE: entering_crisis correctly detected");
}

#[test]
fn test_crisis_detection_entering_critical() {
    println!("=== FSV: Warning -> Critical = entering_critical ===");

    let mut monitor = setup_monitor_at_status(IdentityStatus::Warning);

    // BEFORE
    println!("BEFORE: previous_status = {:?}", monitor.previous_status());
    assert_eq!(monitor.previous_status(), IdentityStatus::Warning);

    // Drop to Critical (IC ~ 0.30)
    monitor.compute_continuity(&uniform_pv(1.0), 0.30, "drop_to_critical");

    // EXECUTE
    let result = monitor.detect_crisis();

    // AFTER
    println!("AFTER: IC = {:.4}", result.identity_coherence);
    println!("AFTER: entering_critical = {}", result.entering_critical);

    assert!(result.status_changed);
    assert!(!result.entering_crisis); // Not from Healthy
    assert!(result.entering_critical);
    assert!(!result.recovering);
    assert_eq!(result.current_status, IdentityStatus::Critical);

    println!("EVIDENCE: entering_critical correctly detected");
}

#[test]
fn test_crisis_detection_direct_to_critical() {
    println!("=== FSV: Healthy -> Critical = both flags ===");

    let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

    // Crash directly to Critical
    monitor.compute_continuity(&uniform_pv(1.0), 0.20, "crash_to_critical");

    // EXECUTE
    let result = monitor.detect_crisis();

    // AFTER
    println!(
        "AFTER: entering_crisis = {}, entering_critical = {}",
        result.entering_crisis, result.entering_critical
    );

    assert!(result.status_changed);
    assert!(result.entering_crisis); // From Healthy
    assert!(result.entering_critical); // To Critical

    println!("EVIDENCE: Direct Healthy->Critical sets both flags");
}

#[test]
fn test_crisis_detection_recovery_from_critical() {
    println!("=== FSV: Critical -> Degraded = recovering ===");

    let mut monitor = setup_monitor_at_status(IdentityStatus::Critical);

    // BEFORE
    println!("BEFORE: previous_status = {:?}", monitor.previous_status());

    // Recover to Degraded (IC ~ 0.55)
    monitor.compute_continuity(&uniform_pv(1.0), 0.55, "recovery");

    // EXECUTE
    let result = monitor.detect_crisis();

    // AFTER
    println!("AFTER: recovering = {}", result.recovering);

    assert!(result.status_changed);
    assert!(!result.entering_crisis);
    assert!(!result.entering_critical);
    assert!(result.recovering);
    assert_eq!(result.previous_status, IdentityStatus::Critical);
    assert_eq!(result.current_status, IdentityStatus::Degraded);

    println!("EVIDENCE: Recovery correctly detected");
}

// ============================================================================
// Cooldown Mechanism Tests
// ============================================================================

#[test]
fn test_crisis_detection_cooldown_initially_can_emit() {
    println!("=== FSV: No cooldown initially ===");

    let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);
    monitor.compute_continuity(&uniform_pv(1.0), 0.30, "crisis");

    let result = monitor.detect_crisis();

    assert!(result.can_emit_event);
    assert!(result.time_since_last_event.is_none());

    println!("EVIDENCE: First event can be emitted (no cooldown)");
}

#[test]
fn test_crisis_detection_cooldown_blocks_rapid_events() {
    println!("=== FSV: Cooldown blocks rapid events ===");

    let mut monitor = setup_monitor_at_status(IdentityStatus::Healthy);

    // First crisis
    monitor.compute_continuity(&uniform_pv(1.0), 0.30, "crisis1");
    let result1 = monitor.detect_crisis();
    assert!(result1.can_emit_event);

    // Mark event emitted
    monitor.mark_event_emitted();

    // Immediately detect again
    monitor.compute_continuity(&uniform_pv(1.0), 0.25, "crisis2");
    let result2 = monitor.detect_crisis();

    // VERIFY
    println!("AFTER: can_emit_event = {}", result2.can_emit_event);
    println!(
        "AFTER: time_since_last_event = {:?}",
        result2.time_since_last_event
    );

    assert!(!result2.can_emit_event);
    assert!(result2.time_since_last_event.unwrap() < CRISIS_EVENT_COOLDOWN);

    println!("EVIDENCE: Cooldown correctly blocks rapid event emission");
}

#[test]
fn test_mark_event_emitted_sets_time() {
    println!("=== FSV: mark_event_emitted sets last_event_time ===");

    let mut monitor = IdentityContinuityMonitor::new();

    // BEFORE
    assert!(monitor.time_since_last_event().is_none());

    // EXECUTE
    monitor.mark_event_emitted();

    // AFTER
    let elapsed = monitor.time_since_last_event();
    assert!(elapsed.is_some());
    // Should be very recent (< 100ms)
    assert!(elapsed.unwrap() < Duration::from_millis(100));

    println!("EVIDENCE: mark_event_emitted correctly sets time");
}

// ============================================================================
// Helper Method Tests
// ============================================================================

#[test]
fn test_previous_status_getter() {
    println!("=== FSV: previous_status getter ===");

    let monitor = IdentityContinuityMonitor::new();

    // Default should be Healthy
    assert_eq!(monitor.previous_status(), IdentityStatus::Healthy);

    println!("EVIDENCE: previous_status getter returns default Healthy");
}

#[test]
fn test_status_changed_method() {
    println!("=== FSV: status_changed method ===");

    let mut monitor = IdentityContinuityMonitor::new();

    // No computation yet - status_changed should be false
    assert!(!monitor.status_changed());

    // First computation (Healthy)
    monitor.compute_continuity(&uniform_pv(1.0), 0.95, "first");
    // Now current_status is Healthy, previous_status is also Healthy (default)
    assert!(!monitor.status_changed());

    // Drop to Warning
    monitor.compute_continuity(&uniform_pv(1.0), 0.75, "warning");
    // current_status = Warning, previous_status = Healthy (not updated until detect_crisis)
    assert!(monitor.status_changed());

    println!("EVIDENCE: status_changed correctly detects transitions");
}

#[test]
fn test_entering_critical_method() {
    println!("=== FSV: entering_critical method ===");

    let mut monitor = IdentityContinuityMonitor::new();

    // No computation yet
    assert!(!monitor.entering_critical());

    // Compute healthy
    monitor.compute_continuity(&uniform_pv(1.0), 0.95, "healthy");
    assert!(!monitor.entering_critical());

    // Drop to Critical
    monitor.compute_continuity(&uniform_pv(1.0), 0.30, "critical");
    // current = Critical, previous_status = Healthy (not updated until detect_crisis)
    assert!(monitor.entering_critical());

    println!("EVIDENCE: entering_critical correctly detects Critical transition");
}

// ============================================================================
// State Ordinal Tests
// ============================================================================

#[test]
fn test_status_ordinal_ordering() {
    println!("=== FSV: Status ordinal ordering ===");

    // We can verify ordinal ordering through recovery detection
    let mut monitor = setup_monitor_at_status(IdentityStatus::Critical);

    // Recovery tests verify ordinal ordering implicitly
    // Critical(0) -> Degraded(1) = recovery
    monitor.compute_continuity(&uniform_pv(1.0), 0.55, "to_degraded");
    let r1 = monitor.detect_crisis();
    assert!(r1.recovering, "Critical -> Degraded should be recovering");

    // Degraded(1) -> Warning(2) = recovery
    monitor.compute_continuity(&uniform_pv(1.0), 0.75, "to_warning");
    let r2 = monitor.detect_crisis();
    assert!(r2.recovering, "Degraded -> Warning should be recovering");

    // Warning(2) -> Healthy(3) = recovery
    monitor.compute_continuity(&uniform_pv(1.0), 0.95, "to_healthy");
    let r3 = monitor.detect_crisis();
    assert!(r3.recovering, "Warning -> Healthy should be recovering");

    println!("EVIDENCE: Status ordinals correctly ordered: Critical < Degraded < Warning < Healthy");
}

// ============================================================================
// CrisisDetectionResult Struct Tests
// ============================================================================

#[test]
fn test_crisis_detection_result_all_fields() {
    println!("=== FSV: CrisisDetectionResult has all expected fields ===");

    let result = CrisisDetectionResult {
        identity_coherence: 0.5,
        previous_status: IdentityStatus::Healthy,
        current_status: IdentityStatus::Warning,
        status_changed: true,
        entering_crisis: true,
        entering_critical: false,
        recovering: false,
        time_since_last_event: None,
        can_emit_event: true,
    };

    // Verify all fields are accessible
    assert!((result.identity_coherence - 0.5).abs() < 1e-6);
    assert_eq!(result.previous_status, IdentityStatus::Healthy);
    assert_eq!(result.current_status, IdentityStatus::Warning);
    assert!(result.status_changed);
    assert!(result.entering_crisis);
    assert!(!result.entering_critical);
    assert!(!result.recovering);
    assert!(result.time_since_last_event.is_none());
    assert!(result.can_emit_event);

    println!("EVIDENCE: CrisisDetectionResult struct has all required fields");
}

#[test]
fn test_crisis_detection_result_debug_and_clone() {
    println!("=== FSV: CrisisDetectionResult implements Debug and Clone ===");

    let result = CrisisDetectionResult {
        identity_coherence: 0.75,
        previous_status: IdentityStatus::Warning,
        current_status: IdentityStatus::Degraded,
        status_changed: true,
        entering_crisis: false,
        entering_critical: false,
        recovering: false,
        time_since_last_event: Some(Duration::from_secs(5)),
        can_emit_event: true,
    };

    // Test Debug
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("CrisisDetectionResult"));
    assert!(debug_str.contains("identity_coherence"));

    // Test Clone
    let cloned = result.clone();
    assert_eq!(cloned.identity_coherence, result.identity_coherence);
    assert_eq!(cloned.current_status, result.current_status);

    // Test PartialEq
    assert_eq!(cloned, result);

    println!("EVIDENCE: CrisisDetectionResult correctly implements Debug, Clone, PartialEq");
}

// ============================================================================
// Integration/Lifecycle Tests
// ============================================================================

#[test]
fn test_full_crisis_lifecycle() {
    println!("=== FSV: Full crisis detection lifecycle ===");

    let mut monitor = IdentityContinuityMonitor::new();

    // Phase 1: Start healthy
    println!("PHASE 1: Initial state");
    monitor.compute_continuity(&uniform_pv(1.0), 0.95, "init_healthy");
    let r1 = monitor.detect_crisis();
    println!(
        "  status={:?}, changed={}, entering_crisis={}",
        r1.current_status, r1.status_changed, r1.entering_crisis
    );
    assert_eq!(r1.current_status, IdentityStatus::Healthy);
    assert!(!r1.entering_crisis);

    // Phase 2: Enter crisis (Warning)
    println!("PHASE 2: Enter crisis");
    monitor.compute_continuity(&uniform_pv(1.0), 0.75, "warning");
    let r2 = monitor.detect_crisis();
    println!(
        "  status={:?}, changed={}, entering_crisis={}",
        r2.current_status, r2.status_changed, r2.entering_crisis
    );
    assert!(r2.entering_crisis);
    assert!(!r2.entering_critical);

    // Phase 3: Deteriorate to Critical
    println!("PHASE 3: Deteriorate to Critical");
    monitor.compute_continuity(&uniform_pv(1.0), 0.30, "critical");
    let r3 = monitor.detect_crisis();
    println!(
        "  status={:?}, changed={}, entering_critical={}",
        r3.current_status, r3.status_changed, r3.entering_critical
    );
    assert!(r3.entering_critical);
    assert!(!r3.entering_crisis); // Not from Healthy

    // Phase 4: Recover
    println!("PHASE 4: Recovery");
    monitor.compute_continuity(&uniform_pv(1.0), 0.95, "recovery");
    let r4 = monitor.detect_crisis();
    println!(
        "  status={:?}, changed={}, recovering={}",
        r4.current_status, r4.status_changed, r4.recovering
    );
    assert!(r4.recovering);
    assert_eq!(r4.current_status, IdentityStatus::Healthy);

    println!("EVIDENCE: Full lifecycle correctly tracked all transitions");
}

#[test]
fn test_cooldown_constant_value() {
    println!("=== FSV: CRISIS_EVENT_COOLDOWN constant value ===");

    // Verify the constant is 30 seconds as specified
    assert_eq!(CRISIS_EVENT_COOLDOWN, Duration::from_secs(30));

    println!("EVIDENCE: CRISIS_EVENT_COOLDOWN = 30 seconds");
}
