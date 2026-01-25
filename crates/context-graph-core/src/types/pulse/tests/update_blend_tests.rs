//! Tests for CognitivePulse update() and blend() methods.

use crate::types::pulse::{CognitivePulse, SuggestedAction};

// =======================================================================
// CognitivePulse::update() Tests (TASK-M02-022)
// =======================================================================

#[test]
fn test_update_modifies_entropy_and_coherence() {
    let mut pulse = CognitivePulse::default();
    assert_eq!(pulse.entropy, 0.5);
    assert_eq!(pulse.coherence, 0.5);

    pulse.update(0.2, -0.1);

    assert_eq!(pulse.entropy, 0.7);
    assert_eq!(pulse.coherence, 0.4);
}

#[test]
fn test_update_clamps_values() {
    let mut pulse = CognitivePulse::default();

    // Try to exceed bounds
    pulse.update(1.0, 1.0);
    assert_eq!(pulse.entropy, 1.0);
    assert_eq!(pulse.coherence, 1.0);

    // Try to go below zero
    pulse.update(-2.0, -2.0);
    assert_eq!(pulse.entropy, 0.0);
    assert_eq!(pulse.coherence, 0.0);
}

#[test]
fn test_update_recomputes_action() {
    let mut pulse = CognitivePulse::from_values(0.5, 0.5);
    assert_eq!(pulse.suggested_action, SuggestedAction::Continue);

    // Push to high entropy, low coherence -> Stabilize
    pulse.update(0.3, -0.2);
    assert_eq!(pulse.entropy, 0.8);
    assert_eq!(pulse.coherence, 0.3);
    assert_eq!(pulse.suggested_action, SuggestedAction::Stabilize);
}

#[test]
fn test_update_updates_coherence_delta() {
    let mut pulse = CognitivePulse::default();
    assert_eq!(pulse.coherence_delta, 0.0);

    pulse.update(0.0, 0.2);

    // coherence went from 0.5 to 0.7, so delta = 0.2
    assert!((pulse.coherence_delta - 0.2).abs() < 0.001);
}

#[test]
fn test_update_updates_timestamp() {
    let mut pulse = CognitivePulse::default();
    let original_timestamp = pulse.timestamp;

    std::thread::sleep(std::time::Duration::from_millis(10));
    pulse.update(0.1, 0.1);

    assert!(pulse.timestamp > original_timestamp);
}

// =======================================================================
// CognitivePulse::blend() Tests (TASK-M02-022)
// =======================================================================

#[test]
fn test_blend_at_zero_equals_self() {
    let pulse1 = CognitivePulse::from_values(0.2, 0.8);
    let pulse2 = CognitivePulse::from_values(0.8, 0.2);

    let blended = pulse1.blend(&pulse2, 0.0);

    assert_eq!(blended.entropy, 0.2);
    assert_eq!(blended.coherence, 0.8);
}

#[test]
fn test_blend_at_one_equals_other() {
    let pulse1 = CognitivePulse::from_values(0.2, 0.8);
    let pulse2 = CognitivePulse::from_values(0.8, 0.2);

    let blended = pulse1.blend(&pulse2, 1.0);

    // Use approximate comparison for floating-point values
    assert!((blended.entropy - 0.8).abs() < 0.0001);
    assert!((blended.coherence - 0.2).abs() < 0.0001);
}

#[test]
fn test_blend_at_midpoint() {
    let pulse1 = CognitivePulse::from_values(0.2, 0.8);
    let pulse2 = CognitivePulse::from_values(0.8, 0.2);

    let blended = pulse1.blend(&pulse2, 0.5);

    assert_eq!(blended.entropy, 0.5);
    assert_eq!(blended.coherence, 0.5);
}

#[test]
fn test_blend_clamps_t() {
    let pulse1 = CognitivePulse::from_values(0.2, 0.8);
    let pulse2 = CognitivePulse::from_values(0.8, 0.2);

    // t > 1.0 should clamp to 1.0
    let blended = pulse1.blend(&pulse2, 2.0);
    assert_eq!(blended.entropy, 0.8);

    // t < 0.0 should clamp to 0.0
    let blended = pulse1.blend(&pulse2, -1.0);
    assert_eq!(blended.entropy, 0.2);
}

#[test]
fn test_blend_interpolates_all_numeric_fields() {
    let pulse1 = CognitivePulse::new(
        0.2,
        0.8,
        0.1,
        1.0,
        SuggestedAction::Ready,
    );
    let pulse2 = CognitivePulse::new(
        0.8,
        0.2,
        -0.1,
        1.4,
        SuggestedAction::Explore,
    );

    let blended = pulse1.blend(&pulse2, 0.5);

    assert_eq!(blended.entropy, 0.5);
    assert_eq!(blended.coherence, 0.5);
    assert_eq!(blended.coherence_delta, 0.0); // (0.1 + -0.1) / 2
    assert_eq!(blended.emotional_weight, 1.2); // (1.0 + 1.4) / 2
}

#[test]
fn test_blend_recomputes_action() {
    let pulse1 = CognitivePulse::from_values(0.2, 0.9); // Ready
    let pulse2 = CognitivePulse::from_values(0.9, 0.2); // Stabilize

    // Midpoint should compute a new action
    let blended = pulse1.blend(&pulse2, 0.5);

    // entropy=0.55, coherence=0.55 -> should compute appropriate action
    // Not testing specific action, just that it computes something
    assert!(matches!(
        blended.suggested_action,
        SuggestedAction::Continue | SuggestedAction::Explore | SuggestedAction::Review
    ));
}

#[test]
fn test_blend_creates_new_timestamp() {
    let pulse1 = CognitivePulse::default();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let pulse2 = CognitivePulse::default();
    std::thread::sleep(std::time::Duration::from_millis(10));

    let blended = pulse1.blend(&pulse2, 0.5);

    // Blended timestamp should be newest
    assert!(blended.timestamp >= pulse2.timestamp);
}

// =======================================================================
// Edge Case Tests (TASK-M02-022)
// =======================================================================

#[test]
fn edge_case_update_extreme_deltas() {
    let mut pulse = CognitivePulse::from_values(0.5, 0.5);

    println!(
        "BEFORE: entropy={}, coherence={}",
        pulse.entropy, pulse.coherence
    );

    pulse.update(100.0, -100.0);

    println!(
        "AFTER: entropy={}, coherence={}",
        pulse.entropy, pulse.coherence
    );

    assert_eq!(
        pulse.entropy, 1.0,
        "Extreme positive delta should clamp to 1.0"
    );
    assert_eq!(
        pulse.coherence, 0.0,
        "Extreme negative delta should clamp to 0.0"
    );
}

#[test]
fn edge_case_blend_identical_pulses() {
    let pulse = CognitivePulse::from_values(0.6, 0.7);

    println!(
        "ORIGINAL: entropy={}, coherence={}",
        pulse.entropy, pulse.coherence
    );

    let blended = pulse.blend(&pulse, 0.5);

    println!(
        "BLENDED: entropy={}, coherence={}",
        blended.entropy, blended.coherence
    );

    assert_eq!(blended.entropy, pulse.entropy);
    assert_eq!(blended.coherence, pulse.coherence);
}

#[test]
fn edge_case_update_action_transition() {
    let mut pulse = CognitivePulse::from_values(0.35, 0.75);

    println!(
        "STATE 1: entropy={}, coherence={}, action={:?}",
        pulse.entropy, pulse.coherence, pulse.suggested_action
    );

    // Low entropy + high coherence = Ready
    assert_eq!(pulse.suggested_action, SuggestedAction::Ready);

    pulse.update(0.5, -0.5);

    println!(
        "STATE 2: entropy={}, coherence={}, action={:?}",
        pulse.entropy, pulse.coherence, pulse.suggested_action
    );

    // High entropy + low coherence = Stabilize
    assert_eq!(pulse.suggested_action, SuggestedAction::Stabilize);
}
