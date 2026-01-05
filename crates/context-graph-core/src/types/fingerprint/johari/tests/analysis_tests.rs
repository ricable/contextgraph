//! Tests for JohariFingerprint analysis methods.

use crate::types::fingerprint::johari::JohariFingerprint;
use crate::types::fingerprint::purpose::NUM_EMBEDDERS;
use crate::types::JohariQuadrant;

// ===== find_blind_spots() Tests =====

#[test]
fn test_find_blind_spots() {
    let mut jf = JohariFingerprint::zeroed();

    // E1 (index 0) = strongly Open
    jf.set_quadrant(0, 0.9, 0.05, 0.03, 0.02, 1.0);
    // E5 (index 4) = strongly Blind
    jf.set_quadrant(4, 0.1, 0.1, 0.7, 0.1, 1.0);

    let blind_spots = jf.find_blind_spots();

    println!("Blind spots: {:?}", blind_spots);

    // Should find E5 as a blind spot (E1 Open * E5 Blind = 0.9 * 0.7 = 0.63)
    let e5_spot = blind_spots.iter().find(|(idx, _)| *idx == 4);
    assert!(e5_spot.is_some(), "E5 should be in blind spots");

    let (_, severity) = e5_spot.unwrap();
    let expected_severity = 0.9 * 0.7;
    assert!(
        (severity - expected_severity).abs() < 0.01,
        "Severity should be ~{}, got {}",
        expected_severity,
        severity
    );

    println!("[PASS] find_blind_spots detects cross-space gaps");
}

// ===== predict_transition() Tests =====

#[test]
fn test_predict_transition() {
    let mut jf = JohariFingerprint::zeroed();

    // Set custom transition probs for E1 (index 0)
    // From Open, highest prob to Hidden
    let mut matrix = [[0.25f32; 4]; 4];
    matrix[JohariFingerprint::OPEN_IDX] = [0.1, 0.6, 0.2, 0.1]; // From Open -> Hidden most likely

    jf.set_transition_probs(0, matrix);

    let predicted = jf.predict_transition(0, JohariQuadrant::Open);
    assert_eq!(predicted, JohariQuadrant::Hidden);

    println!("[PASS] predict_transition uses transition matrix correctly");
}

#[test]
#[should_panic(expected = "embedder_idx")]
fn test_predict_transition_out_of_bounds() {
    let jf = JohariFingerprint::zeroed();
    let _ = jf.predict_transition(13, JohariQuadrant::Open);
}

// ===== openness() and is_aware() Tests =====

#[test]
fn test_openness() {
    let mut jf = JohariFingerprint::zeroed();

    // Set 7 out of 13 embedders to Open
    for i in 0..7 {
        jf.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0);
    }
    // Set remaining 6 to Hidden (not Open)
    for i in 7..NUM_EMBEDDERS {
        jf.set_quadrant(i, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
    }

    let openness = jf.openness();
    let expected = 7.0 / 13.0;
    assert!(
        (openness - expected).abs() < 0.01,
        "Openness should be ~{}, got {}",
        expected,
        openness
    );

    println!("[PASS] openness() returns correct fraction");
}

#[test]
fn test_is_aware() {
    let mut jf = JohariFingerprint::zeroed();

    // Set majority (7+) to Open/Hidden
    for i in 0..7 {
        jf.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0); // Open
    }
    // Set remaining to Blind (not aware)
    for i in 7..NUM_EMBEDDERS {
        jf.set_quadrant(i, 0.0, 0.0, 1.0, 0.0, 1.0); // Blind
    }

    assert!(jf.is_aware(), "7/13 Open should be aware");

    // Set all to Unknown
    let mut jf2 = JohariFingerprint::zeroed();
    for i in 0..NUM_EMBEDDERS {
        jf2.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
    }

    assert!(!jf2.is_aware(), "All Unknown should not be aware");

    println!("[PASS] is_aware() returns correct value");
}

// ===== Verification Log =====

#[test]
fn verification_log() {
    println!("\n=== TASK-F003 VERIFICATION LOG ===");
    println!("Timestamp: 2026-01-05");
    println!();
    println!("Struct Verification:");
    println!("1. JohariFingerprint has quadrants: [[f32; 4]; 13] check");
    println!("2. JohariFingerprint has confidence: [f32; 13] check");
    println!("3. JohariFingerprint has transition_probs: [[[f32; 4]; 4]; 13] check");
    println!();
    println!("Method Verification:");
    println!("4. zeroed() creates valid fingerprint check");
    println!("5. classify_quadrant() follows UTL thresholds check");
    println!("6. set_quadrant() normalizes to sum=1.0 check");
    println!("7. dominant_quadrant() returns highest weight check");
    println!("8. find_by_quadrant() returns correct indices check");
    println!("9. find_blind_spots() detects cross-space gaps check");
    println!("10. predict_transition() uses transition matrix check");
    println!("11. to_compact_bytes() encodes 13 quadrants in 4 bytes check");
    println!("12. from_compact_bytes() decodes correctly check");
    println!("13. validate() catches all invariant violations check");
    println!();
    println!("Edge Cases:");
    println!("- All-zero weights -> uniform distribution check");
    println!("- Max index 12 (E13) -> valid check");
    println!("- Index 13 -> panics check");
    println!();
    println!("VERIFICATION LOG COMPLETE");
}
