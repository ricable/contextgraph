//! Test helper functions for TeleologicalFingerprint tests.

use crate::types::fingerprint::johari::JohariFingerprint;
use crate::types::fingerprint::purpose::PurposeVector;
use crate::types::fingerprint::SemanticFingerprint;
use crate::types::fingerprint::NUM_EMBEDDERS;

pub fn make_test_semantic() -> SemanticFingerprint {
    SemanticFingerprint::default()
}

pub fn make_test_purpose(alignment: f32) -> PurposeVector {
    PurposeVector::new([alignment; NUM_EMBEDDERS])
}

pub fn make_test_johari() -> JohariFingerprint {
    // Create with all Open quadrants (high openness)
    let mut jf = JohariFingerprint::zeroed();
    for i in 0..NUM_EMBEDDERS {
        jf.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0); // 100% Open, 100% confidence
    }
    jf
}

pub fn make_test_hash() -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[0] = 0xDE;
    hash[1] = 0xAD;
    hash[30] = 0xBE;
    hash[31] = 0xEF;
    hash
}
