// crates/context-graph-core/src/gwt/session_identity/types.rs
//! SessionIdentitySnapshot - Core data structure for cross-session identity persistence.
//!
//! # Constitution Reference
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - GWT-003: Identity continuity tracking
//! - AP-25: Kuramoto must have exactly 13 oscillators

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Maximum trajectory length (50 entries = ~2.6KB max)
/// Reduced from 1000 to minimize serialization size
pub const MAX_TRAJECTORY_LEN: usize = 50;

/// Number of Kuramoto oscillators (one per embedder)
/// MUST match KURAMOTO_N in layers/coherence/constants.rs
pub const KURAMOTO_N: usize = 13;

/// Flattened session identity for fast serialization.
/// Target size: <30KB typical (down from 80KB).
///
/// # Size Breakdown
/// - Header (session_id + timestamp + previous + ic): ~100 bytes
/// - Kuramoto state (13x8 + 8): 112 bytes
/// - Purpose vector (13x4): 52 bytes
/// - Trajectory (50x13x4): ~2,600 bytes max
/// - IC monitor state: 8 bytes
/// - Consciousness snapshot: 16 bytes
/// - **Total**: ~3KB typical, <30KB with full trajectory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionIdentitySnapshot {
    // Header (fixed size: ~100 bytes)
    /// UUID string for session identification
    pub session_id: String,
    /// Unix milliseconds timestamp
    pub timestamp_ms: i64,
    /// Previous session UUID for continuity linking
    pub previous_session_id: Option<String>,
    /// Cross-session identity continuity score [0.0, 1.0]
    pub cross_session_ic: f32,

    // Kuramoto state (fixed size: 13*8 + 8 = 112 bytes)
    /// Kuramoto oscillator phases for 13 embedders
    pub kuramoto_phases: [f64; KURAMOTO_N],
    /// Kuramoto coupling strength
    pub coupling: f64,

    // Purpose vector (fixed size: 13*4 = 52 bytes)
    /// 13D purpose vector representing teleological alignment
    pub purpose_vector: [f32; KURAMOTO_N],

    // Identity trajectory (variable, capped at 50 entries ~2.6KB max)
    /// FIFO buffer of recent purpose vectors
    pub trajectory: Vec<[f32; KURAMOTO_N]>,

    // IC monitor state (small)
    /// Last computed identity continuity value
    pub last_ic: f32,
    /// Crisis threshold for dream triggering (default: 0.5)
    pub crisis_threshold: f32,

    // Consciousness snapshot (single, not history)
    /// Current consciousness level C(t)
    pub consciousness: f32,
    /// Integration factor (Kuramoto r)
    pub integration: f32,
    /// Reflection factor (meta-cognitive)
    pub reflection: f32,
    /// Differentiation factor (purpose entropy)
    pub differentiation: f32,
}

impl SessionIdentitySnapshot {
    /// Create new snapshot with given session ID
    ///
    /// # Arguments
    /// * `session_id` - UUID string for this session
    ///
    /// # Returns
    /// New snapshot with default values and specified session_id
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            previous_session_id: None,
            cross_session_ic: 1.0, // Fresh session starts with full continuity

            kuramoto_phases: [0.0; KURAMOTO_N],
            coupling: 0.5, // Default coupling per constitution

            purpose_vector: [0.0; KURAMOTO_N],
            trajectory: Vec::with_capacity(MAX_TRAJECTORY_LEN),

            last_ic: 1.0,
            crisis_threshold: 0.5, // IC_CRITICAL_THRESHOLD

            consciousness: 0.0,
            integration: 0.0,
            reflection: 0.0,
            differentiation: 0.0,
        }
    }

    /// Append purpose vector to trajectory with FIFO eviction
    ///
    /// When trajectory reaches MAX_TRAJECTORY_LEN, oldest entry is removed.
    ///
    /// # Arguments
    /// * `pv` - 13D purpose vector to append
    pub fn append_to_trajectory(&mut self, pv: [f32; KURAMOTO_N]) {
        if self.trajectory.len() >= MAX_TRAJECTORY_LEN {
            self.trajectory.remove(0); // FIFO eviction
        }
        self.trajectory.push(pv);
    }

    /// Estimate serialized size in bytes
    ///
    /// Formula: ~300 + (trajectory.len() * 52) + session_id.len()
    #[inline]
    pub fn estimated_size(&self) -> usize {
        // Base struct overhead: ~300 bytes
        // Each trajectory entry: 13 * 4 = 52 bytes
        // Session ID: variable length
        300 + (self.trajectory.len() * 52) + self.session_id.len()
    }
}

impl Default for SessionIdentitySnapshot {
    /// Create default snapshot with new UUID
    fn default() -> Self {
        Self::new(Uuid::new_v4().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TC-SESSION-01: Serialization Round-Trip
    // =========================================================================
    #[test]
    fn test_snapshot_serialization_roundtrip() {
        println!("\n=== TC-SESSION-01: Serialization Round-Trip ===");

        // SOURCE OF TRUTH: In-memory snapshot
        let snapshot = SessionIdentitySnapshot::default();
        println!(
            "BEFORE: Created snapshot with session_id={}",
            snapshot.session_id
        );
        println!("  estimated_size={} bytes", snapshot.estimated_size());

        // EXECUTE: Serialize
        let bytes = bincode::serialize(&snapshot).expect("Serialization must succeed");
        println!("AFTER SERIALIZE: {} bytes", bytes.len());

        // VERIFY: Size under 30KB
        assert!(
            bytes.len() < 30_000,
            "Serialized size {} must be < 30KB",
            bytes.len()
        );

        // EXECUTE: Deserialize
        let restored: SessionIdentitySnapshot =
            bincode::deserialize(&bytes).expect("Deserialization must succeed");

        // VERIFY: Round-trip equality
        assert_eq!(snapshot, restored, "Round-trip must preserve all fields");
        println!("AFTER DESERIALIZE: session_id={}", restored.session_id);
        println!(
            "RESULT: PASS - Round-trip successful, size={} bytes < 30KB",
            bytes.len()
        );
    }

    // =========================================================================
    // TC-SESSION-02: Trajectory FIFO Eviction
    // =========================================================================
    #[test]
    fn test_trajectory_fifo_eviction() {
        println!("\n=== TC-SESSION-02: Trajectory FIFO Eviction ===");

        // SOURCE OF TRUTH: snapshot.trajectory
        let mut snapshot = SessionIdentitySnapshot::default();
        println!("BEFORE: trajectory.len()={}", snapshot.trajectory.len());

        // EXECUTE: Add 100 entries (exceeds MAX_TRAJECTORY_LEN=50)
        for i in 0..100 {
            snapshot.append_to_trajectory([i as f32; KURAMOTO_N]);
        }

        // VERIFY: Length capped at MAX_TRAJECTORY_LEN
        println!("AFTER: trajectory.len()={}", snapshot.trajectory.len());
        assert_eq!(
            snapshot.trajectory.len(),
            MAX_TRAJECTORY_LEN,
            "Trajectory must cap at MAX_TRAJECTORY_LEN={}",
            MAX_TRAJECTORY_LEN
        );

        // VERIFY: FIFO behavior - first 50 entries evicted
        // Entry 0-49 should be gone, entry 50-99 should remain
        // First remaining entry (index 0) should have value 50.0
        assert_eq!(
            snapshot.trajectory[0][0], 50.0,
            "First entry should be 50.0 (first 50 evicted)"
        );
        assert_eq!(
            snapshot.trajectory[49][0], 99.0,
            "Last entry should be 99.0"
        );

        println!("RESULT: PASS - FIFO eviction working correctly");
        println!("  First entry value: {}", snapshot.trajectory[0][0]);
        println!("  Last entry value: {}", snapshot.trajectory[49][0]);
    }

    // =========================================================================
    // EDGE CASE 1: Empty Trajectory Serialization
    // =========================================================================
    #[test]
    fn edge_case_empty_trajectory() {
        println!("\n=== EDGE CASE: Empty Trajectory Serialization ===");

        let snapshot = SessionIdentitySnapshot::new("empty-test");
        println!("BEFORE: trajectory.len()={}", snapshot.trajectory.len());
        assert!(snapshot.trajectory.is_empty());

        let bytes = bincode::serialize(&snapshot).expect("Must serialize");
        let restored: SessionIdentitySnapshot =
            bincode::deserialize(&bytes).expect("Must deserialize");

        println!(
            "AFTER: Serialized {} bytes, restored trajectory.len()={}",
            bytes.len(),
            restored.trajectory.len()
        );
        assert!(restored.trajectory.is_empty());
        println!("RESULT: PASS - Empty trajectory handled correctly");
    }

    // =========================================================================
    // EDGE CASE 2: Maximum Trajectory Size
    // =========================================================================
    #[test]
    fn edge_case_max_trajectory_size() {
        println!("\n=== EDGE CASE: Maximum Trajectory Size ===");

        let mut snapshot = SessionIdentitySnapshot::default();

        // Fill to max
        for i in 0..MAX_TRAJECTORY_LEN {
            snapshot.append_to_trajectory([i as f32; KURAMOTO_N]);
        }

        println!(
            "BEFORE: trajectory.len()={}, estimated_size={}",
            snapshot.trajectory.len(),
            snapshot.estimated_size()
        );

        let bytes = bincode::serialize(&snapshot).expect("Must serialize");
        println!("AFTER: Actual serialized size={} bytes", bytes.len());

        assert!(bytes.len() < 30_000, "Max trajectory must still be < 30KB");
        println!("RESULT: PASS - Max trajectory size under 30KB limit");
    }

    // =========================================================================
    // EDGE CASE 3: Long Session ID
    // =========================================================================
    #[test]
    fn edge_case_long_session_id() {
        println!("\n=== EDGE CASE: Long Session ID (1000 chars) ===");

        let long_id = "x".repeat(1000);
        let snapshot = SessionIdentitySnapshot::new(long_id.clone());

        println!("BEFORE: session_id.len()={}", snapshot.session_id.len());

        let bytes = bincode::serialize(&snapshot).expect("Must serialize");
        let restored: SessionIdentitySnapshot =
            bincode::deserialize(&bytes).expect("Must deserialize");

        println!("AFTER: Serialized {} bytes", bytes.len());
        assert_eq!(restored.session_id, long_id);
        assert!(bytes.len() < 30_000, "Long session_id must still be < 30KB");
        println!("RESULT: PASS - Long session ID handled correctly");
    }

    // =========================================================================
    // TC-SESSION-01a: Verify Field Count
    // =========================================================================
    #[test]
    fn test_struct_has_14_fields() {
        println!("\n=== TC-SESSION-01a: Verify 14 Fields ===");

        let snapshot = SessionIdentitySnapshot::default();

        // Verify all 14 fields are accessible (compile-time check)
        let _ = &snapshot.session_id; // 1
        let _ = &snapshot.timestamp_ms; // 2
        let _ = &snapshot.previous_session_id; // 3
        let _ = &snapshot.cross_session_ic; // 4
        let _ = &snapshot.kuramoto_phases; // 5
        let _ = &snapshot.coupling; // 6
        let _ = &snapshot.purpose_vector; // 7
        let _ = &snapshot.trajectory; // 8
        let _ = &snapshot.last_ic; // 9
        let _ = &snapshot.crisis_threshold; // 10
        let _ = &snapshot.consciousness; // 11
        let _ = &snapshot.integration; // 12
        let _ = &snapshot.reflection; // 13
        let _ = &snapshot.differentiation; // 14

        println!("RESULT: PASS - All 14 fields accessible");
    }

    // =========================================================================
    // TC-SESSION-01b: Verify Constants
    // =========================================================================
    #[test]
    fn test_constants_match_constitution() {
        println!("\n=== TC-SESSION-01b: Verify Constants ===");

        // KURAMOTO_N must be 13 per constitution AP-25
        assert_eq!(KURAMOTO_N, 13, "KURAMOTO_N must be 13 per AP-25");

        // MAX_TRAJECTORY_LEN should be 50 (reduced from 1000)
        assert_eq!(MAX_TRAJECTORY_LEN, 50, "MAX_TRAJECTORY_LEN should be 50");

        // Verify array sizes match KURAMOTO_N
        let snapshot = SessionIdentitySnapshot::default();
        assert_eq!(snapshot.kuramoto_phases.len(), KURAMOTO_N);
        assert_eq!(snapshot.purpose_vector.len(), KURAMOTO_N);

        println!("RESULT: PASS - Constants match constitution");
        println!("  KURAMOTO_N={}", KURAMOTO_N);
        println!("  MAX_TRAJECTORY_LEN={}", MAX_TRAJECTORY_LEN);
    }

    // =========================================================================
    // TC-SESSION-01c: Estimated Size Accuracy
    // =========================================================================
    #[test]
    fn test_estimated_size_accuracy() {
        println!("\n=== TC-SESSION-01c: Estimated Size Accuracy ===");

        let snapshot = SessionIdentitySnapshot::default();
        let estimated = snapshot.estimated_size();
        let actual = bincode::serialize(&snapshot).unwrap().len();

        println!("Estimated: {} bytes", estimated);
        println!("Actual: {} bytes", actual);

        // Estimate should be within 50% of actual
        let ratio = estimated as f64 / actual as f64;
        println!("Ratio: {:.2}", ratio);

        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Estimate should be within 50%-200% of actual"
        );
        println!("RESULT: PASS - Estimate within acceptable range");
    }

    // =========================================================================
    // TC-SESSION-01d: Send + Sync Compatibility
    // =========================================================================
    #[test]
    fn test_send_sync_compatible() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<SessionIdentitySnapshot>();
        assert_sync::<SessionIdentitySnapshot>();

        println!("RESULT: PASS - SessionIdentitySnapshot is Send + Sync");
    }
}
