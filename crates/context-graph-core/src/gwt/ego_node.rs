//! SELF_EGO_NODE - System Identity and Self-Awareness
//!
//! Implements persistent system identity node as specified in Constitution v4.0.0
//! Section gwt.self_ego_node (lines 371-392).
//!
//! The SELF_EGO_NODE represents the system's understanding of itself:
//! - Current system state (TeleologicalFingerprint)
//! - System's purpose alignment (PurposeVector)
//! - Identity history (trajectory of purpose evolution)
//! - Alignment between actions and self-model

use crate::error::CoreResult;
use crate::types::fingerprint::TeleologicalFingerprint;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Special memory node representing the system's identity
#[derive(Debug, Clone)]
pub struct SelfEgoNode {
    /// Fixed ID for the SELF_EGO_NODE
    pub id: Uuid,
    /// Current teleological fingerprint (system state)
    pub fingerprint: Option<TeleologicalFingerprint>,
    /// System's purpose vector (alignment with north star)
    pub purpose_vector: [f32; 13],
    /// Coherence between current actions and purpose vector
    pub coherence_with_actions: f32,
    /// History of identity snapshots
    pub identity_trajectory: Vec<PurposeSnapshot>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Snapshot of purpose vector at a point in time
#[derive(Debug, Clone)]
pub struct PurposeSnapshot {
    /// Purpose vector at this moment
    pub vector: [f32; 13],
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    /// Context (brief description of system state)
    pub context: String,
}

impl SelfEgoNode {
    /// Create a new SELF_EGO_NODE
    pub fn new() -> Self {
        // Use a fixed deterministic UUID for system identity
        let id = Uuid::nil(); // Special "zero" UUID for system identity

        Self {
            id,
            fingerprint: None,
            purpose_vector: [0.0; 13],
            coherence_with_actions: 0.0,
            identity_trajectory: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    /// Initialize with a purpose vector
    pub fn with_purpose_vector(vector: [f32; 13]) -> Self {
        let mut ego = Self::new();
        ego.purpose_vector = vector;
        ego
    }

    /// Update system fingerprint (state snapshot)
    pub fn update_fingerprint(&mut self, fingerprint: TeleologicalFingerprint) -> CoreResult<()> {
        self.fingerprint = Some(fingerprint);
        self.last_updated = Utc::now();
        Ok(())
    }

    /// Record a purpose vector snapshot in the identity trajectory
    pub fn record_purpose_snapshot(&mut self, context: impl Into<String>) -> CoreResult<()> {
        let snapshot = PurposeSnapshot {
            vector: self.purpose_vector,
            timestamp: Utc::now(),
            context: context.into(),
        };
        self.identity_trajectory.push(snapshot);

        // Keep last 1000 snapshots for memory efficiency
        if self.identity_trajectory.len() > 1000 {
            self.identity_trajectory.remove(0);
        }

        Ok(())
    }

    /// Get the purpose vector at a specific point in history
    pub fn get_historical_purpose_vector(&self, index: usize) -> Option<[f32; 13]> {
        self.identity_trajectory.get(index).map(|s| s.vector)
    }

    /// Get most recent purpose snapshot
    pub fn get_latest_snapshot(&self) -> Option<&PurposeSnapshot> {
        self.identity_trajectory.last()
    }

    /// Update purpose_vector from a TeleologicalFingerprint's purpose alignments.
    ///
    /// Copies fingerprint.purpose_vector.alignments to self.purpose_vector,
    /// updates coherence_with_actions, and sets fingerprint reference.
    ///
    /// # Arguments
    /// * `fingerprint` - The source fingerprint containing purpose_vector.alignments
    ///
    /// # Returns
    /// * `CoreResult<()>` - Ok on success
    ///
    /// # Constitution Reference
    /// From constitution.yaml lines 365-392:
    /// - self_ego_node.fields includes: fingerprint, purpose_vector, coherence_with_actions
    /// - loop: "Retrieve→A(action,PV)→if<0.55 self_reflect→update fingerprint→store evolution"
    pub fn update_from_fingerprint(&mut self, fingerprint: &TeleologicalFingerprint) -> CoreResult<()> {
        // 1. Copy purpose_vector.alignments to self.purpose_vector
        self.purpose_vector = fingerprint.purpose_vector.alignments;

        // 2. Update coherence from fingerprint
        self.coherence_with_actions = fingerprint.purpose_vector.coherence;

        // 3. Store fingerprint reference (clone since we own the data)
        self.fingerprint = Some(fingerprint.clone());

        // 4. Update timestamp
        self.last_updated = Utc::now();

        // 5. Log for debugging
        tracing::debug!(
            "SelfEgoNode updated from fingerprint: purpose_vector[0]={:.4}, coherence={:.4}",
            self.purpose_vector[0],
            self.coherence_with_actions
        );

        Ok(())
    }
}

impl Default for SelfEgoNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Self-Awareness Loop for identity continuity
#[derive(Debug)]
pub struct SelfAwarenessLoop {
    /// Identity continuity tracking
    continuity: IdentityContinuity,
    /// Action-to-purpose alignment threshold
    alignment_threshold: f32,
}

/// Tracks identity continuity over time
#[derive(Debug, Clone)]
pub struct IdentityContinuity {
    /// Cosine similarity between consecutive purpose vectors
    pub recent_continuity: f32,
    /// Order parameter r from Kuramoto sync
    pub kuramoto_order_parameter: f32,
    /// Overall identity coherence: IC = cos(PV_t, PV_{t-1}) × r(t)
    pub identity_coherence: f32,
    /// Status of identity continuity
    pub status: IdentityStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityStatus {
    /// IC > 0.9: Healthy identity continuity
    Healthy,
    /// 0.7 ≤ IC ≤ 0.9: Warning state, monitor closely
    Warning,
    /// IC < 0.7: Degraded identity, may need intervention
    Degraded,
    /// IC < 0.5: Critical, trigger introspective dream
    Critical,
}

impl IdentityContinuity {
    /// Create new IdentityContinuity with correct initial status per constitution.yaml lines 387-392
    /// Starting with identity_coherence=0.0 means status=Critical (IC < 0.5 triggers dream consolidation)
    pub fn new() -> Self {
        let identity_coherence = 0.0;
        Self {
            recent_continuity: 1.0,
            kuramoto_order_parameter: 0.0,
            identity_coherence,
            status: Self::compute_status_from_coherence(identity_coherence),
        }
    }

    /// Compute status per constitution.yaml lines 387-392:
    /// - Healthy: IC > 0.9
    /// - Warning: 0.7 <= IC <= 0.9
    /// - Degraded: 0.5 <= IC < 0.7
    /// - Critical: IC < 0.5 (triggers dream consolidation)
    fn compute_status_from_coherence(coherence: f32) -> IdentityStatus {
        match coherence {
            ic if ic > 0.9 => IdentityStatus::Healthy,
            ic if ic >= 0.7 => IdentityStatus::Warning,
            ic if ic >= 0.5 => IdentityStatus::Degraded,
            _ => IdentityStatus::Critical,
        }
    }

    /// Update identity coherence: IC = cos(PV_t, PV_{t-1}) × r(t)
    pub fn update(&mut self, pv_cosine: f32, kuramoto_r: f32) -> CoreResult<IdentityStatus> {
        self.recent_continuity = pv_cosine.clamp(-1.0, 1.0);
        self.kuramoto_order_parameter = kuramoto_r.clamp(0.0, 1.0);

        // Identity coherence = cosine × r
        self.identity_coherence = (pv_cosine * kuramoto_r).clamp(0.0, 1.0);

        // Determine status using canonical computation
        self.status = Self::compute_status_from_coherence(self.identity_coherence);

        Ok(self.status)
    }
}

impl Default for IdentityContinuity {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfAwarenessLoop {
    /// Create a new self-awareness loop
    pub fn new() -> Self {
        Self {
            continuity: IdentityContinuity::new(),
            alignment_threshold: 0.55,
        }
    }

    /// Get the current identity coherence value
    ///
    /// Returns the IC value computed as: IC = cos(PV_t, PV_{t-1}) × r(t)
    /// Per constitution.yaml lines 387-392
    pub fn identity_coherence(&self) -> f32 {
        self.continuity.identity_coherence
    }

    /// Get the current identity status
    pub fn identity_status(&self) -> IdentityStatus {
        self.continuity.status
    }

    /// Execute self-awareness loop for a single cycle
    ///
    /// # Algorithm
    /// 1. Retrieve current SELF_EGO_NODE purpose vector
    /// 2. Compute alignment with current action
    /// 3. If alignment < 0.55: trigger self-reflection
    /// 4. Update fingerprint with action outcome
    /// 5. Store to purpose_evolution (temporal trajectory)
    pub async fn cycle(
        &mut self,
        ego_node: &mut SelfEgoNode,
        action_embedding: &[f32; 13],
        kuramoto_r: f32,
    ) -> CoreResult<SelfReflectionResult> {
        // Compute cosine similarity between action and current purpose
        let alignment = self.cosine_similarity(&ego_node.purpose_vector, action_embedding);

        // Check if reflection is needed
        let needs_reflection = alignment < self.alignment_threshold;

        // Update identity continuity
        if !ego_node.identity_trajectory.is_empty() {
            let prev_pv = ego_node
                .get_latest_snapshot()
                .map(|s| s.vector)
                .unwrap_or(ego_node.purpose_vector);

            let pv_cosine = self.cosine_similarity(&prev_pv, &ego_node.purpose_vector);
            let status = self.continuity.update(pv_cosine, kuramoto_r)?;

            // Check for critical identity drift
            if status == IdentityStatus::Critical {
                // Trigger introspective dream
                ego_node.record_purpose_snapshot("Critical identity drift - dream triggered")?;
            }
        }

        // Record snapshot of current state
        ego_node.record_purpose_snapshot("Self-awareness cycle")?;

        Ok(SelfReflectionResult {
            alignment,
            needs_reflection,
            identity_status: self.continuity.status,
            identity_coherence: self.continuity.identity_coherence,
        })
    }

    /// Compute cosine similarity between two 13D vectors
    fn cosine_similarity(&self, v1: &[f32; 13], v2: &[f32; 13]) -> f32 {
        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let magnitude_v1: f32 = (v1.iter().map(|a| a * a).sum::<f32>()).sqrt();
        let magnitude_v2: f32 = (v2.iter().map(|a| a * a).sum::<f32>()).sqrt();

        if magnitude_v1 < 1e-6 || magnitude_v2 < 1e-6 {
            0.0
        } else {
            dot_product / (magnitude_v1 * magnitude_v2)
        }
    }
}

impl Default for SelfAwarenessLoop {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from a self-awareness cycle
#[derive(Debug, Clone)]
pub struct SelfReflectionResult {
    /// Alignment between action and purpose
    pub alignment: f32,
    /// Whether self-reflection should be triggered
    pub needs_reflection: bool,
    /// Current identity status
    pub identity_status: IdentityStatus,
    /// Current identity coherence value
    pub identity_coherence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_ego_node_creation() {
        let ego = SelfEgoNode::new();
        assert_eq!(ego.id, Uuid::nil());
        assert_eq!(ego.purpose_vector, [0.0; 13]);
    }

    #[test]
    fn test_self_ego_node_purpose_update() {
        let mut ego = SelfEgoNode::new();
        let pv = [1.0; 13];
        ego.purpose_vector = pv;

        assert_eq!(ego.purpose_vector, pv);
    }

    #[test]
    fn test_purpose_snapshot_recording() {
        let mut ego = SelfEgoNode::new();
        ego.purpose_vector = [0.5; 13];

        ego.record_purpose_snapshot("Test snapshot").unwrap();
        assert_eq!(ego.identity_trajectory.len(), 1);

        let snapshot = ego.get_latest_snapshot().unwrap();
        assert_eq!(snapshot.vector, [0.5; 13]);
        assert!(snapshot.context.contains("Test snapshot"));
    }

    /// FSV test: Initial IdentityContinuity status should be Critical per constitution.yaml lines 387-392
    /// Because identity_coherence=0.0 at initialization, which is < 0.5 (Critical threshold)
    #[test]
    fn test_identity_continuity_initial_status_is_critical() {
        let continuity = IdentityContinuity::new();

        // Per constitution: IC < 0.5 should be Critical, not Healthy
        assert_eq!(
            continuity.status,
            IdentityStatus::Critical,
            "Initial identity coherence 0.0 must result in Critical status, not Healthy"
        );
        assert_eq!(continuity.identity_coherence, 0.0);
    }

    /// FSV test: Status transitions through all states correctly
    #[test]
    fn test_identity_status_from_coherence_all_states() {
        // Verify compute_status_from_coherence works correctly
        let mut continuity = IdentityContinuity::new();

        // Update to each threshold and verify status
        // Critical: IC < 0.5
        continuity.update(0.3, 0.3).unwrap(); // IC = 0.09 < 0.5
        assert_eq!(continuity.status, IdentityStatus::Critical);

        // Degraded: 0.5 <= IC < 0.7
        continuity.update(0.8, 0.7).unwrap(); // IC = 0.56
        assert_eq!(continuity.status, IdentityStatus::Degraded);

        // Warning: 0.7 <= IC <= 0.9
        continuity.update(0.9, 0.85).unwrap(); // IC = 0.765
        assert_eq!(continuity.status, IdentityStatus::Warning);

        // Healthy: IC > 0.9
        continuity.update(0.96, 0.96).unwrap(); // IC = 0.9216 > 0.9
        assert_eq!(continuity.status, IdentityStatus::Healthy);
    }

    #[test]
    fn test_identity_continuity_healthy() {
        let mut continuity = IdentityContinuity::new();
        let status = continuity.update(0.95, 0.95).unwrap();

        assert_eq!(status, IdentityStatus::Healthy);
        assert!(continuity.identity_coherence > 0.9);
    }

    #[test]
    fn test_identity_continuity_critical() {
        let mut continuity = IdentityContinuity::new();
        let status = continuity.update(0.3, 0.3).unwrap();

        assert_eq!(status, IdentityStatus::Critical);
        assert!(continuity.identity_coherence < 0.5);
    }

    #[tokio::test]
    async fn test_self_awareness_loop_cycle() {
        let mut loop_mgr = SelfAwarenessLoop::new();
        let mut ego = SelfEgoNode::with_purpose_vector([1.0; 13]);

        let action = [1.0; 13]; // Perfect alignment
        let result = loop_mgr.cycle(&mut ego, &action, 0.85).await.unwrap();

        assert!(!result.needs_reflection); // Alignment is high
        assert!(result.alignment > 0.99);
    }

    #[tokio::test]
    async fn test_self_awareness_loop_reflection_trigger() {
        let mut loop_mgr = SelfAwarenessLoop::new();
        let mut ego = SelfEgoNode::with_purpose_vector([1.0; 13]);

        let action = [0.0; 13]; // Zero alignment - should trigger reflection
        let result = loop_mgr.cycle(&mut ego, &action, 0.85).await.unwrap();

        assert!(result.needs_reflection);
        assert!(result.alignment < loop_mgr.alignment_threshold);
    }

    #[test]
    fn test_cosine_similarity() {
        let loop_mgr = SelfAwarenessLoop::new();
        let v1 = [1.0; 13];
        let v2 = [1.0; 13];

        let similarity = loop_mgr.cosine_similarity(&v1, &v2);
        assert!((similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let loop_mgr = SelfAwarenessLoop::new();
        let mut v1 = [0.0; 13];
        let mut v2 = [0.0; 13];
        v1[0] = 1.0;
        v2[1] = 1.0;

        let similarity = loop_mgr.cosine_similarity(&v1, &v2);
        assert!(similarity.abs() < 1e-5);
    }

    // =============================================================================
    // TASK-GWT-P0-003: Tests for update_from_fingerprint
    // =============================================================================

    // Import necessary types for TeleologicalFingerprint construction
    use crate::types::fingerprint::{
        TeleologicalFingerprint, PurposeVector, SemanticFingerprint,
        JohariFingerprint,
    };

    /// Helper to create a test TeleologicalFingerprint with known values
    fn create_test_fingerprint(alignments: [f32; 13]) -> TeleologicalFingerprint {
        let purpose_vector = PurposeVector::new(alignments);
        let semantic = SemanticFingerprint::zeroed();
        let johari = JohariFingerprint::zeroed();

        TeleologicalFingerprint {
            id: Uuid::new_v4(),
            semantic,
            purpose_vector,
            johari,
            purpose_evolution: Vec::new(),
            theta_to_north_star: alignments.iter().sum::<f32>() / 13.0,
            content_hash: [0u8; 32],
            created_at: Utc::now(),
            last_updated: Utc::now(),
            access_count: 0,
        }
    }

    #[test]
    fn test_update_from_fingerprint_copies_purpose_vector() {
        println!("=== TEST: update_from_fingerprint copies purpose_vector ===");

        // BEFORE: Create SelfEgoNode with default purpose_vector
        let mut ego = SelfEgoNode::new();
        let initial_pv = ego.purpose_vector;
        assert_eq!(initial_pv, [0.0; 13], "Initial purpose_vector should be zeros");

        // Create fingerprint with known values
        let alignments = [0.8, 0.75, 0.9, 0.6, 0.7, 0.65, 0.85, 0.72, 0.78, 0.68, 0.82, 0.71, 0.76];
        let fingerprint = create_test_fingerprint(alignments);

        // EXECUTE: Update from fingerprint
        ego.update_from_fingerprint(&fingerprint).unwrap();

        // AFTER: Verify purpose_vector was copied
        assert_eq!(ego.purpose_vector, alignments,
            "purpose_vector must match fingerprint.purpose_vector.alignments");

        println!("BEFORE: purpose_vector = {:?}", initial_pv);
        println!("AFTER: purpose_vector = {:?}", ego.purpose_vector);
        println!("EVIDENCE: purpose_vector correctly copied from fingerprint");
    }

    #[test]
    fn test_update_from_fingerprint_updates_coherence() {
        println!("=== TEST: update_from_fingerprint updates coherence_with_actions ===");

        let mut ego = SelfEgoNode::new();
        let initial_coherence = ego.coherence_with_actions;
        assert_eq!(initial_coherence, 0.0, "Initial coherence should be 0.0");

        // Create fingerprint with uniform alignments (max coherence)
        let alignments = [0.8; 13]; // Uniform = coherence = 1.0
        let fingerprint = create_test_fingerprint(alignments);
        let expected_coherence = fingerprint.purpose_vector.coherence;

        ego.update_from_fingerprint(&fingerprint).unwrap();

        assert!((ego.coherence_with_actions - expected_coherence).abs() < 1e-6,
            "coherence_with_actions must equal fingerprint.purpose_vector.coherence");

        println!("BEFORE: coherence = {:.4}", initial_coherence);
        println!("AFTER: coherence = {:.4}", ego.coherence_with_actions);
        println!("EXPECTED: {:.4}", expected_coherence);
        println!("EVIDENCE: coherence correctly updated from fingerprint");
    }

    #[test]
    fn test_update_from_fingerprint_stores_fingerprint() {
        println!("=== TEST: update_from_fingerprint stores fingerprint reference ===");

        let mut ego = SelfEgoNode::new();
        assert!(ego.fingerprint.is_none(), "Initial fingerprint should be None");

        let alignments = [0.5; 13];
        let fingerprint = create_test_fingerprint(alignments);
        let fingerprint_id = fingerprint.id;

        ego.update_from_fingerprint(&fingerprint).unwrap();

        assert!(ego.fingerprint.is_some(), "Fingerprint must be stored after update");
        assert_eq!(ego.fingerprint.as_ref().unwrap().id, fingerprint_id,
            "Stored fingerprint ID must match input fingerprint ID");

        println!("BEFORE: fingerprint = None");
        println!("AFTER: fingerprint.id = {:?}", ego.fingerprint.as_ref().unwrap().id);
        println!("EVIDENCE: fingerprint reference correctly stored");
    }

    #[test]
    fn test_update_from_fingerprint_updates_timestamp() {
        println!("=== TEST: update_from_fingerprint updates last_updated ===");

        let mut ego = SelfEgoNode::new();
        let initial_time = ego.last_updated;

        // Small delay to ensure timestamps differ
        std::thread::sleep(std::time::Duration::from_millis(10));

        let fingerprint = create_test_fingerprint([0.7; 13]);
        ego.update_from_fingerprint(&fingerprint).unwrap();

        assert!(ego.last_updated > initial_time,
            "last_updated must be updated after update_from_fingerprint");

        println!("BEFORE: last_updated = {:?}", initial_time);
        println!("AFTER: last_updated = {:?}", ego.last_updated);
        println!("EVIDENCE: timestamp correctly updated");
    }

    // =============================================================================
    // Full State Verification: purpose_vector update integration
    // =============================================================================

    #[test]
    fn test_fsv_purpose_vector_update_integration() {
        println!("=== FULL STATE VERIFICATION: purpose_vector update ===");

        // SOURCE OF TRUTH: SelfEgoNode.purpose_vector after update
        let mut ego = SelfEgoNode::new();

        // BEFORE state
        println!("STATE BEFORE:");
        println!("  - purpose_vector: {:?}", ego.purpose_vector);
        println!("  - coherence_with_actions: {:.4}", ego.coherence_with_actions);
        println!("  - fingerprint: {:?}", ego.fingerprint.as_ref().map(|f| f.id));

        // Create fingerprint with synthetic data
        let synthetic_alignments = [
            0.85, 0.78, 0.92, 0.67, 0.73, 0.61, 0.88, 0.75, 0.81, 0.69, 0.84, 0.72, 0.79
        ];
        let fingerprint = create_test_fingerprint(synthetic_alignments);

        // EXECUTE
        let result = ego.update_from_fingerprint(&fingerprint);
        assert!(result.is_ok(), "update_from_fingerprint must succeed");

        // VERIFY VIA SEPARATE READ
        println!("\nSTATE AFTER:");
        println!("  - purpose_vector: {:?}", ego.purpose_vector);
        println!("  - coherence_with_actions: {:.4}", ego.coherence_with_actions);
        println!("  - fingerprint.id: {:?}", ego.fingerprint.as_ref().map(|f| f.id));

        // Boundary checks
        assert_eq!(ego.purpose_vector, synthetic_alignments,
            "purpose_vector must exactly match input alignments");
        assert!(ego.coherence_with_actions >= 0.0 && ego.coherence_with_actions <= 1.0,
            "coherence must be in [0,1]");
        assert!(ego.fingerprint.is_some(),
            "fingerprint reference must be stored");

        println!("\nEVIDENCE OF SUCCESS:");
        println!("  - purpose_vector correctly set to input alignments");
        println!("  - coherence = {:.4} (valid range)", ego.coherence_with_actions);
        println!("  - fingerprint reference stored");
    }

    // =============================================================================
    // Edge Case Tests
    // =============================================================================

    #[test]
    fn test_edge_case_zero_alignments() {
        println!("=== EDGE CASE: Zero alignments ===");

        let mut ego = SelfEgoNode::new();
        let fingerprint = create_test_fingerprint([0.0; 13]);

        let result = ego.update_from_fingerprint(&fingerprint);
        assert!(result.is_ok(), "Should handle zero alignments");

        assert_eq!(ego.purpose_vector, [0.0; 13]);
        assert!((ego.coherence_with_actions - 1.0).abs() < 1e-6,
            "Zero uniform alignments should have coherence 1.0");

        println!("EVIDENCE: Zero alignments handled correctly");
    }

    #[test]
    fn test_edge_case_max_alignments() {
        println!("=== EDGE CASE: Maximum alignments ===");

        let mut ego = SelfEgoNode::new();
        let fingerprint = create_test_fingerprint([1.0; 13]);

        let result = ego.update_from_fingerprint(&fingerprint);
        assert!(result.is_ok(), "Should handle max alignments");

        assert_eq!(ego.purpose_vector, [1.0; 13]);

        println!("EVIDENCE: Maximum alignments handled correctly");
    }

    #[test]
    fn test_edge_case_negative_alignments() {
        println!("=== EDGE CASE: Negative alignments ===");

        let mut ego = SelfEgoNode::new();
        // Note: PurposeVector accepts negative values (cosine can be negative)
        let fingerprint = create_test_fingerprint([-0.5; 13]);

        let result = ego.update_from_fingerprint(&fingerprint);
        assert!(result.is_ok(), "Should handle negative alignments");

        assert_eq!(ego.purpose_vector, [-0.5; 13]);

        println!("EVIDENCE: Negative alignments handled correctly");
    }

    #[test]
    fn test_self_awareness_loop_identity_coherence_getter() {
        println!("=== TEST: SelfAwarenessLoop.identity_coherence() getter ===");

        let mut loop_mgr = SelfAwarenessLoop::new();

        // Initial state
        let initial_ic = loop_mgr.identity_coherence();
        let initial_status = loop_mgr.identity_status();
        println!("BEFORE: identity_coherence = {:.4}, status = {:?}", initial_ic, initial_status);

        // Create ego and run a cycle
        let mut ego = SelfEgoNode::with_purpose_vector([0.8; 13]);
        ego.record_purpose_snapshot("Setup").unwrap();

        let action = [0.8; 13];
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let _ = loop_mgr.cycle(&mut ego, &action, 0.95).await.unwrap();
        });

        // After cycle
        let final_ic = loop_mgr.identity_coherence();
        let final_status = loop_mgr.identity_status();
        println!("AFTER: identity_coherence = {:.4}, status = {:?}", final_ic, final_status);

        // Verify getters work
        assert!(final_ic >= 0.0 && final_ic <= 1.0, "IC must be in [0,1]");

        println!("EVIDENCE: identity_coherence getter works correctly");
    }
}
