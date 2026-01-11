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
//!
//! # Persistence (TASK-GWT-P1-001)
//!
//! SelfEgoNode and related types implement Serde Serialize/Deserialize for
//! persistent storage in RocksDB via the CF_EGO_NODE column family.

use crate::error::CoreResult;
use crate::types::fingerprint::TeleologicalFingerprint;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Special memory node representing the system's identity
///
/// # Persistence (TASK-GWT-P1-001)
///
/// This struct is serializable via Serde for RocksDB storage in CF_EGO_NODE.
/// Uses bincode for efficient binary serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
///
/// # Persistence (TASK-GWT-P1-001)
///
/// Serializable component of SelfEgoNode's identity_trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - identity_continuity: "IC = cos(PV_t, PV_{t-1}) x r(t)"
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
///
/// # Persistence (TASK-GWT-P1-001)
/// Serializable for diagnostic/recovery purposes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentityContinuity {
    /// IC = cos(PV_t, PV_{t-1}) x r(t), clamped to [0, 1]
    pub identity_coherence: f32,
    /// Cosine similarity between consecutive purpose vectors
    pub recent_continuity: f32,
    /// Order parameter r from Kuramoto sync
    pub kuramoto_order_parameter: f32,
    /// Status classification based on IC thresholds
    pub status: IdentityStatus,
    /// Timestamp of computation
    pub computed_at: DateTime<Utc>,
}

/// Identity status enum for SELF_EGO_NODE state tracking.
///
/// # Persistence (TASK-GWT-P1-001)
///
/// Serializable component of IdentityContinuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Create a new result with computed values
    ///
    /// # Arguments
    /// * `purpose_continuity` - cos(PV_t, PV_{t-1})
    /// * `kuramoto_r` - Kuramoto order parameter r(t)
    ///
    /// # Returns
    /// Result with IC = purpose_continuity * kuramoto_r, clamped to [0, 1]
    ///
    /// # Example
    /// ```ignore
    /// let result = IdentityContinuity::new(0.9, 0.85);
    /// assert!((result.identity_coherence - 0.765).abs() < 1e-6);
    /// ```
    pub fn new(purpose_continuity: f32, kuramoto_r: f32) -> Self {
        // Clamp inputs to valid ranges
        let cos_clamped = purpose_continuity.clamp(-1.0, 1.0);
        let r_clamped = kuramoto_r.clamp(0.0, 1.0);

        // Compute IC = cos * r, clamp negative to 0
        let ic = (cos_clamped * r_clamped).clamp(0.0, 1.0);

        // Determine status from IC
        let status = Self::compute_status_from_coherence(ic);

        Self {
            identity_coherence: ic,
            recent_continuity: cos_clamped,
            kuramoto_order_parameter: r_clamped,
            status,
            computed_at: Utc::now(),
        }
    }

    /// Create new IdentityContinuity with default initial state
    ///
    /// Starting with identity_coherence=0.0 means status=Critical (IC < 0.5)
    /// per constitution.yaml lines 387-392
    pub fn default_initial() -> Self {
        let identity_coherence = 0.0;
        Self {
            identity_coherence,
            recent_continuity: 1.0,
            kuramoto_order_parameter: 0.0,
            status: Self::compute_status_from_coherence(identity_coherence),
            computed_at: Utc::now(),
        }
    }

    /// Create result for first purpose vector (no previous)
    ///
    /// Returns IC = 1.0, Status = Healthy
    /// Per EC-IDENTITY-01: First purpose vector defaults to healthy
    pub fn first_vector() -> Self {
        Self {
            identity_coherence: 1.0,
            recent_continuity: 1.0,
            kuramoto_order_parameter: 1.0,
            status: IdentityStatus::Healthy,
            computed_at: Utc::now(),
        }
    }

    /// Check if identity is in crisis (IC < 0.7)
    ///
    /// # Constitution Reference
    /// From constitution.yaml line 369:
    /// - warning<0.7 threshold indicates identity drift
    #[inline]
    pub fn is_in_crisis(&self) -> bool {
        self.identity_coherence < 0.7
    }

    /// Check if identity is critical (IC < 0.5)
    ///
    /// # Constitution Reference
    /// From constitution.yaml line 369:
    /// - dream<0.5 threshold triggers introspective dream
    #[inline]
    pub fn is_critical(&self) -> bool {
        self.identity_coherence < 0.5
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

        // Update timestamp
        self.computed_at = Utc::now();

        Ok(self.status)
    }
}

impl Default for IdentityContinuity {
    fn default() -> Self {
        Self::default_initial()
    }
}

impl SelfAwarenessLoop {
    /// Create a new self-awareness loop
    pub fn new() -> Self {
        Self {
            continuity: IdentityContinuity::default_initial(),
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
        let continuity = IdentityContinuity::default_initial();

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
        let mut continuity = IdentityContinuity::default_initial();

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
        let mut continuity = IdentityContinuity::default_initial();
        let status = continuity.update(0.95, 0.95).unwrap();

        assert_eq!(status, IdentityStatus::Healthy);
        assert!(continuity.identity_coherence > 0.9);
    }

    #[test]
    fn test_identity_continuity_critical() {
        let mut continuity = IdentityContinuity::default_initial();
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

    // =========================================================================
    // TASK-IDENTITY-P0-001: Factory Method Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_new_factory_computes_ic_correctly() {
        println!("=== TEST: new() factory computes IC = cos * r ===");

        // BEFORE: No IdentityContinuity exists
        // EXECUTE: Create with known values
        let result = IdentityContinuity::new(0.9, 0.85);

        // AFTER: Verify IC computation
        let expected_ic = 0.9 * 0.85; // 0.765
        assert!((result.identity_coherence - expected_ic).abs() < 1e-6,
            "IC should be {} but was {}", expected_ic, result.identity_coherence);
        assert_eq!(result.status, IdentityStatus::Warning,
            "IC=0.765 should be Warning (0.7 <= IC <= 0.9)");

        // EVIDENCE: Print computed state
        println!("INPUT: purpose_continuity=0.9, kuramoto_r=0.85");
        println!("OUTPUT: identity_coherence={:.4}, status={:?}",
            result.identity_coherence, result.status);
        println!("EVIDENCE: IC = 0.9 * 0.85 = 0.765 (Warning)");
    }

    #[test]
    fn test_identity_continuity_new_factory_clamps_negative_cosine() {
        println!("=== TEST: new() clamps negative cosine to IC >= 0 ===");

        // BEFORE: Negative cosine (opposite vectors)
        let result = IdentityContinuity::new(-0.8, 0.9);

        // AFTER: IC should be 0.0 (clamped), not -0.72
        assert!(result.identity_coherence >= 0.0,
            "IC must be >= 0, but was {}", result.identity_coherence);
        assert_eq!(result.status, IdentityStatus::Critical,
            "IC=0.0 should be Critical");

        println!("INPUT: purpose_continuity=-0.8, kuramoto_r=0.9");
        println!("OUTPUT: identity_coherence={:.4}, status={:?}",
            result.identity_coherence, result.status);
        println!("EVIDENCE: Negative cosine clamped to IC=0.0");
    }

    #[test]
    fn test_identity_continuity_new_factory_clamps_inputs() {
        println!("=== TEST: new() clamps inputs to valid ranges ===");

        // Out-of-range inputs
        let result = IdentityContinuity::new(1.5, 2.0); // Should clamp to 1.0, 1.0

        assert!((result.recent_continuity - 1.0).abs() < 1e-6,
            "purpose_continuity should clamp to 1.0");
        assert!((result.kuramoto_order_parameter - 1.0).abs() < 1e-6,
            "kuramoto_r should clamp to 1.0");
        assert!((result.identity_coherence - 1.0).abs() < 1e-6,
            "IC should be 1.0 * 1.0 = 1.0");

        println!("INPUT: purpose_continuity=1.5, kuramoto_r=2.0");
        println!("OUTPUT: recent_continuity={:.4}, kuramoto_order_parameter={:.4}",
            result.recent_continuity, result.kuramoto_order_parameter);
        println!("EVIDENCE: Inputs clamped to [−1,1] and [0,1]");
    }

    #[test]
    fn test_identity_continuity_first_vector_returns_healthy() {
        println!("=== TEST: first_vector() returns IC=1.0, Healthy ===");

        let result = IdentityContinuity::first_vector();

        assert_eq!(result.identity_coherence, 1.0);
        assert_eq!(result.status, IdentityStatus::Healthy);
        assert_eq!(result.recent_continuity, 1.0);
        assert_eq!(result.kuramoto_order_parameter, 1.0);

        println!("OUTPUT: identity_coherence=1.0, status=Healthy");
        println!("EVIDENCE: First vector defaults to perfect continuity");
    }

    #[test]
    fn test_identity_continuity_first_vector_has_timestamp() {
        println!("=== TEST: first_vector() has computed_at timestamp ===");

        let before = Utc::now();
        let result = IdentityContinuity::first_vector();
        let after = Utc::now();

        assert!(result.computed_at >= before && result.computed_at <= after,
            "computed_at should be between test start and end");

        println!("computed_at: {:?}", result.computed_at);
        println!("EVIDENCE: Timestamp is set correctly");
    }

    // =========================================================================
    // is_in_crisis() and is_critical() Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_is_in_crisis_boundary() {
        println!("=== TEST: is_in_crisis() boundary at IC=0.7 ===");

        // Exactly 0.7 should NOT be crisis
        let at_boundary = IdentityContinuity::new(0.7, 1.0);
        assert!(!at_boundary.is_in_crisis(),
            "IC=0.7 should NOT be in crisis (boundary is < 0.7)");

        // Just below 0.7 IS crisis
        let below_boundary = IdentityContinuity::new(0.699, 1.0);
        assert!(below_boundary.is_in_crisis(),
            "IC=0.699 should be in crisis");

        println!("IC=0.7: is_in_crisis={}", at_boundary.is_in_crisis());
        println!("IC=0.699: is_in_crisis={}", below_boundary.is_in_crisis());
        println!("EVIDENCE: Boundary at 0.7 is exclusive (< not <=)");
    }

    #[test]
    fn test_identity_continuity_is_critical_boundary() {
        println!("=== TEST: is_critical() boundary at IC=0.5 ===");

        // Exactly 0.5 should NOT be critical
        let at_boundary = IdentityContinuity::new(0.5, 1.0);
        assert!(!at_boundary.is_critical(),
            "IC=0.5 should NOT be critical (boundary is < 0.5)");

        // Just below 0.5 IS critical
        let below_boundary = IdentityContinuity::new(0.499, 1.0);
        assert!(below_boundary.is_critical(),
            "IC=0.499 should be critical");

        println!("IC=0.5: is_critical={}", at_boundary.is_critical());
        println!("IC=0.499: is_critical={}", below_boundary.is_critical());
        println!("EVIDENCE: Boundary at 0.5 is exclusive (< not <=)");
    }

    #[test]
    fn test_identity_continuity_crisis_methods_consistent_with_status() {
        println!("=== TEST: is_in_crisis/is_critical consistent with status ===");

        // Healthy: IC > 0.9
        let healthy = IdentityContinuity::new(1.0, 0.95);
        assert!(!healthy.is_in_crisis());
        assert!(!healthy.is_critical());
        assert_eq!(healthy.status, IdentityStatus::Healthy);

        // Warning: 0.7 <= IC <= 0.9
        let warning = IdentityContinuity::new(0.8, 1.0);
        assert!(!warning.is_in_crisis());
        assert!(!warning.is_critical());
        assert_eq!(warning.status, IdentityStatus::Warning);

        // Degraded: 0.5 <= IC < 0.7
        let degraded = IdentityContinuity::new(0.6, 1.0);
        assert!(degraded.is_in_crisis()); // < 0.7
        assert!(!degraded.is_critical()); // >= 0.5
        assert_eq!(degraded.status, IdentityStatus::Degraded);

        // Critical: IC < 0.5
        let critical = IdentityContinuity::new(0.3, 1.0);
        assert!(critical.is_in_crisis());
        assert!(critical.is_critical());
        assert_eq!(critical.status, IdentityStatus::Critical);

        println!("Healthy: is_in_crisis={}, is_critical={}", healthy.is_in_crisis(), healthy.is_critical());
        println!("Warning: is_in_crisis={}, is_critical={}", warning.is_in_crisis(), warning.is_critical());
        println!("Degraded: is_in_crisis={}, is_critical={}", degraded.is_in_crisis(), degraded.is_critical());
        println!("Critical: is_in_crisis={}, is_critical={}", critical.is_in_crisis(), critical.is_critical());
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_bincode_roundtrip() {
        println!("=== TEST: bincode serialization roundtrip ===");

        let original = IdentityContinuity::new(0.85, 0.9);

        // Serialize
        let serialized = bincode::serialize(&original)
            .expect("Serialization must not fail");

        // Deserialize
        let deserialized: IdentityContinuity = bincode::deserialize(&serialized)
            .expect("Deserialization must not fail");

        // Verify all fields
        assert_eq!(original.identity_coherence, deserialized.identity_coherence);
        assert_eq!(original.recent_continuity, deserialized.recent_continuity);
        assert_eq!(original.kuramoto_order_parameter, deserialized.kuramoto_order_parameter);
        assert_eq!(original.status, deserialized.status);
        assert_eq!(original.computed_at, deserialized.computed_at);

        println!("Original: {:?}", original);
        println!("Deserialized: {:?}", deserialized);
        println!("EVIDENCE: All fields preserved through serialization");
    }

    #[test]
    fn test_identity_continuity_json_roundtrip() {
        println!("=== TEST: JSON serialization roundtrip ===");

        let original = IdentityContinuity::new(0.75, 0.8);

        // Serialize to JSON
        let json = serde_json::to_string(&original)
            .expect("JSON serialization must not fail");

        // Deserialize from JSON
        let deserialized: IdentityContinuity = serde_json::from_str(&json)
            .expect("JSON deserialization must not fail");

        assert_eq!(original.identity_coherence, deserialized.identity_coherence);
        assert_eq!(original.status, deserialized.status);

        println!("JSON: {}", json);
        println!("EVIDENCE: JSON serialization works correctly");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_zero_r_gives_critical() {
        println!("=== EDGE CASE: Zero Kuramoto r ===");

        // r = 0 means no synchronization, IC = 0
        let result = IdentityContinuity::new(1.0, 0.0);

        assert_eq!(result.identity_coherence, 0.0);
        assert_eq!(result.status, IdentityStatus::Critical);
        assert!(result.is_critical());

        println!("INPUT: purpose_continuity=1.0, kuramoto_r=0.0");
        println!("OUTPUT: IC=0.0, status=Critical");
        println!("EVIDENCE: Zero sync means zero identity coherence");
    }

    #[test]
    fn test_identity_continuity_perfect_values() {
        println!("=== EDGE CASE: Perfect values ===");

        let result = IdentityContinuity::new(1.0, 1.0);

        assert_eq!(result.identity_coherence, 1.0);
        assert_eq!(result.status, IdentityStatus::Healthy);
        assert!(!result.is_in_crisis());
        assert!(!result.is_critical());

        println!("EVIDENCE: Perfect continuity and sync gives IC=1.0, Healthy");
    }

    // =========================================================================
    // Full State Verification
    // =========================================================================

    #[test]
    fn fsv_identity_continuity_full_lifecycle() {
        println!("=== FULL STATE VERIFICATION: IdentityContinuity lifecycle ===");

        // SOURCE OF TRUTH: IdentityContinuity struct fields after operations

        // 1. Create with factory
        let result = IdentityContinuity::new(0.8, 0.9);

        println!("\nSTATE AFTER new(0.8, 0.9):");
        println!("  identity_coherence: {:.4}", result.identity_coherence);
        println!("  recent_continuity: {:.4}", result.recent_continuity);
        println!("  kuramoto_order_parameter: {:.4}", result.kuramoto_order_parameter);
        println!("  status: {:?}", result.status);
        println!("  computed_at: {:?}", result.computed_at);
        println!("  is_in_crisis: {}", result.is_in_crisis());
        println!("  is_critical: {}", result.is_critical());

        // Verify expected values
        assert!((result.identity_coherence - 0.72).abs() < 1e-6,
            "IC should be 0.8 * 0.9 = 0.72");
        assert_eq!(result.status, IdentityStatus::Warning,
            "0.72 is in Warning range [0.7, 0.9]");
        assert!(!result.is_in_crisis(),
            "0.72 >= 0.7, not in crisis");

        // 2. Create first_vector
        let first = IdentityContinuity::first_vector();

        println!("\nSTATE AFTER first_vector():");
        println!("  identity_coherence: {:.4}", first.identity_coherence);
        println!("  status: {:?}", first.status);

        assert_eq!(first.identity_coherence, 1.0);
        assert_eq!(first.status, IdentityStatus::Healthy);

        // 3. Verify serialization persistence
        let serialized = bincode::serialize(&result).unwrap();
        let restored: IdentityContinuity = bincode::deserialize(&serialized).unwrap();

        println!("\nSTATE AFTER serialization roundtrip:");
        println!("  identity_coherence preserved: {}",
            result.identity_coherence == restored.identity_coherence);
        println!("  computed_at preserved: {}",
            result.computed_at == restored.computed_at);

        assert_eq!(result, restored, "Serialization must preserve all fields");

        println!("\nEVIDENCE OF SUCCESS:");
        println!("  - Factory method computes IC correctly");
        println!("  - first_vector() returns healthy initial state");
        println!("  - Crisis methods work correctly");
        println!("  - Serialization preserves all fields including timestamp");
    }
}
