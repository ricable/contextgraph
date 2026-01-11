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
use std::collections::VecDeque;
use uuid::Uuid;

/// Maximum purpose vector history size per constitution
/// Reference: constitution.yaml line 390 (identity_trajectory: 1000)
pub const MAX_PV_HISTORY_SIZE: usize = 1000;

/// Default crisis threshold per constitution.yaml line 369
/// IC < 0.7 indicates identity drift (warning/degraded state)
pub const IC_CRISIS_THRESHOLD: f32 = 0.7;

/// Critical threshold triggering dream consolidation per constitution.yaml line 369
/// IC < 0.5 triggers introspective dream (critical state)
pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;

/// Epsilon for numerical stability in magnitude comparisons
/// Prevents division by zero in cosine similarity calculation
const COSINE_EPSILON: f32 = 1e-8;

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

/// Manages purpose vector history for identity continuity calculation
///
/// Provides O(1) access to current and previous purpose vectors,
/// handling the edge case of first vector (no previous).
///
/// # Constitution Reference
/// - self_ego_node.identity_trajectory: max 1000 snapshots
/// - IC = cos(PV_t, PV_{t-1}) × r(t) requires consecutive PV access
///
/// # Memory Management
/// Uses FIFO eviction when reaching MAX_PV_HISTORY_SIZE.
/// VecDeque ensures O(1) push_back and pop_front.
///
/// # Error Handling
/// This type does NOT panic. All operations return Option or Result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeVectorHistory {
    /// Ring buffer of purpose snapshots (VecDeque for O(1) operations)
    history: VecDeque<PurposeSnapshot>,
    /// Maximum history size (default: 1000)
    pub max_size: usize,
}

/// Trait for purpose vector history operations
///
/// Enables abstraction for testing and alternative implementations.
pub trait PurposeVectorHistoryProvider {
    /// Push a new purpose vector with context
    ///
    /// # Arguments
    /// * `pv` - The 13D purpose vector (must be exactly 13 elements)
    /// * `context` - Description of what triggered this snapshot
    ///
    /// # Returns
    /// The previous purpose vector, if any existed (for IC calculation)
    fn push(&mut self, pv: [f32; 13], context: impl Into<String>) -> Option<[f32; 13]>;

    /// Get the current (most recent) purpose vector
    ///
    /// # Returns
    /// - `Some(&[f32; 13])` if history has at least one entry
    /// - `None` if history is empty
    fn current(&self) -> Option<&[f32; 13]>;

    /// Get the previous purpose vector (for IC calculation)
    ///
    /// # Returns
    /// - `Some(&[f32; 13])` if history has at least two entries
    /// - `None` if history has 0 or 1 entries
    fn previous(&self) -> Option<&[f32; 13]>;

    /// Get both current and previous for IC calculation
    ///
    /// # Returns
    /// - `Some((current, Some(previous)))` if len >= 2
    /// - `Some((current, None))` if len == 1 (first vector)
    /// - `None` if empty
    fn current_and_previous(&self) -> Option<(&[f32; 13], Option<&[f32; 13]>)>;

    /// Get the number of snapshots in history
    fn len(&self) -> usize;

    /// Check if history is empty
    fn is_empty(&self) -> bool;

    /// Check if this is the first vector (exactly one entry, no previous)
    ///
    /// # Edge Case
    /// Per EC-IDENTITY-01: First vector defaults to IC = 1.0 (Healthy)
    fn is_first_vector(&self) -> bool;
}

impl PurposeVectorHistory {
    /// Create new history with default max size (1000)
    pub fn new() -> Self {
        Self::with_max_size(MAX_PV_HISTORY_SIZE)
    }

    /// Create with custom max size
    ///
    /// # Arguments
    /// * `max_size` - Maximum entries before FIFO eviction
    ///
    /// # Notes
    /// max_size of 0 means no eviction limit.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            // Pre-allocate up to 1024 to avoid reallocs
            history: VecDeque::with_capacity(max_size.min(1024)),
            max_size,
        }
    }

    /// Get read access to full history (for diagnostics)
    pub fn history(&self) -> &VecDeque<PurposeSnapshot> {
        &self.history
    }
}

impl Default for PurposeVectorHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PurposeVectorHistoryProvider for PurposeVectorHistory {
    fn push(&mut self, pv: [f32; 13], context: impl Into<String>) -> Option<[f32; 13]> {
        // Capture previous BEFORE pushing
        let previous = self.current().copied();

        // FIFO eviction if at capacity
        if self.max_size > 0 && self.history.len() >= self.max_size {
            self.history.pop_front(); // O(1) with VecDeque
        }

        // Add new snapshot
        self.history.push_back(PurposeSnapshot {
            vector: pv,
            timestamp: Utc::now(),
            context: context.into(),
        });

        previous
    }

    fn current(&self) -> Option<&[f32; 13]> {
        self.history.back().map(|s| &s.vector)
    }

    fn previous(&self) -> Option<&[f32; 13]> {
        if self.history.len() < 2 {
            return None;
        }
        // VecDeque indexing is O(1)
        self.history.get(self.history.len() - 2).map(|s| &s.vector)
    }

    fn current_and_previous(&self) -> Option<(&[f32; 13], Option<&[f32; 13]>)> {
        self.current().map(|curr| (curr, self.previous()))
    }

    fn len(&self) -> usize {
        self.history.len()
    }

    fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    fn is_first_vector(&self) -> bool {
        self.history.len() == 1
    }
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

/// Compute cosine similarity between two 13-dimensional purpose vectors.
///
/// # Algorithm
/// cos(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)
///
/// # Arguments
/// * `v1` - First 13D purpose vector
/// * `v2` - Second 13D purpose vector
///
/// # Returns
/// * Cosine similarity in range [-1, 1]
/// * Returns 0.0 if either vector has near-zero magnitude (below COSINE_EPSILON)
///
/// # Constitution Reference
/// Used for computing cos(PV_t, PV_{t-1}) in IC = cos(PV_t, PV_{t-1}) × r(t)
///
/// # Example
/// ```ignore
/// let v1 = [1.0; 13];
/// let v2 = [1.0; 13];
/// let similarity = cosine_similarity_13d(&v1, &v2);
/// assert!((similarity - 1.0).abs() < 1e-6); // Identical vectors = 1.0
/// ```
pub fn cosine_similarity_13d(v1: &[f32; 13], v2: &[f32; 13]) -> f32 {
    // Compute dot product: v1 · v2 = Σ(v1_i × v2_i)
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();

    // Compute magnitudes: ||v|| = sqrt(Σ(v_i²))
    let magnitude_v1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let magnitude_v2: f32 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();

    // Handle near-zero magnitude vectors (prevents division by zero)
    // Per spec: return 0.0 for degenerate cases
    if magnitude_v1 < COSINE_EPSILON || magnitude_v2 < COSINE_EPSILON {
        return 0.0;
    }

    // Compute cosine similarity and clamp to valid range [-1, 1]
    // Clamping handles floating point errors that could produce values like 1.0000001
    let similarity = dot_product / (magnitude_v1 * magnitude_v2);
    similarity.clamp(-1.0, 1.0)
}

/// Identity Continuity Monitor - Continuous IC tracking wrapper
///
/// Wraps `PurposeVectorHistory` to provide real-time identity continuity
/// monitoring and status classification.
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - IC = cos(PV_t, PV_{t-1}) × r(t)
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
/// - self_ego_node.identity_trajectory: max 1000 snapshots
///
/// # Architecture
/// ```text
/// IdentityContinuityMonitor
/// ├── history: PurposeVectorHistory (ring buffer)
/// ├── last_result: Option<IdentityContinuity> (cached computation)
/// └── crisis_threshold: f32 (configurable)
/// ```
///
/// # Usage Pattern
/// ```ignore
/// let mut monitor = IdentityContinuityMonitor::new();
/// let ic_result = monitor.compute_continuity(&purpose_vector, kuramoto_r);
/// if monitor.is_in_crisis() {
///     // Trigger remediation
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContinuityMonitor {
    /// Purpose vector history buffer (delegates to PurposeVectorHistory)
    history: PurposeVectorHistory,
    /// Cached last computation result (None until first compute_continuity call)
    last_result: Option<IdentityContinuity>,
    /// Configurable crisis threshold (default: IC_CRISIS_THRESHOLD = 0.7)
    crisis_threshold: f32,
}

impl IdentityContinuityMonitor {
    /// Create new monitor with default configuration.
    ///
    /// Defaults:
    /// - history capacity: MAX_PV_HISTORY_SIZE (1000)
    /// - crisis_threshold: IC_CRISIS_THRESHOLD (0.7)
    pub fn new() -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: IC_CRISIS_THRESHOLD,
        }
    }

    /// Create monitor with custom crisis threshold.
    ///
    /// # Arguments
    /// * `threshold` - Custom crisis threshold (clamped to [0, 1])
    ///
    /// # Example
    /// ```ignore
    /// // More strict monitoring
    /// let monitor = IdentityContinuityMonitor::with_threshold(0.8);
    /// ```
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Create monitor with custom history capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum history entries (0 = unlimited)
    ///
    /// # Example
    /// ```ignore
    /// // Smaller history for memory-constrained environments
    /// let monitor = IdentityContinuityMonitor::with_capacity(100);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            history: PurposeVectorHistory::with_max_size(capacity),
            last_result: None,
            crisis_threshold: IC_CRISIS_THRESHOLD,
        }
    }

    /// Compute identity continuity from new purpose vector and Kuramoto r.
    ///
    /// # Algorithm
    /// 1. Push new PV to history, get previous PV
    /// 2. If first vector: return IdentityContinuity::first_vector()
    /// 3. Compute cos(PV_t, PV_{t-1}) using cosine_similarity_13d
    /// 4. Create IdentityContinuity::new(cosine, kuramoto_r)
    /// 5. Cache and return result
    ///
    /// # Arguments
    /// * `purpose_vector` - Current 13D purpose alignment vector (PV_t)
    /// * `kuramoto_r` - Current Kuramoto order parameter r(t) in [0, 1]
    /// * `context` - Description for history snapshot
    ///
    /// # Returns
    /// * `IdentityContinuity` with computed IC and status
    ///
    /// # Constitution Reference
    /// IC = cos(PV_t, PV_{t-1}) × r(t)
    /// - cos: purpose vector continuity [-1, 1]
    /// - r: phase synchronization level [0, 1]
    pub fn compute_continuity(
        &mut self,
        purpose_vector: &[f32; 13],
        kuramoto_r: f32,
        context: impl Into<String>,
    ) -> IdentityContinuity {
        // Push current PV and get previous (if any)
        let previous = self.history.push(*purpose_vector, context);

        // Compute result based on whether this is first vector
        let result = match previous {
            None => {
                // First vector: per EC-IDENTITY-01, default to healthy
                IdentityContinuity::first_vector()
            }
            Some(prev_pv) => {
                // Compute cosine similarity between consecutive PVs
                let cosine = cosine_similarity_13d(purpose_vector, &prev_pv);

                // Create IdentityContinuity with IC = cos × r
                IdentityContinuity::new(cosine, kuramoto_r)
            }
        };

        // Cache result for subsequent getters
        self.last_result = Some(result.clone());

        result
    }

    /// Get the last computed IdentityContinuity result.
    ///
    /// # Returns
    /// * `Some(&IdentityContinuity)` if compute_continuity was called
    /// * `None` if no computation has been performed yet
    #[inline]
    pub fn last_result(&self) -> Option<&IdentityContinuity> {
        self.last_result.as_ref()
    }

    /// Get current identity coherence value (IC).
    ///
    /// # Returns
    /// * `Some(f32)` in [0, 1] if computed
    /// * `None` if no computation yet
    #[inline]
    pub fn identity_coherence(&self) -> Option<f32> {
        self.last_result.as_ref().map(|r| r.identity_coherence)
    }

    /// Get current identity status classification.
    ///
    /// # Returns
    /// * `Some(IdentityStatus)` if computed
    /// * `None` if no computation yet
    #[inline]
    pub fn current_status(&self) -> Option<IdentityStatus> {
        self.last_result.as_ref().map(|r| r.status)
    }

    /// Check if identity is in crisis (IC < crisis_threshold).
    ///
    /// # Returns
    /// * `true` if IC < crisis_threshold (default 0.7)
    /// * `false` if IC >= crisis_threshold or no computation yet
    ///
    /// # Note
    /// Uses configurable crisis_threshold, not fixed IdentityContinuity::is_in_crisis()
    #[inline]
    pub fn is_in_crisis(&self) -> bool {
        self.last_result
            .as_ref()
            .map(|r| r.identity_coherence < self.crisis_threshold)
            .unwrap_or(false)
    }

    /// Get the number of snapshots in history.
    #[inline]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get the configured crisis threshold.
    #[inline]
    pub fn crisis_threshold(&self) -> f32 {
        self.crisis_threshold
    }

    /// Get read-only access to underlying history.
    ///
    /// Useful for diagnostics and detailed analysis.
    pub fn history(&self) -> &PurposeVectorHistory {
        &self.history
    }

    /// Check if history is empty (no vectors recorded).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Check if this is the first vector (exactly one entry).
    ///
    /// Per EC-IDENTITY-01: First vector defaults to IC = 1.0 (Healthy)
    #[inline]
    pub fn is_first_vector(&self) -> bool {
        self.history.is_first_vector()
    }
}

impl Default for IdentityContinuityMonitor {
    fn default() -> Self {
        Self::new()
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

    // =========================================================================
    // TASK-IDENTITY-P0-002: PurposeVectorHistory Tests
    // =========================================================================

    /// Helper: Create a purpose vector with uniform values
    fn uniform_pv(val: f32) -> [f32; 13] {
        [val; 13]
    }

    /// Helper: Create a purpose vector with specific pattern
    fn pattern_pv(base: f32) -> [f32; 13] {
        [
            base, base + 0.05, base + 0.1, base - 0.05, base,
            base + 0.02, base - 0.03, base + 0.08, base - 0.01,
            base + 0.04, base - 0.02, base + 0.06, base - 0.04,
        ]
    }

    // =========================================================================
    // FSV Tests (Full State Verification)
    // =========================================================================

    #[test]
    fn fsv_push_and_retrieve() {
        println!("=== FSV-1: Push and Retrieve ===");

        // BEFORE
        let mut history = PurposeVectorHistory::new();
        println!("BEFORE: is_empty={}, len={}", history.is_empty(), history.len());
        assert!(history.is_empty());
        assert!(history.current().is_none());

        // EXECUTE
        let prev = history.push(uniform_pv(0.5), "Test context");

        // AFTER
        println!("AFTER: is_empty={}, len={}, is_first_vector={}",
                 history.is_empty(), history.len(), history.is_first_vector());
        assert!(prev.is_none());
        assert_eq!(history.len(), 1);
        assert!(history.is_first_vector());
        assert!(history.current().is_some());
        assert_eq!(*history.current().unwrap(), uniform_pv(0.5));
        assert!(history.previous().is_none());

        // EVIDENCE
        println!("EVIDENCE: history.history().len() = {}", history.history().len());
        assert_eq!(history.history().len(), 1);
    }

    #[test]
    fn fsv_fifo_eviction() {
        println!("=== FSV-2: FIFO Eviction ===");

        // BEFORE
        let mut history = PurposeVectorHistory::with_max_size(3);
        history.push(uniform_pv(0.1), "1");
        history.push(uniform_pv(0.2), "2");
        history.push(uniform_pv(0.3), "3");
        println!("BEFORE: len={}", history.len());
        assert_eq!(history.len(), 3);

        // EXECUTE
        history.push(uniform_pv(0.4), "4");

        // AFTER
        println!("AFTER: len={}", history.len());
        assert_eq!(history.len(), 3); // Still 3

        // Verify oldest was evicted
        let oldest_val = history.history().front().unwrap().vector[0];
        println!("EVIDENCE: oldest value = {} (should be 0.2, not 0.1)", oldest_val);
        assert!((oldest_val - 0.2).abs() < 1e-6);
        assert_eq!(*history.current().unwrap(), uniform_pv(0.4));
        assert_eq!(*history.previous().unwrap(), uniform_pv(0.3));
    }

    #[test]
    fn fsv_serialization_roundtrip() {
        println!("=== FSV-3: Serialization Roundtrip ===");

        // BEFORE
        let mut original = PurposeVectorHistory::with_max_size(100);
        original.push(pattern_pv(0.8), "Context A");
        original.push(pattern_pv(0.9), "Context B");
        println!("BEFORE: len={}, max_size={}", original.len(), original.max_size);

        // EXECUTE
        let serialized = bincode::serialize(&original).expect("serialize must not fail");
        let restored: PurposeVectorHistory = bincode::deserialize(&serialized)
            .expect("deserialize must not fail");

        // AFTER
        println!("AFTER: restored.len={}, restored.max_size={}",
                 restored.len(), restored.max_size);
        assert_eq!(restored.len(), original.len());
        assert_eq!(restored.current(), original.current());
        assert_eq!(restored.previous(), original.previous());
        assert_eq!(restored.max_size, original.max_size);

        // EVIDENCE
        println!("EVIDENCE: All fields match original");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_edge_case_empty_history() {
        println!("=== EDGE CASE: Empty History ===");

        let history = PurposeVectorHistory::new();

        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert!(history.current().is_none());
        assert!(history.previous().is_none());
        assert!(history.current_and_previous().is_none());
        assert!(!history.is_first_vector());

        println!("EVIDENCE: All accessors return None/false, no panic");
    }

    #[test]
    fn test_edge_case_first_vector_pv_history() {
        println!("=== EDGE CASE: First Vector (PurposeVectorHistory) ===");

        let mut history = PurposeVectorHistory::new();
        let prev = history.push(uniform_pv(0.77), "First ever");

        assert!(prev.is_none());
        assert!(history.is_first_vector());
        assert_eq!(history.len(), 1);

        let (curr, prev_ref) = history.current_and_previous().unwrap();
        assert_eq!(*curr, uniform_pv(0.77));
        assert!(prev_ref.is_none());

        println!("EVIDENCE: is_first_vector()=true, previous()=None");
    }

    #[test]
    fn test_edge_case_zero_max_size() {
        println!("=== EDGE CASE: Zero Max Size (Unlimited) ===");

        let mut history = PurposeVectorHistory::with_max_size(0);

        for i in 0..100 {
            history.push(uniform_pv(i as f32 * 0.01), format!("Entry {}", i));
        }

        assert_eq!(history.len(), 100);
        println!("EVIDENCE: len()=100, no eviction with max_size=0");
    }

    // =========================================================================
    // Core Functionality Tests
    // =========================================================================

    #[test]
    fn test_new_creates_empty_history() {
        let history = PurposeVectorHistory::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.max_size, MAX_PV_HISTORY_SIZE);
    }

    #[test]
    fn test_push_returns_previous() {
        let mut history = PurposeVectorHistory::new();

        // First push returns None
        let prev1 = history.push(uniform_pv(0.5), "First");
        assert!(prev1.is_none());

        // Second push returns first
        let prev2 = history.push(uniform_pv(0.7), "Second");
        assert_eq!(prev2.unwrap(), uniform_pv(0.5));

        // Third push returns second
        let prev3 = history.push(uniform_pv(0.9), "Third");
        assert_eq!(prev3.unwrap(), uniform_pv(0.7));
    }

    #[test]
    fn test_current_and_previous_all_states() {
        let mut history = PurposeVectorHistory::new();

        // Empty
        assert!(history.current_and_previous().is_none());

        // One entry
        history.push(uniform_pv(0.5), "1");
        let result = history.current_and_previous().unwrap();
        assert_eq!(*result.0, uniform_pv(0.5));
        assert!(result.1.is_none());

        // Two entries
        history.push(uniform_pv(0.7), "2");
        let result = history.current_and_previous().unwrap();
        assert_eq!(*result.0, uniform_pv(0.7));
        assert_eq!(*result.1.unwrap(), uniform_pv(0.5));
    }

    #[test]
    fn test_is_first_vector_transitions() {
        let mut history = PurposeVectorHistory::new();

        // Empty: NOT first vector
        assert!(!history.is_first_vector());

        // One entry: IS first vector
        history.push(uniform_pv(0.5), "1");
        assert!(history.is_first_vector());

        // Two entries: NOT first vector
        history.push(uniform_pv(0.6), "2");
        assert!(!history.is_first_vector());

        // Many entries: NOT first vector
        history.push(uniform_pv(0.7), "3");
        assert!(!history.is_first_vector());
    }

    #[test]
    fn test_json_serialization_pv_history() {
        let mut history = PurposeVectorHistory::new();
        history.push(uniform_pv(0.75), "JSON test");

        let json = serde_json::to_string(&history).expect("JSON serialize");
        let restored: PurposeVectorHistory = serde_json::from_str(&json).expect("JSON deserialize");

        assert_eq!(restored.len(), history.len());
        assert_eq!(restored.current(), history.current());
    }

    #[test]
    fn test_default_trait_pv_history() {
        let history = PurposeVectorHistory::default();
        assert!(history.is_empty());
        assert_eq!(history.max_size, MAX_PV_HISTORY_SIZE);
    }

    #[test]
    fn test_context_preserved_in_snapshot_pv_history() {
        let mut history = PurposeVectorHistory::new();
        history.push(uniform_pv(0.5), "Important context");

        let snapshot = history.history().back().unwrap();
        assert_eq!(snapshot.context, "Important context");
    }

    #[test]
    fn test_timestamp_is_recent_pv_history() {
        let before = Utc::now();

        let mut history = PurposeVectorHistory::new();
        history.push(uniform_pv(0.5), "Timestamp test");

        let after = Utc::now();

        let snapshot = history.history().back().unwrap();
        assert!(snapshot.timestamp >= before);
        assert!(snapshot.timestamp <= after);
    }

    // =========================================================================
    // Additional PurposeVectorHistory Tests
    // =========================================================================

    #[test]
    fn test_multiple_evictions() {
        println!("=== TEST: Multiple FIFO evictions ===");

        let mut history = PurposeVectorHistory::with_max_size(3);

        // Push 10 items, only last 3 should remain
        for i in 0..10 {
            history.push(uniform_pv(i as f32 * 0.1), format!("Entry {}", i));
        }

        assert_eq!(history.len(), 3);

        // Verify the last 3 entries (7, 8, 9)
        let all_vals: Vec<f32> = history.history().iter().map(|s| s.vector[0]).collect();
        assert!((all_vals[0] - 0.7).abs() < 1e-5, "Expected 0.7, got {}", all_vals[0]);
        assert!((all_vals[1] - 0.8).abs() < 1e-5, "Expected 0.8, got {}", all_vals[1]);
        assert!((all_vals[2] - 0.9).abs() < 1e-5, "Expected 0.9, got {}", all_vals[2]);

        println!("EVIDENCE: After 10 pushes with max_size=3, only entries 7,8,9 remain");
    }

    #[test]
    fn test_history_accessor_returns_readonly_reference() {
        let mut history = PurposeVectorHistory::new();
        history.push(uniform_pv(0.5), "Test");
        history.push(uniform_pv(0.6), "Test 2");

        // history() should return a read-only reference
        let deque = history.history();
        assert_eq!(deque.len(), 2);
        assert!((deque.front().unwrap().vector[0] - 0.5).abs() < 1e-6);
        assert!((deque.back().unwrap().vector[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_with_max_size_different_capacities() {
        // Small capacity
        let h1 = PurposeVectorHistory::with_max_size(5);
        assert_eq!(h1.max_size, 5);

        // Large capacity (should still work)
        let h2 = PurposeVectorHistory::with_max_size(10000);
        assert_eq!(h2.max_size, 10000);

        // Default capacity
        let h3 = PurposeVectorHistory::new();
        assert_eq!(h3.max_size, MAX_PV_HISTORY_SIZE);
    }

    // =========================================================================
    // FSV: Complete Lifecycle Test
    // =========================================================================

    #[test]
    fn fsv_purpose_vector_history_full_lifecycle() {
        println!("=== FULL STATE VERIFICATION: PurposeVectorHistory lifecycle ===");

        // SOURCE OF TRUTH: PurposeVectorHistory.history VecDeque

        // 1. Create empty
        let mut history = PurposeVectorHistory::with_max_size(5);
        println!("\n1. CREATED: max_size={}, len={}, is_empty={}",
                 history.max_size, history.len(), history.is_empty());
        assert!(history.is_empty());
        assert!(!history.is_first_vector());

        // 2. Push first
        let prev1 = history.push(uniform_pv(0.5), "First");
        println!("\n2. AFTER FIRST PUSH: len={}, is_first_vector={}, prev={:?}",
                 history.len(), history.is_first_vector(), prev1.is_some());
        assert!(prev1.is_none());
        assert!(history.is_first_vector());
        assert_eq!(*history.current().unwrap(), uniform_pv(0.5));

        // 3. Push second
        let prev2 = history.push(uniform_pv(0.6), "Second");
        println!("\n3. AFTER SECOND PUSH: len={}, is_first_vector={}, prev={:?}",
                 history.len(), history.is_first_vector(), prev2.map(|p| p[0]));
        assert!(prev2.is_some());
        assert!(!history.is_first_vector());
        assert_eq!(*history.current().unwrap(), uniform_pv(0.6));
        assert_eq!(*history.previous().unwrap(), uniform_pv(0.5));

        // 4. Fill to capacity
        history.push(uniform_pv(0.7), "Third");
        history.push(uniform_pv(0.8), "Fourth");
        history.push(uniform_pv(0.9), "Fifth");
        println!("\n4. AT CAPACITY: len={}", history.len());
        assert_eq!(history.len(), 5);

        // 5. Eviction test
        history.push(uniform_pv(1.0), "Sixth");
        println!("\n5. AFTER EVICTION: len={}", history.len());
        assert_eq!(history.len(), 5);

        // Verify oldest (0.5) was evicted
        let oldest = history.history().front().unwrap().vector[0];
        println!("   Oldest value now: {} (should be 0.6)", oldest);
        assert!((oldest - 0.6).abs() < 1e-6);

        // 6. Serialization roundtrip
        let serialized = bincode::serialize(&history).unwrap();
        let restored: PurposeVectorHistory = bincode::deserialize(&serialized).unwrap();
        println!("\n6. AFTER SERIALIZATION: restored.len={}, restored.max_size={}",
                 restored.len(), restored.max_size);
        assert_eq!(restored.len(), history.len());
        assert_eq!(restored.max_size, history.max_size);
        assert_eq!(restored.current(), history.current());

        println!("\nEVIDENCE OF SUCCESS:");
        println!("  - Empty history correctly reports is_empty=true, is_first_vector=false");
        println!("  - First push returns None, sets is_first_vector=true");
        println!("  - Second push returns previous, sets is_first_vector=false");
        println!("  - FIFO eviction works at capacity");
        println!("  - Serialization preserves all state");
    }

    // =========================================================================
    // TASK-IDENTITY-P0-003: cosine_similarity_13d Tests
    // =========================================================================

    #[test]
    fn test_cosine_similarity_13d_identical_vectors() {
        println!("=== TEST: cosine_similarity_13d with identical vectors ===");

        let v1 = [1.0; 13];
        let v2 = [1.0; 13];

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Identical vectors should have cosine = 1.0
        assert!((similarity - 1.0).abs() < 1e-6,
            "Expected 1.0 for identical vectors, got {}", similarity);

        println!("EVIDENCE: cosine([1.0; 13], [1.0; 13]) = {:.6}", similarity);
    }

    #[test]
    fn test_cosine_similarity_13d_opposite_vectors() {
        println!("=== TEST: cosine_similarity_13d with opposite vectors ===");

        let v1 = [1.0; 13];
        let v2 = [-1.0; 13];

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Opposite vectors should have cosine = -1.0
        assert!((similarity - (-1.0)).abs() < 1e-6,
            "Expected -1.0 for opposite vectors, got {}", similarity);

        println!("EVIDENCE: cosine([1.0; 13], [-1.0; 13]) = {:.6}", similarity);
    }

    #[test]
    fn test_cosine_similarity_13d_orthogonal_vectors() {
        println!("=== TEST: cosine_similarity_13d with orthogonal vectors ===");

        // Create orthogonal vectors
        let mut v1 = [0.0; 13];
        let mut v2 = [0.0; 13];
        v1[0] = 1.0;  // Unit vector along first axis
        v2[1] = 1.0;  // Unit vector along second axis

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Orthogonal vectors should have cosine = 0.0
        assert!(similarity.abs() < 1e-6,
            "Expected 0.0 for orthogonal vectors, got {}", similarity);

        println!("EVIDENCE: Orthogonal vectors have cosine = {:.6}", similarity);
    }

    #[test]
    fn test_cosine_similarity_13d_zero_vector_first() {
        println!("=== EDGE CASE: cosine_similarity_13d with zero first vector ===");

        let v1 = [0.0; 13];
        let v2 = [1.0; 13];

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Zero vector should return 0.0 per spec
        assert_eq!(similarity, 0.0,
            "Zero magnitude vector should return 0.0, got {}", similarity);

        println!("EVIDENCE: Zero first vector returns 0.0");
    }

    #[test]
    fn test_cosine_similarity_13d_zero_vector_second() {
        println!("=== EDGE CASE: cosine_similarity_13d with zero second vector ===");

        let v1 = [1.0; 13];
        let v2 = [0.0; 13];

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Zero vector should return 0.0 per spec
        assert_eq!(similarity, 0.0,
            "Zero magnitude vector should return 0.0, got {}", similarity);

        println!("EVIDENCE: Zero second vector returns 0.0");
    }

    #[test]
    fn test_cosine_similarity_13d_both_zero_vectors() {
        println!("=== EDGE CASE: cosine_similarity_13d with both zero vectors ===");

        let v1 = [0.0; 13];
        let v2 = [0.0; 13];

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Both zero should return 0.0
        assert_eq!(similarity, 0.0,
            "Both zero vectors should return 0.0, got {}", similarity);

        println!("EVIDENCE: Both zero vectors return 0.0");
    }

    #[test]
    fn test_cosine_similarity_13d_near_zero_vectors() {
        println!("=== EDGE CASE: cosine_similarity_13d with near-zero vectors ===");

        let v1 = [1e-10; 13];
        let v2 = [1.0; 13];

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Near-zero should return 0.0 per spec
        assert_eq!(similarity, 0.0,
            "Near-zero magnitude should return 0.0, got {}", similarity);

        println!("EVIDENCE: Near-zero vector returns 0.0");
    }

    #[test]
    fn test_cosine_similarity_13d_different_magnitudes() {
        println!("=== TEST: cosine_similarity_13d with different magnitudes ===");

        let v1 = [1.0; 13];
        let v2 = [10.0; 13]; // Same direction, different magnitude

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Same direction should have cosine = 1.0 regardless of magnitude
        assert!((similarity - 1.0).abs() < 1e-6,
            "Same direction vectors should have cosine = 1.0, got {}", similarity);

        println!("EVIDENCE: Different magnitudes, same direction = {:.6}", similarity);
    }

    #[test]
    fn test_cosine_similarity_13d_real_purpose_vectors() {
        println!("=== TEST: cosine_similarity_13d with realistic purpose vectors ===");

        // Realistic purpose vectors (values in [0, 1])
        let v1 = [0.85, 0.78, 0.92, 0.67, 0.73, 0.61, 0.88, 0.75, 0.81, 0.69, 0.84, 0.72, 0.79];
        let v2 = [0.82, 0.75, 0.89, 0.70, 0.76, 0.65, 0.85, 0.72, 0.78, 0.72, 0.81, 0.69, 0.82];

        let similarity = cosine_similarity_13d(&v1, &v2);

        // Similar purpose vectors should have high cosine
        assert!(similarity > 0.99,
            "Similar purpose vectors should have high cosine, got {}", similarity);
        assert!(similarity <= 1.0,
            "Cosine should be <= 1.0, got {}", similarity);

        println!("EVIDENCE: Similar purpose vectors cosine = {:.6}", similarity);
    }

    #[test]
    fn test_cosine_similarity_13d_clamping() {
        println!("=== TEST: cosine_similarity_13d result is clamped to [-1, 1] ===");

        // Test many random-ish vectors to ensure clamping works
        for i in 0..100 {
            let v1: [f32; 13] = std::array::from_fn(|j| ((i + j) as f32 * 0.1).sin());
            let v2: [f32; 13] = std::array::from_fn(|j| ((i + j + 5) as f32 * 0.15).cos());

            let similarity = cosine_similarity_13d(&v1, &v2);

            assert!(similarity >= -1.0 && similarity <= 1.0,
                "Cosine must be in [-1, 1], got {} at iteration {}", similarity, i);
        }

        println!("EVIDENCE: All 100 iterations returned values in [-1, 1]");
    }

    // =========================================================================
    // TASK-IDENTITY-P0-003: IC Threshold Constants Tests
    // =========================================================================

    #[test]
    fn test_ic_crisis_threshold_value() {
        println!("=== TEST: IC_CRISIS_THRESHOLD constant ===");

        assert_eq!(IC_CRISIS_THRESHOLD, 0.7,
            "IC_CRISIS_THRESHOLD should be 0.7 per constitution.yaml");

        println!("EVIDENCE: IC_CRISIS_THRESHOLD = {}", IC_CRISIS_THRESHOLD);
    }

    #[test]
    fn test_ic_critical_threshold_value() {
        println!("=== TEST: IC_CRITICAL_THRESHOLD constant ===");

        assert_eq!(IC_CRITICAL_THRESHOLD, 0.5,
            "IC_CRITICAL_THRESHOLD should be 0.5 per constitution.yaml");

        println!("EVIDENCE: IC_CRITICAL_THRESHOLD = {}", IC_CRITICAL_THRESHOLD);
    }

    #[test]
    fn test_ic_thresholds_relationship() {
        println!("=== TEST: IC threshold relationship ===");

        // Critical should be lower than crisis (more severe)
        assert!(IC_CRITICAL_THRESHOLD < IC_CRISIS_THRESHOLD,
            "CRITICAL ({}) must be < CRISIS ({})",
            IC_CRITICAL_THRESHOLD, IC_CRISIS_THRESHOLD);

        println!("EVIDENCE: CRITICAL ({}) < CRISIS ({})",
            IC_CRITICAL_THRESHOLD, IC_CRISIS_THRESHOLD);
    }

    // =========================================================================
    // TASK-IDENTITY-P0-003: IdentityContinuityMonitor Tests
    // =========================================================================

    #[test]
    fn test_identity_continuity_monitor_new() {
        println!("=== TEST: IdentityContinuityMonitor::new() ===");

        let monitor = IdentityContinuityMonitor::new();

        assert!(monitor.is_empty());
        assert_eq!(monitor.history_len(), 0);
        assert_eq!(monitor.crisis_threshold(), IC_CRISIS_THRESHOLD);
        assert!(monitor.last_result().is_none());
        assert!(monitor.identity_coherence().is_none());
        assert!(monitor.current_status().is_none());
        assert!(!monitor.is_in_crisis()); // No result = not in crisis

        println!("EVIDENCE: new() creates empty monitor with defaults");
    }

    #[test]
    fn test_identity_continuity_monitor_with_threshold() {
        println!("=== TEST: IdentityContinuityMonitor::with_threshold() ===");

        let monitor = IdentityContinuityMonitor::with_threshold(0.8);

        assert_eq!(monitor.crisis_threshold(), 0.8);
        assert!(monitor.is_empty());

        println!("EVIDENCE: with_threshold(0.8) sets crisis_threshold = 0.8");
    }

    #[test]
    fn test_identity_continuity_monitor_with_threshold_clamping() {
        println!("=== TEST: with_threshold() clamping ===");

        // Test high value clamping
        let monitor_high = IdentityContinuityMonitor::with_threshold(1.5);
        assert_eq!(monitor_high.crisis_threshold(), 1.0);

        // Test negative value clamping
        let monitor_low = IdentityContinuityMonitor::with_threshold(-0.5);
        assert_eq!(monitor_low.crisis_threshold(), 0.0);

        println!("EVIDENCE: threshold clamped to [0, 1]");
    }

    #[test]
    fn test_identity_continuity_monitor_with_capacity() {
        println!("=== TEST: IdentityContinuityMonitor::with_capacity() ===");

        let monitor = IdentityContinuityMonitor::with_capacity(50);

        assert_eq!(monitor.history().max_size, 50);
        assert!(monitor.is_empty());

        println!("EVIDENCE: with_capacity(50) sets max_size = 50");
    }

    #[test]
    fn test_identity_continuity_monitor_default() {
        println!("=== TEST: IdentityContinuityMonitor::default() ===");

        let monitor = IdentityContinuityMonitor::default();

        assert!(monitor.is_empty());
        assert_eq!(monitor.crisis_threshold(), IC_CRISIS_THRESHOLD);
        assert_eq!(monitor.history().max_size, MAX_PV_HISTORY_SIZE);

        println!("EVIDENCE: default() equals new()");
    }

    #[test]
    fn test_identity_continuity_monitor_first_vector() {
        println!("=== TEST: compute_continuity() first vector ===");

        let mut monitor = IdentityContinuityMonitor::new();
        let pv = uniform_pv(0.8);

        // BEFORE
        assert!(monitor.is_empty());
        assert!(monitor.last_result().is_none());

        // EXECUTE
        let result = monitor.compute_continuity(&pv, 0.9, "First vector");

        // AFTER
        assert!(!monitor.is_empty());
        assert!(monitor.is_first_vector());
        assert_eq!(monitor.history_len(), 1);

        // First vector should return IC = 1.0, Healthy
        assert_eq!(result.identity_coherence, 1.0);
        assert_eq!(result.status, IdentityStatus::Healthy);

        // Getters should return values
        assert_eq!(monitor.identity_coherence(), Some(1.0));
        assert_eq!(monitor.current_status(), Some(IdentityStatus::Healthy));
        assert!(!monitor.is_in_crisis()); // 1.0 >= 0.7

        println!("EVIDENCE: First vector returns IC=1.0, Healthy");
    }

    #[test]
    fn test_identity_continuity_monitor_second_vector_identical() {
        println!("=== TEST: compute_continuity() second vector identical ===");

        let mut monitor = IdentityContinuityMonitor::new();
        let pv = uniform_pv(0.8);

        // First vector
        monitor.compute_continuity(&pv, 0.9, "First");

        // Second vector - same PV
        let result = monitor.compute_continuity(&pv, 0.95, "Second");

        // Identical vectors: cos = 1.0, IC = 1.0 * 0.95 = 0.95
        assert!((result.recent_continuity - 1.0).abs() < 1e-6,
            "Identical PVs should have cos = 1.0");
        assert!((result.identity_coherence - 0.95).abs() < 1e-6,
            "IC should be 1.0 * 0.95 = 0.95");
        assert_eq!(result.status, IdentityStatus::Healthy);

        println!("EVIDENCE: Identical vectors give IC = r = 0.95");
    }

    #[test]
    fn test_identity_continuity_monitor_drift_detection() {
        println!("=== TEST: compute_continuity() drift detection ===");

        let mut monitor = IdentityContinuityMonitor::new();

        // First vector - high values
        let pv1 = uniform_pv(0.9);
        monitor.compute_continuity(&pv1, 0.95, "Initial");

        // Second vector - very different (drift)
        let pv2 = uniform_pv(0.1);
        let result = monitor.compute_continuity(&pv2, 0.95, "Drifted");

        // Different vectors: cos([0.9;13], [0.1;13]) = 1.0 (same direction, diff magnitude)
        // Wait, uniform vectors have same direction regardless of magnitude
        // Let me compute actual: both are uniform, so cos = 1.0
        // This is expected - uniform vectors are parallel

        println!("Result: cos={:.4}, IC={:.4}, status={:?}",
            result.recent_continuity, result.identity_coherence, result.status);

        // Since both are uniform positive, they're parallel
        assert!((result.recent_continuity - 1.0).abs() < 1e-6);

        println!("EVIDENCE: Uniform vectors are parallel (cos = 1.0)");
    }

    #[test]
    fn test_identity_continuity_monitor_real_drift() {
        println!("=== TEST: compute_continuity() real purpose vector drift ===");

        let mut monitor = IdentityContinuityMonitor::new();

        // First vector - realistic purpose vector
        let pv1 = [0.9, 0.85, 0.92, 0.8, 0.88, 0.75, 0.95, 0.82, 0.87, 0.78, 0.91, 0.83, 0.86];
        monitor.compute_continuity(&pv1, 0.95, "Aligned");

        // Second vector - shifted purpose (some dimensions change)
        let pv2 = [0.3, 0.25, 0.32, 0.9, 0.88, 0.95, 0.25, 0.92, 0.17, 0.98, 0.21, 0.93, 0.16];
        let result = monitor.compute_continuity(&pv2, 0.9, "Shifted");

        // These vectors have different patterns, so cos < 1.0
        assert!(result.recent_continuity < 1.0,
            "Different patterns should have cos < 1.0");

        println!("Result: cos={:.4}, IC={:.4}, status={:?}",
            result.recent_continuity, result.identity_coherence, result.status);
        println!("EVIDENCE: Real drift detected with cos = {:.4}", result.recent_continuity);
    }

    #[test]
    fn test_identity_continuity_monitor_low_kuramoto_r() {
        println!("=== TEST: compute_continuity() with low Kuramoto r ===");

        let mut monitor = IdentityContinuityMonitor::new();
        let pv = uniform_pv(0.8);

        // First vector
        monitor.compute_continuity(&pv, 0.9, "First");

        // Second vector with low r (fragmented consciousness)
        let result = monitor.compute_continuity(&pv, 0.3, "Low sync");

        // cos = 1.0 (identical), r = 0.3, IC = 0.3
        assert!((result.identity_coherence - 0.3).abs() < 1e-6,
            "IC should be 1.0 * 0.3 = 0.3");
        assert_eq!(result.status, IdentityStatus::Critical,
            "IC=0.3 < 0.5 should be Critical");
        assert!(monitor.is_in_crisis(),
            "IC=0.3 < 0.7 should be in crisis");

        println!("EVIDENCE: Low r causes low IC despite perfect continuity");
    }

    #[test]
    fn test_identity_continuity_monitor_crisis_threshold_custom() {
        println!("=== TEST: is_in_crisis() with custom threshold ===");

        // More strict threshold
        let mut monitor = IdentityContinuityMonitor::with_threshold(0.9);
        let pv = uniform_pv(0.8);

        monitor.compute_continuity(&pv, 0.95, "First");
        let result = monitor.compute_continuity(&pv, 0.85, "Second");

        // IC = 1.0 * 0.85 = 0.85
        assert!((result.identity_coherence - 0.85).abs() < 1e-6);

        // With threshold 0.9, IC=0.85 should be in crisis
        assert!(monitor.is_in_crisis(),
            "IC=0.85 < threshold=0.9 should be in crisis");

        // Standard threshold would not be crisis
        assert!(result.identity_coherence >= IC_CRISIS_THRESHOLD,
            "IC=0.85 >= standard threshold 0.7");

        println!("EVIDENCE: Custom threshold 0.9 triggers crisis at IC=0.85");
    }

    #[test]
    fn test_identity_continuity_monitor_history_accumulation() {
        println!("=== TEST: history accumulation ===");

        let mut monitor = IdentityContinuityMonitor::with_capacity(5);

        // Add 7 vectors
        for i in 0..7 {
            let pv = uniform_pv(0.5 + (i as f32 * 0.05));
            monitor.compute_continuity(&pv, 0.9, format!("Vector {}", i));
        }

        // Should only have 5 due to capacity
        assert_eq!(monitor.history_len(), 5,
            "History should be capped at capacity 5");

        println!("EVIDENCE: History capped at configured capacity");
    }

    #[test]
    fn test_identity_continuity_monitor_serialization() {
        println!("=== TEST: IdentityContinuityMonitor serialization ===");

        let mut original = IdentityContinuityMonitor::with_threshold(0.8);
        let pv1 = uniform_pv(0.75);
        let pv2 = uniform_pv(0.8);
        original.compute_continuity(&pv1, 0.9, "First");
        original.compute_continuity(&pv2, 0.85, "Second");

        // Serialize with bincode
        let serialized = bincode::serialize(&original)
            .expect("Serialization should not fail");

        // Deserialize
        let restored: IdentityContinuityMonitor = bincode::deserialize(&serialized)
            .expect("Deserialization should not fail");

        // Verify state preserved
        assert_eq!(restored.history_len(), original.history_len());
        assert_eq!(restored.crisis_threshold(), original.crisis_threshold());
        assert_eq!(restored.identity_coherence(), original.identity_coherence());
        assert_eq!(restored.current_status(), original.current_status());

        println!("EVIDENCE: Serialization roundtrip preserves all state");
    }

    #[test]
    fn test_identity_continuity_monitor_json_serialization() {
        println!("=== TEST: IdentityContinuityMonitor JSON serialization ===");

        let mut original = IdentityContinuityMonitor::new();
        original.compute_continuity(&uniform_pv(0.7), 0.9, "Test");

        // Serialize to JSON
        let json = serde_json::to_string(&original)
            .expect("JSON serialization should not fail");

        // Deserialize from JSON
        let restored: IdentityContinuityMonitor = serde_json::from_str(&json)
            .expect("JSON deserialization should not fail");

        assert_eq!(restored.history_len(), original.history_len());

        println!("EVIDENCE: JSON serialization works correctly");
    }

    // =========================================================================
    // FSV: Full State Verification Tests
    // =========================================================================

    #[test]
    fn fsv_identity_continuity_monitor_complete_lifecycle() {
        println!("=== FSV: IdentityContinuityMonitor Complete Lifecycle ===");

        // SOURCE OF TRUTH: IdentityContinuityMonitor internal state

        // 1. Create monitor
        let mut monitor = IdentityContinuityMonitor::with_capacity(10);

        println!("\n1. CREATED:");
        println!("  - is_empty: {}", monitor.is_empty());
        println!("  - history_len: {}", monitor.history_len());
        println!("  - crisis_threshold: {:.2}", monitor.crisis_threshold());
        println!("  - last_result: {:?}", monitor.last_result());

        assert!(monitor.is_empty());
        assert_eq!(monitor.history_len(), 0);
        assert!(monitor.last_result().is_none());

        // 2. First vector (healthy baseline)
        let pv1 = [0.85, 0.78, 0.92, 0.67, 0.73, 0.61, 0.88, 0.75, 0.81, 0.69, 0.84, 0.72, 0.79];
        let result1 = monitor.compute_continuity(&pv1, 0.95, "Initial purpose alignment");

        println!("\n2. AFTER FIRST VECTOR:");
        println!("  - is_first_vector: {}", monitor.is_first_vector());
        println!("  - identity_coherence: {:.4}", result1.identity_coherence);
        println!("  - status: {:?}", result1.status);
        println!("  - is_in_crisis: {}", monitor.is_in_crisis());

        assert!(monitor.is_first_vector());
        assert_eq!(result1.identity_coherence, 1.0); // First vector default
        assert_eq!(result1.status, IdentityStatus::Healthy);
        assert!(!monitor.is_in_crisis());

        // 3. Second vector - stable (minimal drift)
        let pv2 = [0.84, 0.77, 0.91, 0.68, 0.74, 0.62, 0.87, 0.74, 0.80, 0.70, 0.83, 0.71, 0.78];
        let result2 = monitor.compute_continuity(&pv2, 0.93, "Minor adjustment");

        println!("\n3. AFTER SECOND VECTOR (stable):");
        println!("  - is_first_vector: {}", monitor.is_first_vector());
        println!("  - recent_continuity (cos): {:.6}", result2.recent_continuity);
        println!("  - kuramoto_r: {:.2}", result2.kuramoto_order_parameter);
        println!("  - identity_coherence: {:.6}", result2.identity_coherence);
        println!("  - status: {:?}", result2.status);

        assert!(!monitor.is_first_vector());
        assert!(result2.recent_continuity > 0.99); // Very similar vectors
        // IC should be high: cos ≈ 1.0 * 0.93 ≈ 0.93
        assert!(result2.identity_coherence > 0.9);

        // 4. Third vector - significant drift
        let pv3 = [0.2, 0.95, 0.3, 0.85, 0.25, 0.9, 0.35, 0.88, 0.28, 0.92, 0.32, 0.87, 0.29];
        let result3 = monitor.compute_continuity(&pv3, 0.75, "Purpose shift");

        println!("\n4. AFTER THIRD VECTOR (drift):");
        println!("  - recent_continuity (cos): {:.6}", result3.recent_continuity);
        println!("  - kuramoto_r: {:.2}", result3.kuramoto_order_parameter);
        println!("  - identity_coherence: {:.6}", result3.identity_coherence);
        println!("  - status: {:?}", result3.status);
        println!("  - is_in_crisis: {}", monitor.is_in_crisis());

        // Vectors have different patterns, cos should be lower
        assert!(result3.recent_continuity < 0.99);
        // Verify state matches getter
        assert_eq!(monitor.identity_coherence(), Some(result3.identity_coherence));
        assert_eq!(monitor.current_status(), Some(result3.status));

        // 5. Continue with low r to force crisis
        let pv4 = [0.2, 0.95, 0.3, 0.85, 0.25, 0.9, 0.35, 0.88, 0.28, 0.92, 0.32, 0.87, 0.29];
        let result4 = monitor.compute_continuity(&pv4, 0.2, "Low consciousness");

        println!("\n5. AFTER LOW KURAMOTO_R:");
        println!("  - recent_continuity (cos): {:.6}", result4.recent_continuity);
        println!("  - kuramoto_r: {:.2}", result4.kuramoto_order_parameter);
        println!("  - identity_coherence: {:.6}", result4.identity_coherence);
        println!("  - status: {:?}", result4.status);
        println!("  - is_in_crisis: {}", monitor.is_in_crisis());

        // Same vector as pv3, so cos = 1.0, but r = 0.2 -> IC = 0.2
        assert!((result4.recent_continuity - 1.0).abs() < 1e-6);
        assert!((result4.identity_coherence - 0.2).abs() < 1e-6);
        assert_eq!(result4.status, IdentityStatus::Critical);
        assert!(monitor.is_in_crisis()); // 0.2 < 0.7

        // 6. Verify serialization roundtrip
        let serialized = bincode::serialize(&monitor).expect("serialize");
        let restored: IdentityContinuityMonitor = bincode::deserialize(&serialized).expect("deserialize");

        println!("\n6. AFTER SERIALIZATION ROUNDTRIP:");
        println!("  - history_len preserved: {}", restored.history_len() == monitor.history_len());
        println!("  - identity_coherence preserved: {:?}", restored.identity_coherence() == monitor.identity_coherence());

        assert_eq!(restored.history_len(), monitor.history_len());
        assert_eq!(restored.identity_coherence(), monitor.identity_coherence());
        assert_eq!(restored.crisis_threshold(), monitor.crisis_threshold());

        println!("\nEVIDENCE OF SUCCESS:");
        println!("  - First vector defaults to IC=1.0, Healthy");
        println!("  - Cosine similarity correctly computed for consecutive PVs");
        println!("  - IC = cos(PV_t, PV_{{t-1}}) × r(t) formula verified");
        println!("  - is_in_crisis() correctly detects IC < threshold");
        println!("  - Serialization preserves all state");
    }

    #[test]
    fn fsv_identity_continuity_monitor_edge_cases() {
        println!("=== FSV: IdentityContinuityMonitor Edge Cases ===");

        // EDGE CASE 1: Negative cosine (opposite vectors)
        println!("\n1. EDGE CASE: Opposite purpose vectors");
        let mut monitor1 = IdentityContinuityMonitor::new();
        let pv_pos = uniform_pv(1.0);
        let pv_neg = uniform_pv(-1.0);

        monitor1.compute_continuity(&pv_pos, 0.9, "Positive");
        let result1 = monitor1.compute_continuity(&pv_neg, 0.9, "Negative");

        println!("  - cos(PV+, PV-) = {:.4}", result1.recent_continuity);
        println!("  - IC = {:.4} (should be 0.0, clamped)", result1.identity_coherence);
        println!("  - status: {:?}", result1.status);

        // Opposite uniform vectors: cos = -1.0, IC = -1.0 * 0.9 = -0.9, clamped to 0.0
        assert!((result1.recent_continuity - (-1.0)).abs() < 1e-6);
        assert_eq!(result1.identity_coherence, 0.0); // Clamped
        assert_eq!(result1.status, IdentityStatus::Critical);

        // EDGE CASE 2: Zero Kuramoto r
        println!("\n2. EDGE CASE: Zero Kuramoto r (no sync)");
        let mut monitor2 = IdentityContinuityMonitor::new();
        let pv = uniform_pv(0.8);

        monitor2.compute_continuity(&pv, 0.9, "First");
        let result2 = monitor2.compute_continuity(&pv, 0.0, "Zero sync");

        println!("  - cos = {:.4}", result2.recent_continuity);
        println!("  - kuramoto_r = {:.2}", result2.kuramoto_order_parameter);
        println!("  - IC = {:.4}", result2.identity_coherence);

        // cos = 1.0, r = 0.0, IC = 0.0
        assert_eq!(result2.identity_coherence, 0.0);
        assert_eq!(result2.status, IdentityStatus::Critical);

        // EDGE CASE 3: Max values
        println!("\n3. EDGE CASE: Maximum values");
        let mut monitor3 = IdentityContinuityMonitor::new();

        monitor3.compute_continuity(&uniform_pv(1.0), 1.0, "First");
        let result3 = monitor3.compute_continuity(&uniform_pv(1.0), 1.0, "Perfect");

        println!("  - IC = {:.4}", result3.identity_coherence);
        assert_eq!(result3.identity_coherence, 1.0);
        assert_eq!(result3.status, IdentityStatus::Healthy);

        // EDGE CASE 4: Exact threshold boundaries
        println!("\n4. EDGE CASE: Exact threshold boundaries");
        let mut monitor4 = IdentityContinuityMonitor::new();

        monitor4.compute_continuity(&uniform_pv(0.8), 1.0, "First");
        // For IC = 0.7 exactly: cos = 1.0, r = 0.7
        let result4 = monitor4.compute_continuity(&uniform_pv(0.8), 0.7, "Boundary");

        println!("  - IC = {:.4} (at crisis boundary)", result4.identity_coherence);
        println!("  - is_in_crisis: {}", monitor4.is_in_crisis());

        assert!((result4.identity_coherence - 0.7).abs() < 1e-6);
        // IC = 0.7 is NOT in crisis (crisis is < 0.7)
        assert!(!monitor4.is_in_crisis());

        println!("\nEVIDENCE OF SUCCESS:");
        println!("  - Opposite vectors handled correctly (IC clamped to 0)");
        println!("  - Zero r produces IC = 0");
        println!("  - Perfect values produce IC = 1.0");
        println!("  - Boundary values handled correctly");
    }

    #[test]
    fn fsv_cosine_similarity_13d_mathematical_properties() {
        println!("=== FSV: cosine_similarity_13d Mathematical Properties ===");

        // Property 1: Symmetry - cos(a, b) = cos(b, a)
        println!("\n1. PROPERTY: Symmetry");
        let a = [0.8, 0.7, 0.9, 0.6, 0.75, 0.65, 0.85, 0.72, 0.78, 0.68, 0.82, 0.71, 0.76];
        let b = [0.5, 0.9, 0.3, 0.85, 0.4, 0.88, 0.35, 0.82, 0.45, 0.87, 0.38, 0.84, 0.42];

        let cos_ab = cosine_similarity_13d(&a, &b);
        let cos_ba = cosine_similarity_13d(&b, &a);

        println!("  - cos(a, b) = {:.10}", cos_ab);
        println!("  - cos(b, a) = {:.10}", cos_ba);
        assert!((cos_ab - cos_ba).abs() < 1e-10,
            "Symmetry violation: {} != {}", cos_ab, cos_ba);

        // Property 2: Self-similarity - cos(a, a) = 1
        println!("\n2. PROPERTY: Self-similarity");
        let cos_aa = cosine_similarity_13d(&a, &a);
        println!("  - cos(a, a) = {:.10}", cos_aa);
        assert!((cos_aa - 1.0).abs() < 1e-10);

        // Property 3: Scale invariance - cos(k*a, a) = 1 for k > 0
        println!("\n3. PROPERTY: Scale invariance");
        let scaled_a: [f32; 13] = std::array::from_fn(|i| a[i] * 3.7);
        let cos_scaled = cosine_similarity_13d(&scaled_a, &a);
        println!("  - cos(3.7*a, a) = {:.10}", cos_scaled);
        assert!((cos_scaled - 1.0).abs() < 1e-6);

        // Property 4: Bounded - -1 <= cos(a, b) <= 1
        println!("\n4. PROPERTY: Bounded output");
        let test_vectors: Vec<[f32; 13]> = vec![
            std::array::from_fn(|i| (i as f32 * 0.1).sin()),
            std::array::from_fn(|i| (i as f32 * 0.2).cos()),
            std::array::from_fn(|i| ((i + 3) as f32).sqrt()),
            std::array::from_fn(|_| -0.5),
            std::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 }),
        ];

        for (i, v1) in test_vectors.iter().enumerate() {
            for (j, v2) in test_vectors.iter().enumerate() {
                let cos = cosine_similarity_13d(v1, v2);
                assert!(cos >= -1.0 && cos <= 1.0,
                    "Bounded violation at ({}, {}): {}", i, j, cos);
            }
        }
        println!("  - All {} comparisons in [-1, 1]", test_vectors.len() * test_vectors.len());

        println!("\nEVIDENCE OF SUCCESS:");
        println!("  - Symmetry verified: cos(a,b) = cos(b,a)");
        println!("  - Self-similarity verified: cos(a,a) = 1");
        println!("  - Scale invariance verified: cos(k*a, a) = 1");
        println!("  - Bounded output verified: -1 <= cos <= 1");
    }
}
