//! IdentityContinuityMonitor - Continuous IC tracking wrapper
//!
//! Wraps PurposeVectorHistory to provide real-time identity continuity
//! monitoring and status classification.
//!
//! # TASK-IDENTITY-P0-004: Crisis Detection
//!
//! This module implements crisis detection with:
//! - `CrisisDetectionResult`: Captures all transition information
//! - Status transition tracking: Tracks previous vs current status
//! - Cooldown mechanism: Prevents event spam during IC fluctuations
//! - Helper methods: For downstream consumers (P0-005, P0-006)

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::cosine::cosine_similarity_13d;
use super::identity_continuity::IdentityContinuity;
use super::purpose_vector_history::{PurposeVectorHistory, PurposeVectorHistoryProvider};
use super::types::{IdentityStatus, CRISIS_EVENT_COOLDOWN, IC_CRISIS_THRESHOLD};

/// Result of crisis detection analysis
///
/// Contains all information needed by CrisisProtocol (P0-005)
/// to decide what actions to take.
///
/// # TASK-IDENTITY-P0-004: Crisis Detection
///
/// # Fields
/// - `identity_coherence`: Current IC value (0.0-1.0)
/// - `previous_status`: Status before this detection
/// - `current_status`: Status after this detection
/// - `status_changed`: True if status transitioned
/// - `entering_crisis`: True if transitioned FROM Healthy to any lower state
/// - `entering_critical`: True if transitioned TO Critical from any other state
/// - `recovering`: True if status improved (lower ordinal -> higher ordinal)
/// - `time_since_last_event`: Time since last crisis event was emitted
/// - `can_emit_event`: True if cooldown allows new event emission
#[derive(Debug, Clone, PartialEq)]
pub struct CrisisDetectionResult {
    /// Current IC value
    pub identity_coherence: f32,
    /// Previous status (before this computation)
    pub previous_status: IdentityStatus,
    /// Current status (after this computation)
    pub current_status: IdentityStatus,
    /// Whether status changed
    pub status_changed: bool,
    /// Whether entering crisis (transition from Healthy to Warning/Degraded/Critical)
    pub entering_crisis: bool,
    /// Whether entering critical (transition to Critical specifically)
    pub entering_critical: bool,
    /// Whether recovering (transition from lower to higher status)
    pub recovering: bool,
    /// Time since last crisis event emission
    pub time_since_last_event: Option<Duration>,
    /// Whether cooldown allows event emission
    pub can_emit_event: bool,
}

/// Convert status to ordinal for comparison
/// Higher ordinal = healthier state
/// Critical=0, Degraded=1, Warning=2, Healthy=3
#[inline]
fn status_ordinal(status: IdentityStatus) -> u8 {
    match status {
        IdentityStatus::Critical => 0,
        IdentityStatus::Degraded => 1,
        IdentityStatus::Warning => 2,
        IdentityStatus::Healthy => 3,
    }
}

/// Default status for deserialization
fn default_healthy_status() -> IdentityStatus {
    IdentityStatus::Healthy
}

/// Identity Continuity Monitor - Continuous IC tracking wrapper
///
/// Wraps `PurposeVectorHistory` to provide real-time identity continuity
/// monitoring and status classification.
///
/// # Constitution Reference
/// From constitution.yaml lines 365-392:
/// - IC = cos(PV_t, PV_{t-1}) x r(t)
/// - Thresholds: healthy>0.9, warning<0.7, dream<0.5
/// - self_ego_node.identity_trajectory: max 1000 snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContinuityMonitor {
    /// Purpose vector history buffer (delegates to PurposeVectorHistory)
    history: PurposeVectorHistory,
    /// Cached last computation result (None until first compute_continuity call)
    last_result: Option<IdentityContinuity>,
    /// Configurable crisis threshold (default: IC_CRISIS_THRESHOLD = 0.7)
    crisis_threshold: f32,

    // === TASK-IDENTITY-P0-004: Crisis Detection Fields ===

    /// Previous status for transition detection (default: Healthy)
    #[serde(default = "default_healthy_status")]
    previous_status: IdentityStatus,
    /// Last time a crisis event was emitted (not serialized - transient state)
    #[serde(skip)]
    last_event_time: Option<Instant>,

    // === TASK-IDENTITY-P0-007: MCP Tool Exposure Fields ===

    /// Cached last crisis detection result for MCP tool exposure.
    /// Not serialized - transient state reconstructed on detect_crisis() calls.
    #[serde(skip)]
    last_detection: Option<CrisisDetectionResult>,
}

impl IdentityContinuityMonitor {
    /// Create new monitor with default configuration.
    ///
    /// Defaults:
    /// - history capacity: MAX_PV_HISTORY_SIZE (1000)
    /// - crisis_threshold: IC_CRISIS_THRESHOLD (0.7)
    /// - previous_status: Healthy (TASK-P0-004)
    /// - last_event_time: None (TASK-P0-004)
    pub fn new() -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: IC_CRISIS_THRESHOLD,
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None, // TASK-IDENTITY-P0-007
        }
    }

    /// Create monitor with custom crisis threshold.
    ///
    /// # Arguments
    /// * `threshold` - Custom crisis threshold (clamped to [0, 1])
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            history: PurposeVectorHistory::new(),
            last_result: None,
            crisis_threshold: threshold.clamp(0.0, 1.0),
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None, // TASK-IDENTITY-P0-007
        }
    }

    /// Create monitor with custom history capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum history entries (0 = unlimited)
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            history: PurposeVectorHistory::with_max_size(capacity),
            last_result: None,
            crisis_threshold: IC_CRISIS_THRESHOLD,
            previous_status: IdentityStatus::Healthy,
            last_event_time: None,
            last_detection: None, // TASK-IDENTITY-P0-007
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

                // Create IdentityContinuity with IC = cos x r
                IdentityContinuity::new(cosine, kuramoto_r)
            }
        };

        // Cache result for subsequent getters
        self.last_result = Some(result.clone());

        result
    }

    /// Get the last computed IdentityContinuity result.
    #[inline]
    pub fn last_result(&self) -> Option<&IdentityContinuity> {
        self.last_result.as_ref()
    }

    /// Get current identity coherence value (IC).
    #[inline]
    pub fn identity_coherence(&self) -> Option<f32> {
        self.last_result.as_ref().map(|r| r.identity_coherence)
    }

    /// Get current identity status classification.
    #[inline]
    pub fn current_status(&self) -> Option<IdentityStatus> {
        self.last_result.as_ref().map(|r| r.status)
    }

    /// Check if identity is in crisis (IC < crisis_threshold).
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
    pub fn history(&self) -> &PurposeVectorHistory {
        &self.history
    }

    /// Check if history is empty (no vectors recorded).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Check if this is the first vector (exactly one entry).
    #[inline]
    pub fn is_first_vector(&self) -> bool {
        self.history.is_first_vector()
    }

    // === TASK-IDENTITY-P0-004: Crisis Detection Methods ===

    /// Detect crisis state and track transitions.
    ///
    /// # Algorithm
    /// 1. Get current status from last_result
    /// 2. Compare with previous_status to detect transitions
    /// 3. Compute entering_crisis (from Healthy to lower)
    /// 4. Compute entering_critical (to Critical from any other)
    /// 5. Compute recovering (status improvement)
    /// 6. Check cooldown for event emission
    /// 7. Update previous_status for next call
    ///
    /// # Returns
    /// `CrisisDetectionResult` with all transition information
    ///
    /// # Panics
    /// Never panics. Returns default result if no computation has occurred.
    pub fn detect_crisis(&mut self) -> CrisisDetectionResult {
        // Get current values (default to Healthy if no computation yet)
        let current_status = self.current_status().unwrap_or(IdentityStatus::Healthy);
        let ic = self.identity_coherence().unwrap_or(1.0);
        let prev_status = self.previous_status;

        // Detect transitions
        let status_changed = current_status != prev_status;

        // Entering crisis = transitioning FROM Healthy to any lower state
        let entering_crisis = status_changed
            && prev_status == IdentityStatus::Healthy
            && current_status != IdentityStatus::Healthy;

        // Entering critical = transitioning TO Critical from any other state
        let entering_critical = status_changed
            && current_status == IdentityStatus::Critical
            && prev_status != IdentityStatus::Critical;

        // Recovering = improving status (lower ordinal to higher ordinal)
        let recovering =
            status_changed && status_ordinal(current_status) > status_ordinal(prev_status);

        // Cooldown check
        let time_since_last_event = self.last_event_time.map(|t| t.elapsed());
        let can_emit_event = match time_since_last_event {
            None => true, // No previous event, can emit
            Some(elapsed) => elapsed >= CRISIS_EVENT_COOLDOWN,
        };

        // Update previous_status for next detection
        self.previous_status = current_status;

        let result = CrisisDetectionResult {
            identity_coherence: ic,
            previous_status: prev_status,
            current_status,
            status_changed,
            entering_crisis,
            entering_critical,
            recovering,
            time_since_last_event,
            can_emit_event,
        };

        // TASK-IDENTITY-P0-007: Cache result for MCP tool exposure
        self.last_detection = Some(result.clone());

        result
    }

    /// Get the previous status (before last detection).
    #[inline]
    pub fn previous_status(&self) -> IdentityStatus {
        self.previous_status
    }

    /// Check if status changed compared to previous detection.
    ///
    /// Note: This compares current computed status with previously recorded status,
    /// NOT the result of the last detect_crisis call.
    #[inline]
    pub fn status_changed(&self) -> bool {
        self.current_status()
            .map(|curr| curr != self.previous_status)
            .unwrap_or(false)
    }

    /// Check if currently entering critical state.
    ///
    /// Returns true if current status is Critical and previous status was not Critical.
    #[inline]
    pub fn entering_critical(&self) -> bool {
        self.current_status()
            .map(|curr| {
                curr == IdentityStatus::Critical
                    && self.previous_status != IdentityStatus::Critical
            })
            .unwrap_or(false)
    }

    /// Mark that a crisis event was emitted (resets cooldown timer).
    #[inline]
    pub fn mark_event_emitted(&mut self) {
        self.last_event_time = Some(Instant::now());
    }

    /// Get time since last event emission.
    #[inline]
    pub fn time_since_last_event(&self) -> Option<Duration> {
        self.last_event_time.map(|t| t.elapsed())
    }

    // === TASK-IDENTITY-P0-007: MCP Tool Exposure Methods ===

    /// Get the last crisis detection result.
    ///
    /// Returns `None` if `detect_crisis()` has never been called.
    /// The cached result allows MCP tools to access crisis state
    /// without triggering a new detection cycle.
    ///
    /// # TASK-IDENTITY-P0-007
    #[inline]
    pub fn last_detection(&self) -> Option<CrisisDetectionResult> {
        self.last_detection.clone()
    }
}

impl Default for IdentityContinuityMonitor {
    fn default() -> Self {
        Self::new()
    }
}
