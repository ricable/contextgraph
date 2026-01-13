//! Crisis Protocol - Identity crisis response execution
//!
//! Implements the crisis response protocol per constitution.yaml lines 387-392:
//! - IC < 0.7: Record purpose snapshot
//! - IC < 0.5: Generate IdentityCrisisEvent for broadcast
//!
//! # TASK-IDENTITY-P0-005: Crisis Protocol

use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::monitor::{CrisisDetectionResult, IdentityContinuityMonitor};
use super::self_ego_node::SelfEgoNode;
use super::types::{IdentityStatus, CRISIS_EVENT_COOLDOWN};
use crate::error::CoreResult;
use crate::gwt::workspace::WorkspaceEvent;

/// Actions taken during crisis protocol execution
///
/// Each action is recorded in `CrisisProtocolResult.actions` for
/// auditability and MCP tool reporting.
#[derive(Debug, Clone, PartialEq)]
pub enum CrisisAction {
    /// Purpose snapshot was recorded in SelfEgoNode
    SnapshotRecorded {
        /// Context message saved with snapshot
        context: String,
    },
    /// IdentityCrisisEvent was generated
    EventGenerated {
        /// Event type identifier
        event_type: String,
    },
    /// Event was not emitted due to cooldown
    EventSkippedCooldown {
        /// Time remaining until cooldown expires
        remaining: Duration,
    },
    /// Introspection mode was triggered (event emitted successfully)
    IntrospectionTriggered,
}

/// Event payload for identity crisis
///
/// Contains all information needed for workspace broadcast and
/// Dream system consumption.
#[derive(Debug, Clone)]
pub struct IdentityCrisisEvent {
    /// Current identity coherence (IC) value
    pub identity_coherence: f32,
    /// Status before crisis detection
    pub previous_status: IdentityStatus,
    /// Current status (should be Critical)
    pub current_status: IdentityStatus,
    /// Human-readable reason for crisis
    pub reason: String,
    /// Timestamp of event generation
    pub timestamp: DateTime<Utc>,
}

impl IdentityCrisisEvent {
    /// Create event from detection result
    ///
    /// # Arguments
    /// * `detection` - Result from `detect_crisis()`
    /// * `reason` - Human-readable reason for crisis
    ///
    /// # Returns
    /// `IdentityCrisisEvent` with all fields populated
    pub fn from_detection(detection: &CrisisDetectionResult, reason: impl Into<String>) -> Self {
        Self {
            identity_coherence: detection.identity_coherence,
            previous_status: detection.previous_status,
            current_status: detection.current_status,
            reason: reason.into(),
            timestamp: Utc::now(),
        }
    }

    /// Convert to WorkspaceEvent for broadcasting
    ///
    /// # Returns
    /// `WorkspaceEvent::IdentityCritical` variant
    pub fn to_workspace_event(&self) -> WorkspaceEvent {
        WorkspaceEvent::IdentityCritical {
            identity_coherence: self.identity_coherence,
            previous_status: format!("{:?}", self.previous_status),
            current_status: format!("{:?}", self.current_status),
            reason: self.reason.clone(),
            timestamp: self.timestamp,
        }
    }
}

/// Result of crisis protocol execution
///
/// Contains all information about what actions were taken during
/// crisis protocol execution, for MCP reporting and debugging.
#[derive(Debug, Clone)]
pub struct CrisisProtocolResult {
    /// The detection result that triggered this protocol
    pub detection: CrisisDetectionResult,
    /// Whether a purpose snapshot was recorded
    pub snapshot_recorded: bool,
    /// Context message for recorded snapshot (if any)
    pub snapshot_context: Option<String>,
    /// Generated event (if any)
    pub event: Option<IdentityCrisisEvent>,
    /// Whether event was actually emittable (cooldown check passed)
    pub event_emitted: bool,
    /// All actions taken during execution
    pub actions: Vec<CrisisAction>,
}

/// Executes crisis protocol based on detection results
///
/// Holds reference to `SelfEgoNode` for snapshot recording.
/// Does NOT hold broadcaster reference - P0-006 handles broadcasting.
pub struct CrisisProtocol {
    /// Reference to ego node for snapshot recording
    ego_node: Arc<RwLock<SelfEgoNode>>,
}

impl CrisisProtocol {
    /// Create new protocol with ego node reference
    ///
    /// # Arguments
    /// * `ego_node` - Arc-wrapped SelfEgoNode for snapshot recording
    pub fn new(ego_node: Arc<RwLock<SelfEgoNode>>) -> Self {
        Self { ego_node }
    }

    /// Execute crisis protocol based on detection result
    ///
    /// # Algorithm
    /// 1. If status is NOT Healthy (IC < 0.7):
    ///    a. Record purpose snapshot with context in SelfEgoNode
    /// 2. If entering_critical OR current_status == Critical:
    ///    a. Generate IdentityCrisisEvent
    ///    b. Check cooldown via detection.can_emit_event
    ///    c. If can emit: mark event_emitted = true, add IntrospectionTriggered
    ///    d. If cannot emit: add EventSkippedCooldown with remaining time
    /// 3. Return CrisisProtocolResult with all actions
    ///
    /// # Arguments
    /// * `detection` - Result from `detect_crisis()`
    /// * `monitor` - Mutable reference to monitor for `mark_event_emitted()`
    ///
    /// # Returns
    /// `CoreResult<CrisisProtocolResult>` with all actions taken
    ///
    /// # Errors
    /// Returns `CoreError` if `record_purpose_snapshot` fails
    pub async fn execute(
        &self,
        detection: CrisisDetectionResult,
        monitor: &mut IdentityContinuityMonitor,
    ) -> CoreResult<CrisisProtocolResult> {
        let mut actions = Vec::new();
        let mut snapshot_recorded = false;
        let mut snapshot_context: Option<String> = None;
        let mut event: Option<IdentityCrisisEvent> = None;
        let mut event_emitted = false;

        // Step 1: Handle any crisis state (IC < 0.7 = not Healthy)
        if detection.current_status != IdentityStatus::Healthy {
            let context = format!(
                "Identity crisis: IC={:.4}, status={:?} (was {:?})",
                detection.identity_coherence, detection.current_status, detection.previous_status
            );

            // Record snapshot in ego node
            {
                let mut ego = self.ego_node.write().await;
                ego.record_purpose_snapshot(&context)?;
            }

            snapshot_recorded = true;
            actions.push(CrisisAction::SnapshotRecorded {
                context: context.clone(),
            });
            snapshot_context = Some(context);

            tracing::debug!(
                ic = %detection.identity_coherence,
                status = ?detection.current_status,
                "Crisis protocol: snapshot recorded"
            );
        }

        // Step 2: Handle critical state (IC < 0.5)
        if detection.entering_critical || detection.current_status == IdentityStatus::Critical {
            let reason = if detection.entering_critical {
                format!(
                    "Identity entered critical state: IC dropped from {:?} to Critical (IC={:.4})",
                    detection.previous_status, detection.identity_coherence
                )
            } else {
                format!(
                    "Identity remains critical: IC={:.4}",
                    detection.identity_coherence
                )
            };

            let crisis_event = IdentityCrisisEvent::from_detection(&detection, &reason);
            event = Some(crisis_event.clone());

            actions.push(CrisisAction::EventGenerated {
                event_type: "IdentityCritical".to_string(),
            });

            // Check cooldown
            if detection.can_emit_event {
                event_emitted = true;
                monitor.mark_event_emitted();
                actions.push(CrisisAction::IntrospectionTriggered);

                tracing::warn!(
                    ic = %detection.identity_coherence,
                    "Crisis protocol: IdentityCritical event ready for emission"
                );
            } else {
                // Calculate remaining cooldown time
                let remaining = detection
                    .time_since_last_event
                    .map(|elapsed| CRISIS_EVENT_COOLDOWN.saturating_sub(elapsed))
                    .unwrap_or(Duration::ZERO);

                actions.push(CrisisAction::EventSkippedCooldown { remaining });

                tracing::debug!(
                    remaining_secs = %remaining.as_secs(),
                    "Crisis protocol: event skipped due to cooldown"
                );
            }
        }

        Ok(CrisisProtocolResult {
            detection,
            snapshot_recorded,
            snapshot_context,
            event,
            event_emitted,
            actions,
        })
    }
}
