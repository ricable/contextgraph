//! GwtSystem - Self-awareness loop methods
//!
//! This module contains impl blocks for GwtSystem that handle
//! self-awareness processing and identity crisis management.

use crate::types::fingerprint::TeleologicalFingerprint;

use super::{GwtSystem, IdentityStatus, SelfReflectionResult, WorkspaceEvent};

impl GwtSystem {
    /// Process an action through the self-awareness loop.
    ///
    /// This method:
    /// 1. Updates self_ego_node.purpose_vector from fingerprint
    /// 2. Computes action_embedding from fingerprint.purpose_vector.alignments
    /// 3. Gets kuramoto_r from internal Kuramoto network
    /// 4. Calls self_awareness_loop.cycle()
    /// 5. Triggers dream if IdentityStatus::Critical
    ///
    /// # Arguments
    /// * `fingerprint` - The action's TeleologicalFingerprint
    ///
    /// # Returns
    /// * `SelfReflectionResult` containing alignment and identity status
    ///
    /// # Constitution Reference
    /// From constitution.yaml lines 365-392:
    /// - loop: "Retrieve→A(action,PV)→if<0.55 self_reflect→update fingerprint→store evolution"
    /// - identity_continuity: "IC = cos(PV_t, PV_{t-1}) × r(t); healthy>0.9, warning<0.7, dream<0.5"
    pub async fn process_action_awareness(
        &self,
        fingerprint: &TeleologicalFingerprint,
    ) -> crate::CoreResult<SelfReflectionResult> {
        // 1. Get kuramoto_r from internal network
        let kuramoto_r = self.get_kuramoto_r().await;

        // 2. Extract action_embedding from fingerprint
        let action_embedding = fingerprint.purpose_vector.alignments;

        // 3. Acquire write lock on self_ego_node
        let mut ego_node = self.self_ego_node.write().await;

        // 4. Update purpose_vector from fingerprint
        ego_node.update_from_fingerprint(fingerprint)?;

        // 5. Acquire write lock on self_awareness_loop
        let mut loop_mgr = self.self_awareness_loop.write().await;

        // 6. Execute self-awareness cycle
        let result = loop_mgr.cycle(&mut ego_node, &action_embedding, kuramoto_r).await?;

        // 7. Log the result
        tracing::info!(
            "Self-awareness cycle: alignment={:.4}, identity_status={:?}, identity_coherence={:.4}",
            result.alignment,
            result.identity_status,
            result.identity_coherence
        );

        // 8. Check for Critical identity status - MUST trigger dream
        if result.identity_status == IdentityStatus::Critical {
            // Drop locks before async call to prevent deadlock
            drop(ego_node);
            drop(loop_mgr);
            self.trigger_identity_dream("Identity coherence critical").await?;
        }

        // 9. Return result
        Ok(result)
    }

    /// Trigger dream consolidation when identity is Critical (IC < 0.5).
    ///
    /// If dream controller is not available, logs warning and records
    /// purpose snapshot (graceful degradation).
    ///
    /// # Arguments
    /// * `reason` - Description of why dream is triggered
    ///
    /// # Constitution Reference
    /// From constitution.yaml line 391: "dream<0.5" triggers introspective dream
    pub(crate) async fn trigger_identity_dream(&self, reason: &str) -> crate::CoreResult<()> {
        // 1. Log critical warning
        tracing::warn!("IDENTITY CRITICAL: Triggering dream consolidation. Reason: {}", reason);

        // 2. Record purpose snapshot with dream trigger context
        {
            let mut ego_node = self.self_ego_node.write().await;
            ego_node.record_purpose_snapshot(format!("Dream triggered: {}", reason))?;
        }

        // 3. Get identity coherence from self_awareness_loop
        let identity_coherence = {
            let loop_mgr = self.self_awareness_loop.read().await;
            loop_mgr.identity_coherence()
        };

        // 4. Broadcast workspace event for dream trigger
        // (DreamController will be wired in TASK-GWT-P1-002)
        self.event_broadcaster.broadcast(WorkspaceEvent::IdentityCritical {
            identity_coherence,
            previous_status: "Unknown".to_string(), // System awareness doesn't track status transitions
            current_status: "Critical".to_string(),
            reason: reason.to_string(),
            timestamp: chrono::Utc::now(),
        }).await;

        // 5. Log graceful degradation message
        // TODO(TASK-GWT-P1-002): Wire to actual DreamController
        tracing::info!("Dream trigger recorded. DreamController integration pending.");

        Ok(())
    }
}
