//! Status query tool implementations (get_memetic_status).

use serde_json::json;
use tracing::error;

use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

impl Handlers {
    /// get_memetic_status tool implementation.
    ///
    /// TASK-S001: Updated to use TeleologicalMemoryStore count.
    ///
    /// Returns comprehensive system status including:
    /// - Fingerprint count from TeleologicalMemoryStore
    /// - Live UTL metrics from UtlProcessor (NOT hardcoded)
    /// - `_cognitive_pulse` with live system state
    ///
    /// # Constitution References
    /// - UTL formula: constitution.yaml:152
    pub(crate) async fn call_get_memetic_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        let fingerprint_count = match self.teleological_store.count().await {
            Ok(count) => count,
            Err(e) => {
                error!(error = %e, "get_memetic_status: TeleologicalStore.count() FAILED");
                return JsonRpcResponse::error(
                    id,
                    error_codes::STORAGE_ERROR,
                    format!("Failed to get fingerprint count: {}", e),
                );
            }
        };

        // Get LIVE UTL status from the processor
        let utl_status = self.utl_processor.get_status();

        // FAIL-FAST: UTL processor MUST return all required fields.
        // Per constitution AP-007: No stubs or fallbacks in production code paths.
        // If the UTL processor doesn't have these fields, the system is broken.
        let lifecycle_phase = match utl_status.get("lifecycle_phase").and_then(|v| v.as_str()) {
            Some(phase) => phase,
            None => {
                error!("get_memetic_status: UTL processor missing 'lifecycle_phase' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'lifecycle_phase'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let entropy = match utl_status.get("entropy").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!(
                    "get_memetic_status: UTL processor missing 'entropy' field - system is broken"
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'entropy'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let coherence = match utl_status.get("coherence").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!("get_memetic_status: UTL processor missing 'coherence' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'coherence'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let learning_score = match utl_status.get("learning_score").and_then(|v| v.as_f64()) {
            Some(v) => v as f32,
            None => {
                error!("get_memetic_status: UTL processor missing 'learning_score' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'learning_score'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        let consolidation_phase = match utl_status
            .get("consolidation_phase")
            .and_then(|v| v.as_str())
        {
            Some(phase) => phase,
            None => {
                error!("get_memetic_status: UTL processor missing 'consolidation_phase' field - system is broken");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    "UTL processor returned incomplete status: missing 'consolidation_phase'. \
                     This indicates a broken UTL system that must be fixed."
                        .to_string(),
                );
            }
        };

        // TASK-EMB-024: Get REAL layer statuses from LayerStatusProvider
        let perception_status = self
            .layer_status_provider
            .perception_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: perception_status FAILED");
                "error".to_string()
            });
        let memory_status = self
            .layer_status_provider
            .memory_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: memory_status FAILED");
                "error".to_string()
            });
        let action_status = self
            .layer_status_provider
            .action_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: action_status FAILED");
                "error".to_string()
            });
        let meta_status = self
            .layer_status_provider
            .meta_status()
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|e| {
                error!(error = %e, "get_memetic_status: meta_status FAILED");
                "error".to_string()
            });

        self.tool_result_with_pulse(
            id,
            json!({
                "phase": lifecycle_phase,
                "fingerprintCount": fingerprint_count,
                "embedderCount": NUM_EMBEDDERS,
                "storageBackend": self.teleological_store.backend_type().to_string(),
                "storageSizeBytes": self.teleological_store.storage_size_bytes(),
                "utl": {
                    "entropy": entropy,
                    "coherence": coherence,
                    "learningScore": learning_score,
                    "consolidationPhase": consolidation_phase
                },
                "layers": {
                    "perception": perception_status,
                    "memory": memory_status,
                    "action": action_status,
                    "meta": meta_status
                }
            }),
        )
    }
}
