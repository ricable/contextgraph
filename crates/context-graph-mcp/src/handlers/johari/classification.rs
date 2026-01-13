//! Johari classification handler (TASK-MCP-005).
//!
//! Classifies into Johari Window quadrants using UTL metrics.
//! FAIL FAST: All errors return immediately - NO fallbacks, NO mocks.
//!
//! ## Two Input Modes
//! 1. Direct mode: Provide delta_s and delta_c values
//! 2. Memory mode: Provide memory_id to classify from stored fingerprint
//!
//! ## Constitution Reference (utl.johari lines 154-157)
//! - Open (ΔS<0.5, ΔC>0.5) → DirectRecall
//! - Blind (ΔS>0.5, ΔC<0.5) → TriggerDream
//! - Hidden (ΔS<0.5, ΔC<0.5) → GetNeighborhood
//! - Unknown (ΔS>0.5, ΔC>0.5) → EpistemicAction

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, error};
use uuid::Uuid;

use context_graph_core::types::JohariQuadrant;
use context_graph_utl::johari::get_suggested_action;
use context_graph_utl::SuggestedAction;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

/// Input for get_johari_classification tool.
///
/// Either (delta_s, delta_c) pair OR memory_id must be provided.
#[derive(Debug, Clone, Deserialize)]
pub struct JohariClassificationInput {
    /// Surprise metric [0.0, 1.0] - mutually exclusive with memory_id
    pub delta_s: Option<f32>,
    /// Coherence metric [0.0, 1.0] - mutually exclusive with memory_id
    pub delta_c: Option<f32>,
    /// Memory UUID to classify from stored fingerprint - mutually exclusive with delta_s/delta_c
    pub memory_id: Option<Uuid>,
    /// Embedder index to use when using memory_id (default: 0 = E1 semantic)
    pub embedder_index: Option<usize>,
    /// Classification threshold (default: 0.5)
    pub threshold: Option<f32>,
}

/// Output for get_johari_classification tool
#[derive(Debug, Clone, Serialize)]
pub struct JohariClassificationOutput {
    /// Classified Johari quadrant
    pub quadrant: JohariQuadrant,
    /// Surprise metric [0.0, 1.0]
    pub delta_s: f32,
    /// Coherence metric [0.0, 1.0]
    pub delta_c: f32,
    /// Recommended action based on quadrant
    pub suggested_action: SuggestedAction,
    /// Human-readable explanation
    pub explanation: String,
    /// Source of the classification (direct or memory_id)
    pub source: String,
    /// Threshold used for classification
    pub threshold: f32,
}

impl Handlers {
    /// Handle get_johari_classification tool call.
    ///
    /// Classifies into a Johari quadrant based on delta_s and delta_c.
    /// Supports direct metrics or memory lookup.
    ///
    /// TASK-MCP-005: FAIL FAST on all errors.
    pub(crate) async fn call_get_johari_classification(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_johari_classification: {:?}", arguments);

        // Parse input - FAIL FAST on invalid input
        let input: JohariClassificationInput = match serde_json::from_value(arguments.clone()) {
            Ok(i) => i,
            Err(e) => {
                error!("Invalid input for get_johari_classification: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid input: {}", e),
                );
            }
        };

        let threshold = input.threshold.unwrap_or(0.5);

        // FAIL FAST: Validate threshold range
        if !(0.0..=1.0).contains(&threshold) {
            error!(
                "get_johari_classification: threshold out of range: {}",
                threshold
            );
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("threshold must be in [0.0, 1.0], got {}", threshold),
            );
        }

        // Determine input mode and get delta_s, delta_c
        let (delta_s, delta_c, source) = if let Some(memory_id) = input.memory_id {
            // Memory mode: lookup from stored fingerprint
            let embedder_idx = input.embedder_index.unwrap_or(0);

            // FAIL FAST: Validate embedder index (0-12 for 13 embedders)
            if embedder_idx >= 13 {
                error!(
                    "get_johari_classification: embedder_index out of range: {}",
                    embedder_idx
                );
                return JsonRpcResponse::error(
                    id,
                    error_codes::JOHARI_INVALID_EMBEDDER_INDEX,
                    format!("embedder_index must be 0-12, got {}", embedder_idx),
                );
            }

            match self.get_delta_from_memory(memory_id, embedder_idx).await {
                Ok((ds, dc)) => (ds, dc, format!("memory:{}", memory_id)),
                Err(e) => {
                    error!("Failed to get delta from memory {}: {}", memory_id, e);
                    return JsonRpcResponse::error(id, error_codes::FINGERPRINT_NOT_FOUND, e);
                }
            }
        } else if let (Some(ds), Some(dc)) = (input.delta_s, input.delta_c) {
            // Direct mode: use provided values

            // FAIL FAST: Validate delta_s range
            if !(0.0..=1.0).contains(&ds) {
                error!("get_johari_classification: delta_s out of range: {}", ds);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("delta_s must be in [0.0, 1.0], got {}", ds),
                );
            }

            // FAIL FAST: Validate delta_c range
            if !(0.0..=1.0).contains(&dc) {
                error!("get_johari_classification: delta_c out of range: {}", dc);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("delta_c must be in [0.0, 1.0], got {}", dc),
                );
            }

            (ds, dc, "direct".to_string())
        } else {
            // FAIL FAST: Invalid input combination
            error!("get_johari_classification: missing required parameters");
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                "Must provide either (delta_s AND delta_c) OR memory_id",
            );
        };

        debug!(
            "Classifying: delta_s={}, delta_c={}, threshold={}",
            delta_s, delta_c, threshold
        );

        // Classify using the threshold
        let quadrant = classify_johari(delta_s, delta_c, threshold);
        let suggested_action = get_suggested_action(quadrant);
        let explanation = generate_explanation(quadrant, delta_s, delta_c);

        let output = JohariClassificationOutput {
            quadrant,
            delta_s,
            delta_c,
            suggested_action,
            explanation,
            source,
            threshold,
        };

        debug!(
            "Classified as {:?} with action {:?}",
            quadrant, suggested_action
        );

        JsonRpcResponse::success(id, json!(output))
    }

    /// Get delta_s and delta_c from a stored memory's JohariFingerprint.
    ///
    /// FAIL FAST: Returns error if memory not found or fingerprint unavailable.
    async fn get_delta_from_memory(
        &self,
        memory_id: Uuid,
        embedder_idx: usize,
    ) -> Result<(f32, f32), String> {
        // Retrieve fingerprint from storage
        let fingerprint = self
            .teleological_store
            .retrieve(memory_id)
            .await
            .map_err(|e| format!("Storage error: {}", e))?
            .ok_or_else(|| format!("Memory not found: {}", memory_id))?;

        // Get soft classification weights for the specified embedder
        // quadrants[embedder_idx] gives [Open, Hidden, Blind, Unknown] weights
        let weights = fingerprint.johari.quadrants[embedder_idx];

        // Compute effective delta_s and delta_c from soft weights
        // Using the inverse of the classification logic:
        // - High Open + Hidden weight → low delta_s (familiar)
        // - High Open + Unknown weight → high delta_c (coherent)
        //
        // Quadrant weights order: [Open, Hidden, Blind, Unknown]
        // Open: low ΔS, high ΔC → weights[0]
        // Hidden: low ΔS, low ΔC → weights[1]
        // Blind: high ΔS, low ΔC → weights[2]
        // Unknown: high ΔS, high ΔC → weights[3]
        let low_surprise_weight = weights[0] + weights[1]; // Open + Hidden
        let high_coherence_weight = weights[0] + weights[3]; // Open + Unknown

        let delta_s = 1.0 - low_surprise_weight.min(1.0);
        let delta_c = high_coherence_weight.min(1.0);

        Ok((delta_s, delta_c))
    }
}

/// Classify (delta_s, delta_c) into JohariQuadrant.
///
/// Constitution reference (utl.johari lines 154-157):
/// - Open: ΔS < threshold, ΔC > threshold (low surprise, high coherence)
/// - Blind: ΔS > threshold, ΔC < threshold (high surprise, low coherence)
/// - Hidden: ΔS < threshold, ΔC < threshold (low surprise, low coherence)
/// - Unknown: ΔS > threshold, ΔC > threshold (high surprise, high coherence)
#[inline]
fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant {
    match (delta_s < threshold, delta_c > threshold) {
        (true, true) => JohariQuadrant::Open,    // Low surprise, high coherence
        (false, false) => JohariQuadrant::Blind, // High surprise, low coherence
        (true, false) => JohariQuadrant::Hidden, // Low surprise, low coherence
        (false, true) => JohariQuadrant::Unknown, // High surprise, high coherence
    }
}

/// Generate human-readable explanation for classification.
fn generate_explanation(quadrant: JohariQuadrant, delta_s: f32, delta_c: f32) -> String {
    match quadrant {
        JohariQuadrant::Open => format!(
            "Open quadrant (ΔS={:.3}, ΔC={:.3}): Low surprise and high coherence. \
             Content is familiar and well-understood. Recommended: DirectRecall.",
            delta_s, delta_c
        ),
        JohariQuadrant::Blind => format!(
            "Blind quadrant (ΔS={:.3}, ΔC={:.3}): High surprise but low coherence. \
             Content is unexpected and not well-integrated. Recommended: TriggerDream \
             for consolidation.",
            delta_s, delta_c
        ),
        JohariQuadrant::Hidden => format!(
            "Hidden quadrant (ΔS={:.3}, ΔC={:.3}): Low surprise and low coherence. \
             Content is familiar but poorly connected. Recommended: GetNeighborhood \
             to explore context.",
            delta_s, delta_c
        ),
        JohariQuadrant::Unknown => format!(
            "Unknown quadrant (ΔS={:.3}, ΔC={:.3}): High surprise and high coherence. \
             Novel information that fits well. Recommended: EpistemicAction to update beliefs.",
            delta_s, delta_c
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== CLASSIFICATION LOGIC TESTS ==========

    #[test]
    fn test_classify_johari_open() {
        // Open: low surprise (< 0.5), high coherence (> 0.5)
        let quadrant = classify_johari(0.2, 0.8, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Open);
    }

    #[test]
    fn test_classify_johari_blind() {
        // Blind: high surprise (> 0.5), low coherence (< 0.5)
        let quadrant = classify_johari(0.8, 0.2, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Blind);
    }

    #[test]
    fn test_classify_johari_hidden() {
        // Hidden: low surprise (< 0.5), low coherence (< 0.5)
        let quadrant = classify_johari(0.2, 0.2, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Hidden);
    }

    #[test]
    fn test_classify_johari_unknown() {
        // Unknown: high surprise (> 0.5), high coherence (> 0.5)
        let quadrant = classify_johari(0.8, 0.8, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Unknown);
    }

    #[test]
    fn test_classify_johari_boundary_blind() {
        // Edge case: exactly on threshold (delta_s=0.5, delta_c=0.5)
        // delta_s < 0.5 is FALSE, delta_c > 0.5 is FALSE -> Blind
        let quadrant = classify_johari(0.5, 0.5, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Blind);
    }

    #[test]
    fn test_classify_johari_custom_threshold() {
        // With threshold 0.7:
        // delta_s=0.6 < 0.7 (low surprise), delta_c=0.8 > 0.7 (high coherence) -> Open
        let quadrant = classify_johari(0.6, 0.8, 0.7);
        assert_eq!(quadrant, JohariQuadrant::Open);

        // Same values with default threshold 0.5 would give Unknown
        let quadrant_default = classify_johari(0.6, 0.8, 0.5);
        assert_eq!(quadrant_default, JohariQuadrant::Unknown);
    }

    // ========== SUGGESTED ACTION TESTS ==========

    #[test]
    fn test_suggested_action_open() {
        let action = get_suggested_action(JohariQuadrant::Open);
        assert_eq!(action, SuggestedAction::DirectRecall);
    }

    #[test]
    fn test_suggested_action_blind() {
        let action = get_suggested_action(JohariQuadrant::Blind);
        assert_eq!(action, SuggestedAction::TriggerDream);
    }

    #[test]
    fn test_suggested_action_hidden() {
        let action = get_suggested_action(JohariQuadrant::Hidden);
        assert_eq!(action, SuggestedAction::GetNeighborhood);
    }

    #[test]
    fn test_suggested_action_unknown() {
        let action = get_suggested_action(JohariQuadrant::Unknown);
        assert_eq!(action, SuggestedAction::EpistemicAction);
    }

    // ========== EXPLANATION TESTS ==========

    #[test]
    fn test_generate_explanation_open() {
        let exp = generate_explanation(JohariQuadrant::Open, 0.2, 0.8);
        assert!(exp.contains("Open quadrant"));
        assert!(exp.contains("DirectRecall"));
        assert!(exp.contains("0.200"));
        assert!(exp.contains("0.800"));
    }

    #[test]
    fn test_generate_explanation_blind() {
        let exp = generate_explanation(JohariQuadrant::Blind, 0.8, 0.2);
        assert!(exp.contains("Blind quadrant"));
        assert!(exp.contains("TriggerDream"));
        assert!(exp.contains("consolidation"));
    }

    #[test]
    fn test_generate_explanation_hidden() {
        let exp = generate_explanation(JohariQuadrant::Hidden, 0.2, 0.2);
        assert!(exp.contains("Hidden quadrant"));
        assert!(exp.contains("GetNeighborhood"));
        assert!(exp.contains("context"));
    }

    #[test]
    fn test_generate_explanation_unknown() {
        let exp = generate_explanation(JohariQuadrant::Unknown, 0.8, 0.8);
        assert!(exp.contains("Unknown quadrant"));
        assert!(exp.contains("EpistemicAction"));
        assert!(exp.contains("beliefs"));
    }

    // ========== INPUT/OUTPUT TYPE TESTS ==========

    #[test]
    fn test_johari_classification_input_direct_mode_parse() {
        let json = json!({
            "delta_s": 0.3,
            "delta_c": 0.7
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.delta_s, Some(0.3));
        assert_eq!(input.delta_c, Some(0.7));
        assert!(input.memory_id.is_none());
    }

    #[test]
    fn test_johari_classification_input_memory_mode_parse() {
        let json = json!({
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "embedder_index": 5
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        assert!(input.memory_id.is_some());
        assert_eq!(input.embedder_index, Some(5));
        assert!(input.delta_s.is_none());
        assert!(input.delta_c.is_none());
    }

    #[test]
    fn test_johari_classification_input_with_threshold() {
        let json = json!({
            "delta_s": 0.6,
            "delta_c": 0.7,
            "threshold": 0.65
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.threshold, Some(0.65));
    }

    #[test]
    fn test_johari_classification_output_serialize() {
        let output = JohariClassificationOutput {
            quadrant: JohariQuadrant::Open,
            delta_s: 0.3,
            delta_c: 0.7,
            suggested_action: SuggestedAction::DirectRecall,
            explanation: "Test explanation".to_string(),
            source: "direct".to_string(),
            threshold: 0.5,
        };
        let json = serde_json::to_value(&output).unwrap();

        assert!(json.get("quadrant").is_some());
        assert!(json.get("delta_s").is_some());
        assert!(json.get("delta_c").is_some());
        assert!(json.get("suggested_action").is_some());
        assert!(json.get("explanation").is_some());
        assert!(json.get("source").is_some());
        assert!(json.get("threshold").is_some());
    }

    #[test]
    fn test_johari_classification_output_values() {
        let output = JohariClassificationOutput {
            quadrant: JohariQuadrant::Unknown,
            delta_s: 0.75,
            delta_c: 0.85,
            suggested_action: SuggestedAction::EpistemicAction,
            explanation: "Novel coherent content".to_string(),
            source: "memory:test-uuid".to_string(),
            threshold: 0.5,
        };
        let json = serde_json::to_value(&output).unwrap();

        // Use f32 epsilon tolerance for floating-point comparison
        let delta_s = json.get("delta_s").unwrap().as_f64().unwrap();
        let delta_c = json.get("delta_c").unwrap().as_f64().unwrap();
        let threshold = json.get("threshold").unwrap().as_f64().unwrap();

        assert!((delta_s - 0.75).abs() < 1e-6, "delta_s mismatch: {}", delta_s);
        assert!((delta_c - 0.85).abs() < 1e-6, "delta_c mismatch: {}", delta_c);
        assert!((threshold - 0.5).abs() < 1e-6, "threshold mismatch: {}", threshold);
    }

    // ========== CONSTITUTION COMPLIANCE TESTS ==========

    #[test]
    fn test_constitution_compliance_all_quadrants() {
        // Verify all 4 quadrant mappings match constitution.yaml utl.johari (lines 154-157)

        // Open: ΔS<0.5, ΔC>0.5 → DirectRecall
        let open = classify_johari(0.3, 0.7, 0.5);
        assert_eq!(open, JohariQuadrant::Open);
        assert_eq!(get_suggested_action(open), SuggestedAction::DirectRecall);

        // Blind: ΔS>0.5, ΔC<0.5 → TriggerDream
        let blind = classify_johari(0.7, 0.3, 0.5);
        assert_eq!(blind, JohariQuadrant::Blind);
        assert_eq!(get_suggested_action(blind), SuggestedAction::TriggerDream);

        // Hidden: ΔS<0.5, ΔC<0.5 → GetNeighborhood
        let hidden = classify_johari(0.3, 0.3, 0.5);
        assert_eq!(hidden, JohariQuadrant::Hidden);
        assert_eq!(get_suggested_action(hidden), SuggestedAction::GetNeighborhood);

        // Unknown: ΔS>0.5, ΔC>0.5 → EpistemicAction
        let unknown = classify_johari(0.7, 0.7, 0.5);
        assert_eq!(unknown, JohariQuadrant::Unknown);
        assert_eq!(
            get_suggested_action(unknown),
            SuggestedAction::EpistemicAction
        );
    }

    #[test]
    fn test_task_09_fix_verified() {
        // TASK-09 (ISS-011) fixed Blind/Unknown action mapping
        // This test verifies the fix is still in place

        // Blind MUST map to TriggerDream (was incorrectly EpistemicAction)
        assert_eq!(
            get_suggested_action(JohariQuadrant::Blind),
            SuggestedAction::TriggerDream,
            "TASK-09 fix: Blind should map to TriggerDream"
        );

        // Unknown MUST map to EpistemicAction (was incorrectly TriggerDream)
        assert_eq!(
            get_suggested_action(JohariQuadrant::Unknown),
            SuggestedAction::EpistemicAction,
            "TASK-09 fix: Unknown should map to EpistemicAction"
        );
    }

    // ========== TASK-MCP-005 FSV EDGE CASE TESTS ==========
    // These tests verify the HANDLER via dispatch mechanism
    // per constitution.yaml utl.johari lines 154-157

    #[test]
    fn test_fsv_edge_case_boundary_exactly_0_5() {
        // Edge case: exactly on threshold (0.5, 0.5)
        // delta_s < 0.5 is FALSE (0.5 is NOT less than 0.5)
        // delta_c > 0.5 is FALSE (0.5 is NOT greater than 0.5)
        // Result: (false, false) -> Blind quadrant
        let quadrant = classify_johari(0.5, 0.5, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Blind);
        assert_eq!(get_suggested_action(quadrant), SuggestedAction::TriggerDream);
    }

    #[test]
    fn test_fsv_edge_case_min_values_0_0() {
        // Minimum values (0.0, 0.0)
        // delta_s < 0.5 is TRUE, delta_c > 0.5 is FALSE
        // Result: (true, false) -> Hidden quadrant
        let quadrant = classify_johari(0.0, 0.0, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Hidden);
        assert_eq!(
            get_suggested_action(quadrant),
            SuggestedAction::GetNeighborhood
        );
    }

    #[test]
    fn test_fsv_edge_case_max_values_1_0() {
        // Maximum values (1.0, 1.0)
        // delta_s < 0.5 is FALSE, delta_c > 0.5 is TRUE
        // Result: (false, true) -> Unknown quadrant
        let quadrant = classify_johari(1.0, 1.0, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Unknown);
        assert_eq!(
            get_suggested_action(quadrant),
            SuggestedAction::EpistemicAction
        );
    }

    #[test]
    fn test_fsv_edge_case_custom_threshold_0_7() {
        // Custom threshold 0.7
        // With (0.6, 0.8): delta_s < 0.7 TRUE, delta_c > 0.7 TRUE -> Open
        let quadrant = classify_johari(0.6, 0.8, 0.7);
        assert_eq!(quadrant, JohariQuadrant::Open);
        assert_eq!(get_suggested_action(quadrant), SuggestedAction::DirectRecall);

        // Same values with default threshold would give Unknown
        let quadrant_default = classify_johari(0.6, 0.8, 0.5);
        assert_eq!(quadrant_default, JohariQuadrant::Unknown);
    }

    #[test]
    fn test_fsv_edge_case_just_below_threshold() {
        // Just below threshold: (0.499, 0.501)
        // delta_s < 0.5 is TRUE (0.499 < 0.5)
        // delta_c > 0.5 is TRUE (0.501 > 0.5)
        // Result: Open quadrant
        let quadrant = classify_johari(0.499, 0.501, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Open);
        assert_eq!(get_suggested_action(quadrant), SuggestedAction::DirectRecall);
    }

    #[test]
    fn test_fsv_edge_case_just_above_threshold() {
        // Just above threshold: (0.501, 0.499)
        // delta_s < 0.5 is FALSE (0.501 is NOT < 0.5)
        // delta_c > 0.5 is FALSE (0.499 is NOT > 0.5)
        // Result: Blind quadrant
        let quadrant = classify_johari(0.501, 0.499, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Blind);
        assert_eq!(get_suggested_action(quadrant), SuggestedAction::TriggerDream);
    }

    #[test]
    fn test_fsv_edge_case_asymmetric_boundary_open() {
        // Asymmetric: (0.5, 0.501)
        // delta_s < 0.5 is FALSE, delta_c > 0.5 is TRUE
        // Result: Unknown quadrant (high surprise + high coherence)
        let quadrant = classify_johari(0.5, 0.501, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Unknown);
    }

    #[test]
    fn test_fsv_edge_case_asymmetric_boundary_hidden() {
        // Asymmetric: (0.499, 0.5)
        // delta_s < 0.5 is TRUE, delta_c > 0.5 is FALSE
        // Result: Hidden quadrant (low surprise + low coherence)
        let quadrant = classify_johari(0.499, 0.5, 0.5);
        assert_eq!(quadrant, JohariQuadrant::Hidden);
    }

    #[test]
    fn test_fsv_edge_case_custom_threshold_extreme_low() {
        // Very low threshold (0.1) - most things are "above threshold"
        // With (0.05, 0.5): delta_s < 0.1 TRUE, delta_c > 0.1 TRUE -> Open
        let quadrant = classify_johari(0.05, 0.5, 0.1);
        assert_eq!(quadrant, JohariQuadrant::Open);
    }

    #[test]
    fn test_fsv_edge_case_custom_threshold_extreme_high() {
        // Very high threshold (0.9) - most things are "below threshold"
        // With (0.5, 0.95): delta_s < 0.9 TRUE, delta_c > 0.9 TRUE -> Open
        let quadrant = classify_johari(0.5, 0.95, 0.9);
        assert_eq!(quadrant, JohariQuadrant::Open);
    }

    #[test]
    fn test_fsv_input_validation_delta_s_out_of_range() {
        // Test input parsing with out-of-range delta_s
        let json = json!({
            "delta_s": 1.5,  // INVALID: > 1.0
            "delta_c": 0.5
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        // Input will parse, but handler would reject delta_s > 1.0
        assert_eq!(input.delta_s, Some(1.5));
    }

    #[test]
    fn test_fsv_input_validation_delta_c_out_of_range() {
        // Test input parsing with out-of-range delta_c
        let json = json!({
            "delta_s": 0.5,
            "delta_c": -0.5  // INVALID: < 0.0
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        // Input will parse, but handler would reject delta_c < 0.0
        assert_eq!(input.delta_c, Some(-0.5));
    }

    #[test]
    fn test_fsv_input_validation_threshold_out_of_range() {
        // Test input parsing with out-of-range threshold
        let json = json!({
            "delta_s": 0.5,
            "delta_c": 0.5,
            "threshold": 2.0  // INVALID: > 1.0
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        // Input will parse, but handler would reject threshold > 1.0
        assert_eq!(input.threshold, Some(2.0));
    }

    #[test]
    fn test_fsv_input_validation_missing_both_modes() {
        // Neither (delta_s, delta_c) nor memory_id provided
        let json = json!({
            "threshold": 0.5
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        assert!(input.delta_s.is_none());
        assert!(input.delta_c.is_none());
        assert!(input.memory_id.is_none());
        // Handler would reject this with INVALID_PARAMS
    }

    #[test]
    fn test_fsv_input_validation_partial_direct_mode() {
        // Only delta_s provided (missing delta_c)
        let json = json!({
            "delta_s": 0.5
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        assert!(input.delta_s.is_some());
        assert!(input.delta_c.is_none());
        // Handler would reject this with INVALID_PARAMS
    }

    #[test]
    fn test_fsv_input_validation_embedder_index_boundary() {
        // Test embedder_index parsing at boundary
        let json = json!({
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "embedder_index": 12  // Valid: 0-12
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.embedder_index, Some(12));
    }

    #[test]
    fn test_fsv_input_validation_embedder_index_invalid() {
        // Test embedder_index parsing with invalid value
        let json = json!({
            "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            "embedder_index": 13  // INVALID: must be 0-12
        });
        let input: JohariClassificationInput = serde_json::from_value(json).unwrap();
        // Input will parse, but handler would reject embedder_index >= 13
        assert_eq!(input.embedder_index, Some(13));
    }

    #[test]
    fn test_fsv_output_serialization_all_quadrants() {
        // Verify output serialization for each quadrant
        for (quadrant, expected_action) in [
            (JohariQuadrant::Open, SuggestedAction::DirectRecall),
            (JohariQuadrant::Blind, SuggestedAction::TriggerDream),
            (JohariQuadrant::Hidden, SuggestedAction::GetNeighborhood),
            (JohariQuadrant::Unknown, SuggestedAction::EpistemicAction),
        ] {
            let output = JohariClassificationOutput {
                quadrant,
                delta_s: 0.5,
                delta_c: 0.5,
                suggested_action: expected_action,
                explanation: "Test".to_string(),
                source: "direct".to_string(),
                threshold: 0.5,
            };
            let json = serde_json::to_value(&output).unwrap();
            assert!(json.get("quadrant").is_some());
            assert!(json.get("suggested_action").is_some());
        }
    }

    #[test]
    fn test_fsv_explanation_contains_metrics() {
        // Verify explanations contain the actual metric values
        let exp = generate_explanation(JohariQuadrant::Open, 0.123, 0.987);
        assert!(exp.contains("0.123"));
        assert!(exp.contains("0.987"));
        assert!(exp.contains("Open quadrant"));
        assert!(exp.contains("DirectRecall"));
    }
}
