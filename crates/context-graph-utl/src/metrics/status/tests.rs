//! Tests for UtlStatus and MCP response types.

#![allow(clippy::field_reassign_with_default)]

use super::*;

#[test]
fn test_status_default() {
    let status = UtlStatus::default();

    assert_eq!(status.lifecycle_stage, LifecycleStage::Infancy);
    assert_eq!(status.interaction_count, 0);
    assert_eq!(status.phase_angle, 0.0);
}

#[test]
fn test_status_is_encoding() {
    let mut status = UtlStatus::default();
    status.consolidation_phase = ConsolidationPhase::NREM;

    assert!(status.is_encoding());
    assert!(!status.is_consolidating());
    assert!(!status.is_wake());
}

#[test]
fn test_status_is_consolidating() {
    let mut status = UtlStatus::default();
    status.consolidation_phase = ConsolidationPhase::REM;

    assert!(!status.is_encoding());
    assert!(status.is_consolidating());
    assert!(!status.is_wake());
}

#[test]
fn test_status_is_wake() {
    let mut status = UtlStatus::default();
    status.consolidation_phase = ConsolidationPhase::Wake;

    assert!(!status.is_encoding());
    assert!(!status.is_consolidating());
    assert!(status.is_wake());
}

#[test]
fn test_status_is_novelty_seeking() {
    let mut status = UtlStatus::default();
    status.lifecycle_stage = LifecycleStage::Infancy;

    assert!(status.is_novelty_seeking());
    assert!(!status.is_consolidation_focused());
    assert!(!status.is_balanced());
}

#[test]
fn test_status_is_balanced() {
    let mut status = UtlStatus::default();
    status.lifecycle_stage = LifecycleStage::Growth;

    assert!(!status.is_novelty_seeking());
    assert!(!status.is_consolidation_focused());
    assert!(status.is_balanced());
}

#[test]
fn test_status_is_consolidation_focused() {
    let mut status = UtlStatus::default();
    status.lifecycle_stage = LifecycleStage::Maturity;

    assert!(!status.is_novelty_seeking());
    assert!(status.is_consolidation_focused());
    assert!(!status.is_balanced());
}

#[test]
fn test_status_summary() {
    let status = UtlStatus {
        lifecycle_stage: LifecycleStage::Growth,
        interaction_count: 150,
        consolidation_phase: ConsolidationPhase::REM,
        metrics: UtlComputationMetrics {
            avg_learning_magnitude: 0.654,
            ..Default::default()
        },
        ..Default::default()
    };

    let summary = status.summary();

    assert!(summary.contains("UTL:"));
    assert!(summary.contains("Growth"));
    assert!(summary.contains("150"));
    assert!(summary.contains("REM"));
    assert!(summary.contains("0.654"));
}

#[test]
fn test_status_to_mcp_response() {
    let status = UtlStatus {
        lifecycle_stage: LifecycleStage::Growth,
        interaction_count: 100,
        phase_angle: 1.57,
        consolidation_phase: ConsolidationPhase::Wake,
        current_thresholds: StageThresholds {
            entropy_trigger: 0.7,
            coherence_trigger: 0.5,
            min_importance_store: 0.3,
            consolidation_threshold: 0.6,
        },
        metrics: UtlComputationMetrics {
            avg_learning_magnitude: 0.6,
            avg_delta_s: 0.4,
            avg_delta_c: 0.5,
            ..Default::default()
        },
        ..Default::default()
    };

    let response = status.to_mcp_response();

    assert_eq!(response.lifecycle_phase, "Growth");
    assert_eq!(response.interaction_count, 100);
    assert_eq!(response.entropy, 0.4);
    assert_eq!(response.coherence, 0.5);
    assert_eq!(response.learning_score, 0.6);
    assert_eq!(response.phase_angle, 1.57);
    assert_eq!(response.consolidation_phase, "Wake");
    assert_eq!(response.thresholds.entropy_trigger, 0.7);
}

#[test]
fn test_status_serialization() {
    let status = UtlStatus {
        lifecycle_stage: LifecycleStage::Maturity,
        interaction_count: 500,
        phase_angle: 2.5,
        consolidation_phase: ConsolidationPhase::REM,
        ..Default::default()
    };

    let json = serde_json::to_string(&status).expect("serialize");
    let parsed: UtlStatus = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(status, parsed);
}

#[test]
fn test_thresholds_response_from() {
    let thresholds = StageThresholds {
        entropy_trigger: 0.9,
        coherence_trigger: 0.2,
        min_importance_store: 0.1,
        consolidation_threshold: 0.3,
    };

    let response = ThresholdsResponse::from(&thresholds);

    assert_eq!(response.entropy_trigger, 0.9);
    assert_eq!(response.coherence_trigger, 0.2);
    assert_eq!(response.min_importance_store, 0.1);
    assert_eq!(response.consolidation_threshold, 0.3);
}

#[test]
fn test_thresholds_response_serialization() {
    let response = ThresholdsResponse {
        entropy_trigger: 0.8,
        coherence_trigger: 0.4,
        min_importance_store: 0.2,
        consolidation_threshold: 0.5,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: ThresholdsResponse = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(response, parsed);
}

#[test]
fn test_status_response_serialization() {
    let response = UtlStatusResponse {
        lifecycle_phase: "Growth".to_string(),
        interaction_count: 200,
        entropy: 0.45,
        coherence: 0.55,
        learning_score: 0.7,
        johari_quadrant: "Open".to_string(),
        consolidation_phase: "Wake".to_string(),
        phase_angle: 1.0,
        thresholds: ThresholdsResponse::default(),
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: UtlStatusResponse = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(response, parsed);
}
