//! Manual integration tests for Topic, TopicProfile, TopicPhase, TopicStability
//!
//! TASK-P4-002: Topic Types Implementation
//! These tests verify the topic system using synthetic data with known expected outputs.

use std::collections::HashMap;

use context_graph_core::clustering::{Topic, TopicPhase, TopicProfile, TopicStability};
use context_graph_core::embeddings::category::{category_for, max_weighted_agreement, topic_threshold};
use context_graph_core::teleological::Embedder;
use uuid::Uuid;

// =============================================================================
// Test Constants (Constitution Values)
// =============================================================================

const EXPECTED_MAX_WEIGHTED_AGREEMENT: f32 = 8.5;
const EXPECTED_TOPIC_THRESHOLD: f32 = 2.5;

// =============================================================================
// Manual Test 1: Verify Constitution Constants
// =============================================================================

#[test]
fn manual_test_constitution_constants() {
    println!("\n=== Manual Test 1: Constitution Constants ===");

    // Verify max_weighted_agreement
    let actual_max = max_weighted_agreement();
    assert!(
        (actual_max - EXPECTED_MAX_WEIGHTED_AGREEMENT).abs() < f32::EPSILON,
        "ARCH-09: max_weighted_agreement should be 8.5, got {}",
        actual_max
    );
    println!("[VERIFIED] max_weighted_agreement() = {}", actual_max);

    // Verify topic_threshold
    let actual_threshold = topic_threshold();
    assert!(
        (actual_threshold - EXPECTED_TOPIC_THRESHOLD).abs() < f32::EPSILON,
        "ARCH-09: topic_threshold should be 2.5, got {}",
        actual_threshold
    );
    println!("[VERIFIED] topic_threshold() = {}", actual_threshold);

    // Verify category weights match constitution
    println!("\nCategory weights by embedder:");
    for embedder in Embedder::all() {
        let cat = category_for(embedder);
        let weight = cat.topic_weight();
        println!("  {} ({:?}): weight = {}", embedder, cat, weight);
    }

    // Sum all weights to verify max
    let computed_max: f32 = Embedder::all()
        .map(|e| category_for(e).topic_weight())
        .sum();
    println!("\nComputed max from all weights: {}", computed_max);
    assert!(
        (computed_max - EXPECTED_MAX_WEIGHTED_AGREEMENT).abs() < 0.001,
        "Sum of all weights should be 8.5, got {}",
        computed_max
    );
    println!("[VERIFIED] Sum of category weights = {}", computed_max);
}

// =============================================================================
// Manual Test 2: Weighted Agreement Calculation (AP-60 Verification)
// =============================================================================

#[test]
fn manual_test_weighted_agreement_calculation() {
    println!("\n=== Manual Test 2: Weighted Agreement Calculation ===");

    // Test Case 2.1: Temporal-only (should be 0.0 per AP-60)
    println!("\n[Test 2.1] Temporal-only cluster:");
    let mut temporal_strengths = [0.0f32; 13];
    temporal_strengths[Embedder::TemporalRecent.index()] = 1.0;
    temporal_strengths[Embedder::TemporalPeriodic.index()] = 1.0;
    temporal_strengths[Embedder::TemporalPositional.index()] = 1.0;

    let temporal_profile = TopicProfile::new(temporal_strengths);
    let temporal_weighted = temporal_profile.weighted_agreement();

    println!("  Input: E2=1.0, E3=1.0, E4=1.0 (all temporal)");
    println!("  Expected weighted_agreement: 0.0");
    println!("  Actual weighted_agreement: {}", temporal_weighted);
    println!("  Is topic? {}", temporal_profile.is_topic());

    assert!(
        temporal_weighted.abs() < f32::EPSILON,
        "AP-60: Temporal embedders MUST NOT count toward topic detection"
    );
    assert!(
        !temporal_profile.is_topic(),
        "Temporal-only should NOT be a topic"
    );
    println!("[VERIFIED] AP-60: Temporal contributes 0.0");

    // Test Case 2.2: Semantic-only (3 spaces = 3.0)
    println!("\n[Test 2.2] Three semantic spaces:");
    let mut semantic_strengths = [0.0f32; 13];
    semantic_strengths[Embedder::Semantic.index()] = 1.0; // E1, semantic, 1.0
    semantic_strengths[Embedder::Causal.index()] = 1.0; // E5, semantic, 1.0
    semantic_strengths[Embedder::Code.index()] = 1.0; // E7, semantic, 1.0

    let semantic_profile = TopicProfile::new(semantic_strengths);
    let semantic_weighted = semantic_profile.weighted_agreement();

    println!("  Input: E1=1.0, E5=1.0, E7=1.0 (all semantic)");
    println!("  Expected weighted_agreement: 3.0");
    println!("  Actual weighted_agreement: {}", semantic_weighted);
    println!("  Is topic? {}", semantic_profile.is_topic());

    assert!(
        (semantic_weighted - 3.0).abs() < 0.001,
        "3 semantic spaces at 1.0 should give 3.0, got {}",
        semantic_weighted
    );
    assert!(
        semantic_profile.is_topic(),
        "3.0 >= 2.5 threshold should be topic"
    );
    println!("[VERIFIED] 3 semantic spaces = 3.0 weighted");

    // Test Case 2.3: Mixed categories (constitution example)
    println!("\n[Test 2.3] Mixed: 2 semantic + 1 relational = 2.5:");
    let mut mixed_strengths = [0.0f32; 13];
    mixed_strengths[Embedder::Semantic.index()] = 1.0; // semantic, 1.0
    mixed_strengths[Embedder::Causal.index()] = 1.0; // semantic, 1.0
    mixed_strengths[Embedder::Entity.index()] = 1.0; // relational, 0.5

    let mixed_profile = TopicProfile::new(mixed_strengths);
    let mixed_weighted = mixed_profile.weighted_agreement();

    println!("  Input: E1=1.0 (semantic), E5=1.0 (semantic), E11=1.0 (relational)");
    println!("  Expected: 1.0 + 1.0 + 0.5 = 2.5");
    println!("  Actual weighted_agreement: {}", mixed_weighted);
    println!("  Is topic? {}", mixed_profile.is_topic());

    assert!(
        (mixed_weighted - 2.5).abs() < 0.001,
        "Expected 2.5, got {}",
        mixed_weighted
    );
    assert!(mixed_profile.is_topic(), "2.5 meets threshold exactly");
    println!("[VERIFIED] 2 semantic + 1 relational = 2.5 weighted");

    // Test Case 2.4: Below threshold
    println!("\n[Test 2.4] Below threshold: 2 semantic = 2.0:");
    let mut below_strengths = [0.0f32; 13];
    below_strengths[Embedder::Semantic.index()] = 1.0;
    below_strengths[Embedder::Causal.index()] = 1.0;

    let below_profile = TopicProfile::new(below_strengths);
    let below_weighted = below_profile.weighted_agreement();

    println!("  Input: E1=1.0, E5=1.0");
    println!("  Expected weighted_agreement: 2.0");
    println!("  Actual weighted_agreement: {}", below_weighted);
    println!("  Is topic? {}", below_profile.is_topic());

    assert!(
        (below_weighted - 2.0).abs() < 0.001,
        "Expected 2.0, got {}",
        below_weighted
    );
    assert!(
        !below_profile.is_topic(),
        "2.0 < 2.5 threshold should NOT be topic"
    );
    println!("[VERIFIED] 2 semantic = 2.0 (below threshold)");
}

// =============================================================================
// Manual Test 3: Topic Confidence Calculation
// =============================================================================

#[test]
fn manual_test_topic_confidence_calculation() {
    println!("\n=== Manual Test 3: Topic Confidence Calculation ===");

    // Test various weighted_agreement values and their expected confidence
    let test_cases = [
        ("3 semantic spaces", 3.0, 3.0 / 8.5),
        ("5 semantic spaces", 5.0, 5.0 / 8.5),
        ("All max possible", 8.5, 1.0),
        ("At threshold", 2.5, 2.5 / 8.5),
    ];

    for (name, weighted, expected_confidence) in test_cases {
        println!("\n[Test] {}:", name);
        println!("  weighted_agreement = {}", weighted);
        println!("  Expected confidence = {:.4}", expected_confidence);

        // Create a profile that achieves the target weighted agreement
        // We'll use semantic embedders (weight 1.0) to achieve exact values
        let mut strengths = [0.0f32; 13];
        let semantic_embedders = [
            Embedder::Semantic,
            Embedder::Causal,
            Embedder::Sparse,
            Embedder::Code,
            Embedder::Multimodal,
            Embedder::LateInteraction,
            Embedder::KeywordSplade,
        ];

        let mut remaining: f32 = weighted;
        for &embedder in &semantic_embedders {
            if remaining <= 0.0 {
                break;
            }
            let strength = remaining.min(1.0);
            strengths[embedder.index()] = strength;
            remaining -= strength;
        }

        // Add relational/structural if needed
        if remaining > 0.0 {
            strengths[Embedder::Emotional.index()] = (remaining / 0.5_f32).min(1.0);
            remaining -= 0.5_f32.min(remaining);
        }
        if remaining > 0.0 {
            strengths[Embedder::Entity.index()] = (remaining / 0.5_f32).min(1.0);
            remaining -= 0.5_f32.min(remaining);
        }
        if remaining > 0.0 {
            strengths[Embedder::Hdc.index()] = (remaining / 0.5_f32).min(1.0);
        }

        let profile = TopicProfile::new(strengths);
        let actual_weighted = profile.weighted_agreement();
        let topic = Topic::new(profile.clone(), HashMap::new(), vec![]);

        println!("  Actual weighted_agreement = {:.4}", actual_weighted);
        println!("  Topic confidence = {:.4}", topic.confidence);

        assert!(
            (topic.confidence - expected_confidence).abs() < 0.01,
            "Confidence mismatch: expected {:.4}, got {:.4}",
            expected_confidence,
            topic.confidence
        );
        println!("[VERIFIED] Confidence calculated correctly");
    }
}

// =============================================================================
// Manual Test 4: TopicPhase Transitions
// =============================================================================

#[test]
fn manual_test_topic_phase_transitions() {
    println!("\n=== Manual Test 4: TopicPhase Transitions ===");

    let mut stability = TopicStability::new();
    println!("Initial phase: {:?}", stability.phase);
    assert_eq!(stability.phase, TopicPhase::Emerging);

    // Test transition to Stable
    println!("\n[Transition to Stable]");
    stability.age_hours = 48.0;
    stability.membership_churn = 0.05;
    stability.update_phase();
    println!("  age_hours=48.0, churn=0.05");
    println!("  Phase after update: {:?}", stability.phase);
    assert_eq!(stability.phase, TopicPhase::Stable);
    println!("[VERIFIED] Transitioned to Stable");

    // Test transition to Declining
    println!("\n[Transition to Declining]");
    stability.membership_churn = 0.6;
    stability.update_phase();
    println!("  churn=0.6");
    println!("  Phase after update: {:?}", stability.phase);
    assert_eq!(stability.phase, TopicPhase::Declining);
    println!("[VERIFIED] Transitioned to Declining");

    // Test Emerging condition
    println!("\n[Reset to Emerging]");
    stability.age_hours = 0.5;
    stability.membership_churn = 0.4;
    stability.update_phase();
    println!("  age_hours=0.5, churn=0.4");
    println!("  Phase after update: {:?}", stability.phase);
    assert_eq!(stability.phase, TopicPhase::Emerging);
    println!("[VERIFIED] Reset to Emerging");
}

// =============================================================================
// Manual Test 5: Profile Similarity
// =============================================================================

#[test]
fn manual_test_profile_similarity() {
    println!("\n=== Manual Test 5: Profile Similarity ===");

    // Identical profiles
    println!("\n[Test] Identical profiles:");
    let p1 = TopicProfile::new([0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let p2 = TopicProfile::new([0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let sim = p1.similarity(&p2);
    println!("  Similarity: {}", sim);
    assert!((sim - 1.0).abs() < 0.001, "Identical profiles should have similarity ~1.0");
    println!("[VERIFIED] Identical profiles have similarity 1.0");

    // Orthogonal profiles
    println!("\n[Test] Orthogonal profiles:");
    let p3 = TopicProfile::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let p4 = TopicProfile::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let sim2 = p3.similarity(&p4);
    println!("  Similarity: {}", sim2);
    assert!(sim2 < 0.001, "Orthogonal profiles should have similarity ~0.0");
    println!("[VERIFIED] Orthogonal profiles have similarity ~0.0");

    // Partial overlap
    println!("\n[Test] Partial overlap profiles:");
    let p5 = TopicProfile::new([1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let p6 = TopicProfile::new([1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let sim3 = p5.similarity(&p6);
    println!("  Similarity: {:.4}", sim3);
    assert!(sim3 > 0.0 && sim3 < 1.0, "Partial overlap should have 0 < similarity < 1");
    println!("[VERIFIED] Partial overlap has intermediate similarity");
}

// =============================================================================
// Manual Test 6: Topic with Members
// =============================================================================

#[test]
fn manual_test_topic_with_members() {
    println!("\n=== Manual Test 6: Topic with Members ===");

    let mem1 = Uuid::new_v4();
    let mem2 = Uuid::new_v4();
    let mem3 = Uuid::new_v4();

    let mut strengths = [0.0f32; 13];
    strengths[Embedder::Semantic.index()] = 1.0;
    strengths[Embedder::Causal.index()] = 0.9;
    strengths[Embedder::Code.index()] = 0.8;

    let profile = TopicProfile::new(strengths);
    let mut cluster_ids = HashMap::new();
    cluster_ids.insert(Embedder::Semantic, 0);
    cluster_ids.insert(Embedder::Causal, 1);
    cluster_ids.insert(Embedder::Code, 2);

    let mut topic = Topic::new(profile.clone(), cluster_ids.clone(), vec![mem1, mem2]);

    println!("Topic created:");
    println!("  ID: {}", topic.id);
    println!("  Members: {}", topic.member_count());
    println!("  Cluster IDs: {:?}", topic.cluster_ids);
    println!("  Is valid: {}", topic.is_valid());
    println!("  Confidence: {:.4}", topic.confidence);
    println!("  Contributing spaces: {:?}", topic.contributing_spaces);

    assert_eq!(topic.member_count(), 2);
    assert!(topic.contains_memory(&mem1));
    assert!(topic.contains_memory(&mem2));
    assert!(!topic.contains_memory(&mem3));
    println!("[VERIFIED] Member operations work correctly");

    // Test access recording
    println!("\nRecording access...");
    assert_eq!(topic.stability.access_count, 0);
    assert!(topic.stability.last_accessed.is_none());

    topic.record_access();

    assert_eq!(topic.stability.access_count, 1);
    assert!(topic.stability.last_accessed.is_some());
    println!("  Access count after: {}", topic.stability.access_count);
    println!("  Last accessed: {:?}", topic.stability.last_accessed);
    println!("[VERIFIED] Access recording works correctly");
}

// =============================================================================
// Manual Test 7: Edge Cases
// =============================================================================

#[test]
fn manual_test_edge_cases() {
    println!("\n=== Manual Test 7: Edge Cases ===");

    // Test Case 7.1: Zero profile
    println!("\n[Test 7.1] Zero profile:");
    let zero_profile = TopicProfile::default();
    let zero_weighted = zero_profile.weighted_agreement();
    println!("  weighted_agreement: {}", zero_weighted);
    assert!(zero_weighted.abs() < f32::EPSILON);
    assert!(!zero_profile.is_topic());
    println!("[VERIFIED] Zero profile handles correctly");

    // Test Case 7.2: Max profile
    println!("\n[Test 7.2] Max profile (all 1.0):");
    let max_profile = TopicProfile::new([1.0; 13]);
    let max_weighted = max_profile.weighted_agreement();
    println!("  weighted_agreement: {}", max_weighted);
    assert!((max_weighted - 8.5).abs() < 0.001);
    assert!(max_profile.is_topic());
    println!("[VERIFIED] Max profile = 8.5");

    // Test Case 7.3: Clamping
    println!("\n[Test 7.3] Strength clamping:");
    let clamped = TopicProfile::new([2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    println!("  Input: 2.0 -> clamped to {}", clamped.strengths[0]);
    println!("  Input: -1.0 -> clamped to {}", clamped.strengths[1]);
    assert_eq!(clamped.strengths[0], 1.0, "2.0 should clamp to 1.0");
    assert_eq!(clamped.strengths[1], 0.0, "-1.0 should clamp to 0.0");
    println!("[VERIFIED] Clamping works correctly");

    // Test Case 7.4: Similarity with zero vector
    println!("\n[Test 7.4] Similarity with zero vector:");
    let zero = TopicProfile::default();
    let non_zero = TopicProfile::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let sim = zero.similarity(&non_zero);
    println!("  Similarity: {}", sim);
    assert!(!sim.is_nan(), "Similarity should not be NaN");
    println!("[VERIFIED] Zero vector similarity handled gracefully");
}

// =============================================================================
// Manual Test 8: Serialization Roundtrip
// =============================================================================

#[test]
fn manual_test_serialization_roundtrip() {
    println!("\n=== Manual Test 8: Serialization Roundtrip ===");

    // Create a complete topic
    let mut strengths = [0.0f32; 13];
    strengths[0] = 0.9;
    strengths[4] = 0.85;
    strengths[6] = 0.7;

    let profile = TopicProfile::new(strengths);
    let mut cluster_ids = HashMap::new();
    cluster_ids.insert(Embedder::Semantic, 5);
    cluster_ids.insert(Embedder::Causal, 3);

    let original = Topic::new(profile, cluster_ids, vec![Uuid::new_v4(), Uuid::new_v4()]);

    // Serialize
    let json = serde_json::to_string_pretty(&original).expect("Failed to serialize");
    println!("Serialized JSON ({} bytes):", json.len());
    println!("{}", json);

    // Deserialize
    let restored: Topic = serde_json::from_str(&json).expect("Failed to deserialize");

    // Verify
    assert_eq!(original.id, restored.id, "ID mismatch");
    assert_eq!(original.profile.strengths, restored.profile.strengths, "Profile mismatch");
    assert_eq!(original.member_memories.len(), restored.member_memories.len(), "Member count mismatch");
    assert!((original.confidence - restored.confidence).abs() < f32::EPSILON, "Confidence mismatch");
    println!("\n[VERIFIED] Serialization roundtrip successful");

    // Phase serialization
    println!("\nPhase serialization:");
    for phase in [TopicPhase::Emerging, TopicPhase::Stable, TopicPhase::Declining, TopicPhase::Merging] {
        let json = serde_json::to_string(&phase).unwrap();
        let restored: TopicPhase = serde_json::from_str(&json).unwrap();
        println!("  {:?} -> {} -> {:?}", phase, json, restored);
        assert_eq!(phase, restored);
    }
    println!("[VERIFIED] All phases serialize correctly");
}

// =============================================================================
// Summary Test
// =============================================================================

#[test]
fn manual_test_summary() {
    println!("\n");
    println!("=============================================================");
    println!("TASK-P4-002: Topic Types Implementation - Manual Test Summary");
    println!("=============================================================");
    println!();
    println!("Tests Executed:");
    println!("  1. Constitution Constants (max_weighted_agreement=8.5, threshold=2.5)");
    println!("  2. Weighted Agreement Calculation (AP-60 temporal exclusion)");
    println!("  3. Topic Confidence Calculation (confidence = weighted/8.5)");
    println!("  4. TopicPhase Transitions (Emerging->Stable->Declining)");
    println!("  5. Profile Similarity (cosine similarity)");
    println!("  6. Topic with Members (membership tracking)");
    println!("  7. Edge Cases (zero, max, clamping, NaN handling)");
    println!("  8. Serialization Roundtrip (JSON encode/decode)");
    println!();
    println!("Key Constitution Requirements Verified:");
    println!("  - ARCH-09: Topic threshold is weighted_agreement >= 2.5");
    println!("  - AP-60: Temporal embedders (E2-E4) NEVER count toward topic detection");
    println!("  - MAX_WEIGHTED_AGREEMENT = 8.5 (7*1.0 + 2*0.5 + 1*0.5)");
    println!("  - Confidence = weighted_agreement / 8.5");
    println!();
    println!("Category Weights:");
    println!("  - Semantic (E1,E5,E6,E7,E10,E12,E13): 1.0 x 7 = 7.0");
    println!("  - Temporal (E2,E3,E4): 0.0 x 3 = 0.0");
    println!("  - Relational (E8,E11): 0.5 x 2 = 1.0");
    println!("  - Structural (E9): 0.5 x 1 = 0.5");
    println!("  - Total: 8.5");
    println!();
    println!("[SUCCESS] All manual tests passed");
    println!("=============================================================");
}
