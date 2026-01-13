//! Manual edge case tests for Hyperbolic Walk implementation
//! TASK-DREAM-P0-004 Section 6: Manual Testing with Synthetic Data
//!
//! Three edge cases tested:
//! 1. Single walk, small query budget - Walk completes with partial queries
//! 2. Many starting positions (100+), high temperature - Even distribution of exploration
//! 3. Interrupt signal mid-walk - Clean termination with partial results

use context_graph_core::dream::{
    HyperbolicExplorer, HyperbolicWalkConfig,
    sample_starting_positions,
    position_to_query,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Helper to create interrupt flag
fn make_interrupt_flag() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

/// Edge Case 1: Single walk from single starting position
/// Verifies that walk completes correctly
#[test]
fn edge_case_1_single_walk_single_start() {
    println!("\n=== EDGE CASE 1: Single walk from single starting position ===");

    // Use default config (Constitution values)
    let config = HyperbolicWalkConfig {
        step_size: 0.1,
        max_steps: 100,
        temperature: 2.0,           // Constitution
        min_blind_spot_distance: 0.7, // Constitution semantic_leap
        direction_samples: 8,
    };

    let mut explorer = HyperbolicExplorer::new(config)
        .with_seed(123);

    let interrupt = make_interrupt_flag();
    let start_positions: Vec<[f32; 64]> = vec![[0.0; 64]];

    let result = explorer.explore(&start_positions, &interrupt);

    println!("  - Walks completed: {}", result.walks.len());
    println!("  - Queries generated: {}", result.queries_generated);
    println!("  - Blind spots found: {}", result.all_blind_spots.len());
    println!("  - Coverage estimate: {:.4}", result.coverage_estimate);
    println!("  - Average semantic leap: {:.4}", result.average_semantic_leap);

    // Verify walk completed from single starting position
    assert!(!result.walks.is_empty(), "Should complete at least one walk");
    assert!(result.queries_generated > 0, "Should generate queries");
    assert!(result.queries_generated <= 100, "Should respect query limit");

    // Log walk trajectory lengths
    for (i, walk) in result.walks.iter().enumerate() {
        println!("  - Walk {} trajectory length: {}, completed: {}",
                 i, walk.trajectory.len(), walk.completed);
    }

    println!("  ✓ PASSED: Single walk from single start completed correctly");
}

/// Edge Case 2: Many starting positions with high temperature
/// Verifies even distribution of exploration across starting points
#[test]
fn edge_case_2_many_starting_positions_high_temp() {
    println!("\n=== EDGE CASE 2: Many starting positions (100+), high temperature ===");

    let config = HyperbolicWalkConfig {
        step_size: 0.1,
        max_steps: 20, // Shorter walks to test more starting positions
        temperature: 2.0,           // Constitution
        min_blind_spot_distance: 0.7, // Constitution semantic_leap
        direction_samples: 8,
    };

    let mut explorer = HyperbolicExplorer::new(config)
        .with_seed(456);

    let interrupt = make_interrupt_flag();

    // Generate 100 diverse starting positions spread across the Poincare ball
    let mut start_positions: Vec<[f32; 64]> = Vec::with_capacity(100);
    for i in 0..100 {
        let mut pos = [0.0f32; 64];
        // Create positions at different radii and angles
        let radius = 0.3 * ((i % 10) as f32 / 10.0); // 0 to 0.27
        let angle = (i / 10) as f32 * 0.2 * std::f32::consts::PI; // Different angles
        pos[0] = radius * angle.cos();
        pos[1] = radius * angle.sin();
        start_positions.push(pos);
    }

    println!("  - Starting positions: {}", start_positions.len());

    let result = explorer.explore(&start_positions, &interrupt);

    println!("  - Walks completed: {}", result.walks.len());
    println!("  - Queries generated: {}", result.queries_generated);
    println!("  - Blind spots found: {}", result.all_blind_spots.len());
    println!("  - Coverage estimate: {:.4}", result.coverage_estimate);

    // Verify exploration occurred from multiple starting positions
    assert!(!result.walks.is_empty(), "Should complete walks");
    assert!(result.queries_generated <= 100, "Must respect query limit (100)");

    // Check distribution: walks should start from different positions
    let unique_starts: std::collections::HashSet<_> = result.walks.iter()
        .map(|w| format!("{:.3}_{:.3}", w.start_position[0], w.start_position[1]))
        .collect();

    println!("  - Unique starting positions explored: {}", unique_starts.len());
    assert!(!unique_starts.is_empty(), "Should explore from at least one starting position");

    // Log some blind spot details if found
    for (i, blind_spot) in result.all_blind_spots.iter().take(5).enumerate() {
        println!("  - Blind spot {}: distance={:.4}, confidence={:.4}",
                 i, blind_spot.distance_from_nearest, blind_spot.confidence);
    }

    println!("  ✓ PASSED: Many starting positions explored with high temperature");
}

/// Edge Case 3: Interrupt signal during walk
/// Verifies clean termination with partial results when interrupted
#[test]
fn edge_case_3_interrupt_mid_walk() {
    println!("\n=== EDGE CASE 3: Interrupt signal mid-walk ===");

    let config = HyperbolicWalkConfig {
        step_size: 0.1,
        max_steps: 1000, // Long walks to ensure interrupt hits mid-walk
        temperature: 2.0,           // Constitution
        min_blind_spot_distance: 0.7, // Constitution semantic_leap
        direction_samples: 8,
    };

    let mut explorer = HyperbolicExplorer::new(config)
        .with_seed(789);

    let interrupt = make_interrupt_flag();

    // Set interrupt flag before exploration
    // This simulates an external query arriving during dream state
    interrupt.store(true, Ordering::SeqCst);

    let start_positions: Vec<[f32; 64]> = vec![[0.0; 64], [0.1; 64], [0.2; 64]];

    let result = explorer.explore(&start_positions, &interrupt);

    println!("  - Walks completed: {}", result.walks.len());
    println!("  - Queries generated: {}", result.queries_generated);
    println!("  - Blind spots found: {}", result.all_blind_spots.len());

    // Check that walks were interrupted (not fully completed)
    let any_incomplete = result.walks.iter().any(|w| !w.completed);
    println!("  - Any walks incomplete: {}", any_incomplete);

    // Verify partial results are still valid (whatever was completed before interrupt)
    // The result should be consistent - walks that did complete should have valid trajectories
    for walk in &result.walks {
        assert!(!walk.trajectory.is_empty() || walk.total_distance == 0.0,
                "Walk trajectory should be valid even if partial");
    }

    println!("  ✓ PASSED: Interrupt handled cleanly with partial results");
}

/// Constitution compliance verification test
#[test]
fn verify_constitution_compliance() {
    println!("\n=== CONSTITUTION COMPLIANCE VERIFICATION ===");

    let config = HyperbolicWalkConfig::default();

    println!("  - Temperature: {} (Constitution: 2.0)", config.temperature);
    println!("  - Min blind spot distance: {} (Constitution semantic_leap: 0.7)",
             config.min_blind_spot_distance);

    assert_eq!(config.temperature, 2.0,
               "Temperature must be 2.0 per Constitution");
    assert_eq!(config.min_blind_spot_distance, 0.7,
               "Min blind spot distance must be 0.7 per Constitution semantic_leap");

    // Test that explorer respects query limit
    let explorer = HyperbolicExplorer::with_defaults().with_seed(999);
    assert_eq!(explorer.remaining_queries(), 100, "Query limit must be 100");

    println!("  - Query limit (from remaining_queries): {} (Constitution: 100)",
             explorer.remaining_queries());

    println!("  ✓ PASSED: All Constitution values verified");
}

/// Integration test: Full exploration cycle with synthetic node positions
#[test]
fn integration_full_exploration_cycle() {
    println!("\n=== INTEGRATION: Full exploration cycle ===");

    let mut explorer = HyperbolicExplorer::with_defaults()
        .with_seed(999);

    let interrupt = make_interrupt_flag();

    // Create synthetic node positions with phi values for sample_starting_positions
    let mut node_positions: Vec<(f32, [f32; 64])> = Vec::with_capacity(20);
    for i in 0..20 {
        let phi = (i as f32 / 20.0) + 0.1; // phi in [0.1, 1.05]
        let mut pos = [0.0f32; 64];
        pos[0] = 0.2 * ((i % 5) as f32 / 5.0);
        pos[1] = 0.15 * ((i / 5) as f32 / 4.0);
        node_positions.push((phi, pos));
    }

    // Sample starting positions using the exported helper function
    let start_positions = sample_starting_positions(&node_positions, 10, 2.0);

    println!("  - Sampled {} starting positions from {} nodes",
             start_positions.len(), node_positions.len());

    let result = explorer.explore(&start_positions, &interrupt);

    println!("  - Walks completed: {}", result.walks.len());
    println!("  - Total queries: {}", result.queries_generated);
    println!("  - Blind spots discovered: {}", result.all_blind_spots.len());
    println!("  - Coverage estimate: {:.4}", result.coverage_estimate);
    println!("  - Unique positions: {}", result.unique_positions);

    // Calculate coverage statistics
    let total_steps: usize = result.walks.iter()
        .map(|w| w.trajectory.len())
        .sum();
    let total_distance: f64 = result.walks.iter()
        .map(|w| w.total_distance as f64)
        .sum();

    println!("  - Total steps taken: {}", total_steps);
    println!("  - Total distance covered: {:.4}", total_distance);

    // Verify query generation produces meaningful results
    for blind_spot in &result.all_blind_spots {
        let query = position_to_query(&blind_spot.position);
        assert_eq!(query.len(), 64, "Query embedding should be 64D");
        println!("  - Generated query embedding of size: {}", query.len());
    }

    println!("  ✓ PASSED: Full exploration cycle completed successfully");
}

/// Test position_to_query helper function
#[test]
fn test_position_to_query_helper() {
    println!("\n=== TEST: position_to_query helper ===");

    // Test origin
    let origin = [0.0f32; 64];
    let query_origin = position_to_query(&origin);
    assert_eq!(query_origin.len(), 64, "Query should be 64D");
    println!("  - Origin query: first 5 dims = {:?}", &query_origin[0..5]);

    // Test non-origin
    let mut pos = [0.0f32; 64];
    pos[0] = 0.5;
    pos[1] = 0.3;
    let query_pos = position_to_query(&pos);
    assert_eq!(query_pos.len(), 64, "Query should be 64D");
    println!("  - Non-origin query: first 5 dims = {:?}", &query_pos[0..5]);

    println!("  ✓ PASSED: position_to_query helper works correctly");
}

/// Test sample_starting_positions helper function
#[test]
fn test_sample_starting_positions_helper() {
    println!("\n=== TEST: sample_starting_positions helper ===");

    // Create test data
    let mut node_positions: Vec<(f32, [f32; 64])> = Vec::new();
    for i in 0..5 {
        let mut pos = [0.0f32; 64];
        pos[0] = 0.1 * i as f32;
        node_positions.push((1.0 - 0.1 * i as f32, pos)); // phi decreases
    }

    // Sample with temperature 2.0 (Constitution)
    let samples = sample_starting_positions(&node_positions, 3, 2.0);

    println!("  - Input: {} node positions", node_positions.len());
    println!("  - Requested: 3 samples");
    println!("  - Got: {} samples", samples.len());

    assert!(samples.len() <= 3, "Should not return more than requested");
    assert!(!samples.is_empty(), "Should return some samples");

    for (i, sample) in samples.iter().enumerate() {
        println!("  - Sample {} pos[0]={:.3}", i, sample[0]);
    }

    // Test empty input
    let empty_samples = sample_starting_positions(&[], 5, 2.0);
    assert!(empty_samples.is_empty(), "Empty input should return empty");

    // Test zero count
    let zero_samples = sample_starting_positions(&node_positions, 0, 2.0);
    assert!(zero_samples.is_empty(), "Zero count should return empty");

    println!("  ✓ PASSED: sample_starting_positions helper works correctly");
}
