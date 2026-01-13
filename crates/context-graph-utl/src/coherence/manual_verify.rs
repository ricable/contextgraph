//! Manual verification module for three-component coherence formula.
//! Run with: cargo test -p context-graph-utl manual_verify -- --nocapture

#[cfg(test)]
mod manual_verification {
    use crate::coherence::cluster_fit::ClusterContext;
    use crate::coherence::{CoherenceTracker, GraphContext};
    use crate::config::CoherenceConfig;

    /// Test Case 1: Perfect coherence scenario
    ///
    /// - Connectivity: 1.0 (vertex identical to neighbors)
    /// - ClusterFit: ~1.0 (vertex in tight same_cluster, far from nearest_cluster)
    /// - Consistency: ~1.0 (stable history with similar embeddings)
    ///
    /// Expected: ΔC ≈ 1.0
    #[test]
    fn manual_test_perfect_coherence() {
        println!("\n=== MANUAL TEST 1: Perfect Coherence ===");
        
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);
        
        // Build stable history with identical embeddings
        let stable_embedding = vec![1.0, 0.0, 0.0, 0.0];
        for _ in 0..10 {
            tracker.update(&stable_embedding);
        }
        
        // Vertex = stable embedding
        let vertex = vec![1.0, 0.0, 0.0, 0.0];
        
        // Same cluster: identical to vertex
        let same_cluster = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.99, 0.01, 0.0, 0.0], // Very close
        ];
        
        // Nearest cluster: opposite direction
        let nearest_cluster = vec![
            vec![-1.0, 0.0, 0.0, 0.0],
            vec![-0.9, 0.1, 0.0, 0.0],
        ];
        
        let cluster_context = ClusterContext::new(same_cluster.clone(), nearest_cluster);
        
        // Connectivity = 1.0 (perfect match with neighbors)
        let connectivity = 1.0f32;
        
        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);
        
        println!("  Vertex: {:?}", vertex);
        println!("  Connectivity: {:.4}", connectivity);
        println!("  Final Coherence: {:.4}", coherence);
        
        // ΔC = 0.4×1.0 + 0.4×~1.0 + 0.2×~1.0 ≈ 1.0
        assert!(coherence > 0.9, "Expected high coherence > 0.9, got {}", coherence);
        println!("  ✓ PASSED: coherence = {:.4} > 0.9", coherence);
    }

    /// Test Case 2: Zero coherence scenario
    ///
    /// - Connectivity: 0.0 (vertex orthogonal to neighbors)
    /// - ClusterFit: ~0.0 (vertex closer to nearest_cluster than same_cluster)
    /// - Consistency: ~0.0 (chaotic history)
    ///
    /// Expected: ΔC ≈ 0.0
    #[test]
    fn manual_test_zero_coherence() {
        println!("\n=== MANUAL TEST 2: Zero Coherence ===");
        
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);
        
        // Build chaotic history with random embeddings
        let chaotic_embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![-1.0, 0.0, 0.0, 0.0],
        ];
        for emb in &chaotic_embeddings {
            tracker.update(emb);
        }
        
        // Vertex is far from same_cluster, close to nearest_cluster
        let vertex = vec![0.9, 0.1, 0.0, 0.0];
        
        // Same cluster: far from vertex
        let same_cluster = vec![
            vec![0.0, 0.0, 0.9, 0.1],
            vec![0.0, 0.0, 0.8, 0.2],
        ];
        
        // Nearest cluster: close to vertex (misclassified point)
        let nearest_cluster = vec![
            vec![0.85, 0.15, 0.0, 0.0],
            vec![0.88, 0.12, 0.0, 0.0],
        ];
        
        let cluster_context = ClusterContext::new(same_cluster, nearest_cluster);
        
        // Connectivity = 0.0 (no connection to neighbors)
        let connectivity = 0.0f32;
        
        let coherence = tracker.compute_coherence(&vertex, connectivity, &cluster_context);
        
        println!("  Vertex: {:?}", vertex);
        println!("  Connectivity: {:.4}", connectivity);
        println!("  Final Coherence: {:.4}", coherence);
        
        // ΔC = 0.4×0.0 + 0.4×~0.0 + 0.2×~0.0 ≈ 0.0
        assert!(coherence < 0.5, "Expected low coherence < 0.5, got {}", coherence);
        println!("  ✓ PASSED: coherence = {:.4} < 0.5", coherence);
    }

    /// Test Case 3: Verify formula with explicit components
    /// Constitution line 166: ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
    #[test]
    fn manual_test_formula_verification() {
        println!("\n=== MANUAL TEST 3: Formula Verification ===");
        
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);
        
        // Build moderate history
        let embedding = vec![0.5, 0.5, 0.0, 0.0];
        for _ in 0..5 {
            tracker.update(&embedding);
        }
        
        let vertex = vec![0.5, 0.5, 0.0, 0.0];
        let same_cluster = vec![
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.6, 0.4, 0.0, 0.0],
        ];
        let nearest_cluster = vec![
            vec![0.0, 0.0, 0.5, 0.5],
        ];
        
        let cluster_context = ClusterContext::new(same_cluster, nearest_cluster);
        let graph_context = GraphContext::new(
            vertex.clone(),
            vec![vec![0.5, 0.5, 0.0, 0.0], vec![0.55, 0.45, 0.0, 0.0]],
        );
        
        // Use compute_coherence_full to get all components
        let result = tracker.compute_coherence_full(&vertex, &graph_context, &cluster_context);
        
        println!("  Connectivity: {:.4}", result.connectivity);
        println!("  ClusterFit: {:.4}", result.cluster_fit);
        println!("  Consistency: {:.4}", result.consistency);
        println!("  Final Score: {:.4}", result.score);
        
        // Verify formula: ΔC = 0.4×C + 0.4×CF + 0.2×Con
        let expected = 0.4 * result.connectivity + 0.4 * result.cluster_fit + 0.2 * result.consistency;
        let diff = (result.score - expected).abs();
        
        println!("  Expected (formula): {:.4}", expected);
        println!("  Difference: {:.6}", diff);
        
        assert!(diff < 0.001, "Formula mismatch: expected {}, got {}", expected, result.score);
        println!("  ✓ PASSED: Formula verified within tolerance");
        
        // Verify all components in [0, 1]
        assert!((0.0..=1.0).contains(&result.connectivity), "Connectivity out of range");
        assert!((0.0..=1.0).contains(&result.cluster_fit), "ClusterFit out of range");
        assert!((0.0..=1.0).contains(&result.consistency), "Consistency out of range");
        assert!((0.0..=1.0).contains(&result.score), "Score out of range");
        println!("  ✓ PASSED: All components in [0, 1]");
    }

    /// Test Case 4: Weight adjustment verification
    #[test]
    fn manual_test_weight_adjustment() {
        println!("\n=== MANUAL TEST 4: Weight Adjustment ===");
        
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);
        
        // Build history
        for i in 0..5 {
            tracker.update(&[0.5 + i as f32 * 0.01, 0.5 - i as f32 * 0.01, 0.0, 0.0]);
        }
        
        let vertex = vec![0.55, 0.45, 0.0, 0.0];
        let cluster_context = ClusterContext::new(
            vec![vec![0.5, 0.5, 0.0, 0.0], vec![0.55, 0.45, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 0.5, 0.5]],
        );
        let connectivity = 0.8;
        
        // Default weights: 0.4, 0.4, 0.2
        let coherence_default = tracker.compute_coherence(&vertex, connectivity, &cluster_context);
        println!("  Default weights (0.4, 0.4, 0.2): {:.4}", coherence_default);
        
        // Change to emphasize connectivity
        tracker.set_weights(0.8, 0.1, 0.1);
        let coherence_connectivity = tracker.compute_coherence(&vertex, connectivity, &cluster_context);
        println!("  Connectivity emphasis (0.8, 0.1, 0.1): {:.4}", coherence_connectivity);
        
        // Change to emphasize cluster fit
        tracker.set_weights(0.1, 0.8, 0.1);
        let coherence_clusterfit = tracker.compute_coherence(&vertex, connectivity, &cluster_context);
        println!("  ClusterFit emphasis (0.1, 0.8, 0.1): {:.4}", coherence_clusterfit);
        
        // Verify different weights produce different results
        assert!((coherence_default - coherence_connectivity).abs() > 0.01 ||
                (coherence_default - coherence_clusterfit).abs() > 0.01,
                "Weight changes should affect output");
        println!("  ✓ PASSED: Weight adjustment affects coherence");
    }

    /// Test Case 5: NaN/Inf handling (AP-10 compliance)
    #[test]
    fn manual_test_nan_inf_handling() {
        println!("\n=== MANUAL TEST 5: NaN/Inf Handling (AP-10) ===");
        
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);
        
        for _ in 0..3 {
            tracker.update(&[0.5, 0.5, 0.0, 0.0]);
        }
        
        let vertex = vec![0.5, 0.5, 0.0, 0.0];
        let cluster_context = ClusterContext::new(
            vec![vec![0.5, 0.5, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 0.5, 0.5]],
        );
        
        // Test NaN connectivity
        let coherence_nan = tracker.compute_coherence(&vertex, f32::NAN, &cluster_context);
        assert!(!coherence_nan.is_nan(), "Output should not be NaN");
        assert!((0.0..=1.0).contains(&coherence_nan), "Output should be in [0, 1]");
        println!("  NaN connectivity -> {:.4} (using fallback 0.5)", coherence_nan);
        
        // Test Inf connectivity
        let coherence_inf = tracker.compute_coherence(&vertex, f32::INFINITY, &cluster_context);
        assert!(!coherence_inf.is_infinite(), "Output should not be Inf");
        assert!((0.0..=1.0).contains(&coherence_inf), "Output should be in [0, 1]");
        println!("  Inf connectivity -> {:.4} (using fallback 0.5)", coherence_inf);
        
        println!("  ✓ PASSED: AP-10 compliance verified");
    }
}
