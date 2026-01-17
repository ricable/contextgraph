//! Manual edge case testing for TASK-P2-002
//! This validates all boundary and edge case scenarios

use context_graph_core::embeddings::vector::{DenseVector, BinaryVector};
use context_graph_core::types::fingerprint::SparseVector;

fn main() {
    println!("=== TASK-P2-002 Manual Edge Case Testing ===\n");
    
    // =========================================================================
    // Edge Case 1: Zero Vector Cosine (AP-10 compliance)
    // =========================================================================
    println!("EC-001: Zero Vector Cosine Similarity");
    let zero = DenseVector::zeros(3);
    let non_zero = DenseVector::new(vec![1.0, 2.0, 3.0]);
    
    let result1 = zero.cosine_similarity(&non_zero);
    let result2 = non_zero.cosine_similarity(&zero);
    let result3 = zero.cosine_similarity(&zero);
    
    println!("  BEFORE: zero_vec = [0.0, 0.0, 0.0], non_zero = [1.0, 2.0, 3.0]");
    println!("  AFTER: cosine(zero, non_zero) = {}", result1);
    println!("         cosine(non_zero, zero) = {}", result2);
    println!("         cosine(zero, zero) = {}", result3);
    
    assert_eq!(result1, 0.0, "EC-001a FAILED: zero<->non_zero should be 0.0");
    assert_eq!(result2, 0.0, "EC-001b FAILED: non_zero<->zero should be 0.0");
    assert_eq!(result3, 0.0, "EC-001c FAILED: zero<->zero should be 0.0");
    println!("  RESULT: PASS - Zero vectors return 0.0 (not NaN)\n");
    
    // =========================================================================
    // Edge Case 2: Empty Vectors
    // =========================================================================
    println!("EC-002: Empty Vector Handling");
    let empty_dense = DenseVector::default();
    let empty_binary = BinaryVector::default();
    
    println!("  BEFORE: empty_dense.len() = {}", empty_dense.len());
    println!("          empty_binary.bit_len() = {}", empty_binary.bit_len());
    
    let empty_cosine = empty_dense.cosine_similarity(&empty_dense);
    let empty_hamming = empty_binary.hamming_distance(&empty_binary);
    
    println!("  AFTER: cosine(empty, empty) = {}", empty_cosine);
    println!("         hamming(empty, empty) = {}", empty_hamming);
    
    assert!(empty_dense.is_empty(), "EC-002a FAILED: DenseVector::default() should be empty");
    assert_eq!(empty_binary.bit_len(), 0, "EC-002b FAILED: BinaryVector::default() should have 0 bits");
    assert_eq!(empty_cosine, 0.0, "EC-002c FAILED: Empty cosine should be 0.0");
    assert_eq!(empty_hamming, 0, "EC-002d FAILED: Empty hamming should be 0");
    println!("  RESULT: PASS - Empty vectors handled gracefully\n");
    
    // =========================================================================
    // Edge Case 3: Out of Bounds Binary Access
    // =========================================================================
    println!("EC-003: Out of Bounds Binary Vector Access");
    let mut binary = BinaryVector::zeros(64);
    
    println!("  BEFORE: binary.bit_len() = 64");
    println!("          Attempting get_bit(100) and set_bit(100, true)");
    
    let oob_get = binary.get_bit(100);
    binary.set_bit(100, true); // Should be no-op
    let after_set = binary.get_bit(100);
    
    println!("  AFTER: get_bit(100) = {}", oob_get);
    println!("         After set_bit(100, true), get_bit(100) = {}", after_set);
    
    assert!(!oob_get, "EC-003a FAILED: Out of bounds get should return false");
    assert!(!after_set, "EC-003b FAILED: Out of bounds set should be no-op");
    println!("  RESULT: PASS - Out of bounds access handled safely\n");
    
    // =========================================================================
    // Edge Case 4: Disjoint Sparse Jaccard
    // =========================================================================
    println!("EC-004: Disjoint Sparse Vector Jaccard");
    let sparse_a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
    let sparse_b = SparseVector::new(vec![3, 4, 5], vec![1.0, 1.0, 1.0]).unwrap();
    
    println!("  BEFORE: sparse_a indices = [0, 1, 2]");
    println!("          sparse_b indices = [3, 4, 5]");
    
    let jaccard_disjoint = sparse_a.jaccard_similarity(&sparse_b);
    
    println!("  AFTER: jaccard(disjoint) = {}", jaccard_disjoint);
    
    assert_eq!(jaccard_disjoint, 0.0, "EC-004 FAILED: Disjoint sets should have Jaccard 0.0");
    println!("  RESULT: PASS - Disjoint sparse vectors have Jaccard 0.0\n");
    
    // =========================================================================
    // Edge Case 5: Empty Sparse Sparsity
    // =========================================================================
    println!("EC-005: Empty Sparse Vector Sparsity");
    let empty_sparse = SparseVector::empty();
    
    println!("  BEFORE: empty_sparse.nnz() = {}", empty_sparse.nnz());
    
    let sparsity = empty_sparse.sparsity();
    
    println!("  AFTER: empty_sparse.sparsity() = {}", sparsity);
    
    assert_eq!(sparsity, 1.0, "EC-005 FAILED: Empty sparse should be 100% sparse (1.0)");
    println!("  RESULT: PASS - Empty sparse vector has sparsity 1.0\n");
    
    // =========================================================================
    // Edge Case 6: Identical Vectors
    // =========================================================================
    println!("EC-006: Identical Vector Operations");
    let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
    let normalized = v.normalized();
    
    println!("  BEFORE: v = [1.0, 2.0, 3.0]");
    
    let cosine_identical = normalized.cosine_similarity(&normalized);
    let euclidean_identical = v.euclidean_distance(&v);
    let sparse_c = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
    let jaccard_identical = sparse_c.jaccard_similarity(&sparse_c);
    
    println!("  AFTER: cosine(normalized, normalized) = {}", cosine_identical);
    println!("         euclidean(v, v) = {}", euclidean_identical);
    println!("         jaccard(sparse_c, sparse_c) = {}", jaccard_identical);
    
    assert!((cosine_identical - 1.0).abs() < 1e-6, "EC-006a FAILED: Identical cosine should be 1.0");
    assert_eq!(euclidean_identical, 0.0, "EC-006b FAILED: Identical euclidean should be 0.0");
    assert!((jaccard_identical - 1.0).abs() < 1e-6, "EC-006c FAILED: Identical jaccard should be 1.0");
    println!("  RESULT: PASS - Identical vectors compute correctly\n");
    
    // =========================================================================
    // Edge Case 7: Euclidean Distance (3,4,5 triangle)
    // =========================================================================
    println!("EC-007: 3-4-5 Triangle Euclidean Distance");
    let origin = DenseVector::new(vec![0.0, 0.0]);
    let point = DenseVector::new(vec![3.0, 4.0]);
    
    println!("  BEFORE: origin = [0.0, 0.0], point = [3.0, 4.0]");
    
    let distance = origin.euclidean_distance(&point);
    
    println!("  AFTER: euclidean_distance = {}", distance);
    
    assert!((distance - 5.0).abs() < 1e-6, "EC-007 FAILED: sqrt(9+16)=5");
    println!("  RESULT: PASS - 3-4-5 triangle computes distance 5.0\n");
    
    // =========================================================================
    // Edge Case 8: Hamming Distance (0b011 vs 0b110)
    // =========================================================================
    println!("EC-008: Hamming Distance XOR");
    let mut h_a = BinaryVector::zeros(64);
    let mut h_b = BinaryVector::zeros(64);
    // a = 011 (bits 0, 1), b = 110 (bits 1, 2)
    h_a.set_bit(0, true);
    h_a.set_bit(1, true);
    h_b.set_bit(1, true);
    h_b.set_bit(2, true);
    
    println!("  BEFORE: h_a bits [0,1]=true, h_b bits [1,2]=true");
    
    let hamming = h_a.hamming_distance(&h_b);
    
    println!("  AFTER: hamming_distance = {}", hamming);
    println!("         XOR = 011 ^ 110 = 101 = 2 differing bits");
    
    assert_eq!(hamming, 2, "EC-008 FAILED: Hamming(011,110) should be 2");
    println!("  RESULT: PASS - Hamming distance = 2\n");
    
    // =========================================================================
    // Edge Case 9: Normalize Zero Vector
    // =========================================================================
    println!("EC-009: Normalize Zero Vector");
    let mut zero_for_norm = DenseVector::zeros(3);
    
    println!("  BEFORE: zero_for_norm = [0.0, 0.0, 0.0]");
    
    zero_for_norm.normalize(); // Should not panic
    
    println!("  AFTER: zero_for_norm = {:?}", zero_for_norm.data());
    
    assert_eq!(zero_for_norm.data(), &[0.0, 0.0, 0.0], "EC-009 FAILED: Normalizing zero should leave it zero");
    println!("  RESULT: PASS - Normalize zero vector is no-op\n");
    
    // =========================================================================
    // Edge Case 10: Opposite Vectors Cosine
    // =========================================================================
    println!("EC-010: Opposite Vectors Cosine Similarity");
    let pos = DenseVector::new(vec![1.0, 1.0]);
    let neg = DenseVector::new(vec![-1.0, -1.0]);
    
    println!("  BEFORE: pos = [1.0, 1.0], neg = [-1.0, -1.0]");
    
    let opposite_cosine = pos.cosine_similarity(&neg);
    
    println!("  AFTER: cosine(pos, neg) = {}", opposite_cosine);
    
    assert!((opposite_cosine - (-1.0)).abs() < 1e-6, "EC-010 FAILED: Opposite vectors should have cosine -1.0");
    println!("  RESULT: PASS - Opposite vectors have cosine -1.0\n");
    
    // =========================================================================
    // Summary
    // =========================================================================
    println!("=== ALL 10 EDGE CASES PASSED ===");
    println!("\nSummary:");
    println!("  EC-001: Zero Vector Cosine         -> PASS (AP-10 compliant)");
    println!("  EC-002: Empty Vector Handling      -> PASS");
    println!("  EC-003: Out of Bounds Binary       -> PASS");
    println!("  EC-004: Disjoint Sparse Jaccard    -> PASS");
    println!("  EC-005: Empty Sparse Sparsity      -> PASS");
    println!("  EC-006: Identical Vectors          -> PASS");
    println!("  EC-007: 3-4-5 Triangle Euclidean   -> PASS");
    println!("  EC-008: Hamming XOR                -> PASS");
    println!("  EC-009: Normalize Zero Vector      -> PASS");
    println!("  EC-010: Opposite Vectors Cosine    -> PASS");
}
