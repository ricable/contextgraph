//! Tests for SearchResultItem type and conversions.

use crate::index::search_result::SearchResultItem;

// ========== L2 to Similarity Conversion ==========

#[test]
fn test_l2_to_similarity_zero_distance() {
    // Zero L2 distance = identical vectors = similarity 1.0
    let item = SearchResultItem::from_l2(1, 0.0);
    assert!((item.similarity - 1.0).abs() < 1e-6);
}

#[test]
fn test_l2_to_similarity_max_distance() {
    // L2^2 = 2 for orthogonal normalized vectors -> similarity = 0
    let item = SearchResultItem::from_l2(1, 2.0);
    assert!((item.similarity - 0.0).abs() < 1e-6);
}

#[test]
fn test_l2_to_similarity_opposite() {
    // L2^2 = 4 for opposite normalized vectors -> similarity = -1
    let item = SearchResultItem::from_l2(1, 4.0);
    assert!((item.similarity - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_from_similarity_roundtrip() {
    let original_sim = 0.75;
    let item = SearchResultItem::from_similarity(42, original_sim);

    assert_eq!(item.id, 42);
    assert!((item.similarity - original_sim).abs() < 1e-6);

    // Verify distance conversion
    // d^2 = 2(1 - 0.75) = 0.5
    assert!((item.distance - 0.5).abs() < 1e-6);
}

// ========== SearchResultItem Equality ==========

#[test]
fn test_search_result_item_equality() {
    let a = SearchResultItem::from_l2(42, 0.5);
    let b = SearchResultItem::from_l2(42, 0.5);
    assert_eq!(a, b);

    let c = SearchResultItem::from_l2(42, 0.6);
    assert_ne!(a, c);
}

#[test]
fn test_search_result_item_clone() {
    let item = SearchResultItem::from_l2(99, 1.5);
    let cloned = item.clone();
    assert_eq!(item, cloned);
}
