//! Tests for batch entailment operations.

use crate::config::HyperbolicConfig;
use crate::entailment::query::entailment_check_batch;
use crate::error::GraphError;

use super::helpers::{
    create_test_cone, create_test_point, create_test_storage, store_cone, store_point,
};

// ========== entailment_check_batch Tests ==========

#[test]
fn test_batch_check_empty() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    let results = entailment_check_batch(&storage, &[], &config).expect("batch should succeed");
    assert!(results.is_empty());
}

#[test]
fn test_batch_check_single_pair() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Setup data
    let ancestor_cone = create_test_cone(0.0, 1.5, 0);
    store_cone(&storage, 1, &ancestor_cone);

    let descendant_point = create_test_point(0.3);
    store_point(&storage, 2, &descendant_point);

    let pairs = vec![(1, 2)];
    let results = entailment_check_batch(&storage, &pairs, &config).expect("batch should succeed");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].ancestor_id, 1);
    assert_eq!(results[0].descendant_id, 2);
    assert!(results[0].is_entailed);
    assert_eq!(results[0].score, 1.0);
}

#[test]
fn test_batch_check_multiple_pairs() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Ancestor cone at origin
    let ancestor_cone = create_test_cone(0.0, 1.5, 0);
    store_cone(&storage, 1, &ancestor_cone);

    // Multiple descendants
    for i in 2..=5 {
        let point = create_test_point(0.1 * i as f32);
        store_point(&storage, i, &point);
    }

    let pairs: Vec<(i64, i64)> = (2..=5).map(|i| (1, i)).collect();
    let results = entailment_check_batch(&storage, &pairs, &config).expect("batch should succeed");

    assert_eq!(results.len(), 4);
    for (idx, result) in results.iter().enumerate() {
        assert_eq!(result.ancestor_id, 1);
        assert_eq!(result.descendant_id, (idx + 2) as i64);
    }
}

#[test]
fn test_batch_check_caching() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Same ancestor, multiple descendants
    let ancestor_cone = create_test_cone(0.0, 1.5, 0);
    store_cone(&storage, 1, &ancestor_cone);

    for i in 2..=10 {
        let point = create_test_point(0.05 * i as f32);
        store_point(&storage, i, &point);
    }

    // All pairs share ancestor - should benefit from caching
    let pairs: Vec<(i64, i64)> = (2..=10).map(|i| (1, i)).collect();
    let results = entailment_check_batch(&storage, &pairs, &config).expect("batch should succeed");

    assert_eq!(results.len(), 9);
}

#[test]
fn test_batch_check_missing_data() {
    let (storage, _temp_dir) = create_test_storage();
    let config = HyperbolicConfig::default();

    // Only partial data
    let cone = create_test_cone(0.0, 1.5, 0);
    store_cone(&storage, 1, &cone);
    // No descendant point

    let pairs = vec![(1, 2)];
    let result = entailment_check_batch(&storage, &pairs, &config);

    assert!(matches!(result, Err(GraphError::MissingHyperbolicData(2))));
}
