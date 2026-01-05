//! Tests for column family constants.

use crate::storage::constants::*;

#[test]
fn test_cf_names() {
    assert_eq!(CF_ADJACENCY, "adjacency");
    assert_eq!(CF_HYPERBOLIC, "hyperbolic");
    assert_eq!(CF_CONES, "entailment_cones");
    assert_eq!(CF_FAISS_IDS, "faiss_ids");
    assert_eq!(CF_NODES, "nodes");
    assert_eq!(CF_METADATA, "metadata");
}

#[test]
fn test_all_column_families_count() {
    assert_eq!(ALL_COLUMN_FAMILIES.len(), 7); // M04-T15: added CF_EDGES
}

#[test]
fn test_all_column_families_contains_all() {
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_ADJACENCY));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_HYPERBOLIC));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_CONES));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_FAISS_IDS));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_NODES));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_METADATA));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_EDGES)); // M04-T15
}

#[test]
fn test_all_column_families_order() {
    // Order must match descriptor generation
    assert_eq!(ALL_COLUMN_FAMILIES[0], CF_ADJACENCY);
    assert_eq!(ALL_COLUMN_FAMILIES[1], CF_HYPERBOLIC);
    assert_eq!(ALL_COLUMN_FAMILIES[2], CF_CONES);
    assert_eq!(ALL_COLUMN_FAMILIES[3], CF_FAISS_IDS);
    assert_eq!(ALL_COLUMN_FAMILIES[4], CF_NODES);
    assert_eq!(ALL_COLUMN_FAMILIES[5], CF_METADATA);
    assert_eq!(ALL_COLUMN_FAMILIES[6], CF_EDGES); // M04-T15
}
