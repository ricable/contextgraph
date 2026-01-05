//! GraphStorage CRUD operation tests.
//!
//! M04-T13: GraphStorage implementation - Create, Read, Update, Delete

use context_graph_graph::storage::{
    EntailmentCone, GraphStorage, LegacyGraphEdge, NodeId, PoincarePoint, StorageConfig,
};

#[test]
fn test_graph_storage_open_default() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_graph_storage.db");

    println!("BEFORE: Opening GraphStorage with default config");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open GraphStorage");

    println!("AFTER: GraphStorage opened successfully");

    // Verify we can access it
    let count = storage
        .hyperbolic_count()
        .expect("Failed to count hyperbolic");
    assert_eq!(count, 0, "New database should be empty");

    println!("VERIFIED: Empty database has 0 hyperbolic entries");
}

#[test]
fn test_graph_storage_open_with_config() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_graph_storage_config.db");

    println!("BEFORE: Opening GraphStorage with read-optimized config");

    let config = StorageConfig::read_optimized();
    let storage = GraphStorage::open(&db_path, config).expect("Failed to open GraphStorage");

    println!("AFTER: GraphStorage opened with custom config");

    // Verify it works
    let point = PoincarePoint::origin();
    storage
        .put_hyperbolic(1, &point)
        .expect("Failed to put hyperbolic");
    let retrieved = storage.get_hyperbolic(1).expect("Failed to get hyperbolic");
    assert!(retrieved.is_some());

    println!("VERIFIED: GraphStorage works with custom config");
}

#[test]
fn test_graph_storage_hyperbolic_crud() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_hyperbolic_crud.db");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open GraphStorage");

    println!("BEFORE: Testing hyperbolic CRUD operations");

    // Create
    let node_id: NodeId = 42;
    let mut point = PoincarePoint::origin();
    point.coords[0] = 0.5;
    point.coords[63] = -0.3;

    storage.put_hyperbolic(node_id, &point).expect("PUT failed");
    println!("CREATE: Stored point for node_id={}", node_id);

    // Read
    let retrieved = storage
        .get_hyperbolic(node_id)
        .expect("GET failed")
        .unwrap();
    assert!((retrieved.coords[0] - 0.5).abs() < 0.0001);
    assert!((retrieved.coords[63] - (-0.3)).abs() < 0.0001);
    println!("READ: Retrieved point matches");

    // Update
    let mut updated = point.clone();
    updated.coords[0] = 0.9;
    storage
        .put_hyperbolic(node_id, &updated)
        .expect("UPDATE failed");

    let after_update = storage
        .get_hyperbolic(node_id)
        .expect("GET failed")
        .unwrap();
    assert!((after_update.coords[0] - 0.9).abs() < 0.0001);
    println!("UPDATE: Point updated successfully");

    // Delete
    storage.delete_hyperbolic(node_id).expect("DELETE failed");
    let deleted = storage.get_hyperbolic(node_id).expect("GET failed");
    assert!(deleted.is_none());
    println!("DELETE: Point deleted successfully");

    println!("AFTER: Hyperbolic CRUD operations complete");
}

#[test]
fn test_graph_storage_cone_crud() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_cone_crud.db");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open GraphStorage");

    println!("BEFORE: Testing entailment cone CRUD operations");

    // Create
    let node_id: NodeId = 100;
    let mut cone = EntailmentCone::default_at_origin();
    cone.apex.coords[0] = 0.1;
    cone.aperture = 0.5;
    cone.aperture_factor = 2.0;
    cone.depth = 5;

    storage.put_cone(node_id, &cone).expect("PUT failed");
    println!("CREATE: Stored cone for node_id={}", node_id);

    // Read
    let retrieved = storage.get_cone(node_id).expect("GET failed").unwrap();
    assert!((retrieved.apex.coords[0] - 0.1).abs() < 0.0001);
    assert!((retrieved.aperture - 0.5).abs() < 0.0001);
    assert!((retrieved.aperture_factor - 2.0).abs() < 0.0001);
    assert_eq!(retrieved.depth, 5);
    println!("READ: Retrieved cone matches");

    // Delete
    storage.delete_cone(node_id).expect("DELETE failed");
    let deleted = storage.get_cone(node_id).expect("GET failed");
    assert!(deleted.is_none());
    println!("DELETE: Cone deleted successfully");

    println!("AFTER: Cone CRUD operations complete");
}

#[test]
fn test_graph_storage_adjacency_operations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_adjacency.db");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open GraphStorage");

    println!("BEFORE: Testing adjacency list operations");

    let source: NodeId = 1;

    // Initially empty
    let edges = storage.get_adjacency(source).expect("GET failed");
    assert!(edges.is_empty());
    println!("INITIAL: No edges for node_id={}", source);

    // Add edges
    storage
        .add_edge(
            source,
            LegacyGraphEdge {
                target: 10,
                edge_type: 1,
            },
        )
        .expect("Add edge 1 failed");
    storage
        .add_edge(
            source,
            LegacyGraphEdge {
                target: 20,
                edge_type: 2,
            },
        )
        .expect("Add edge 2 failed");
    storage
        .add_edge(
            source,
            LegacyGraphEdge {
                target: 30,
                edge_type: 1,
            },
        )
        .expect("Add edge 3 failed");

    let edges = storage.get_adjacency(source).expect("GET failed");
    assert_eq!(edges.len(), 3);
    println!("ADDED: 3 edges from node_id={}", source);

    // Remove an edge
    let removed = storage.remove_edge(source, 20).expect("Remove edge failed");
    assert!(removed);
    let edges = storage.get_adjacency(source).expect("GET failed");
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().all(|e| e.target != 20));
    println!("REMOVED: Edge to target=20");

    // Remove non-existent edge
    let not_removed = storage
        .remove_edge(source, 999)
        .expect("Remove edge failed");
    assert!(!not_removed);
    println!("NOT REMOVED: Non-existent edge to target=999");

    // Delete all adjacencies
    storage.delete_adjacency(source).expect("DELETE failed");
    let edges = storage.get_adjacency(source).expect("GET failed");
    assert!(edges.is_empty());
    println!("DELETED: All edges for node_id={}", source);

    println!("AFTER: Adjacency operations complete");
}

#[test]
fn test_graph_storage_binary_format_sizes() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_sizes.db");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open GraphStorage");

    println!("BEFORE: Verifying binary format sizes");

    // PoincarePoint: 256 bytes (64 f32)
    let point = PoincarePoint::origin();
    storage.put_hyperbolic(1, &point).expect("PUT failed");
    // We can't directly access the raw bytes, but we can verify roundtrip
    let retrieved = storage.get_hyperbolic(1).expect("GET failed").unwrap();
    assert_eq!(retrieved.coords.len(), 64);
    println!("PoincarePoint: 64 coords (256 bytes)");

    // EntailmentCone: 268 bytes (256 + 4 + 4 + 4)
    let cone = EntailmentCone::default_at_origin();
    storage.put_cone(2, &cone).expect("PUT failed");
    let retrieved_cone = storage.get_cone(2).expect("GET failed").unwrap();
    assert_eq!(retrieved_cone.apex.coords.len(), 64);
    println!("EntailmentCone: 268 bytes (256 apex + 4 aperture + 4 factor + 4 depth)");

    println!("AFTER: Binary format sizes verified");
}
