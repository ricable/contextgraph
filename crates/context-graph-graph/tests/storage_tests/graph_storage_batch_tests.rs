//! GraphStorage batch operations and advanced tests.
//!
//! M04-T13: GraphStorage implementation - Batch, Iteration, Persistence

use context_graph_graph::storage::{
    EntailmentCone, GraphStorage, LegacyGraphEdge, NodeId, PoincarePoint,
};

#[test]
fn test_graph_storage_batch_operations() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_batch.db");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open GraphStorage");

    println!("BEFORE: Testing batch operations");

    // Create batch
    let mut batch = storage.new_batch();

    // Add multiple hyperbolic points
    for i in 0..10 {
        let mut point = PoincarePoint::origin();
        point.coords[0] = i as f32 * 0.1;
        storage
            .batch_put_hyperbolic(&mut batch, i, &point)
            .expect("Batch put failed");
    }

    // Add multiple cones
    for i in 10..15 {
        let mut cone = EntailmentCone::default_at_origin();
        cone.depth = i as u32;
        storage
            .batch_put_cone(&mut batch, i, &cone)
            .expect("Batch put failed");
    }

    // Add edges
    let edges = vec![
        LegacyGraphEdge {
            target: 1,
            edge_type: 0,
        },
        LegacyGraphEdge {
            target: 2,
            edge_type: 1,
        },
    ];
    storage
        .batch_put_adjacency(&mut batch, 100, &edges)
        .expect("Batch put failed");

    println!("BATCH PREPARED: 10 points, 5 cones, 1 adjacency list");

    // Write batch atomically
    storage.write_batch(batch).expect("Batch write failed");

    println!("BATCH WRITTEN: Atomically");

    // Verify all data
    let count = storage.hyperbolic_count().expect("Count failed");
    assert_eq!(count, 10);

    let cone_count = storage.cone_count().expect("Count failed");
    assert_eq!(cone_count, 5);

    let adj_count = storage.adjacency_count().expect("Count failed");
    assert_eq!(adj_count, 1);

    println!("VERIFIED: All batch data persisted");
    println!("AFTER: Batch operations complete");
}

#[test]
fn test_graph_storage_iteration() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_iteration.db");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open GraphStorage");

    println!("BEFORE: Testing iteration");

    // Insert test data
    for i in 0..5 {
        let mut point = PoincarePoint::origin();
        point.coords[0] = i as f32;
        storage.put_hyperbolic(i, &point).expect("PUT failed");
    }

    // Iterate and collect
    let mut collected: Vec<(NodeId, PoincarePoint)> = Vec::new();
    for result in storage.iter_hyperbolic().expect("Iter failed") {
        collected.push(result.expect("Iter item failed"));
    }

    assert_eq!(collected.len(), 5);
    println!("ITERATED: Collected {} hyperbolic points", collected.len());

    println!("AFTER: Iteration complete");
}

#[test]
fn test_graph_storage_reopen_preserves_data() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_reopen_graph_storage.db");

    println!("BEFORE: Testing data persistence across reopen");

    // First open: write data
    {
        let storage = GraphStorage::open_default(&db_path).expect("Failed to open");

        let mut point = PoincarePoint::origin();
        point.coords[0] = 0.42;
        storage.put_hyperbolic(1, &point).expect("PUT failed");

        let cone = EntailmentCone::default_at_origin();
        storage.put_cone(2, &cone).expect("PUT failed");

        storage
            .put_adjacency(
                3,
                &[LegacyGraphEdge {
                    target: 4,
                    edge_type: 5,
                }],
            )
            .expect("PUT failed");

        println!("FIRST OPEN: Wrote point, cone, edges");
    }

    // Second open: verify data
    {
        let storage = GraphStorage::open_default(&db_path).expect("Failed to reopen");

        let point = storage.get_hyperbolic(1).expect("GET failed").unwrap();
        assert!((point.coords[0] - 0.42).abs() < 0.0001);

        let cone = storage.get_cone(2).expect("GET failed");
        assert!(cone.is_some());

        let edges = storage.get_adjacency(3).expect("GET failed");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, 4);

        println!("SECOND OPEN: All data persisted");
    }

    println!("AFTER: Data persistence verified");
}

#[test]
fn test_graph_storage_clone_is_cheap() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_clone.db");

    let storage = GraphStorage::open_default(&db_path).expect("Failed to open");

    println!("BEFORE: Testing cheap clone via Arc<DB>");

    // Write via original
    let point = PoincarePoint::origin();
    storage.put_hyperbolic(1, &point).expect("PUT failed");

    // Clone
    let storage2 = storage.clone();

    // Read via clone
    let retrieved = storage2.get_hyperbolic(1).expect("GET failed");
    assert!(retrieved.is_some());

    // Write via clone
    let point2 = PoincarePoint::origin();
    storage2.put_hyperbolic(2, &point2).expect("PUT failed");

    // Read via original
    let retrieved2 = storage.get_hyperbolic(2).expect("GET failed");
    assert!(retrieved2.is_some());

    println!("AFTER: Clone shares same underlying DB");
}
