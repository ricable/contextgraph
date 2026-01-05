//! Tests for GraphEdge storage operations.

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::storage::edges::{Domain, EdgeType, GraphEdge};
    use crate::storage::storage_impl::core::GraphStorage;

    #[test]
    fn test_graph_edge_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edge_roundtrip.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = GraphEdge::new(42, source, target, EdgeType::Semantic, 0.75, Domain::Code);

        // Put edge
        storage.put_edge(&edge).unwrap();

        // Get edge back
        let retrieved = storage.get_edge(42).unwrap().expect("edge should exist");

        assert_eq!(retrieved.id, 42);
        assert_eq!(retrieved.source, source);
        assert_eq!(retrieved.target, target);
        assert_eq!(retrieved.edge_type, EdgeType::Semantic);
        assert!((retrieved.weight - 0.75).abs() < 0.0001);
        assert_eq!(retrieved.domain, Domain::Code);
    }

    #[test]
    fn test_graph_edge_not_found() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edge_not_found.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let result = storage.get_edge(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_graph_edge_delete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edge_delete.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let edge = GraphEdge::new(
            100,
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Causal,
            0.8,
            Domain::General,
        );

        storage.put_edge(&edge).unwrap();
        assert!(storage.get_edge(100).unwrap().is_some());

        storage.delete_edge(100).unwrap();
        assert!(storage.get_edge(100).unwrap().is_none());
    }

    #[test]
    fn test_graph_edges_bulk_operations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edges_bulk.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let edges: Vec<GraphEdge> = (0..10)
            .map(|i| {
                GraphEdge::new(
                    i,
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    EdgeType::Semantic,
                    0.5 + (i as f32 * 0.05),
                    Domain::General,
                )
            })
            .collect();

        // Put all edges
        storage.put_edges(&edges).unwrap();

        // Get subset of edges
        let edge_ids: Vec<i64> = vec![0, 3, 5, 9];
        let retrieved = storage.get_edges(&edge_ids).unwrap();

        assert_eq!(retrieved.len(), 4);
        assert_eq!(retrieved[0].0, 0);
        assert_eq!(retrieved[1].0, 3);
        assert_eq!(retrieved[2].0, 5);
        assert_eq!(retrieved[3].0, 9);

        // Verify edge count
        assert_eq!(storage.edge_count().unwrap(), 10);
    }

    #[test]
    fn test_graph_edges_bulk_delete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edges_bulk_delete.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let edges: Vec<GraphEdge> = (0..5)
            .map(|i| {
                GraphEdge::new(
                    i,
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    EdgeType::Temporal,
                    0.6,
                    Domain::Research,
                )
            })
            .collect();

        storage.put_edges(&edges).unwrap();
        assert_eq!(storage.edge_count().unwrap(), 5);

        // Delete some edges
        storage.delete_edges(&[1, 3]).unwrap();
        assert_eq!(storage.edge_count().unwrap(), 3);

        // Verify specific edges are gone
        assert!(storage.get_edge(1).unwrap().is_none());
        assert!(storage.get_edge(3).unwrap().is_none());

        // Verify others remain
        assert!(storage.get_edge(0).unwrap().is_some());
        assert!(storage.get_edge(2).unwrap().is_some());
        assert!(storage.get_edge(4).unwrap().is_some());
    }

    #[test]
    fn test_graph_edge_iteration() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edge_iter.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let edges: Vec<GraphEdge> = (100..103)
            .map(|i| {
                GraphEdge::new(
                    i,
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    EdgeType::Hierarchical,
                    0.9,
                    Domain::Code,
                )
            })
            .collect();

        storage.put_edges(&edges).unwrap();

        // Iterate and collect
        let mut collected: Vec<GraphEdge> =
            storage.iter_edges().unwrap().map(|r| r.unwrap()).collect();

        collected.sort_by_key(|e| e.id);
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].id, 100);
        assert_eq!(collected[1].id, 101);
        assert_eq!(collected[2].id, 102);
    }

    #[test]
    fn test_graph_edge_batch_put() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edge_batch.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let edge1 = GraphEdge::new(
            1,
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            0.5,
            Domain::General,
        );
        let edge2 = GraphEdge::new(
            2,
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Causal,
            0.7,
            Domain::General,
        );

        let mut batch = storage.new_batch();
        storage.batch_put_edge(&mut batch, &edge1).unwrap();
        storage.batch_put_edge(&mut batch, &edge2).unwrap();
        storage.write_batch(batch).unwrap();

        assert_eq!(storage.edge_count().unwrap(), 2);
        assert!(storage.get_edge(1).unwrap().is_some());
        assert!(storage.get_edge(2).unwrap().is_some());
    }

    #[test]
    fn test_graph_edge_modulated_weight() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_edge_modulation.db");
        let storage = GraphStorage::open_default(&db_path).unwrap();

        let edge = GraphEdge::new(
            42,
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            0.5,
            Domain::Code,
        );

        storage.put_edge(&edge).unwrap();
        let retrieved = storage.get_edge(42).unwrap().unwrap();

        // Verify modulated weight works after roundtrip
        // Same domain gives bonus
        let code_weight = retrieved.get_modulated_weight(Domain::Code);
        // Different domain no bonus
        let general_weight = retrieved.get_modulated_weight(Domain::General);

        // Code domain should have higher modulated weight due to domain bonus
        assert!(
            code_weight > general_weight,
            "Same domain should boost weight"
        );
    }
}
