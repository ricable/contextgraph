//! In-memory stub implementation of MemoryStore.

use async_trait::async_trait;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{CoreError, CoreResult};
use crate::traits::{MemoryStore, SearchOptions, StorageBackend};
use crate::types::{MemoryNode, NodeId};

/// In-memory store for Ghost System phase.
///
/// Uses a simple HashMap for storage with RwLock for concurrent access.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    nodes: Arc<RwLock<HashMap<NodeId, MemoryNode>>>,
}

impl InMemoryStore {
    /// Create a new in-memory store.
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn store(&self, node: MemoryNode) -> CoreResult<NodeId> {
        let id = node.id;
        let mut nodes = self.nodes.write().await;
        nodes.insert(id, node);
        Ok(id)
    }

    async fn retrieve(&self, id: NodeId) -> CoreResult<Option<MemoryNode>> {
        let nodes = self.nodes.read().await;
        Ok(nodes.get(&id).cloned())
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        options: SearchOptions,
    ) -> CoreResult<Vec<(MemoryNode, f32)>> {
        let nodes = self.nodes.read().await;
        let mut results: Vec<(MemoryNode, f32)> = nodes
            .values()
            .filter(|n| {
                // Apply filters
                if !options.include_deleted && n.metadata.deleted {
                    return false;
                }
                if let Some(ref quadrant) = options.johari_filter {
                    if &n.quadrant != quadrant {
                        return false;
                    }
                }
                if let Some(ref modality) = options.modality_filter {
                    if &n.metadata.modality != modality {
                        return false;
                    }
                }
                true
            })
            .map(|n| {
                let similarity = Self::cosine_similarity(query_embedding, &n.embedding);
                (n.clone(), similarity)
            })
            .filter(|(_, sim)| {
                if let Some(min_sim) = options.min_similarity {
                    *sim >= min_sim
                } else {
                    true
                }
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        results.truncate(options.top_k);
        Ok(results)
    }

    async fn search_text(
        &self,
        _query: &str,
        options: SearchOptions,
    ) -> CoreResult<Vec<(MemoryNode, f32)>> {
        // In stub, just return random nodes since we don't have real embeddings
        let nodes = self.nodes.read().await;
        let mut results: Vec<(MemoryNode, f32)> = nodes
            .values()
            .filter(|n| !n.metadata.deleted || options.include_deleted)
            .take(options.top_k)
            .map(|n| (n.clone(), 0.5)) // Mock similarity
            .collect();
        results.truncate(options.top_k);
        Ok(results)
    }

    async fn delete(&self, id: NodeId, soft: bool) -> CoreResult<bool> {
        let mut nodes = self.nodes.write().await;
        if soft {
            if let Some(node) = nodes.get_mut(&id) {
                node.metadata.deleted = true;
                return Ok(true);
            }
        } else if nodes.remove(&id).is_some() {
            return Ok(true);
        }
        Ok(false)
    }

    async fn update(&self, node: MemoryNode) -> CoreResult<bool> {
        use std::collections::hash_map::Entry;
        let mut nodes = self.nodes.write().await;
        if let Entry::Occupied(mut e) = nodes.entry(node.id) {
            e.insert(node);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn count(&self) -> CoreResult<usize> {
        let nodes = self.nodes.read().await;
        Ok(nodes.values().filter(|n| !n.metadata.deleted).count())
    }

    async fn compact(&self) -> CoreResult<()> {
        let mut nodes = self.nodes.write().await;
        nodes.retain(|_, n| !n.metadata.deleted);
        Ok(())
    }

    // =========================================================================
    // Persistence Operations
    // =========================================================================

    async fn flush(&self) -> CoreResult<()> {
        // In-memory: no-op, all writes are immediately visible
        Ok(())
    }

    async fn checkpoint(&self) -> CoreResult<PathBuf> {
        // In-memory: serialize to temp file for testing
        let nodes = self.nodes.read().await;
        let data = serde_json::to_vec(&*nodes)
            .map_err(|e| CoreError::StorageError(format!("Failed to serialize checkpoint: {}", e)))?;

        let path = std::env::temp_dir().join(format!(
            "inmemory_checkpoint_{}.json",
            chrono::Utc::now().timestamp_millis()
        ));

        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&data))
            .map_err(|e| CoreError::StorageError(format!("Failed to write checkpoint file: {}", e)))?;

        Ok(path)
    }

    async fn restore(&self, checkpoint: &Path) -> CoreResult<()> {
        let mut data = Vec::new();
        std::fs::File::open(checkpoint)
            .and_then(|mut f| f.read_to_end(&mut data))
            .map_err(|e| CoreError::StorageError(format!("Failed to read checkpoint file: {}", e)))?;

        let restored: HashMap<NodeId, MemoryNode> =
            serde_json::from_slice(&data)
                .map_err(|e| CoreError::StorageError(format!("Failed to deserialize checkpoint: {}", e)))?;

        let mut nodes = self.nodes.write().await;
        *nodes = restored;
        Ok(())
    }

    // =========================================================================
    // Statistics (Sync)
    // =========================================================================

    fn node_count_sync(&self) -> usize {
        // Use try_read for sync access - may undercount during writes
        self.nodes
            .try_read()
            .map(|n| n.values().filter(|n| !n.metadata.deleted).count())
            .unwrap_or(0)
    }

    fn storage_size_bytes(&self) -> usize {
        // Estimate: each node is roughly content_len + embedding_len * 4 + metadata
        self.nodes
            .try_read()
            .map(|nodes| {
                nodes
                    .values()
                    .map(|n| {
                        n.content.len() + (n.embedding.len() * 4) + 256 // 256 for metadata overhead
                    })
                    .sum()
            })
            .unwrap_or(0)
    }

    fn backend_type(&self) -> StorageBackend {
        StorageBackend::InMemory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let store = InMemoryStore::new();
        let embedding = vec![0.1; 1536];
        let node = MemoryNode::new("test content".to_string(), embedding);
        let id = node.id;

        store.store(node.clone()).await.unwrap();
        let retrieved = store.retrieve(id).await.unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "test content");
    }

    #[tokio::test]
    async fn test_retrieve_not_found() {
        let store = InMemoryStore::new();
        let result = store.retrieve(NodeId::new_v4()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_soft_delete() {
        let store = InMemoryStore::new();
        let embedding = vec![0.1; 1536];
        let node = MemoryNode::new("test".to_string(), embedding);
        let id = node.id;

        store.store(node).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        store.delete(id, true).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);

        // Node still exists but is marked deleted
        let retrieved = store.retrieve(id).await.unwrap();
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().metadata.deleted);
    }

    #[tokio::test]
    async fn test_hard_delete() {
        let store = InMemoryStore::new();
        let embedding = vec![0.1; 1536];
        let node = MemoryNode::new("test".to_string(), embedding);
        let id = node.id;

        store.store(node).await.unwrap();
        store.delete(id, false).await.unwrap();

        let retrieved = store.retrieve(id).await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_search() {
        let store = InMemoryStore::new();

        // Store some nodes
        for i in 0..5 {
            let mut embedding = vec![0.0; 1536];
            embedding[0] = i as f32;
            let node = MemoryNode::new(format!("content {}", i), embedding);
            store.store(node).await.unwrap();
        }

        let query = vec![1.0; 1536];
        let options = SearchOptions::new(3);
        let results = store.search(&query, options).await.unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((InMemoryStore::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(InMemoryStore::cosine_similarity(&a, &c).abs() < 0.001);
    }

    // =========================================================================
    // TC-GHOST-005: Memory Round-Trip Exact Match Tests
    // =========================================================================

    #[tokio::test]
    async fn test_memory_round_trip_exact_match() {
        // TC-GHOST-005: Store and retrieve MemoryNode by UUID - exact match
        let store = InMemoryStore::new();
        let embedding = vec![0.5; 1536];

        let mut node =
            MemoryNode::new("Test content for round-trip".to_string(), embedding.clone());
        node.importance = 0.8;
        let original_id = node.id;
        let original_content = node.content.clone();
        let original_importance = node.importance;

        // Store
        let stored_id = store.store(node).await.unwrap();
        assert_eq!(stored_id, original_id, "Stored ID must match original");

        // Retrieve
        let retrieved = store
            .retrieve(original_id)
            .await
            .unwrap()
            .expect("Node must exist");

        // Verify exact match
        assert_eq!(retrieved.id, original_id, "ID must match");
        assert_eq!(retrieved.content, original_content, "Content must match");
        assert_eq!(
            retrieved.importance, original_importance,
            "Importance must match"
        );
        assert_eq!(
            retrieved.embedding.len(),
            1536,
            "Embedding dimension must be preserved"
        );
    }

    #[tokio::test]
    async fn test_memory_round_trip_embedding_integrity() {
        // TC-GHOST-005: Embedding data must be preserved exactly
        let store = InMemoryStore::new();

        // Create embedding with specific values to test precision
        let mut embedding = vec![0.0; 1536];
        for i in 0..1536 {
            embedding[i] = (i as f32 / 1536.0) * 2.0 - 1.0; // Range [-1.0, 1.0]
        }

        let node = MemoryNode::new("Embedding integrity test".to_string(), embedding.clone());
        let id = node.id;

        store.store(node).await.unwrap();
        let retrieved = store.retrieve(id).await.unwrap().expect("Node must exist");

        // Verify embedding is exactly preserved
        assert_eq!(
            retrieved.embedding.len(),
            embedding.len(),
            "Embedding length must be preserved"
        );

        for (i, (&original, &stored)) in
            embedding.iter().zip(retrieved.embedding.iter()).enumerate()
        {
            assert_eq!(
                original, stored,
                "Embedding value at index {} must be exactly preserved: {} != {}",
                i, original, stored
            );
        }
    }

    #[tokio::test]
    async fn test_memory_round_trip_metadata_integrity() {
        // TC-GHOST-005: All metadata fields must be preserved
        let store = InMemoryStore::new();
        let embedding = vec![0.1; 1536];

        let mut node = MemoryNode::new("Metadata integrity test".to_string(), embedding);
        node.importance = 0.95;
        node.access_count = 42;
        node.metadata.deleted = false;
        node.metadata.source = Some("test-source".to_string());
        node.metadata.language = Some("en".to_string());
        node.metadata.tags = vec!["tag1".to_string(), "tag2".to_string()];
        node.metadata.utl_score = Some(0.75);
        node.metadata.consolidated = true;
        node.metadata.rationale = Some("Test rationale".to_string());

        let id = node.id;
        let original_created_at = node.created_at;
        let original_accessed_at = node.accessed_at;

        store.store(node).await.unwrap();
        let retrieved = store.retrieve(id).await.unwrap().expect("Node must exist");

        // Verify all fields
        assert_eq!(retrieved.importance, 0.95, "Importance must match");
        assert_eq!(retrieved.access_count, 42, "Access count must match");
        assert!(!retrieved.metadata.deleted, "Deleted flag must match");
        assert_eq!(
            retrieved.created_at, original_created_at,
            "Created timestamp must match"
        );
        assert_eq!(
            retrieved.accessed_at, original_accessed_at,
            "Accessed at must match"
        );

        // Verify metadata
        assert_eq!(retrieved.metadata.source, Some("test-source".to_string()));
        assert_eq!(retrieved.metadata.language, Some("en".to_string()));
        assert_eq!(
            retrieved.metadata.tags,
            vec!["tag1".to_string(), "tag2".to_string()]
        );
        assert_eq!(retrieved.metadata.utl_score, Some(0.75));
        assert_eq!(retrieved.metadata.consolidated, true);
        assert_eq!(
            retrieved.metadata.rationale,
            Some("Test rationale".to_string())
        );
    }

    #[tokio::test]
    async fn test_memory_multiple_nodes_independent() {
        // TC-GHOST-005: Multiple nodes must be stored and retrieved independently
        let store = InMemoryStore::new();

        let mut nodes = Vec::new();
        for i in 0..10 {
            let embedding = vec![i as f32 / 10.0; 1536];
            let mut node = MemoryNode::new(format!("Node content {}", i), embedding);
            node.importance = i as f32 / 10.0;
            nodes.push(node);
        }

        // Store all nodes
        let mut ids = Vec::new();
        for node in &nodes {
            let id = store.store(node.clone()).await.unwrap();
            ids.push(id);
        }

        // Verify each node independently
        for (i, id) in ids.iter().enumerate() {
            let retrieved = store.retrieve(*id).await.unwrap().expect("Node must exist");
            assert_eq!(retrieved.content, format!("Node content {}", i));
            assert_eq!(retrieved.importance, i as f32 / 10.0);
        }

        // Verify count
        assert_eq!(store.count().await.unwrap(), 10);
    }

    #[tokio::test]
    async fn test_memory_update_preserves_unmodified_fields() {
        // TC-GHOST-005: Update must preserve unmodified fields
        let store = InMemoryStore::new();
        let embedding = vec![0.5; 1536];

        let mut node = MemoryNode::new("Original content".to_string(), embedding.clone());
        node.importance = 0.5;
        node.metadata.source = Some("original-source".to_string());
        let id = node.id;
        let original_created_at = node.created_at;

        store.store(node).await.unwrap();

        // Retrieve and modify only specific fields
        let mut modified = store.retrieve(id).await.unwrap().unwrap();
        modified.content = "Modified content".to_string();
        modified.importance = 0.9;

        store.update(modified).await.unwrap();

        // Verify modifications and preserved fields
        let final_node = store.retrieve(id).await.unwrap().unwrap();
        assert_eq!(final_node.content, "Modified content");
        assert_eq!(final_node.importance, 0.9);
        assert_eq!(
            final_node.metadata.source,
            Some("original-source".to_string())
        );
        assert_eq!(final_node.created_at, original_created_at);
        assert_eq!(final_node.embedding, embedding);
    }

    #[tokio::test]
    async fn test_memory_search_returns_correct_nodes() {
        // TC-GHOST-005: Search must return nodes with correct similarity ordering
        let store = InMemoryStore::new();

        // Create nodes with specific embeddings for predictable similarity
        let query = vec![1.0; 1536];

        // Node 1: Perfect match with query
        let mut node1 = MemoryNode::new("Perfect match".to_string(), vec![1.0; 1536]);
        node1.importance = 0.8;

        // Node 2: Orthogonal (zero similarity)
        let mut node2_emb = vec![0.0; 1536];
        node2_emb[0] = 1.0; // Only first dimension non-zero
        let node2 = MemoryNode::new("Partial match".to_string(), node2_emb);

        store.store(node1).await.unwrap();
        store.store(node2).await.unwrap();

        let options = SearchOptions::new(10);
        let results = store.search(&query, options).await.unwrap();

        // Perfect match should be first
        assert!(!results.is_empty());
        assert_eq!(results[0].0.content, "Perfect match");

        // Similarity should be close to 1.0 for perfect match
        assert!(
            (results[0].1 - 1.0).abs() < 0.001,
            "Perfect match similarity should be ~1.0, got {}",
            results[0].1
        );
    }

    // =========================================================================
    // M06-T03: Persistence Operations Tests
    // =========================================================================

    #[tokio::test]
    async fn test_flush_is_noop() {
        let store = InMemoryStore::new();
        // Should complete without error
        store.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_checkpoint_restore() {
        println!("=== CHECKPOINT RESTORE TEST ===");
        let store = InMemoryStore::new();

        // Store some nodes
        let embedding = vec![0.5; 1536];
        let node1 = MemoryNode::new("content 1".to_string(), embedding.clone());
        let node2 = MemoryNode::new("content 2".to_string(), embedding);
        let id1 = node1.id;
        let id2 = node2.id;

        store.store(node1).await.unwrap();
        store.store(node2).await.unwrap();

        println!("BEFORE CHECKPOINT: count = {}", store.count().await.unwrap());
        assert_eq!(store.count().await.unwrap(), 2);

        // Create checkpoint
        let checkpoint_path = store.checkpoint().await.unwrap();
        println!("CHECKPOINT PATH: {:?}", checkpoint_path);
        assert!(checkpoint_path.exists(), "Checkpoint file must exist");

        // Verify checkpoint file has content
        let metadata = std::fs::metadata(&checkpoint_path).unwrap();
        assert!(metadata.len() > 0, "Checkpoint file must not be empty");
        println!("CHECKPOINT FILE SIZE: {} bytes", metadata.len());

        // Clear the store by hard deleting
        store.delete(id1, false).await.unwrap();
        store.delete(id2, false).await.unwrap();
        assert_eq!(
            store.count().await.unwrap(),
            0,
            "Store should be empty after delete"
        );
        println!("AFTER DELETE: count = {}", store.count().await.unwrap());

        // Restore from checkpoint
        store.restore(&checkpoint_path).await.unwrap();
        println!("AFTER RESTORE: count = {}", store.count().await.unwrap());

        // Verify restoration
        assert_eq!(
            store.count().await.unwrap(),
            2,
            "Store should have 2 nodes after restore"
        );
        assert!(
            store.retrieve(id1).await.unwrap().is_some(),
            "Node 1 should exist"
        );
        assert!(
            store.retrieve(id2).await.unwrap().is_some(),
            "Node 2 should exist"
        );

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
        println!("RESULT: PASS - Checkpoint/restore works correctly");
    }

    #[tokio::test]
    async fn test_checkpoint_restore_empty_store() {
        // Edge case: checkpoint empty store
        let store = InMemoryStore::new();
        assert_eq!(store.count().await.unwrap(), 0);

        let checkpoint_path = store.checkpoint().await.unwrap();
        assert!(checkpoint_path.exists());

        // Restore empty checkpoint
        store.restore(&checkpoint_path).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
    }

    #[tokio::test]
    async fn test_restore_invalid_path() {
        let store = InMemoryStore::new();
        let invalid_path = std::path::Path::new("/nonexistent/path/checkpoint.json");

        let result = store.restore(invalid_path).await;
        assert!(result.is_err(), "Restore from invalid path must fail");

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("Storage error"),
                "Error should be StorageError, got: {}",
                error_msg
            );
        }
    }

    #[tokio::test]
    async fn test_checkpoint_with_empty_content() {
        // Edge case: node with empty content
        let store = InMemoryStore::new();

        let node = MemoryNode::new(String::new(), vec![0.0; 1536]);
        let id = node.id;
        store.store(node).await.unwrap();

        let checkpoint_path = store.checkpoint().await.unwrap();

        // Clear and restore
        store.delete(id, false).await.unwrap();
        store.restore(&checkpoint_path).await.unwrap();

        let retrieved = store.retrieve(id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "");

        // Cleanup
        std::fs::remove_file(&checkpoint_path).ok();
    }

    // =========================================================================
    // M06-T03: Statistics Tests
    // =========================================================================

    #[tokio::test]
    async fn test_node_count_sync() {
        let store = InMemoryStore::new();

        // Initially empty
        assert_eq!(store.node_count_sync(), 0);

        // Add nodes
        for i in 0..5 {
            let embedding = vec![i as f32 / 10.0; 1536];
            let node = MemoryNode::new(format!("content {}", i), embedding);
            store.store(node).await.unwrap();
        }

        assert_eq!(store.node_count_sync(), 5);

        // Soft delete should reduce count
        let embedding = vec![0.1; 1536];
        let node_to_delete = MemoryNode::new("to delete".to_string(), embedding);
        let delete_id = node_to_delete.id;
        store.store(node_to_delete).await.unwrap();
        assert_eq!(store.node_count_sync(), 6);

        store.delete(delete_id, true).await.unwrap();
        assert_eq!(store.node_count_sync(), 5);
    }

    #[tokio::test]
    async fn test_storage_size_bytes() {
        let store = InMemoryStore::new();

        // Initially zero
        let initial_size = store.storage_size_bytes();
        assert_eq!(initial_size, 0);

        // Add a node
        let embedding = vec![0.1; 1536];
        let node = MemoryNode::new("test content".to_string(), embedding);
        store.store(node).await.unwrap();

        let size = store.storage_size_bytes();
        // Should be at least content (12) + embedding (1536 * 4) + metadata (256) = ~6412
        let expected_min = 12 + (1536 * 4) + 256;
        assert!(
            size >= expected_min,
            "Size should be >= {}, got {}",
            expected_min,
            size
        );
    }

    #[tokio::test]
    async fn test_backend_type() {
        let store = InMemoryStore::new();
        assert_eq!(store.backend_type(), StorageBackend::InMemory);
    }
}
