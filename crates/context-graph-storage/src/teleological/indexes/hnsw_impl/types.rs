//! HNSW index types and helpers.
//!
//! Contains the `HnswEmbedderIndex` struct and helper functions for usearch integration.

use std::collections::HashMap;
// HIGH-17 FIX: parking_lot::RwLock is non-poisonable. One panic no longer
// permanently breaks all subsequent HNSW operations via poison cascade.
use parking_lot::RwLock;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use uuid::Uuid;

use super::super::get_hnsw_config;
use super::super::hnsw_config::{DistanceMetric, EmbedderIndex, HnswConfig};

/// Convert our DistanceMetric to usearch MetricKind.
///
/// # Panics
///
/// Panics with "METRIC ERROR" if MaxSim is passed (requires token-level computation).
pub(crate) fn metric_to_usearch(metric: DistanceMetric) -> MetricKind {
    match metric {
        DistanceMetric::Cosine => MetricKind::Cos,
        DistanceMetric::DotProduct => MetricKind::IP,
        DistanceMetric::Euclidean => MetricKind::L2sq,
        DistanceMetric::AsymmetricCosine => MetricKind::Cos, // Asymmetry handled at query time
        DistanceMetric::MaxSim => {
            panic!("METRIC ERROR: MaxSim not supported for HNSW - use E12 ColBERT index")
        }
    }
}

/// HNSW index for a single embedder using usearch for O(log n) graph traversal.
///
/// Stores vectors with UUID associations and supports approximate nearest neighbor search.
///
/// # Thread Safety
///
/// Uses `RwLock` for interior mutability. usearch::Index itself is Send + Sync.
/// Multiple readers can access concurrently, but writes are exclusive.
///
/// # Performance
///
/// - Insert: O(log n) via HNSW graph construction
/// - Search: O(log n) via HNSW graph traversal (NOT brute force!)
/// - Target: <10ms search @ 1M vectors
///
/// # Example
///
/// ```
/// use context_graph_storage::teleological::indexes::{
///     EmbedderIndex, HnswEmbedderIndex, EmbedderIndexOps,
/// };
/// use uuid::Uuid;
///
/// let index = HnswEmbedderIndex::new(EmbedderIndex::E8Graph);
/// assert_eq!(index.config().dimension, 1024);
///
/// let id = Uuid::new_v4();
/// let vector = vec![0.5f32; 1024];
/// index.insert(id, &vector).unwrap();
///
/// let results = index.search(&vector, 1, None).unwrap();
/// assert_eq!(results[0].0, id);
/// ```
pub struct HnswEmbedderIndex {
    pub(crate) embedder: EmbedderIndex,
    pub(crate) config: HnswConfig,
    /// usearch HNSW index - provides O(log n) graph traversal
    pub(crate) index: RwLock<Index>,
    /// UUID to usearch key mapping
    pub(crate) id_to_key: RwLock<HashMap<Uuid, u64>>,
    /// usearch key to UUID mapping (for result conversion)
    pub(crate) key_to_id: RwLock<HashMap<u64, Uuid>>,
    /// Next available key for usearch (monotonically increasing)
    pub(crate) next_key: RwLock<u64>,
}

impl HnswEmbedderIndex {
    /// Create new index for specified embedder.
    ///
    /// Initializes usearch Index with configuration from HnswConfig:
    /// - dimensions from config.dimension
    /// - metric from config.metric (mapped to usearch MetricKind)
    /// - connectivity (M) from config.m
    /// - expansion_add (ef_construction) from config.ef_construction
    /// - expansion_search (ef_search) from config.ef_search
    ///
    /// # Panics
    ///
    /// - Panics with "FAIL FAST" message if embedder has no HNSW config (E6, E12, E13).
    /// - Panics if usearch Index creation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_storage::teleological::indexes::{
    ///     EmbedderIndex, EmbedderIndexOps, HnswEmbedderIndex,
    /// };
    ///
    /// let index = HnswEmbedderIndex::new(EmbedderIndex::E1Semantic);
    /// assert_eq!(index.config().dimension, 1024);
    /// ```
    ///
    /// ```should_panic
    /// use context_graph_storage::teleological::indexes::{EmbedderIndex, HnswEmbedderIndex};
    ///
    /// // This will panic - E6 uses inverted index
    /// let _index = HnswEmbedderIndex::new(EmbedderIndex::E6Sparse);
    /// ```
    pub fn new(embedder: EmbedderIndex) -> Self {
        let config = get_hnsw_config(embedder).unwrap_or_else(|| {
            panic!(
                "FAIL FAST: No HNSW config for {:?}. Use InvertedIndex for E6/E13, MaxSim for E12.",
                embedder
            )
        });

        let usearch_metric = metric_to_usearch(config.metric);

        let options = IndexOptions {
            dimensions: config.dimension,
            metric: usearch_metric,
            quantization: ScalarKind::F32,
            connectivity: config.m,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            ..Default::default()
        };

        let index = Index::new(&options).unwrap_or_else(|e| {
            panic!(
                "FAIL FAST: Failed to create usearch index for {:?}: {}",
                embedder, e
            )
        });

        // Reserve initial capacity - usearch requires this before adding vectors
        // Start with reasonable initial capacity, will grow as needed
        const INITIAL_CAPACITY: usize = 1024;
        index.reserve(INITIAL_CAPACITY).unwrap_or_else(|e| {
            panic!(
                "FAIL FAST: Failed to reserve capacity for {:?}: {}",
                embedder, e
            )
        });

        Self {
            embedder,
            config,
            index: RwLock::new(index),
            id_to_key: RwLock::new(HashMap::new()),
            key_to_id: RwLock::new(HashMap::new()),
            next_key: RwLock::new(0),
        }
    }

    /// Create index with custom config (for testing).
    ///
    /// # Arguments
    ///
    /// * `embedder` - Embedder type this index serves
    /// * `config` - Custom HNSW configuration
    ///
    /// # Note
    ///
    /// Use `new()` for production - this bypasses config validation.
    #[allow(dead_code)]
    pub fn with_config(embedder: EmbedderIndex, config: HnswConfig) -> Self {
        let usearch_metric = metric_to_usearch(config.metric);

        let options = IndexOptions {
            dimensions: config.dimension,
            metric: usearch_metric,
            quantization: ScalarKind::F32,
            connectivity: config.m,
            expansion_add: config.ef_construction,
            expansion_search: config.ef_search,
            ..Default::default()
        };

        let index = Index::new(&options).unwrap_or_else(|e| {
            panic!(
                "FAIL FAST: Failed to create usearch index for {:?}: {}",
                embedder, e
            )
        });

        // Reserve initial capacity - usearch requires this before adding vectors
        const INITIAL_CAPACITY: usize = 1024;
        index.reserve(INITIAL_CAPACITY).unwrap_or_else(|e| {
            panic!(
                "FAIL FAST: Failed to reserve capacity for {:?}: {}",
                embedder, e
            )
        });

        Self {
            embedder,
            config,
            index: RwLock::new(index),
            id_to_key: RwLock::new(HashMap::new()),
            key_to_id: RwLock::new(HashMap::new()),
            next_key: RwLock::new(0),
        }
    }

    /// Check if a vector ID exists in the index.
    pub fn contains(&self, id: Uuid) -> bool {
        self.id_to_key.read().contains_key(&id)
    }

    /// Get all vector IDs in the index.
    pub fn ids(&self) -> Vec<Uuid> {
        self.id_to_key.read().keys().copied().collect()
    }

    /// Serialize the HNSW graph to a byte buffer.
    ///
    /// Uses usearch's `save_to_buffer` for the graph structure.
    /// Returns None if the index is empty (nothing to persist).
    pub fn serialize_graph(&self) -> Result<Option<Vec<u8>>, String> {
        let idx = self.index.read();
        if idx.size() == 0 {
            return Ok(None);
        }
        let len = idx.serialized_length();
        let mut buf = vec![0u8; len];
        idx.save_to_buffer(&mut buf)
            .map_err(|e| format!("usearch save_to_buffer failed for {:?}: {}", self.embedder, e))?;
        Ok(Some(buf))
    }

    /// Serialize UUID↔key mappings + next_key as JSON.
    ///
    /// Returns None if the index is empty.
    pub fn serialize_metadata(&self) -> Option<Vec<u8>> {
        let id_to_key = self.id_to_key.read();
        if id_to_key.is_empty() {
            return None;
        }
        let next_key = *self.next_key.read();
        let meta = HnswPersistMetadata {
            mappings: id_to_key.iter().map(|(id, k)| (*id, *k)).collect(),
            next_key,
        };
        Some(serde_json::to_vec(&meta).expect("HnswPersistMetadata serialization cannot fail"))
    }

    /// Restore index state from persisted graph + metadata bytes.
    ///
    /// Replaces the current HNSW graph and UUID mappings with the persisted data.
    /// The usearch index options (dimension, metric, etc.) must match the persisted graph.
    ///
    /// # Errors
    ///
    /// Returns error if usearch `load_from_buffer` fails or metadata is corrupted.
    pub fn restore_from_persisted(
        &self,
        graph_data: &[u8],
        meta_data: &[u8],
    ) -> Result<usize, String> {
        // Deserialize metadata first (cheaper, validates before touching index)
        let meta: HnswPersistMetadata = serde_json::from_slice(meta_data)
            .map_err(|e| format!("metadata deserialization failed for {:?}: {}", self.embedder, e))?;

        // Load graph into usearch index
        let idx = self.index.write();
        idx.load_from_buffer(graph_data)
            .map_err(|e| format!("usearch load_from_buffer failed for {:?}: {}", self.embedder, e))?;

        // Rebuild UUID mappings
        let mut id_to_key = self.id_to_key.write();
        let mut key_to_id = self.key_to_id.write();
        let mut next_key = self.next_key.write();

        id_to_key.clear();
        key_to_id.clear();

        let count = meta.mappings.len();
        id_to_key.reserve(count);
        key_to_id.reserve(count);

        for (uuid, key) in &meta.mappings {
            id_to_key.insert(*uuid, *key);
            key_to_id.insert(*key, *uuid);
        }
        *next_key = meta.next_key;

        Ok(count)
    }
}

/// Metadata for persisting HNSW UUID↔key mappings.
#[derive(serde::Serialize, serde::Deserialize)]
struct HnswPersistMetadata {
    mappings: Vec<(Uuid, u64)>,
    next_key: u64,
}
