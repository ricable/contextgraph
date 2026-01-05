//! Column family name constants for RocksDB storage.
//!
//! These constants define the names of all column families used by the graph storage.
//! Order matters for RocksDB - must match descriptor generation order.

/// Column family for adjacency lists (edge data).
/// Key: node_id (16 bytes UUID)
/// Value: Vec<GraphEdge> (variable length, bincode)
/// Optimized for: prefix scans (listing all edges from a node)
pub const CF_ADJACENCY: &str = "adjacency";

/// Column family for hyperbolic coordinates.
/// Key: node_id (16 bytes UUID)
/// Value: [f32; 64] = 256 bytes (Poincaré ball coordinates)
/// Optimized for: point lookups, GPU batch loading
pub const CF_HYPERBOLIC: &str = "hyperbolic";

/// Column family for entailment cones.
/// Key: node_id (16 bytes UUID)
/// Value: EntailmentCone = 268 bytes (256 coords + 4 aperture + 4 factor + 4 depth)
/// Optimized for: range scans with bloom filter
pub const CF_CONES: &str = "entailment_cones";

/// Column family for FAISS ID mapping.
/// Key: node_id (16 bytes UUID)
/// Value: FAISS internal ID (i64 = 8 bytes)
/// Optimized for: point lookups, bidirectional mapping
pub const CF_FAISS_IDS: &str = "faiss_ids";

/// Column family for node data.
/// Key: node_id (16 bytes UUID)
/// Value: MemoryNode (variable length, bincode)
/// Optimized for: point lookups
pub const CF_NODES: &str = "nodes";

/// Column family for metadata (schema version, stats, etc.).
/// Key: key string
/// Value: JSON value
/// Optimized for: small dataset, infrequent access
pub const CF_METADATA: &str = "metadata";

/// Column family for full GraphEdge storage (M04-T15).
/// Key: edge_id (8 bytes i64)
/// Value: GraphEdge (bincode serialized, all 13 Marblestone fields)
/// Optimized for: point lookups by edge ID
///
/// This CF stores full edges with NT weights, domain, steering_reward, etc.
/// Used by BFS traversal to get modulated weights.
pub const CF_EDGES: &str = "edges";

/// All column family names in order.
/// Order matters for RocksDB - must match descriptor generation order.
pub const ALL_COLUMN_FAMILIES: &[&str] = &[
    CF_ADJACENCY,
    CF_HYPERBOLIC,
    CF_CONES,
    CF_FAISS_IDS,
    CF_NODES,
    CF_METADATA,
    CF_EDGES,
];
