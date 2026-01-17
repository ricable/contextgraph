# TASK-P4-007: MultiSpaceClusterManager

```xml
<task_spec id="TASK-P4-007" version="3.0">
<metadata>
  <title>MultiSpaceClusterManager Implementation</title>
  <status>COMPLETE</status>
  <layer>logic</layer>
  <sequence>33</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-02</requirement_ref>
    <requirement_ref>REQ-P4-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P4-005</task_ref>
    <task_ref status="COMPLETE">TASK-P4-006</task_ref>
    <task_ref status="COMPLETE">TASK-P1-005</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <last_audit>2025-01-17</last_audit>
</metadata>

<context>
Implements MultiSpaceClusterManager which coordinates clustering across all 13
embedding spaces. Maintains HDBSCAN clusterers for batch reclustering and BIRCH
trees for incremental updates per space. Handles memory insertion and full
reclustering operations.

**IMPORTANT:** Clustering runs on ALL 13 spaces, but topic synthesis (TASK-P4-008)
uses weighted agreement where:
- Semantic embedders (E1, E5, E6, E7, E10, E12, E13): weight 1.0
- Temporal embedders (E2, E3, E4): weight 0.0 (excluded from topics)
- Relational embedders (E8, E11): weight 0.5
- Structural embedder (E9): weight 0.5

This is the central coordination point for all clustering operations.
</context>

<current_codebase_state>
CRITICAL: Read this section to understand what ALREADY EXISTS vs what needs implementation.

=====================================================================================
EXISTING FILES AND TYPES (VERIFIED 2025-01-17)
=====================================================================================

File: crates/context-graph-core/src/clustering/mod.rs (42 lines)
CURRENT EXPORTS:
```rust
pub use birch::{birch_defaults, BIRCHEntry, BIRCHNode, BIRCHParams, BIRCHTree, ClusteringFeature};
pub use cluster::Cluster;
pub use error::ClusterError;
pub use hdbscan::{hdbscan_defaults, ClusterSelectionMethod, HDBSCANClusterer, HDBSCANParams};
pub use membership::ClusterMembership;
pub use topic::{Topic, TopicPhase, TopicProfile, TopicStability};
```

File: crates/context-graph-core/src/clustering/hdbscan.rs (1,774 lines) - COMPLETE
AVAILABLE:
- HDBSCANClusterer::new(params) / with_defaults() / for_space(Embedder)
- HDBSCANClusterer::fit(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;], ids: &amp;[Uuid], space: Embedder) -> Result&lt;Vec&lt;ClusterMembership&gt;, ClusterError&gt;
- HDBSCANParams with builder pattern and validate()

File: crates/context-graph-core/src/clustering/birch.rs (2,528 lines) - COMPLETE
AVAILABLE:
- BIRCHTree::new(params, dimension) -> Result&lt;Self, ClusterError&gt;
- BIRCHTree::insert(&amp;mut self, embedding: &amp;[f32], memory_id: Uuid) -> Result&lt;usize, ClusterError&gt;
- BIRCHTree::get_clusters() -> Vec&lt;ClusteringFeature&gt;
- BIRCHTree::get_cluster_members() -> Vec&lt;(usize, Vec&lt;Uuid&gt;)&gt;
- BIRCHTree::cluster_count() -> usize
- BIRCHTree::total_points() -> usize
- BIRCHParams with builder pattern and validate()

File: crates/context-graph-core/src/clustering/membership.rs (286 lines) - COMPLETE
AVAILABLE:
- ClusterMembership::new(memory_id, space, cluster_id, probability, is_core_point)
- ClusterMembership::noise(memory_id, space) -> Self (cluster_id = -1)
- ClusterMembership::is_noise() -> bool
- ClusterMembership::is_confident() -> bool (probability >= 0.8)

File: crates/context-graph-core/src/clustering/error.rs (142 lines) - COMPLETE
AVAILABLE:
- ClusterError::InsufficientData { required, actual }
- ClusterError::DimensionMismatch { expected, actual }
- ClusterError::NoValidClusters
- ClusterError::InvalidParameter(String)
- ClusterError::SpaceNotInitialized(Embedder)
- ClusterError::StorageError(String)
- Helper constructors: insufficient_data(), dimension_mismatch(), invalid_parameter()

=====================================================================================
MEMORY AND EMBEDDING TYPES (VERIFIED ACTUAL SIGNATURES)
=====================================================================================

File: crates/context-graph-core/src/memory/mod.rs
Memory struct fields:
```rust
pub struct Memory {
    pub id: Uuid,
    pub content: String,
    pub source: MemorySource,
    pub created_at: DateTime&lt;Utc&gt;,
    pub session_id: String,
    pub teleological_array: TeleologicalArray,  // THIS IS THE 13-EMBEDDING ARRAY
    pub chunk_metadata: Option&lt;ChunkMetadata&gt;,
    pub word_count: u32,
}
```

File: crates/context-graph-core/src/memory/store.rs
MemoryStore IS SYNCHRONOUS (NOT ASYNC):
```rust
impl MemoryStore {
    pub fn new(path: &amp;Path) -> Result&lt;Self, StorageError&gt;
    pub fn store(&amp;self, memory: &amp;Memory) -> Result&lt;(), StorageError&gt;
    pub fn get(&amp;self, id: Uuid) -> Result&lt;Option&lt;Memory&gt;, StorageError&gt;
    pub fn get_by_session(&amp;self, session_id: &amp;str) -> Result&lt;Vec&lt;Memory&gt;, StorageError&gt;
    pub fn count(&amp;self) -> Result&lt;u64, StorageError&gt;
    pub fn delete(&amp;self, id: Uuid) -> Result&lt;bool, StorageError&gt;
    // NOTE: NO get_all() method exists - iterate sessions or add one
}
```

File: crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs
TeleologicalArray is type alias for SemanticFingerprint:
```rust
pub type TeleologicalArray = SemanticFingerprint;

pub struct SemanticFingerprint {
    pub e1_semantic: Vec&lt;f32&gt;,           // E1: 1024D dense
    pub e2_temporal_recent: Vec&lt;f32&gt;,    // E2: 512D dense
    pub e3_temporal_periodic: Vec&lt;f32&gt;,  // E3: 512D dense
    pub e4_temporal_positional: Vec&lt;f32&gt;,// E4: 512D dense
    pub e5_causal: Vec&lt;f32&gt;,             // E5: 768D dense (asymmetric)
    pub e6_sparse: SparseVector,          // E6: ~30K sparse SPLADE
    pub e7_code: Vec&lt;f32&gt;,               // E7: 1536D dense
    pub e8_graph: Vec&lt;f32&gt;,              // E8: 384D dense (Emotional/Relational)
    pub e9_hdc: Vec&lt;f32&gt;,                // E9: 1024D dense (projected from 10K)
    pub e10_multimodal: Vec&lt;f32&gt;,        // E10: 768D dense
    pub e11_entity: Vec&lt;f32&gt;,            // E11: 384D dense
    pub e12_late_interaction: Vec&lt;Vec&lt;f32&gt;&gt;, // E12: 128D per token (variable)
    pub e13_splade: SparseVector,         // E13: ~30K sparse SPLADE v3
}

// Accessor method:
impl SemanticFingerprint {
    pub fn get(&amp;self, embedder: Embedder) -> EmbeddingRef&lt;'_&gt;
}

pub enum EmbeddingRef&lt;'a&gt; {
    Dense(&amp;'a [f32]),           // E1-E5, E7-E11
    Sparse(&amp;'a SparseVector),   // E6, E13
    TokenLevel(&amp;'a [Vec&lt;f32&gt;]), // E12
}
```

File: crates/context-graph-core/src/teleological/embedder.rs
Embedder enum (CANONICAL):
```rust
#[repr(u8)]
pub enum Embedder {
    Semantic = 0,           // E1
    TemporalRecent = 1,     // E2
    TemporalPeriodic = 2,   // E3
    TemporalPositional = 3, // E4
    Causal = 4,             // E5
    Sparse = 5,             // E6
    Code = 6,               // E7
    Emotional = 7,          // E8 (NOT E8Graph, NOT E8Emotional)
    Hdc = 8,                // E9 (NOT E9HDC)
    Multimodal = 9,         // E10
    Entity = 10,            // E11
    LateInteraction = 11,   // E12
    KeywordSplade = 12,     // E13
}

impl Embedder {
    pub const COUNT: usize = 13;
    pub fn index(self) -> usize
    pub fn from_index(idx: usize) -> Option&lt;Self&gt;
    pub fn all() -> impl ExactSizeIterator&lt;Item = Embedder&gt;
}
```

File: crates/context-graph-core/src/types/fingerprint/sparse.rs
SparseVector:
```rust
impl SparseVector {
    pub fn nnz(&amp;self) -> usize  // Non-zero count
    pub fn to_dense(&amp;self, vocab_size: usize) -> Vec&lt;f32&gt;  // Convert to dense
    pub fn indices(&amp;self) -> &amp;[u32]
    pub fn values(&amp;self) -> &amp;[f32]
}
```

=====================================================================================
DIMENSION CONSTANTS
=====================================================================================

E1:  1024   E2:  512   E3:  512   E4:  512   E5:  768
E6:  sparse E7:  1536  E8:  384   E9:  1024  E10: 768
E11: 384    E12: 128/token        E13: sparse

File: crates/context-graph-core/src/embeddings/config.rs
```rust
pub fn get_dimension(embedder: Embedder) -> usize {
    match embedder {
        Embedder::Semantic => 1024,
        Embedder::TemporalRecent | Embedder::TemporalPeriodic | Embedder::TemporalPositional => 512,
        Embedder::Causal => 768,
        Embedder::Sparse => 30522,      // SPLADE vocab size
        Embedder::Code => 1536,
        Embedder::Emotional => 384,
        Embedder::Hdc => 1024,
        Embedder::Multimodal => 768,
        Embedder::Entity => 384,
        Embedder::LateInteraction => 128, // Per-token dimension
        Embedder::KeywordSplade => 30522, // SPLADE vocab size
    }
}
```

=====================================================================================
NOT YET IMPLEMENTED (THIS TASK)
=====================================================================================

File: crates/context-graph-core/src/clustering/manager.rs (NEW FILE)
- MultiSpaceClusterManager struct
- Per-space BIRCH tree management
- Batch HDBSCAN clustering across all spaces
- Incremental memory insertion
- Cluster membership tracking per memory
- Progressive activation tiers
</current_codebase_state>

<constitution_requirements>
Source: docs2/constitution.yaml

CRITICAL ARCHITECTURE RULES:
  ARCH-01: TeleologicalArray is atomic - store all 13 embeddings or nothing
  ARCH-02: Apples-to-apples only - compare E1&lt;-&gt;E1, never E1&lt;-&gt;E5
  ARCH-04: Temporal embedders (E2-E4) NEVER count toward topic detection - metadata only
  ARCH-09: Topic threshold is weighted_agreement >= 2.5 (not raw space count)
  AP-14: No .unwrap() in library code - use expect() with context or propagate errors
  AP-60: Temporal embedders MUST NOT count toward topic detection

CLUSTERING SPEC:
  algorithms.batch: "HDBSCAN per embedding space"
  algorithms.online: "BIRCH CF-trees for incremental updates"
  parameters.min_cluster_size: 3
  parameters.silhouette_threshold: 0.3

PROGRESSIVE ACTIVATION TIERS:
  tier_0: 0 memories - No clustering, defaults
  tier_1: 1-2 memories - Pairwise similarity
  tier_2: 3-9 memories - Basic clustering
  tier_3: 10-29 memories - Multiple clusters, Divergence detection (REAL CLUSTERING STARTS)
  tier_4: 30-99 memories - Reliable statistics
  tier_5: 100-499 memories - Sub-clustering
  tier_6: 500+ memories - Full personalization
</constitution_requirements>

<input_context_files>
  <file purpose="constitution" MUST_READ="true">docs2/constitution.yaml</file>
  <file purpose="traceability" MUST_READ="true">docs2/impplan/tasks/_traceability.md</file>
  <file purpose="hdbscan_impl">crates/context-graph-core/src/clustering/hdbscan.rs</file>
  <file purpose="birch_impl">crates/context-graph-core/src/clustering/birch.rs</file>
  <file purpose="memory_types">crates/context-graph-core/src/memory/mod.rs</file>
  <file purpose="memory_store">crates/context-graph-core/src/memory/store.rs</file>
  <file purpose="fingerprint">crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs</file>
  <file purpose="embedder_enum">crates/context-graph-core/src/teleological/embedder.rs</file>
  <file purpose="error_types">crates/context-graph-core/src/clustering/error.rs</file>
  <file purpose="embeddings_config">crates/context-graph-core/src/embeddings/config.rs</file>
</input_context_files>

<prerequisites>
  <check status="VERIFIED">TASK-P4-005 complete - HDBSCANClusterer exists with fit() method</check>
  <check status="VERIFIED">TASK-P4-006 complete - BIRCHTree exists with insert() method</check>
  <check status="VERIFIED">TASK-P1-005 complete - MemoryStore exists (SYNCHRONOUS, not async)</check>
  <check status="VERIFIED">ClusterMembership exists with noise() constructor</check>
  <check status="VERIFIED">Embedder::all() iterator available</check>
</prerequisites>

<scope>
  <in_scope>
    - Create manager.rs file in clustering module
    - Create MultiSpaceClusterManager struct
    - Maintain per-space BIRCH trees (HashMap&lt;Embedder, BIRCHTree&gt;)
    - Implement cluster_all_spaces batch method using HDBSCAN
    - Implement insert_memory for incremental BIRCH updates
    - Implement recluster_space for single space
    - Track ClusterMemberships per memory
    - Progressive activation based on memory count
    - Helper function to extract embedding vectors from TeleologicalArray
    - Handle all 3 embedding types: Dense, Sparse, TokenLevel
  </in_scope>
  <out_of_scope>
    - Topic synthesis (TASK-P4-008)
    - Persistence of cluster state (handled by storage layer)
    - Distributed clustering
    - GPU acceleration
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/manager.rs">
/// Progressive activation tier thresholds
const TIER_THRESHOLDS: [usize; 7] = [0, 1, 3, 10, 30, 100, 500];

/// Manager for clustering across all 13 embedding spaces.
///
/// Coordinates HDBSCAN (batch) and BIRCH (incremental) clustering.
/// Maintains per-space BIRCH trees and tracks cluster memberships.
///
/// # Constitution Compliance
///
/// - Clusters ALL 13 spaces (weighting applied in TopicSynthesizer)
/// - Progressive activation: real clustering starts at tier 3 (10+ memories)
/// - Uses HDBSCAN min_cluster_size=3 per constitution
pub struct MultiSpaceClusterManager {
    /// Per-space BIRCH trees for incremental clustering
    birch_trees: HashMap&lt;Embedder, BIRCHTree&gt;,
    /// Cluster memberships per memory
    memberships: HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt;,
    /// HDBSCAN params per space
    hdbscan_params: HashMap&lt;Embedder, HDBSCANParams&gt;,
    /// BIRCH params per space
    birch_params: HashMap&lt;Embedder, BIRCHParams&gt;,
    /// Total memory count for tier calculation
    memory_count: usize,
    /// Last recluster time per space
    last_recluster: HashMap&lt;Embedder, DateTime&lt;Utc&gt;&gt;,
}

impl MultiSpaceClusterManager {
    /// Create a new manager with default per-space parameters.
    pub fn new() -> Result&lt;Self, ClusterError&gt;;

    /// Get current activation tier (0-6).
    pub fn activation_tier(&amp;self) -> usize;

    /// Check if real clustering is active (tier >= 3, memory_count >= 10).
    pub fn is_clustering_active(&amp;self) -> bool;

    /// Cluster all spaces using HDBSCAN (batch operation).
    ///
    /// For tier 0-2: Returns all noise memberships.
    /// For tier 3+: Runs HDBSCAN per space.
    ///
    /// # Errors
    /// - ClusterError::StorageError if extraction fails
    pub fn cluster_all_spaces(
        &amp;mut self,
        memories: &amp;[Memory],
    ) -> Result&lt;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;, ClusterError&gt;;

    /// Insert a memory incrementally using BIRCH trees.
    ///
    /// Updates all 13 BIRCH trees and tracks membership.
    ///
    /// # Errors
    /// - ClusterError::DimensionMismatch if embedding dimensions wrong
    pub fn insert_memory(&amp;mut self, memory: &amp;Memory) -> Result&lt;(), ClusterError&gt;;

    /// Recluster a single space using HDBSCAN.
    ///
    /// Rebuilds the BIRCH tree for that space.
    ///
    /// # Arguments
    /// * `space` - The embedding space to recluster
    /// * `memories` - All memories (need their embeddings for this space)
    pub fn recluster_space(
        &amp;mut self,
        space: Embedder,
        memories: &amp;[Memory],
    ) -> Result&lt;(), ClusterError&gt;;

    /// Get memberships for a memory.
    pub fn get_memberships(&amp;self, memory_id: &amp;Uuid) -> Option&lt;&amp;Vec&lt;ClusterMembership&gt;&gt;;

    /// Get all memberships.
    pub fn get_all_memberships(&amp;self) -> &amp;HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt;;

    /// Check if space should be reclustered.
    ///
    /// Returns true if:
    /// - Never reclustered AND memory_count >= 10
    /// - 24+ hours since last recluster
    /// - BIRCH cluster count error > 15%
    pub fn should_recluster(&amp;self, space: Embedder) -> bool;

    /// Get cluster for a memory in a specific space.
    pub fn get_cluster_for_memory(&amp;self, memory_id: &amp;Uuid, space: Embedder) -> Option&lt;i32&gt;;

    /// Get memory count.
    pub fn memory_count(&amp;self) -> usize;

    /// Get BIRCH cluster count for a space.
    pub fn cluster_count(&amp;self, space: Embedder) -> usize;
}

/// Extract embedding vector for a specific space from TeleologicalArray.
///
/// Handles all three embedding types:
/// - Dense: Direct Vec&lt;f32&gt; clone
/// - Sparse (E6, E13): Convert to dense using vocab size
/// - TokenLevel (E12): Mean pooling across tokens
///
/// # Returns
/// Dense vector suitable for clustering (always Vec&lt;f32&gt;)
pub fn extract_embedding_for_space(
    array: &amp;TeleologicalArray,
    space: Embedder,
) -> Vec&lt;f32&gt;;
    </signature>
  </signatures>

  <constraints>
    - Tier 0-2 (0-9 memories): No real clustering, all noise
    - Tier 3+ (10+ memories): Real clustering active
    - BIRCH used for incremental O(log n), HDBSCAN for batch
    - Each space clustered independently (ARCH-02)
    - All 13 spaces clustered; topic weighting applied in TopicSynthesizer (TASK-P4-008)
    - MemoryStore is SYNCHRONOUS - do NOT add async/await
    - Use ClusterError for all errors, NEVER panic
    - No .unwrap() - use expect() with context or ? propagation
  </constraints>

  <verification>
    - cluster_all_spaces clusters all 13 spaces
    - insert_memory updates all 13 BIRCH trees
    - Memberships stored per memory (13 entries per memory)
    - Progressive activation tiers respected
    - recluster_space rebuilds single space
    - extract_embedding_for_space handles Dense, Sparse, TokenLevel
    - All tests pass with [PASS] output visibility
  </verification>
</definition_of_done>

<implementation_code>
File: crates/context-graph-core/src/clustering/manager.rs

```rust
//! Multi-space cluster manager for coordinating clustering across all 13 spaces.
//!
//! Maintains per-space BIRCH trees for incremental clustering and coordinates
//! HDBSCAN batch clustering operations.
//!
//! # Constitution Compliance
//!
//! - ARCH-02: Each space clustered independently (E1&lt;-&gt;E1, never E1&lt;-&gt;E5)
//! - Progressive activation: real clustering at tier 3 (10+ memories)
//! - HDBSCAN min_cluster_size=3 per constitution

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use crate::embeddings::config::get_dimension;
use crate::memory::Memory;
use crate::teleological::Embedder;
use crate::types::fingerprint::{EmbeddingRef, TeleologicalArray};

use super::birch::{birch_defaults, BIRCHParams, BIRCHTree};
use super::error::ClusterError;
use super::hdbscan::{HDBSCANClusterer, HDBSCANParams};
use super::membership::ClusterMembership;

/// Progressive activation tier thresholds per constitution.
const TIER_THRESHOLDS: [usize; 7] = [0, 1, 3, 10, 30, 100, 500];

/// SPLADE vocabulary size for sparse-&gt;dense conversion.
const SPLADE_VOCAB_SIZE: usize = 30522;

/// E12 token dimension for mean pooling.
const E12_TOKEN_DIM: usize = 128;

/// Manager for clustering across all 13 embedding spaces.
///
/// Coordinates HDBSCAN (batch) and BIRCH (incremental) clustering.
/// Maintains per-space BIRCH trees and tracks cluster memberships.
#[derive(Debug)]
pub struct MultiSpaceClusterManager {
    /// Per-space BIRCH trees for incremental clustering
    birch_trees: HashMap&lt;Embedder, BIRCHTree&gt;,
    /// Cluster memberships per memory (memory_id -&gt; 13 memberships)
    memberships: HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt;,
    /// HDBSCAN params per space
    hdbscan_params: HashMap&lt;Embedder, HDBSCANParams&gt;,
    /// BIRCH params per space
    birch_params: HashMap&lt;Embedder, BIRCHParams&gt;,
    /// Total memory count for tier calculation
    memory_count: usize,
    /// Last recluster time per space
    last_recluster: HashMap&lt;Embedder, DateTime&lt;Utc&gt;&gt;,
}

impl Default for MultiSpaceClusterManager {
    fn default() -> Self {
        Self::new().expect("default MultiSpaceClusterManager should always succeed")
    }
}

impl MultiSpaceClusterManager {
    /// Create a new manager with default per-space parameters.
    ///
    /// Initializes BIRCH trees and parameters for all 13 spaces.
    ///
    /// # Errors
    ///
    /// Returns `ClusterError::InvalidParameter` if BIRCH tree creation fails
    /// (should not happen with default parameters).
    pub fn new() -> Result&lt;Self, ClusterError&gt; {
        let mut birch_trees = HashMap::with_capacity(Embedder::COUNT);
        let mut hdbscan_params = HashMap::with_capacity(Embedder::COUNT);
        let mut birch_params_map = HashMap::with_capacity(Embedder::COUNT);

        for embedder in Embedder::all() {
            // Get space-specific parameters
            let h_params = HDBSCANParams::default_for_space(embedder);
            let b_params = BIRCHParams::default_for_space(embedder);

            // Get dimension for this space
            let dim = get_dimension(embedder);

            // Create BIRCH tree (validates params internally)
            let tree = BIRCHTree::new(b_params.clone(), dim)?;

            birch_trees.insert(embedder, tree);
            hdbscan_params.insert(embedder, h_params);
            birch_params_map.insert(embedder, b_params);
        }

        Ok(Self {
            birch_trees,
            memberships: HashMap::new(),
            hdbscan_params,
            birch_params: birch_params_map,
            memory_count: 0,
            last_recluster: HashMap::new(),
        })
    }

    /// Get current activation tier (0-6).
    ///
    /// Tiers per constitution progressive_tiers:
    /// - 0: 0 memories
    /// - 1: 1-2 memories
    /// - 2: 3-9 memories
    /// - 3: 10-29 memories (real clustering starts)
    /// - 4: 30-99 memories
    /// - 5: 100-499 memories
    /// - 6: 500+ memories
    #[must_use]
    pub fn activation_tier(&amp;self) -> usize {
        for (tier, &amp;threshold) in TIER_THRESHOLDS.iter().enumerate().rev() {
            if self.memory_count >= threshold {
                return tier;
            }
        }
        0
    }

    /// Check if real clustering is active (tier >= 3, memory_count >= 10).
    #[must_use]
    pub fn is_clustering_active(&amp;self) -> bool {
        self.activation_tier() >= 3
    }

    /// Cluster all spaces using HDBSCAN (batch operation).
    ///
    /// For tier 0-2 (&lt;10 memories): Returns all noise memberships.
    /// For tier 3+ (10+ memories): Runs HDBSCAN per space.
    ///
    /// # Arguments
    ///
    /// * `memories` - All memories to cluster
    ///
    /// # Returns
    ///
    /// HashMap mapping each Embedder to Vec of ClusterMembership for all memories.
    ///
    /// # Errors
    ///
    /// Returns ClusterError if embedding extraction or clustering fails.
    pub fn cluster_all_spaces(
        &amp;mut self,
        memories: &amp;[Memory],
    ) -> Result&lt;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;, ClusterError&gt; {
        let mut all_memberships: HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt; =
            HashMap::with_capacity(Embedder::COUNT);

        // Check if we have enough memories for real clustering
        if memories.len() &lt; TIER_THRESHOLDS[3] {
            // Tier 0-2: Return all noise memberships
            for embedder in Embedder::all() {
                let noise: Vec&lt;ClusterMembership&gt; = memories
                    .iter()
                    .map(|m| ClusterMembership::noise(m.id, embedder))
                    .collect();
                all_memberships.insert(embedder, noise);
            }

            // Update stored memberships
            for memory in memories {
                let mut mem_memberships = Vec::with_capacity(Embedder::COUNT);
                for embedder in Embedder::all() {
                    mem_memberships.push(ClusterMembership::noise(memory.id, embedder));
                }
                self.memberships.insert(memory.id, mem_memberships);
            }

            self.memory_count = memories.len();
            return Ok(all_memberships);
        }

        // Tier 3+: Cluster each space with HDBSCAN
        for embedder in Embedder::all() {
            let memberships = self.cluster_space(embedder, memories)?;
            all_memberships.insert(embedder, memberships);
        }

        // Update stored memberships per memory
        for memory in memories {
            let mut mem_memberships = Vec::with_capacity(Embedder::COUNT);
            for embedder in Embedder::all() {
                if let Some(space_memberships) = all_memberships.get(&amp;embedder) {
                    if let Some(m) = space_memberships.iter().find(|m| m.memory_id == memory.id) {
                        mem_memberships.push(m.clone());
                    } else {
                        mem_memberships.push(ClusterMembership::noise(memory.id, embedder));
                    }
                } else {
                    mem_memberships.push(ClusterMembership::noise(memory.id, embedder));
                }
            }
            self.memberships.insert(memory.id, mem_memberships);
        }

        self.memory_count = memories.len();
        Ok(all_memberships)
    }

    /// Cluster a single space using HDBSCAN.
    fn cluster_space(
        &amp;self,
        space: Embedder,
        memories: &amp;[Memory],
    ) -> Result&lt;Vec&lt;ClusterMembership&gt;, ClusterError&gt; {
        let params = self.hdbscan_params.get(&amp;space).ok_or_else(|| {
            ClusterError::SpaceNotInitialized(space)
        })?;

        // Extract embeddings for this space
        let mut embeddings: Vec&lt;Vec&lt;f32&gt;&gt; = Vec::with_capacity(memories.len());
        let mut ids: Vec&lt;Uuid&gt; = Vec::with_capacity(memories.len());

        for memory in memories {
            let embedding = extract_embedding_for_space(&amp;memory.teleological_array, space);
            embeddings.push(embedding);
            ids.push(memory.id);
        }

        // Run HDBSCAN
        let clusterer = HDBSCANClusterer::new(params.clone());
        clusterer.fit(&amp;embeddings, &amp;ids, space)
    }

    /// Insert a memory incrementally using BIRCH trees.
    ///
    /// Updates all 13 BIRCH trees and tracks membership.
    ///
    /// # Arguments
    ///
    /// * `memory` - The memory to insert
    ///
    /// # Errors
    ///
    /// Returns ClusterError if embedding extraction or BIRCH insertion fails.
    pub fn insert_memory(&amp;mut self, memory: &amp;Memory) -> Result&lt;(), ClusterError&gt; {
        let mut memberships = Vec::with_capacity(Embedder::COUNT);

        for embedder in Embedder::all() {
            let embedding = extract_embedding_for_space(&amp;memory.teleological_array, embedder);

            let tree = self.birch_trees.get_mut(&amp;embedder).ok_or_else(|| {
                ClusterError::SpaceNotInitialized(embedder)
            })?;

            // Insert into BIRCH tree (returns cluster index)
            let cluster_idx = tree.insert(&amp;embedding, memory.id)?;

            // Create membership
            let membership = ClusterMembership::new(
                memory.id,
                embedder,
                cluster_idx as i32,
                0.8, // Default probability for incremental (BIRCH doesn't compute this)
                false, // Core point determination requires HDBSCAN analysis
            );

            memberships.push(membership);
        }

        self.memberships.insert(memory.id, memberships);
        self.memory_count += 1;

        Ok(())
    }

    /// Recluster a single space using HDBSCAN.
    ///
    /// Rebuilds the BIRCH tree for that space after reclustering.
    ///
    /// # Arguments
    ///
    /// * `space` - The embedding space to recluster
    /// * `memories` - All memories (need their embeddings for this space)
    ///
    /// # Errors
    ///
    /// Returns ClusterError if clustering or BIRCH rebuild fails.
    pub fn recluster_space(
        &amp;mut self,
        space: Embedder,
        memories: &amp;[Memory],
    ) -> Result&lt;(), ClusterError&gt; {
        if memories.is_empty() {
            return Ok(());
        }

        // Run HDBSCAN for this space
        let new_memberships = self.cluster_space(space, memories)?;

        // Update stored memberships
        for membership in new_memberships {
            if let Some(mem_list) = self.memberships.get_mut(&amp;membership.memory_id) {
                // Replace existing membership for this space
                if let Some(idx) = mem_list.iter().position(|m| m.space == space) {
                    mem_list[idx] = membership;
                } else {
                    mem_list.push(membership);
                }
            }
        }

        // Rebuild BIRCH tree for this space
        let dim = get_dimension(space);
        let params = self
            .birch_params
            .get(&amp;space)
            .cloned()
            .unwrap_or_else(birch_defaults);

        let mut new_tree = BIRCHTree::new(params, dim)?;

        for memory in memories {
            let embedding = extract_embedding_for_space(&amp;memory.teleological_array, space);
            new_tree.insert(&amp;embedding, memory.id)?;
        }

        self.birch_trees.insert(space, new_tree);
        self.last_recluster.insert(space, Utc::now());

        Ok(())
    }

    /// Get memberships for a memory.
    #[must_use]
    pub fn get_memberships(&amp;self, memory_id: &amp;Uuid) -> Option&lt;&amp;Vec&lt;ClusterMembership&gt;&gt; {
        self.memberships.get(memory_id)
    }

    /// Get all memberships.
    #[must_use]
    pub fn get_all_memberships(&amp;self) -> &amp;HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt; {
        &amp;self.memberships
    }

    /// Check if space should be reclustered.
    ///
    /// Returns true if:
    /// - Never reclustered AND memory_count >= 10
    /// - 24+ hours since last recluster
    /// - BIRCH cluster count error > 15% from expected
    #[must_use]
    pub fn should_recluster(&amp;self, space: Embedder) -> bool {
        // Don't recluster if &lt; 10 memories
        if self.memory_count &lt; TIER_THRESHOLDS[3] {
            return false;
        }

        // Check if never reclustered
        let last = match self.last_recluster.get(&amp;space) {
            Some(t) => t,
            None => return true, // Never reclustered, should do it
        };

        // Check time since last recluster (24 hour threshold)
        let elapsed = Utc::now() - *last;
        if elapsed >= Duration::hours(24) {
            return true;
        }

        // Check BIRCH cluster count error
        if let Some(tree) = self.birch_trees.get(&amp;space) {
            let cluster_count = tree.cluster_count();
            // Expected clusters roughly = memory_count / 10
            let expected = (self.memory_count as f32 / 10.0).ceil() as usize;
            if expected > 0 {
                let error =
                    (cluster_count as f32 - expected as f32).abs() / expected as f32;
                if error > 0.15 {
                    return true;
                }
            }
        }

        false
    }

    /// Get cluster for a memory in a specific space.
    #[must_use]
    pub fn get_cluster_for_memory(&amp;self, memory_id: &amp;Uuid, space: Embedder) -> Option&lt;i32&gt; {
        self.memberships
            .get(memory_id)?
            .iter()
            .find(|m| m.space == space)
            .map(|m| m.cluster_id)
    }

    /// Get memory count.
    #[must_use]
    pub fn memory_count(&amp;self) -> usize {
        self.memory_count
    }

    /// Get BIRCH cluster count for a space.
    #[must_use]
    pub fn cluster_count(&amp;self, space: Embedder) -> usize {
        self.birch_trees
            .get(&amp;space)
            .map(|t| t.cluster_count())
            .unwrap_or(0)
    }
}

/// Extract embedding vector for a specific space from TeleologicalArray.
///
/// Handles all three embedding types:
/// - Dense (E1-E5, E7-E11): Direct clone
/// - Sparse (E6, E13): Convert to dense using SPLADE vocab size
/// - TokenLevel (E12): Mean pooling across tokens
///
/// # Arguments
///
/// * `array` - The TeleologicalArray containing all 13 embeddings
/// * `space` - The embedding space to extract
///
/// # Returns
///
/// Dense Vec&lt;f32&gt; suitable for clustering algorithms.
#[must_use]
pub fn extract_embedding_for_space(array: &amp;TeleologicalArray, space: Embedder) -> Vec&lt;f32&gt; {
    match array.get(space) {
        EmbeddingRef::Dense(slice) => slice.to_vec(),
        EmbeddingRef::Sparse(sparse_vec) => {
            // Convert sparse to dense for clustering
            sparse_vec.to_dense(SPLADE_VOCAB_SIZE)
        }
        EmbeddingRef::TokenLevel(tokens) => {
            // Mean pooling for E12 ColBERT
            if tokens.is_empty() {
                return vec![0.0; E12_TOKEN_DIM];
            }

            let n = tokens.len() as f32;
            let dim = tokens.first().map(|t| t.len()).unwrap_or(E12_TOKEN_DIM);
            let mut mean = vec![0.0; dim];

            for token in tokens {
                for (i, &amp;v) in token.iter().enumerate() {
                    if i &lt; mean.len() {
                        mean[i] += v;
                    }
                }
            }

            mean.iter_mut().for_each(|v| *v /= n);
            mean
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let result = MultiSpaceClusterManager::new();
        assert!(result.is_ok(), "Manager creation should succeed");

        let manager = result.expect("already checked is_ok");
        assert_eq!(manager.memory_count(), 0);
        assert_eq!(manager.activation_tier(), 0);
        assert!(!manager.is_clustering_active());

        // Verify all 13 spaces initialized
        for embedder in Embedder::all() {
            assert!(
                manager.birch_trees.contains_key(&amp;embedder),
                "BIRCH tree missing for {:?}",
                embedder
            );
            assert!(
                manager.hdbscan_params.contains_key(&amp;embedder),
                "HDBSCAN params missing for {:?}",
                embedder
            );
        }

        println!("[PASS] Manager creation with all 13 spaces");
    }

    #[test]
    fn test_activation_tiers() {
        let mut manager = MultiSpaceClusterManager::new().expect("manager creation");

        // Tier 0: 0 memories
        assert_eq!(manager.activation_tier(), 0);
        assert!(!manager.is_clustering_active());

        // Simulate memory counts
        manager.memory_count = 1;
        assert_eq!(manager.activation_tier(), 1);

        manager.memory_count = 3;
        assert_eq!(manager.activation_tier(), 2);

        manager.memory_count = 10;
        assert_eq!(manager.activation_tier(), 3);
        assert!(manager.is_clustering_active());

        manager.memory_count = 100;
        assert_eq!(manager.activation_tier(), 5);

        manager.memory_count = 500;
        assert_eq!(manager.activation_tier(), 6);

        println!("[PASS] Activation tier calculation");
    }

    #[test]
    fn test_should_recluster_insufficient_memories() {
        let manager = MultiSpaceClusterManager::new().expect("manager creation");

        // Should not recluster with &lt; 10 memories
        for embedder in Embedder::all() {
            assert!(
                !manager.should_recluster(embedder),
                "Should not recluster {:?} with 0 memories",
                embedder
            );
        }

        println!("[PASS] should_recluster returns false for insufficient memories");
    }

    #[test]
    fn test_should_recluster_never_reclustered() {
        let mut manager = MultiSpaceClusterManager::new().expect("manager creation");
        manager.memory_count = 10; // Tier 3

        // Never reclustered, so should recluster
        for embedder in Embedder::all() {
            assert!(
                manager.should_recluster(embedder),
                "Should recluster {:?} when never reclustered",
                embedder
            );
        }

        println!("[PASS] should_recluster returns true when never reclustered");
    }

    #[test]
    fn test_cluster_all_spaces_empty() {
        let mut manager = MultiSpaceClusterManager::new().expect("manager creation");
        let memories: Vec&lt;Memory&gt; = vec![];

        let result = manager.cluster_all_spaces(&amp;memories);
        assert!(result.is_ok());

        let all_memberships = result.expect("already checked is_ok");
        assert_eq!(all_memberships.len(), Embedder::COUNT);

        // All spaces should have empty results
        for embedder in Embedder::all() {
            let memberships = all_memberships.get(&amp;embedder).expect("space should exist");
            assert!(memberships.is_empty());
        }

        println!("[PASS] cluster_all_spaces with empty input");
    }

    #[test]
    fn test_get_memberships_unknown_id() {
        let manager = MultiSpaceClusterManager::new().expect("manager creation");
        let unknown_id = Uuid::new_v4();

        assert!(manager.get_memberships(&amp;unknown_id).is_none());
        assert!(manager.get_cluster_for_memory(&amp;unknown_id, Embedder::Semantic).is_none());

        println!("[PASS] get_memberships returns None for unknown ID");
    }

    #[test]
    fn test_cluster_count_empty() {
        let manager = MultiSpaceClusterManager::new().expect("manager creation");

        for embedder in Embedder::all() {
            let count = manager.cluster_count(embedder);
            assert_eq!(count, 0, "Empty tree should have 0 clusters for {:?}", embedder);
        }

        println!("[PASS] cluster_count returns 0 for empty trees");
    }

    #[test]
    fn test_default_impl() {
        let manager = MultiSpaceClusterManager::default();
        assert_eq!(manager.memory_count(), 0);
        assert_eq!(manager.activation_tier(), 0);

        println!("[PASS] Default impl works");
    }
}
```
</implementation_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/manager.rs">MultiSpaceClusterManager implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs" action="add_module_and_exports">
Add after line 34:
```rust
pub mod manager;
```

Add to pub use section:
```rust
pub use manager::{extract_embedding_for_space, MultiSpaceClusterManager};
```
  </file>
</files_to_modify>

<exact_imports>
Add these imports at the top of manager.rs:
```rust
use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use crate::embeddings::config::get_dimension;
use crate::memory::Memory;
use crate::teleological::Embedder;
use crate::types::fingerprint::{EmbeddingRef, TeleologicalArray};

use super::birch::{birch_defaults, BIRCHParams, BIRCHTree};
use super::error::ClusterError;
use super::hdbscan::{HDBSCANClusterer, HDBSCANParams};
use super::membership::ClusterMembership;
```
</exact_imports>

<full_state_verification>
  <source_of_truth>
    <item>constitution.yaml: Clustering defaults, progressive tiers, embedder categories</item>
    <item>birch.rs: BIRCHTree API (new, insert, cluster_count, get_cluster_members)</item>
    <item>hdbscan.rs: HDBSCANClusterer::fit() signature</item>
    <item>membership.rs: ClusterMembership::noise() constructor</item>
    <item>fingerprint.rs: TeleologicalArray::get(Embedder) -&gt; EmbeddingRef</item>
    <item>embedder.rs: Embedder::all() iterator, Embedder::COUNT</item>
    <item>config.rs: get_dimension(Embedder) function</item>
  </source_of_truth>

  <execute_and_inspect>
    After implementation, run these commands and VERIFY output:

    1. cargo check --package context-graph-core
       EXPECTED: No errors, no warnings about unused code

    2. cargo test --package context-graph-core manager -- --nocapture
       EXPECTED: All tests pass, [PASS] lines visible

    3. cargo test --package context-graph-core clustering -- --nocapture
       EXPECTED: All clustering tests pass

    4. cargo clippy --package context-graph-core -- -D warnings
       EXPECTED: No warnings in manager.rs
  </execute_and_inspect>

  <boundary_and_edge_case_audit>
    <case name="empty_memories">
      Input: cluster_all_spaces(&amp;[])
      Before: manager.memory_count=0, memberships empty
      Expected After: Returns HashMap with 13 empty Vecs, memory_count=0
      Verification: Check all_memberships.len() == 13, each Vec is empty
    </case>
    <case name="below_tier_3">
      Input: cluster_all_spaces with 5 memories
      Before: manager.memory_count=0
      Expected After: All memberships are NOISE (cluster_id=-1), memory_count=5
      Verification: For each membership, assert is_noise() == true
    </case>
    <case name="tier_3_active">
      Input: cluster_all_spaces with 15 memories
      Before: manager.memory_count=0
      Expected After: HDBSCAN runs, real clusters formed, memory_count=15
      Verification: At least some memberships have cluster_id >= 0
    </case>
    <case name="insert_single_memory">
      Input: insert_memory with valid Memory
      Before: memory_count=0, memberships empty
      Expected After: memory_count=1, memberships has 1 entry with 13 memberships
      Verification: get_memberships(id).len() == 13
    </case>
    <case name="should_recluster_time_based">
      Input: Set last_recluster to 25 hours ago, memory_count=15
      Expected: should_recluster returns true
      Verification: Assert should_recluster(Embedder::Semantic) == true
    </case>
  </boundary_and_edge_case_audit>

  <evidence_of_success>
    - All cargo test commands pass
    - cargo clippy shows no warnings
    - All 13 spaces have BIRCH trees after new()
    - insert_memory creates exactly 13 memberships per memory
    - cluster_all_spaces respects progressive activation tiers
    - get_memberships returns correct memberships after operations
    - Memory ID tracking is verifiable via get_all_memberships()
  </evidence_of_success>
</full_state_verification>

<manual_verification_protocol>
  <synthetic_test_data>
    Create a comprehensive integration test with KNOWN expected outputs:

    ```rust
    // File: crates/context-graph-core/tests/manager_manual_test.rs

    //! Manual verification tests for MultiSpaceClusterManager.
    //!
    //! These tests verify actual behavior, NOT mocks.
    //! Run with: cargo test --package context-graph-core --test manager_manual_test -- --nocapture

    use context_graph_core::clustering::{extract_embedding_for_space, MultiSpaceClusterManager};
    use context_graph_core::memory::{Memory, MemorySource, HookType};
    use context_graph_core::teleological::Embedder;
    use context_graph_core::types::fingerprint::{TeleologicalArray, SparseVector};
    use uuid::Uuid;
    use chrono::Utc;

    /// Create a test memory with a specific pattern
    fn create_test_memory(id: Uuid, pattern: f32) -> Memory {
        // Create TeleologicalArray with predictable values
        let array = TeleologicalArray {
            e1_semantic: vec![pattern; 1024],
            e2_temporal_recent: vec![pattern * 0.5; 512],
            e3_temporal_periodic: vec![pattern * 0.3; 512],
            e4_temporal_positional: vec![pattern * 0.2; 512],
            e5_causal: vec![pattern; 768],
            e6_sparse: SparseVector::empty(),
            e7_code: vec![pattern; 1536],
            e8_graph: vec![pattern; 384],
            e9_hdc: vec![pattern; 1024],
            e10_multimodal: vec![pattern; 768],
            e11_entity: vec![pattern; 384],
            e12_late_interaction: vec![vec![pattern; 128]; 5], // 5 tokens
            e13_splade: SparseVector::empty(),
        };

        Memory {
            id,
            content: format!("Test memory with pattern {}", pattern),
            source: MemorySource::HookDescription {
                hook_type: HookType::PostToolUse,
                tool_name: Some("test_tool".to_string()),
            },
            created_at: Utc::now(),
            session_id: "test_session".to_string(),
            teleological_array: array,
            chunk_metadata: None,
            word_count: 10,
        }
    }

    #[test]
    fn test_full_clustering_workflow() {
        println!("\n=== MANUAL VERIFICATION: Full Clustering Workflow ===\n");

        // STEP 1: Create manager
        let mut manager = MultiSpaceClusterManager::new()
            .expect("Manager creation should succeed");

        println!("[STATE BEFORE] memory_count: {}", manager.memory_count());
        println!("[STATE BEFORE] activation_tier: {}", manager.activation_tier());
        println!("[STATE BEFORE] is_clustering_active: {}", manager.is_clustering_active());
        assert_eq!(manager.memory_count(), 0);
        assert_eq!(manager.activation_tier(), 0);

        // STEP 2: Create test memories (15 to reach tier 3)
        let memories: Vec&lt;Memory&gt; = (0..15)
            .map(|i| {
                let id = Uuid::from_u128(i as u128);
                let pattern = (i as f32) / 10.0;
                create_test_memory(id, pattern)
            })
            .collect();

        println!("\n[INPUT] Created {} test memories", memories.len());

        // STEP 3: Cluster all spaces
        let result = manager.cluster_all_spaces(&amp;memories);
        assert!(result.is_ok(), "Clustering should succeed");

        let all_memberships = result.expect("already checked");

        println!("\n[STATE AFTER] memory_count: {}", manager.memory_count());
        println!("[STATE AFTER] activation_tier: {}", manager.activation_tier());
        println!("[STATE AFTER] is_clustering_active: {}", manager.is_clustering_active());

        // VERIFY: Correct memory count
        assert_eq!(manager.memory_count(), 15);
        assert_eq!(manager.activation_tier(), 3);
        assert!(manager.is_clustering_active());

        // VERIFY: All 13 spaces have results
        assert_eq!(all_memberships.len(), 13);
        println!("\n[VERIFY] All 13 spaces have clustering results");

        // VERIFY: Each space has 15 memberships
        for embedder in Embedder::all() {
            let memberships = all_memberships.get(&amp;embedder).expect("space exists");
            println!(
                "[VERIFY] {:?}: {} memberships, {} non-noise",
                embedder,
                memberships.len(),
                memberships.iter().filter(|m| !m.is_noise()).count()
            );
            assert_eq!(memberships.len(), 15, "Each space should have 15 memberships");
        }

        // VERIFY: Stored memberships accessible
        for memory in &amp;memories {
            let stored = manager.get_memberships(&amp;memory.id);
            assert!(stored.is_some(), "Membership should be stored for {}", memory.id);
            assert_eq!(stored.unwrap().len(), 13, "Should have 13 memberships per memory");
        }

        println!("\n[PASS] Full clustering workflow verified");
    }

    #[test]
    fn test_incremental_insertion() {
        println!("\n=== MANUAL VERIFICATION: Incremental Insertion ===\n");

        let mut manager = MultiSpaceClusterManager::new().expect("manager");

        // Insert 5 memories one at a time
        for i in 0..5 {
            let id = Uuid::from_u128((100 + i) as u128);
            let memory = create_test_memory(id, i as f32 / 5.0);

            println!("[BEFORE INSERT {}] memory_count: {}", i, manager.memory_count());

            let result = manager.insert_memory(&amp;memory);
            assert!(result.is_ok(), "Insert {} should succeed", i);

            println!("[AFTER INSERT {}] memory_count: {}", i, manager.memory_count());

            // VERIFY: Memory was tracked
            let memberships = manager.get_memberships(&amp;id);
            assert!(memberships.is_some());
            assert_eq!(memberships.unwrap().len(), 13);

            // VERIFY: BIRCH trees updated
            for embedder in Embedder::all() {
                let count = manager.cluster_count(embedder);
                println!(
                    "  {:?} cluster_count: {}",
                    embedder, count
                );
            }
        }

        assert_eq!(manager.memory_count(), 5);
        println!("\n[PASS] Incremental insertion verified");
    }

    #[test]
    fn test_extract_embedding_for_space() {
        println!("\n=== MANUAL VERIFICATION: Embedding Extraction ===\n");

        let memory = create_test_memory(Uuid::new_v4(), 1.0);

        // Verify each space extraction
        for embedder in Embedder::all() {
            let embedding = extract_embedding_for_space(&amp;memory.teleological_array, embedder);

            println!(
                "[VERIFY] {:?}: extracted {} dimensions",
                embedder,
                embedding.len()
            );

            // Check dimension is non-zero
            assert!(!embedding.is_empty(), "{:?} embedding should not be empty", embedder);

            // Check no NaN/Infinity
            for (i, &amp;v) in embedding.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "{:?}[{}] should be finite, got {}",
                    embedder, i, v
                );
            }
        }

        // Specific dimension checks
        let e1 = extract_embedding_for_space(&amp;memory.teleological_array, Embedder::Semantic);
        assert_eq!(e1.len(), 1024, "E1 should be 1024D");

        let e12 = extract_embedding_for_space(&amp;memory.teleological_array, Embedder::LateInteraction);
        assert_eq!(e12.len(), 128, "E12 mean pooled should be 128D");

        println!("\n[PASS] Embedding extraction verified");
    }

    #[test]
    fn test_tier_transitions() {
        println!("\n=== MANUAL VERIFICATION: Tier Transitions ===\n");

        let mut manager = MultiSpaceClusterManager::new().expect("manager");

        // Test each tier boundary
        let test_counts = [0, 1, 2, 3, 9, 10, 29, 30, 99, 100, 499, 500, 1000];

        for &amp;count in &amp;test_counts {
            manager.memory_count = count;
            let tier = manager.activation_tier();
            let active = manager.is_clustering_active();

            println!(
                "[VERIFY] memory_count={:4} -&gt; tier={}, clustering_active={}",
                count, tier, active
            );

            // Verify tier boundaries per constitution
            match count {
                0 =&gt; assert_eq!(tier, 0),
                1..=2 =&gt; assert_eq!(tier, 1),
                3..=9 =&gt; assert_eq!(tier, 2),
                10..=29 =&gt; { assert_eq!(tier, 3); assert!(active); }
                30..=99 =&gt; { assert_eq!(tier, 4); assert!(active); }
                100..=499 =&gt; { assert_eq!(tier, 5); assert!(active); }
                500.. =&gt; { assert_eq!(tier, 6); assert!(active); }
                _ =&gt; {}
            }
        }

        println!("\n[PASS] Tier transitions verified");
    }
    ```
  </synthetic_test_data>

  <physical_output_verification>
    After running tests, verify these conditions:

    1. Memory IDs are ACTUALLY stored (not just counted):
       - get_memberships(id) returns actual memberships with correct memory_id
       - get_all_memberships() size equals memory_count

    2. Cluster assignments are valid:
       - cluster_id is either -1 (noise) or &gt;= 0
       - membership_probability is in [0.0, 1.0]
       - space field matches the embedder it was computed for

    3. BIRCH tree state is correct:
       - cluster_count(space) returns actual cluster count
       - Inserting same memory twice doesn't create duplicate memberships

    4. Progressive activation enforced:
       - tier 0-2: All noise memberships (cluster_id = -1)
       - tier 3+: HDBSCAN runs, some clusters may form
  </physical_output_verification>
</manual_verification_protocol>

<implementation_requirements>
  <no_backwards_compatibility>
    This is a NEW implementation. Do NOT:
    - Add compatibility shims for non-existent code
    - Create migration paths (nothing to migrate from)
    - Add deprecated method aliases
    - Reference MemoryStore methods that don't exist (like get_all)
  </no_backwards_compatibility>

  <fail_fast_error_handling>
    REQUIRED error handling pattern:
    - Never .unwrap() - use .expect("context") or propagate with ?
    - Return ClusterError for all failure cases
    - Log errors with sufficient context for debugging
  </fail_fast_error_handling>

  <no_mock_data_in_tests>
    Tests MUST use real data structures:
    - Create actual TeleologicalArray with realistic dimensions
    - Use actual Memory structs, not mocks
    - Verify actual outputs, not call counts
  </no_mock_data_in_tests>

  <synchronous_api>
    MemoryStore is SYNCHRONOUS. Do NOT:
    - Add async fn signatures
    - Use .await
    - For async contexts, use tokio::spawn_blocking()
  </synchronous_api>
</implementation_requirements>

<validation_criteria>
  <criterion>cargo check --package context-graph-core succeeds</criterion>
  <criterion>cargo test --package context-graph-core manager -- --nocapture shows [PASS] for all tests</criterion>
  <criterion>cargo clippy --package context-graph-core shows no warnings</criterion>
  <criterion>All 13 spaces have BIRCH trees initialized in new()</criterion>
  <criterion>cluster_all_spaces returns HashMap with 13 entries</criterion>
  <criterion>insert_memory creates exactly 13 memberships</criterion>
  <criterion>Progressive activation tiers correctly computed</criterion>
  <criterion>extract_embedding_for_space handles Dense, Sparse, TokenLevel</criterion>
  <criterion>No .unwrap() in library code</criterion>
</validation_criteria>

<test_commands>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run manager tests">cargo test --package context-graph-core manager -- --nocapture</command>
  <command description="Run integration test">cargo test --package context-graph-core --test manager_manual_test -- --nocapture</command>
  <command description="Run clippy">cargo clippy --package context-graph-core -- -D warnings</command>
  <command description="Generate docs">cargo doc --package context-graph-core --no-deps</command>
</test_commands>

<notes>
  <note category="architecture">
    MultiSpaceClusterManager clusters ALL 13 spaces. Topic weighting (SEMANTIC=1.0,
    TEMPORAL=0.0, RELATIONAL=0.5, STRUCTURAL=0.5) is applied by TopicSynthesizer
    (TASK-P4-008), NOT by this manager. Per ARCH-02, each space is clustered
    independently without cross-space comparison.
  </note>
  <note category="sparse_vectors">
    E6 (Sparse) and E13 (KeywordSplade) use SparseVector. The extract_embedding_for_space
    function converts these to dense Vec&lt;f32&gt; using vocab size 30522. This allows
    uniform treatment by BIRCH/HDBSCAN which expect dense vectors.
  </note>
  <note category="token_level">
    E12 (LateInteraction/ColBERT) stores per-token embeddings (Vec&lt;Vec&lt;f32&gt;&gt;).
    The extract function uses mean pooling to produce a single 128D vector.
    This is a simplification; more sophisticated aggregation could be added.
  </note>
  <note category="thread_safety">
    MultiSpaceClusterManager is NOT thread-safe. For concurrent access, wrap in
    Arc&lt;RwLock&lt;MultiSpaceClusterManager&gt;&gt;. This matches the pattern used by
    other components in the codebase.
  </note>
  <note category="memory_store">
    MemoryStore.get_all() does NOT exist. The manager expects memories to be
    passed in externally. If iterating all memories is needed, caller should
    implement their own iteration over sessions or add get_all() to MemoryStore.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Read birch.rs to verify BIRCHTree API: new(params, dim), insert(&amp;embedding, id), cluster_count()
- [ ] Read hdbscan.rs to verify HDBSCANClusterer::fit() signature
- [ ] Read fingerprint.rs to verify TeleologicalArray::get(Embedder) -&gt; EmbeddingRef
- [ ] Read embedder.rs to verify Embedder::all() and Embedder::COUNT
- [ ] Read config.rs to verify get_dimension(Embedder) function
- [ ] Create manager.rs file
- [ ] Implement MultiSpaceClusterManager struct with all fields
- [ ] Implement new() with per-space initialization
- [ ] Implement activation_tier() and is_clustering_active()
- [ ] Implement cluster_all_spaces() with tier check
- [ ] Implement cluster_space() helper
- [ ] Implement insert_memory() with all 13 BIRCH updates
- [ ] Implement recluster_space()
- [ ] Implement should_recluster() with time/error check
- [ ] Implement extract_embedding_for_space() helper
- [ ] Write unit tests with [PASS] output
- [ ] Update mod.rs to export manager module
- [ ] Run: `cargo check --package context-graph-core`
- [ ] Run: `cargo test --package context-graph-core manager -- --nocapture`
- [ ] Verify [PASS] output for all tests
- [ ] Run: `cargo clippy --package context-graph-core -- -D warnings`
- [ ] Create integration test file (manager_manual_test.rs)
- [ ] Run integration tests and verify outputs
- [ ] Proceed to TASK-P4-008

## Dependencies Verified

| Task | Status | Verification |
|------|--------|--------------|
| TASK-P4-005 | COMPLETE | HDBSCANClusterer::fit() exists in hdbscan.rs |
| TASK-P4-006 | COMPLETE | BIRCHTree with insert(), cluster_count() exists in birch.rs |
| TASK-P1-005 | COMPLETE | MemoryStore exists (synchronous) in memory/store.rs |
| TASK-P4-001 | COMPLETE | ClusterMembership with noise() exists in membership.rs |
