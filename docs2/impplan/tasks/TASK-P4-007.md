# TASK-P4-007: MultiSpaceClusterManager

```xml
<task_spec id="TASK-P4-007" version="1.0">
<metadata>
  <title>MultiSpaceClusterManager Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>33</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-02</requirement_ref>
    <requirement_ref>REQ-P4-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P4-005</task_ref>
    <task_ref>TASK-P4-006</task_ref>
    <task_ref>TASK-P1-005</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
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

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#component_contracts</file>
  <file purpose="hdbscan">crates/context-graph-core/src/clustering/hdbscan.rs</file>
  <file purpose="birch">crates/context-graph-core/src/clustering/birch.rs</file>
  <file purpose="memory_store">crates/context-graph-core/src/memory/store.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-005 complete (HDBSCANClusterer exists)</check>
  <check>TASK-P4-006 complete (BIRCHTree exists)</check>
  <check>TASK-P1-005 complete (MemoryStore exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create MultiSpaceClusterManager struct
    - Maintain per-space BIRCH trees
    - Implement cluster_all_spaces batch method
    - Implement insert_memory for incremental updates
    - Implement recluster_space for single space
    - Track ClusterMemberships per memory
    - Progressive activation based on memory count
  </in_scope>
  <out_of_scope>
    - Topic synthesis (TASK-P4-008)
    - Persistence of cluster state
    - Distributed clustering
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/manager.rs">
      pub struct MultiSpaceClusterManager {
          store: Arc&lt;MemoryStore&gt;,
          birch_trees: HashMap&lt;Embedder, BIRCHTree&gt;,
          memberships: HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt;,
          hdbscan_params: HashMap&lt;Embedder, HDBSCANParams&gt;,
          birch_params: HashMap&lt;Embedder, BIRCHParams&gt;,
      }

      impl MultiSpaceClusterManager {
          pub fn new(store: Arc&lt;MemoryStore&gt;) -> Self;
          pub async fn cluster_all_spaces(&amp;mut self, memories: &amp;[Memory]) -> Result&lt;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;, ClusterError&gt;;
          pub async fn insert_memory(&amp;mut self, memory: &amp;Memory) -> Result&lt;(), ClusterError&gt;;
          pub async fn recluster_space(&amp;mut self, space: Embedder) -> Result&lt;(), ClusterError&gt;;
          pub fn get_memberships(&amp;self, memory_id: &amp;Uuid) -> Option&lt;&amp;Vec&lt;ClusterMembership&gt;&gt;;
          pub fn get_all_memberships(&amp;self) -> &amp;HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt;;
          pub fn should_recluster(&amp;self, space: Embedder) -> bool;
          fn activation_tier(&amp;self) -> usize;
      }
    </signature>
  </signatures>

  <constraints>
    - Tier 0-2 (0-9 memories): No real clustering, all noise
    - Tier 3+ (10+ memories): Real clustering active
    - BIRCH used for incremental, HDBSCAN for batch
    - Each space clustered independently
    - All 13 spaces clustered; topic weighting applied in TopicSynthesizer
  </constraints>

  <verification>
    - cluster_all_spaces clusters all 13 spaces
    - insert_memory updates BIRCH trees
    - Memberships stored per memory
    - Progressive activation tiers respected
    - recluster_space rebuilds single space
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/manager.rs

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use crate::embedding::{Embedder, TeleologicalArray};
use crate::embedding::config::get_dimension;
use crate::memory::{Memory, MemoryStore};
use super::hdbscan::{HDBSCANClusterer, HDBSCANParams};
use super::birch::{BIRCHTree, BIRCHParams};
use super::membership::ClusterMembership;
use super::error::ClusterError;

/// Progressive activation tiers
const TIER_THRESHOLD: [usize; 7] = [0, 1, 3, 10, 30, 100, 500];

/// Manager for clustering across all 13 embedding spaces
pub struct MultiSpaceClusterManager {
    store: Arc&lt;MemoryStore&gt;,
    /// Per-space BIRCH trees for incremental clustering
    birch_trees: HashMap&lt;Embedder, BIRCHTree&gt;,
    /// Cluster memberships per memory
    memberships: HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt;,
    /// HDBSCAN params per space
    hdbscan_params: HashMap&lt;Embedder, HDBSCANParams&gt;,
    /// BIRCH params per space
    birch_params: HashMap&lt;Embedder, BIRCHParams&gt;,
    /// Total memory count
    memory_count: usize,
    /// Last recluster time per space
    last_recluster: HashMap&lt;Embedder, chrono::DateTime&lt;chrono::Utc&gt;&gt;,
}

impl MultiSpaceClusterManager {
    /// Create a new manager
    pub fn new(store: Arc&lt;MemoryStore&gt;) -> Self {
        let mut hdbscan_params = HashMap::new();
        let mut birch_params = HashMap::new();
        let mut birch_trees = HashMap::new();

        // Initialize per-space configurations
        for embedder in Embedder::all() {
            hdbscan_params.insert(embedder, HDBSCANParams::default_for_space(embedder));
            birch_params.insert(embedder, BIRCHParams::default_for_space(embedder));

            let dim = get_dimension(embedder);
            let params = BIRCHParams::default_for_space(embedder);
            birch_trees.insert(embedder, BIRCHTree::new(params, dim));
        }

        Self {
            store,
            birch_trees,
            memberships: HashMap::new(),
            hdbscan_params,
            birch_params,
            memory_count: 0,
            last_recluster: HashMap::new(),
        }
    }

    /// Get current activation tier
    pub fn activation_tier(&amp;self) -> usize {
        for (tier, &amp;threshold) in TIER_THRESHOLD.iter().enumerate().rev() {
            if self.memory_count >= threshold {
                return tier;
            }
        }
        0
    }

    /// Check if clustering is active
    pub fn is_clustering_active(&amp;self) -> bool {
        self.activation_tier() >= 3
    }

    /// Cluster all spaces using HDBSCAN (batch operation)
    pub async fn cluster_all_spaces(
        &amp;mut self,
        memories: &amp;[Memory],
    ) -> Result&lt;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;, ClusterError&gt; {
        let mut all_memberships: HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt; = HashMap::new();

        // Check if we have enough memories
        if memories.len() < 3 {
            // Return all noise for insufficient data
            for embedder in Embedder::all() {
                let noise: Vec&lt;ClusterMembership&gt; = memories
                    .iter()
                    .map(|m| ClusterMembership::noise(m.id, embedder))
                    .collect();
                all_memberships.insert(embedder, noise);
            }
            return Ok(all_memberships);
        }

        // Cluster each space
        for embedder in Embedder::all() {
            let memberships = self.cluster_space(embedder, memories)?;
            all_memberships.insert(embedder, memberships);
        }

        // Update stored memberships
        for memory in memories {
            let mut mem_memberships = Vec::with_capacity(13);
            for embedder in Embedder::all() {
                if let Some(space_memberships) = all_memberships.get(&amp;embedder) {
                    if let Some(m) = space_memberships.iter().find(|m| m.memory_id == memory.id) {
                        mem_memberships.push(m.clone());
                    }
                }
            }
            self.memberships.insert(memory.id, mem_memberships);
        }

        self.memory_count = memories.len();
        Ok(all_memberships)
    }

    /// Cluster a single space
    fn cluster_space(
        &amp;mut self,
        space: Embedder,
        memories: &amp;[Memory],
    ) -> Result&lt;Vec&lt;ClusterMembership&gt;, ClusterError&gt; {
        let params = self.hdbscan_params.get(&amp;space)
            .ok_or(ClusterError::SpaceNotInitialized(space))?;

        // Extract embeddings for this space
        let (embeddings, ids): (Vec&lt;Vec&lt;f32&gt;&gt;, Vec&lt;Uuid&gt;) = memories
            .iter()
            .map(|m| {
                let embedding = extract_embedding_for_space(&amp;m.teleological_array, space);
                (embedding, m.id)
            })
            .unzip();

        // Run HDBSCAN
        let clusterer = HDBSCANClusterer::new(params.clone());
        clusterer.fit(&amp;embeddings, &amp;ids, space)
    }

    /// Insert a memory incrementally using BIRCH
    pub async fn insert_memory(&amp;mut self, memory: &amp;Memory) -> Result&lt;(), ClusterError&gt; {
        let mut memberships = Vec::with_capacity(13);

        for embedder in Embedder::all() {
            let embedding = extract_embedding_for_space(&amp;memory.teleological_array, embedder);

            let tree = self.birch_trees.get_mut(&amp;embedder)
                .ok_or(ClusterError::SpaceNotInitialized(embedder))?;

            // Insert into BIRCH tree
            let cluster_idx = tree.insert(&amp;embedding, memory.id)?;

            // Create membership
            let membership = ClusterMembership::new(
                memory.id,
                embedder,
                cluster_idx as i32,
                0.8, // Default probability for incremental
                false, // Core point determination requires more analysis
            );

            memberships.push(membership);
        }

        self.memberships.insert(memory.id, memberships);
        self.memory_count += 1;

        Ok(())
    }

    /// Recluster a single space using HDBSCAN
    pub async fn recluster_space(&amp;mut self, space: Embedder) -> Result&lt;(), ClusterError&gt; {
        // Load all memories
        let memories = self.store.get_all().await?;

        if memories.is_empty() {
            return Ok(());
        }

        // Run HDBSCAN for this space
        let new_memberships = self.cluster_space(space, &amp;memories)?;

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
        let params = self.birch_params.get(&amp;space)
            .cloned()
            .unwrap_or_default();

        let mut new_tree = BIRCHTree::new(params, dim);

        for memory in &amp;memories {
            let embedding = extract_embedding_for_space(&amp;memory.teleological_array, space);
            new_tree.insert(&amp;embedding, memory.id)?;
        }

        self.birch_trees.insert(space, new_tree);
        self.last_recluster.insert(space, chrono::Utc::now());

        Ok(())
    }

    /// Get memberships for a memory
    pub fn get_memberships(&amp;self, memory_id: &amp;Uuid) -> Option&lt;&amp;Vec&lt;ClusterMembership&gt;&gt; {
        self.memberships.get(memory_id)
    }

    /// Get all memberships
    pub fn get_all_memberships(&amp;self) -> &amp;HashMap&lt;Uuid, Vec&lt;ClusterMembership&gt;&gt; {
        &amp;self.memberships
    }

    /// Check if space should be reclustered
    pub fn should_recluster(&amp;self, space: Embedder) -> bool {
        // Recluster if:
        // 1. Never reclustered and have enough memories
        // 2. BIRCH error exceeds threshold
        // 3. Enough time has passed

        if self.memory_count < 10 {
            return false;
        }

        if let Some(last) = self.last_recluster.get(&amp;space) {
            let elapsed = chrono::Utc::now() - *last;
            if elapsed < chrono::Duration::hours(24) {
                return false;
            }
        } else {
            // Never reclustered
            return true;
        }

        // Check BIRCH cluster count vs expected
        if let Some(tree) = self.birch_trees.get(&amp;space) {
            let cluster_count = tree.cluster_count();
            let expected = (self.memory_count as f32 / 10.0).ceil() as usize;
            let error = (cluster_count as f32 - expected as f32).abs() / expected as f32;
            return error > 0.15; // 15% error threshold
        }

        false
    }

    /// Get cluster for a memory in a specific space
    pub fn get_cluster_for_memory(&amp;self, memory_id: &amp;Uuid, space: Embedder) -> Option&lt;i32&gt; {
        self.memberships
            .get(memory_id)?
            .iter()
            .find(|m| m.space == space)
            .map(|m| m.cluster_id)
    }

    /// Get memory count
    pub fn memory_count(&amp;self) -> usize {
        self.memory_count
    }
}

/// Extract embedding vector for a specific space from TeleologicalArray
fn extract_embedding_for_space(array: &amp;TeleologicalArray, space: Embedder) -> Vec&lt;f32&gt; {
    match space {
        Embedder::E1Semantic => array.e1_semantic.as_slice().to_vec(),
        Embedder::E2TempRecent => array.e2_temp_recent.as_slice().to_vec(),
        Embedder::E3TempPeriodic => array.e3_temp_periodic.as_slice().to_vec(),
        Embedder::E4TempPosition => array.e4_temp_position.as_slice().to_vec(),
        Embedder::E5Causal => array.e5_causal.as_slice().to_vec(),
        Embedder::E6Sparse => array.e6_sparse.to_dense(), // Convert sparse to dense
        Embedder::E7Code => array.e7_code.as_slice().to_vec(),
        Embedder::E8Emotional => array.e8_emotional.as_slice().to_vec(),
        Embedder::E9HDC => array.e9_hdc.to_dense(), // Convert binary to dense
        Embedder::E10Multimodal => array.e10_multimodal.as_slice().to_vec(),
        Embedder::E11Entity => array.e11_entity.as_slice().to_vec(),
        Embedder::E12LateInteract => {
            // For late interaction, use mean of token embeddings
            if array.e12_late_interact.is_empty() {
                vec![0.0; 128]
            } else {
                let dim = array.e12_late_interact[0].len();
                let mut mean = vec![0.0; dim];
                for token in &amp;array.e12_late_interact {
                    for (i, v) in token.as_slice().iter().enumerate() {
                        mean[i] += v;
                    }
                }
                let n = array.e12_late_interact.len() as f32;
                mean.iter_mut().for_each(|v| *v /= n);
                mean
            }
        }
        Embedder::E13SPLADE => array.e13_splade.to_dense(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_manager() -> MultiSpaceClusterManager {
        let dir = tempdir().unwrap();
        let store = Arc::new(MemoryStore::new(dir.path()).unwrap());
        MultiSpaceClusterManager::new(store)
    }

    #[tokio::test]
    async fn test_activation_tiers() {
        let manager = create_test_manager().await;
        assert_eq!(manager.activation_tier(), 0);
        assert!(!manager.is_clustering_active());
    }

    #[tokio::test]
    async fn test_cluster_all_spaces_empty() {
        let mut manager = create_test_manager().await;
        let memories: Vec&lt;Memory&gt; = vec![];

        let result = manager.cluster_all_spaces(&amp;memories).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_should_recluster() {
        let manager = create_test_manager().await;

        // Should not recluster with < 10 memories
        assert!(!manager.should_recluster(Embedder::E1Semantic));
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/manager.rs">MultiSpaceClusterManager implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">Add pub mod manager and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>cluster_all_spaces processes all 13 spaces</criterion>
  <criterion>insert_memory updates BIRCH trees for all spaces</criterion>
  <criterion>Memberships stored per memory</criterion>
  <criterion>activation_tier returns correct tier</criterion>
  <criterion>should_recluster returns false when &lt;10 memories</criterion>
  <criterion>recluster_space rebuilds single space</criterion>
</validation_criteria>

<test_commands>
  <command description="Run manager tests">cargo test --package context-graph-core manager</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create manager.rs
- [ ] Implement MultiSpaceClusterManager struct
- [ ] Initialize per-space BIRCH trees
- [ ] Implement cluster_all_spaces
- [ ] Implement insert_memory
- [ ] Implement recluster_space
- [ ] Implement should_recluster logic
- [ ] Implement activation_tier
- [ ] Implement extract_embedding_for_space helper
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-008
