# TASK-P4-006: BIRCHTree

```xml
<task_spec id="TASK-P4-006" version="1.0">
<metadata>
  <title>BIRCHTree Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>32</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-02</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P4-001</task_ref>
    <task_ref>TASK-P4-004</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

<context>
Implements the BIRCHTree for O(log n) incremental clustering. BIRCH maintains
a CF-tree (Clustering Feature tree) where each leaf entry represents a sub-cluster.
New points are inserted by navigating to the closest leaf and either merging
or splitting nodes.

This is the primary real-time clustering algorithm for new memory insertion.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#component_contracts</file>
  <file purpose="params">crates/context-graph-core/src/clustering/birch.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-001 complete (ClusterMembership exists)</check>
  <check>TASK-P4-004 complete (BIRCHParams and ClusteringFeature exist)</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement BIRCHTree struct
    - Implement insert method
    - Implement node splitting
    - Implement get_clusters method
    - Implement adapt_threshold method
    - CF propagation to parents
    - Track cluster assignments
  </in_scope>
  <out_of_scope>
    - Full rebuild from scratch (use HDBSCAN)
    - GPU acceleration
    - Persistent storage of tree
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/birch.rs">
      pub struct BIRCHNode {
          is_leaf: bool,
          entries: Vec&lt;BIRCHEntry&gt;,
      }

      pub struct BIRCHEntry {
          cf: ClusteringFeature,
          child: Option&lt;Box&lt;BIRCHNode&gt;&gt;,
          memory_ids: Vec&lt;Uuid&gt;,
      }

      pub struct BIRCHTree {
          params: BIRCHParams,
          root: BIRCHNode,
          dimension: usize,
          total_points: usize,
      }

      impl BIRCHTree {
          pub fn new(params: BIRCHParams, dimension: usize) -> Self;
          pub fn insert(&amp;mut self, embedding: &amp;[f32], memory_id: Uuid) -> Result&lt;usize, ClusterError&gt;;
          pub fn get_clusters(&amp;self) -> Vec&lt;ClusteringFeature&gt;;
          pub fn get_cluster_members(&amp;self) -> Vec&lt;(usize, Vec&lt;Uuid&gt;)&gt;;
          pub fn adapt_threshold(&amp;mut self, target_cluster_count: usize);
          pub fn cluster_count(&amp;self) -> usize;
      }
    </signature>
  </signatures>

  <constraints>
    - insert is O(B log n) where B = branching_factor
    - Node splits when entries > max_node_entries
    - Points merge into entry if distance < threshold
    - Leaf entries track member memory IDs
  </constraints>

  <verification>
    - insert returns cluster index
    - Node splits correctly on overflow
    - CF values propagated to parents
    - adapt_threshold adjusts cluster granularity
    - Memory IDs tracked correctly
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/birch.rs (tree section)

use uuid::Uuid;
use super::error::ClusterError;

/// Entry in a BIRCH node
#[derive(Debug, Clone)]
pub struct BIRCHEntry {
    /// Clustering feature summary
    pub cf: ClusteringFeature,
    /// Child node (None for leaf entries)
    pub child: Option&lt;Box&lt;BIRCHNode&gt;&gt;,
    /// Memory IDs in this entry (leaf only)
    pub memory_ids: Vec&lt;Uuid&gt;,
}

impl BIRCHEntry {
    /// Create a new leaf entry from a point
    pub fn from_point(embedding: &amp;[f32], memory_id: Uuid) -> Self {
        Self {
            cf: ClusteringFeature::from_point(embedding),
            child: None,
            memory_ids: vec![memory_id],
        }
    }

    /// Create a new non-leaf entry with a child
    pub fn with_child(cf: ClusteringFeature, child: BIRCHNode) -> Self {
        Self {
            cf,
            child: Some(Box::new(child)),
            memory_ids: Vec::new(),
        }
    }

    /// Check if this is a leaf entry
    pub fn is_leaf(&amp;self) -> bool {
        self.child.is_none()
    }

    /// Merge a point into this entry
    pub fn merge_point(&amp;mut self, embedding: &amp;[f32], memory_id: Uuid) {
        self.cf.add_point(embedding);
        self.memory_ids.push(memory_id);
    }
}

/// Node in the BIRCH CF-tree
#[derive(Debug, Clone)]
pub struct BIRCHNode {
    /// Whether this is a leaf node
    pub is_leaf: bool,
    /// Entries in this node
    pub entries: Vec&lt;BIRCHEntry&gt;,
}

impl BIRCHNode {
    /// Create a new empty leaf node
    pub fn new_leaf() -> Self {
        Self {
            is_leaf: true,
            entries: Vec::new(),
        }
    }

    /// Create a new empty non-leaf node
    pub fn new_internal() -> Self {
        Self {
            is_leaf: false,
            entries: Vec::new(),
        }
    }

    /// Get total CF for this node
    pub fn total_cf(&amp;self) -> ClusteringFeature {
        let dim = self.entries.first()
            .map(|e| e.cf.dimension())
            .unwrap_or(0);

        let mut total = ClusteringFeature::new(dim);
        for entry in &amp;self.entries {
            total.merge(&amp;entry.cf);
        }
        total
    }

    /// Find closest entry to a point
    pub fn find_closest(&amp;self, point: &amp;[f32]) -> Option&lt;usize&gt; {
        if self.entries.is_empty() {
            return None;
        }

        let point_cf = ClusteringFeature::from_point(point);
        let mut min_dist = f32::INFINITY;
        let mut min_idx = 0;

        for (i, entry) in self.entries.iter().enumerate() {
            let dist = entry.cf.distance(&amp;point_cf);
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        Some(min_idx)
    }
}

/// BIRCH CF-tree for incremental clustering
#[derive(Debug)]
pub struct BIRCHTree {
    params: BIRCHParams,
    root: BIRCHNode,
    dimension: usize,
    total_points: usize,
}

impl BIRCHTree {
    /// Create a new empty BIRCH tree
    pub fn new(params: BIRCHParams, dimension: usize) -> Self {
        Self {
            params,
            root: BIRCHNode::new_leaf(),
            dimension,
            total_points: 0,
        }
    }

    /// Insert a point into the tree
    /// Returns the cluster index for this point
    pub fn insert(
        &amp;mut self,
        embedding: &amp;[f32],
        memory_id: Uuid,
    ) -> Result&lt;usize, ClusterError&gt; {
        if embedding.len() != self.dimension {
            return Err(ClusterError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let cluster_idx = self.insert_recursive(embedding, memory_id, &amp;mut self.root.clone());

        // Handle root split if needed
        if self.root.entries.len() > self.params.max_node_entries {
            self.split_root();
        }

        self.total_points += 1;
        Ok(cluster_idx)
    }

    /// Recursive insertion helper
    fn insert_recursive(
        &amp;mut self,
        embedding: &amp;[f32],
        memory_id: Uuid,
        node: &amp;mut BIRCHNode,
    ) -> usize {
        if node.is_leaf {
            // Find closest entry or create new one
            if let Some(idx) = node.find_closest(embedding) {
                let entry = &amp;mut node.entries[idx];

                // Check if point fits within threshold
                if entry.cf.would_fit(embedding, self.params.threshold) {
                    entry.merge_point(embedding, memory_id);
                    return idx;
                }
            }

            // Create new entry
            let new_entry = BIRCHEntry::from_point(embedding, memory_id);
            let new_idx = node.entries.len();
            node.entries.push(new_entry);

            // Handle node split if needed
            if node.entries.len() > self.params.max_node_entries {
                self.split_leaf(node);
            }

            new_idx
        } else {
            // Non-leaf: descend to closest child
            let closest_idx = node.find_closest(embedding).unwrap_or(0);

            if let Some(ref mut child) = node.entries[closest_idx].child {
                let cluster_idx = self.insert_recursive(embedding, memory_id, child);

                // Update CF
                node.entries[closest_idx].cf.add_point(embedding);

                // Handle child split if needed
                if child.entries.len() > self.params.max_node_entries {
                    self.split_non_leaf(node, closest_idx);
                }

                cluster_idx
            } else {
                0
            }
        }
    }

    /// Split a leaf node
    fn split_leaf(&amp;self, node: &amp;mut BIRCHNode) {
        if node.entries.len() <= self.params.max_node_entries {
            return;
        }

        // Find two most distant entries as seeds
        let (seed1, seed2) = self.find_farthest_pair(&amp;node.entries);

        let mut entries1 = vec![node.entries[seed1].clone()];
        let mut entries2 = vec![node.entries[seed2].clone()];

        // Distribute other entries
        for (i, entry) in node.entries.iter().enumerate() {
            if i == seed1 || i == seed2 {
                continue;
            }

            let dist1 = entry.cf.distance(&amp;entries1[0].cf);
            let dist2 = entry.cf.distance(&amp;entries2[0].cf);

            if dist1 <= dist2 {
                entries1.push(entry.clone());
            } else {
                entries2.push(entry.clone());
            }
        }

        // Keep smaller group in current node, promote larger
        if entries1.len() <= entries2.len() {
            node.entries = entries1;
            // entries2 would be returned for parent to handle
        } else {
            node.entries = entries2;
        }
    }

    /// Split root node
    fn split_root(&amp;mut self) {
        if self.root.entries.len() <= self.params.max_node_entries {
            return;
        }

        let (seed1, seed2) = self.find_farthest_pair(&amp;self.root.entries);

        let mut node1 = BIRCHNode::new_leaf();
        let mut node2 = BIRCHNode::new_leaf();
        node1.is_leaf = self.root.is_leaf;
        node2.is_leaf = self.root.is_leaf;

        node1.entries.push(self.root.entries[seed1].clone());
        node2.entries.push(self.root.entries[seed2].clone());

        for (i, entry) in self.root.entries.iter().enumerate() {
            if i == seed1 || i == seed2 {
                continue;
            }

            let dist1 = entry.cf.distance(&amp;node1.entries[0].cf);
            let dist2 = entry.cf.distance(&amp;node2.entries[0].cf);

            if dist1 <= dist2 {
                node1.entries.push(entry.clone());
            } else {
                node2.entries.push(entry.clone());
            }
        }

        // Create new root
        let entry1 = BIRCHEntry::with_child(node1.total_cf(), node1);
        let entry2 = BIRCHEntry::with_child(node2.total_cf(), node2);

        self.root = BIRCHNode::new_internal();
        self.root.entries.push(entry1);
        self.root.entries.push(entry2);
    }

    /// Split a non-leaf node at the given child index
    fn split_non_leaf(&amp;self, _parent: &amp;mut BIRCHNode, _child_idx: usize) {
        // Similar to split_leaf but for internal nodes
        // Simplified: defer to split_root pattern
    }

    /// Find two most distant entries
    fn find_farthest_pair(&amp;self, entries: &amp;[BIRCHEntry]) -> (usize, usize) {
        if entries.len() < 2 {
            return (0, 0);
        }

        let mut max_dist = 0.0f32;
        let mut pair = (0, 1);

        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let dist = entries[i].cf.distance(&amp;entries[j].cf);
                if dist > max_dist {
                    max_dist = dist;
                    pair = (i, j);
                }
            }
        }

        pair
    }

    /// Get all leaf CFs as cluster summaries
    pub fn get_clusters(&amp;self) -> Vec&lt;ClusteringFeature&gt; {
        let mut clusters = Vec::new();
        self.collect_leaf_cfs(&amp;self.root, &amp;mut clusters);
        clusters
    }

    /// Recursively collect leaf CFs
    fn collect_leaf_cfs(&amp;self, node: &amp;BIRCHNode, clusters: &amp;mut Vec&lt;ClusteringFeature&gt;) {
        if node.is_leaf {
            for entry in &amp;node.entries {
                clusters.push(entry.cf.clone());
            }
        } else {
            for entry in &amp;node.entries {
                if let Some(ref child) = entry.child {
                    self.collect_leaf_cfs(child, clusters);
                }
            }
        }
    }

    /// Get cluster members (cluster_idx -> memory_ids)
    pub fn get_cluster_members(&amp;self) -> Vec&lt;(usize, Vec&lt;Uuid&gt;)&gt; {
        let mut members = Vec::new();
        self.collect_members(&amp;self.root, &amp;mut members, &amp;mut 0);
        members
    }

    fn collect_members(
        &amp;self,
        node: &amp;BIRCHNode,
        members: &amp;mut Vec&lt;(usize, Vec&lt;Uuid&gt;)&gt;,
        idx: &amp;mut usize,
    ) {
        if node.is_leaf {
            for entry in &amp;node.entries {
                members.push((*idx, entry.memory_ids.clone()));
                *idx += 1;
            }
        } else {
            for entry in &amp;node.entries {
                if let Some(ref child) = entry.child {
                    self.collect_members(child, members, idx);
                }
            }
        }
    }

    /// Adapt threshold to achieve target cluster count
    pub fn adapt_threshold(&amp;mut self, target_cluster_count: usize) {
        let current_count = self.cluster_count();

        if current_count == target_cluster_count || target_cluster_count == 0 {
            return;
        }

        // Binary search for appropriate threshold
        let mut low = 0.01f32;
        let mut high = 1.0f32;

        for _ in 0..10 {
            let mid = (low + high) / 2.0;
            self.params.threshold = mid;

            // Would need to rebuild tree to test - simplified here
            if current_count > target_cluster_count {
                low = mid; // Increase threshold to reduce clusters
            } else {
                high = mid; // Decrease threshold to increase clusters
            }
        }
    }

    /// Get current cluster count
    pub fn cluster_count(&amp;self) -> usize {
        self.get_clusters().len()
    }

    /// Get total points in tree
    pub fn total_points(&amp;self) -> usize {
        self.total_points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_birch_insert() {
        let params = birch_defaults();
        let mut tree = BIRCHTree::new(params, 3);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let result1 = tree.insert(&amp;[1.0, 2.0, 3.0], id1);
        assert!(result1.is_ok());

        let result2 = tree.insert(&amp;[1.1, 2.1, 3.1], id2);
        assert!(result2.is_ok());

        assert_eq!(tree.total_points(), 2);
    }

    #[test]
    fn test_birch_dimension_check() {
        let params = birch_defaults();
        let mut tree = BIRCHTree::new(params, 3);

        let result = tree.insert(&amp;[1.0, 2.0], Uuid::new_v4()); // Wrong dimension
        assert!(matches!(result, Err(ClusterError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_birch_clustering() {
        let params = BIRCHParams::default().with_threshold(0.5);
        let mut tree = BIRCHTree::new(params, 2);

        // Insert two clear clusters
        for i in 0..5 {
            tree.insert(&amp;[0.0 + i as f32 * 0.1, 0.0], Uuid::new_v4()).unwrap();
        }
        for i in 0..5 {
            tree.insert(&amp;[10.0 + i as f32 * 0.1, 10.0], Uuid::new_v4()).unwrap();
        }

        let clusters = tree.get_clusters();
        assert!(clusters.len() >= 2); // Should have at least 2 clusters
    }

    #[test]
    fn test_birch_get_members() {
        let params = birch_defaults();
        let mut tree = BIRCHTree::new(params, 2);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        tree.insert(&amp;[0.0, 0.0], id1).unwrap();
        tree.insert(&amp;[0.1, 0.1], id2).unwrap();

        let members = tree.get_cluster_members();
        let total_members: usize = members.iter().map(|(_, m)| m.len()).sum();
        assert_eq!(total_members, 2);
    }
}
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/birch.rs">Add BIRCHTree, BIRCHNode, BIRCHEntry implementations</file>
</files_to_modify>

<validation_criteria>
  <criterion>insert returns cluster index</criterion>
  <criterion>Dimension mismatch detected</criterion>
  <criterion>Node splits when exceeding max_entries</criterion>
  <criterion>get_clusters returns all leaf CFs</criterion>
  <criterion>Memory IDs tracked in leaf entries</criterion>
  <criterion>adapt_threshold adjusts granularity</criterion>
</validation_criteria>

<test_commands>
  <command description="Run BIRCH tree tests">cargo test --package context-graph-core birch</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>

<notes>
  <note category="performance">
    Insert is O(B log n) where B = branching_factor.
    Good for real-time incremental updates.
  </note>
  <note category="sparse">
    BIRCH uses Euclidean distance internally.
    For sparse vectors (E6, E13), consider alternative approaches.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Implement BIRCHEntry struct
- [ ] Implement BIRCHNode struct
- [ ] Implement BIRCHTree struct
- [ ] Implement insert method with threshold check
- [ ] Implement node splitting logic
- [ ] Implement root splitting
- [ ] Implement get_clusters
- [ ] Implement get_cluster_members
- [ ] Implement adapt_threshold
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-007
