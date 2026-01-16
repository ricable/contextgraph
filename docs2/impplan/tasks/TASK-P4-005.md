# TASK-P4-005: HDBSCANClusterer

```xml
<task_spec id="TASK-P4-005" version="1.0">
<metadata>
  <title>HDBSCANClusterer Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>31</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P4-001</task_ref>
    <task_ref>TASK-P4-003</task_ref>
    <task_ref>TASK-P3-004</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

<context>
Implements the HDBSCANClusterer that performs batch density-based clustering.
HDBSCAN (Hierarchical DBSCAN) builds a cluster hierarchy and extracts clusters
using Excess of Mass or Leaf selection. Returns ClusterMembership with probability
and core point status.

This is the primary batch clustering algorithm for reclustering all memories.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#component_contracts</file>
  <file purpose="params">crates/context-graph-core/src/clustering/hdbscan.rs</file>
  <file purpose="distance">crates/context-graph-core/src/retrieval/distance.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-001 complete (ClusterMembership exists)</check>
  <check>TASK-P4-003 complete (HDBSCANParams exists)</check>
  <check>TASK-P3-004 complete (distance functions exist)</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement fit method (main clustering)
    - Build mutual reachability graph
    - Construct minimum spanning tree
    - Build cluster hierarchy (dendrogram)
    - Extract clusters (EOM/Leaf)
    - Compute membership probabilities
    - Compute silhouette score
    - Mark core points
  </in_scope>
  <out_of_scope>
    - Incremental updates (BIRCH handles that)
    - Per-space indexing (future optimization)
    - GPU acceleration
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/hdbscan.rs">
      pub struct HDBSCANClusterer {
          params: HDBSCANParams,
      }

      impl HDBSCANClusterer {
          pub fn new(params: HDBSCANParams) -> Self;
          pub fn fit(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;], memory_ids: &amp;[Uuid], space: Embedder) -> Result&lt;Vec&lt;ClusterMembership&gt;, ClusterError&gt;;
          pub fn compute_silhouette(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;], labels: &amp;[i32]) -> f32;
          fn compute_core_distances(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;]) -> Vec&lt;f32&gt;;
          fn compute_mutual_reachability(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;], core_distances: &amp;[f32]) -> Vec&lt;Vec&lt;f32&gt;&gt;;
          fn build_mst(&amp;self, distances: &amp;[Vec&lt;f32&gt;]) -> Vec&lt;(usize, usize, f32)&gt;;
          fn extract_clusters(&amp;self, mst: &amp;[(usize, usize, f32)], n_points: usize) -> (Vec&lt;i32&gt;, Vec&lt;f32&gt;);
      }
    </signature>
  </signatures>

  <constraints>
    - fit requires min_cluster_size points
    - O(n² log n) for dense distance matrix
    - Noise points get cluster_id = -1
    - membership_probability in 0.0..=1.0
    - silhouette_score in -1.0..=1.0
  </constraints>

  <verification>
    - fit returns correct number of memberships
    - Noise points have probability = 0.0
    - Core points identified correctly
    - Silhouette computed correctly
    - Clusters respect min_cluster_size
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/hdbscan.rs (clusterer section)

use std::collections::{HashSet, HashMap};
use uuid::Uuid;
use crate::embedding::Embedder;
use crate::retrieval::distance::{cosine_similarity, jaccard_similarity, hamming_similarity};
use super::membership::ClusterMembership;
use super::error::ClusterError;

/// HDBSCAN clusterer for batch density-based clustering
pub struct HDBSCANClusterer {
    params: HDBSCANParams,
}

impl HDBSCANClusterer {
    /// Create a new HDBSCAN clusterer
    pub fn new(params: HDBSCANParams) -> Self {
        Self { params }
    }

    /// Create with default parameters
    pub fn with_defaults() -> Self {
        Self::new(HDBSCANParams::default())
    }

    /// Fit the clusterer to embeddings and return cluster assignments
    pub fn fit(
        &amp;self,
        embeddings: &amp;[Vec&lt;f32&gt;],
        memory_ids: &amp;[Uuid],
        space: Embedder,
    ) -> Result&lt;Vec&lt;ClusterMembership&gt;, ClusterError&gt; {
        let n = embeddings.len();

        if n < self.params.min_cluster_size {
            return Err(ClusterError::InsufficientData {
                required: self.params.min_cluster_size,
                actual: n,
            });
        }

        if n != memory_ids.len() {
            return Err(ClusterError::DimensionMismatch {
                expected: n,
                actual: memory_ids.len(),
            });
        }

        // Step 1: Compute core distances
        let core_distances = self.compute_core_distances(embeddings);

        // Step 2: Compute mutual reachability distances
        let mutual_reach = self.compute_mutual_reachability(embeddings, &amp;core_distances);

        // Step 3: Build minimum spanning tree
        let mst = self.build_mst(&amp;mutual_reach);

        // Step 4: Extract clusters from hierarchy
        let (labels, probabilities) = self.extract_clusters(&amp;mst, n);

        // Step 5: Identify core points
        let core_points = self.identify_core_points(embeddings, &amp;labels);

        // Build ClusterMemberships
        let memberships: Vec&lt;ClusterMembership&gt; = memory_ids
            .iter()
            .zip(labels.iter())
            .zip(probabilities.iter())
            .zip(core_points.iter())
            .map(|(((id, &amp;label), &amp;prob), &amp;is_core)| {
                ClusterMembership::new(*id, space, label, prob, is_core)
            })
            .collect();

        Ok(memberships)
    }

    /// Compute core distances (distance to k-th nearest neighbor)
    fn compute_core_distances(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;]) -> Vec&lt;f32&gt; {
        let k = self.params.min_samples;
        let n = embeddings.len();
        let mut core_distances = Vec::with_capacity(n);

        for i in 0..n {
            // Compute distances to all other points
            let mut distances: Vec&lt;f32&gt; = (0..n)
                .filter(|&amp;j| j != i)
                .map(|j| self.point_distance(&amp;embeddings[i], &amp;embeddings[j]))
                .collect();

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Core distance is distance to k-th nearest (0-indexed: k-1)
            let core_dist = if k <= distances.len() {
                distances[k - 1]
            } else {
                distances.last().copied().unwrap_or(f32::INFINITY)
            };

            core_distances.push(core_dist);
        }

        core_distances
    }

    /// Compute mutual reachability distances
    fn compute_mutual_reachability(
        &amp;self,
        embeddings: &amp;[Vec&lt;f32&gt;],
        core_distances: &amp;[f32],
    ) -> Vec&lt;Vec&lt;f32&gt;&gt; {
        let n = embeddings.len();
        let mut mutual_reach = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = self.point_distance(&amp;embeddings[i], &amp;embeddings[j]);
                let mr = dist.max(core_distances[i]).max(core_distances[j]);
                mutual_reach[i][j] = mr;
                mutual_reach[j][i] = mr;
            }
        }

        mutual_reach
    }

    /// Build minimum spanning tree using Prim's algorithm
    fn build_mst(&amp;self, distances: &amp;[Vec&lt;f32&gt;]) -> Vec&lt;(usize, usize, f32)&gt; {
        let n = distances.len();
        if n == 0 {
            return vec![];
        }

        let mut in_tree = vec![false; n];
        let mut edges = Vec::with_capacity(n - 1);
        let mut min_dist = vec![f32::INFINITY; n];
        let mut min_edge = vec![0usize; n];

        // Start from node 0
        in_tree[0] = true;
        for j in 1..n {
            min_dist[j] = distances[0][j];
            min_edge[j] = 0;
        }

        for _ in 1..n {
            // Find minimum distance node not in tree
            let mut min_val = f32::INFINITY;
            let mut min_idx = 0;

            for j in 0..n {
                if !in_tree[j] &amp;&amp; min_dist[j] < min_val {
                    min_val = min_dist[j];
                    min_idx = j;
                }
            }

            // Add to tree
            in_tree[min_idx] = true;
            edges.push((min_edge[min_idx], min_idx, min_val));

            // Update distances
            for j in 0..n {
                if !in_tree[j] &amp;&amp; distances[min_idx][j] < min_dist[j] {
                    min_dist[j] = distances[min_idx][j];
                    min_edge[j] = min_idx;
                }
            }
        }

        // Sort edges by weight
        edges.sort_by(|a, b| a.2.partial_cmp(&amp;b.2).unwrap_or(std::cmp::Ordering::Equal));
        edges
    }

    /// Extract clusters from MST hierarchy using EOM or Leaf
    fn extract_clusters(
        &amp;self,
        mst: &amp;[(usize, usize, f32)],
        n_points: usize,
    ) -> (Vec&lt;i32&gt;, Vec&lt;f32&gt;) {
        if n_points == 0 {
            return (vec![], vec![]);
        }

        // Build condensed tree and extract clusters
        // Simplified version: use single-linkage at appropriate cutoff
        let mut labels = vec![-1i32; n_points];
        let mut probabilities = vec![0.0f32; n_points];

        // Union-Find for connected components
        let mut parent: Vec&lt;usize&gt; = (0..n_points).collect();

        fn find(parent: &amp;mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &amp;mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        // Add edges until we have stable clusters
        let mut cluster_sizes: HashMap&lt;usize, usize&gt; = HashMap::new();
        for i in 0..n_points {
            cluster_sizes.insert(i, 1);
        }

        // Process edges in order of weight
        for (i, j, _weight) in mst {
            let pi = find(&amp;mut parent, *i);
            let pj = find(&amp;mut parent, *j);

            if pi != pj {
                let size_i = cluster_sizes.get(&amp;pi).copied().unwrap_or(1);
                let size_j = cluster_sizes.get(&amp;pj).copied().unwrap_or(1);

                union(&amp;mut parent, pi, pj);
                let new_root = find(&amp;mut parent, pi);
                cluster_sizes.insert(new_root, size_i + size_j);
            }
        }

        // Assign cluster labels
        let mut cluster_map: HashMap&lt;usize, i32&gt; = HashMap::new();
        let mut next_cluster = 0i32;

        for i in 0..n_points {
            let root = find(&amp;mut parent, i);
            let cluster_size = cluster_sizes.get(&amp;root).copied().unwrap_or(1);

            if cluster_size >= self.params.min_cluster_size {
                let cluster_id = *cluster_map.entry(root).or_insert_with(|| {
                    let id = next_cluster;
                    next_cluster += 1;
                    id
                });
                labels[i] = cluster_id;
                probabilities[i] = 1.0; // Simplified: full probability for cluster members
            } else {
                labels[i] = -1; // Noise
                probabilities[i] = 0.0;
            }
        }

        (labels, probabilities)
    }

    /// Identify core points (points with >= min_samples neighbors within core distance)
    fn identify_core_points(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;], labels: &amp;[i32]) -> Vec&lt;bool&gt; {
        let n = embeddings.len();
        let mut is_core = vec![false; n];

        for i in 0..n {
            if labels[i] == -1 {
                continue; // Noise is not core
            }

            // Count neighbors in same cluster
            let mut neighbor_count = 0;
            for j in 0..n {
                if i != j &amp;&amp; labels[j] == labels[i] {
                    neighbor_count += 1;
                }
            }

            is_core[i] = neighbor_count >= self.params.min_samples;
        }

        is_core
    }

    /// Compute distance between two points
    fn point_distance(&amp;self, a: &amp;[f32], b: &amp;[f32]) -> f32 {
        match self.params.metric {
            DistanceMetric::Cosine => {
                let dense_a = DenseVector::new(a.to_vec());
                let dense_b = DenseVector::new(b.to_vec());
                1.0 - cosine_similarity(&amp;dense_a, &amp;dense_b)
            }
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt()
            }
            _ => {
                // Default to Euclidean for other metrics
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt()
            }
        }
    }

    /// Compute silhouette score for clustering quality
    pub fn compute_silhouette(&amp;self, embeddings: &amp;[Vec&lt;f32&gt;], labels: &amp;[i32]) -> f32 {
        let n = embeddings.len();
        if n < 2 {
            return 0.0;
        }

        // Get unique non-noise clusters
        let clusters: HashSet&lt;i32&gt; = labels.iter()
            .filter(|&amp;&amp;l| l != -1)
            .copied()
            .collect();

        if clusters.len() < 2 {
            return 0.0; // Need at least 2 clusters
        }

        let mut total_silhouette = 0.0;
        let mut count = 0;

        for i in 0..n {
            if labels[i] == -1 {
                continue; // Skip noise
            }

            // a(i) = mean distance to same cluster
            let mut same_cluster_sum = 0.0;
            let mut same_cluster_count = 0;

            for j in 0..n {
                if i != j &amp;&amp; labels[j] == labels[i] {
                    same_cluster_sum += self.point_distance(&amp;embeddings[i], &amp;embeddings[j]);
                    same_cluster_count += 1;
                }
            }

            let a_i = if same_cluster_count > 0 {
                same_cluster_sum / same_cluster_count as f32
            } else {
                0.0
            };

            // b(i) = min mean distance to other clusters
            let mut min_other_mean = f32::INFINITY;

            for &amp;cluster in &amp;clusters {
                if cluster == labels[i] {
                    continue;
                }

                let mut other_sum = 0.0;
                let mut other_count = 0;

                for j in 0..n {
                    if labels[j] == cluster {
                        other_sum += self.point_distance(&amp;embeddings[i], &amp;embeddings[j]);
                        other_count += 1;
                    }
                }

                if other_count > 0 {
                    let mean = other_sum / other_count as f32;
                    min_other_mean = min_other_mean.min(mean);
                }
            }

            let b_i = min_other_mean;

            // s(i) = (b(i) - a(i)) / max(a(i), b(i))
            let s_i = if a_i.max(b_i) > 0.0 {
                (b_i - a_i) / a_i.max(b_i)
            } else {
                0.0
            };

            total_silhouette += s_i;
            count += 1;
        }

        if count > 0 {
            total_silhouette / count as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embeddings() -> Vec&lt;Vec&lt;f32&gt;&gt; {
        // Two clear clusters
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![5.2, 5.0],
        ]
    }

    #[test]
    fn test_hdbscan_fit() {
        let embeddings = create_test_embeddings();
        let ids: Vec&lt;Uuid&gt; = (0..6).map(|_| Uuid::new_v4()).collect();

        let params = HDBSCANParams::default()
            .with_min_cluster_size(2)
            .with_min_samples(1);

        let clusterer = HDBSCANClusterer::new(params);
        let result = clusterer.fit(&amp;embeddings, &amp;ids, Embedder::E1Semantic);

        assert!(result.is_ok());
        let memberships = result.unwrap();
        assert_eq!(memberships.len(), 6);
    }

    #[test]
    fn test_hdbscan_insufficient_data() {
        let embeddings = vec![vec![0.0, 0.0]];
        let ids = vec![Uuid::new_v4()];

        let clusterer = HDBSCANClusterer::with_defaults();
        let result = clusterer.fit(&amp;embeddings, &amp;ids, Embedder::E1Semantic);

        assert!(matches!(result, Err(ClusterError::InsufficientData { .. })));
    }

    #[test]
    fn test_silhouette_score() {
        let embeddings = create_test_embeddings();
        let labels = vec![0, 0, 0, 1, 1, 1]; // Two perfect clusters

        let clusterer = HDBSCANClusterer::with_defaults();
        let silhouette = clusterer.compute_silhouette(&amp;embeddings, &amp;labels);

        // Should be high for well-separated clusters
        assert!(silhouette > 0.5);
    }
}
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/hdbscan.rs">Add HDBSCANClusterer implementation</file>
</files_to_modify>

<validation_criteria>
  <criterion>fit returns ClusterMembership for each input</criterion>
  <criterion>Noise points have cluster_id = -1 and probability = 0.0</criterion>
  <criterion>Core points correctly identified</criterion>
  <criterion>Silhouette score computed correctly</criterion>
  <criterion>min_cluster_size respected</criterion>
  <criterion>Handles insufficient data with proper error</criterion>
</validation_criteria>

<test_commands>
  <command description="Run HDBSCAN tests">cargo test --package context-graph-core hdbscan</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>

<notes>
  <note category="complexity">
    Current implementation is O(n²) for distance matrix.
    For large datasets (&gt;1000 points), consider spatial indexing.
  </note>
  <note category="algorithm">
    This is a simplified HDBSCAN implementation.
    Production use may benefit from the hdbscan crate.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Implement HDBSCANClusterer struct
- [ ] Implement compute_core_distances
- [ ] Implement compute_mutual_reachability
- [ ] Implement build_mst (Prim's algorithm)
- [ ] Implement extract_clusters (EOM)
- [ ] Implement identify_core_points
- [ ] Implement compute_silhouette
- [ ] Write unit tests for two-cluster case
- [ ] Test noise detection
- [ ] Test silhouette calculation
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-006
