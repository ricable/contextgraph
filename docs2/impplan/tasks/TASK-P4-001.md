# TASK-P4-001: ClusterMembership and Cluster Types

```xml
<task_spec id="TASK-P4-001" version="1.0">
<metadata>
  <title>ClusterMembership and Cluster Type Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>27</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P2-003</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the foundational types for cluster assignments. ClusterMembership tracks
which cluster a memory belongs to in each embedding space, including probability
and core point status. Cluster represents a cluster with centroid, member count,
and quality metrics.

These types are used by both HDBSCAN and BIRCH clustering algorithms.
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#data_models</file>
  <file purpose="embedder">crates/context-graph-core/src/embedding/embedder.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P2-003 complete (Embedder enum exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ClusterMembership struct
    - Create Cluster struct
    - Create ClusterError enum
    - Implement Debug, Clone, Serialize, Deserialize
    - Implement constructors and accessors
    - Handle noise points (cluster_id = -1)
  </in_scope>
  <out_of_scope>
    - Clustering algorithm logic (TASK-P4-005, P4-006)
    - Cluster storage (TASK-P4-007)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/membership.rs">
      #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
      pub struct ClusterMembership {
          pub memory_id: Uuid,
          pub space: Embedder,
          pub cluster_id: i32,
          pub membership_probability: f32,
          pub is_core_point: bool,
      }

      impl ClusterMembership {
          pub fn new(memory_id: Uuid, space: Embedder, cluster_id: i32, probability: f32, is_core: bool) -> Self;
          pub fn noise(memory_id: Uuid, space: Embedder) -> Self;
          pub fn is_noise(&amp;self) -> bool;
      }
    </signature>
    <signature file="crates/context-graph-core/src/clustering/cluster.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct Cluster {
          pub id: i32,
          pub space: Embedder,
          pub centroid: Vec&lt;f32&gt;,
          pub member_count: u32,
          pub silhouette_score: f32,
          pub created_at: DateTime&lt;Utc&gt;,
          pub updated_at: DateTime&lt;Utc&gt;,
      }

      impl Cluster {
          pub fn new(id: i32, space: Embedder, centroid: Vec&lt;f32&gt;, member_count: u32) -> Self;
          pub fn update_silhouette(&amp;mut self, score: f32);
          pub fn touch(&amp;mut self);
      }
    </signature>
    <signature file="crates/context-graph-core/src/clustering/error.rs">
      #[derive(Debug, Error)]
      pub enum ClusterError {
          #[error("Insufficient data: required {required}, actual {actual}")]
          InsufficientData { required: usize, actual: usize },
          #[error("Dimension mismatch: expected {expected}, actual {actual}")]
          DimensionMismatch { expected: usize, actual: usize },
          #[error("No valid clusters found")]
          NoValidClusters,
          #[error("Storage error: {0}")]
          StorageError(#[from] StorageError),
      }
    </signature>
  </signatures>

  <constraints>
    - cluster_id = -1 indicates noise (not in any cluster)
    - membership_probability in 0.0..=1.0
    - silhouette_score in -1.0..=1.0
    - centroid dimension must match embedder config
  </constraints>

  <verification>
    - noise() creates membership with cluster_id = -1
    - is_noise() returns true for cluster_id = -1
    - Serialization roundtrip works
    - Cluster.touch() updates updated_at
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/membership.rs

use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::embedding::Embedder;

/// Represents a memory's cluster assignment in a specific embedding space
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClusterMembership {
    /// The memory this membership belongs to
    pub memory_id: Uuid,
    /// The embedding space this assignment is for
    pub space: Embedder,
    /// Cluster ID (-1 = noise, not in any cluster)
    pub cluster_id: i32,
    /// Probability of belonging to this cluster (0.0..=1.0)
    pub membership_probability: f32,
    /// Whether this point is a core point of the cluster
    pub is_core_point: bool,
}

impl ClusterMembership {
    /// Create a new cluster membership
    pub fn new(
        memory_id: Uuid,
        space: Embedder,
        cluster_id: i32,
        probability: f32,
        is_core: bool,
    ) -> Self {
        Self {
            memory_id,
            space,
            cluster_id,
            membership_probability: probability.clamp(0.0, 1.0),
            is_core_point: is_core,
        }
    }

    /// Create a noise membership (not in any cluster)
    pub fn noise(memory_id: Uuid, space: Embedder) -> Self {
        Self {
            memory_id,
            space,
            cluster_id: -1,
            membership_probability: 0.0,
            is_core_point: false,
        }
    }

    /// Check if this is a noise point
    pub fn is_noise(&amp;self) -> bool {
        self.cluster_id == -1
    }

    /// Check if this is a high-confidence assignment
    pub fn is_confident(&amp;self) -> bool {
        self.membership_probability >= 0.8
    }
}

---
File: crates/context-graph-core/src/clustering/cluster.rs

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::embedding::Embedder;

/// Represents a cluster in an embedding space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster identifier (unique per space)
    pub id: i32,
    /// The embedding space this cluster belongs to
    pub space: Embedder,
    /// Cluster centroid (mean of all member embeddings)
    pub centroid: Vec&lt;f32&gt;,
    /// Number of members in this cluster
    pub member_count: u32,
    /// Silhouette score (-1.0..=1.0, higher is better)
    pub silhouette_score: f32,
    /// When the cluster was created
    pub created_at: DateTime&lt;Utc&gt;,
    /// When the cluster was last updated
    pub updated_at: DateTime&lt;Utc&gt;,
}

impl Cluster {
    /// Create a new cluster
    pub fn new(id: i32, space: Embedder, centroid: Vec&lt;f32&gt;, member_count: u32) -> Self {
        let now = Utc::now();
        Self {
            id,
            space,
            centroid,
            member_count,
            silhouette_score: 0.0, // Compute later
            created_at: now,
            updated_at: now,
        }
    }

    /// Update the silhouette score
    pub fn update_silhouette(&amp;mut self, score: f32) {
        self.silhouette_score = score.clamp(-1.0, 1.0);
        self.touch();
    }

    /// Update the updated_at timestamp
    pub fn touch(&amp;mut self) {
        self.updated_at = Utc::now();
    }

    /// Check if this cluster has good quality
    pub fn is_high_quality(&amp;self) -> bool {
        self.silhouette_score >= 0.3
    }

    /// Update centroid and member count
    pub fn update_centroid(&amp;mut self, centroid: Vec&lt;f32&gt;, member_count: u32) {
        self.centroid = centroid;
        self.member_count = member_count;
        self.touch();
    }
}

---
File: crates/context-graph-core/src/clustering/error.rs

use thiserror::Error;
use crate::memory::store::StorageError;

/// Errors that can occur during clustering operations
#[derive(Debug, Error)]
pub enum ClusterError {
    #[error("Insufficient data: required {required}, actual {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("No valid clusters found")]
    NoValidClusters,

    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),

    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },

    #[error("Space not initialized: {0:?}")]
    SpaceNotInitialized(crate::embedding::Embedder),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_membership() {
        let mem_id = Uuid::new_v4();
        let membership = ClusterMembership::noise(mem_id, Embedder::E1Semantic);

        assert!(membership.is_noise());
        assert_eq!(membership.cluster_id, -1);
        assert_eq!(membership.membership_probability, 0.0);
        assert!(!membership.is_core_point);
    }

    #[test]
    fn test_cluster_membership() {
        let mem_id = Uuid::new_v4();
        let membership = ClusterMembership::new(
            mem_id,
            Embedder::E1Semantic,
            5,
            0.95,
            true,
        );

        assert!(!membership.is_noise());
        assert_eq!(membership.cluster_id, 5);
        assert!(membership.is_confident());
    }

    #[test]
    fn test_probability_clamping() {
        let mem_id = Uuid::new_v4();
        let membership = ClusterMembership::new(
            mem_id,
            Embedder::E1Semantic,
            1,
            1.5, // Should be clamped to 1.0
            false,
        );

        assert_eq!(membership.membership_probability, 1.0);
    }

    #[test]
    fn test_cluster_touch() {
        let mut cluster = Cluster::new(
            1,
            Embedder::E1Semantic,
            vec![0.0; 1024],
            10,
        );

        let old_updated = cluster.updated_at;
        std::thread::sleep(std::time::Duration::from_millis(10));
        cluster.touch();

        assert!(cluster.updated_at > old_updated);
    }

    #[test]
    fn test_cluster_silhouette() {
        let mut cluster = Cluster::new(
            1,
            Embedder::E1Semantic,
            vec![0.0; 1024],
            10,
        );

        cluster.update_silhouette(0.75);
        assert!(cluster.is_high_quality());

        cluster.update_silhouette(0.2);
        assert!(!cluster.is_high_quality());
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/membership.rs">ClusterMembership type</file>
  <file path="crates/context-graph-core/src/clustering/cluster.rs">Cluster type</file>
  <file path="crates/context-graph-core/src/clustering/error.rs">ClusterError enum</file>
  <file path="crates/context-graph-core/src/clustering/mod.rs">Module re-exports</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/lib.rs">Add pub mod clustering</file>
</files_to_modify>

<validation_criteria>
  <criterion>ClusterMembership::noise creates -1 cluster_id</criterion>
  <criterion>is_noise() returns true for -1 cluster_id</criterion>
  <criterion>membership_probability clamped to 0.0..=1.0</criterion>
  <criterion>silhouette_score clamped to -1.0..=1.0</criterion>
  <criterion>Cluster.touch() updates updated_at timestamp</criterion>
  <criterion>Serialization/deserialization roundtrip works</criterion>
</validation_criteria>

<test_commands>
  <command description="Run membership tests">cargo test --package context-graph-core membership</command>
  <command description="Run cluster tests">cargo test --package context-graph-core cluster</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create clustering module directory
- [ ] Create membership.rs with ClusterMembership
- [ ] Create cluster.rs with Cluster type
- [ ] Create error.rs with ClusterError enum
- [ ] Create mod.rs with re-exports
- [ ] Add pub mod clustering to lib.rs
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-002
