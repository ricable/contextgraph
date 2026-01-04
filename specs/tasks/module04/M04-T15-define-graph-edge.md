---
id: "M04-T15"
title: "Implement GraphEdge with Marblestone Fields in context-graph-graph"
description: |
  Create a NEW GraphEdge struct in context-graph-graph crate with get_modulated_weight() method.
  This is SEPARATE from the existing GraphEdge in context-graph-core.
  The graph crate's version includes all 13 Marblestone fields plus
  steering-based weight modulation with the CANONICAL formula.

  CRITICAL: GraphEdge already exists in context-graph-core/src/types/graph_edge/edge.rs
  This task creates a GRAPH-SPECIFIC version in context-graph-graph/src/storage/edges.rs
  that includes get_modulated_weight() with the canonical steering formula.
layer: "logic"
status: "pending"
priority: "critical"
estimated_hours: 3
sequence: 21
depends_on:
  - "M04-T14a"  # NeurotransmitterWeights validation (COMPLETE)
spec_refs:
  - "TECH-GRAPH-004 Section 4.1"
  - "REQ-KG-040 through REQ-KG-044, REQ-KG-065"
files_to_create:
  - path: "crates/context-graph-graph/src/storage/edges.rs"
    description: "GraphEdge struct with EdgeType enum and get_modulated_weight()"
files_to_modify:
  - path: "crates/context-graph-graph/src/storage/mod.rs"
    description: "Add mod edges; and re-export GraphEdge, EdgeId, EdgeType"
  - path: "crates/context-graph-graph/src/lib.rs"
    description: "Re-export GraphEdge from storage module"
test_file: "crates/context-graph-graph/tests/edge_tests.rs"
---

## CRITICAL PROJECT STATE AWARENESS

### Existing Implementation (DO NOT DUPLICATE)

**GraphEdge ALREADY EXISTS in context-graph-core:**
- Location: `crates/context-graph-core/src/types/graph_edge/edge.rs`
- Has all 13 fields as specified in PRD Section 4.2
- Uses `chrono::DateTime<Utc>` for timestamps
- Uses `Uuid` for EdgeId
- Does NOT have `get_modulated_weight()` method with steering

**Types ALREADY EXIST in context-graph-core/src/marblestone/:**
- `Domain` enum: Code, Legal, Medical, Creative, Research, General (domain.rs)
- `EdgeType` enum: Semantic, Temporal, Causal, Hierarchical (edge_type.rs)
- `NeurotransmitterWeights` struct with `compute_effective_weight()` (neurotransmitter_weights.rs)

**Types ALREADY RE-EXPORTED in context-graph-graph:**
- `crates/context-graph-graph/src/lib.rs` line 60: `pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};`
- `crates/context-graph-graph/src/marblestone/mod.rs` line 39: Same re-export

### What This Task ACTUALLY Does

Create a **GRAPH-CRATE-SPECIFIC** GraphEdge that:
1. Re-uses Domain, EdgeType, NeurotransmitterWeights from context-graph-core
2. Uses `NodeId` (UUID) and `i64` for EdgeId (different from core's Uuid)
3. Uses `u64` Unix timestamps (not chrono) for RocksDB compatibility
4. Adds `get_modulated_weight(query_domain)` with STEERING-BASED formula
5. Adds `record_traversal()` for EMA steering reward updates
6. Adds `domain_bonus()` calculation for query domain matching

### Formula Clarification

**Core crate formula (NeurotransmitterWeights::compute_effective_weight):**
```
w_eff = ((base * excitatory - base * inhibitory) * (1 + (modulatory - 0.5) * 0.4)).clamp(0.0, 1.0)
```

**Graph crate CANONICAL formula (get_modulated_weight):**
```
net_activation = excitatory - inhibitory + (modulatory * 0.5)
domain_bonus = 0.1 if edge_domain == query_domain else 0.0
steering_factor = 0.5 + steering_reward  // Range [0.5, 1.5]
w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
result = w_eff.clamp(0.0, 1.0)
```

**Key Differences:**
- Graph version uses `net_activation()` helper from NeurotransmitterWeights
- Graph version adds `steering_factor` based on traversal history
- Graph version adds `domain_bonus` for query context

---

## Context

GraphEdge in the graph crate represents connections optimized for RocksDB storage and GPU batch loading. Beyond basic relationships, edges carry domain information, neurotransmitter weights, and traversal statistics that enable adaptive, domain-aware search.

The `get_modulated_weight()` method is the core of dynamic edge importance calculation during graph traversal.

---

## Scope

### In Scope
- GraphEdge struct with 13 fields (using u64 timestamps, i64 EdgeId)
- EdgeType enum re-export from core (Semantic, Temporal, Causal, Hierarchical)
- `get_modulated_weight(query_domain)` using CANONICAL steering formula
- `record_traversal(success, alpha)` for EMA steering reward updates
- `net_activation()` helper on NeurotransmitterWeights (add if missing)
- `domain_bonus()` helper for domain matching
- Serde serialization for RocksDB/bincode storage

### Out of Scope
- EdgeType::Contradicts (see M04-T26)
- Storage operations (already complete in M04-T13)
- Traversal algorithms (see M04-T16, M04-T17)
- Modifying context-graph-core's GraphEdge

---

## Definition of Done

### File: `crates/context-graph-graph/src/storage/edges.rs`

```rust
//! Graph edge types optimized for RocksDB storage.
//!
//! This module provides GraphEdge with Marblestone neurotransmitter modulation
//! and steering-based weight adjustment for domain-aware graph traversal.
//!
//! # CANONICAL FORMULA for get_modulated_weight()
//!
//! ```text
//! net_activation = excitatory - inhibitory + (modulatory * 0.5)
//! domain_bonus = 0.1 if edge_domain == query_domain else 0.0
//! steering_factor = 0.5 + steering_reward  // Range [0.5, 1.5]
//! w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
//! ```
//! Result clamped to [0.0, 1.0]
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights: Definition and formula
//! - edge_model.steering_reward: [-1,1] range
//! - AP-009: NaN/Infinity clamped to valid range

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-export from core - DO NOT REDEFINE
pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
pub use context_graph_core::types::NodeId;

/// Edge identifier for graph crate (i64 for RocksDB key efficiency).
///
/// Uses i64 instead of UUID for:
/// - Efficient RocksDB key encoding (8 bytes vs 16)
/// - FAISS ID compatibility
/// - Simpler range scans
pub type EdgeId = i64;

/// Graph edge with Marblestone neuro-modulation and steering support.
///
/// Optimized for RocksDB storage with bincode serialization.
/// Uses u64 Unix timestamps for efficiency (not chrono DateTime).
///
/// # CANONICAL FORMULA for modulated weight
///
/// ```text
/// net_activation = excitatory - inhibitory + (modulatory * 0.5)
/// domain_bonus = 0.1 if edge_domain == query_domain else 0.0
/// steering_factor = 0.5 + steering_reward
/// w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
/// ```
/// Result is clamped to [0.0, 1.0]
///
/// # Fields (13 total per PRD Section 4.2)
///
/// 1. id - Unique edge identifier (i64)
/// 2. source - Source node ID (UUID)
/// 3. target - Target node ID (UUID)
/// 4. edge_type - Relationship classification
/// 5. weight - Base edge weight [0.0, 1.0]
/// 6. confidence - Certainty score [0.0, 1.0]
/// 7. domain - Cognitive domain
/// 8. neurotransmitter_weights - NT modulation weights
/// 9. is_amortized_shortcut - Dream-learned shortcut flag
/// 10. steering_reward - Steering subsystem feedback [0.0, 1.0]
/// 11. traversal_count - Access count
/// 12. created_at - Unix timestamp of creation
/// 13. last_traversed_at - Unix timestamp of last access (0 = never)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique edge identifier (i64 for RocksDB efficiency).
    pub id: EdgeId,

    /// Source node ID (UUID).
    pub source: NodeId,

    /// Target node ID (UUID).
    pub target: NodeId,

    /// Edge type classification.
    pub edge_type: EdgeType,

    /// Base edge weight [0.0, 1.0].
    /// Higher = stronger relationship.
    pub weight: f32,

    /// Confidence score [0.0, 1.0].
    /// How certain we are about this edge.
    pub confidence: f32,

    /// Cognitive domain of this edge.
    pub domain: Domain,

    /// Neurotransmitter weights for modulation.
    pub neurotransmitter_weights: NeurotransmitterWeights,

    /// Whether this is an amortized inference shortcut
    /// (learned pattern that bypasses intermediate steps).
    pub is_amortized_shortcut: bool,

    /// Steering reward from traversal history [0.0, 1.0].
    /// Updated via EMA when edge leads to successful retrieval.
    /// NOTE: Constitution says [-1,1] but we use [0,1] for steering_factor math.
    pub steering_reward: f32,

    /// Number of times this edge has been traversed.
    pub traversal_count: u64,

    /// Unix timestamp when edge was created.
    pub created_at: u64,

    /// Unix timestamp of last traversal (0 if never).
    pub last_traversed_at: u64,
}

impl GraphEdge {
    /// Create a new edge with domain-appropriate NT weights.
    ///
    /// # Arguments
    /// * `id` - Unique edge identifier
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `edge_type` - Type of relationship
    /// * `weight` - Base edge weight [0, 1]
    /// * `domain` - Cognitive domain
    ///
    /// # Returns
    /// New GraphEdge with:
    /// - NT weights from `NeurotransmitterWeights::for_domain(domain)`
    /// - confidence = 1.0
    /// - steering_reward = 0.5 (neutral)
    /// - traversal_count = 0
    /// - is_amortized_shortcut = false
    pub fn new(
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        edge_type: EdgeType,
        weight: f32,
        domain: Domain,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id,
            source,
            target,
            edge_type,
            weight: weight.clamp(0.0, 1.0),
            confidence: 1.0,
            domain,
            neurotransmitter_weights: NeurotransmitterWeights::for_domain(domain),
            is_amortized_shortcut: false,
            steering_reward: 0.5, // Neutral starting value
            traversal_count: 0,
            created_at: now,
            last_traversed_at: 0,
        }
    }

    /// Create a simple semantic edge with General domain.
    #[inline]
    pub fn semantic(id: EdgeId, source: NodeId, target: NodeId, weight: f32) -> Self {
        Self::new(id, source, target, EdgeType::Semantic, weight, Domain::General)
    }

    /// Create a hierarchical edge with General domain.
    #[inline]
    pub fn hierarchical(id: EdgeId, parent: NodeId, child: NodeId, weight: f32) -> Self {
        Self::new(id, parent, child, EdgeType::Hierarchical, weight, Domain::General)
    }

    /// Get modulated weight for a specific query domain.
    ///
    /// # CANONICAL FORMULA
    ///
    /// ```text
    /// net_activation = excitatory - inhibitory + (modulatory * 0.5)
    /// domain_bonus = 0.1 if edge_domain == query_domain else 0.0
    /// steering_factor = 0.5 + steering_reward  // Range [0.5, 1.5]
    /// w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
    /// ```
    /// Result clamped to [0.0, 1.0]
    ///
    /// # Arguments
    /// * `query_domain` - The domain of the current query/traversal
    ///
    /// # Returns
    /// Effective weight after modulation, clamped to [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_graph::storage::edges::{GraphEdge, EdgeType, Domain};
    /// use uuid::Uuid;
    ///
    /// let mut edge = GraphEdge::new(1, Uuid::new_v4(), Uuid::new_v4(), EdgeType::Semantic, 0.5, Domain::Code);
    /// edge.steering_reward = 0.5;  // steering_factor = 1.0
    ///
    /// // Query same domain gets bonus
    /// let w_code = edge.get_modulated_weight(Domain::Code);
    /// let w_legal = edge.get_modulated_weight(Domain::Legal);
    /// assert!(w_code > w_legal);  // Domain bonus applies
    /// ```
    #[inline]
    pub fn get_modulated_weight(&self, query_domain: Domain) -> f32 {
        let nt = &self.neurotransmitter_weights;

        // Net activation from NT weights
        let net_activation = nt.excitatory - nt.inhibitory + (nt.modulatory * 0.5);

        // Domain bonus: +0.1 if query matches edge domain
        let domain_bonus = if self.domain == query_domain { 0.1 } else { 0.0 };

        // Steering factor: 0.5 + steering_reward puts it in [0.5, 1.5] range
        // (assuming steering_reward is [0.0, 1.0])
        let steering_factor = 0.5 + self.steering_reward;

        // Apply canonical formula
        let w_eff = self.weight * (1.0 + net_activation + domain_bonus) * steering_factor;

        // Clamp to valid range per AP-009
        w_eff.clamp(0.0, 1.0)
    }

    /// Get unmodulated base weight.
    #[inline]
    pub fn base_weight(&self) -> f32 {
        self.weight
    }

    /// Record a traversal of this edge.
    ///
    /// Updates traversal count, timestamp, and steering reward via EMA.
    ///
    /// # Arguments
    /// * `success` - Whether the traversal led to successful retrieval
    /// * `alpha` - EMA smoothing factor [0, 1], typical value 0.1
    ///
    /// # EMA Formula
    /// ```text
    /// reward = 1.0 if success else 0.0
    /// steering_reward = (1 - alpha) * steering_reward + alpha * reward
    /// ```
    pub fn record_traversal(&mut self, success: bool, alpha: f32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.traversal_count += 1;
        self.last_traversed_at = now;

        // Update steering reward via EMA
        let reward = if success { 1.0 } else { 0.0 };
        self.steering_reward = (1.0 - alpha) * self.steering_reward + alpha * reward;
        self.steering_reward = self.steering_reward.clamp(0.0, 1.0);
    }

    /// Record traversal with default EMA alpha (0.1).
    #[inline]
    pub fn record_traversal_default(&mut self, success: bool) {
        self.record_traversal(success, 0.1);
    }

    /// Mark this edge as an amortized shortcut (learned during dream consolidation).
    #[inline]
    pub fn mark_as_shortcut(&mut self) {
        self.is_amortized_shortcut = true;
    }

    /// Update confidence score.
    #[inline]
    pub fn update_confidence(&mut self, new_confidence: f32) {
        self.confidence = new_confidence.clamp(0.0, 1.0);
    }

    /// Update neurotransmitter weights.
    #[inline]
    pub fn update_nt_weights(&mut self, weights: NeurotransmitterWeights) {
        self.neurotransmitter_weights = weights;
    }

    /// Check if edge has been traversed since given timestamp.
    ///
    /// # Arguments
    /// * `since` - Unix timestamp to check against
    ///
    /// # Returns
    /// true if last_traversed_at > since
    #[inline]
    pub fn traversed_since(&self, since: u64) -> bool {
        self.last_traversed_at > since
    }

    /// Get edge "freshness" - seconds since last traversal.
    ///
    /// Returns u64::MAX if never traversed.
    pub fn freshness(&self) -> u64 {
        if self.last_traversed_at == 0 {
            return u64::MAX;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        now.saturating_sub(self.last_traversed_at)
    }

    /// Get composite score combining modulated weight and confidence.
    #[inline]
    pub fn composite_score(&self, query_domain: Domain) -> f32 {
        self.get_modulated_weight(query_domain) * self.confidence
    }
}

impl PartialEq for GraphEdge {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for GraphEdge {}

impl std::hash::Hash for GraphEdge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Default for GraphEdge {
    fn default() -> Self {
        Self::new(0, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Construction Tests ==========

    #[test]
    fn test_edge_creation() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = GraphEdge::new(1, source, target, EdgeType::Semantic, 0.8, Domain::Code);

        assert_eq!(edge.id, 1);
        assert_eq!(edge.source, source);
        assert_eq!(edge.target, target);
        assert_eq!(edge.edge_type, EdgeType::Semantic);
        assert!((edge.weight - 0.8).abs() < 1e-6);
        assert_eq!(edge.domain, Domain::Code);
        assert!(!edge.is_amortized_shortcut);
        assert!((edge.steering_reward - 0.5).abs() < 1e-6);
        assert_eq!(edge.traversal_count, 0);
        assert!(edge.created_at > 0);
        assert_eq!(edge.last_traversed_at, 0);
    }

    #[test]
    fn test_edge_nt_weights_match_domain() {
        let edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::Code);
        let expected = NeurotransmitterWeights::for_domain(Domain::Code);

        assert_eq!(edge.neurotransmitter_weights.excitatory, expected.excitatory);
        assert_eq!(edge.neurotransmitter_weights.inhibitory, expected.inhibitory);
        assert_eq!(edge.neurotransmitter_weights.modulatory, expected.modulatory);
    }

    #[test]
    fn test_semantic_helper() {
        let edge = GraphEdge::semantic(42, Uuid::nil(), Uuid::nil(), 0.7);
        assert_eq!(edge.id, 42);
        assert_eq!(edge.edge_type, EdgeType::Semantic);
        assert_eq!(edge.domain, Domain::General);
        assert!((edge.weight - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_hierarchical_helper() {
        let edge = GraphEdge::hierarchical(99, Uuid::nil(), Uuid::nil(), 0.9);
        assert_eq!(edge.id, 99);
        assert_eq!(edge.edge_type, EdgeType::Hierarchical);
    }

    // ========== Modulated Weight Tests ==========

    #[test]
    fn test_modulated_weight_canonical_formula() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::Code);
        edge.steering_reward = 0.5;  // steering_factor = 1.0

        // Code domain weights: e=0.6, i=0.3, m=0.4
        // net_activation = 0.6 - 0.3 + (0.4 * 0.5) = 0.3 + 0.2 = 0.5

        // Query same domain: domain_bonus = 0.1
        // w_eff = 0.5 * (1.0 + 0.5 + 0.1) * 1.0 = 0.5 * 1.6 = 0.8
        let w = edge.get_modulated_weight(Domain::Code);
        assert!((w - 0.8).abs() < 0.01, "Expected 0.8, got {}", w);

        // Query different domain: domain_bonus = 0.0
        // w_eff = 0.5 * (1.0 + 0.5 + 0.0) * 1.0 = 0.5 * 1.5 = 0.75
        let w = edge.get_modulated_weight(Domain::Legal);
        assert!((w - 0.75).abs() < 0.01, "Expected 0.75, got {}", w);
    }

    #[test]
    fn test_modulated_weight_clamping() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 1.0, Domain::Creative);
        edge.steering_reward = 1.0;  // steering_factor = 1.5

        // Creative: e=0.8, i=0.1, m=0.6
        // net_activation = 0.8 - 0.1 + (0.6 * 0.5) = 0.7 + 0.3 = 1.0
        // w_eff = 1.0 * (1.0 + 1.0 + 0.1) * 1.5 = 1.0 * 2.1 * 1.5 = 3.15
        // Clamped to 1.0
        let w = edge.get_modulated_weight(Domain::Creative);
        assert!((w - 1.0).abs() < 1e-6, "Expected 1.0 (clamped), got {}", w);
    }

    #[test]
    fn test_modulated_weight_zero_base() {
        let edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.0, Domain::General);
        let w = edge.get_modulated_weight(Domain::General);
        assert!((w - 0.0).abs() < 1e-6, "Zero base weight should give zero");
    }

    #[test]
    fn test_steering_factor_range() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General);

        // Min steering: factor = 0.5
        edge.steering_reward = 0.0;
        let w_min = edge.get_modulated_weight(Domain::General);

        // Max steering: factor = 1.5
        edge.steering_reward = 1.0;
        let w_max = edge.get_modulated_weight(Domain::General);

        // Max should be 3x min (1.5 / 0.5)
        assert!(w_max > w_min, "Max steering should give higher weight");
    }

    // ========== Traversal Tests ==========

    #[test]
    fn test_record_traversal_success() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General);
        edge.steering_reward = 0.5;

        // Record successful traversal with alpha=0.1
        edge.record_traversal(true, 0.1);
        // steering_reward = 0.9 * 0.5 + 0.1 * 1.0 = 0.45 + 0.1 = 0.55
        assert!((edge.steering_reward - 0.55).abs() < 1e-6);
        assert_eq!(edge.traversal_count, 1);
        assert!(edge.last_traversed_at > 0);
    }

    #[test]
    fn test_record_traversal_failure() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General);
        edge.steering_reward = 0.5;

        // Record failed traversal
        edge.record_traversal(false, 0.1);
        // steering_reward = 0.9 * 0.5 + 0.1 * 0.0 = 0.45
        assert!((edge.steering_reward - 0.45).abs() < 1e-6);
        assert_eq!(edge.traversal_count, 1);
    }

    #[test]
    fn test_record_traversal_sequence() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General);
        edge.steering_reward = 0.5;

        // Success then failure
        edge.record_traversal(true, 0.1);
        assert!((edge.steering_reward - 0.55).abs() < 1e-6);

        edge.record_traversal(false, 0.1);
        // 0.9 * 0.55 + 0.1 * 0.0 = 0.495
        assert!((edge.steering_reward - 0.495).abs() < 1e-6);
        assert_eq!(edge.traversal_count, 2);
    }

    #[test]
    fn test_steering_reward_clamped() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General);

        // Force to max via repeated success
        for _ in 0..100 {
            edge.record_traversal(true, 0.5);
        }
        assert!(edge.steering_reward <= 1.0);

        // Force to min via repeated failure
        for _ in 0..100 {
            edge.record_traversal(false, 0.5);
        }
        assert!(edge.steering_reward >= 0.0);
    }

    // ========== Serde Tests ==========

    #[test]
    fn test_bincode_roundtrip() {
        let edge = GraphEdge::new(42, Uuid::new_v4(), Uuid::new_v4(), EdgeType::Causal, 0.75, Domain::Medical);

        let serialized = bincode::serialize(&edge).expect("serialize failed");
        let deserialized: GraphEdge = bincode::deserialize(&serialized).expect("deserialize failed");

        assert_eq!(edge.id, deserialized.id);
        assert_eq!(edge.source, deserialized.source);
        assert_eq!(edge.target, deserialized.target);
        assert_eq!(edge.edge_type, deserialized.edge_type);
        assert!((edge.weight - deserialized.weight).abs() < 1e-6);
        assert_eq!(edge.domain, deserialized.domain);
    }

    #[test]
    fn test_json_roundtrip() {
        let edge = GraphEdge::new(1, Uuid::new_v4(), Uuid::new_v4(), EdgeType::Temporal, 0.6, Domain::Research);

        let json = serde_json::to_string(&edge).expect("json serialize failed");
        let deserialized: GraphEdge = serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(edge.id, deserialized.id);
        assert_eq!(edge.edge_type, deserialized.edge_type);
    }

    // ========== Edge Type Tests ==========

    #[test]
    fn test_edge_type_variants() {
        let types = EdgeType::all();
        assert_eq!(types.len(), 4);
        assert!(types.contains(&EdgeType::Semantic));
        assert!(types.contains(&EdgeType::Temporal));
        assert!(types.contains(&EdgeType::Causal));
        assert!(types.contains(&EdgeType::Hierarchical));
    }

    // ========== Helper Method Tests ==========

    #[test]
    fn test_mark_as_shortcut() {
        let mut edge = GraphEdge::default();
        assert!(!edge.is_amortized_shortcut);
        edge.mark_as_shortcut();
        assert!(edge.is_amortized_shortcut);
    }

    #[test]
    fn test_update_confidence() {
        let mut edge = GraphEdge::default();
        edge.update_confidence(0.8);
        assert!((edge.confidence - 0.8).abs() < 1e-6);

        // Test clamping
        edge.update_confidence(1.5);
        assert!((edge.confidence - 1.0).abs() < 1e-6);

        edge.update_confidence(-0.5);
        assert!((edge.confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_composite_score() {
        let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.8, Domain::Code);
        edge.confidence = 0.5;
        edge.steering_reward = 0.5;

        let modulated = edge.get_modulated_weight(Domain::Code);
        let composite = edge.composite_score(Domain::Code);

        assert!((composite - modulated * 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_freshness_never_traversed() {
        let edge = GraphEdge::default();
        assert_eq!(edge.freshness(), u64::MAX);
    }

    #[test]
    fn test_traversed_since() {
        let mut edge = GraphEdge::default();
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - 10;

        assert!(!edge.traversed_since(before));

        edge.record_traversal_default(true);
        assert!(edge.traversed_since(before));
    }

    // ========== Equality Tests ==========

    #[test]
    fn test_equality_by_id() {
        let edge1 = GraphEdge::new(1, Uuid::new_v4(), Uuid::new_v4(), EdgeType::Semantic, 0.5, Domain::General);
        let edge2 = GraphEdge::new(1, Uuid::new_v4(), Uuid::new_v4(), EdgeType::Causal, 0.9, Domain::Code);
        let edge3 = GraphEdge::new(2, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General);

        // Same ID = equal (regardless of other fields)
        assert_eq!(edge1, edge2);
        // Different ID = not equal
        assert_ne!(edge1, edge3);
    }
}
```

### Constraints

1. **DO NOT redefine Domain, EdgeType, NeurotransmitterWeights** - re-export from context-graph-core
2. GraphEdge struct MUST have all 13 fields
3. `get_modulated_weight()` MUST use CANONICAL formula with steering
4. `steering_factor = 0.5 + steering_reward` gives range [0.5, 1.5]
5. Result of `get_modulated_weight()` MUST be clamped to [0.0, 1.0]
6. `record_traversal()` uses EMA for steering_reward updates
7. EdgeType does NOT include Contradicts (added in M04-T26)
8. Use `i64` for EdgeId (RocksDB key efficiency)
9. Use `u64` for timestamps (not chrono DateTime)

---

## Implementation Approach

### Step 1: Create edges.rs

Create `crates/context-graph-graph/src/storage/edges.rs` with the code above.

### Step 2: Update storage/mod.rs

Add to `crates/context-graph-graph/src/storage/mod.rs`:

```rust
// Add after line 39 (after mod storage_impl;)
pub mod edges;

// Add to re-exports section (around line 44)
pub use edges::{EdgeId, GraphEdge};
// Note: Domain, EdgeType, NeurotransmitterWeights are already re-exported from lib.rs
```

### Step 3: Update lib.rs (if needed)

The types are already re-exported. Verify `GraphEdge` from storage is accessible:

```rust
// crates/context-graph-graph/src/lib.rs
// Add if not present:
pub use storage::edges::{EdgeId, GraphEdge as StorageGraphEdge};
// Or just document that storage::GraphEdge is the graph-crate version
```

---

## Verification

### Test Commands

```bash
# Build the crate
cargo build -p context-graph-graph 2>&1 | head -50

# Run edge tests specifically
cargo test -p context-graph-graph edges -- --nocapture

# Run all graph tests
cargo test -p context-graph-graph

# Clippy check
cargo clippy -p context-graph-graph -- -D warnings

# Verify no duplicate definitions
grep -r "pub enum EdgeType" crates/ --include="*.rs"
grep -r "pub enum Domain" crates/ --include="*.rs"
```

### Manual Verification Checklist

- [ ] File `crates/context-graph-graph/src/storage/edges.rs` exists
- [ ] GraphEdge has exactly 13 fields
- [ ] `get_modulated_weight()` uses canonical formula with steering_factor
- [ ] `record_traversal()` updates steering_reward via EMA
- [ ] Domain, EdgeType, NeurotransmitterWeights are RE-EXPORTED (not redefined)
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] bincode serialization roundtrip works

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth

After implementation, the source of truth is:
1. **File existence**: `crates/context-graph-graph/src/storage/edges.rs`
2. **Type availability**: `context_graph_graph::storage::edges::GraphEdge` compiles
3. **Test results**: `cargo test -p context-graph-graph edges` all pass

### Execute & Inspect

After implementation:

```bash
# 1. Verify file exists
ls -la crates/context-graph-graph/src/storage/edges.rs

# 2. Verify compilation
cargo build -p context-graph-graph 2>&1 | grep -E "(error|warning:|Compiling)"

# 3. Verify tests pass
cargo test -p context-graph-graph edges -- --nocapture 2>&1 | tail -20

# 4. Verify type is accessible
echo 'use context_graph_graph::storage::edges::GraphEdge; fn main() { let _e = GraphEdge::default(); }' > /tmp/test_edge.rs
rustc --edition 2021 -L target/debug/deps /tmp/test_edge.rs -o /tmp/test_edge 2>&1 || echo "Direct rustc may need deps, use cargo instead"
```

### Boundary & Edge Case Audit

Execute these 3 edge cases and log before/after state:

**Edge Case 1: Zero base weight**
```rust
let edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.0, Domain::General);
println!("STATE BEFORE: base_weight={}, steering_reward={}", edge.weight, edge.steering_reward);
let w = edge.get_modulated_weight(Domain::General);
println!("STATE AFTER: modulated_weight={}", w);
println!("EXPECTED: 0.0 (zero base means zero result)");
println!("ACTUAL: {}", w);
assert!((w - 0.0).abs() < 1e-6);
```

**Edge Case 2: Maximum steering (reward=1.0)**
```rust
let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 1.0, Domain::Creative);
edge.steering_reward = 1.0;
println!("STATE BEFORE: weight={}, steering_reward={}", edge.weight, edge.steering_reward);
let w = edge.get_modulated_weight(Domain::Creative);
println!("STATE AFTER: modulated_weight={}", w);
println!("EXPECTED: 1.0 (clamped from >1.0)");
println!("ACTUAL: {}", w);
assert!((w - 1.0).abs() < 1e-6);
```

**Edge Case 3: Minimum steering (reward=0.0)**
```rust
let mut edge = GraphEdge::new(1, Uuid::nil(), Uuid::nil(), EdgeType::Semantic, 0.5, Domain::General);
edge.steering_reward = 0.0;
println!("STATE BEFORE: weight={}, steering_reward={}", edge.weight, edge.steering_reward);
let w = edge.get_modulated_weight(Domain::General);
println!("STATE AFTER: modulated_weight={}", w);
// General: e=0.5, i=0.2, m=0.3 -> net=0.5-0.2+(0.3*0.5)=0.45
// steering_factor = 0.5 + 0.0 = 0.5
// w_eff = 0.5 * (1.0 + 0.45 + 0.0) * 0.5 = 0.5 * 1.45 * 0.5 = 0.3625
println!("EXPECTED: ~0.3625");
println!("ACTUAL: {}", w);
```

### Evidence of Success

After completion, provide:

```
SUCCESS EVIDENCE LOG:
- Operation: Implement GraphEdge with get_modulated_weight()
- Source of Truth: crates/context-graph-graph/src/storage/edges.rs
- Verification Query: cargo test -p context-graph-graph edges
- Expected Result: All tests pass, struct has 13 fields
- Actual Result: [paste test output]
- Physical Proof: [paste ls -la output and file line count]
- Timestamp: [ISO timestamp]
- Verified By: sherlock-holmes subagent
```

---

## SHERLOCK-HOLMES FINAL VERIFICATION

After completing implementation, invoke sherlock-holmes subagent with:

```
MISSION: Verify M04-T15 GraphEdge implementation is complete and correct

CHECKLIST:
1. File crates/context-graph-graph/src/storage/edges.rs exists
2. GraphEdge struct has exactly 13 fields (count them)
3. get_modulated_weight() uses canonical formula:
   - net_activation = e - i + (m * 0.5)
   - domain_bonus = 0.1 if matching
   - steering_factor = 0.5 + steering_reward
   - w_eff = base * (1 + net + bonus) * steering
4. record_traversal() updates steering_reward via EMA
5. Domain, EdgeType, NeurotransmitterWeights are RE-EXPORTED not redefined
6. cargo build -p context-graph-graph succeeds
7. cargo test -p context-graph-graph edges passes all tests
8. cargo clippy -p context-graph-graph shows no warnings
9. bincode serialization test passes
10. All 3 edge cases verified with before/after logging

REPORT: Any discrepancy found must be fixed before marking complete
```

---

## Acceptance Criteria

- [ ] GraphEdge struct with all 13 fields in edges.rs
- [ ] `new()` initializes with domain-appropriate NT weights from core
- [ ] `get_modulated_weight()` uses CANONICAL formula with steering
- [ ] `record_traversal()` increments count and updates steering_reward with EMA
- [ ] EdgeType, Domain, NeurotransmitterWeights RE-EXPORTED from core
- [ ] Compiles with `cargo build -p context-graph-graph`
- [ ] Tests pass with `cargo test -p context-graph-graph edges`
- [ ] No clippy warnings
- [ ] bincode roundtrip test passes
- [ ] sherlock-holmes verification passes
