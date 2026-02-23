//! RVF (RuVector Format) bridge module for context-graph-storage.
//!
//! This module provides integration between context-graph-storage's existing usearch HNSW
//! and the RVF cognitive container format. It supports:
//! - Dual-write: write to both usearch and RVF
//! - Progressive recall: Layer A (70%, ef=16) → B (85%, ef=64) → C (95%, ef=256)
//! - COW branching: derive child stores from parent
//! - SONA learning hooks for adaptive retrieval
//! - Hyperbolic (Poincaré) distance for document hierarchy
//!
//! # Segment Types
//!
//! | Segment | Code | Purpose |
//! |---------|------|---------|
//! | VEC_SEG | 0x01 | Raw vector data |
//! | INDEX_SEG | 0x02 | HNSW progressive index |
//! | META_SEG | 0x03 | Vector metadata |
//! | OVERLAY_SEG | 0x05 | LoRA adapter deltas (SONA Loop B) |
//! | GRAPH_SEG | 0x06 | Property graph (SONA Loop C) |
//! | MODEL_SEG | 0x09 | ML model weights |
//! | CRYPTO_SEG | 0x0A | Signatures and key material |
//! | WITNESS_SEG | 0x0B | Append-only witness/audit chain |
//! | WASM_SEG | 0x08 | Embedded WASM modules |
//! | COW_MAP_SEG | 0x20 | Copy-on-write cluster map |
//!
//! # Quick Start
//!
//! ```ignore
//! use context_graph_storage::rvf::{RvfBridge, RvfConfig};
//!
//! let bridge = RvfBridge::new(RvfConfig::default()).await?;
//! bridge.store_memory(id, &vector, metadata).await?;
//! let results = bridge.search(&query, top_k).await?;
//! ```

pub mod bridge;
pub mod client;
pub mod segments;
pub mod sona;

pub use bridge::{
    BridgeSearchResult, CowFilterType, ProgressiveRecallLayer, ResultSource, RvfBridge,
    RvfBridgeConfig, RvfBridgeError, RvfBridgeResult, RvfBridgeStatus,
};
pub use client::{RvfClient, RvfClientConfig, RvfClientError, RvfClientResult, RvfFileIdentity, RvfSearchResult};
pub use segments::{RvfSegment, RvfSegmentHeader, RvfSegmentStats, RvfSegmentType};
pub use sona::{
    SonaConfidence, SonaConfig, SonaFeedback, SonaLearning, SonaLoop, SonaRecommendation,
    SonaState,
};
