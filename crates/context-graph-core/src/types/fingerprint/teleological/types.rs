//! TeleologicalFingerprint type definition.
//!
//! This module contains the struct definition for the complete teleological fingerprint.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::fingerprint::{SemanticFingerprint, SparseVector};

/// Complete teleological fingerprint for a memory node.
///
/// This struct combines semantic content with metadata for tracking
/// and retrieval.
///
/// From constitution.yaml:
/// - Expected size: ~46KB per node (+ ~2KB if e6_sparse is present)
///
/// # E6 Sparse Vector
///
/// The optional `e6_sparse` field stores the original sparse vector from E6
/// (V_selectivity) embedder before projection to dense. This enables:
/// - Stage 1 sparse recall via inverted index
/// - Exact keyword matching for technical queries
/// - E6 tie-breaking when E1 scores are close
///
/// See docs/e6upgrade.md for the full E6 enhancement proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalFingerprint {
    /// Unique identifier for this fingerprint (UUID v4)
    pub id: Uuid,

    /// The 13-embedding semantic fingerprint (from TASK-F001)
    pub semantic: SemanticFingerprint,

    /// SHA-256 hash of the source content (32 bytes)
    pub content_hash: [u8; 32],

    /// When this fingerprint was created
    pub created_at: DateTime<Utc>,

    /// When this fingerprint was last updated
    pub last_updated: DateTime<Utc>,

    /// Number of times this memory has been accessed
    pub access_count: u64,

    /// Importance score [0.0, 1.0] for memory prioritization.
    /// Used by consolidation, boost_importance, and dream phases.
    /// Default: 0.5
    pub importance: f32,

    /// Original E6 sparse vector for Stage 1 recall and keyword matching.
    ///
    /// This field is optional for backward compatibility with existing fingerprints.
    /// New fingerprints should populate this during embedding generation.
    ///
    /// - Typical size: ~235 active terms (~1.4KB)
    /// - Used for: inverted index recall, exact term matching, tie-breaking
    /// - NOT used for: semantic similarity (use projected dense in `semantic`)
    ///
    /// NOTE: We use #[serde(default)] but NOT skip_serializing_if because bincode
    /// uses a fixed format and doesn't support field skipping. All fields must
    /// be serialized for bincode compatibility.
    #[serde(default)]
    pub e6_sparse: Option<SparseVector>,
}
