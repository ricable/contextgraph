//! TeleologicalFingerprint type definition.
//!
//! This module contains the struct definition for the complete teleological fingerprint.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::fingerprint::SemanticFingerprint;

/// Complete teleological fingerprint for a memory node.
///
/// This struct combines semantic content with metadata for tracking
/// and retrieval.
///
/// From constitution.yaml:
/// - Expected size: ~46KB per node
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
}
