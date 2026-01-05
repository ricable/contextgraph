//! TeleologicalFingerprint type definition.
//!
//! This module contains the struct definition for the complete teleological fingerprint.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::fingerprint::evolution::PurposeSnapshot;
use crate::types::fingerprint::johari::JohariFingerprint;
use crate::types::fingerprint::purpose::PurposeVector;
use crate::types::fingerprint::SemanticFingerprint;

/// Complete teleological fingerprint for a memory node.
///
/// This struct combines semantic content (what) with purpose (why),
/// enabling retrieval that considers both similarity and goal alignment.
///
/// From constitution.yaml:
/// - Expected size: ~46KB per node
/// - MAX_EVOLUTION_SNAPSHOTS: 100 (older snapshots archived to TimescaleDB)
/// - Misalignment warning: delta_A < -0.15 predicts failure 72 hours ahead
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalFingerprint {
    /// Unique identifier for this fingerprint (UUID v4)
    pub id: Uuid,

    /// The 13-embedding semantic fingerprint (from TASK-F001)
    pub semantic: SemanticFingerprint,

    /// 13D alignment signature to North Star goal
    pub purpose_vector: PurposeVector,

    /// Per-embedder Johari awareness classification
    pub johari: JohariFingerprint,

    /// Time-series of purpose evolution snapshots
    pub purpose_evolution: Vec<PurposeSnapshot>,

    /// Current alignment angle to North Star goal (aggregate)
    pub theta_to_north_star: f32,

    /// SHA-256 hash of the source content (32 bytes)
    pub content_hash: [u8; 32],

    /// When this fingerprint was created
    pub created_at: DateTime<Utc>,

    /// When this fingerprint was last updated
    pub last_updated: DateTime<Utc>,

    /// Number of times this memory has been accessed
    pub access_count: u64,
}
