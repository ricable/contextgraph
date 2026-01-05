# Technical Specification: Multi-Array Teleological Fingerprint

**Document ID**: TECH-SPEC-001
**Version**: 1.0.0
**Status**: Draft
**Date**: 2026-01-04
**Traces To**: FUNC-SPEC-001
**Implements**: FR-100 through FR-600

---

## 1. Data Structures (TS-100 Series)

### 1.1 SemanticFingerprint (TS-101)

**Requirement**: FR-101, FR-102, FR-103, FR-104
**Module**: `crates/context-graph-core/src/types/fingerprint/semantic.rs`

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Total dimension count across all embedders (excluding variable E12)
pub const TOTAL_DENSE_DIMS: usize = 1024 + 512 + 512 + 512 + 768 + 1536 + 384 + 1024 + 768 + 384;

/// Sparse vector with ~5% active indices from 30K vocabulary
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseVector30K {
    /// Active dimension indices (typically ~1500 of 30000)
    pub indices: Vec<u16>,
    /// Activation values for active dimensions (same length as indices)
    pub values: Vec<f32>,
}

impl SparseVector30K {
    /// Maximum vocabulary size for sparse embeddings
    pub const VOCAB_SIZE: usize = 30_000;

    /// Expected sparsity ratio (5% active)
    pub const SPARSITY: f32 = 0.05;

    /// Create new sparse vector with validation
    pub fn new(indices: Vec<u16>, values: Vec<f32>) -> Result<Self, &'static str> {
        if indices.len() != values.len() {
            return Err("indices and values must have same length");
        }
        if indices.iter().any(|&i| i as usize >= Self::VOCAB_SIZE) {
            return Err("index out of vocabulary bounds");
        }
        Ok(Self { indices, values })
    }

    /// Compute sparse similarity (Jaccard + weighted overlap)
    pub fn similarity(&self, other: &Self) -> f32 {
        use std::collections::HashSet;

        let self_set: HashSet<u16> = self.indices.iter().copied().collect();
        let other_set: HashSet<u16> = other.indices.iter().copied().collect();

        let intersection: HashSet<_> = self_set.intersection(&other_set).collect();
        let union_size = self_set.len() + other_set.len() - intersection.len();

        if union_size == 0 {
            return 0.0;
        }

        let jaccard = intersection.len() as f32 / union_size as f32;

        // Weighted overlap by activation values
        let weighted_overlap: f32 = intersection
            .iter()
            .filter_map(|&&idx| {
                let self_pos = self.indices.iter().position(|&i| i == idx)?;
                let other_pos = other.indices.iter().position(|&i| i == idx)?;
                Some(self.values[self_pos] * other.values[other_pos])
            })
            .sum();

        0.5 * jaccard + 0.5 * weighted_overlap.min(1.0)
    }
}

/// SemanticFingerprint - THE core struct replacing Vec<f32>
///
/// Stores all 12 embeddings without fusion, preserving 100% semantic information.
/// Total storage: ~46KB per memory (vs 6KB for legacy fused Vector1536)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFingerprint {
    /// E1: text-embedding-3-large (general semantic meaning)
    /// Dimension: 1024
    pub e1_text_general: [f32; 1024],

    /// E2: text-embedding-3-small (compressed semantic)
    /// Dimension: 512
    pub e2_text_small: [f32; 512],

    /// E3: multilingual-e5-base (cross-lingual semantics)
    /// Dimension: 512
    pub e3_multilingual: [f32; 512],

    /// E4: codet5p-110m (code semantics)
    /// Dimension: 512
    pub e4_code: [f32; 512],

    /// E5: Asymmetric query/document embeddings (retrieval optimization)
    /// Dimension: (768, 768) = 1536 total
    pub e5_query_doc: ([f32; 768], [f32; 768]),

    /// E6: splade-v3 sparse (lexical-semantic hybrid)
    /// Dimension: ~1500 active of 30K vocabulary
    pub e6_sparse: SparseVector30K,

    /// E7: text-embedding-ada-002 (OpenAI general purpose)
    /// Dimension: 1536
    pub e7_openai_ada: [f32; 1536],

    /// E8: all-MiniLM-L6 (lightweight dense)
    /// Dimension: 384
    pub e8_minilm: [f32; 384],

    /// E9: SimHash projected from 10Kbit binary hypervector
    /// Dimension: 1024 (projected from 10K-bit)
    pub e9_simhash: [f32; 1024],

    /// E10: instructor-xl (instruction-following embeddings)
    /// Dimension: 768
    pub e10_instructor: [f32; 768],

    /// E11: e5-small-v2 (fast dense retrieval)
    /// Dimension: 384
    pub e11_fast: [f32; 384],

    /// E12: ColBERT-style per-token late interaction
    /// Dimension: 128 per token, variable length
    pub e12_token_level: Vec<[f32; 128]>,
}

impl SemanticFingerprint {
    /// Calculate total storage size in bytes
    pub fn storage_size(&self) -> usize {
        let dense_size = 4 * (
            1024 +   // e1
            512 +    // e2
            512 +    // e3
            512 +    // e4
            768 * 2 +// e5 (query + doc)
            1536 +   // e7
            384 +    // e8
            1024 +   // e9
            768 +    // e10
            384      // e11
        );
        let sparse_size = (self.e6_sparse.indices.len() * 2) + (self.e6_sparse.values.len() * 4);
        let late_interaction_size = self.e12_token_level.len() * 128 * 4;

        dense_size + sparse_size + late_interaction_size
    }

    /// Create empty fingerprint (for initialization)
    pub fn zeroed() -> Self {
        Self {
            e1_text_general: [0.0; 1024],
            e2_text_small: [0.0; 512],
            e3_multilingual: [0.0; 512],
            e4_code: [0.0; 512],
            e5_query_doc: ([0.0; 768], [0.0; 768]),
            e6_sparse: SparseVector30K { indices: vec![], values: vec![] },
            e7_openai_ada: [0.0; 1536],
            e8_minilm: [0.0; 384],
            e9_simhash: [0.0; 1024],
            e10_instructor: [0.0; 768],
            e11_fast: [0.0; 384],
            e12_token_level: vec![],
        }
    }

    /// Get embedding by index (0-11)
    pub fn get_embedding(&self, idx: usize) -> Option<&[f32]> {
        match idx {
            0 => Some(&self.e1_text_general),
            1 => Some(&self.e2_text_small),
            2 => Some(&self.e3_multilingual),
            3 => Some(&self.e4_code),
            4 => Some(&self.e5_query_doc.0), // Query embedding
            5 => Some(&self.e7_openai_ada),
            6 => Some(&self.e8_minilm),
            7 => Some(&self.e9_simhash),
            8 => Some(&self.e10_instructor),
            9 => Some(&self.e11_fast),
            _ => None,
        }
    }
}
```

---

### 1.2 TeleologicalFingerprint (TS-102)

**Requirement**: FR-201, FR-202, FR-203, FR-204
**Module**: `crates/context-graph-core/src/types/fingerprint/teleological.rs`

```rust
use super::semantic::SemanticFingerprint;
use crate::types::johari::JohariQuadrant;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Number of embedders in the system
pub const NUM_EMBEDDERS: usize = 12;

/// Alignment threshold classifications (from Royse 2026)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignmentThreshold {
    /// theta >= 0.75: Strongly aligned to purpose
    Optimal,
    /// theta in [0.70, 0.75): Adequately aligned
    Acceptable,
    /// theta in [0.55, 0.70): Alignment degrading
    Warning,
    /// theta < 0.55: Misalignment requiring attention
    Critical,
}

impl AlignmentThreshold {
    /// Classify alignment score into threshold category
    pub fn classify(theta: f32) -> Self {
        match theta {
            t if t >= 0.75 => Self::Optimal,
            t if t >= 0.70 => Self::Acceptable,
            t if t >= 0.55 => Self::Warning,
            _ => Self::Critical,
        }
    }

    /// Check if this threshold indicates misalignment
    pub fn is_misaligned(&self) -> bool {
        matches!(self, Self::Warning | Self::Critical)
    }
}

/// 12-dimensional purpose vector: alignment to North Star per embedding space
/// Formula: PV = [A(E1,V), A(E2,V), ..., A(E12,V)]
/// where A(Ei,V) = cos(Ei, V) is alignment of embedder i to North Star V
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeVector {
    /// Per-embedder alignment scores [0.0, 1.0]
    pub alignments: [f32; NUM_EMBEDDERS],

    /// Which embedding space dominates (1-12)
    pub dominant_embedder: u8,

    /// How coherent are all spaces with each other? [0.0, 1.0]
    pub coherence: f32,

    /// Stability over time [0.0, 1.0]
    pub stability: f32,
}

impl PurposeVector {
    /// Compute aggregate alignment across all spaces
    pub fn aggregate_alignment(&self) -> f32 {
        self.alignments.iter().sum::<f32>() / NUM_EMBEDDERS as f32
    }

    /// Get the threshold classification for aggregate alignment
    pub fn threshold_status(&self) -> AlignmentThreshold {
        AlignmentThreshold::classify(self.aggregate_alignment())
    }

    /// Find the dominant (highest alignment) embedder index
    pub fn find_dominant(&self) -> u8 {
        self.alignments
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u8 + 1)
            .unwrap_or(1)
    }

    /// Compute cosine similarity with another purpose vector (12D)
    pub fn similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.alignments.iter()
            .zip(other.alignments.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_self: f32 = self.alignments.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other: f32 = other.alignments.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_self == 0.0 || norm_other == 0.0 {
            return 0.0;
        }

        dot / (norm_self * norm_other)
    }
}

/// Trigger event for purpose evolution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionTrigger {
    /// Initial creation of memory
    Created,
    /// Memory accessed and alignment recomputed
    Accessed { query_context: String },
    /// Goal hierarchy changed
    GoalChanged { old_goal: Uuid, new_goal: Uuid },
    /// Periodic recalibration
    Recalibration,
    /// Misalignment detected
    MisalignmentDetected { delta_a: f32 },
}

/// Snapshot of purpose at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeSnapshot {
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    /// Purpose vector at this time
    pub purpose: PurposeVector,
    /// Johari quadrants at this time (per embedder)
    pub johari: JohariFingerprint,
    /// Event that triggered the snapshot
    pub trigger: EvolutionTrigger,
}

/// TeleologicalFingerprint - Full node representation with purpose awareness
///
/// Wraps SemanticFingerprint with:
/// - PurposeVector (12D alignment signature)
/// - JohariFingerprint (per-embedder awareness classification)
/// - Purpose evolution tracking (time-series)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalFingerprint {
    /// Unique identifier
    pub id: Uuid,

    /// The raw 12-embedding array
    pub semantic: SemanticFingerprint,

    /// 12D alignment signature to current North Star
    pub purpose_vector: PurposeVector,

    /// Per-embedder Johari Window classification
    pub johari: JohariFingerprint,

    /// Temporal evolution of purpose (most recent snapshots)
    pub purpose_evolution: Vec<PurposeSnapshot>,

    /// Current aggregate alignment to North Star
    pub theta_to_north_star: f32,

    /// Content hash for deduplication
    pub content_hash: [u8; 32],

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub last_updated: DateTime<Utc>,

    /// Access count for decay calculations
    pub access_count: u64,
}

impl TeleologicalFingerprint {
    /// Expected storage size per fingerprint (~46KB)
    pub const EXPECTED_SIZE_BYTES: usize = 46_000;

    /// Maximum purpose evolution snapshots to keep in memory
    pub const MAX_EVOLUTION_SNAPSHOTS: usize = 100;

    /// Create new teleological fingerprint
    pub fn new(
        semantic: SemanticFingerprint,
        purpose_vector: PurposeVector,
        johari: JohariFingerprint,
        content_hash: [u8; 32],
    ) -> Self {
        let theta = purpose_vector.aggregate_alignment();
        let now = Utc::now();

        Self {
            id: Uuid::new_v4(),
            semantic,
            purpose_vector: purpose_vector.clone(),
            johari: johari.clone(),
            purpose_evolution: vec![PurposeSnapshot {
                timestamp: now,
                purpose: purpose_vector,
                johari,
                trigger: EvolutionTrigger::Created,
            }],
            theta_to_north_star: theta,
            content_hash,
            created_at: now,
            last_updated: now,
            access_count: 0,
        }
    }

    /// Record a purpose evolution snapshot
    pub fn record_snapshot(&mut self, trigger: EvolutionTrigger) {
        let snapshot = PurposeSnapshot {
            timestamp: Utc::now(),
            purpose: self.purpose_vector.clone(),
            johari: self.johari.clone(),
            trigger,
        };

        self.purpose_evolution.push(snapshot);

        // Keep only most recent snapshots in memory
        if self.purpose_evolution.len() > Self::MAX_EVOLUTION_SNAPSHOTS {
            self.purpose_evolution.remove(0);
        }

        self.last_updated = Utc::now();
    }

    /// Compute alignment delta from previous snapshot
    pub fn compute_alignment_delta(&self) -> f32 {
        if self.purpose_evolution.len() < 2 {
            return 0.0;
        }

        let current = self.theta_to_north_star;
        let previous = self.purpose_evolution[self.purpose_evolution.len() - 2]
            .purpose
            .aggregate_alignment();

        current - previous
    }

    /// Check for misalignment warning (delta_A < -0.15)
    pub fn check_misalignment_warning(&self) -> Option<f32> {
        let delta = self.compute_alignment_delta();
        if delta < -0.15 {
            Some(delta)
        } else {
            None
        }
    }

    /// Get threshold classification
    pub fn alignment_status(&self) -> AlignmentThreshold {
        AlignmentThreshold::classify(self.theta_to_north_star)
    }
}
```

---

### 1.3 JohariFingerprint (TS-103)

**Requirement**: FR-203
**Module**: `crates/context-graph-core/src/types/fingerprint/johari.rs`

```rust
use crate::types::johari::JohariQuadrant;
use serde::{Deserialize, Serialize};

/// Number of embedders in the system
pub const NUM_EMBEDDERS: usize = 12;

/// Per-embedder Johari Window classification
///
/// Classifies each embedding space into Johari quadrants:
/// - Open: Low entropy, High coherence (aware in this space)
/// - Blind: High entropy, Low coherence (discovery opportunity)
/// - Hidden: Low entropy, Low coherence (latent in this space)
/// - Unknown: High entropy, High coherence (frontier in this space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohariFingerprint {
    /// Quadrant classification per embedding space [Open, Hidden, Blind, Unknown]
    /// Stored as 4 f32 weights per embedder for soft classification
    pub quadrants: [[f32; 4]; NUM_EMBEDDERS],

    /// Confidence of classification per embedder [0.0, 1.0]
    pub confidence: [f32; NUM_EMBEDDERS],

    /// Transition probability matrix for evolution prediction
    /// transitions[i][from][to] = P(quadrant_to | quadrant_from) for embedder i
    pub transition_probs: [[[f32; 4]; 4]; NUM_EMBEDDERS],
}

impl JohariFingerprint {
    /// Entropy threshold for quadrant classification
    pub const ENTROPY_THRESHOLD: f32 = 0.5;

    /// Coherence threshold for quadrant classification
    pub const COHERENCE_THRESHOLD: f32 = 0.5;

    /// Create zeroed Johari fingerprint
    pub fn zeroed() -> Self {
        Self {
            quadrants: [[0.0; 4]; NUM_EMBEDDERS],
            confidence: [0.0; NUM_EMBEDDERS],
            transition_probs: [[[0.25; 4]; 4]; NUM_EMBEDDERS], // Uniform prior
        }
    }

    /// Classify based on entropy (delta_S) and coherence (delta_C)
    pub fn classify_quadrant(entropy: f32, coherence: f32) -> JohariQuadrant {
        match (
            entropy < Self::ENTROPY_THRESHOLD,
            coherence > Self::COHERENCE_THRESHOLD,
        ) {
            (true, true) => JohariQuadrant::Open,    // Low entropy, high coherence
            (true, false) => JohariQuadrant::Hidden, // Low entropy, low coherence
            (false, false) => JohariQuadrant::Blind, // High entropy, low coherence
            (false, true) => JohariQuadrant::Unknown, // High entropy, high coherence
        }
    }

    /// Get dominant quadrant for an embedder
    pub fn dominant_quadrant(&self, embedder_idx: usize) -> JohariQuadrant {
        if embedder_idx >= NUM_EMBEDDERS {
            return JohariQuadrant::Unknown;
        }

        let weights = &self.quadrants[embedder_idx];
        let max_idx = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(3);

        match max_idx {
            0 => JohariQuadrant::Open,
            1 => JohariQuadrant::Hidden,
            2 => JohariQuadrant::Blind,
            _ => JohariQuadrant::Unknown,
        }
    }

    /// Set quadrant weights for an embedder
    pub fn set_quadrant(
        &mut self,
        embedder_idx: usize,
        open: f32,
        hidden: f32,
        blind: f32,
        unknown: f32,
        confidence: f32,
    ) {
        if embedder_idx >= NUM_EMBEDDERS {
            return;
        }

        // Normalize weights
        let sum = open + hidden + blind + unknown;
        if sum > 0.0 {
            self.quadrants[embedder_idx] = [
                open / sum,
                hidden / sum,
                blind / sum,
                unknown / sum,
            ];
        }

        self.confidence[embedder_idx] = confidence.clamp(0.0, 1.0);
    }

    /// Find embedders in a specific quadrant
    pub fn find_by_quadrant(&self, quadrant: JohariQuadrant) -> Vec<usize> {
        (0..NUM_EMBEDDERS)
            .filter(|&i| self.dominant_quadrant(i) == quadrant)
            .collect()
    }

    /// Find blind spots: embedders where we have high semantic awareness but low causal awareness
    pub fn find_blind_spots(&self) -> Vec<(usize, f32)> {
        let semantic_idx = 0; // E1
        let causal_idx = 4;   // E5

        let semantic_open = self.quadrants[semantic_idx][0]; // Open weight
        let causal_blind = self.quadrants[causal_idx][2];    // Blind weight

        if semantic_open > 0.5 && causal_blind > 0.5 {
            vec![(causal_idx, causal_blind)]
        } else {
            vec![]
        }
    }

    /// Predict next quadrant for an embedder given current quadrant
    pub fn predict_transition(&self, embedder_idx: usize, current: JohariQuadrant) -> JohariQuadrant {
        if embedder_idx >= NUM_EMBEDDERS {
            return current;
        }

        let from_idx = current as usize;
        let probs = &self.transition_probs[embedder_idx][from_idx];

        // Return quadrant with highest transition probability
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(from_idx);

        match max_idx {
            0 => JohariQuadrant::Open,
            1 => JohariQuadrant::Hidden,
            2 => JohariQuadrant::Blind,
            _ => JohariQuadrant::Unknown,
        }
    }

    /// Encode quadrants as compact bytes (2 bits per quadrant = 3 bytes for 12 embedders)
    pub fn to_compact_bytes(&self) -> [u8; 3] {
        let mut bytes = [0u8; 3];
        for i in 0..NUM_EMBEDDERS {
            let quadrant_idx = self.dominant_quadrant(i) as u8;
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            bytes[byte_idx] |= quadrant_idx << bit_offset;
        }
        bytes
    }

    /// Decode quadrants from compact bytes
    pub fn from_compact_bytes(bytes: [u8; 3]) -> Self {
        let mut fingerprint = Self::zeroed();
        for i in 0..NUM_EMBEDDERS {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let quadrant_idx = (bytes[byte_idx] >> bit_offset) & 0b11;

            // Set dominant quadrant to 1.0, others to 0.0
            fingerprint.quadrants[i] = [0.0; 4];
            fingerprint.quadrants[i][quadrant_idx as usize] = 1.0;
            fingerprint.confidence[i] = 1.0;
        }
        fingerprint
    }
}
```

---

## 2. Storage Architecture (TS-200 Series)

### 2.1 RocksDB Schema (TS-201)

**Requirement**: FR-301, FR-304
**Module**: `crates/context-graph-storage/src/rocksdb/schema.rs`

```rust
use rocksdb::{ColumnFamilyDescriptor, Options, DB};
use std::path::Path;

/// RocksDB column families for teleological storage
pub const CF_FINGERPRINTS: &str = "fingerprints";         // Primary 46KB fingerprints
pub const CF_PURPOSE_VECTORS: &str = "purpose_vectors";   // 12D purpose vectors (48 bytes)
pub const CF_JOHARI_INDEX: &str = "johari_index";         // Bitmap index by quadrant
pub const CF_GOAL_ALIGNMENT: &str = "goal_alignment";     // Memory-to-goal cache
pub const CF_EVOLUTION: &str = "purpose_evolution";       // Time-series snapshots
pub const CF_METADATA: &str = "metadata";                 // Node metadata

/// Schema definition for teleological fingerprint storage
pub struct TeleologicalSchema;

impl TeleologicalSchema {
    /// Create column family options optimized for large fingerprints (~46KB)
    pub fn fingerprint_cf_options() -> Options {
        let mut opts = Options::default();

        // Optimize for large values
        opts.set_target_file_size_base(256 * 1024 * 1024); // 256MB SST files
        opts.set_write_buffer_size(128 * 1024 * 1024);     // 128MB write buffer
        opts.set_max_write_buffer_number(4);
        opts.set_min_write_buffer_number_to_merge(2);

        // Compression for space efficiency
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Block size for 46KB values
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_size(64 * 1024); // 64KB blocks
        opts.set_block_based_table_factory(&block_opts);

        opts
    }

    /// Create column family options for small purpose vectors (48 bytes)
    pub fn purpose_vector_cf_options() -> Options {
        let mut opts = Options::default();

        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB SST files
        opts.set_write_buffer_size(32 * 1024 * 1024);     // 32MB write buffer
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Enable bloom filter for fast lookups
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, true);
        opts.set_block_based_table_factory(&block_opts);

        opts
    }

    /// Open database with all column families
    pub fn open(path: impl AsRef<Path>) -> Result<DB, rocksdb::Error> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_FINGERPRINTS, Self::fingerprint_cf_options()),
            ColumnFamilyDescriptor::new(CF_PURPOSE_VECTORS, Self::purpose_vector_cf_options()),
            ColumnFamilyDescriptor::new(CF_JOHARI_INDEX, Options::default()),
            ColumnFamilyDescriptor::new(CF_GOAL_ALIGNMENT, Options::default()),
            ColumnFamilyDescriptor::new(CF_EVOLUTION, Options::default()),
            ColumnFamilyDescriptor::new(CF_METADATA, Options::default()),
        ];

        DB::open_cf_descriptors(&db_opts, path, cf_descriptors)
    }
}

/// Key format for fingerprint storage: UUID as 16 bytes
pub fn fingerprint_key(id: &uuid::Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Key format for purpose vector: UUID as 16 bytes
pub fn purpose_vector_key(id: &uuid::Uuid) -> [u8; 16] {
    *id.as_bytes()
}

/// Key format for Johari index: (quadrant_u8, embedder_u8, memory_id_bytes)
pub fn johari_index_key(quadrant: u8, embedder: u8, memory_id: &uuid::Uuid) -> Vec<u8> {
    let mut key = Vec::with_capacity(18);
    key.push(quadrant);
    key.push(embedder);
    key.extend_from_slice(memory_id.as_bytes());
    key
}

/// Key format for goal alignment: (memory_id_bytes, goal_id_bytes)
pub fn goal_alignment_key(memory_id: &uuid::Uuid, goal_id: &uuid::Uuid) -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(memory_id.as_bytes());
    key.extend_from_slice(goal_id.as_bytes());
    key
}

/// Key format for evolution: (memory_id_bytes, timestamp_i64_be)
pub fn evolution_key(memory_id: &uuid::Uuid, timestamp_nanos: i64) -> Vec<u8> {
    let mut key = Vec::with_capacity(24);
    key.extend_from_slice(memory_id.as_bytes());
    key.extend_from_slice(&timestamp_nanos.to_be_bytes());
    key
}
```

---

### 2.2 HNSW Index Configuration (TS-202)

**Requirement**: FR-302
**Module**: `crates/context-graph-storage/src/indexes/hnsw_config.rs`

```rust
use std::collections::HashMap;

/// HNSW index configuration per embedding space
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Number of connections per node (M parameter)
    pub m: usize,
    /// Size of dynamic candidate list during construction (ef_construction)
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (ef_search)
    pub ef_search: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Embedding dimension
    pub dimension: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
    /// For asymmetric query/document embeddings
    AsymmetricCosine,
}

/// Index types for different embedding spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedderIndex {
    E1TextGeneral,      // 1024D HNSW
    E2TextSmall,        // 512D HNSW
    E3Multilingual,     // 512D HNSW
    E4Code,             // 512D HNSW
    E5QueryDoc,         // 768D x 2 (separate query/doc indexes)
    E6Sparse,           // Inverted index (NOT HNSW)
    E7OpenaiAda,        // 1536D HNSW
    E8Minilm,           // 384D HNSW
    E9Simhash,          // 1024D HNSW (with LSH option)
    E10Instructor,      // 768D HNSW
    E11Fast,            // 384D HNSW
    E12TokenLevel,      // ColBERT-style (NOT HNSW)
    PurposeVector,      // 12D HNSW
}

/// Get HNSW configuration for each index type
pub fn get_hnsw_config(index: EmbedderIndex) -> Option<HnswConfig> {
    match index {
        EmbedderIndex::E1TextGeneral => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 1024,
        }),
        EmbedderIndex::E2TextSmall => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 512,
        }),
        EmbedderIndex::E3Multilingual => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 512,
        }),
        EmbedderIndex::E4Code => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 512,
        }),
        EmbedderIndex::E5QueryDoc => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::AsymmetricCosine,
            dimension: 768,
        }),
        EmbedderIndex::E6Sparse => None, // Uses inverted index
        EmbedderIndex::E7OpenaiAda => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 1536,
        }),
        EmbedderIndex::E8Minilm => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 384,
        }),
        EmbedderIndex::E9Simhash => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 1024,
        }),
        EmbedderIndex::E10Instructor => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 768,
        }),
        EmbedderIndex::E11Fast => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 384,
        }),
        EmbedderIndex::E12TokenLevel => None, // Uses ColBERT MaxSim
        EmbedderIndex::PurposeVector => Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
            dimension: 12,
        }),
    }
}

/// Get all HNSW indexes that need to be created
pub fn all_hnsw_configs() -> HashMap<EmbedderIndex, HnswConfig> {
    let indices = [
        EmbedderIndex::E1TextGeneral,
        EmbedderIndex::E2TextSmall,
        EmbedderIndex::E3Multilingual,
        EmbedderIndex::E4Code,
        EmbedderIndex::E5QueryDoc,
        EmbedderIndex::E7OpenaiAda,
        EmbedderIndex::E8Minilm,
        EmbedderIndex::E9Simhash,
        EmbedderIndex::E10Instructor,
        EmbedderIndex::E11Fast,
        EmbedderIndex::PurposeVector,
    ];

    indices
        .into_iter()
        .filter_map(|idx| get_hnsw_config(idx).map(|cfg| (idx, cfg)))
        .collect()
}
```

---

### 2.3 Serialization (TS-203)

**Requirement**: FR-301, FR-103
**Module**: `crates/context-graph-storage/src/serialization.rs`

```rust
use crate::types::fingerprint::{
    SemanticFingerprint, TeleologicalFingerprint, PurposeVector, JohariFingerprint, SparseVector30K
};
use bincode::{config, Decode, Encode};
use std::io::{Read, Write};

/// Bincode configuration for fingerprint serialization
pub fn bincode_config() -> impl bincode::config::Config {
    config::standard()
        .with_little_endian()
        .with_variable_int_encoding()
}

/// Serialization format version for future compatibility
pub const SERIALIZATION_VERSION: u8 = 1;

/// Header for serialized fingerprint
#[derive(Debug, Clone, Encode, Decode)]
pub struct FingerprintHeader {
    /// Format version
    pub version: u8,
    /// Total size in bytes (excluding header)
    pub total_size: u32,
    /// Number of tokens in E12 (for pre-allocation)
    pub e12_token_count: u16,
    /// Sparse vector active count
    pub e6_active_count: u16,
}

/// Serialize SemanticFingerprint to bytes
pub fn serialize_semantic_fingerprint(fp: &SemanticFingerprint) -> Vec<u8> {
    let header = FingerprintHeader {
        version: SERIALIZATION_VERSION,
        total_size: fp.storage_size() as u32,
        e12_token_count: fp.e12_token_level.len() as u16,
        e6_active_count: fp.e6_sparse.indices.len() as u16,
    };

    let config = bincode_config();
    let mut buf = Vec::with_capacity(header.total_size as usize + 8);

    // Write header
    bincode::encode_into_std_write(&header, &mut buf, config).unwrap();

    // Write dense embeddings (fixed size, direct write)
    write_f32_array(&mut buf, &fp.e1_text_general);
    write_f32_array(&mut buf, &fp.e2_text_small);
    write_f32_array(&mut buf, &fp.e3_multilingual);
    write_f32_array(&mut buf, &fp.e4_code);
    write_f32_array(&mut buf, &fp.e5_query_doc.0);
    write_f32_array(&mut buf, &fp.e5_query_doc.1);
    write_f32_array(&mut buf, &fp.e7_openai_ada);
    write_f32_array(&mut buf, &fp.e8_minilm);
    write_f32_array(&mut buf, &fp.e9_simhash);
    write_f32_array(&mut buf, &fp.e10_instructor);
    write_f32_array(&mut buf, &fp.e11_fast);

    // Write sparse embedding
    bincode::encode_into_std_write(&fp.e6_sparse.indices, &mut buf, config).unwrap();
    bincode::encode_into_std_write(&fp.e6_sparse.values, &mut buf, config).unwrap();

    // Write late interaction (variable length)
    for token_emb in &fp.e12_token_level {
        write_f32_array(&mut buf, token_emb);
    }

    buf
}

/// Deserialize SemanticFingerprint from bytes
pub fn deserialize_semantic_fingerprint(data: &[u8]) -> Result<SemanticFingerprint, &'static str> {
    let config = bincode_config();
    let mut cursor = std::io::Cursor::new(data);

    // Read header
    let header: FingerprintHeader = bincode::decode_from_std_read(&mut cursor, config)
        .map_err(|_| "failed to decode header")?;

    if header.version != SERIALIZATION_VERSION {
        return Err("unsupported serialization version");
    }

    // Read dense embeddings
    let e1_text_general = read_f32_array::<1024>(&mut cursor)?;
    let e2_text_small = read_f32_array::<512>(&mut cursor)?;
    let e3_multilingual = read_f32_array::<512>(&mut cursor)?;
    let e4_code = read_f32_array::<512>(&mut cursor)?;
    let e5_query = read_f32_array::<768>(&mut cursor)?;
    let e5_doc = read_f32_array::<768>(&mut cursor)?;
    let e7_openai_ada = read_f32_array::<1536>(&mut cursor)?;
    let e8_minilm = read_f32_array::<384>(&mut cursor)?;
    let e9_simhash = read_f32_array::<1024>(&mut cursor)?;
    let e10_instructor = read_f32_array::<768>(&mut cursor)?;
    let e11_fast = read_f32_array::<384>(&mut cursor)?;

    // Read sparse embedding
    let indices: Vec<u16> = bincode::decode_from_std_read(&mut cursor, config)
        .map_err(|_| "failed to decode sparse indices")?;
    let values: Vec<f32> = bincode::decode_from_std_read(&mut cursor, config)
        .map_err(|_| "failed to decode sparse values")?;

    // Read late interaction
    let mut e12_token_level = Vec::with_capacity(header.e12_token_count as usize);
    for _ in 0..header.e12_token_count {
        e12_token_level.push(read_f32_array::<128>(&mut cursor)?);
    }

    Ok(SemanticFingerprint {
        e1_text_general,
        e2_text_small,
        e3_multilingual,
        e4_code,
        e5_query_doc: (e5_query, e5_doc),
        e6_sparse: SparseVector30K { indices, values },
        e7_openai_ada,
        e8_minilm,
        e9_simhash,
        e10_instructor,
        e11_fast,
        e12_token_level,
    })
}

/// Write f32 array directly to buffer (no length prefix for fixed arrays)
fn write_f32_array<W: Write>(w: &mut W, arr: &[f32]) {
    for &f in arr {
        w.write_all(&f.to_le_bytes()).unwrap();
    }
}

/// Read f32 array from cursor
fn read_f32_array<const N: usize>(r: &mut impl Read) -> Result<[f32; N], &'static str> {
    let mut arr = [0.0f32; N];
    let mut buf = [0u8; 4];
    for i in 0..N {
        r.read_exact(&mut buf).map_err(|_| "unexpected end of data")?;
        arr[i] = f32::from_le_bytes(buf);
    }
    Ok(arr)
}
```

---

## 3. Trait Definitions (TS-300 Series)

### 3.1 EmbeddingProvider Changes (TS-301)

**Requirement**: FR-101, FR-102, FR-104
**Module**: `crates/context-graph-core/src/traits/embedding_provider.rs`

```rust
use async_trait::async_trait;
use std::time::Duration;
use crate::error::CoreResult;
use crate::types::fingerprint::SemanticFingerprint;

/// Output from multi-array embedding generation
#[derive(Debug, Clone)]
pub struct MultiArrayEmbeddingOutput {
    /// The complete 12-embedding fingerprint
    pub fingerprint: SemanticFingerprint,

    /// Total latency for all 12 embeddings
    pub total_latency: Duration,

    /// Per-embedder latencies (for optimization)
    pub per_embedder_latency: [Duration; 12],

    /// Per-embedder model IDs
    pub model_ids: [String; 12],
}

impl MultiArrayEmbeddingOutput {
    /// Expected total latency target: <30ms for all 12 embedders
    pub const TARGET_LATENCY_MS: u64 = 30;

    /// Check if latency is within target
    pub fn is_within_latency_target(&self) -> bool {
        self.total_latency.as_millis() < Self::TARGET_LATENCY_MS as u128
    }
}

/// Multi-Array Embedding Provider trait
///
/// REPLACES the legacy EmbeddingProvider that returned Vec<f32>.
/// Returns complete SemanticFingerprint with all 12 embeddings.
///
/// NO FUSION - each embedder output stored independently.
#[async_trait]
pub trait MultiArrayEmbeddingProvider: Send + Sync {
    /// Generate complete 12-embedding fingerprint for content
    ///
    /// # Performance Target
    /// - Single content: <30ms for all 12 embeddings
    /// - Uses parallel embedding generation internally
    ///
    /// # Returns
    /// - MultiArrayEmbeddingOutput containing SemanticFingerprint
    ///
    /// # Errors
    /// - Returns CoreError::Embedding if any embedder fails
    async fn embed_all(&self, content: &str) -> CoreResult<MultiArrayEmbeddingOutput>;

    /// Generate fingerprints for multiple contents in batch
    ///
    /// # Performance Target
    /// - 64 contents: <100ms for all 12 embeddings per content
    ///
    /// # Returns
    /// - Vec of MultiArrayEmbeddingOutput, one per input content
    async fn embed_batch_all(&self, contents: &[String]) -> CoreResult<Vec<MultiArrayEmbeddingOutput>>;

    /// Get expected dimensions for each embedder
    fn dimensions(&self) -> [usize; 12] {
        [
            1024,  // E1
            512,   // E2
            512,   // E3
            512,   // E4
            768,   // E5 (query)
            0,     // E6 (sparse - variable)
            1536,  // E7
            384,   // E8
            1024,  // E9
            768,   // E10
            384,   // E11
            128,   // E12 (per token)
        ]
    }

    /// Get model IDs for each embedder
    fn model_ids(&self) -> [&str; 12];

    /// Check if all embedders are ready
    fn is_ready(&self) -> bool;

    /// Get health status per embedder
    fn health_status(&self) -> [bool; 12];
}

/// Individual embedder trait for composing MultiArrayEmbeddingProvider
#[async_trait]
pub trait SingleEmbedder: Send + Sync {
    /// Embedding dimension for this embedder
    fn dimension(&self) -> usize;

    /// Model identifier
    fn model_id(&self) -> &str;

    /// Generate single embedding
    async fn embed(&self, content: &str) -> CoreResult<Vec<f32>>;

    /// Check if ready
    fn is_ready(&self) -> bool;
}
```

---

### 3.2 MemoryStore Changes (TS-302)

**Requirement**: FR-301, FR-302, FR-303, FR-401
**Module**: `crates/context-graph-core/src/traits/memory_store.rs`

```rust
use async_trait::async_trait;
use uuid::Uuid;
use crate::error::CoreResult;
use crate::types::fingerprint::{TeleologicalFingerprint, PurposeVector, JohariFingerprint};
use crate::types::johari::JohariQuadrant;

/// Query weights for multi-embedding similarity
#[derive(Debug, Clone)]
pub struct SimilarityWeights {
    /// Weights for each embedder [0.0, 1.0], must sum to 1.0
    pub weights: [f32; 12],
    /// Query type for automatic weight selection
    pub query_type: QueryType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    SemanticSearch,      // Heavy E1, E7
    CausalReasoning,     // Heavy E5
    CodeSearch,          // Heavy E4, E7
    TemporalNavigation,  // Heavy E2-E4 temporal embedders
    FactChecking,        // Heavy E11 entity
    Balanced,            // Equal weights
}

impl SimilarityWeights {
    /// Create weights for a specific query type
    pub fn for_query_type(qt: QueryType) -> Self {
        let weights = match qt {
            QueryType::SemanticSearch => [
                0.30, 0.05, 0.05, 0.05, 0.10, 0.05, 0.20, 0.05, 0.05, 0.05, 0.03, 0.02
            ],
            QueryType::CausalReasoning => [
                0.15, 0.03, 0.03, 0.03, 0.45, 0.03, 0.10, 0.03, 0.03, 0.05, 0.05, 0.02
            ],
            QueryType::CodeSearch => [
                0.15, 0.02, 0.02, 0.35, 0.05, 0.03, 0.25, 0.02, 0.02, 0.03, 0.03, 0.03
            ],
            QueryType::TemporalNavigation => [
                0.15, 0.20, 0.20, 0.20, 0.05, 0.02, 0.05, 0.02, 0.03, 0.03, 0.03, 0.02
            ],
            QueryType::FactChecking => [
                0.10, 0.02, 0.02, 0.02, 0.20, 0.05, 0.05, 0.02, 0.02, 0.05, 0.43, 0.02
            ],
            QueryType::Balanced => [
                0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.087
            ],
        };
        Self { weights, query_type: qt }
    }

    /// Validate weights sum to ~1.0
    pub fn validate(&self) -> bool {
        let sum: f32 = self.weights.iter().sum();
        (sum - 1.0).abs() < 0.01
    }
}

/// Multi-embedding search options
#[derive(Debug, Clone)]
pub struct MultiEmbeddingSearchOptions {
    /// Maximum results to return
    pub top_k: usize,
    /// Minimum aggregate similarity threshold
    pub min_similarity: f32,
    /// Similarity weights per embedder
    pub weights: SimilarityWeights,
    /// Filter by Johari quadrant (for specific embedder)
    pub johari_filter: Option<(usize, JohariQuadrant)>,
    /// Minimum alignment to North Star
    pub min_alignment: Option<f32>,
    /// Specific embedder index for single-space search (0-11)
    pub single_space: Option<usize>,
}

impl Default for MultiEmbeddingSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.5,
            weights: SimilarityWeights::for_query_type(QueryType::Balanced),
            johari_filter: None,
            min_alignment: None,
            single_space: None,
        }
    }
}

/// Search result with multi-embedding similarity breakdown
#[derive(Debug, Clone)]
pub struct MultiEmbeddingSearchResult {
    /// The matched fingerprint
    pub fingerprint: TeleologicalFingerprint,
    /// Aggregate similarity score
    pub similarity: f32,
    /// Per-embedder similarity scores
    pub per_embedder_similarity: [f32; 12],
    /// Top contributing embedders (indices)
    pub top_contributors: Vec<(usize, f32)>,
    /// Alignment to North Star
    pub alignment_score: f32,
}

/// Teleological Memory Store trait
///
/// REPLACES legacy MemoryStore that accepted Vec<f32> embeddings.
/// Operates on TeleologicalFingerprint with multi-embedding search.
#[async_trait]
pub trait TeleologicalMemoryStore: Send + Sync {
    /// Store a teleological fingerprint
    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid>;

    /// Retrieve fingerprint by ID
    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>>;

    /// Multi-embedding semantic search
    ///
    /// Uses weighted similarity across all 12 embedding spaces.
    async fn search_multi(
        &self,
        query: &TeleologicalFingerprint,
        options: MultiEmbeddingSearchOptions,
    ) -> CoreResult<Vec<MultiEmbeddingSearchResult>>;

    /// Single-space search (uses only one embedder's index)
    async fn search_single_space(
        &self,
        query_embedding: &[f32],
        embedder_index: usize,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    /// Search by purpose vector similarity
    async fn search_by_purpose(
        &self,
        purpose: &PurposeVector,
        top_k: usize,
        min_similarity: f32,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    /// Find memories aligned with a goal
    async fn find_aligned_to_goal(
        &self,
        goal_id: Uuid,
        min_alignment: f32,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    /// Find memories by Johari quadrant for specific embedder
    async fn find_by_johari(
        &self,
        embedder_index: usize,
        quadrant: JohariQuadrant,
        top_k: usize,
    ) -> CoreResult<Vec<Uuid>>;

    /// Update fingerprint alignment and Johari classification
    async fn update_alignment(
        &self,
        id: Uuid,
        purpose_vector: PurposeVector,
        johari: JohariFingerprint,
    ) -> CoreResult<bool>;

    /// Record purpose evolution snapshot
    async fn record_evolution(
        &self,
        id: Uuid,
        trigger: super::fingerprint::EvolutionTrigger,
    ) -> CoreResult<()>;

    /// Delete fingerprint
    async fn delete(&self, id: Uuid) -> CoreResult<bool>;

    /// Get total count
    async fn count(&self) -> CoreResult<usize>;

    /// Compact storage and rebuild indexes
    async fn compact(&self) -> CoreResult<()>;
}
```

---

## 4. Query System (TS-400 Series)

### 4.1 Weighted Similarity (TS-401)

**Requirement**: FR-401, FR-402
**Module**: `crates/context-graph-core/src/similarity/weighted.rs`

```rust
use crate::types::fingerprint::{SemanticFingerprint, TeleologicalFingerprint, SparseVector30K};

/// Compute weighted multi-embedding similarity
///
/// Formula: S(A,B) = sum_i(w_i * sim_i(A_i, B_i))
/// where sim_i is the appropriate similarity function for embedder i
pub fn multi_embedding_similarity(
    query: &SemanticFingerprint,
    document: &SemanticFingerprint,
    weights: &[f32; 12],
) -> MultiSimilarityResult {
    let mut per_embedder = [0.0f32; 12];

    // E1: Cosine similarity (1024D)
    per_embedder[0] = cosine_similarity(&query.e1_text_general, &document.e1_text_general);

    // E2: Cosine similarity (512D)
    per_embedder[1] = cosine_similarity(&query.e2_text_small, &document.e2_text_small);

    // E3: Cosine similarity (512D)
    per_embedder[2] = cosine_similarity(&query.e3_multilingual, &document.e3_multilingual);

    // E4: Cosine similarity (512D)
    per_embedder[3] = cosine_similarity(&query.e4_code, &document.e4_code);

    // E5: Asymmetric similarity - query embedding vs doc embedding
    per_embedder[4] = asymmetric_similarity(
        &query.e5_query_doc.0,    // Query uses query embedding
        &document.e5_query_doc.1, // Document uses doc embedding
    );

    // E6: Sparse similarity (Jaccard + weighted overlap)
    per_embedder[5] = query.e6_sparse.similarity(&document.e6_sparse);

    // E7: Cosine similarity (1536D)
    per_embedder[6] = cosine_similarity(&query.e7_openai_ada, &document.e7_openai_ada);

    // E8: Cosine similarity (384D)
    per_embedder[7] = cosine_similarity(&query.e8_minilm, &document.e8_minilm);

    // E9: Cosine similarity (1024D) - could use Hamming on binary representation
    per_embedder[8] = cosine_similarity(&query.e9_simhash, &document.e9_simhash);

    // E10: Cosine similarity (768D)
    per_embedder[9] = cosine_similarity(&query.e10_instructor, &document.e10_instructor);

    // E11: Cosine similarity (384D)
    per_embedder[10] = cosine_similarity(&query.e11_fast, &document.e11_fast);

    // E12: MaxSim late interaction
    per_embedder[11] = maxsim_similarity(&query.e12_token_level, &document.e12_token_level);

    // Compute weighted aggregate
    let aggregate: f32 = per_embedder
        .iter()
        .zip(weights.iter())
        .map(|(sim, w)| sim * w)
        .sum();

    // Find top contributors
    let mut indexed: Vec<(usize, f32)> = per_embedder
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s * weights[i]))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_contributors: Vec<(usize, f32)> = indexed.into_iter().take(3).collect();

    MultiSimilarityResult {
        aggregate,
        per_embedder,
        top_contributors,
    }
}

/// Result of multi-embedding similarity computation
#[derive(Debug, Clone)]
pub struct MultiSimilarityResult {
    /// Weighted aggregate similarity [0.0, 1.0]
    pub aggregate: f32,
    /// Per-embedder similarity scores
    pub per_embedder: [f32; 12],
    /// Top contributing (embedder_index, weighted_contribution)
    pub top_contributors: Vec<(usize, f32)>,
}

/// Cosine similarity for dense vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Asymmetric similarity for query/document embeddings
/// Uses query embedding from query, doc embedding from document
pub fn asymmetric_similarity(query_emb: &[f32], doc_emb: &[f32]) -> f32 {
    cosine_similarity(query_emb, doc_emb)
}

/// ColBERT-style MaxSim for late interaction embeddings
///
/// For each query token, find max similarity to any document token
/// Average over all query tokens
pub fn maxsim_similarity(query_tokens: &[[f32; 128]], doc_tokens: &[[f32; 128]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }

    let mut total: f32 = 0.0;

    for q_emb in query_tokens {
        let max_sim = doc_tokens
            .iter()
            .map(|d_emb| cosine_similarity(q_emb, d_emb))
            .fold(f32::NEG_INFINITY, f32::max);
        total += max_sim.max(0.0);
    }

    total / query_tokens.len() as f32
}

/// Teleological similarity: not just similar, but similar FOR THE SAME PURPOSE
pub fn teleological_similarity(
    a: &TeleologicalFingerprint,
    b: &TeleologicalFingerprint,
    weights: &[f32; 12],
) -> f32 {
    // Raw embedding similarity
    let embedding_sim = multi_embedding_similarity(&a.semantic, &b.semantic, weights);

    // Purpose alignment (both aligned to same purpose?)
    let purpose_alignment = a.purpose_vector.similarity(&b.purpose_vector);

    // Teleological similarity = embedding similarity * purpose alignment
    embedding_sim.aggregate * purpose_alignment
}
```

---

### 4.2 Distance Metrics (TS-402)

**Requirement**: FR-403
**Module**: `crates/context-graph-core/src/similarity/metrics.rs`

```rust
/// Distance metrics for different embedding types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cosine_similarity
    Cosine,
    /// Dot product (for normalized vectors, equivalent to cosine)
    DotProduct,
    /// Euclidean (L2) distance
    Euclidean,
    /// Hamming distance for binary vectors
    Hamming,
    /// Jaccard distance for sparse sets
    Jaccard,
    /// MaxSim for late interaction
    MaxSim,
}

/// Per-embedder recommended distance metrics
pub fn recommended_metric(embedder_index: usize) -> DistanceMetric {
    match embedder_index {
        0..=4 => DistanceMetric::Cosine,   // E1-E5: Dense semantic
        5 => DistanceMetric::Jaccard,       // E6: Sparse
        6..=10 => DistanceMetric::Cosine,  // E7-E11: Dense
        11 => DistanceMetric::MaxSim,       // E12: Late interaction
        _ => DistanceMetric::Cosine,
    }
}

/// Compute distance using specified metric
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => {
            1.0 - super::weighted::cosine_similarity(a, b)
        }
        DistanceMetric::DotProduct => {
            -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
        }
        DistanceMetric::Euclidean => {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        }
        DistanceMetric::Hamming => {
            // For binary vectors encoded as f32 (0.0 or 1.0)
            a.iter()
                .zip(b.iter())
                .filter(|(x, y)| (*x > 0.5) != (*y > 0.5))
                .count() as f32
                / a.len() as f32
        }
        _ => 1.0 - super::weighted::cosine_similarity(a, b),
    }
}

/// Convert distance to similarity [0.0, 1.0]
pub fn distance_to_similarity(distance: f32, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine | DistanceMetric::Hamming | DistanceMetric::Jaccard => {
            1.0 - distance.clamp(0.0, 1.0)
        }
        DistanceMetric::Euclidean => {
            1.0 / (1.0 + distance)
        }
        DistanceMetric::DotProduct => {
            // Dot product can be negative, normalize to [0, 1]
            (1.0 + distance.tanh()) / 2.0
        }
        DistanceMetric::MaxSim => {
            distance.clamp(0.0, 1.0) // MaxSim is already similarity
        }
    }
}
```

---

## 5. Meta-UTL System (TS-500 Series)

### 5.1 Architecture (TS-501)

**Requirement**: FR-501, FR-502
**Module**: `crates/context-graph-core/src/meta_utl/mod.rs`

```rust
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Meta-UTL prediction for storage impact
#[derive(Debug, Clone)]
pub struct StoragePrediction {
    /// Predicted impact on system coherence
    pub coherence_delta: f32,
    /// Predicted alignment change
    pub alignment_delta: f32,
    /// Confidence in prediction [0.0, 1.0]
    pub confidence: f32,
    /// Prediction timestamp
    pub predicted_at: Instant,
}

/// Meta-UTL prediction for retrieval quality
#[derive(Debug, Clone)]
pub struct RetrievalPrediction {
    /// Predicted relevance of top-k results
    pub expected_relevance: f32,
    /// Predicted alignment of results to query goal
    pub expected_alignment: f32,
    /// Confidence in prediction [0.0, 1.0]
    pub confidence: f32,
}

/// Learning trajectory per embedding space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceLearningTrajectory {
    /// Embedder index (0-11)
    pub embedder_index: usize,
    /// Historical prediction accuracy
    pub accuracy_history: VecDeque<f32>,
    /// Current weight (adjusted based on accuracy)
    pub current_weight: f32,
    /// Space-specific alignment threshold
    pub alignment_threshold: f32,
    /// Total predictions made
    pub prediction_count: u64,
    /// Correct predictions
    pub correct_predictions: u64,
}

impl SpaceLearningTrajectory {
    /// Maximum history entries
    const MAX_HISTORY: usize = 100;

    /// Create new trajectory for embedder
    pub fn new(embedder_index: usize, initial_weight: f32) -> Self {
        Self {
            embedder_index,
            accuracy_history: VecDeque::with_capacity(Self::MAX_HISTORY),
            current_weight: initial_weight,
            alignment_threshold: 0.70, // Default threshold
            prediction_count: 0,
            correct_predictions: 0,
        }
    }

    /// Record prediction outcome
    pub fn record_outcome(&mut self, was_correct: bool) {
        self.prediction_count += 1;
        if was_correct {
            self.correct_predictions += 1;
        }

        let accuracy = if was_correct { 1.0 } else { 0.0 };
        self.accuracy_history.push_back(accuracy);

        if self.accuracy_history.len() > Self::MAX_HISTORY {
            self.accuracy_history.pop_front();
        }
    }

    /// Get recent accuracy (rolling window)
    pub fn recent_accuracy(&self) -> f32 {
        if self.accuracy_history.is_empty() {
            return 0.5; // Prior
        }
        self.accuracy_history.iter().sum::<f32>() / self.accuracy_history.len() as f32
    }

    /// Adjust weight based on accuracy
    pub fn adjust_weight(&mut self) {
        let accuracy = self.recent_accuracy();

        // Increase weight for high accuracy, decrease for low
        if accuracy > 0.85 {
            self.current_weight = (self.current_weight * 1.05).min(0.25);
        } else if accuracy < 0.70 {
            self.current_weight = (self.current_weight * 0.95).max(0.02);
        }
    }
}

/// Meta-UTL system that learns about its own learning
pub struct MetaUTL {
    /// Per-space learning trajectories
    pub trajectories: [SpaceLearningTrajectory; 12],

    /// Overall system accuracy
    pub system_accuracy: f32,

    /// Recent predictions for validation
    pub recent_predictions: VecDeque<(StoragePrediction, Option<f32>)>,

    /// Meta-learning rate
    pub meta_lr: f32,

    /// Escalation threshold (accuracy below this triggers alert)
    pub escalation_threshold: f32,
}

impl MetaUTL {
    /// Target accuracy for storage predictions
    pub const STORAGE_ACCURACY_TARGET: f32 = 0.85;

    /// Target accuracy for retrieval predictions
    pub const RETRIEVAL_ACCURACY_TARGET: f32 = 0.80;

    /// Operations before escalation check
    pub const ESCALATION_WINDOW: usize = 100;

    /// Create new Meta-UTL system
    pub fn new() -> Self {
        let initial_weight = 1.0 / 12.0;
        Self {
            trajectories: std::array::from_fn(|i| SpaceLearningTrajectory::new(i, initial_weight)),
            system_accuracy: 0.5,
            recent_predictions: VecDeque::with_capacity(100),
            meta_lr: 0.1,
            escalation_threshold: 0.70,
        }
    }

    /// Predict impact of storing a fingerprint
    pub fn predict_storage_impact(&self, _fingerprint: &super::TeleologicalFingerprint) -> StoragePrediction {
        // Compute weighted prediction based on per-space accuracies
        let weighted_confidence: f32 = self.trajectories
            .iter()
            .map(|t| t.recent_accuracy() * t.current_weight)
            .sum();

        StoragePrediction {
            coherence_delta: 0.0,  // Placeholder - requires actual computation
            alignment_delta: 0.0,
            confidence: weighted_confidence,
            predicted_at: Instant::now(),
        }
    }

    /// Validate prediction against actual outcome
    pub fn validate_prediction(&mut self, predicted: &StoragePrediction, actual_delta: f32) {
        let prediction_error = (predicted.coherence_delta - actual_delta).abs();
        let was_accurate = prediction_error < 0.2;

        // Update per-space trajectories
        for trajectory in &mut self.trajectories {
            trajectory.record_outcome(was_accurate);
            trajectory.adjust_weight();
        }

        // Update system accuracy
        self.update_system_accuracy();

        // Check for escalation
        if self.system_accuracy < self.escalation_threshold {
            self.escalate_low_accuracy();
        }
    }

    /// Update overall system accuracy
    fn update_system_accuracy(&mut self) {
        self.system_accuracy = self.trajectories
            .iter()
            .map(|t| t.recent_accuracy())
            .sum::<f32>()
            / 12.0;
    }

    /// Handle low accuracy escalation
    fn escalate_low_accuracy(&self) {
        tracing::warn!(
            "Meta-UTL accuracy below threshold: {} < {}",
            self.system_accuracy,
            self.escalation_threshold
        );
        // Could trigger alerts, parameter reset, etc.
    }

    /// Get current optimized weights
    pub fn optimized_weights(&self) -> [f32; 12] {
        let mut weights: [f32; 12] = std::array::from_fn(|i| self.trajectories[i].current_weight);

        // Normalize to sum to 1.0
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }
}

impl Default for MetaUTL {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 5.2 Metrics (TS-502)

**Requirement**: FR-503
**Module**: `crates/context-graph-core/src/meta_utl/metrics.rs`

```rust
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// System health metrics from Meta-UTL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthMetrics {
    /// Overall learning score (UTL avg target > 0.6)
    pub learning_score: f32,

    /// Coherence recovery time (target < 10s)
    pub coherence_recovery_time: Duration,

    /// Attack detection rate (target > 95%)
    pub attack_detection_rate: f32,

    /// False positive rate (target < 2%)
    pub false_positive_rate: f32,

    /// Per-space prediction accuracy
    pub per_space_accuracy: [f32; 12],

    /// Timestamp of metrics collection
    pub collected_at: chrono::DateTime<chrono::Utc>,
}

impl SystemHealthMetrics {
    /// Learning score target
    pub const LEARNING_SCORE_TARGET: f32 = 0.6;

    /// Coherence recovery target
    pub const COHERENCE_RECOVERY_TARGET: Duration = Duration::from_secs(10);

    /// Attack detection target
    pub const ATTACK_DETECTION_TARGET: f32 = 0.95;

    /// False positive target
    pub const FALSE_POSITIVE_TARGET: f32 = 0.02;

    /// Check if all targets are met
    pub fn meets_targets(&self) -> bool {
        self.learning_score >= Self::LEARNING_SCORE_TARGET
            && self.coherence_recovery_time <= Self::COHERENCE_RECOVERY_TARGET
            && self.attack_detection_rate >= Self::ATTACK_DETECTION_TARGET
            && self.false_positive_rate <= Self::FALSE_POSITIVE_TARGET
    }

    /// Get list of failed targets
    pub fn failed_targets(&self) -> Vec<&'static str> {
        let mut failed = Vec::new();

        if self.learning_score < Self::LEARNING_SCORE_TARGET {
            failed.push("learning_score");
        }
        if self.coherence_recovery_time > Self::COHERENCE_RECOVERY_TARGET {
            failed.push("coherence_recovery_time");
        }
        if self.attack_detection_rate < Self::ATTACK_DETECTION_TARGET {
            failed.push("attack_detection_rate");
        }
        if self.false_positive_rate > Self::FALSE_POSITIVE_TARGET {
            failed.push("false_positive_rate");
        }

        failed
    }
}

/// Compute learning score using UTL formula
/// L = sigmoid(2.0 * weighted_deltas)
pub fn compute_learning_score(
    semantic_deltas: &[f32; 12],
    causal_deltas: &[f32; 12],
    weights: &[f32; 12],
    w_e: f32, // Surprise factor
) -> f32 {
    // Weighted sum of semantic deltas
    let weighted_semantic: f32 = semantic_deltas
        .iter()
        .zip(weights.iter())
        .map(|(d, w)| d * w)
        .sum();

    // Weighted sum of causal deltas
    let weighted_causal: f32 = causal_deltas
        .iter()
        .zip(weights.iter())
        .map(|(d, w)| d * w)
        .sum();

    // UTL formula
    let raw = weighted_semantic * weighted_causal * w_e;

    // Sigmoid normalization
    sigmoid(2.0 * raw)
}

/// Sigmoid function for UTL score normalization
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

---

## 6. File Operations (TS-600 Series)

### 6.1 Removal Order (TS-601)

**Requirement**: FR-601, FR-602
**Impact**: 36 files to be removed

**CRITICAL**: Files must be removed in dependency order to prevent compilation errors.

#### Phase 1: Remove Tests First (Lowest Risk)
```
DELETE ORDER - TESTS:
1. tests/fusion_tests.rs
2. tests/integration/fusion_integration_tests.rs
3. benches/fusion_bench.rs
4. tests/unit/fused_embedding_tests.rs
5. tests/unit/gating_tests.rs
6. tests/unit/expert_selector_tests.rs
```

#### Phase 2: Remove MCP Handlers
```
DELETE ORDER - MCP:
7. src/mcp/handlers/fused_search.rs
8. src/mcp/handlers/fusion_query.rs
```

#### Phase 3: Remove Search Components
```
DELETE ORDER - SEARCH:
9. src/search/fused_similarity.rs
10. src/search/gated_retrieval.rs
```

#### Phase 4: Remove Storage Components
```
DELETE ORDER - STORAGE:
11. src/storage/fused_vector_store.rs
12. src/storage/fusion_cache.rs
```

#### Phase 5: Remove Embedding Pipeline
```
DELETE ORDER - EMBEDDINGS:
13. src/embeddings/fusion_pipeline.rs
14. src/embeddings/fused_embedding.rs
15. src/embeddings/vector_1536.rs
```

#### Phase 6: Remove Core Fusion (Highest Risk - Do Last)
```
DELETE ORDER - CORE FUSION:
16. src/fusion/expert_selector.rs
17. src/fusion/gating.rs
18. src/fusion/fusion_config.rs
19. src/fusion/fuse_moe.rs
20. src/fusion/mod.rs
```

#### Phase 7: Remove Configuration
```
DELETE ORDER - CONFIG:
21. config/fusion.toml
22. config/gating_weights.yaml
23. config/expert_routing.yaml
```

#### Phase 8: Remove Types
```
DELETE ORDER - TYPES:
24. src/types/fused_types.rs
25. src/types/gating_types.rs
```

---

### 6.2 Modification Order (TS-602)

**Requirement**: Modify existing files to use new types

#### Trait Modifications (Highest Priority)
```
MODIFY ORDER - TRAITS:
1. crates/context-graph-core/src/traits/embedding_provider.rs
   - ADD: MultiArrayEmbeddingProvider trait
   - ADD: MultiArrayEmbeddingOutput struct
   - KEEP: Legacy EmbeddingProvider (deprecated marker)

2. crates/context-graph-core/src/traits/memory_store.rs
   - ADD: TeleologicalMemoryStore trait
   - ADD: MultiEmbeddingSearchOptions
   - ADD: SimilarityWeights
   - MODIFY: SearchOptions to support multi-embedding

3. crates/context-graph-core/src/traits/mod.rs
   - EXPORT: New traits and types
```

#### Type Additions
```
ADD ORDER - TYPES:
4. crates/context-graph-core/src/types/fingerprint/mod.rs (NEW)
5. crates/context-graph-core/src/types/fingerprint/semantic.rs (NEW)
6. crates/context-graph-core/src/types/fingerprint/teleological.rs (NEW)
7. crates/context-graph-core/src/types/fingerprint/johari.rs (NEW)
8. crates/context-graph-core/src/types/mod.rs
   - EXPORT: fingerprint module
```

#### Similarity Module
```
ADD ORDER - SIMILARITY:
9. crates/context-graph-core/src/similarity/mod.rs (NEW)
10. crates/context-graph-core/src/similarity/weighted.rs (NEW)
11. crates/context-graph-core/src/similarity/metrics.rs (NEW)
```

#### Meta-UTL Module
```
ADD ORDER - META_UTL:
12. crates/context-graph-core/src/meta_utl/mod.rs (NEW)
13. crates/context-graph-core/src/meta_utl/metrics.rs (NEW)
```

#### Storage Module
```
MODIFY ORDER - STORAGE:
14. crates/context-graph-storage/src/rocksdb/schema.rs (NEW)
15. crates/context-graph-storage/src/indexes/hnsw_config.rs (NEW)
16. crates/context-graph-storage/src/serialization.rs (NEW)
```

---

## 7. Traceability Matrix (TS -> FR)

| TS ID | TS Title | FR Reference | Status |
|-------|----------|--------------|--------|
| TS-101 | SemanticFingerprint | FR-101, FR-102, FR-103, FR-104 | Specified |
| TS-102 | TeleologicalFingerprint | FR-201, FR-202, FR-203, FR-204 | Specified |
| TS-103 | JohariFingerprint | FR-203 | Specified |
| TS-201 | RocksDB Schema | FR-301, FR-304 | Specified |
| TS-202 | HNSW Index Configuration | FR-302 | Specified |
| TS-203 | Serialization | FR-301, FR-103 | Specified |
| TS-301 | EmbeddingProvider Changes | FR-101, FR-102, FR-104 | Specified |
| TS-302 | MemoryStore Changes | FR-301, FR-302, FR-303, FR-401 | Specified |
| TS-401 | Weighted Similarity | FR-401, FR-402 | Specified |
| TS-402 | Distance Metrics | FR-403 | Specified |
| TS-501 | Meta-UTL Architecture | FR-501, FR-502 | Specified |
| TS-502 | Meta-UTL Metrics | FR-503 | Specified |
| TS-601 | Removal Order | FR-601, FR-602 | Specified |
| TS-602 | Modification Order | All FR | Specified |

---

## 8. Implementation Notes

### 8.1 Compilation Verification

All Rust code in this specification MUST compile. To verify:

```bash
# Create a test crate
cargo new --lib tech_spec_verify
cd tech_spec_verify

# Add dependencies to Cargo.toml
cat >> Cargo.toml << 'EOF'
async-trait = "0.1"
serde = { version = "1.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
bincode = "2.0.0-rc.3"
rocksdb = "0.22"
tracing = "0.1"
EOF

# Copy each code block and verify compilation
cargo check
```

### 8.2 Size Verification

Expected storage per fingerprint:

```
E1:  1024 * 4 =   4,096 bytes
E2:   512 * 4 =   2,048 bytes
E3:   512 * 4 =   2,048 bytes
E4:   512 * 4 =   2,048 bytes
E5:   768 * 4 * 2 = 6,144 bytes (query + doc)
E6:  ~1500 * 6 =   9,000 bytes (index u16 + value f32)
E7:  1536 * 4 =   6,144 bytes
E8:   384 * 4 =   1,536 bytes
E9:  1024 * 4 =   4,096 bytes
E10:  768 * 4 =   3,072 bytes
E11:  384 * 4 =   1,536 bytes
E12: ~100 * 128 * 4 = 51,200 bytes (100 tokens avg)
-----------------------------------
TOTAL:           ~92,968 bytes (~90KB with typical E12)
DENSE ONLY:      ~46,000 bytes (~46KB)
```

### 8.3 Performance Targets

| Operation | Target | Verification |
|-----------|--------|--------------|
| Single embed (all 12) | <30ms | Benchmark |
| Batch embed (64 x 12) | <100ms | Benchmark |
| Per-space HNSW search | <2ms | Benchmark |
| Purpose vector search | <1ms | Benchmark |
| Multi-space weighted sim | <5ms | Benchmark |
| Serialization (46KB) | <1ms | Benchmark |
| Deserialization (46KB) | <1ms | Benchmark |

---

**END OF TECHNICAL SPECIFICATION**
