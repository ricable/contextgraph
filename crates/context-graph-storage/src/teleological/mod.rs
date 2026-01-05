//! Teleological fingerprint storage extensions.
//!
//! Adds 4 column families for ~63KB TeleologicalFingerprint storage
//! and 5-stage pipeline indexing.
//!
//! # Column Families (4 new, 16 total)
//!
//! | Name | Purpose | Key Format | Value Size |
//! |------|---------|------------|------------|
//! | fingerprints | Primary ~63KB TeleologicalFingerprints | UUID (16 bytes) | ~63KB |
//! | purpose_vectors | 13D purpose vectors | UUID (16 bytes) | 52 bytes |
//! | e13_splade_inverted | Inverted index for E13 SPLADE | term_id (2 bytes) | Vec<UUID> |
//! | e1_matryoshka_128 | E1 Matryoshka 128D truncated vectors | UUID (16 bytes) | 512 bytes |
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**
//!
//! All errors panic with full context. No silent fallbacks or default values.
//! This ensures data integrity and makes bugs immediately visible.

pub mod column_families;
pub mod indexes;
pub mod schema;
pub mod serialization;

#[cfg(test)]
mod tests;

// Re-export column family types
pub use column_families::{
    e1_matryoshka_128_cf_options, e13_splade_inverted_cf_options, fingerprint_cf_options,
    get_teleological_cf_descriptors, purpose_vector_cf_options, CF_E1_MATRYOSHKA_128,
    CF_E13_SPLADE_INVERTED, CF_FINGERPRINTS, CF_PURPOSE_VECTORS, TELEOLOGICAL_CFS,
};

// Re-export schema types
pub use schema::{
    e13_splade_inverted_key, e1_matryoshka_128_key, fingerprint_key, parse_e13_splade_key,
    parse_fingerprint_key, purpose_vector_key, parse_purpose_vector_key, parse_e1_matryoshka_key,
};

// Re-export serialization types
pub use serialization::{
    deserialize_e1_matryoshka_128, deserialize_memory_id_list, deserialize_purpose_vector,
    deserialize_teleological_fingerprint, serialize_e1_matryoshka_128, serialize_memory_id_list,
    serialize_purpose_vector, serialize_teleological_fingerprint, TELEOLOGICAL_VERSION,
};

// Re-export index configuration types (TASK-F005)
pub use indexes::{
    // Enums
    DistanceMetric, EmbedderIndex,
    // Structs
    HnswConfig, InvertedIndexConfig,
    // Config functions
    all_hnsw_configs, get_hnsw_config, get_inverted_index_config,
    // Metric functions
    compute_distance, cosine_similarity, distance_to_similarity, recommended_metric,
    // Dimension constants
    E1_DIM, E1_MATRYOSHKA_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM, E6_SPARSE_VOCAB, E7_DIM, E8_DIM,
    E9_DIM, E10_DIM, E11_DIM, E12_TOKEN_DIM, E13_SPLADE_VOCAB, NUM_EMBEDDERS, PURPOSE_VECTOR_DIM,
};
