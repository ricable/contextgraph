# Task: TASK-F004 - Implement RocksDB Storage Schema for 46KB Fingerprints

## Metadata
- **ID**: TASK-F004
- **Layer**: Foundation
- **Priority**: P0 (Critical Path)
- **Estimated Effort**: L (Large)
- **Dependencies**: TASK-F001, TASK-F002, TASK-F003
- **Traces To**: TS-201, TS-203, FR-301, FR-304

## Description

Implement the RocksDB column family schema and bincode serialization for storing 46KB TeleologicalFingerprints. This task creates the storage layer that persists the multi-array fingerprints efficiently.

The storage architecture uses 6 column families:
1. **fingerprints** - Primary 46KB fingerprints (optimized for large values)
2. **purpose_vectors** - 12D purpose vectors (48 bytes, fast lookups)
3. **johari_index** - Bitmap index by quadrant per embedder
4. **goal_alignment** - Memory-to-goal alignment cache
5. **purpose_evolution** - Time-series snapshots
6. **metadata** - Node metadata

## Acceptance Criteria

- [ ] RocksDB column family definitions for all 6 families
- [ ] Column family options optimized for value sizes (46KB vs 48 bytes)
- [ ] Key format functions for each family
- [ ] Bincode serialization for SemanticFingerprint
- [ ] Bincode deserialization with version checking
- [ ] FingerprintHeader for variable-length fields (E6, E12)
- [ ] Compression configuration (LZ4)
- [ ] Unit tests for serialization round-trip
- [ ] Integration tests with actual RocksDB instance

## Implementation Steps

1. Create `crates/context-graph-storage/src/teleological/mod.rs`:
   - Define module structure
2. Create `crates/context-graph-storage/src/teleological/schema.rs`:
   - Define column family name constants
   - Implement `TeleologicalSchema` with CF options
   - Implement key format functions
3. Create `crates/context-graph-storage/src/teleological/serialization.rs`:
   - Define `SERIALIZATION_VERSION = 1`
   - Implement `FingerprintHeader` struct
   - Implement `serialize_semantic_fingerprint()`
   - Implement `deserialize_semantic_fingerprint()`
   - Implement `serialize_teleological_fingerprint()`
   - Implement `deserialize_teleological_fingerprint()`
4. Update `crates/context-graph-storage/src/lib.rs` to export teleological module
5. Add bincode dependency to Cargo.toml if not present

## Files Affected

### Files to Create
- `crates/context-graph-storage/src/teleological/mod.rs` - Module definition
- `crates/context-graph-storage/src/teleological/schema.rs` - RocksDB schema
- `crates/context-graph-storage/src/teleological/serialization.rs` - Bincode serialization

### Files to Modify
- `crates/context-graph-storage/src/lib.rs` - Export teleological module
- `crates/context-graph-storage/Cargo.toml` - Add bincode dependency

### Existing Files to Reference
- `crates/context-graph-storage/src/column_families.rs` - Existing CF patterns
- `crates/context-graph-storage/src/serialization.rs` - Existing serialization patterns

## Code Signature (Definition of Done)

```rust
// schema.rs
pub const CF_FINGERPRINTS: &str = "fingerprints";
pub const CF_PURPOSE_VECTORS: &str = "purpose_vectors";
pub const CF_JOHARI_INDEX: &str = "johari_index";
pub const CF_GOAL_ALIGNMENT: &str = "goal_alignment";
pub const CF_EVOLUTION: &str = "purpose_evolution";
pub const CF_METADATA: &str = "metadata";

pub struct TeleologicalSchema;

impl TeleologicalSchema {
    /// Options optimized for 46KB values
    pub fn fingerprint_cf_options() -> Options;

    /// Options optimized for 48-byte purpose vectors
    pub fn purpose_vector_cf_options() -> Options;

    /// Open DB with all column families
    pub fn open(path: impl AsRef<Path>) -> Result<DB, rocksdb::Error>;
}

/// Key: UUID as 16 bytes
pub fn fingerprint_key(id: &Uuid) -> [u8; 16];

/// Key: UUID as 16 bytes
pub fn purpose_vector_key(id: &Uuid) -> [u8; 16];

/// Key: (quadrant_u8, embedder_u8, memory_id_bytes)
pub fn johari_index_key(quadrant: u8, embedder: u8, memory_id: &Uuid) -> Vec<u8>;

/// Key: (memory_id_bytes, goal_id_bytes)
pub fn goal_alignment_key(memory_id: &Uuid, goal_id: &Uuid) -> Vec<u8>;

/// Key: (memory_id_bytes, timestamp_i64_be)
pub fn evolution_key(memory_id: &Uuid, timestamp_nanos: i64) -> Vec<u8>;

// serialization.rs
pub const SERIALIZATION_VERSION: u8 = 1;

#[derive(Debug, Clone, Encode, Decode)]
pub struct FingerprintHeader {
    pub version: u8,
    pub total_size: u32,
    pub e12_token_count: u16,
    pub e6_active_count: u16,
}

pub fn serialize_semantic_fingerprint(fp: &SemanticFingerprint) -> Vec<u8>;
pub fn deserialize_semantic_fingerprint(data: &[u8]) -> Result<SemanticFingerprint, &'static str>;

pub fn serialize_teleological_fingerprint(fp: &TeleologicalFingerprint) -> Vec<u8>;
pub fn deserialize_teleological_fingerprint(data: &[u8]) -> Result<TeleologicalFingerprint, &'static str>;
```

## Testing Requirements

### Unit Tests
- `test_fingerprint_key_format` - 16-byte UUID key
- `test_johari_index_key_format` - Correct composite key
- `test_goal_alignment_key_format` - 32-byte composite key
- `test_evolution_key_format` - Sortable timestamp key
- `test_serialize_semantic_roundtrip` - Encode then decode matches
- `test_serialize_teleological_roundtrip` - Full fingerprint round-trip
- `test_serialize_sparse_vector` - E6 variable length handling
- `test_serialize_token_level` - E12 variable length handling
- `test_version_check` - Rejects wrong version

### Integration Tests
- `test_rocksdb_open_all_cfs` - Opens DB with 6 column families
- `test_rocksdb_store_retrieve_fingerprint` - Full write/read cycle
- `test_rocksdb_cf_options_large_values` - Correct settings for 46KB

## Verification

```bash
# Compile check
cargo check -p context-graph-storage

# Run unit tests
cargo test -p context-graph-storage teleological

# Run integration tests
cargo test -p context-graph-storage --test rocksdb_integration
```

## Constraints

- Bincode 2.0 (rc.3) for efficient binary serialization
- LZ4 compression for space efficiency
- 64KB block size for 46KB values
- 256MB SST files for fingerprints CF
- 64MB SST files for purpose vectors CF
- Big-endian timestamps for sorted range scans
- Bloom filter on purpose vectors for fast lookups

## Performance Targets

| Operation | Target |
|-----------|--------|
| Serialization (46KB) | <1ms |
| Deserialization (46KB) | <1ms |
| RocksDB write | <5ms |
| RocksDB read | <2ms |

## Notes

This task creates the storage foundation. The actual store implementation that uses these schemas comes in Logic Layer tasks.

The 46KB storage per node is ~7.5x larger than legacy 6KB fused vectors, but preserves 100% information vs 33% with fusion.

Reference implementation in TECH-SPEC-001 Sections 2.1 and 2.3 (TS-201, TS-203).
