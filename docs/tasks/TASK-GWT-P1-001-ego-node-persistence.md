# TASK-GWT-P1-001: SELF_EGO_NODE Persistence Layer

## ✅ STATUS: COMPLETED (2026-01-10)

All implementation complete. 12 tests passing. Code merged.

---

## Quick Reference

| Item | Value |
|------|-------|
| Column Family | `CF_EGO_NODE` ("ego_node") |
| Key | Fixed 8-byte string `"ego_node"` |
| Serialization | bincode v1.3 with version byte prefix |
| Version | `EGO_NODE_VERSION = 1` |
| Total CFs | 9 teleological + 13 quantized = 22 |

---

## Files Modified (Verified)

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | +32 | Added `Serialize, Deserialize` derives |
| `crates/context-graph-core/src/stubs/teleological_store_stub.rs` | +61 | Added `ego_node` field and stub methods |
| `crates/context-graph-core/src/traits/teleological_memory_store.rs` | +53 | Added `save_ego_node()`, `load_ego_node()` trait methods |
| `crates/context-graph-storage/src/teleological/column_families.rs` | +81 | Added `CF_EGO_NODE`, `ego_node_cf_options()` |
| `crates/context-graph-storage/src/teleological/schema.rs` | +26 | Added `EGO_NODE_KEY`, `ego_node_key()` |
| `crates/context-graph-storage/src/teleological/serialization.rs` | +161 | Added `serialize_ego_node()`, `deserialize_ego_node()` |
| `crates/context-graph-storage/src/teleological/rocksdb_store.rs` | +107 | Implemented trait methods with `cf_ego_node()` helper |
| `crates/context-graph-storage/src/teleological/mod.rs` | +26 | Re-exports for ego node functions |
| `crates/context-graph-storage/src/teleological/tests.rs` | +385 | 12 comprehensive tests |

---

## Test Results (All Pass)

```
cargo test -p context-graph-storage ego_node

running 12 tests
test teleological::tests::test_ego_node_deserialize_empty_panics - should panic ... ok
test teleological::tests::test_ego_node_in_cf_array ... ok
test teleological::tests::test_ego_node_deserialize_wrong_version_panics - should panic ... ok
test teleological::tests::test_ego_node_key_constant ... ok
test teleological::tests::test_ego_node_cf_options_valid ... ok
test teleological::tests::test_serialize_ego_node_roundtrip ... ok
test teleological::tests::test_ego_node_version_constant ... ok
test teleological::tests::test_serialize_ego_node_with_large_trajectory ... ok
test teleological::tests::test_in_memory_store_ego_node_roundtrip ... ok
test teleological::tests::test_ego_node_overwrite ... ok
test teleological::tests::test_ego_node_save_load_roundtrip ... ok
test teleological::tests::test_ego_node_persistence_across_reopen ... ok

test result: ok. 12 passed; 0 failed; 0 ignored
```

---

## Key Implementation Details

### 1. Column Family: CF_EGO_NODE

**Location:** `crates/context-graph-storage/src/teleological/column_families.rs`

```rust
pub const CF_EGO_NODE: &str = "ego_node";

pub fn ego_node_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.optimize_for_point_lookup(16);
    opts.create_if_missing(true);
    opts
}
```

### 2. Key Schema

**Location:** `crates/context-graph-storage/src/teleological/schema.rs`

```rust
pub const EGO_NODE_KEY: &[u8] = b"ego_node";  // 8 bytes, fixed

#[inline]
pub const fn ego_node_key() -> &'static [u8] {
    EGO_NODE_KEY
}
```

### 3. Serialization (FAIL FAST)

**Location:** `crates/context-graph-storage/src/teleological/serialization.rs`

```rust
pub const EGO_NODE_VERSION: u8 = 1;
const MIN_EGO_NODE_SIZE: usize = 50;
const MAX_EGO_NODE_SIZE: usize = 300_000;

pub fn serialize_ego_node(ego: &SelfEgoNode) -> Vec<u8>;
pub fn deserialize_ego_node(data: &[u8]) -> SelfEgoNode;
```

- Panics on empty data
- Panics on version mismatch
- Panics on serialization failure
- Size bounds: 50 bytes to 300KB

### 4. Trait Methods

**Location:** `crates/context-graph-core/src/traits/teleological_memory_store.rs`

```rust
async fn save_ego_node(&self, ego_node: &SelfEgoNode) -> CoreResult<()>;
async fn load_ego_node(&self) -> CoreResult<Option<SelfEgoNode>>;
```

### 5. Serde Derives

**Location:** `crates/context-graph-core/src/gwt/ego_node.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEgoNode { ... }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeSnapshot { ... }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityContinuity { ... }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityStatus { ... }
```

---

## Constitution Reference

From `docs2/constitution.yaml` lines 365-369:

```yaml
self_ego_node:
  id: "SELF_EGO_NODE"
  fields: [fingerprint, purpose_vector, identity_trajectory, coherence_with_actions]
  loop: "Retrieve→A(action,PV)→if<0.55 self_reflect→update fingerprint→store evolution"
  identity_continuity: "IC = cos(PV_t, PV_{t-1}) × r(t); healthy>0.9, warning<0.7, dream<0.5"
```

---

## Verification Commands

```bash
# Build both crates
cargo build -p context-graph-core -p context-graph-storage

# Run ego node tests
cargo test -p context-graph-storage ego_node

# Run all teleological tests
cargo test -p context-graph-storage teleological

# Check clippy
cargo clippy -p context-graph-core -p context-graph-storage -- -D warnings
```

---

## Full State Verification Evidence

### Source of Truth
- **Database:** RocksDB `CF_EGO_NODE` column family
- **Key:** Fixed `"ego_node"` (8 bytes)
- **Value:** Version byte + bincode payload

### Test: Persistence Across Reopen (FSV)

```
=== TEST: Ego node persists across store close/reopen ===
STEP 1: Saving ego node id=... with 5 snapshots
STEP 1: Closing store...
STEP 2: Reopening store...
STEP 2: Loaded ego node id=... with 5 snapshots
RESULT: PASS - Ego node persists across store close/reopen
```

### Edge Cases Verified

1. **Empty trajectory** - Minimal ego node with no snapshots
2. **Large trajectory** - 100 snapshots (~10KB serialized)
3. **Overwrite** - Second save replaces first correctly
4. **Version mismatch** - Panics as expected (FAIL FAST)
5. **Empty data** - Panics as expected (FAIL FAST)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   SelfEgoNode                           │
│  ├── id: Uuid                                           │
│  ├── fingerprint: Option<TeleologicalFingerprint>       │
│  ├── purpose_vector: [f32; 13]                          │
│  ├── coherence_with_actions: f32                        │
│  ├── identity_trajectory: Vec<PurposeSnapshot>          │
│  └── last_updated: DateTime<Utc>                        │
└─────────────────────────────────────────────────────────┘
                          │
                          │ save_ego_node() / load_ego_node()
                          ▼
┌─────────────────────────────────────────────────────────┐
│              TeleologicalMemoryStore                    │
│  ├── save_ego_node(&SelfEgoNode) -> CoreResult<()>      │
│  └── load_ego_node() -> CoreResult<Option<SelfEgoNode>> │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              RocksDB Column Family                      │
│  CF_EGO_NODE: "ego_node" (9th teleological CF)          │
│  Key: "ego_node" (8 bytes, singleton)                   │
│  Value: version_byte + bincode(SelfEgoNode)             │
└─────────────────────────────────────────────────────────┘
```

---

## Related Tasks

| Task | Status | Relationship |
|------|--------|--------------|
| TASK-GWT-P0-003 | ✅ COMPLETED | Prerequisite - SelfAwarenessLoop activation |
| TASK-GWT-P1-002 | Ready | Depends on ego node for workspace events |

---

## Build Status

```
cargo build -p context-graph-core -p context-graph-storage
# Finished `dev` profile in 0.22s

cargo clippy -p context-graph-core -- -D warnings
# No warnings

cargo clippy -p context-graph-storage
# Pre-existing warnings in matrix.rs (unrelated to this task)
```
