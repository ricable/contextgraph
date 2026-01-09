# TASK-CORE-006: Storage Trait Alignment & Unification

```xml
<task_spec id="TASK-CORE-006" version="2.0">
<metadata>
  <title>Align Storage Traits with Teleological Architecture</title>
  <status>todo</status>
  <layer>foundation</layer>
  <sequence>6</sequence>
  <implements>
    <requirement_ref>REQ-STORAGE-TRAIT-01</requirement_ref>
    <requirement_ref>REQ-STORAGE-ASYNC-01</requirement_ref>
    <requirement_ref>ARCH-01 (TeleologicalArray is atomic)</requirement_ref>
    <requirement_ref>ARCH-02 (Apples-to-apples comparison)</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-CORE-004 (ComparisonValidationError - COMPLETED)</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
## CRITICAL: READ THIS FIRST

This task is about ALIGNING and UNIFYING existing storage traits, NOT creating new ones from scratch.

### What Already Exists (DO NOT RECREATE)

The storage crate already has substantial teleological infrastructure:

1. **TeleologicalMemoryStore trait** - `crates/context-graph-core/src/traits/teleological_memory_store.rs`
   - Full CRUD operations
   - Multi-space semantic search
   - Purpose vector search
   - Sparse (SPLADE) search
   - Batch operations
   - Persistence and checkpointing

2. **RocksDbTeleologicalStore** - `crates/context-graph-storage/src/teleological/rocksdb_store.rs` (1543 lines)
   - Full implementation of TeleologicalMemoryStore
   - 20 column families configured

3. **QuantizedFingerprintStorage trait** - `crates/context-graph-storage/src/teleological/quantized.rs` (919 lines)
   - Per-embedder quantized storage
   - RocksDbMemex implementation

4. **Memex trait** - `crates/context-graph-storage/src/memex.rs` (619 lines)
   - MemoryNode and GraphEdge storage
   - Different scope than teleological storage

5. **20 Column Families** - `crates/context-graph-storage/src/teleological/column_families.rs`
   - 7 core teleological CFs
   - 13 quantized embedder CFs (emb_0 through emb_12)

### What This Task Must Do

1. **Audit** existing traits for alignment with constitution.yaml
2. **Add** ComparisonValidationError integration from TASK-CORE-004
3. **Ensure** FAIL FAST semantics (no fallbacks)
4. **Verify** object-safety requirements
5. **Document** the unified storage architecture

### What This Task Must NOT Do

- Create redundant traits
- Add backwards compatibility shims
- Create mock implementations (use real data in tests)
- Recreate types that already exist
</context>

<objective>
Align existing storage traits (TeleologicalMemoryStore, QuantizedFingerprintStorage, Memex) with the
teleological architecture, integrating ComparisonValidationError from TASK-CORE-004 and ensuring
FAIL FAST semantics throughout.
</objective>

<rationale>
Per constitution.yaml ARCH-01: "TeleologicalArray is the Atomic Storage Unit"
Per constitution.yaml ARCH-02: "Compare Only Compatible Embedding Types"

The storage traits must enforce these architectural rules at the trait level, making violations
impossible at compile time and panicking at runtime when invalid states are detected.
</rationale>

<current_state>
## Existing File Inventory (ACTUAL PATHS)

### Core Crate Types
| File | Contains | Status |
|------|----------|--------|
| `crates/context-graph-core/src/traits/teleological_memory_store.rs` | TeleologicalMemoryStore trait (657 lines) | EXISTS |
| `crates/context-graph-core/src/types/fingerprint/teleological/mod.rs` | TeleologicalFingerprint struct | EXISTS |
| `crates/context-graph-core/src/types/fingerprint/semantic/mod.rs` | SemanticFingerprint (13-embedding array) | EXISTS |
| `crates/context-graph-core/src/teleological/embedder.rs` | Embedder enum (TASK-CORE-002) | EXISTS |
| `crates/context-graph-core/src/teleological/comparison_error.rs` | ComparisonValidationError (TASK-CORE-004) | EXISTS |

### Storage Crate Implementation
| File | Contains | Size | Status |
|------|----------|------|--------|
| `crates/context-graph-storage/src/teleological/mod.rs` | Module re-exports | 5.3KB | EXISTS |
| `crates/context-graph-storage/src/teleological/rocksdb_store.rs` | RocksDbTeleologicalStore | 54KB | EXISTS |
| `crates/context-graph-storage/src/teleological/quantized.rs` | QuantizedFingerprintStorage trait | 33KB | EXISTS |
| `crates/context-graph-storage/src/teleological/column_families.rs` | 20 CF definitions | 18KB | EXISTS |
| `crates/context-graph-storage/src/teleological/schema.rs` | Key format functions | 8.5KB | EXISTS |
| `crates/context-graph-storage/src/teleological/serialization.rs` | Serde for fingerprints | 11KB | EXISTS |
| `crates/context-graph-storage/src/memex.rs` | Memex trait | 619 lines | EXISTS |

### Completed Dependencies
| Task | Output | Location |
|------|--------|----------|
| TASK-CORE-002 | Embedder enum | `crates/context-graph-core/src/teleological/embedder.rs` |
| TASK-CORE-003 | SemanticFingerprint validation | `crates/context-graph-core/src/types/fingerprint/semantic/validation.rs` |
| TASK-CORE-004 | ComparisonValidationError | `crates/context-graph-core/src/teleological/comparison_error.rs` |
</current_state>

<input_context_files>
  <!-- EXISTING FILES TO READ AND UNDERSTAND -->
  <file purpose="existing_trait" critical="true">crates/context-graph-core/src/traits/teleological_memory_store.rs</file>
  <file purpose="existing_impl" critical="true">crates/context-graph-storage/src/teleological/rocksdb_store.rs</file>
  <file purpose="quantized_trait">crates/context-graph-storage/src/teleological/quantized.rs</file>
  <file purpose="column_families">crates/context-graph-storage/src/teleological/column_families.rs</file>
  <file purpose="comparison_errors">crates/context-graph-core/src/teleological/comparison_error.rs</file>
  <file purpose="fingerprint_types">crates/context-graph-core/src/types/fingerprint/teleological/mod.rs</file>
  <file purpose="semantic_fingerprint">crates/context-graph-core/src/types/fingerprint/semantic/mod.rs</file>
  <file purpose="constitution">docs2/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <check>TASK-CORE-004 complete - ComparisonValidationError exists at crates/context-graph-core/src/teleological/comparison_error.rs</check>
  <check>TeleologicalMemoryStore trait exists at crates/context-graph-core/src/traits/teleological_memory_store.rs</check>
  <check>RocksDbTeleologicalStore implementation exists at crates/context-graph-storage/src/teleological/rocksdb_store.rs</check>
  <check>SemanticFingerprint type exists at crates/context-graph-core/src/types/fingerprint/semantic/mod.rs</check>
</prerequisites>

<scope>
  <in_scope>
    <item>Audit TeleologicalMemoryStore trait for constitution alignment</item>
    <item>Add ComparisonValidationError to storage error types</item>
    <item>Verify FAIL FAST semantics (no Option fallbacks where Result is appropriate)</item>
    <item>Verify object-safety of all traits</item>
    <item>Document trait hierarchy and relationships</item>
    <item>Add validation methods that use ComparisonValidationError</item>
    <item>Ensure all 13 embedders are handled (no partial storage)</item>
  </in_scope>
  <out_of_scope>
    <item>Creating new storage traits from scratch (use existing)</item>
    <item>Per-embedder HNSW index implementation (TASK-CORE-007)</item>
    <item>RocksDB schema changes (TASK-CORE-008)</item>
    <item>Mock implementations (tests use real InMemoryTeleologicalStore)</item>
    <item>Backwards compatibility with deprecated traits</item>
  </out_of_scope>
</scope>

<definition_of_done>
  <modifications>
    <!-- MODIFICATION 1: Add ComparisonValidationError to storage errors -->
    <modification file="crates/context-graph-storage/src/teleological/rocksdb_store.rs">
      Add ComparisonValidationError to TeleologicalStoreError enum:

      ```rust
      use context_graph_core::teleological::ComparisonValidationError;

      #[derive(Debug, thiserror::Error)]
      pub enum TeleologicalStoreError {
          // ... existing variants ...

          /// Validation error for comparison types
          #[error("Comparison validation failed: {0}")]
          ComparisonValidation(#[from] ComparisonValidationError),
      }
      ```
    </modification>

    <!-- MODIFICATION 2: Add validation method to trait extension -->
    <modification file="crates/context-graph-core/src/traits/teleological_memory_store.rs">
      Add validation extension using ComparisonValidationError:

      ```rust
      use crate::teleological::{ComparisonValidationError, ComparisonValidationResult};

      impl<T: TeleologicalMemoryStore> TeleologicalMemoryStoreExt for T {
          // ... existing methods ...

          /// Validate a fingerprint before storage.
          /// Returns ComparisonValidationError on validation failure.
          fn validate_for_storage(
              &self,
              fingerprint: &TeleologicalFingerprint,
          ) -> ComparisonValidationResult<()> {
              // Validate all 13 embeddings are present (ARCH-05)
              fingerprint.semantic_fingerprint.validate()?;
              // Validate purpose vector is normalized
              fingerprint.purpose_vector.validate()?;
              Ok(())
          }
      }
      ```
    </modification>
  </modifications>

  <verification_queries>
    <!-- Query 1: Verify ComparisonValidationError is integrated -->
    <query>
      grep -r "ComparisonValidationError" crates/context-graph-storage/src/teleological/
      Expected: At least one file imports and uses ComparisonValidationError
    </query>

    <!-- Query 2: Verify FAIL FAST (no unwrap_or_default) -->
    <query>
      grep -r "unwrap_or_default\|unwrap_or\|or_else" crates/context-graph-storage/src/teleological/rocksdb_store.rs
      Expected: Zero matches (all errors should propagate)
    </query>

    <!-- Query 3: Verify trait is object-safe -->
    <query>
      cargo check -p context-graph-core 2>&1 | grep "cannot be made into an object"
      Expected: Zero matches
    </query>

    <!-- Query 4: Verify tests pass -->
    <query>
      cargo test -p context-graph-core teleological_memory_store --no-fail-fast
      Expected: All tests pass
    </query>
  </verification_queries>

  <constraints>
    <constraint>TeleologicalMemoryStore must remain object-safe (dyn TeleologicalMemoryStore works)</constraint>
    <constraint>All methods must propagate errors via Result (no silent failures)</constraint>
    <constraint>No backwards compatibility - old traits are deleted not deprecated</constraint>
    <constraint>FAIL FAST: panic on invariant violations, never silently continue</constraint>
    <constraint>All 13 embeddings must be validated before storage (ARCH-05)</constraint>
    <constraint>No mock data in tests - use InMemoryTeleologicalStore with real fingerprints</constraint>
  </constraints>

  <verification>
    <command>cargo check -p context-graph-storage</command>
    <command>cargo test -p context-graph-core traits::teleological_memory_store</command>
    <command>cargo test -p context-graph-storage teleological</command>
  </verification>
</definition_of_done>

<existing_types_reference>
## TeleologicalMemoryStore Trait (EXISTING - DO NOT RECREATE)

Location: `crates/context-graph-core/src/traits/teleological_memory_store.rs`

```rust
#[async_trait]
pub trait TeleologicalMemoryStore: Send + Sync {
    // CRUD Operations
    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid>;
    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>>;
    async fn update(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<bool>;
    async fn delete(&self, id: Uuid, soft: bool) -> CoreResult<bool>;

    // Search Operations
    async fn search_semantic(&self, query: &SemanticFingerprint, options: TeleologicalSearchOptions) -> CoreResult<Vec<TeleologicalSearchResult>>;
    async fn search_purpose(&self, query: &PurposeVector, options: TeleologicalSearchOptions) -> CoreResult<Vec<TeleologicalSearchResult>>;
    async fn search_text(&self, text: &str, options: TeleologicalSearchOptions) -> CoreResult<Vec<TeleologicalSearchResult>>;
    async fn search_sparse(&self, sparse_query: &SparseVector, top_k: usize) -> CoreResult<Vec<(Uuid, f32)>>;

    // Batch Operations
    async fn store_batch(&self, fingerprints: Vec<TeleologicalFingerprint>) -> CoreResult<Vec<Uuid>>;
    async fn retrieve_batch(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<TeleologicalFingerprint>>>;

    // Statistics
    async fn count(&self) -> CoreResult<usize>;
    async fn count_by_quadrant(&self) -> CoreResult<[usize; 4]>;
    fn storage_size_bytes(&self) -> usize;
    fn backend_type(&self) -> TeleologicalStorageBackend;

    // Persistence
    async fn flush(&self) -> CoreResult<()>;
    async fn checkpoint(&self) -> CoreResult<PathBuf>;
    async fn restore(&self, checkpoint_path: &Path) -> CoreResult<()>;
    async fn compact(&self) -> CoreResult<()>;

    // Scanning
    async fn list_by_quadrant(&self, quadrant: usize, limit: usize) -> CoreResult<Vec<(Uuid, JohariFingerprint)>>;
    async fn list_all_johari(&self, limit: usize) -> CoreResult<Vec<(Uuid, JohariFingerprint)>>;
}
```

## ComparisonValidationError (TASK-CORE-004 - COMPLETED)

Location: `crates/context-graph-core/src/teleological/comparison_error.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonValidationError {
    WeightsNotNormalized { actual_sum: f32, expected_sum: f32, tolerance: f32, weights: WeightValues },
    WeightOutOfRange { field_name: &'static str, value: f32, min: f32, max: f32 },
    MatrixNotSymmetric { row: usize, col: usize, value_ij: f32, value_ji: f32, tolerance: f32 },
    DiagonalNotUnity { index: usize, actual: f32, expected: f32, tolerance: f32 },
    SynergyOutOfRange { row: usize, col: usize, value: f32, min: f32, max: f32 },
    SimilarityOutOfRange { component: &'static str, value: f32 },
    BreakdownInconsistent { computed_overall: f32, stored_overall: f32, tolerance: f32 },
}

pub type ComparisonValidationResult<T> = Result<T, ComparisonValidationError>;
```

## TeleologicalStoreError (EXISTING)

Location: `crates/context-graph-storage/src/teleological/rocksdb_store.rs`

```rust
#[derive(Debug, thiserror::Error)]
pub enum TeleologicalStoreError {
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Fingerprint not found: {0}")]
    NotFound(Uuid),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Index error: {0}")]
    Index(String),
    // ADD THIS: ComparisonValidation variant
}
```
</existing_types_reference>

<implementation_steps>
## Step 1: Audit Existing Traits

Read and document the current state of:
1. TeleologicalMemoryStore trait methods
2. TeleologicalStoreError variants
3. RocksDbTeleologicalStore implementation

Verify alignment with constitution.yaml requirements.

## Step 2: Add ComparisonValidationError Integration

1. Import ComparisonValidationError in rocksdb_store.rs
2. Add ComparisonValidation variant to TeleologicalStoreError
3. Implement From<ComparisonValidationError> for TeleologicalStoreError

## Step 3: Add Validation Extension Methods

Add to TeleologicalMemoryStoreExt:
- validate_for_storage() - pre-storage validation
- validate_fingerprint_integrity() - post-retrieval validation

## Step 4: Verify FAIL FAST Semantics

Audit all methods in RocksDbTeleologicalStore:
- Remove any .unwrap_or() or .unwrap_or_default() calls
- Ensure all errors propagate via Result
- Add panic!() for invariant violations that should never occur

## Step 5: Update Tests

Location: `crates/context-graph-core/src/traits/teleological_memory_store_tests.rs`

Add tests:
- test_comparison_validation_error_propagation()
- test_invalid_fingerprint_rejected()
- test_partial_embeddings_rejected()

Use real fingerprints, not mock data.
</implementation_steps>

<full_state_verification>
## Source of Truth Verification

After implementation, verify:

1. **Column Family Existence**
   ```bash
   # Verify 20 CFs are defined
   grep -c "pub const CF_" crates/context-graph-storage/src/teleological/column_families.rs
   # Expected: 20
   ```

2. **Error Variant Integration**
   ```bash
   # Verify ComparisonValidation in error enum
   grep "ComparisonValidation" crates/context-graph-storage/src/teleological/rocksdb_store.rs
   # Expected: At least one match
   ```

3. **FAIL FAST Compliance**
   ```bash
   # Verify no silent fallbacks
   grep -E "unwrap_or|or_else|unwrap_or_default" crates/context-graph-storage/src/teleological/rocksdb_store.rs
   # Expected: 0 matches
   ```

## Execute & Inspect

```bash
# Run trait-specific tests
cargo test -p context-graph-core traits::teleological_memory_store -- --nocapture

# Run storage tests
cargo test -p context-graph-storage teleological -- --nocapture

# Verify object safety
cargo check -p context-graph-core 2>&1 | grep -i "object"
```

## Boundary & Edge Case Audit

Test these scenarios:
1. Store fingerprint with zeroed E3 embedding - should FAIL (ARCH-05)
2. Store fingerprint with NaN in purpose vector - should FAIL (ARCH-02)
3. Store fingerprint with mismatched dimensions - should FAIL (type system)
4. Retrieve non-existent UUID - should return None (not panic)
5. Search with invalid ComparisonType - should return ComparisonValidationError

## Evidence of Success

Manual verification checklist:
- [ ] `cargo test -p context-graph-core traits` passes
- [ ] `cargo test -p context-graph-storage teleological` passes
- [ ] `cargo check -p context-graph-storage` produces no warnings
- [ ] No "deprecated" warnings in output
- [ ] ComparisonValidationError appears in storage error enum
</full_state_verification>

<test_commands>
  <command>cargo check -p context-graph-storage</command>
  <command>cargo test -p context-graph-core traits::teleological_memory_store</command>
  <command>cargo test -p context-graph-storage teleological</command>
  <command>cargo clippy -p context-graph-storage -- -D warnings</command>
</test_commands>

<files_to_modify>
  <file path="crates/context-graph-storage/src/teleological/rocksdb_store.rs" action="modify">
    Add ComparisonValidation variant to TeleologicalStoreError
    Import ComparisonValidationError from context_graph_core
  </file>
  <file path="crates/context-graph-core/src/traits/teleological_memory_store.rs" action="modify">
    Add validate_for_storage() to TeleologicalMemoryStoreExt
    Import ComparisonValidationError
  </file>
</files_to_modify>

<files_to_create>
  <!-- NO NEW FILES - This task modifies existing files only -->
</files_to_create>

<validation_criteria>
  <criterion>ComparisonValidationError is integrated into storage error hierarchy</criterion>
  <criterion>TeleologicalMemoryStore remains object-safe</criterion>
  <criterion>All storage methods propagate errors (no silent failures)</criterion>
  <criterion>Pre-storage validation rejects invalid fingerprints</criterion>
  <criterion>Tests pass without mock data</criterion>
  <criterion>FAIL FAST semantics enforced (panics on invariant violations)</criterion>
</validation_criteria>
</task_spec>
```

## Quick Reference: Key Files

| Purpose | Path |
|---------|------|
| Main trait | `crates/context-graph-core/src/traits/teleological_memory_store.rs` |
| Main impl | `crates/context-graph-storage/src/teleological/rocksdb_store.rs` |
| Error types | `crates/context-graph-core/src/teleological/comparison_error.rs` |
| Column families | `crates/context-graph-storage/src/teleological/column_families.rs` |
| Fingerprint type | `crates/context-graph-core/src/types/fingerprint/teleological/mod.rs` |
| 13-embedding array | `crates/context-graph-core/src/types/fingerprint/semantic/mod.rs` |

## Key Principles

1. **NO NEW TRAITS** - Modify existing TeleologicalMemoryStore
2. **NO MOCK DATA** - Tests use real InMemoryTeleologicalStore
3. **NO BACKWARDS COMPAT** - Delete deprecated code, don't deprecate it
4. **FAIL FAST** - Panic on invariants, Result on expected errors
5. **13 EMBEDDINGS** - All or nothing (ARCH-05)
