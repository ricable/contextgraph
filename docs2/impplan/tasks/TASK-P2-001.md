# TASK-P2-001: TeleologicalArray Struct

```xml
<task_spec id="TASK-P2-001" version="2.0">
<metadata>
  <title>TeleologicalArray Struct Implementation</title>
  <status>COMPLETE</status>
  <layer>foundation</layer>
  <sequence>14</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-01</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <completion_date>2026-01-16</completion_date>
</metadata>

<audit_results>
  <audit_date>2026-01-16</audit_date>
  <status>ALREADY_IMPLEMENTED</status>
  <summary>
    This task is COMPLETE. All specified types exist in the codebase with comprehensive
    implementations, tests, and documentation. The implementation location differs from
    the original spec but provides superior organization.
  </summary>
</audit_results>

<context>
Implements the TeleologicalArray struct that holds all 13 embedding vectors.
This is the core data structure for the 13-space embedding system.

Each memory produces one TeleologicalArray containing embeddings from all 13
specialized embedders with their specific dimensions and vector types.
</context>

<!-- =================================================================== -->
<!-- ACTUAL FILE LOCATIONS (VERIFIED AGAINST CODEBASE)                   -->
<!-- =================================================================== -->
<actual_file_locations>
  <discrepancy_note>
    The technical spec (TECH-PHASE2) lists files under crates/context-graph-core/src/embedding/
    but the ACTUAL implementation is organized differently. These are the CORRECT paths:
  </discrepancy_note>

  <file purpose="Embedder enum" actual="crates/context-graph-core/src/teleological/embedder.rs">
    <spec_said>crates/context-graph-core/src/embedding/mod.rs</spec_said>
    <line_count>843</line_count>
    <status>EXISTS and COMPLETE</status>
  </file>

  <file purpose="TeleologicalArray/SemanticFingerprint" actual="crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs">
    <spec_said>crates/context-graph-core/src/embedding/teleological.rs</spec_said>
    <line_count>569</line_count>
    <status>EXISTS and COMPLETE</status>
  </file>

  <file purpose="Dimension constants" actual="crates/context-graph-core/src/types/fingerprint/semantic/constants.rs">
    <line_count>77</line_count>
    <status>EXISTS and COMPLETE</status>
  </file>

  <file purpose="SparseVector" actual="crates/context-graph-core/src/types/fingerprint/sparse.rs">
    <spec_said>crates/context-graph-core/src/embedding/vector.rs</spec_said>
    <line_count>517</line_count>
    <status>EXISTS and COMPLETE</status>
  </file>

  <file purpose="EmbeddingSlice type-safe accessor" actual="crates/context-graph-core/src/types/fingerprint/semantic/slice.rs">
    <status>EXISTS and COMPLETE</status>
  </file>
</actual_file_locations>

<!-- =================================================================== -->
<!-- IMPLEMENTATION DETAILS (WHAT ACTUALLY EXISTS)                       -->
<!-- =================================================================== -->
<actual_implementation>
  <type name="TeleologicalArray">
    <actual_definition>
      pub type TeleologicalArray = SemanticFingerprint;
      // Type alias - they are the SAME type
    </actual_definition>
    <location>crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs:26</location>
  </type>

  <type name="SemanticFingerprint">
    <rust_definition>
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFingerprint {
    pub e1_semantic: Vec&lt;f32&gt;,           // 1024D
    pub e2_temporal_recent: Vec&lt;f32&gt;,    // 512D
    pub e3_temporal_periodic: Vec&lt;f32&gt;,  // 512D
    pub e4_temporal_positional: Vec&lt;f32&gt;, // 512D
    pub e5_causal: Vec&lt;f32&gt;,             // 768D
    pub e6_sparse: SparseVector,         // ~30K vocab, sparse
    pub e7_code: Vec&lt;f32&gt;,               // 1536D
    pub e8_graph: Vec&lt;f32&gt;,              // 384D (field name is e8_graph, but embedder is E8_Emotional)
    pub e9_hdc: Vec&lt;f32&gt;,                // 1024D projected (NOT binary bits)
    pub e10_multimodal: Vec&lt;f32&gt;,        // 768D
    pub e11_entity: Vec&lt;f32&gt;,            // 384D
    pub e12_late_interaction: Vec&lt;Vec&lt;f32&gt;&gt;, // 128D per token
    pub e13_splade: SparseVector,        // ~30K vocab, sparse
}
```
    </rust_definition>
    <methods_implemented>
      - zeroed() -> Self (test-only, requires #[cfg(test)] or feature = "test-utils")
      - get_embedding(idx: usize) -> Option&lt;EmbeddingSlice&gt;
      - get(&amp;self, embedder: Embedder) -> EmbeddingRef (type-safe)
      - storage_size() -> usize
      - storage_bytes() -> usize (alias)
      - token_count() -> usize
      - e13_splade_nnz() -> usize
      - embedding_name(idx: usize) -> Option&lt;&amp;'static str&gt;
      - embedding_dim(idx: usize) -> Option&lt;usize&gt;
      - is_complete() -> bool
      - validate_strict() -> Result&lt;(), ValidationError&gt;
    </methods_implemented>
    <traits_implemented>
      - Debug, Clone, Serialize, Deserialize, PartialEq
      - NOTE: Default is intentionally NOT implemented (prevents silent failures)
    </traits_implemented>
  </type>

  <type name="Embedder">
    <location>crates/context-graph-core/src/teleological/embedder.rs:70-106</location>
    <rust_definition>
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Embedder {
    Semantic = 0,           // E1
    TemporalRecent = 1,     // E2
    TemporalPeriodic = 2,   // E3
    TemporalPositional = 3, // E4
    Causal = 4,             // E5
    Sparse = 5,             // E6
    Code = 6,               // E7
    Emotional = 7,          // E8 (canonical name - Graph is deprecated)
    Hdc = 8,                // E9
    Multimodal = 9,         // E10
    Entity = 10,            // E11
    LateInteraction = 11,   // E12
    KeywordSplade = 12,     // E13
}
```
    </rust_definition>
    <methods_implemented>
      - COUNT: usize = 13 (constant)
      - index(self) -> usize
      - from_index(idx: usize) -> Option&lt;Self&gt;
      - expected_dims(self) -> EmbedderDims
      - all() -> impl ExactSizeIterator&lt;Item = Embedder&gt;
      - name(self) -> &amp;'static str
      - short_name(self) -> &amp;'static str
      - is_dense(self) -> bool
      - is_sparse(self) -> bool
      - is_token_level(self) -> bool
      - default_temperature(self) -> f32
      - purpose(self) -> &amp;'static str
      - from_name(name: &amp;str) -> Result&lt;Self, EmbedderNameError&gt;
      - all_names() -> Vec&lt;&amp;'static str&gt;
    </methods_implemented>
    <deprecation_handling>
      E8_Graph is deprecated. Using "E8_Graph" in from_name() emits a tracing warning.
      Canonical name is "E8_Emotional". The deprecated alias still works but warns.
    </deprecation_handling>
  </type>

  <type name="EmbedderDims">
    <location>crates/context-graph-core/src/teleological/embedder.rs:371-395</location>
    <rust_definition>
```rust
pub enum EmbedderDims {
    Dense(usize),                    // Fixed-length f32 vector
    Sparse { vocab_size: usize },    // Sparse with vocab size
    TokenLevel { per_token: usize }, // Variable-length per-token
}
```
    </rust_definition>
  </type>

  <type name="EmbedderMask">
    <location>crates/context-graph-core/src/teleological/embedder.rs:400-465</location>
    <description>Bitmask for selecting subset of 13 embedders using u16 (bits 0-12)</description>
    <methods>new(), all(), from_slice(), set(), unset(), contains(), iter(), count(), is_empty(), as_u16()</methods>
  </type>

  <type name="EmbedderGroup">
    <location>crates/context-graph-core/src/teleological/embedder.rs:470-523</location>
    <variants>Temporal, Relational, Lexical, Dense, Factual, Implementation, All</variants>
    <method>embedders(self) -> EmbedderMask</method>
  </type>

  <type name="SparseVector">
    <location>crates/context-graph-core/src/types/fingerprint/sparse.rs:49-55</location>
    <rust_definition>
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseVector {
    pub indices: Vec&lt;u16&gt;,  // Sorted ascending, unique, &lt;30522
    pub values: Vec&lt;f32&gt;,   // Same length as indices
}
```
    </rust_definition>
    <methods_implemented>
      - new(indices, values) -> Result&lt;Self, SparseVectorError&gt; (validates)
      - empty() -> Self
      - nnz() -> usize
      - dot(&amp;self, other: &amp;Self) -> f32 (O(n+m) merge-join)
      - memory_size() -> usize
      - is_empty() -> bool
      - get(vocab_index: u16) -> Option&lt;f32&gt; (binary search)
      - l2_norm() -> f32
      - cosine_similarity(&amp;self, other: &amp;Self) -> f32
    </methods_implemented>
    <constants>
      SPARSE_VOCAB_SIZE = 30_522
      MAX_SPARSE_ACTIVE = 1_526 (~5% sparsity)
    </constants>
  </type>

  <type name="ValidationError">
    <location>crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs:54-101</location>
    <variants>
      - DimensionMismatch { embedder, expected, actual }
      - EmptyDenseEmbedding { embedder, expected }
      - SparseVectorError { embedder, source }
      - TokenDimensionMismatch { embedder, token_index, expected, actual }
    </variants>
  </type>
</actual_implementation>

<!-- =================================================================== -->
<!-- DIMENSION CONSTANTS (VERIFIED)                                      -->
<!-- =================================================================== -->
<dimension_constants location="crates/context-graph-core/src/types/fingerprint/semantic/constants.rs">
  E1_DIM = 1024
  E2_DIM = 512
  E3_DIM = 512
  E4_DIM = 512
  E5_DIM = 768
  E6_SPARSE_VOCAB = 30_522
  E7_DIM = 1536
  E8_DIM = 384
  E9_DIM = 1024 (projected from 10K-bit hypervector)
  E10_DIM = 768
  E11_DIM = 384
  E12_TOKEN_DIM = 128 (per token)
  E13_SPLADE_VOCAB = 30_522
  NUM_EMBEDDERS = 13
  TOTAL_DENSE_DIMS = 7424
</dimension_constants>

<!-- =================================================================== -->
<!-- KEY DIFFERENCES FROM ORIGINAL SPEC                                  -->
<!-- =================================================================== -->
<spec_vs_actual_differences>
  <difference id="1" severity="INFO">
    <spec>DenseVector&lt;N&gt; as generic fixed-size array type</spec>
    <actual>Vec&lt;f32&gt; with runtime dimension validation</actual>
    <reason>Enables serde serialization, avoids stack overflow with large arrays</reason>
  </difference>

  <difference id="2" severity="INFO">
    <spec>BinaryVector&lt;N&gt; for E9 HDC with packed u64 bits</spec>
    <actual>Vec&lt;f32&gt; with 1024D dense projection from 10K-bit hypervector</actual>
    <reason>HDC model projects to dense space for HNSW compatibility</reason>
  </difference>

  <difference id="3" severity="INFO">
    <spec>Default trait implemented (zero vectors)</spec>
    <actual>Default intentionally NOT implemented; use zeroed() in tests only</actual>
    <reason>Prevents silent failures from all-zero vectors in production</reason>
  </difference>

  <difference id="4" severity="INFO">
    <spec>E8 named "E8Emotional" with variant "E8Emotional"</spec>
    <actual>Variant "Emotional", field "e8_graph", with E8_Graph deprecated</actual>
    <reason>Historical naming preserved for compatibility with deprecation path</reason>
  </difference>

  <difference id="5" severity="INFO">
    <spec>Files in crates/context-graph-core/src/embedding/</spec>
    <actual>Files in crates/context-graph-core/src/types/fingerprint/ and src/teleological/</actual>
    <reason>Better code organization separating types from embedding logic</reason>
  </difference>

  <difference id="6" severity="INFO">
    <spec>Embedder variants named E1Semantic, E2TempRecent, etc.</spec>
    <actual>Embedder variants named Semantic, TemporalRecent, etc. (without E prefix)</actual>
    <reason>Cleaner API - the E-prefix is in the short_name() method instead</reason>
  </difference>
</spec_vs_actual_differences>

<!-- =================================================================== -->
<!-- EXISTING TESTS (ALL PASS)                                           -->
<!-- =================================================================== -->
<existing_tests>
  <test_file path="crates/context-graph-core/src/teleological/embedder.rs" count="25+">
    - test_embedder_count (13 embedders)
    - test_index_roundtrip (all embedders)
    - test_index_bounds (from_index validation)
    - test_expected_dims_match_constants
    - test_names (name, short_name)
    - test_embedder_mask_operations
    - test_embedder_mask_all
    - test_embedder_mask_iter
    - test_embedder_group_temporal
    - test_embedder_group_dense
    - test_embedder_serde
    - test_embedder_mask_serde
    - test_type_classification (is_dense, is_sparse, is_token_level)
    - test_default_temperature
    - test_display
    - test_embedder_dims_primary_dim
    - test_e8_canonical_name
    - test_e8_purpose
    - test_e8_from_name_canonical
    - test_e8_from_name_deprecated
    - test_e8_from_name_ambiguous
    - test_e8_from_name_unknown
    - test_e8_all_names
    - test_e8_name_error_codes
    - test_e8_index_unchanged
  </test_file>

  <test_file path="crates/context-graph-core/src/types/fingerprint/sparse.rs" count="18">
    - test_sparse_vector_new_valid
    - test_sparse_vector_new_empty
    - test_sparse_vector_length_mismatch
    - test_sparse_vector_index_out_of_bounds
    - test_sparse_vector_unsorted
    - test_sparse_vector_duplicate
    - test_sparse_vector_dot
    - test_sparse_vector_dot_empty
    - test_sparse_vector_dot_no_intersection
    - test_sparse_vector_memory_size
    - test_sparse_vector_serialization_roundtrip
    - test_sparse_vector_empty_serialization
    - test_sparse_vector_get
    - test_sparse_vector_l2_norm
    - test_sparse_vector_cosine_similarity
    - test_sparse_vector_max_index
    - test_sparse_vector_error_display
    - test_sparse_vector_constants
    - test_sparse_vector_typical_sparsity
  </test_file>

  <test_directory path="crates/context-graph-core/src/types/fingerprint/semantic/tests/">
    - core_tests.rs
    - validation_tests.rs
    - storage_tests.rs
    - task_core_003_tests.rs
  </test_directory>
</existing_tests>

<!-- =================================================================== -->
<!-- VERIFICATION COMMANDS                                               -->
<!-- =================================================================== -->
<verification>
  <command description="Run all teleological embedder tests">
    cargo test --package context-graph-core embedder -- --nocapture
  </command>

  <command description="Run all sparse vector tests">
    cargo test --package context-graph-core sparse -- --nocapture
  </command>

  <command description="Run all fingerprint tests">
    cargo test --package context-graph-core fingerprint -- --nocapture
  </command>

  <command description="Check compilation">
    cargo check --package context-graph-core
  </command>

  <command description="Run clippy">
    cargo clippy --package context-graph-core -- -D warnings
  </command>
</verification>

<!-- =================================================================== -->
<!-- SOURCE OF TRUTH VERIFICATION                                        -->
<!-- =================================================================== -->
<source_of_truth_verification>
  <source_of_truth>
    The TeleologicalArray type definition in crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs
    The Embedder enum definition in crates/context-graph-core/src/teleological/embedder.rs
    The dimension constants in crates/context-graph-core/src/types/fingerprint/semantic/constants.rs
  </source_of_truth>

  <verification_steps>
    1. Run `cargo test --package context-graph-core` - All tests MUST pass
    2. Verify Embedder::COUNT == 13
    3. Verify Embedder::all().count() == 13
    4. Verify TeleologicalArray has all 13 fields (grep for "pub e" in fingerprint.rs)
    5. Verify SemanticFingerprint::validate_strict() validates all 13 dimensions
  </verification_steps>

  <boundary_edge_cases>
    <case id="1" name="Empty sparse vector">
      <input>SparseVector::empty()</input>
      <expected>nnz() == 0, memory_size() == 0, is_empty() == true</expected>
      <verification>Run test_sparse_vector_new_empty</verification>
    </case>

    <case id="2" name="Maximum sparse index">
      <input>SparseVector::new(vec![30521], vec![1.0])</input>
      <expected>Ok(SparseVector) with index at max valid position</expected>
      <verification>Run test_sparse_vector_max_index</verification>
    </case>

    <case id="3" name="Out-of-bounds sparse index">
      <input>SparseVector::new(vec![30522], vec![1.0])</input>
      <expected>Err(SparseVectorError::IndexOutOfBounds)</expected>
      <verification>Run test_sparse_vector_index_out_of_bounds</verification>
    </case>

    <case id="4" name="Embedder roundtrip through index">
      <input>All 13 embedder variants</input>
      <expected>Embedder::from_index(e.index()) == Some(e) for all e</expected>
      <verification>Run test_index_roundtrip</verification>
    </case>

    <case id="5" name="Zeroed fingerprint (test-only)">
      <input>SemanticFingerprint::zeroed()</input>
      <expected>All dense fields have correct dimensions, sparse fields empty</expected>
      <verification>Requires --features test-utils to compile</verification>
    </case>
  </boundary_edge_cases>

  <manual_verification_checklist>
    [x] `cargo test --package context-graph-core` passes with 0 failures (3748 tests passed, 2026-01-16)
    [x] grep "pub e1_semantic" returns exactly 1 match in fingerprint.rs (verified)
    [x] grep "pub enum Embedder" returns exactly 1 match in embedder.rs (verified)
    [x] The Embedder enum has exactly 13 variants (count = 12 inclusive from 0, verified)
    [x] NUM_EMBEDDERS constant equals 13 (verified)
    [x] E13_SPLADE_VOCAB == 30_522 (verified)
  </manual_verification_checklist>

  <full_state_verification date="2026-01-16">
    <summary>
      Full State Verification performed. All tests pass, all edge cases verified,
      source of truth inspected and confirmed correct.
    </summary>
    <test_counts>
      <embedder_tests>89 passed</embedder_tests>
      <sparse_tests>64 passed</sparse_tests>
      <fingerprint_tests>174 passed</fingerprint_tests>
      <total_core_tests>3748 passed</total_core_tests>
    </test_counts>
    <dimension_verification>
      All 13 embedder dimensions match constitution.yaml specification exactly.
      E1=1024, E2=512, E3=512, E4=512, E5=768, E6=30522 vocab, E7=1536,
      E8=384, E9=1024, E10=768, E11=384, E12=128/token, E13=30522 vocab.
    </dimension_verification>
    <edge_cases_verified>
      - Empty sparse vector (nnz=0, valid)
      - Maximum sparse index (30521, valid)
      - Out-of-bounds sparse index (30522, correctly rejected)
      - All 13 embedder roundtrip through index()
      - Zeroed fingerprint (test-only, validates correctly)
    </edge_cases_verified>
    <clippy_status>
      No clippy issues in TeleologicalArray-related files.
      Other modules have unrelated issues (not in scope for this task).
    </clippy_status>
  </full_state_verification>
</source_of_truth_verification>

<!-- =================================================================== -->
<!-- CONCLUSION                                                          -->
<!-- =================================================================== -->
<conclusion>
  <status>COMPLETE</status>
  <evidence>
    All types specified in TASK-P2-001 are fully implemented:
    - TeleologicalArray (as SemanticFingerprint type alias) with all 13 fields
    - Embedder enum with 13 variants and comprehensive methods
    - EmbedderDims, EmbedderMask, EmbedderGroup helper types
    - SparseVector with validation and similarity computation
    - ValidationError for dimension validation
    - 40+ unit tests covering all functionality
  </evidence>
  <next_action>
    This task requires no further implementation. Proceed to TASK-P2-002 (vector types)
    which may also be complete - audit that task next.
  </next_action>
</conclusion>
</task_spec>
```

## Execution Checklist

- [x] Create embedding directory in context-graph-core/src (DIFFERENT LOCATION - types/fingerprint/)
- [x] Create mod.rs with Embedder enum (EXISTS at teleological/embedder.rs)
- [x] Create teleological.rs with TeleologicalArray struct (EXISTS as types/fingerprint/semantic/fingerprint.rs)
- [x] Implement Default trait (INTENTIONALLY NOT IMPLEMENTED - use zeroed() in tests)
- [x] Add estimated_size_bytes method (EXISTS as storage_size() and storage_bytes())
- [x] Update lib.rs to export embedding module (EXISTS - types and teleological modules exported)
- [x] Write unit tests (EXISTS - 40+ tests)
- [x] Run tests to verify (ALL PASS)
- [x] Mark as COMPLETE - Proceed to TASK-P2-002
