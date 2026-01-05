# Functional Specification: Multi-Array Teleological Fingerprint Architecture

**Document ID**: FUNC-SPEC-001
**Version**: 1.0.0
**Status**: Draft
**Date**: 2026-01-04
**Implements**: projectionplan1.md, projectionplan2.md, contextprd.md, constitution.yaml

---

## 1. Document Overview

### 1.1 Purpose

This Functional Specification defines the complete functional requirements for the Multi-Array Teleological Fingerprint Architecture. The architecture replaces the legacy FuseMoE single-vector fusion approach with a 12-embedding array system that preserves 100% of semantic information and enables teleological (purpose-driven) memory operations.

### 1.2 Core Paradigm

**"NO FUSION - Store all 12 embeddings. The array IS the teleological vector."**

The pattern across embedding spaces reveals purpose. Each memory is represented by:
- `SemanticFingerprint`: 12-array of embeddings (E1-E12)
- `TeleologicalFingerprint`: SemanticFingerprint + PurposeVector + JohariFingerprint + PurposeEvolution

### 1.3 Document Scope

This specification covers:
- SemanticFingerprint structure and storage requirements
- TeleologicalFingerprint with purpose vector and Johari awareness
- Storage layer architecture (Primary, Per-Space Indexes, Goal Hierarchy)
- Query routing and weighted similarity computation
- Meta-UTL self-aware learning requirements
- Complete removal of legacy fusion components

### 1.4 Source Traceability

| Source Document | Reference Sections |
|-----------------|-------------------|
| projectionplan1.md | Sections 10, 11.1-11.9, 12, 13, 14, 15 |
| projectionplan2.md | Sections 11.5-11.9, 12, 13, 14, 15 |
| contextprd.md | Sections 2, 3, 4, 14, 18, 19 |
| constitution.yaml | embeddings, storage, teleological, meta_utl |

---

## 2. Scope and Boundaries

### 2.1 In Scope

- Complete 12-embedding array architecture
- Per-embedder dimension and storage requirements
- TeleologicalFingerprint with PurposeVector and JohariFingerprint
- 3-layer storage architecture (Primary + Indexes + Temporal)
- Query routing and weighted similarity
- Meta-UTL self-aware learning system
- Complete removal of 36 fusion-related files

### 2.2 Out of Scope

- Implementation details (covered in Technical Specification)
- CUDA kernel optimization (separate spec)
- MCP handler modifications (separate spec)
- Migration tooling (separate spec)

### 2.3 Constraints

| Constraint | Source | Description |
|------------|--------|-------------|
| No Fusion | constitution.yaml | NO FuseMoE, NO gating, NO single-vector fusion |
| No Backwards Compat | projectionplan1.md | Clean break from legacy architecture |
| Fail Fast | projectionplan1.md | Robust error logging, no silent failures |
| No Mock Data | constitution.yaml | Tests use fixtures, never inline stubs |
| Performance | constitution.yaml | inject_context P95 < 25ms, P99 < 50ms |

---

## 3. Functional Requirements

### 3.1 SemanticFingerprint Requirements (FR-100 Series)

The SemanticFingerprint stores all 12 embedding vectors without fusion, preserving 100% of semantic information.

#### FR-101: 12-Embedding Array Storage

**Requirement**: The SemanticFingerprint SHALL store exactly 12 embedding vectors as an array, NOT fused into a single vector.

**Rationale**: Legacy FuseMoE with top-k=4 fusion loses 67% of information. Storing all 12 preserves complete semantic representation.

**Source**: projectionplan1.md Section 10.2, constitution.yaml embeddings.paradigm

**Acceptance Criteria**:
- AC-101.1: SemanticFingerprint contains array of exactly 12 embedding vectors
- AC-101.2: No gating mechanism or fusion operation exists
- AC-101.3: Each embedder output stored independently

---

#### FR-102: Per-Embedder Dimension Requirements

**Requirement**: Each embedder SHALL produce embeddings of the specified dimension:

| ID | Embedder | Dimension | Storage Estimate |
|----|----------|-----------|------------------|
| E1 | Semantic | 1024D | 4KB |
| E2 | Temporal-Recent | 512D | 2KB |
| E3 | Temporal-Periodic | 512D | 2KB |
| E4 | Temporal-Positional | 512D | 2KB |
| E5 | Causal | 768D | 3KB |
| E6 | Sparse | ~30K (5% active) | ~6KB |
| E7 | Code | 1536D | 6KB |
| E8 | Graph/GNN | 384D | 1.5KB |
| E9 | HDC | 10K-bit -> 1024D | 4KB |
| E10 | Multimodal | 768D | 3KB |
| E11 | Entity/TransE | 384D | 1.5KB |
| E12 | Late-Interaction | 128D/token | Variable |

**Rationale**: Dimension requirements derived from optimal performance of each embedding model type.

**Source**: contextprd.md Section 3, constitution.yaml embeddings.models

**Acceptance Criteria**:
- AC-102.1: E1 produces exactly 1024-dimensional dense vectors
- AC-102.2: E2-E4 produce exactly 512-dimensional temporal vectors
- AC-102.3: E5 produces 768-dimensional causal vectors with direction indicator
- AC-102.4: E6 stores sparse indices and values (5% active from ~30K vocabulary)
- AC-102.5: E7 produces exactly 1536-dimensional code vectors
- AC-102.6: E8 produces exactly 384-dimensional graph vectors
- AC-102.7: E9 produces 1024-dimensional vectors from 10K-bit hyperdimensional encoding
- AC-102.8: E10 produces exactly 768-dimensional multimodal vectors
- AC-102.9: E11 produces exactly 384-dimensional entity vectors
- AC-102.10: E12 produces 128-dimensional vectors per token (ColBERT-style)

---

#### FR-103: Total Storage per Fingerprint

**Requirement**: The complete SemanticFingerprint SHALL require approximately 46KB of storage per memory.

**Rationale**: This is ~7.5x larger than legacy 6KB fused vectors but preserves 100% information (vs 33% with fusion).

**Source**: projectionplan1.md Section 10.2, constitution.yaml embeddings.storage_per_memory

**Acceptance Criteria**:
- AC-103.1: Dense embeddings (E1-E5, E7-E11) stored as contiguous byte arrays
- AC-103.2: Sparse embedding (E6) stored as index/value pairs
- AC-103.3: Late-interaction (E12) stores token count and per-token vectors
- AC-103.4: Total storage falls within 40-50KB range per fingerprint

---

#### FR-104: No Fusion, Gating, or MoE

**Requirement**: The system SHALL NOT implement any form of:
- FuseMoE (Mixture of Experts fusion)
- Gating mechanisms for embedding selection
- Single-vector fusion from multiple embeddings
- Top-k expert selection

**Rationale**: Fusion loses information. The complete 12-array IS the representation.

**Source**: projectionplan1.md Section 15.1, constitution.yaml embeddings.paradigm

**Acceptance Criteria**:
- AC-104.1: No FuseMoE struct, trait, or implementation exists
- AC-104.2: No GatingNetwork or ExpertSelector exists
- AC-104.3: No fusion() method that combines embeddings into single vector
- AC-104.4: Similarity uses weighted per-space computation, not fused vectors

---

### 3.2 TeleologicalFingerprint Requirements (FR-200 Series)

The TeleologicalFingerprint wraps SemanticFingerprint with purpose-aware metadata enabling goal-aligned retrieval.

#### FR-201: PurposeVector (12D Alignment Signature)

**Requirement**: The TeleologicalFingerprint SHALL include a 12-dimensional PurposeVector representing alignment to North Star goal per embedding space.

**Formula**: `PV = [A(E1,V), A(E2,V), ..., A(E12,V)]`

Where `A(Eᵢ,V) = cos(Eᵢ, V)` is alignment of embedder i output to North Star goal V.

**Rationale**: The purpose vector enables retrieval by teleological signature rather than just content similarity.

**Source**: projectionplan2.md Section 13.2, constitution.yaml teleological.purpose_vector

**Acceptance Criteria**:
- AC-201.1: PurposeVector is f32[12] array
- AC-201.2: Each dimension represents cosine similarity to North Star in that space
- AC-201.3: PurposeVector is searchable via 12D HNSW index
- AC-201.4: Computation includes dominant_embedder (1-12) indicating primary purpose space

---

#### FR-202: North Star Alignment Thresholds

**Requirement**: The system SHALL enforce empirically validated alignment thresholds:

| Threshold | Range | Interpretation |
|-----------|-------|----------------|
| Optimal | theta >= 0.75 | Strongly aligned to purpose |
| Acceptable | theta in [0.70, 0.75) | Adequately aligned |
| Warning | theta in [0.55, 0.70) | Alignment degrading |
| Critical | theta < 0.55 | Misalignment requiring attention |
| Failure Prediction | delta_A < -0.15 | Predicts failure 30-60s ahead |

**Rationale**: Thresholds based on Royse 2026 teleological vector research.

**Source**: constitution.yaml teleological.thresholds, contextprd.md Section 4.5

**Acceptance Criteria**:
- AC-202.1: System computes north_star_alignment as aggregate of PurposeVector
- AC-202.2: Alignment classified into optimal/acceptable/warning/critical categories
- AC-202.3: Alignment delta tracking detects failure prediction threshold
- AC-202.4: Alerts generated when alignment drops below warning threshold

---

#### FR-203: JohariFingerprint Per Embedder

**Requirement**: Each TeleologicalFingerprint SHALL include per-embedder Johari Window classification:

| Quadrant | Entropy (delta_S) | Coherence (delta_C) | Meaning |
|----------|-------------------|---------------------|---------|
| Open | Low (<0.5) | High (>0.5) | Aware in this space |
| Blind | High (>0.5) | Low (<0.5) | Discovery opportunity |
| Hidden | Low (<0.5) | Low (<0.5) | Latent in this space |
| Unknown | High (>0.5) | High (>0.5) | Frontier in this space |

**Rationale**: Cross-space Johari analysis enables targeted learning. A memory can be Open(semantic) but Blind(causal).

**Source**: projectionplan2.md Section 11.6, constitution.yaml utl.johari

**Acceptance Criteria**:
- AC-203.1: JohariFingerprint stores quadrant classification for all 12 spaces
- AC-203.2: Confidence score (0.0-1.0) stored per classification
- AC-203.3: Transition probability matrix stored for evolution prediction
- AC-203.4: Bitmap index enables queries by quadrant per embedder

---

#### FR-204: Purpose Evolution Time-Series Tracking

**Requirement**: The system SHALL track how teleological alignment changes over time for each memory.

**Tracked Data**:
- Timestamp of measurement
- PurposeVector at that time
- JohariFingerprint at that time
- Trigger event (Created, Accessed, GoalChanged, Recalibration, MisalignmentDetected)

**Rationale**: Purpose drift detection enables proactive intervention before misalignment causes failures.

**Source**: projectionplan2.md Section 11.7, constitution.yaml storage.temporal

**Acceptance Criteria**:
- AC-204.1: PurposeSnapshot recorded on each significant event
- AC-204.2: TimescaleDB hypertable stores evolution time-series
- AC-204.3: 90-day retention for continuous data, daily samples thereafter
- AC-204.4: Drift detection when delta_alignment exceeds -0.15 per space

---

### 3.3 Storage Requirements (FR-300 Series)

The storage architecture uses 3 layers for efficient teleological operations.

#### FR-301: Primary Storage in RocksDB with 12-Array Serialization

**Requirement**: The primary storage layer SHALL store complete TeleologicalFingerprint per memory in RocksDB (dev) or ScyllaDB (prod).

**Stored Fields**:
- id: UUID (primary key)
- embeddings: [E1..E12 as BYTEA]
- purpose_vector: REAL[12]
- johari_quadrants: BYTEA (12 classifications, 2 bits each)
- johari_confidence: REAL[12]
- north_star_alignment: REAL
- dominant_embedder: INT (1-12)
- coherence_score: REAL
- Metadata: created_at, last_accessed, access_count, source_type

**Rationale**: Complete fingerprint storage enables any query pattern without data loss.

**Source**: constitution.yaml storage.layer1_primary, projectionplan2.md Section 11.8

**Acceptance Criteria**:
- AC-301.1: All 12 embeddings stored as serialized byte arrays
- AC-301.2: Sparse embedding (E6) stored as index/value arrays
- AC-301.3: E5 causal includes direction indicator (1=cause, 2=effect, 3=bidirectional)
- AC-301.4: E12 stores token_count and per-token vectors
- AC-301.5: UUID primary key with content_hash for deduplication

---

#### FR-302: Per-Space HNSW Indexes (12 Separate Indexes)

**Requirement**: Layer 2A SHALL maintain 12 separate HNSW indexes, one per embedding space.

**Index Configuration**:
- M = 16 (connections per node)
- ef_construction = 200
- ef_search = 100
- Distance: cosine similarity

**Rationale**: Per-space indexes enable targeted queries ("find causally similar" uses only E5 index).

**Source**: constitution.yaml storage.layer2a_per_embedder, projectionplan2.md Section 11.3

**Acceptance Criteria**:
- AC-302.1: Separate HNSW index exists for each of E1-E12
- AC-302.2: Each index uses appropriate dimension for its embedder
- AC-302.3: E6 sparse uses inverted index, not HNSW
- AC-302.4: Query router selects appropriate index based on query type

---

#### FR-303: Goal Hierarchy Index (Layer 2C)

**Requirement**: The Goal Hierarchy Index SHALL enable navigation through North Star -> Mid -> Local goal alignments.

**Structure**:
- Tree index linking goals hierarchically
- Per-memory alignment scores cached for each goal
- Inverted index: goal_id -> aligned memory_ids (sorted by alignment)

**Rationale**: Enables queries like "find all memories aligned with goal V_mid[2]".

**Source**: projectionplan2.md Section 11.5, constitution.yaml storage.layer2c_goals

**Acceptance Criteria**:
- AC-303.1: Goal hierarchy stored with level (0=north_star, 1=mid, 2=local)
- AC-303.2: Memory-to-goal alignment cache with per_space_alignment[12]
- AC-303.3: Transitive alignment computed via bound: 2*theta1*theta2 - 1
- AC-303.4: Goal embedding (1536D) for goal similarity search

---

#### FR-304: Storage Size Increase (~46KB per Node)

**Requirement**: The system SHALL accommodate increased storage requirements:
- Previous: ~6KB per node (fused 1536D vector)
- New: ~46KB per node (12-array fingerprint)
- Ratio: ~7.5x increase

**Rationale**: Information preservation requires proportional storage increase.

**Source**: projectionplan1.md Section 10.2, constitution.yaml embeddings.storage_per_memory

**Acceptance Criteria**:
- AC-304.1: Storage estimation functions updated for 46KB baseline
- AC-304.2: Memory budgets account for 7.5x increase
- AC-304.3: GPU memory allocation adjusted (target <24GB with 8GB headroom)
- AC-304.4: Graph capacity maintained at >10M nodes

---

### 3.4 Query Requirements (FR-400 Series)

Query operations use weighted per-space similarity with configurable weights.

#### FR-401: Weighted Similarity Across 12 Embedders

**Requirement**: Similarity computation SHALL use weighted per-space cosine with query-adaptive weights.

**Formula**: `S(A,B) = sum_i(w_i * cos(A_i, B_i))` where `w_i = f(query_type, tau_i)`

**Rationale**: Query type determines which spaces are most relevant. Semantic search weights E1 heavily; causal reasoning weights E5.

**Source**: constitution.yaml embeddings.similarity, contextprd.md Section 3

**Acceptance Criteria**:
- AC-401.1: Similarity computed as weighted sum of per-space cosines
- AC-401.2: Weights normalize to sum = 1.0
- AC-401.3: Per-space similarities available for analysis
- AC-401.4: Teleological similarity includes purpose alignment factor

---

#### FR-402: Per-Embedder Weight Configuration

**Requirement**: Weight profiles SHALL be configured per query type:

| Query Type | E1(Sem) | E5(Causal) | E7(Code) | E11(Entity) | Others |
|------------|---------|------------|----------|-------------|--------|
| semantic_search | 0.40 | 0.15 | 0.10 | 0.15 | balanced |
| causal_reasoning | 0.20 | 0.50 | 0.05 | 0.15 | reduced |
| code_search | 0.20 | 0.05 | 0.50 | 0.10 | reduced |
| temporal_navigation | 0.20 | 0.05 | 0.05 | 0.10 | E2-E4: 0.60 |
| fact_checking | 0.15 | 0.25 | 0.05 | 0.50 | reduced |

**Rationale**: Query-type-specific weights optimize relevance for different use cases.

**Source**: constitution.yaml embeddings.similarity.query_types

**Acceptance Criteria**:
- AC-402.1: Weight profiles stored in configuration
- AC-402.2: Query router selects appropriate profile by query type
- AC-402.3: Custom weight overrides supported per query
- AC-402.4: Weights validated to sum to 1.0

---

#### FR-403: Configurable Distance Metrics Per Space

**Requirement**: Each embedding space MAY use appropriate distance metric:
- Dense spaces (E1-E5, E7-E11): Cosine similarity
- Sparse space (E6): Inverted index with sparse dot product
- Late-interaction (E12): MaxSim aggregation

**Rationale**: Different embedding types require different comparison methods.

**Source**: projectionplan1.md Section 11.2

**Acceptance Criteria**:
- AC-403.1: Distance metric configurable per embedder
- AC-403.2: E6 uses sparse-aware similarity
- AC-403.3: E12 uses ColBERT MaxSim over token vectors
- AC-403.4: Unified interface abstracts metric differences

---

### 3.5 Meta-UTL Requirements (FR-500 Series)

Meta-UTL enables the system to learn about its own learning.

#### FR-501: Self-Aware Learning Monitoring

**Requirement**: Meta-UTL SHALL predict and track learning system performance:
- Predict storage impact before committing
- Predict retrieval quality before executing
- Self-adjust UTL parameters based on outcome accuracy

**Rationale**: Self-awareness enables proactive optimization rather than reactive debugging.

**Source**: constitution.yaml meta_utl, contextprd.md Section 19

**Acceptance Criteria**:
- AC-501.1: Storage impact predictor with >0.85 accuracy
- AC-501.2: Retrieval quality predictor with >0.80 accuracy
- AC-501.3: Parameter adjustment when prediction_error > 0.2
- AC-501.4: Escalation when accuracy < 0.7 for 100 operations

---

#### FR-502: Learning Trajectory Tracking

**Requirement**: The system SHALL track learning trajectory per embedding space:
- Which spaces are most/least predictive
- Space weight adjustment based on accuracy
- Per-space alignment threshold tuning

**Rationale**: Not all embedding spaces contribute equally; tracking enables optimization.

**Source**: constitution.yaml meta_utl.per_space_meta

**Acceptance Criteria**:
- AC-502.1: Per-space prediction accuracy tracked
- AC-502.2: Space weights adjusted based on accuracy history
- AC-502.3: Alignment thresholds tunable per space empirically
- AC-502.4: Meta-learning events logged for analysis

---

#### FR-503: System Health Metrics

**Requirement**: Meta-UTL SHALL expose system health metrics:
- Overall learning score (UTL avg > 0.6)
- Coherence recovery time (<10s)
- Attack detection rate (>95%)
- False positive rate (<2%)

**Rationale**: Health metrics enable monitoring and alerting.

**Source**: constitution.yaml perf.quality, meta_utl.correction

**Acceptance Criteria**:
- AC-503.1: Learning score computed as sigmoid(2.0 * weighted_deltas)
- AC-503.2: Coherence recovery tracked with 10s target
- AC-503.3: Attack detection logged with confidence scores
- AC-503.4: False positive tracking enables threshold tuning

---

### 3.6 Removal Requirements (FR-600 Series)

Complete removal of legacy fusion components with no backwards compatibility.

#### FR-601: Complete Removal of 36 Fusion Files

**Requirement**: The following file patterns SHALL be completely removed:

**Core Fusion (12 files)**:
- `src/fusion/mod.rs`
- `src/fusion/fuse_moe.rs`
- `src/fusion/gating.rs`
- `src/fusion/expert_selector.rs`
- `src/fusion/fusion_config.rs`
- `src/embeddings/fused_embedding.rs`
- `src/embeddings/fusion_pipeline.rs`
- `src/storage/fused_vector_store.rs`
- `src/search/fused_similarity.rs`
- `src/mcp/handlers/fused_search.rs`
- `tests/fusion_tests.rs`
- `benches/fusion_bench.rs`

**Supporting Fusion (12 files)**:
- Any file with `fuse`, `fusion`, `gating`, `expert_select` in name
- `src/embeddings/vector_1536.rs` (legacy single-vector type)
- `src/types/fused_types.rs`
- Integration with removed components

**Configuration (6 files)**:
- Fusion-related TOML/YAML sections
- Legacy embedding pipeline configs
- Gating network weights

**Tests (6 files)**:
- Unit tests for fusion
- Integration tests for fusion
- Benchmarks for fusion

**Rationale**: Clean break required for architectural integrity.

**Source**: projectionplan1.md Section 15.1

**Acceptance Criteria**:
- AC-601.1: No file with "fuse", "fusion", "gating" exists post-removal
- AC-601.2: No imports of removed modules in remaining code
- AC-601.3: Cargo.toml dependencies for fusion crates removed
- AC-601.4: All fusion-related tests deleted

---

#### FR-602: No Backwards Compatibility

**Requirement**: The system SHALL NOT provide migration paths or compatibility layers for:
- Legacy Vector1536 type
- FuseMoE API
- Gating network interfaces
- Single-vector similarity functions

**Rationale**: Compatibility layers add complexity and encourage legacy usage.

**Source**: projectionplan1.md Section 15

**Acceptance Criteria**:
- AC-602.1: No From/Into implementations between legacy and new types
- AC-602.2: No deprecated attributes on fusion-related items
- AC-602.3: No compatibility shims or adapters
- AC-602.4: Documentation explicitly states no migration path

---

#### FR-603: Fail Fast with Robust Error Logging

**Requirement**: The system SHALL fail immediately and loudly when:
- Fusion-related code paths are invoked
- Legacy data formats are encountered
- Invalid embedding dimensions received

**Rationale**: Silent failures mask architectural violations.

**Source**: constitution.yaml forbidden.AP-001, AP-009

**Acceptance Criteria**:
- AC-603.1: Invalid embedding dimension triggers immediate error
- AC-603.2: Legacy format detection logs error with stack trace
- AC-603.3: All errors include context for debugging
- AC-603.4: No unwrap() in production code; use expect() with message

---

#### FR-604: No Mock Data in Tests

**Requirement**: All tests SHALL use proper fixtures, NOT inline mock data:
- Test fixtures in `tests/fixtures/`
- Realistic embedding dimensions
- Valid fingerprint structures

**Rationale**: Mock data can mask dimensional errors; fixtures enforce correctness.

**Source**: constitution.yaml forbidden.AP-007

**Acceptance Criteria**:
- AC-604.1: Test fixtures exist for all 12 embedder types
- AC-604.2: Fixtures use correct dimensions per embedder
- AC-604.3: No inline vector construction in tests
- AC-604.4: Fixture validation on load

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements

| Metric | Target | Source |
|--------|--------|--------|
| Single Embed (all 12) | <30ms | contextprd.md |
| Batch Embed (64 x 12) | <100ms | contextprd.md |
| Per-Space HNSW search | <2ms | contextprd.md |
| Purpose Vector search | <1ms | contextprd.md |
| Multi-space weighted sim | <5ms | contextprd.md |
| inject_context P95 | <25ms | constitution.yaml |
| inject_context P99 | <50ms | constitution.yaml |
| Purpose evolution write | <5ms | contextprd.md |
| Teleological alignment | <1ms | contextprd.md |

### 4.2 Quality Requirements

| Metric | Target | Source |
|--------|--------|--------|
| Unit coverage | >= 90% | constitution.yaml |
| Integration coverage | >= 80% | constitution.yaml |
| UTL avg | > 0.6 | constitution.yaml |
| Coherence recovery | < 10s | constitution.yaml |
| Attack detection | > 95% | constitution.yaml |
| False positive | < 2% | constitution.yaml |
| Info loss | < 15% | constitution.yaml |

### 4.3 Scalability Requirements

| Metric | Target | Source |
|--------|--------|--------|
| Graph capacity | > 10M nodes | constitution.yaml |
| GPU memory | < 24GB (8GB headroom) | constitution.yaml |
| Embed throughput | > 1000/sec | constitution.yaml |
| Search batch 100 | < 5ms | constitution.yaml |

---

## 5. Constraints

### 5.1 Technical Constraints

| ID | Constraint | Source |
|----|------------|--------|
| TC-01 | Rust 1.75+, Edition 2021 | constitution.yaml |
| TC-02 | CUDA 13.1 for GPU operations | constitution.yaml |
| TC-03 | RTX 5090 target GPU (32GB VRAM) | constitution.yaml |
| TC-04 | RocksDB (dev) / ScyllaDB (prod) primary storage | constitution.yaml |
| TC-05 | TimescaleDB for temporal purpose evolution | constitution.yaml |
| TC-06 | One primary type per module, max 500 lines | constitution.yaml |

### 5.2 Security Constraints

| ID | Constraint | Source |
|----|------------|--------|
| SEC-01 | Validate/sanitize all input | constitution.yaml |
| SEC-02 | Scrub PII pre-embed | constitution.yaml |
| SEC-03 | Anomaly threshold 3.0 std | constitution.yaml |
| SEC-04 | Detect prompt injection | constitution.yaml |
| SEC-05 | Quarantine semantic cancer | constitution.yaml |

### 5.3 Architectural Constraints

| ID | Constraint | Source |
|----|------------|--------|
| AC-01 | NO FuseMoE or fusion operations | constitution.yaml |
| AC-02 | NO backwards compatibility with legacy types | projectionplan1.md |
| AC-03 | NO single-vector similarity functions | constitution.yaml |
| AC-04 | Lock order: inner -> faiss_index | constitution.yaml |
| AC-05 | Max 5 unsafe blocks per module | constitution.yaml |

---

## 6. Traceability Matrix

### 6.1 PRD to Functional Requirements

| PRD Section | FR ID | Requirement |
|-------------|-------|-------------|
| contextprd.md 3.0 | FR-101 | 12-Embedding Array Storage |
| contextprd.md 3.0 | FR-102 | Per-Embedder Dimensions |
| constitution.yaml embeddings | FR-103 | 46KB Storage per Fingerprint |
| constitution.yaml embeddings.paradigm | FR-104 | No Fusion/Gating/MoE |
| constitution.yaml teleological | FR-201 | PurposeVector 12D |
| constitution.yaml teleological.thresholds | FR-202 | North Star Alignment |
| constitution.yaml utl.johari | FR-203 | JohariFingerprint |
| constitution.yaml storage.temporal | FR-204 | Purpose Evolution |
| constitution.yaml storage.layer1 | FR-301 | Primary RocksDB Storage |
| constitution.yaml storage.layer2a | FR-302 | 12 Per-Space HNSW |
| constitution.yaml storage.layer2c | FR-303 | Goal Hierarchy Index |
| constitution.yaml perf.memory | FR-304 | Storage Size Increase |
| constitution.yaml embeddings.similarity | FR-401 | Weighted Similarity |
| constitution.yaml embeddings.query_types | FR-402 | Weight Configuration |
| projectionplan1.md 11.2 | FR-403 | Distance Metrics |
| constitution.yaml meta_utl | FR-501 | Self-Aware Learning |
| constitution.yaml meta_utl.per_space | FR-502 | Learning Trajectory |
| constitution.yaml meta_utl.correction | FR-503 | System Health |
| projectionplan1.md 15.1 | FR-601 | Remove 36 Files |
| projectionplan1.md 15.1 | FR-602 | No Backwards Compat |
| constitution.yaml forbidden | FR-603 | Fail Fast |
| constitution.yaml forbidden.AP-007 | FR-604 | No Mock Data |

### 6.2 Requirements Coverage Summary

| Series | Count | Coverage |
|--------|-------|----------|
| FR-100 (SemanticFingerprint) | 4 | Complete |
| FR-200 (TeleologicalFingerprint) | 4 | Complete |
| FR-300 (Storage) | 4 | Complete |
| FR-400 (Query) | 3 | Complete |
| FR-500 (Meta-UTL) | 3 | Complete |
| FR-600 (Removal) | 4 | Complete |
| **Total** | **22** | **100%** |

---

## 7. Acceptance Criteria

### 7.1 System Acceptance Criteria

1. **SAC-01**: All 22 functional requirements pass verification
2. **SAC-02**: No fusion-related code exists in codebase
3. **SAC-03**: 12-embedding array stores all embedder outputs
4. **SAC-04**: TeleologicalFingerprint includes PurposeVector, Johari, Evolution
5. **SAC-05**: 12 HNSW indexes + 1 Purpose index + Goal hierarchy index exist
6. **SAC-06**: Query router selects appropriate indexes
7. **SAC-07**: Meta-UTL tracks learning trajectory
8. **SAC-08**: All performance targets met

### 7.2 Verification Methods

| Criterion | Method |
|-----------|--------|
| SAC-01 | Automated test suite |
| SAC-02 | Code search for fusion patterns |
| SAC-03 | Unit tests on SemanticFingerprint |
| SAC-04 | Unit tests on TeleologicalFingerprint |
| SAC-05 | Integration tests on storage layer |
| SAC-06 | Integration tests on query router |
| SAC-07 | Integration tests on Meta-UTL |
| SAC-08 | Benchmark suite |

### 7.3 Definition of Done

The Multi-Array Teleological Fingerprint Architecture is complete when:

1. All FR-100 through FR-600 requirements implemented
2. All acceptance criteria (AC-*) verified
3. Unit test coverage >= 90%
4. Integration test coverage >= 80%
5. Performance benchmarks pass all targets
6. Zero fusion-related code in repository
7. Documentation updated to reflect new architecture

---

## Appendix A: Embedder Purpose Mapping

| ID | Embedder | Teleological Purpose | Measurement |
|----|----------|---------------------|-------------|
| E1 | Semantic | V_meaning | A(content, V_meaning) |
| E2 | Temporal-Recent | V_freshness | A(timestamp, V_freshness) |
| E3 | Temporal-Periodic | V_periodicity | A(pattern, V_periodicity) |
| E4 | Temporal-Positional | V_ordering | A(position, V_ordering) |
| E5 | Causal | V_causality | A(causation, V_causality) |
| E6 | Sparse | V_selectivity | A(activations, V_selectivity) |
| E7 | Code | V_correctness | A(ast, V_correctness) |
| E8 | Graph | V_connectivity | A(structure, V_connectivity) |
| E9 | HDC | V_robustness | A(hologram, V_robustness) |
| E10 | Multimodal | V_multimodality | A(grounding, V_multimodality) |
| E11 | Entity | V_factuality | A(triple, V_factuality) |
| E12 | Late-Interaction | V_precision | A(tokens, V_precision) |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **SemanticFingerprint** | 12-array of embeddings preserving all semantic information |
| **TeleologicalFingerprint** | SemanticFingerprint + PurposeVector + Johari + Evolution |
| **PurposeVector** | 12D signature of alignment to North Star per embedding space |
| **JohariFingerprint** | Per-embedder awareness classification (Open/Hidden/Blind/Unknown) |
| **North Star** | System-wide teleological goal guiding alignment |
| **FuseMoE** | DEPRECATED - Mixture of Experts fusion (removed) |
| **Meta-UTL** | Self-aware learning system that learns about its own learning |
| **UTL** | Unified Theory of Learning - L = f((delta_S x delta_C) * w_e * cos phi) |

---

**END OF FUNCTIONAL SPECIFICATION**
