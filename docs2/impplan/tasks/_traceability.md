# Task Traceability Matrix

## Coverage: Functional Specs → Technical Specs → Tasks

This matrix ensures every requirement from the functional specifications is covered by at least one task.

---

## Phase 0: North Star Removal

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-P0-01: Delete North Star files | requirement | files_to_delete | TASK-P0-001 | ✓ |
| REQ-P0-02: Remove MCP tools | requirement | mcp_tools_to_remove | TASK-P0-002 | ✓ |
| REQ-P0-03: Update constitution | requirement | constitution_changes | TASK-P0-003 | ✓ |
| REQ-P0-04: Drop database tables | requirement | database_changes | TASK-P0-004 | ✓ |
| REQ-P0-05: Clean up imports | requirement | import_cleanup | TASK-P0-001 | ✓ |
| EC-P0-01: No compilation errors | edge_case | verification_protocol | TASK-P0-005 | ✓ |
| EC-P0-02: Tool count 59→53 | edge_case | verification_protocol | TASK-P0-005 | ✓ |

---

## Phase 1: Memory Capture

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-P1-01: Hook description capture | requirement | capture_hook_description | TASK-P1-007 | ✓ |
| REQ-P1-02: Claude response capture | requirement | capture_claude_response | TASK-P1-007 | ✓ |
| REQ-P1-03: MD file chunking | requirement | chunk_text | TASK-P1-004 | ✓ |
| REQ-P1-04: 200-word chunks, 50 overlap | requirement | TextChunker.constants | TASK-P1-004 | ✓ |
| REQ-P1-05: File watcher | requirement | MDFileWatcher | TASK-P1-008 | ✓ |
| REQ-P1-06: Session tracking | requirement | SessionManager | TASK-P1-006 | ✓ |
| Model: Memory | data_model | Memory struct | TASK-P1-001 | ✓ |
| Model: MemorySource | data_model | MemorySource enum | TASK-P1-001 | ✓ |
| Model: ChunkMetadata | data_model | ChunkMetadata struct | TASK-P1-002 | ✓ |
| Model: TextChunk | data_model | TextChunk struct | TASK-P1-002 | ✓ |
| Model: Session | data_model | Session struct | TASK-P1-003 | ✓ |
| Model: SessionStatus | data_model | SessionStatus enum | TASK-P1-003 | ✓ |
| Component: TextChunker | component | TextChunker | TASK-P1-004 | ✓ |
| Component: MemoryStore | component | MemoryStore | TASK-P1-005 | ✓ |
| Component: SessionManager | component | SessionManager | TASK-P1-006 | ✓ |
| Component: MemoryCaptureService | component | MemoryCaptureService | TASK-P1-007 | ✓ |
| Component: MDFileWatcher | component | MDFileWatcher | TASK-P1-008 | ✓ |
| Error: ChunkerError | error | ChunkerError enum | TASK-P1-004 | ✓ |
| Error: CaptureError | error | CaptureError enum | TASK-P1-007 | ✓ |
| Error: WatcherError | error | WatcherError enum | TASK-P1-008 | ✓ |
| Error: StorageError | error | StorageError enum | TASK-P1-005 | ✓ |

---

## Phase 2: 13-Space Embedding

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-P2-01: Atomic 13-embedding | requirement | embed_all | TASK-P2-005 | ✓ |
| REQ-P2-02: Per-embedder distance | requirement | get_distance_metric | TASK-P2-003 | ✓ |
| REQ-P2-03: Quantization <20KB | requirement | quantize_array | TASK-P2-006 | ✓ |
| REQ-P2-04: Dimension validation | requirement | validate_teleological_array | TASK-P2-004 | ✓ |
| REQ-P2-05: Config registry | requirement | EmbedderConfigRegistry | TASK-P2-003 | ✓ |
| REQ-P2-06: Embedder category classification | requirement | EmbedderCategory enum | TASK-P2-003b | ✓ |
| REQ-P2-07: Category topic_weight method | requirement | topic_weight() | TASK-P2-003b | ✓ |
| Model: TeleologicalArray | data_model | TeleologicalArray | TASK-P2-001 | ✓ |
| Model: DenseVector | data_model | DenseVector<N> | TASK-P2-002 | ✓ |
| Model: SparseVector | data_model | SparseVector | TASK-P2-002 | ✓ |
| Model: BinaryVector | data_model | BinaryVector<N> | TASK-P2-002 | ✓ |
| Model: EmbedderCategory | data_model | EmbedderCategory enum | TASK-P2-003b | ✓ |
| Model: EmbedderConfig | data_model | EmbedderConfig (with category field) | TASK-P2-003 | ✓ |
| Model: DistanceMetric | data_model | DistanceMetric enum | TASK-P2-003 | ✓ |
| Model: QuantizationConfig | data_model | QuantizationConfig | TASK-P2-003 | ✓ |
| Model: Embedder | data_model | Embedder enum | TASK-P2-001 | ✓ |
| Method: topic_weight() | method | EmbedderCategory | TASK-P2-003b | ✓ |
| Method: is_semantic() | method | EmbedderCategory | TASK-P2-003b | ✓ |
| Method: is_temporal() | method | EmbedderCategory | TASK-P2-003b | ✓ |
| Component: MultiArrayProvider | component | MultiArrayProvider | TASK-P2-005 | ✓ |
| Component: DimensionValidator | component | DimensionValidator | TASK-P2-004 | ✓ |
| Component: Quantizer | component | Quantizer | TASK-P2-006 | ✓ |
| Error: EmbedderError | error | EmbedderError enum | TASK-P2-005 | ✓ |
| Error: ValidationError | error | ValidationError enum | TASK-P2-004 | ✓ |

---

## Phase 3: Similarity & Divergence

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-P3-01: Per-space comparison | requirement | compute_similarity | TASK-P3-005 | ✓ |
| REQ-P3-02: ANY() relevance logic | requirement | is_relevant | TASK-P3-005 | ✓ |
| REQ-P3-03: Weighted relevance score (category-weighted) | requirement | compute_relevance_score | TASK-P3-005 | ✓ |
| REQ-P3-04: Divergence detection (semantic only) | requirement | detect_divergence | TASK-P3-006 | ✓ |
| REQ-P3-05: Multi-space retrieval | requirement | retrieve_similar | TASK-P3-007 | ✓ |
| REQ-P3-06: Exclude temporal from weighted calculations | requirement | SPACE_WEIGHTS | TASK-P3-005 | ✓ |
| REQ-P3-07: Only check semantic spaces for divergence | requirement | DIVERGENCE_SPACES | TASK-P3-006 | ✓ |
| Model: SimilarityResult | data_model | SimilarityResult | TASK-P3-001 | ✓ |
| Model: PerSpaceScores | data_model | PerSpaceScores | TASK-P3-001 | ✓ |
| Model: DivergenceAlert | data_model | DivergenceAlert | TASK-P3-002 | ✓ |
| Model: SimilarityThresholds | data_model | SimilarityThresholds | TASK-P3-003 | ✓ |
| Model: SpaceWeights | data_model | SpaceWeights | TASK-P3-003 | ✓ |
| Config: HIGH_THRESHOLDS | config | static_configuration | TASK-P3-003 | ✓ |
| Config: LOW_THRESHOLDS | config | static_configuration | TASK-P3-003 | ✓ |
| Config: SPACE_WEIGHTS (category-derived) | config | static_configuration | TASK-P3-005 | ✓ |
| Config: DIVERGENCE_SPACES (E1, E5-E7, E10, E12-E13) | config | static_configuration | TASK-P3-006 | ✓ |
| Component: MultiSpaceSimilarity | component | MultiSpaceSimilarity | TASK-P3-005 | ✓ |
| Component: DivergenceDetector | component | DivergenceDetector | TASK-P3-006 | ✓ |
| Component: SimilarityRetriever | component | SimilarityRetriever | TASK-P3-007 | ✓ |
| Component: DistanceCalculator | component | DistanceCalculator | TASK-P3-004 | ✓ |
| Method: cosine_similarity | method | DistanceCalculator | TASK-P3-004 | ✓ |
| Method: jaccard_similarity | method | DistanceCalculator | TASK-P3-004 | ✓ |
| Method: hamming_similarity | method | DistanceCalculator | TASK-P3-004 | ✓ |
| Method: max_sim | method | DistanceCalculator | TASK-P3-004 | ✓ |
| Method: transe_similarity | method | DistanceCalculator | TASK-P3-004 | ✓ |
| Method: compute_weighted_similarity | method | MultiSpaceSimilarity | TASK-P3-005 | ✓ |
| Method: is_divergence_space | method | DivergenceDetector | TASK-P3-006 | ✓ |

---

## Phase 4: Multi-Space Clustering

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-P4-01: HDBSCAN clustering | requirement | HDBSCANClusterer.fit | TASK-P4-005 | ✓ |
| REQ-P4-02: BIRCH incremental | requirement | BIRCHTree.insert | TASK-P4-006 | ✓ |
| REQ-P4-03: Per-space clustering | requirement | cluster_all_spaces | TASK-P4-007 | ✓ |
| REQ-P4-04: Topic synthesis | requirement | synthesize_topics | TASK-P4-008 | ✓ |
| REQ-P4-05: Stability tracking | requirement | update_topic_stability | TASK-P4-009 | ✓ |
| Model: ClusterMembership | data_model | ClusterMembership | TASK-P4-001 | ✓ |
| Model: Cluster | data_model | Cluster | TASK-P4-001 | ✓ |
| Model: Topic | data_model | Topic | TASK-P4-002 | ✓ |
| Model: TopicProfile | data_model | TopicProfile | TASK-P4-002 | ✓ |
| Model: TopicStability | data_model | TopicStability | TASK-P4-002 | ✓ |
| Model: TopicPhase | data_model | TopicPhase enum | TASK-P4-002 | ✓ |
| Model: HDBSCANParams | data_model | HDBSCANParams | TASK-P4-003 | ✓ |
| Model: BIRCHParams | data_model | BIRCHParams | TASK-P4-004 | ✓ |
| Model: ClusteringFeature | data_model | ClusteringFeature | TASK-P4-004 | ✓ |
| Component: HDBSCANClusterer | component | HDBSCANClusterer | TASK-P4-005 | ✓ |
| Component: BIRCHTree | component | BIRCHTree | TASK-P4-006 | ✓ |
| Component: MultiSpaceClusterManager | component | MultiSpaceClusterManager | TASK-P4-007 | ✓ |
| Component: TopicSynthesizer | component | TopicSynthesizer | TASK-P4-008 | ✓ |
| Component: TopicStabilityTracker | component | TopicStabilityTracker | TASK-P4-009 | ✓ |
| Config: HDBSCAN_DEFAULTS | config | static_configuration | TASK-P4-003 | ✓ |
| Config: BIRCH_DEFAULTS | config | static_configuration | TASK-P4-004 | ✓ |

---

## Phase 5: Injection Pipeline

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-P5-01: Context generation | requirement | generate_context | TASK-P5-007 | ✓ |
| REQ-P5-02: Divergence priority | requirement | generate_context | TASK-P5-007 | ✓ |
| REQ-P5-03: Priority calculation | requirement | compute_priority | TASK-P5-004 | ✓ |
| REQ-P5-04: Token budgeting | requirement | select_within_budget | TASK-P5-005 | ✓ |
| REQ-P5-05: Context formatting | requirement | format_full_context | TASK-P5-006 | ✓ |
| REQ-P5-06: Temporal enrichment badges | requirement | Priority 5 badges | TASK-P5-003b | ✓ |
| Model: InjectionCandidate | data_model | InjectionCandidate | TASK-P5-001 | ✓ |
| Model: InjectionCategory | data_model | InjectionCategory enum | TASK-P5-001 | ✓ |
| Model: TokenBudget | data_model | TokenBudget | TASK-P5-002 | ✓ |
| Model: InjectionResult | data_model | InjectionResult | TASK-P5-003 | ✓ |
| Model: RecencyFactor | data_model | RecencyFactor | TASK-P5-004 | ✓ |
| Model: DiversityBonus | data_model | DiversityBonus | TASK-P5-004 | ✓ |
| Model: weighted_agreement | data_model | InjectionCandidate.weighted_agreement | TASK-P5-001 | ✓ |
| Model: TemporalBadge | data_model | TemporalBadge | TASK-P5-003b | ✓ |
| Model: TemporalBadgeType | data_model | TemporalBadgeType enum | TASK-P5-003b | ✓ |
| Component: InjectionPipeline | component | InjectionPipeline | TASK-P5-007 | ✓ |
| Component: PriorityRanker | component | PriorityRanker | TASK-P5-004 | ✓ |
| Component: TokenBudgetManager | component | TokenBudgetManager | TASK-P5-005 | ✓ |
| Component: ContextFormatter | component | ContextFormatter | TASK-P5-006 | ✓ |
| Component: TemporalEnrichmentProvider | component | TemporalEnrichmentProvider | TASK-P5-003b | ✓ |
| Config: DEFAULT_TOKEN_BUDGET | config | static_configuration | TASK-P5-002 | ✓ |

---

## Phase 6: CLI & Hooks

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-P6-01: CLI binary | requirement | CliApp | TASK-P6-001 | ✓ |
| REQ-P6-02: Hook scripts | requirement | hook_scripts | TASK-P6-008 | ✓ |
| REQ-P6-03: Output format | requirement | format_full_context | TASK-P6-003 | ✓ |
| REQ-P6-04: Session management | requirement | SessionCommands | TASK-P6-002 | ✓ |
| REQ-P6-05: Setup command | requirement | SetupCommand | TASK-P6-007 | ✓ |
| REQ-P6-06: Timeout compliance | requirement | hook_scripts | TASK-P6-008 | ✓ |
| REQ-P6-07: Env var reading | requirement | EnvReader | TASK-P6-001 | ✓ |
| REQ-P6-08: Verbose mode | requirement | CliConfig | TASK-P6-001 | ✓ |
| Model: CliConfig | data_model | CliConfig | TASK-P6-001 | ✓ |
| Model: HookSettings | data_model | HookSettings | TASK-P6-007 | ✓ |
| Command: session start | cli_command | handle_session_start | TASK-P6-002 | ✓ |
| Command: session end | cli_command | handle_session_end | TASK-P6-002 | ✓ |
| Command: inject-context | cli_command | handle_inject_context | TASK-P6-003 | ✓ |
| Command: inject-brief | cli_command | handle_inject_brief | TASK-P6-004 | ✓ |
| Command: capture-memory | cli_command | handle_capture_memory | TASK-P6-005 | ✓ |
| Command: capture-response | cli_command | handle_capture_response | TASK-P6-006 | ✓ |
| Command: setup | cli_command | handle_setup | TASK-P6-007 | ✓ |
| Script: session-start.sh | script | hook_scripts | TASK-P6-008 | ✓ |
| Script: user-prompt-submit.sh | script | hook_scripts | TASK-P6-008 | ✓ |
| Script: pre-tool-use.sh | script | hook_scripts | TASK-P6-008 | ✓ |
| Script: post-tool-use.sh | script | hook_scripts | TASK-P6-008 | ✓ |
| Script: stop.sh | script | hook_scripts | TASK-P6-008 | ✓ |
| Script: session-end.sh | script | hook_scripts | TASK-P6-008 | ✓ |
| Test: E2E integration | test | test_plan | TASK-P6-009 | ✓ |
| Test: Performance | test | test_plan | TASK-P6-010 | ✓ |

---

## Embedder Categories (Cross-Cutting Requirements)

| Spec Item | Type | Tech Spec Section | Task ID | ✓ |
|-----------|------|-------------------|---------|---|
| REQ-EMB-CAT-01: Embedder category classification | requirement | Embedder Categories | TASK-P4-008 | ✓ |
| REQ-EMB-CAT-02: Temporal exclusion from topic detection | requirement | Category-Specific Rules | TASK-P4-008 | ✓ |
| REQ-EMB-CAT-03: Weighted agreement formula | requirement | Topic Formation Rule | TASK-P4-008 | ✓ |
| Const: TOPIC_THRESHOLD (2.5) | config | Topic Detection Threshold | TASK-P4-008 | ✓ |
| Const: MAX_WEIGHTED_AGREEMENT (8.5) | config | Topic Formation Rule | TASK-P4-008 | ✓ |
| Const: SEMANTIC_WEIGHT (1.0) | config | Embedder Category Weights | TASK-P4-008 | ✓ |
| Const: TEMPORAL_WEIGHT (0.0) | config | Embedder Category Weights | TASK-P4-008 | ✓ |
| Const: RELATIONAL_WEIGHT (0.5) | config | Embedder Category Weights | TASK-P4-008 | ✓ |
| Const: STRUCTURAL_WEIGHT (0.5) | config | Embedder Category Weights | TASK-P4-008 | ✓ |
| Formula: topic_confidence | formula | topic_confidence = weighted_agreement / 8.5 | TASK-P4-008 | ✓ |
| Constraint: Temporal exclusion from relevance | constraint | Similarity Detection | TASK-P3-005 | ✓ |
| Constraint: Temporal exclusion from divergence | constraint | Divergence Detection | TASK-P3-006 | ✓ |
| Feature: Temporal badges in Priority 5 | feature | Temporal Context Enrichment | TASK-P5-003b | ✓ |

---

## Uncovered Items

**(none)**

All specification items are covered by at least one task.

---

## Validation Checklist

- [x] All data models have tasks
- [x] All service methods have tasks
- [x] All CLI commands have tasks
- [x] All error states handled
- [x] All edge cases covered
- [x] Task dependencies form valid DAG (no cycles)
- [x] Layer ordering correct (foundation → logic → surface)
- [x] All hook scripts have tasks
- [x] All configuration items have tasks

---

## Coverage Summary

| Phase | Requirements | Models | Components | Methods | Configs | Total | Tasks | Coverage |
|-------|--------------|--------|------------|---------|---------|-------|-------|----------|
| P0 | 5 | 0 | 0 | 0 | 0 | 7 | 5 | 100% |
| P1 | 6 | 6 | 5 | 0 | 0 | 21 | 8 | 100% |
| P2 | 7 | 9 | 3 | 3 | 0 | 25 | 7 | 100% |
| P3 | 7 | 5 | 4 | 7 | 4 | 29 | 7 | 100% |
| P4 | 5 | 9 | 5 | 0 | 2 | 20 | 9 | 100% |
| P5 | 6 | 9 | 5 | 0 | 1 | 21 | 8 | 100% |
| P6 | 8 | 2 | 0 | 0 | 0 | 12 | 10 | 100% |
| EMB-CAT | 3 | 0 | 0 | 0 | 4 | 13 | — | 100% |
| **Total** | **47** | **40** | **22** | **10** | **11** | **148** | **54** | **100%** |
