# Context Graph Provenance Improvement Plan

## Full System Investigation Report & Implementation Roadmap

**Date:** 2026-02-06
**Branch:** casetrack
**Scope:** Complete provenance audit across all system layers
**Status:** Investigation Complete - Awaiting Implementation

---

## Executive Summary

A comprehensive forensic investigation of the entire Context Graph codebase reveals that **provenance tracking is strong for source content tracing but critically deficient for operational metadata, retrieval transparency, and lifecycle audit trails**. The system can answer "what document does this chunk come from?" but cannot answer "who merged these memories?", "which model generated this relationship?", or "why was this result returned?".

**Overall Provenance Completeness: ~55%**

| Layer | Completeness | Assessment |
|-------|-------------|------------|
| Source Content Tracing | 95% | Excellent - character-level spans, SHA-256 hashes |
| Confidence Tracking | 90% | Good - LLM confidence stored for relationships |
| Temporal Metadata | 75% | Partial - creation timestamps exist, access/modification gaps |
| Retrieval Transparency | 25% | Critical Gap - rich internal data never exposed to users |
| LLM/Model Provenance | 20% | Critical Gap - no model version tracking |
| Operational Audit Trail | 15% | Critical Gap - no persistent audit log |
| Agent/User Attribution | 0% | Missing - no `created_by` or operator tracking |

---

## Table of Contents

1. [Investigation Methodology](#1-investigation-methodology)
2. [Current Provenance Architecture](#2-current-provenance-architecture)
3. [Gap Analysis by System Layer](#3-gap-analysis-by-system-layer)
4. [Findings Summary](#4-findings-summary)
5. [Implementation Plan](#5-implementation-plan)
6. [New Architectural Components](#6-new-architectural-components)
7. [Priority Roadmap](#7-priority-roadmap)
8. [PRD Compliance Assessment](#8-prd-compliance-assessment)

---

## 1. Investigation Methodology

Six parallel investigation threads examined every layer of the system:

1. **Memory Storage Layer** - SourceMetadata, NodeMetadata, TeleologicalFingerprint, CausalRelationship, FileIndexEntry structs and their RocksDB storage
2. **Retrieval & Search Pipeline** - SearchResult structs, RRF fusion, pipeline stages, score breakdowns, MCP tool responses
3. **Hooks & MCP Tool Layer** - Hook execution, tool call tracking, session management, file watcher
4. **Causal & Graph Discovery** - LLM analysis provenance, relationship storage, discovery cycle tracking
5. **Entity & Code Systems** - E11/KEPLER entity tracking, E7/Qodo code embedding, TransE relationships
6. **Consolidation & Lifecycle** - Merge operations, soft delete, importance boosting, topic detection, dream consolidation

### Files Examined

**Core Types:**
- `crates/context-graph-core/src/types/source_metadata.rs`
- `crates/context-graph-core/src/types/memory_node/metadata.rs`
- `crates/context-graph-core/src/types/fingerprint/teleological/types.rs`
- `crates/context-graph-core/src/types/causal_relationship.rs`
- `crates/context-graph-core/src/types/graph_edge/edge.rs`
- `crates/context-graph-core/src/types/file_index.rs`
- `crates/context-graph-core/src/entity/mod.rs`
- `crates/context-graph-core/src/memory/ast_chunker.rs`
- `crates/context-graph-core/src/memory/code_watcher.rs`
- `crates/context-graph-core/src/retrieval/result.rs`
- `crates/context-graph-core/src/retrieval/aggregation.rs`

**Storage:**
- `crates/context-graph-storage/src/teleological/rocksdb_store/source_metadata.rs`
- `crates/context-graph-storage/src/teleological/rocksdb_store/fusion.rs`
- `crates/context-graph-storage/src/teleological/column_families.rs`
- `crates/context-graph-storage/src/code/store.rs`

**MCP Handlers:**
- `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/causal_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/graph_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/consolidation.rs`
- `crates/context-graph-mcp/src/handlers/merge.rs`

**Discovery Agents:**
- `crates/context-graph-causal-agent/src/types/mod.rs`
- `crates/context-graph-causal-agent/src/service/mod.rs`
- `crates/context-graph-causal-agent/src/llm/mod.rs`
- `crates/context-graph-graph-agent/src/types/mod.rs`

---

## 2. Current Provenance Architecture

### What IS Well-Tracked

#### 2.1 Source Content Tracing (95% Complete)

The `SourceMetadata` struct provides strong document-to-chunk provenance:

```
SourceMetadata Fields (EXISTING):
  source_type: SourceType          -- MDFileChunk | HookDescription | ClaudeResponse | Manual | CausalExplanation
  file_path: Option<String>        -- Full path to source file
  chunk_index: Option<u32>         -- 0-based within file
  total_chunks: Option<u32>        -- Total chunks from file
  start_line: Option<u32>          -- 1-based start line
  end_line: Option<u32>            -- 1-based end line (inclusive)
  hook_type: Option<String>        -- Which hook created this
  tool_name: Option<String>        -- Which tool was involved
  session_id: Option<String>       -- Session context
  session_sequence: Option<u64>    -- Order within session (E4)
  causal_direction: Option<String> -- Precomputed from E5 norms
```

#### 2.2 Causal Relationship Provenance (Reference Model)

`CausalRelationship` is the **gold standard** for provenance in the codebase:

```
CausalRelationship Fields (EXISTING):
  source_fingerprint_id: Uuid      -- Links back to source memory
  source_content: String           -- Full original text (up to 500 chars)
  source_spans: Vec<CausalSourceSpan>  -- Character-level offsets + excerpts
  confidence: f32                  -- LLM confidence [0.0, 1.0]
  mechanism_type: String           -- direct | mediated | feedback | temporal
  created_at: i64                  -- Unix timestamp
```

With `CausalSourceSpan` providing character-level provenance:
```
  start_char: usize               -- 0-based character offset
  end_char: usize                 -- Exclusive end offset
  text_excerpt: String            -- Exact text (up to 200 chars)
  span_type: String               -- cause | effect | full
```

#### 2.3 RRF Fusion Internals (Tracked but Not Exposed)

The `CombinedResult` struct tracks rich retrieval provenance internally:

```
CombinedResult Fields (EXISTING but NOT EXPOSED):
  found_by: HashSet<usize>        -- Which embedder indices found this
  primary_embedder: usize          -- Best-ranking embedder
  unique_contribution: bool        -- Blind-spot discovery flag
  best_rank: usize                 -- Position in best embedder
  insight_annotation: Option<String> -- Always None (never populated)
```

And `weighted_rrf_fusion_with_scores()` returns per-embedder scores in a `HashMap<String, f32>` that is computed but discarded before reaching the user.

---

## 3. Gap Analysis by System Layer

### GAP-01: No User/Agent Attribution (ALL LAYERS)

**Severity: CRITICAL**
**Affects: Every operation in the system**

No operation anywhere in the system tracks WHO performed it.

| Operation | `created_by` | `operator_id` |
|-----------|-------------|---------------|
| store_memory | Missing | Missing |
| merge_concepts | Missing | Missing |
| forget_concept | Missing | Missing |
| boost_importance | Missing | Missing |
| trigger_causal_discovery | Missing | Missing |

**Impact:** Cannot comply with GDPR right-to-audit, SOC2, or legal discovery requirements. Multi-tenant systems cannot attribute memories to users.

---

### GAP-02: No LLM Model Version Tracking (Causal & Graph Discovery)

**Severity: CRITICAL**
**Affects:** `CausalRelationship`, `GraphAnalysisResult`, `GraphEdge`

When the causal agent (Hermes 2 Pro / Qwen2.5) or graph agent analyzes memory pairs, the following is NOT stored:

```
MISSING:
  llm_model_name: String           -- "Hermes 2 Pro Mistral 7B"
  llm_model_version: String        -- Version tag or commit hash
  llm_quantization: String         -- "Q5_K_M"
  llm_temperature: f32             -- 0.0
  llm_prompt_template_hash: String -- SHA-256 of prompt template
```

**Impact:** If the LLM model is upgraded, cannot determine which relationships used old vs new model. Cannot reproduce analysis. Cannot correlate model version with relationship quality.

**Current Code:**
- `LlmConfig` exists in `context-graph-causal-agent/src/llm/mod.rs` with model_path, temperature, etc.
- But this config is runtime-only, NEVER stored with relationships.

---

### GAP-03: Retrieval Transparency (Search Pipeline)

**Severity: CRITICAL**
**Affects:** All search MCP tools (search_graph, search_causes, search_connections, search_by_intent, etc.)

Rich internal provenance data is computed but never exposed to users:

| Internal Data | Tracked? | Exposed? |
|---------------|----------|----------|
| Embedder contributions (found_by) | Yes | No |
| Primary embedder | Yes | No |
| Blind-spot discoveries | Yes | No |
| RRF per-result breakdown | Yes | No |
| Strategy used (E1Only/MultiSpace/Pipeline) | Yes | No |
| Weight profile applied | Yes | No |
| Auto-detected query type (causal/code/intent) | Yes | No |
| E10 multiplicative boost value | Yes | No |
| Direction modifier per result | Yes | No |
| Pipeline stage filtering reasons | Yes | No |
| Insight annotations | Struct exists | Never populated |

**Impact:** Users see only final scores without any explanation of WHY a result was returned or HOW scores were computed. Cannot debug poor results.

**Example of current response:**
```json
{ "memory_id": "abc123", "aggregate_score": 0.0512, "space_count": 8 }
```

**What should be available:**
```json
{
  "memory_id": "abc123",
  "aggregate_score": 0.0512,
  "provenance": {
    "strategy": "MultiSpace",
    "weight_profile": "causal_reasoning",
    "found_by": ["E1", "E5", "E6", "E7", "E8", "E10", "E11", "E13"],
    "primary_embedder": "E5_Causal",
    "consensus": "8 of 13 embedders (61%)",
    "blind_spot_discovery": false,
    "per_embedder_scores": { "E1": 0.90, "E5": 0.85, "E6": 0.88 }
  }
}
```

---

### GAP-04: No Persistent Audit Log (ALL Operations)

**Severity: CRITICAL**
**Affects:** Every mutation operation

No append-only audit log exists. Operations modify state without recording history.

| Operation | Audit Logged? | Reversible? | History? |
|-----------|--------------|-------------|----------|
| store_memory | No | N/A | No |
| merge_concepts | 30-day reversal only | 30 days | 30 days then LOST |
| forget_concept | deleted_at flag only | 30 days soft | No reason/who |
| boost_importance | No | No | Current value only |
| trigger_consolidation | Ephemeral response only | N/A | Not persisted at all |
| detect_topics | Final state only | N/A | No membership history |
| causal_discovery | Confidence only | No | No rejected pairs |

**Impact:** After 30 days, merge history is permanently lost. Importance change history does not exist. Consolidation recommendations are discarded after the response. Cannot perform legal discovery or compliance audits.

---

### GAP-05: No Embedding Model Version Tracking (ALL Embedders)

**Severity: HIGH**
**Affects:** TeleologicalFingerprint (all 13 embedders), CodeEntity (E7)

When embeddings are computed, no record of which model version generated them:

```
MISSING in TeleologicalFingerprint:
  embedding_model_versions: HashMap<String, String>
  embeddings_computed_at: DateTime<Utc>

MISSING in CodeStore:
  e7_model_version: String
  e7_computed_at: DateTime<Utc>
```

**Impact:** If embedder models are updated, cannot determine which memories need re-embedding. Cannot detect stale embeddings.

---

### GAP-06: Entity System Provenance (E11)

**Severity: HIGH**
**Affects:** Entity extraction, canonicalization, TransE relationships

| Gap | Detail |
|-----|--------|
| No entity-to-source-memory linkage | Cannot trace where an entity was first discovered |
| Entity confidence not persisted | Computed at API time but never stored |
| Canonicalization not tracked | "postgres" → "postgresql" applied without record |
| TransE scores discarded | Computed during graph building, not stored with edges |
| No extraction method recorded | Cannot distinguish KB-matched vs heuristic entities |
| No entity alias tracking | No synonym registry |

**Reference:** The `CausalSourceSpan` pattern should be replicated for entities.

---

### GAP-07: Code Embedding Provenance (E7)

**Severity: HIGH**
**Affects:** Code entities, AST parsing, code watcher

| Gap | Detail |
|-----|--------|
| No git commit metadata | Watcher runs `git ls-files` but discards commit/author/branch |
| File hash not persisted with chunks | SHA-256 computed in-memory HashMap, lost on restart |
| AST parse errors not stored | Warnings logged but not persisted with entities |
| No E7 model version tracking | Cannot determine which Qodo-Embed version was used |
| No chunk merge/split provenance | AST chunker merges small nodes without recording |

---

### GAP-08: Hook Execution Provenance

**Severity: MEDIUM-HIGH**
**Affects:** All 6 hook types (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop, SessionEnd)

| Gap | Detail |
|-----|--------|
| No hook execution audit log | Hook invocations not persisted |
| No tool_use_id in SourceMetadata | Cannot link memory back to specific tool invocation |
| Hook timestamps not stored | HookInput.timestamp_ms received but discarded |
| Tool success/failure not validated | PostToolUse.tool_success is optional, never persisted |
| No MCP request ID tracking | JsonRpc request ID not linked to resulting memories |
| No call chain tracking | Parent-child tool relationships not tracked |

---

### GAP-09: Merge & Consolidation History

**Severity: HIGH**
**Affects:** merge_concepts, trigger_consolidation

| Gap | Detail |
|-----|--------|
| Reversal records expire in 30 days | After expiration, merge lineage permanently lost |
| No back-link from merged node | Cannot query "what was merged to create this?" |
| Consolidation recommendations ephemeral | Analysis discarded after response |
| No recommendation-to-action linking | Cannot trace if user followed consolidation suggestion |
| No cascading audit | If A merged to B, B deleted - no trace of A's fate |

---

### GAP-10: Importance & Metadata Change History

**Severity: MEDIUM**
**Affects:** boost_importance, any metadata modifications

| Gap | Detail |
|-----|--------|
| No importance change history | Only current value stored, no historical record |
| No modification timestamps per field | NodeMetadata.version increments but no per-field tracking |
| No modification reason | Cannot explain why importance was changed |

---

## 4. Findings Summary

### Provenance Completeness by Data Structure

| Structure | What IS Tracked | What is MISSING |
|-----------|----------------|-----------------|
| **SourceMetadata** | source_type, file_path, chunk_index, lines, hook_type, tool_name, session_id, session_sequence, causal_direction | created_at, created_by, file_hash, tool_use_id, mcp_request_id, hook_timestamp |
| **NodeMetadata** | source, modality, tags, version, consolidated/at, deleted/at, rationale, custom | deleted_by, deletion_reason, consolidated_with, consolidated_by, modification_history |
| **TeleologicalFingerprint** | id, content_hash, created_at, last_updated, access_count, importance | last_accessed_at, embedding_model_versions, embeddings_computed_at |
| **CausalRelationship** | source_fingerprint_id, source_content, source_spans, confidence, mechanism_type, created_at | llm_model_version, llm_prompt_hash, extraction_method, scanner_initial_score, agent_attribution |
| **GraphEdge** | source_id, target_id, edge_type, weight, confidence, created_at, last_traversed_at, traversal_count | created_by_agent, discovery_method, llm_model_version, steering_reward_history |
| **EntityLink** | surface_form, canonical_id, entity_type | source_memory_id, extraction_confidence, extraction_method, discovered_at |
| **CodeEntity** | file_path, line_start/end, content_hash, entity_type, scope_chain | git_commit, git_author, git_branch, e7_model_version, parse_errors |

### Provenance Coverage by Question Type

| Question | Answerable? | Evidence |
|----------|-------------|---------|
| "What file does this chunk come from?" | YES | SourceMetadata.file_path + line numbers |
| "What text was this relationship extracted from?" | YES | CausalSourceSpan with character offsets |
| "How confident was the LLM?" | YES | confidence field on relationships |
| "When was this memory created?" | PARTIAL | Fingerprint.created_at exists, SourceMetadata lacks it |
| "Which model generated this relationship?" | NO | Model version not stored |
| "Who created/deleted/merged this memory?" | NO | No operator tracking |
| "Why was this result returned?" | NO | Internal data not exposed |
| "Which embedders found this result?" | NO | CombinedResult.found_by not exposed |
| "What strategy was used for this search?" | NO | Strategy selection not returned |
| "Why was this memory deleted?" | NO | No deletion_reason field |
| "What was merged to create this?" | 30 DAYS ONLY | Reversal records expire |
| "Were consolidation recommendations followed?" | NO | Recommendations ephemeral |
| "What importance changes occurred?" | NO | No change history |
| "Which git commit introduced this code?" | NO | Git metadata not persisted |
| "Where was this entity first discovered?" | NO | No entity-to-source linkage |

---

## 5. Implementation Plan

### Phase 1: Core Provenance Infrastructure (P0 - Critical)

#### 5.1 Create Audit Log Column Family

**New RocksDB CF:** `CF_AUDIT_LOG`

```rust
pub struct AuditRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub operation: AuditOperation,
    pub target_id: Uuid,                    // Memory/relationship/entity affected
    pub operator_id: Option<String>,        // User/agent who performed operation
    pub session_id: Option<String>,         // Session context
    pub rationale: Option<String>,          // Why this operation was performed
    pub parameters: serde_json::Value,      // Operation-specific parameters
    pub result: AuditResult,                // Success/Failure + details
    pub previous_state: Option<Vec<u8>>,    // Serialized previous state (for reversibility)
}

pub enum AuditOperation {
    MemoryCreated,
    MemoryMerged { source_ids: Vec<Uuid>, strategy: String },
    MemoryDeleted { soft: bool, reason: Option<String> },
    MemoryRestored,
    ImportanceBoosted { old: f32, new: f32, delta: f32 },
    RelationshipDiscovered { relationship_type: String, confidence: f32 },
    ConsolidationAnalyzed { candidates_found: usize },
    TopicDetected { topic_id: String, members: usize },
    EmbeddingRecomputed { embedder: String, model_version: String },
}

pub enum AuditResult {
    Success,
    Failure { error: String },
    Partial { warnings: Vec<String> },
}
```

**Key Design:**
- Append-only (no updates or deletes)
- Keyed by timestamp + UUID for chronological ordering
- Secondary index by target_id for "show all operations on memory X"
- Bloom filter for fast existence checks
- LZ4 compression

**Files to modify:**
- `crates/context-graph-storage/src/teleological/column_families.rs` - Add CF definition
- `crates/context-graph-storage/src/teleological/rocksdb_store/` - Add audit_log.rs module
- `crates/context-graph-core/src/types/` - Add audit.rs type definitions

---

#### 5.2 Add Operator Attribution to All Operations

**Add to SourceMetadata:**
```rust
pub created_by: Option<String>,           // User/agent identifier
pub created_at: Option<DateTime<Utc>>,    // Explicit creation timestamp
```

**Add to all MCP tool handlers:**
- Accept optional `operator_id` parameter
- Pass through to storage layer
- Log in audit trail

**Files to modify:**
- `crates/context-graph-core/src/types/source_metadata.rs` - Add fields
- `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` - Accept operator_id
- `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` - Accept operator_id
- `crates/context-graph-mcp/src/handlers/merge.rs` - Accept operator_id

---

#### 5.3 Add LLM Model Metadata to Relationships

**New struct:**
```rust
pub struct LLMProvenance {
    pub model_name: String,                // "Hermes 2 Pro Mistral 7B"
    pub model_version: String,             // Version tag or file hash
    pub quantization: String,              // "Q5_K_M"
    pub temperature: f32,                  // 0.0
    pub max_tokens: usize,                // 512
    pub prompt_template_hash: String,      // SHA-256 of template used
    pub grammar_type: Option<String>,      // "causal_analysis" | "graph_relationship"
    pub tokens_consumed: Option<u32>,      // Actual tokens used
    pub generation_time_ms: Option<u64>,   // LLM inference time
}
```

**Add to CausalRelationship:**
```rust
pub llm_provenance: Option<LLMProvenance>,
```

**Add to GraphEdge:**
```rust
pub discovery_provenance: Option<LLMProvenance>,
```

**Files to modify:**
- `crates/context-graph-core/src/types/causal_relationship.rs` - Add field
- `crates/context-graph-core/src/types/graph_edge/edge.rs` - Add field
- `crates/context-graph-causal-agent/src/service/mod.rs` - Populate during discovery
- `crates/context-graph-graph-agent/src/service/mod.rs` - Populate during discovery
- `crates/context-graph-causal-agent/src/llm/mod.rs` - Expose config as metadata

---

### Phase 2: Retrieval Transparency (P0 - Critical)

#### 5.4 Expose Embedder Contributions in Search Results

**Modify search result DTOs to include provenance:**

```rust
pub struct SearchResultProvenance {
    pub strategy: String,                    // "E1Only" | "MultiSpace" | "Pipeline"
    pub weight_profile: String,              // "semantic_search" | "causal_reasoning" etc.
    pub query_classification: QueryClassification,
    pub embedder_contributions: Vec<EmbedderContribution>,
    pub consensus_score: f32,                // Fraction of embedders that agreed
    pub primary_embedder: String,            // "E5_Causal"
    pub is_blind_spot_discovery: bool,       // Found by only 1 embedder
}

pub struct QueryClassification {
    pub detected_type: String,               // "causal" | "code" | "intent" | "general"
    pub detection_patterns: Vec<String>,      // ["why", "cause"] for causal
    pub intent_mode: Option<String>,          // "SeekingIntent" | "SeekingContext"
    pub e10_boost_applied: Option<f32>,       // 1.2x or 0.8x
}

pub struct EmbedderContribution {
    pub embedder: String,                    // "E1_Semantic"
    pub similarity: f32,                     // Raw score in this space
    pub rank: usize,                         // Rank in this embedder's results
    pub rrf_contribution: f32,               // 1/(K + rank + 1)
    pub weight: f32,                         // Profile weight applied
}
```

**Add `include_provenance: bool` parameter to all search tools** (default: false for backwards compatibility).

**Files to modify:**
- `crates/context-graph-core/src/retrieval/result.rs` - Add provenance to result types
- `crates/context-graph-core/src/retrieval/aggregation.rs` - Expose CombinedResult data
- `crates/context-graph-storage/src/teleological/rocksdb_store/fusion.rs` - Return per-embedder scores
- `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` - Add include_provenance param
- `crates/context-graph-mcp/src/handlers/tools/causal_tools.rs` - Add include_provenance param
- `crates/context-graph-mcp/src/handlers/tools/graph_tools.rs` - Add include_provenance param

---

#### 5.5 Populate Insight Annotations

The `insight_annotation: Option<String>` field exists in `CombinedResult` but is always `None`.

**Implementation:**
```rust
// In aggregation.rs, after building CombinedResult:
if result.unique_contribution {
    result.insight_annotation = Some(format!(
        "Blind-spot discovery: Only {} found this result (other embedders missed it)",
        result.primary_embedder_name()
    ));
} else if result.found_by.len() >= 10 {
    result.insight_annotation = Some(format!(
        "Strong consensus: {}/13 embedders agree on this result",
        result.found_by.len()
    ));
}
```

**Files to modify:**
- `crates/context-graph-core/src/retrieval/aggregation.rs` - Populate annotations

---

### Phase 3: Entity & Code Provenance (P1 - High)

#### 5.6 Entity Source Provenance

**New struct (modeled after CausalSourceSpan):**
```rust
pub struct EntityProvenance {
    pub entity: EntityLink,
    pub source_memory_id: Uuid,
    pub source_spans: Vec<EntitySourceSpan>,
    pub extraction_method: EntityExtractionMethod,
    pub confidence: f32,
    pub extracted_at: DateTime<Utc>,
}

pub struct EntitySourceSpan {
    pub start_char: usize,
    pub end_char: usize,
    pub text_excerpt: String,
}

pub enum EntityExtractionMethod {
    KnowledgeBase,
    HeuristicPattern,
    TransEInferred,
    LLMExtracted,
}
```

**Add confidence field to EntityLink:**
```rust
pub struct EntityLink {
    pub surface_form: String,
    pub canonical_id: String,
    pub entity_type: EntityType,
    pub confidence: f32,              // NEW: persisted, not just API-time
}
```

**New CF:** `CF_ENTITY_PROVENANCE` mapping entity canonical_id + memory_id to EntityProvenance

**Files to modify:**
- `crates/context-graph-core/src/entity/mod.rs` - Add confidence to EntityLink
- `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs` - Store provenance during extraction
- `crates/context-graph-storage/` - Add entity provenance CF

---

#### 5.7 Code Git Metadata

**Extend CodeEntity:**
```rust
pub struct CodeGitMetadata {
    pub commit_hash: Option<String>,
    pub author: Option<String>,
    pub branch: Option<String>,
    pub commit_timestamp: Option<DateTime<Utc>>,
}
```

**Persist file hash with chunks:**
```rust
// In SourceMetadata (new fields):
pub file_content_hash: Option<String>,     // SHA-256 of source file
pub file_modified_at: Option<DateTime<Utc>>,
```

**Capture during code watcher scan:**
- The watcher already runs `git ls-files` and `git status --porcelain`
- Add `git log -1 --format="%H|%an|%aI" -- {file}` for each changed file
- Store results with CodeEntity

**Files to modify:**
- `crates/context-graph-core/src/memory/code_watcher.rs` - Capture git metadata
- `crates/context-graph-core/src/types/source_metadata.rs` - Add file_content_hash
- `crates/context-graph-storage/src/code/store.rs` - Store git metadata

---

#### 5.8 TransE Score Persistence

**Add to entity edge storage:**
```rust
pub struct EntityEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub weight: f32,
    pub memory_ids: Vec<Uuid>,
    pub transe_score: Option<f32>,          // NEW: TransE confidence
    pub discovery_method: RelationshipOrigin, // NEW: How discovered
}

pub enum RelationshipOrigin {
    CoOccurrence { count: usize },
    TransEInferred { score: f32 },
    LLMInferred { confidence: f32, model: String },
}
```

**Files to modify:**
- `crates/context-graph-mcp/src/handlers/tools/entity_dtos.rs` - Add transe_score
- `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs` - Store during graph building

---

### Phase 4: Lifecycle Provenance (P1 - High)

#### 5.9 Enhanced Deletion Tracking

**Extend NodeMetadata soft delete:**
```rust
pub struct DeletionMetadata {
    pub deleted_by: Option<String>,
    pub deletion_reason: Option<String>,
    pub deleted_at: DateTime<Utc>,
    pub recovery_deadline: DateTime<Utc>,    // Explicit 30-day deadline
}
```

**Add to forget_concept MCP tool:**
- Accept optional `reason` parameter
- Accept optional `operator_id` parameter
- Store in DeletionMetadata + audit log

**Files to modify:**
- `crates/context-graph-core/src/types/memory_node/metadata.rs` - Add DeletionMetadata
- `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` - Accept reason/operator

---

#### 5.10 Permanent Merge History

**Replace 30-day expiring reversal records with permanent merge history:**

**New CF:** `CF_MERGE_HISTORY`

```rust
pub struct MergeRecord {
    pub id: Uuid,
    pub merged_id: Uuid,                    // Resulting merged fingerprint
    pub source_ids: Vec<Uuid>,              // Original fingerprints
    pub strategy: String,                   // union | intersection | weighted_average
    pub rationale: String,                  // User-provided reason
    pub operator_id: Option<String>,        // Who merged
    pub timestamp: DateTime<Utc>,
    pub reversal_hash: String,              // For undo capability
    pub original_fingerprints_json: Vec<String>, // Serialized originals
}
```

**Add back-link from merged fingerprint:**
```rust
// In SourceMetadata or NodeMetadata:
pub derived_from: Option<Vec<Uuid>>,        // Source fingerprint IDs
pub derivation_method: Option<String>,      // "merge" | "consolidation" | "expansion"
```

**Keep reversal records for undo, but make merge history permanent.**

**Files to modify:**
- `crates/context-graph-mcp/src/handlers/merge.rs` - Write to CF_MERGE_HISTORY
- `crates/context-graph-core/src/types/source_metadata.rs` - Add derived_from
- `crates/context-graph-storage/src/teleological/column_families.rs` - Add CF_MERGE_HISTORY

---

#### 5.11 Importance Change History

**New CF:** `CF_IMPORTANCE_HISTORY`

```rust
pub struct ImportanceChangeRecord {
    pub memory_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub old_value: f32,
    pub new_value: f32,
    pub delta: f32,
    pub operator_id: Option<String>,
    pub reason: Option<String>,
}
```

**Files to modify:**
- `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` - Write history on boost
- `crates/context-graph-storage/src/teleological/column_families.rs` - Add CF

---

### Phase 5: Hook & Tool Call Provenance (P2 - Medium)

#### 5.12 Tool Call Registry

**Add tool_use_id to SourceMetadata:**
```rust
pub tool_use_id: Option<String>,           // From HookPayload
pub mcp_request_id: Option<String>,        // JsonRpc request ID
pub hook_execution_timestamp_ms: Option<i64>, // From HookInput
```

**Create tool call → memory mapping:**

**New CF:** `CF_TOOL_CALL_INDEX`
```
Key: tool_use_id
Value: Vec<Uuid>  (fingerprint IDs created by this tool call)
```

**Files to modify:**
- `crates/context-graph-core/src/types/source_metadata.rs` - Add fields
- `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` - Pass through IDs
- `crates/context-graph-mcp/src/handlers/tools/post_tool_use.rs` - Capture tool_use_id

---

#### 5.13 Hook Execution Audit Log

**Record every hook invocation:**
```rust
pub struct HookExecutionRecord {
    pub hook_type: String,
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub exit_code: i32,
    pub tool_name: Option<String>,
    pub tool_use_id: Option<String>,
    pub success: bool,
    pub error_message: Option<String>,
    pub memories_created: Vec<Uuid>,
}
```

**Files to modify:**
- `crates/context-graph-cli/src/commands/hooks/` - Log execution
- New CF or append to CF_AUDIT_LOG

---

#### 5.14 Consolidation Recommendation Persistence

**Store consolidation analysis results:**

**New CF:** `CF_CONSOLIDATION_RECOMMENDATIONS`

```rust
pub struct ConsolidationRecommendation {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub strategy: String,
    pub candidates: Vec<ConsolidationCandidate>,
    pub session_id: Option<String>,
    pub operator_id: Option<String>,
    pub status: RecommendationStatus,       // Pending | Accepted | Rejected | Expired
    pub acted_on_at: Option<DateTime<Utc>>,
    pub resulting_merge_id: Option<Uuid>,   // If accepted, link to merge
}

pub struct ConsolidationCandidate {
    pub source_ids: Vec<Uuid>,
    pub target_id: Uuid,
    pub similarity: f32,
    pub combined_alignment: f32,
}

pub enum RecommendationStatus {
    Pending,
    Accepted { merge_id: Uuid },
    Rejected { reason: Option<String> },
    Expired,
}
```

**Files to modify:**
- `crates/context-graph-mcp/src/handlers/tools/consolidation.rs` - Persist recommendations
- `crates/context-graph-mcp/src/handlers/merge.rs` - Link merge to recommendation if applicable

---

### Phase 6: Embedding Model Tracking (P2 - Medium)

#### 5.15 Embedding Version Registry

**New CF:** `CF_EMBEDDING_REGISTRY`

```rust
pub struct EmbeddingVersionRecord {
    pub fingerprint_id: Uuid,
    pub computed_at: DateTime<Utc>,
    pub embedder_versions: HashMap<String, String>,  // "E1" → "all-MiniLM-L6-v2.1"
    pub e7_model_version: Option<String>,             // "qodo-embed-1-1.5b"
    pub computation_time_ms: Option<u64>,
}
```

**Track during fingerprint creation in store_memory:**

**Files to modify:**
- `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` - Record versions
- `crates/context-graph-storage/src/teleological/column_families.rs` - Add CF

---

## 6. New Architectural Components

### 6.1 Summary of New Column Families

| CF Name | Purpose | Key Format | Retention |
|---------|---------|-----------|-----------|
| `CF_AUDIT_LOG` | Append-only audit trail | timestamp + UUID | Permanent |
| `CF_MERGE_HISTORY` | Permanent merge lineage | merge_id | Permanent |
| `CF_IMPORTANCE_HISTORY` | Importance change records | memory_id + timestamp | Permanent |
| `CF_TOOL_CALL_INDEX` | Tool call → memory mapping | tool_use_id | Session lifetime |
| `CF_CONSOLIDATION_RECOMMENDATIONS` | Persisted recommendations | recommendation_id | 90 days |
| `CF_ENTITY_PROVENANCE` | Entity extraction provenance | entity_id + memory_id | Permanent |
| `CF_EMBEDDING_REGISTRY` | Embedding version tracking | fingerprint_id | Permanent |

### 6.2 Summary of New Struct Fields

| Struct | New Fields |
|--------|-----------|
| **SourceMetadata** | `created_at`, `created_by`, `file_content_hash`, `file_modified_at`, `tool_use_id`, `mcp_request_id`, `hook_execution_timestamp_ms`, `derived_from`, `derivation_method` |
| **NodeMetadata** | `DeletionMetadata { deleted_by, deletion_reason, recovery_deadline }` |
| **CausalRelationship** | `llm_provenance: Option<LLMProvenance>`, `scanner_initial_score: Option<f32>` |
| **GraphEdge** | `discovery_provenance: Option<LLMProvenance>`, `created_by_agent: Option<String>` |
| **EntityLink** | `confidence: f32` |
| **CodeEntity** | `CodeGitMetadata { commit_hash, author, branch, commit_timestamp }` |

### 6.3 New MCP Tool Parameters

| Tool | New Parameters |
|------|---------------|
| All search tools | `include_provenance: bool` (default: false) |
| store_memory | `operator_id: Option<String>` |
| merge_concepts | `operator_id: Option<String>` |
| forget_concept | `reason: Option<String>`, `operator_id: Option<String>` |
| boost_importance | `reason: Option<String>`, `operator_id: Option<String>` |
| trigger_consolidation | `persist_recommendations: bool` (default: true) |

### 6.4 New MCP Tools

| Tool | Purpose |
|------|---------|
| `get_audit_trail` | Query audit log for a specific memory or time range |
| `get_merge_history` | Show merge lineage for a fingerprint |
| `get_provenance_chain` | Full provenance chain from embedding → chunk → source |

---

## 7. Priority Roadmap

### P0 - Critical (Implement First)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | CF_AUDIT_LOG + AuditRecord struct | 3 days | Foundation for all audit capability |
| 2 | Operator attribution (created_by) on all operations | 2 days | Enables user/agent tracking |
| 3 | LLMProvenance on CausalRelationship + GraphEdge | 2 days | Model version tracking |
| 4 | Search result provenance (include_provenance param) | 3 days | Retrieval transparency |
| 5 | Populate insight_annotation | 1 day | Explain retrieval decisions |

**P0 Total: ~11 days**

### P1 - High (Implement Second)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 6 | Entity source provenance (EntityProvenance struct) | 3 days | Entity traceability |
| 7 | Code git metadata capture | 2 days | Code change attribution |
| 8 | Permanent merge history (CF_MERGE_HISTORY) | 2 days | Merge lineage |
| 9 | Enhanced deletion tracking (reason, operator) | 1 day | Deletion audit |
| 10 | Importance change history | 1 day | Metadata audit |
| 11 | TransE score persistence | 1 day | Entity relationship quality |
| 12 | File hash persistence in SourceMetadata | 1 day | File change detection |

**P1 Total: ~11 days**

### P2 - Medium (Implement Third)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 13 | Tool call registry (tool_use_id tracking) | 2 days | Tool-to-memory linkage |
| 14 | Hook execution audit log | 2 days | Hook operation tracking |
| 15 | Consolidation recommendation persistence | 2 days | Consolidation audit |
| 16 | Embedding version registry | 2 days | Embedding freshness |
| 17 | AST parse error persistence | 1 day | Code quality tracking |
| 18 | Entity alias/synonym registry | 2 days | Entity deduplication |
| 19 | Scanner initial score in relationships | 1 day | Discovery traceability |

**P2 Total: ~12 days**

### P3 - Lower Priority (Nice to Have)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 20 | Steering reward history | 1 day | Edge quality tracking |
| 21 | Domain detection confidence | 1 day | Classification trust |
| 22 | Alternative LLM interpretations | 2 days | Analysis transparency |
| 23 | get_audit_trail MCP tool | 2 days | User-facing audit queries |
| 24 | get_provenance_chain MCP tool | 2 days | Full provenance visualization |
| 25 | Topic membership history | 3 days | Topic evolution tracking |

**P3 Total: ~11 days**

---

## 8. PRD Compliance Assessment

### PRD Provenance Mandate (from PRD_COMBINED.md):

> "Every piece of information CaseTrack returns MUST trace back to exact source. The provenance chain is: Embedding vector -> Chunk -> Provenance -> Source document. Every search result, EVERY MCP TOOL RESPONSE includes full provenance. There are ZERO ORPHANED VECTORS. If the provenance chain is broken, the data is useless."

### Current Compliance:

| Requirement | Status | Gap |
|-------------|--------|-----|
| Embedding → Chunk tracing | PASS | Content hash links fingerprint to content |
| Chunk → Source file tracing | PASS | SourceMetadata has file_path + line numbers |
| Every search result has provenance | PARTIAL FAIL | Source metadata returned, but retrieval provenance missing |
| Zero orphaned vectors | PASS | reconcile_files tool exists |
| Full provenance chain | FAIL | Breaks at: operations (merge/delete), model versions, retrieval decisions |

### After Implementation:

| Requirement | Expected Status |
|-------------|----------------|
| Embedding → Chunk tracing | PASS (+ model version tracking) |
| Chunk → Source file tracing | PASS (+ file hash + git metadata) |
| Every search result has provenance | PASS (with include_provenance) |
| Zero orphaned vectors | PASS |
| Full provenance chain | PASS (audit log + merge history + operator tracking) |
| Legal discovery compliance | PASS (permanent audit trail + operator attribution) |

---

## Appendix: Reference Architecture

### Ideal Provenance Chain (After Implementation)

```
User Query: "What databases work with Rust?"
  │
  ├─ Query Classification
  │   strategy: MultiSpace
  │   weight_profile: code_search
  │   detected_type: code
  │   auto_patterns: ["databases", "Rust"]
  │
  ├─ Search Execution
  │   E1 (semantic): 45 results in 3ms
  │   E7 (code): 32 results in 5ms  ← primary
  │   E11 (entity): 28 results in 2ms
  │   E13 (SPLADE): 67 results in 1ms
  │
  ├─ RRF Fusion
  │   candidate: "diesel ORM setup"
  │     found_by: [E1, E7, E11, E13] (4/13 = 31% consensus)
  │     primary_embedder: E7 (rank 0)
  │     scores: { E1: 0.85, E7: 0.92, E11: 0.88, E13: 0.79 }
  │     rrf_score: 0.0512
  │     insight: "E11 found this via entity match (Diesel=database ORM)"
  │
  ├─ Result Provenance
  │   memory_id: abc123
  │   source: MDFileChunk
  │   file_path: /docs/rust-databases.md:42-87
  │   chunk_index: 3/5
  │   created_at: 2026-01-15T10:30:00Z
  │   created_by: "session-xyz"
  │   file_hash: "sha256:a1b2c3..."
  │   embedding_versions: { E1: "v2.1", E7: "qodo-1.5b", E11: "kepler-v1" }
  │
  └─ Audit Trail
      memory abc123 created at 2026-01-15T10:30:00Z by file_watcher
      memory abc123 importance boosted +0.2 at 2026-01-20 by user (reason: "relevant to project")
      memory abc123 retrieved 47 times, last at 2026-02-06T14:22:00Z
```

---

## 9. Subagent Orchestration

### Overview

Implementation is automated through 11 specialized Claude Code subagents located in `.claude/agents/improvementplan/`. Each agent handles a specific workstream with defined inputs, outputs, and dependencies.

### Agent Registry

| Agent | Phase | Items | Model | Key Deliverables |
|-------|-------|-------|-------|------------------|
| `audit-log-architect` | 1.1 | #1 | opus | CF_AUDIT_LOG, AuditRecord struct, storage module |
| `operator-attribution` | 1.2 | #2 | sonnet | created_by/operator_id on SourceMetadata + MCP handlers |
| `llm-provenance-tracker` | 1.3 | #3 | sonnet | LLMProvenance struct on CausalRelationship + GraphEdge |
| `retrieval-transparency` | 2 | #4, #5 | opus | SearchResultProvenance, include_provenance param, insight annotations |
| `entity-provenance` | 3a | #6, #11 | sonnet | EntityProvenance, CF_ENTITY_PROVENANCE, TransE score persistence |
| `code-git-provenance` | 3b | #7, #12 | sonnet | CodeGitMetadata, file_content_hash in SourceMetadata |
| `lifecycle-provenance` | 4 | #8, #9, #10 | opus | CF_MERGE_HISTORY, DeletionMetadata, CF_IMPORTANCE_HISTORY |
| `hook-tool-provenance` | 5 | #13, #14, #15 | sonnet | CF_TOOL_CALL_INDEX, HookExecutionRecord, CF_CONSOLIDATION_RECOMMENDATIONS |
| `embedding-version-tracker` | 6 | #16 | sonnet | CF_EMBEDDING_REGISTRY, EmbeddingVersionRecord |
| `provenance-mcp-tools` | 7 | #23, #24 | sonnet | get_audit_trail, get_merge_history, get_provenance_chain MCP tools |
| `provenance-integration-tester` | 8 | All | opus | Integration tests for all provenance features |

### Orchestration Agent

`provenance-coordinator` (opus) manages the full execution sequence, dependency resolution, and cross-agent conflict management. Use it to run the complete plan.

### Execution Dependency Graph

```
Phase 1 (Sequential - Foundation):
  audit-log-architect ──> operator-attribution ──> llm-provenance-tracker
         │
         │ (CF_AUDIT_LOG must exist first)
         ▼
Phase 2 (Parallel with Phase 3, 6):
  retrieval-transparency ─────────────────────────────────────────────┐
                                                                      │
Phase 3 (Parallel pair):                                              │
  entity-provenance ──┐                                               │
  code-git-provenance ┘── (both modify SourceMetadata, run parallel)  │
                                                                      │
Phase 6 (Independent):                                                │
  embedding-version-tracker ──────────────────────────────────────────┤
                                                                      │
Phase 4 (After Phase 1):                                              │
  lifecycle-provenance ───── (extends operator-attribution work) ─────┤
                                                                      │
Phase 5 (After Phases 1-4):                                           │
  hook-tool-provenance ───── (extends SourceMetadata + audit log) ────┤
                                                                      │
Phase 7 (After ALL implementation):                                   │
  provenance-mcp-tools ───── (queries all CFs) ──────────────────────┤
                                                                      │
Phase 8 (Final):                                                      │
  provenance-integration-tester ──── (validates everything) ──────────┘
```

### Shared Resource Conflicts

These files are modified by multiple agents and require sequential execution or merge coordination:

| File | Modified By | Resolution |
|------|-------------|------------|
| `source_metadata.rs` | operator-attribution, code-git-provenance, hook-tool-provenance, lifecycle-provenance | Run sequentially within same struct |
| `column_families.rs` | audit-log-architect, entity-provenance, lifecycle-provenance, hook-tool-provenance, embedding-version-tracker | Each adds new CF - low conflict risk |
| `curation_tools.rs` | operator-attribution, lifecycle-provenance | lifecycle-provenance runs AFTER operator-attribution |
| `merge.rs` | operator-attribution, lifecycle-provenance, hook-tool-provenance | Run in Phase order |
| `memory_tools.rs` | operator-attribution, retrieval-transparency, hook-tool-provenance, embedding-version-tracker | Run in Phase order |

### Invocation

To execute the full plan:
```
Use the provenance-coordinator subagent to orchestrate the complete implementation.
```

To execute a single phase:
```
Use the {agent-name} subagent to implement Phase N.
Example: Use the audit-log-architect subagent to implement Phase 1.1.
```

### Progress Tracking

Each agent should update this section upon completion:

| Agent | Status | Completion Date | Notes |
|-------|--------|-----------------|-------|
| audit-log-architect | Pending | - | - |
| operator-attribution | Pending | - | - |
| llm-provenance-tracker | Pending | - | - |
| retrieval-transparency | Pending | - | - |
| entity-provenance | Pending | - | - |
| code-git-provenance | Pending | - | - |
| lifecycle-provenance | Pending | - | - |
| hook-tool-provenance | Pending | - | - |
| embedding-version-tracker | Pending | - | - |
| provenance-mcp-tools | Pending | - | - |
| provenance-integration-tester | Pending | - | - |

### Verification Gates

Between phases, run these verification steps:

1. **After Phase 1:** `cargo check -p context-graph-core -p context-graph-storage -p context-graph-mcp`
2. **After Phases 2-6:** `cargo check --workspace`
3. **After Phase 7:** `cargo check -p context-graph-mcp` (new tools compile)
4. **After Phase 8:** `cargo test --workspace` (all tests pass)
5. **Final:** `cargo build --release` (release build succeeds)

---

*End of Improvement Plan*
