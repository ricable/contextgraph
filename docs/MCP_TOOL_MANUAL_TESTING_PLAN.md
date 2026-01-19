# MCP Tool Manual Testing Plan

## Overview

This document outlines a comprehensive manual testing plan for all 14 MCP tools in the Context Graph system. The primary focus is on:

1. **Zero Detection**: Identifying unexpected 0's in responses that may indicate bugs vs. legitimate empty states
2. **Outlier Detection**: Finding anomalous values that suggest calculation errors
3. **Synthetic Data Requirements**: Determining when test data ingestion is necessary
4. **Database State Verification**: Confirming actual stored data matches expectations

---

## Pre-Testing Checklist

Before running tests, verify the following:

### 1. Database State Assessment
```bash
# Check if database exists and has data
ls -la ~/.context-graph/data/

# Check RocksDB directory size
du -sh ~/.context-graph/data/rocksdb/
```

### 2. MCP Server Status
```bash
# Verify MCP server is running
ps aux | grep context-graph-mcp

# Check server logs
tail -f /tmp/context-graph-mcp.log
```

### 3. Initial Memetic Status
Call `get_memetic_status` FIRST to establish baseline:
- Record `node_count` - if 0, synthetic data ingestion is REQUIRED
- Record initial `entropy`, `coherence`, `learning_score`

---

## Testing Decision Tree

```
Is node_count == 0?
  YES -> STOP: Ingest synthetic data first (see Appendix A)
  NO  -> Continue testing

For each zero value encountered:
  1. Is this field expected to be zero given current state?
     - entropy=0 with 0 nodes -> EXPECTED
     - entropy=0 with 100+ nodes -> BUG (should have variation)
  2. Does the database have corresponding data?
     - Use search_graph to verify content exists
  3. Is this a calculation error or empty state?
     - Check related fields for consistency
```

---

## Tool-by-Tool Testing Plan

### Category 1: Core Tools (5)

---

#### 1.1 get_memetic_status

**Purpose**: System health check - run this FIRST and LAST for every test session.

**Call**:
```json
{
  "name": "get_memetic_status",
  "arguments": {}
}
```

**Expected Fields & Zero Analysis**:

| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `node_count` | integer | Fresh database | Never (if data was ingested) |
| `entropy` | float [0,1] | 0-1 nodes exist | 10+ diverse nodes exist |
| `coherence` | float [0,1] | Highly diverse content | Never with homogeneous content |
| `learning_score` | float [0,1] | No recent learning | Active session with new memories |

**Outlier Detection**:
- `entropy` consistently at 0.0 or 1.0 -> Check UTL processor
- `learning_score` > 0 but `node_count` unchanged -> Possible storage failure
- All metrics exactly 0.5 -> Possible default value bug

**Test Steps**:
1. Call with empty database -> Verify `node_count=0`, metrics near 0
2. Inject 5 memories -> Call again -> Verify `node_count=5`, metrics non-zero
3. Inject 50 diverse memories -> Verify `entropy` > 0.3

---

#### 1.2 inject_context

**Purpose**: Store content with UTL processing.

**Call**:
```json
{
  "name": "inject_context",
  "arguments": {
    "content": "Test content about authentication middleware",
    "rationale": "Testing injection flow",
    "importance": 0.7,
    "modality": "text"
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `fingerprint_id` | UUID | Never | N/A (should always exist) |
| `utl_metrics.entropy` | float | Identical content | Unique content |
| `utl_metrics.coherence` | float | No prior context | Rich context exists |

**Test Steps**:
1. Inject with minimum params (content, rationale only)
2. Inject with all params specified
3. Inject duplicate content -> Should show low entropy in UTL metrics
4. Inject unique content -> Should show higher entropy

**Zero Verification**:
- After injection, call `search_graph` with same content
- Result count should be >= 1
- If 0 results, check storage layer logs

---

#### 1.3 store_memory

**Purpose**: Direct storage without UTL processing.

**Call**:
```json
{
  "name": "store_memory",
  "arguments": {
    "content": "PostgreSQL connection string: host=localhost port=5432",
    "importance": 0.8,
    "modality": "text",
    "tags": ["database", "config"]
  }
}
```

**Test Steps**:
1. Store memory with all fields
2. Store memory with minimal fields (content only)
3. Verify with `search_graph` immediately after
4. Check `get_memetic_status` node_count increased by 1

**Zero Detection**:
- If `fingerprint_id` missing in response -> BUG
- If subsequent `search_graph` returns 0 -> Storage failed silently

---

#### 1.4 search_graph

**Purpose**: Semantic search across all 13 embedding spaces.

**Call**:
```json
{
  "name": "search_graph",
  "arguments": {
    "query": "authentication",
    "topK": 20,
    "minSimilarity": 0.3,
    "modality": "text",
    "includeContent": true
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `results.length` | integer | No matching content | Content known to exist |
| `similarity` | float [0,1] | Never (would not be returned) | N/A |
| `dominantEmbedder` | string | Never | N/A |

**Test Steps**:
1. Search empty database -> Should return empty array (not error)
2. Inject content about "machine learning"
3. Search "machine learning" -> Should return >= 1 result
4. Search with high minSimilarity (0.9) -> May return 0 (OK)
5. Search with minSimilarity=0.0, topK=100 -> Should return all memories

**Zero Analysis Protocol**:
```
IF results.length == 0:
  1. Check get_memetic_status().node_count
     - If 0: Expected (no data)
     - If >0: Potential bug
  2. Try query with minSimilarity=0.0
     - If still 0: Embedding or search bug
     - If >0: Original query too specific
  3. Try exact content that was injected
     - If 0: Critical search indexing bug
```

---

#### 1.5 trigger_consolidation

**Purpose**: Merge similar memories.

**Call**:
```json
{
  "name": "trigger_consolidation",
  "arguments": {
    "strategy": "similarity",
    "min_similarity": 0.85,
    "max_memories": 100
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `candidates_found` | integer | No similar memories | Many duplicates exist |
| `merged_count` | integer | No candidates | Candidates found but none merged |
| `memory_reduction` | integer | No merges | Merges occurred |

**Test Steps**:
1. Run on empty database -> All zeros expected
2. Inject 10 nearly-identical memories
3. Run consolidation -> `candidates_found` should be > 0
4. Verify `node_count` decreased after merge

**Outlier Detection**:
- `merged_count` > `candidates_found` -> BUG
- `memory_reduction` negative -> BUG
- Consolidation takes > 30s on small dataset -> Performance bug

---

### Category 2: Topic Tools (4)

---

#### 2.1 get_topic_portfolio

**Purpose**: View emergent topics from multi-space clustering.

**Call**:
```json
{
  "name": "get_topic_portfolio",
  "arguments": {
    "format": "verbose"
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `topics.length` | integer | < 3 memories OR low weighted_agreement | 50+ diverse memories |
| `topic.confidence` | float [0,1] | Never < 0.29 (threshold=2.5/8.5) | N/A |
| `topic.memory_count` | integer | Never | N/A |

**Test Steps**:
1. Run with 0 memories -> Empty topics array (expected)
2. Run with 2 memories -> Empty topics array (expected, need >= 3)
3. Inject 20+ memories in 3 distinct domains (auth, database, frontend)
4. Run `detect_topics` first, then `get_topic_portfolio`
5. Should see 3 topics emerge

**Zero Topics Diagnosis**:
```
IF topics.length == 0 AND node_count >= 3:
  1. Run detect_topics with force=true
  2. Check get_topic_stability for issues
  3. Verify memories are in SEMANTIC categories (not just temporal)
  4. Check weighted_agreement in verbose format
     - If all < 2.5: Need more diverse semantic content
```

---

#### 2.2 get_topic_stability

**Purpose**: Portfolio health metrics including churn and entropy.

**Call**:
```json
{
  "name": "get_topic_stability",
  "arguments": {
    "hours": 24
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `churn_rate` | float [0,1] | Stable topics, no changes | Active topic changes occurring |
| `entropy` | float [0,1] | Single dominant topic | Many diverse topics |
| `phases.emerging` | integer | No new topics | New content being added |
| `phases.stable` | integer | New system | Mature topics exist |

**Test Steps**:
1. Check immediately after fresh start -> churn may be 0 (expected)
2. Inject diverse memories, run detect_topics multiple times
3. Check stability -> Should see non-zero metrics
4. Verify `dream_recommended` flag accuracy:
   - If entropy > 0.7 AND churn > 0.5 -> Should be true
   - Otherwise -> Should be false

**Outlier Detection**:
- churn_rate > 1.0 -> BUG (should be clamped)
- All phases at 0 but topics exist -> Phase classification bug

---

#### 2.3 detect_topics

**Purpose**: Force HDBSCAN clustering recalculation.

**Call**:
```json
{
  "name": "detect_topics",
  "arguments": {
    "force": true
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `topics_detected` | integer | < 3 memories | 50+ diverse memories |
| `clustering_time_ms` | integer | Never (always > 0) | N/A |
| `memories_processed` | integer | Empty database | Non-empty database |

**Test Steps**:
1. Run on empty database -> `memories_processed=0` (expected)
2. Inject 3 similar memories -> May still detect 0 topics (expected)
3. Inject 30 memories across 5 domains -> Should detect >= 2 topics
4. Run with `force=false` immediately after -> Should skip (recently computed)
5. Run with `force=true` -> Should recompute

**Zero Detection**:
- If `memories_processed=0` but `node_count > 0` -> Memory loading bug
- If `topics_detected=0` with 50+ diverse memories -> Clustering parameters may be too strict

---

#### 2.4 get_divergence_alerts

**Purpose**: Detect current work diverging from recent patterns.

**Call**:
```json
{
  "name": "get_divergence_alerts",
  "arguments": {
    "lookback_hours": 4
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `alerts.length` | integer | Consistent work patterns | Sharp topic change occurred |
| `divergence_score` | float [0,1] | Similar content | Very different content |

**Test Steps**:
1. Inject 10 memories about "frontend development"
2. Check divergence -> Should be low/no alerts
3. Inject 5 memories about "database optimization" (different domain)
4. Check divergence -> Should see alerts for semantic shift

**Important**: Only SEMANTIC embedders (E1, E5, E6, E7, E10, E12, E13) contribute.
- Temporal embedders (E2, E3, E4) are EXCLUDED per AP-62, AP-63.

---

### Category 3: Curation Tools (2)

---

#### 3.1 boost_importance

**Purpose**: Adjust memory importance scores.

**Call**:
```json
{
  "name": "boost_importance",
  "arguments": {
    "node_id": "<UUID>",
    "delta": 0.3
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `old_importance` | float [0,1] | Original was 0 (rare) | N/A |
| `new_importance` | float [0,1] | Delta brought to 0 | Positive delta from non-zero |
| `delta_applied` | float | delta=0 passed | Non-zero delta passed |

**Test Steps**:
1. Inject memory with importance=0.5
2. Boost by +0.3 -> Should be 0.8
3. Boost by +0.5 -> Should be clamped to 1.0
4. Boost by -1.5 -> Should be clamped to 0.0
5. Verify clamping per BR-MCP-002

**Outlier Detection**:
- `new_importance` > 1.0 or < 0.0 -> Clamping bug
- `new_importance` != old + delta (within clamp bounds) -> Calculation bug

---

#### 3.2 forget_concept

**Purpose**: Soft-delete with 30-day recovery.

**Call**:
```json
{
  "name": "forget_concept",
  "arguments": {
    "node_id": "<UUID>",
    "soft_delete": true
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `deleted_at` | timestamp | Never (always set) | N/A |
| `recovery_window_days` | integer | soft_delete=false | soft_delete=true (should be 30) |

**Test Steps**:
1. Inject a memory, record UUID
2. Forget with soft_delete=true
3. Search for that content -> Should NOT appear in results
4. Check `get_memetic_status` -> `node_count` should decrease by 1
5. (Advanced) Check raw storage for soft-delete marker

**Zero Detection**:
- If `node_count` unchanged after forget -> Deletion failed silently
- If content still appears in search -> Soft delete not filtering correctly

---

### Category 4: Dream Tools (2)

---

#### 4.1 trigger_dream

**Purpose**: Run NREM/REM consolidation cycles.

**Call**:
```json
{
  "name": "trigger_dream",
  "arguments": {
    "blocking": true,
    "dry_run": false,
    "max_duration_secs": 300
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `nrem_connections_strengthened` | integer | No high-importance patterns | 100+ memories with importance > 0.7 |
| `rem_blind_spots_discovered` | integer | Well-covered topic space | Sparse topic coverage |
| `dream_duration_ms` | integer | Never (always > 0) | N/A |

**Test Steps**:
1. Verify trigger conditions: entropy > 0.7 AND churn > 0.5
2. Run with `dry_run=true` first -> No modifications, get projections
3. Run with `blocking=false` -> Get dream_id
4. Poll with `get_dream_status` until complete
5. Check topic stability before/after

**Trigger Condition Verification**:
```bash
# Check if dream is appropriate
1. Call get_topic_stability
2. IF entropy > 0.7 AND churn > 0.5:
     Dream is recommended (dream_recommended=true expected)
   ELSE:
     Dream may not be beneficial
```

---

#### 4.2 get_dream_status

**Purpose**: Monitor running/completed dream cycles.

**Call**:
```json
{
  "name": "get_dream_status",
  "arguments": {
    "dream_id": "<UUID>"
  }
}
```

**Expected Response Fields**:
| Field | Type | Zero OK When | Zero = BUG When |
|-------|------|--------------|-----------------|
| `progress_percent` | integer [0,100] | Just started | Dream running for > 1 min |
| `elapsed_ms` | integer | Never (always > 0 once started) | N/A |
| `remaining_ms` | integer | Dream completed | Dream in progress |

**Test Steps**:
1. Trigger dream with `blocking=false`
2. Immediately call `get_dream_status` -> Should show progress
3. Poll every 5 seconds until `status=completed`
4. Call without `dream_id` -> Should return most recent

---

## Synthetic Data Ingestion Protocol (Appendix A)

If `get_memetic_status().node_count == 0`, run this synthetic data ingestion:

### Minimum Dataset for Testing

```json
// Category 1: Authentication (5 memories)
[
  {"content": "JWT tokens expire after 24 hours for security", "modality": "text", "importance": 0.8},
  {"content": "OAuth2 refresh tokens stored in httpOnly cookies", "modality": "text", "importance": 0.7},
  {"content": "User authentication middleware checks Bearer token", "modality": "code", "importance": 0.9},
  {"content": "Session invalidation on password change", "modality": "text", "importance": 0.6},
  {"content": "Two-factor auth required for admin roles", "modality": "text", "importance": 0.8}
]

// Category 2: Database (5 memories)
[
  {"content": "PostgreSQL connection pool max 20 connections", "modality": "text", "importance": 0.7},
  {"content": "Redis cache TTL set to 300 seconds for sessions", "modality": "text", "importance": 0.6},
  {"content": "Database migrations run on deploy via Flyway", "modality": "text", "importance": 0.5},
  {"content": "Query optimization using EXPLAIN ANALYZE", "modality": "code", "importance": 0.8},
  {"content": "Indexing strategy for user_id foreign keys", "modality": "text", "importance": 0.7}
]

// Category 3: Frontend (5 memories)
[
  {"content": "React component lifecycle for data fetching", "modality": "code", "importance": 0.6},
  {"content": "Tailwind CSS utility classes for responsive design", "modality": "text", "importance": 0.5},
  {"content": "State management via Redux Toolkit slices", "modality": "code", "importance": 0.8},
  {"content": "Form validation using Yup schema", "modality": "code", "importance": 0.7},
  {"content": "Error boundary components for graceful failures", "modality": "text", "importance": 0.7}
]

// Category 4: API Design (5 memories)
[
  {"content": "REST endpoints follow resource naming conventions", "modality": "text", "importance": 0.6},
  {"content": "GraphQL mutations for write operations", "modality": "code", "importance": 0.7},
  {"content": "Rate limiting at 100 requests per minute per user", "modality": "text", "importance": 0.8},
  {"content": "API versioning via URL path /api/v1/", "modality": "text", "importance": 0.5},
  {"content": "OpenAPI spec generated from code annotations", "modality": "text", "importance": 0.6}
]

// Category 5: DevOps (5 memories)
[
  {"content": "Docker containers deployed to Kubernetes cluster", "modality": "text", "importance": 0.7},
  {"content": "CI/CD pipeline runs tests before merge", "modality": "text", "importance": 0.8},
  {"content": "Prometheus metrics exposed on /metrics endpoint", "modality": "text", "importance": 0.6},
  {"content": "Log aggregation via Elasticsearch and Kibana", "modality": "text", "importance": 0.5},
  {"content": "Terraform modules for infrastructure as code", "modality": "code", "importance": 0.7}
]
```

### Ingestion Script

```bash
#!/bin/bash
# Ingest via MCP tools

for memory in "${MEMORIES[@]}"; do
  echo "Injecting: $memory"
  # Call inject_context via MCP
done

# Force topic detection after ingestion
# Call detect_topics with force=true
```

---

## Test Execution Checklist

### Phase 1: Baseline (Empty Database)
- [ ] `get_memetic_status` returns `node_count=0`
- [ ] `search_graph` returns empty results (not error)
- [ ] `get_topic_portfolio` returns empty topics
- [ ] `detect_topics` processes 0 memories

### Phase 2: After Synthetic Data (25 memories)
- [ ] `get_memetic_status` returns `node_count=25`
- [ ] `search_graph "authentication"` returns 3-5 results
- [ ] `detect_topics` with `force=true` finds topics
- [ ] `get_topic_portfolio` shows >= 2 topics

### Phase 3: Curation Operations
- [ ] `boost_importance` correctly modifies a memory
- [ ] `forget_concept` removes memory from search
- [ ] `trigger_consolidation` finds candidates (if duplicates exist)

### Phase 4: Dream Operations
- [ ] Check trigger conditions via `get_topic_stability`
- [ ] Run `trigger_dream` with `dry_run=true`
- [ ] Run actual dream cycle
- [ ] Verify topic stability improved

### Phase 5: Edge Cases
- [ ] Search with `minSimilarity=1.0` (may return 0)
- [ ] Boost importance beyond 1.0 (should clamp)
- [ ] Forget non-existent UUID (should error gracefully)
- [ ] Detect topics with < 3 memories (should return 0)

---

## Reporting Template

For each test run, document:

```markdown
## Test Run: [DATE]

### Environment
- Database state: [empty/seeded/production]
- Node count at start: [N]
- MCP server version: [X.Y.Z]

### Results Summary
| Tool | Expected | Actual | Status |
|------|----------|--------|--------|
| get_memetic_status | node_count=25 | node_count=25 | PASS |
| search_graph | results >= 1 | results = 3 | PASS |
| ... | ... | ... | ... |

### Zero Values Found
| Tool | Field | Value | Analysis |
|------|-------|-------|----------|
| get_topic_portfolio | topics.length | 0 | Expected: < 3 memories |

### Outliers Found
| Tool | Field | Value | Expected Range | Analysis |
|------|-------|-------|----------------|----------|
| get_topic_stability | churn_rate | 1.5 | [0.0, 1.0] | BUG: Not clamped |

### Actions Required
1. [Bug to file or fix]
2. [Data to ingest]
```

---

## Quick Reference: Zero Value Decision Matrix

| Tool | Field | 0 = OK When | 0 = INVESTIGATE When |
|------|-------|-------------|---------------------|
| get_memetic_status | node_count | Fresh DB | After successful inject |
| get_memetic_status | entropy | 0-1 nodes | 10+ diverse nodes |
| search_graph | results.length | No matches | Content known to exist |
| get_topic_portfolio | topics.length | < 3 nodes | 50+ diverse nodes |
| get_topic_stability | churn_rate | Stable system | Active changes |
| detect_topics | topics_detected | < 3 nodes | 30+ diverse nodes |
| trigger_consolidation | merged_count | No duplicates | Duplicates exist |
| trigger_dream | blind_spots | Well-covered | Sparse coverage |

---

## Constitution Compliance Checks

During testing, verify these rules:

- **AP-70**: Dream triggers only when entropy > 0.7 AND churn > 0.5
- **SEC-06**: Forget uses 30-day soft delete by default
- **BR-MCP-002**: Importance clamped to [0.0, 1.0]
- **ARCH-09**: Topic threshold is weighted_agreement >= 2.5
- **AP-62/AP-63**: Divergence uses SEMANTIC embedders only (not temporal)
