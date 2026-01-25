# E12 (ColBERT) and E13 (SPLADE) Manual Verification Plan

**Date:** 2026-01-25
**Purpose:** Full State Verification with Source of Truth Inspection

---

## 1. Source of Truth Identification

### 1.1 E12 (ColBERT) Storage Locations

| Data | Storage Location | Column Family | Key Format |
|------|------------------|---------------|------------|
| E12 Token Embeddings | RocksDB | `CF_E12_LATE_INTERACTION` | UUID (16 bytes) |
| Token Count per Memory | Value in CF | - | bincode Vec<Vec<f32>> |
| Token Dimension | Constant | - | 128D per token |

### 1.2 E13 (SPLADE) Storage Locations

| Data | Storage Location | Column Family | Key Format |
|------|------------------|---------------|------------|
| E13 Sparse Vectors | SemanticFingerprint | `fingerprints` | UUID (16 bytes) |
| Inverted Index | RocksDB | `CF_E13_SPLADE_INVERTED` | term_id (2 bytes) |
| Vocabulary Size | Constant | - | 30,522 (BERT vocab) |

### 1.3 Pipeline Execution Traces

| Data | Location | How to Verify |
|------|----------|---------------|
| Stage 1 E13 usage | Debug logs | `Stage 1: E13 SPLADE returned X candidates` |
| Stage 4 E12 usage | Debug logs | `E12 rerank: fusion=X, maxsim=Y, blended=Z` |
| Pipeline timing | Response metadata | `causal.colbertApplied` field |

---

## 2. Synthetic Test Data

### 2.1 Test Memory Set (5 memories with known relationships)

```
Memory 1: "Rust's ownership system prevents data races at compile time"
  - Expected: High E1 similarity to "Rust memory safety"
  - Expected: E13 SPLADE should index terms: rust, ownership, data, races, compile

Memory 2: "The Diesel ORM provides type-safe database access for Rust"
  - Expected: E11 should link "Diesel" to database concept
  - Expected: E13 SPLADE should index: diesel, orm, database, rust

Memory 3: "PostgreSQL is a powerful open-source relational database"
  - Expected: Direct E1 match for "database" queries
  - Expected: E13 SPLADE should index: postgresql, database, relational

Memory 4: "fn connect_db() { let conn = PgConnection::establish(&url)?; }"
  - Expected: E7 Code should recognize function pattern
  - Expected: E12 should provide token-level precision for "connect_db"

Memory 5: "Memory leaks cause system crashes in long-running applications"
  - Expected: E5 Causal should link "cause" relationship
  - Expected: E13 SPLADE should index: memory, leaks, crashes, cause
```

### 2.2 Test Queries

| Query | Expected Top Results | E12 Value | E13 Value |
|-------|---------------------|-----------|-----------|
| "Rust memory safety" | Memory 1, 2 | Rerank for precision | Stage 1 recall |
| "database" | Memory 2, 3 | Rerank differentiators | Exact term match |
| "connect_db function" | Memory 4 | Token-level match | Code tokens |
| "what causes crashes" | Memory 5 | Rerank causal | "cause", "crashes" |

---

## 3. Verification Steps

### Phase 1: Memory Storage Verification
1. Store each synthetic memory via `store_memory` tool
2. Query RocksDB to verify:
   - Fingerprint exists in `fingerprints` CF
   - E12 tokens exist in `CF_E12_LATE_INTERACTION`
   - E13 terms exist in `CF_E13_SPLADE_INVERTED`

### Phase 2: Search Verification
1. Execute `search_graph` with `strategy: "pipeline"` and `enableRerank: true`
2. Verify response contains:
   - `colbertApplied: true`
   - Results in expected order
3. Check debug logs for E12/E13 stage execution

### Phase 3: Edge Case Verification
1. Empty query → Should return error
2. Very long query (1000+ chars) → Should handle gracefully
3. Query with only stop words → E13 should still work

---

## 4. Expected Evidence

### 4.1 E12 Token Storage Evidence
```
CF_E12_LATE_INTERACTION contains:
  - Key: <memory_uuid>
  - Value: Vec<Vec<f32>> with 20-50 tokens, each 128D
```

### 4.2 E13 Inverted Index Evidence
```
CF_E13_SPLADE_INVERTED contains:
  - Key: term_id (e.g., 7592 for "database")
  - Value: Vec<Uuid> of memories containing that term
```

### 4.3 Pipeline Execution Evidence
```
search_graph response:
{
  "causal": {
    "colbertApplied": true,
    ...
  },
  "searchStrategy": "pipeline"
}
```

---

## 5. Success Criteria

| Criterion | Verification Method | Pass Condition |
|-----------|---------------------|----------------|
| E12 tokens stored | RocksDB read | Non-empty Vec<Vec<f32>> for each memory |
| E13 terms indexed | Inverted index query | Terms map to correct memory UUIDs |
| Pipeline Stage 1 | Debug logs | "E13 SPLADE returned X candidates" |
| Pipeline Stage 4 | Response field | `colbertApplied: true` |
| Reranking improves results | Score comparison | MaxSim reranked scores differ from fusion |
