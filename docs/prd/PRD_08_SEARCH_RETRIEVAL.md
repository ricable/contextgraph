# PRD 08: Search & Retrieval

**Version**: 5.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. 3-Stage Search Pipeline

```
+-----------------------------------------------------------------------+
|                        3-STAGE SEARCH PIPELINE                         |
+-----------------------------------------------------------------------+
|                                                                       |
|  Query: "What does the report say about customer retention?"          |
|                                                                       |
|  +---------------------------------------------------------------+   |
|  | STAGE 1: BM25 RECALL                                  [<5ms]   |   |
|  |                                                                |   |
|  | - E13 inverted index lookup                                   |   |
|  | - Terms: "report", "customer", "retention"                    |   |
|  | - Fast lexical matching                                       |   |
|  |                                                                |   |
|  | Output: 500 candidate chunks                                  |   |
|  +---------------------------------------------------------------+   |
|                              |                                        |
|                              v                                        |
|  +---------------------------------------------------------------+   |
|  | STAGE 2: SEMANTIC RANKING                             [<80ms]  |   |
|  |                                                                |   |
|  | - E1: Semantic similarity (384D dense cosine)                 |   |
|  | - E6: Keyword expansion (sparse dot product)                  |   |
|  | - Score fusion via Reciprocal Rank Fusion (RRF)               |   |
|  |                                                                |   |
|  | Output: 100 candidates, ranked                                |   |
|  +---------------------------------------------------------------+   |
|                              |                                        |
|                              v                                        |
|  +---------------------------------------------------------------+   |
|  | STAGE 3: COLBERT RERANK (PRO TIER ONLY)              [<100ms] |   |
|  |                                                                |   |
|  | - E12: Token-level MaxSim scoring                             |   |
|  | - Ensures exact phrase matches rank highest                   |   |
|  | - "customer retention" > "retention of the customer"          |   |
|  |                                                                |   |
|  | Output: Top K results with provenance                         |   |
|  +---------------------------------------------------------------+   |
|                                                                       |
|  LATENCY TARGETS                                                      |
|  ----------------                                                     |
|  Free tier (Stages 1-2):  <100ms                                     |
|  Pro tier (Stages 1-3):   <200ms                                     |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## 2. Search Engine Implementation

```rust
pub struct SearchEngine {
    embedder: Arc<EmbeddingEngine>,
    tier: LicenseTier,
}

impl SearchEngine {
    pub fn search(
        &self,
        collection: &CollectionHandle,
        query: &str,
        top_k: usize,
        document_filter: Option<Uuid>,
    ) -> Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();

        // Stage 1: BM25 recall
        let bm25_candidates = self.bm25_recall(collection, query, 500, document_filter)?;

        if bm25_candidates.is_empty() {
            return Ok(vec![]);
        }

        // Stage 2: Semantic ranking
        let query_e1 = self.embedder.embed_query(query, EmbedderId::E1)?;
        let query_e6 = self.embedder.embed_query(query, EmbedderId::E6)?;

        let mut scored: Vec<(Uuid, f32)> = bm25_candidates
            .iter()
            .map(|chunk_id| {
                let e1_score = self.score_dense(collection, "e1", chunk_id, &query_e1)?;
                let e6_score = self.score_sparse(collection, "e6", chunk_id, &query_e6)?;

                let rrf = rrf_fusion(&[
                    (e1_score, 1.0),   // E1: weight 1.0
                    (e6_score, 0.8),   // E6: weight 0.8
                ]);

                Ok((*chunk_id, rrf))
            })
            .collect::<Result<Vec<_>>>()?;

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(100);

        // Stage 3: ColBERT rerank (Pro only)
        if self.tier.is_pro() {
            scored = self.colbert_rerank(collection, query, scored)?;
        }

        // Build results with provenance
        let results: Vec<SearchResult> = scored
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, score)| self.build_result(collection, chunk_id, score))
            .collect::<Result<Vec<_>>>()?;

        let elapsed = start.elapsed();
        tracing::info!(
            "Search completed: {} results in {}ms (query: '{}')",
            results.len(),
            elapsed.as_millis(),
            query
        );

        Ok(results)
    }

    fn build_result(
        &self,
        collection: &CollectionHandle,
        chunk_id: Uuid,
        score: f32,
    ) -> Result<SearchResult> {
        let chunk = collection.get_chunk(chunk_id)?;
        let (ctx_before, ctx_after) = collection.get_surrounding_context(&chunk, 1)?;

        Ok(SearchResult {
            text: chunk.text,
            score,
            provenance: chunk.provenance.clone(),
            citation: chunk.provenance.cite(),
            citation_short: chunk.provenance.cite_short(),
            context_before: ctx_before,
            context_after: ctx_after,
        })
    }
}
```

---

## 3. BM25 Implementation

Standard BM25 with `k1=1.2, b=0.75`. Stored in `bm25_index` column family.

**Key schema**: `term:{token}` -> bincode `PostingList`, `stats` -> bincode `Bm25Stats`

**Tokenization**: lowercase, split on non-alphanumeric (preserving apostrophes), filter stopwords and single-char tokens.

```rust
pub struct Bm25Index;

impl Bm25Index {
    /// Tokenize query -> lookup postings per term -> accumulate BM25 scores
    /// per chunk -> apply optional document_filter -> return top `limit` chunk IDs
    pub fn search(collection: &CollectionHandle, query: &str, limit: usize,
                  document_filter: Option<Uuid>) -> Result<Vec<Uuid>>;

    /// Tokenize chunk text -> upsert PostingList per term -> update Bm25Stats
    pub fn index_chunk(collection: &CollectionHandle, chunk: &Chunk) -> Result<()>;
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Bm25Stats {
    pub total_docs: u32,
    pub total_tokens: u64,
    pub avg_doc_length: f32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PostingList {
    pub doc_freq: u32,
    pub entries: Vec<PostingEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PostingEntry {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub term_freq: u32,
    pub doc_length: u32,
}
```

---

## 4. Reciprocal Rank Fusion (RRF)

```rust
/// Combine scores from multiple embedders using RRF
/// Each (score, weight) pair represents one embedder's score and its importance
pub fn rrf_fusion(scored_weights: &[(f32, f32)]) -> f32 {
    const K: f32 = 60.0;

    scored_weights
        .iter()
        .map(|(score, weight)| {
            if *score <= 0.0 {
                0.0
            } else {
                // Convert similarity score to rank-like value, then apply RRF
                weight / (K + (1.0 / score))
            }
        })
        .sum()
}

/// RRF constant. Higher K smooths out rank differences.
const RRF_K: f32 = 60.0;
}
```

---

## 5. ColBERT Reranking (Stage 3)

```rust
impl SearchEngine {
    fn colbert_rerank(
        &self,
        collection: &CollectionHandle,
        query: &str,
        candidates: Vec<(Uuid, f32)>,
    ) -> Result<Vec<(Uuid, f32)>> {
        // Embed query at token level
        let query_tokens = self.embedder.embed_query(query, EmbedderId::E12)?;
        let query_vecs = match query_tokens {
            QueryEmbedding::Token(t) => t,
            _ => unreachable!(),
        };

        let mut reranked: Vec<(Uuid, f32)> = candidates
            .into_iter()
            .map(|(chunk_id, base_score)| {
                // Load chunk's token embeddings
                let chunk_tokens = self.load_token_embeddings(collection, &chunk_id)?;

                // MaxSim: for each query token, find max similarity to any chunk token
                let maxsim_score = query_vecs.vectors.iter()
                    .map(|q_vec| {
                        chunk_tokens.vectors.iter()
                            .map(|c_vec| cosine_similarity(q_vec, c_vec))
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .sum::<f32>() / query_vecs.vectors.len() as f32;

                // Blend ColBERT score with previous ranking
                let final_score = base_score * 0.4 + maxsim_score * 0.6;
                Ok((chunk_id, final_score))
            })
            .collect::<Result<Vec<_>>>()?;

        reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(reranked)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenEmbeddings {
    pub vectors: Vec<Vec<f32>>,  // One 64D vector per token
    pub token_count: usize,
}
```

---

## 6. Knowledge Graph Integration in Search

After vector search returns chunks, results can optionally be expanded via the collection's knowledge graph to surface related content the user did not directly query.

```
KNOWLEDGE GRAPH EXPANSION (POST-RETRIEVAL)
=================================================================================

  1. Vector search returns top K chunks (from Stages 1-3)
  2. For each result chunk:
     a. Look up entities mentioned in that chunk
     b. Find other chunks/documents sharing those entities -> "Related documents"
     c. Traverse chunk-to-chunk edges (semantic similarity, co-reference) -> "Related chunks"
  3. Deduplicate and rank expanded results by graph edge weight
  4. Return expanded results alongside primary results

  Enables:
    - "Related documents" via entity overlap
    - "Related chunks" via graph edges
    - Cross-document discovery without explicit search terms
```

```rust
impl SearchEngine {
    /// Expand search results via knowledge graph edges
    pub fn expand_via_graph(
        &self,
        collection: &CollectionHandle,
        results: &[SearchResult],
        max_expansions: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut expanded = Vec::new();

        for result in results {
            // Find entities in this chunk
            let entities = collection.get_chunk_entities(result.provenance.document_id)?;

            // Find other documents mentioning the same entities
            let related_docs = collection.find_documents_by_entities(&entities)?;

            // Find chunks connected via graph edges
            let related_chunks = collection.get_related_chunks(
                result.provenance.document_id,
                max_expansions,
            )?;

            for chunk in related_chunks {
                expanded.push(self.build_result(collection, chunk.id, chunk.edge_weight)?);
            }
        }

        // Deduplicate by chunk_id
        expanded.dedup_by_key(|r| r.provenance.document_id);
        expanded.truncate(max_expansions);

        Ok(expanded)
    }
}
```

---

## 7. Search Response Format (Canonical)

This is the canonical MCP response format for `search_collection` (also referenced by PRD 09).
Document-scoped search uses the same pipeline via the `document_filter` parameter on `SearchEngine::search`.

Every search result includes full provenance: file path, document name, page, paragraph, line, and character offsets.

```json
{
  "query": "customer retention strategy",
  "collection": "Project Alpha",
  "results_count": 5,
  "search_time_ms": 87,
  "tier": "pro",
  "stages_used": ["bm25", "semantic", "colbert"],
  "results": [
    {
      "text": "The recommended customer retention strategy focuses on quarterly business reviews and proactive account management...",
      "score": 0.94,
      "citation": "Q3_Report.pdf, p. 12, para. 8",
      "citation_short": "Q3_Report, p. 12",
      "source": {
        "document": "Q3_Report.pdf",
        "document_path": "/Users/sarah/Projects/Alpha/originals/Q3_Report.pdf",
        "document_id": "abc-123",
        "chunk_id": "chunk-456",
        "chunk_index": 14,
        "page": 12,
        "paragraph_start": 8,
        "paragraph_end": 8,
        "line_start": 1,
        "line_end": 4,
        "char_start": 24580,
        "char_end": 26580,
        "extraction_method": "Native",
        "ocr_confidence": null,
        "chunk_created_at": "2026-01-15T14:30:00Z",
        "chunk_embedded_at": "2026-01-15T14:30:12Z",
        "document_ingested_at": "2026-01-15T14:29:48Z"
      },
      "context": {
        "before": "...the previous paragraph text...",
        "after": "...the next paragraph text..."
      }
    }
  ]
}
```

---

*CaseTrack PRD v5.0.0 -- Document 8 of 10*
