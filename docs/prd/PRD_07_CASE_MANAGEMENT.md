# PRD 07: Collection Management & Provenance

**Version**: 5.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Collection Model

```rust
/// A document collection containing related files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub collection_type: CollectionType,
    pub status: CollectionStatus,
    pub tags: Vec<String>,
    pub created_by: Option<String>,
    pub created_at: i64,     // Unix timestamp
    pub updated_at: i64,     // Unix timestamp
    pub stats: CollectionStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub document_count: u32,
    pub page_count: u32,
    pub chunk_count: u32,
    pub storage_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CollectionType {
    Business,
    Research,
    Project,
    Archive,
    Compliance,
    Financial,
    Technical,
    HR,
    Sales,
    Other,
}

// Derive FromStr via case-insensitive match on variant names. Default: Other.

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CollectionStatus {
    Active,
    Closed,
    Archived,
}
```

---

## 2. Collection Registry

Shared RocksDB instance indexing all collections. Key schema: `collection:{uuid}` -> bincode-serialized `Collection`.

```rust
pub struct CollectionRegistry {
    db: rocksdb::DB,        // registry.db in data_dir
    data_dir: PathBuf,
    active_collection: Option<Uuid>,
}

pub struct CreateCollectionParams {
    pub name: String,
    pub description: Option<String>,
    pub collection_type: Option<CollectionType>,
    pub tags: Option<Vec<String>>,
    pub created_by: Option<String>,
}

impl CollectionRegistry {
    /// Opens registry.db from data_dir
    pub fn open(data_dir: &Path) -> Result<Self>;

    /// Creates collection dir + originals subdir, initializes CollectionHandle DB,
    /// stores in registry, auto-switches active_collection to new collection
    pub fn create_collection(&mut self, params: CreateCollectionParams) -> Result<Collection>;

    /// Lookup by "collection:{id}" key. Error: CollectionNotFound
    pub fn get_collection(&self, collection_id: Uuid) -> Result<Collection>;

    /// Prefix scan "collection:", returns all collections sorted by updated_at DESC
    pub fn list_collections(&self) -> Result<Vec<Collection>>;

    /// Upsert collection metadata
    pub fn update_collection(&mut self, collection: &Collection) -> Result<()>;

    /// Deletes registry entry + entire collection directory. Clears active_collection if matched.
    pub fn delete_collection(&mut self, collection_id: Uuid) -> Result<()>;

    /// Validates collection exists, opens CollectionHandle, sets active_collection
    pub fn switch_collection(&mut self, collection_id: Uuid) -> Result<CollectionHandle>;

    pub fn active_collection_id(&self) -> Option<Uuid>;
    pub fn count_collections(&self) -> Result<u32>;
}
```

---

## 3. Collection Handle

Each collection has its own `collection.db` RocksDB with column families defined in `super::COLUMN_FAMILIES`.

Key schemas:
- Documents CF: `doc:{uuid}` -> bincode `DocumentMetadata`
- Chunks CF: `chunk:{uuid}` -> bincode `Chunk`
- Chunks CF index: `doc_chunks:{doc_uuid}:{sequence:06}` -> chunk UUID string

```rust
/// Handle to an open collection database
pub struct CollectionHandle {
    pub db: rocksdb::DB,
    pub collection_id: Uuid,       // Parsed from collection_dir directory name
    pub collection_dir: PathBuf,
}

impl CollectionHandle {
    /// Create collection.db with all column families (DB dropped after init, reopened by open())
    pub fn initialize(collection_dir: &Path) -> Result<()>;

    /// Open existing collection.db. Error: CollectionDbOpenFailed
    pub fn open(collection_dir: &Path) -> Result<Self>;

    // --- Document Operations (all use "documents" CF) ---
    pub fn store_document(&self, doc: &DocumentMetadata) -> Result<()>;
    pub fn get_document(&self, doc_id: Uuid) -> Result<DocumentMetadata>;
    /// Prefix scan "doc:", sorted by ingested_at DESC
    pub fn list_documents(&self) -> Result<Vec<DocumentMetadata>>;
    /// Deletes doc metadata + all chunks via doc_chunks index + embeddings + provenance
    pub fn delete_document(&self, doc_id: Uuid) -> Result<()>;

    // --- Chunk Operations (all use "chunks" CF) ---
    /// Stores chunk + doc_chunks index entry (keyed by doc_id + zero-padded sequence)
    pub fn store_chunk(&self, chunk: &Chunk) -> Result<()>;
    pub fn get_chunk(&self, chunk_id: Uuid) -> Result<Chunk>;
}
```

---

## 4. Provenance System (THE MOST IMPORTANT SYSTEM IN CASETRACK)

### 4.1 Provenance Model

```
PROVENANCE IS NON-NEGOTIABLE
=================================================================================

Every piece of information CaseTrack stores or returns MUST trace back to:
  1. The SOURCE FILE (file path + filename on disk)
  2. The exact LOCATION (page, paragraph, line, character offsets)
  3. The EXTRACTION METHOD (Native text, OCR, Hybrid)
  4. TIMESTAMPS (when created, when last embedded)

This applies to:
  - Every text chunk
  - Every embedding vector (linked via chunk_id)
  - Every entity mention (stores chunk_id + char offsets)
  - Every reference record (stores chunk_id + document_id)
  - Every search result (includes full provenance)
  - Every MCP tool response that returns text

If the provenance chain is broken, the data is USELESS.
A search result without a source reference is worthless to a professional.
```

Every chunk tracks exactly where it came from:

```rust
/// EVERY chunk stores full provenance. This is THE MOST IMPORTANT DATA STRUCTURE
/// in CaseTrack. When the AI returns information, the user must know EXACTLY where
/// it came from -- which document, which file on disk, which page, which paragraph,
/// which line, which character range. Without provenance, the data is useless.
///
/// The Provenance chain: Embedding vector -> chunk_id -> ChunkData.provenance -> source file
/// This chain is NEVER broken. Every embedding, every entity mention, every reference,
/// every search result carries its Provenance. If you can't cite the source, you can't
/// return the information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    // === Source Document (WHERE did this come from?) ===
    /// UUID of the ingested document
    pub document_id: Uuid,
    /// Original filename ("Report.pdf") -- always stored, never empty
    pub document_name: String,
    /// Full filesystem path where the file was when ingested
    /// ("/Users/sarah/Projects/Alpha/Report.pdf")
    /// Used for: reindexing (re-reads the file), sync (detects changes), display
    pub document_path: Option<PathBuf>,

    // === Location in Document (EXACTLY where in the document?) ===
    /// Page number (1-indexed) -- which page of the PDF/DOCX/XLSX
    pub page: u32,
    /// First paragraph index included in this chunk (0-indexed within page)
    pub paragraph_start: u32,
    /// Last paragraph index included in this chunk
    pub paragraph_end: u32,
    /// First line number (1-indexed within page)
    pub line_start: u32,
    /// Last line number
    pub line_end: u32,

    // === Character Offsets (for exact highlighting and cursor positioning) ===
    /// Character offset from start of page -- pinpoints exactly where the text starts
    pub char_start: u64,
    /// Character offset end -- pinpoints exactly where the text ends
    pub char_end: u64,

    // === Extraction Metadata (HOW was the text obtained?) ===
    /// How the text was extracted from the original file
    pub extraction_method: ExtractionMethod,
    /// OCR confidence score (0.0-1.0) if extracted via OCR. Lets the AI warn when
    /// text may be unreliable ("This text was OCR'd with 72% confidence").
    pub ocr_confidence: Option<f32>,

    // === Chunk Position ===
    /// Sequential position of this chunk within the entire document (0-indexed)
    pub chunk_index: u32,

    // === Timestamps (WHEN was this data created/updated?) ===
    /// When this chunk was first created from the source document (Unix timestamp)
    pub created_at: i64,
    /// When the embedding vectors for this chunk were last computed (Unix timestamp)
    /// Updated on reindex. Lets the system detect stale embeddings.
    pub embedded_at: i64,
}

impl Provenance {
    /// Generate a source reference string
    pub fn cite(&self) -> String {
        let mut parts = vec![self.document_name.clone()];
        parts.push(format!("p. {}", self.page));

        if self.paragraph_start == self.paragraph_end {
            parts.push(format!("para. {}", self.paragraph_start));
        } else {
            parts.push(format!("paras. {}-{}", self.paragraph_start, self.paragraph_end));
        }

        if self.line_start > 0 {
            parts.push(format!("ll. {}-{}", self.line_start, self.line_end));
        }

        parts.join(", ")
    }

    /// Short reference for inline use
    pub fn cite_short(&self) -> String {
        format!("{}, p. {}",
            self.document_name.split('.').next().unwrap_or(&self.document_name),
            self.page
        )
    }
}
```

### 4.2 Search Results with Provenance

```rust
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub provenance: Provenance,
    pub citation: String,
    pub citation_short: String,
    pub context_before: Option<String>,
    pub context_after: Option<String>,
}

impl SearchResult {
    pub fn to_mcp_content(&self) -> serde_json::Value {
        json!({
            "text": self.text,
            "score": self.score,
            "citation": self.citation,
            "citation_short": self.citation_short,
            "source": {
                "document": self.provenance.document_name,
                "page": self.provenance.page,
                "paragraph_start": self.provenance.paragraph_start,
                "paragraph_end": self.provenance.paragraph_end,
                "lines": format!("{}-{}", self.provenance.line_start, self.provenance.line_end),
                "extraction_method": format!("{:?}", self.provenance.extraction_method),
                "ocr_confidence": self.provenance.ocr_confidence,
            },
            "context": {
                "before": self.context_before,
                "after": self.context_after,
            }
        })
    }
}
```

### 4.3 Context Window

Search results include surrounding chunks for comprehension. Uses the `doc_chunks` index to look up adjacent chunks by `sequence +/- window`.

```rust
impl CollectionHandle {
    /// Returns (before_text, after_text) by looking up adjacent chunks
    /// via doc_chunks:{doc_id}:{sequence +/- 1} index keys
    pub fn get_surrounding_context(
        &self,
        chunk: &Chunk,
        window: usize,
    ) -> Result<(Option<String>, Option<String>)>;
}
```

---

## 5. Collection Summary

Each collection maintains a summary structure that provides an at-a-glance overview of the collection's contents, automatically updated as documents are ingested or removed.

```rust
/// Per-collection summary providing an overview of all contents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSummary {
    pub collection_id: Uuid,

    // === Key Entities ===
    /// People, organizations, and other named entities mentioned across all documents
    pub entities: Vec<EntitySummary>,

    // === Key Dates & Timelines ===
    /// Important dates extracted from documents with context
    pub key_dates: Vec<DateEntry>,

    // === Top Topics/Themes ===
    /// Dominant themes identified across the collection
    pub top_topics: Vec<TopicSummary>,

    // === Document Statistics ===
    pub document_count: u32,
    pub total_pages: u32,
    pub total_chunks: u32,
    pub storage_bytes: u64,
    pub file_types: HashMap<String, u32>,  // e.g., {"pdf": 12, "docx": 5, "xlsx": 3}

    // === Entity Statistics ===
    pub unique_entity_count: u32,
    pub entity_type_counts: HashMap<String, u32>,  // e.g., {"person": 45, "org": 12}

    pub last_updated: i64,  // Unix timestamp
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    pub name: String,
    pub entity_type: String,        // "person", "organization", "location", etc.
    pub mention_count: u32,
    pub document_ids: Vec<Uuid>,    // Which documents mention this entity
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateEntry {
    pub date: String,               // ISO 8601
    pub context: String,            // What the date refers to
    pub document_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicSummary {
    pub label: String,
    pub chunk_count: u32,           // How many chunks belong to this topic
    pub representative_terms: Vec<String>,
}
```

---

## 6. Reference Network

The reference network is a graph of cross-document references within a collection. It enables navigation between related documents based on shared entities, semantic similarity, and explicit references.

```rust
/// Edge in the reference network connecting two documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceEdge {
    pub source_doc_id: Uuid,
    pub target_doc_id: Uuid,
    pub edge_type: ReferenceEdgeType,
    pub weight: f32,                // Strength of the reference (0.0-1.0)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReferenceEdgeType {
    SharedEntity,           // Both documents mention the same entity
    SemanticSimilarity,     // Documents contain semantically similar chunks
    ExplicitReference,      // One document explicitly references another
}
```

---

## 7. Knowledge Graph

Every collection maintains a knowledge graph linking chunks, documents, and entities with full provenance.

```
KNOWLEDGE GRAPH STRUCTURE
=================================================================================

  Nodes:
    - Document nodes (one per ingested file)
    - Chunk nodes (one per text chunk, linked to parent document)
    - Entity nodes (people, organizations, dates, etc. extracted from chunks)

  Edges:
    - Chunk-to-Document: Every chunk linked to its source document with full provenance
    - Entity-to-Chunk: Entity mention links with character offsets
    - Document-to-Document: Shared entities, semantic similarity, explicit references
    - Chunk-to-Chunk: Semantic similarity above threshold, co-reference

  Enables queries like:
    - "Show me all documents mentioning Company X"
    - "What other documents relate to this one?"
    - "Which entities appear across multiple documents?"
    - "Trace the provenance of this information back to the source"
```

```rust
/// Node in the collection's knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphNode {
    Document { id: Uuid, name: String },
    Chunk { id: Uuid, document_id: Uuid, text_preview: String },
    Entity { id: Uuid, name: String, entity_type: String },
}

/// Edge in the collection's knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: Uuid,
    pub target: Uuid,
    pub edge_type: GraphEdgeType,
    pub weight: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GraphEdgeType {
    ChunkToDocument,        // Chunk belongs to document (provenance link)
    EntityToChunk,          // Entity mentioned in chunk
    DocumentToDocument,     // Cross-document reference
    ChunkToChunk,           // Semantic similarity or co-reference
}
```

---

## 8. Collection Lifecycle

```
COLLECTION LIFECYCLE
=================================================================================

  create_collection("Project Alpha")
       |
       v
  [ACTIVE] -----> ingest_pdf, ingest_docx, ingest_xlsx, search_collection
       |
       |  close_collection()          reopen_collection()
       v                                   |
  [CLOSED] --------> (read-only) ---------+
       |
       |  archive_collection()
       v
  [ARCHIVED] -----> (read-only, not shown in default list)
       |
       |  delete_collection()
       v
  [DELETED] -----> collection directory removed from disk

Notes:
  - ACTIVE: Full read/write. Can ingest, search, modify.
  - CLOSED: Read-only. Search works. Cannot ingest new documents.
  - ARCHIVED: Same as closed but hidden from default list_collections.
  - DELETED: Completely removed. Not recoverable.
```

---

## 9. Collection Management via MCP Tools -- Operations Guide

This section is the definitive reference for how the AI (Claude) and the user manage collections, documents, embeddings, and databases through MCP tools. **Every operation below is exposed as an MCP tool** (see PRD 09 for full input/output schemas).

### 9.1 Isolation Guarantee

```
CRITICAL: DATA NEVER CROSSES COLLECTION BOUNDARIES
=================================================================================

- Each collection = its own RocksDB database on disk (separate files, separate directory)
- Embeddings from Collection A are in a DIFFERENT DATABASE FILE than Collection B
- Search operates within a SINGLE COLLECTION ONLY -- there is no cross-collection search
- Ingestion targets the ACTIVE COLLECTION ONLY -- documents go into exactly one collection
- Deleting a collection deletes ONLY that collection's database, chunks, embeddings, and index
- No shared vector index, no shared embedding store, no shared anything

The AI MUST switch_collection before performing ANY operation on a different collection.
There is no way to accidentally mix data between collections.
```

### 9.2 Collection Lifecycle Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Create a collection | `create_collection` | Creates a new collection directory, initializes an empty RocksDB instance with all column families, registers in the collection registry, auto-switches to the new collection | New database on disk |
| List all collections | `list_collections` | Lists all collections with status, document count, chunk count, creation date | Read-only |
| Switch active collection | `switch_collection` | Changes which collection all subsequent operations target. Opens that collection's RocksDB database. | Changes active DB handle |
| Get collection details | `get_collection_info` | Shows all documents, total pages, total chunks, storage usage, embedder info | Read-only |
| Delete a collection | `delete_collection` | **Permanently removes**: collection directory, RocksDB database, ALL chunks, ALL embeddings, ALL indexes, ALL provenance records, optionally stored original files. Requires `confirm=true`. Not recoverable. | **Destroys entire database** |

### 9.3 Document Management Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Ingest one file | `ingest_document` | Reads file -> extracts text -> chunks into 2000-char segments -> embeds with all active models -> stores in active collection's DB | Adds chunks + embeddings to active collection |
| Ingest a folder | `ingest_folder` | Recursively walks directory -> ingests all supported files (PDF, DOCX, XLSX, TXT, etc.) -> skips already-ingested (SHA256) | Bulk add to active collection |
| Sync a folder | `sync_folder` | Compares disk vs DB -> ingests new files, reindexes changed files, optionally removes deleted | Add/update/remove in active collection |
| List documents | `list_documents` | Lists all documents in active collection with page count, chunk count, type | Read-only |
| Get document details | `get_document` | Shows one document's metadata, extraction method, chunk stats | Read-only |
| **Delete a document** | `delete_document` | **Removes from active collection**: document metadata, ALL chunks for that document, ALL embeddings for those chunks, ALL provenance records, ALL BM25 index entries. Requires `confirm=true`. | **Destroys document data** |

### 9.4 Embedding & Index Management Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Check index health | `get_index_status` | Per-document report: embedder coverage (2/4 vs 4/4), SHA256 staleness, missing source files | Read-only |
| Reindex one document | `reindex_document` | Deletes ALL old chunks + embeddings -> re-reads source file -> re-chunks -> re-embeds -> rebuilds BM25 entries. Option: `reparse=false` keeps chunks, only rebuilds embeddings. | **Replaces** old embeddings with fresh ones |
| Reindex entire collection | `reindex_collection` | Full rebuild of every document in the collection. Option: `skip_unchanged=true` only touches stale documents. Requires `confirm=true`. | **Replaces** all embeddings in collection |
| Get chunk provenance | `get_chunk` | Retrieves one chunk with full text and provenance (file, page, paragraph, line, char offsets) | Read-only |
| List document chunks | `get_document_chunks` | Lists all chunks in a document with their provenance | Read-only |
| Get surrounding context | `get_source_context` | Gets the chunks before/after a given chunk for context | Read-only |

### 9.5 Folder Watch & Auto-Sync Operations (MCP Tools)

| Operation | MCP Tool | What It Does | Data Impact |
|-----------|----------|-------------|-------------|
| Watch a folder | `watch_folder` | Starts OS-level file monitoring. New/modified/deleted files automatically trigger ingestion/reindex/removal in the target collection. | Automatic ongoing changes |
| Stop watching | `unwatch_folder` | Stops auto-sync. Existing collection data is untouched. | No data change |
| List watches | `list_watches` | Shows all active watches, their schedule, last sync, health status | Read-only |
| Change schedule | `set_sync_schedule` | Changes how often a watch syncs (on_change, hourly, daily, manual) | No data change |

### 9.6 Typical AI Workflow

```
User: "New collection for Project Alpha. Docs in ~/Projects/Alpha/"

Claude:
  1. create_collection("Project Alpha", collection_type="project")  -> isolated DB, auto-switched
  2. ingest_folder("~/Projects/Alpha/", recursive=true)             -> chunks + embeds all files
  3. watch_folder("~/Projects/Alpha/", schedule="on_change")        -> auto-sync future changes

User: "Search for customer retention strategy"
  4. search_collection("customer retention strategy", top_k=5)      -> results with full provenance

User: "Switch to Q3 Reports collection and search revenue figures"
  5. switch_collection("Q3 Reports")                                -> separate DB, Alpha inaccessible
  6. search_collection("revenue figures")                           -> Q3-only results

Key invariant: delete_collection/delete_document/reindex always cascade through
chunks -> embeddings -> provenance -> BM25 entries. Original source files on
disk are NEVER removed. See PRD 09 for full tool schemas.
```

---

*CaseTrack PRD v5.0.0 -- Document 7 of 10*
