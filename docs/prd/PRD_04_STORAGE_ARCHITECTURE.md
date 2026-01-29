# PRD 04: Storage Architecture

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Core Principle: Everything on YOUR Machine

**CaseTrack stores ALL data locally on the user's device:**

- **Embedding models**: Downloaded once, stored in `~/Documents/CaseTrack/models/`
- **Vector embeddings**: Stored in RocksDB on your device
- **Document chunks**: Stored in RocksDB on your device
- **Collection databases**: Each collection is an isolated RocksDB instance
- **Original documents**: Optionally copied to your CaseTrack folder

**Nothing is sent to any cloud service. Ever.**

---

## 2. Directory Structure

```
~/Documents/CaseTrack/                       <-- All CaseTrack data lives here
|
|-- config.toml                              <-- User configuration (optional)
|-- watches.json                             <-- Folder watch registry (auto-sync)
|
|-- models/                                  <-- Embedding models (~400MB)
|   |-- bge-small-en-v1.5/                     Downloaded on first use
|   |   |-- model.onnx                         Cached permanently
|   |   +-- tokenizer.json                     No re-download needed
|   |-- splade-distil/
|   +-- colbert-small/
|
|-- registry.db/                             <-- Collection index (RocksDB)
|   +-- [collection metadata, schema version]
|
+-- collections/                             <-- Per-collection databases
    |
    |-- {collection-uuid-1}/                 <-- Collection "Project Alpha"
    |   |-- collection.db/                     (Isolated RocksDB instance)
    |   |   |-- documents     CF              Document metadata
    |   |   |-- chunks        CF              Text chunks (bincode)
    |   |   |-- embeddings    CF              All embedder vectors + chunk text + provenance
    |   |   |   |-- e1                        384D vectors
    |   |   |   |-- e6                        Sparse vectors
    |   |   |   +-- ...                       All active embedders
    |   |   |-- bm25_index    CF              Inverted index for keyword search
    |   |   +-- ...                           Additional column families
    |   +-- originals/                        Original files (optional copy)
    |       |-- Report.pdf
    |       |-- Summary.docx
    |       +-- Data.xlsx
    |
    |-- {collection-uuid-2}/                 <-- Collection "Q4 Analysis"
    |   +-- ...                                (Completely isolated)
    |
    +-- {collection-uuid-N}/                 <-- More collections...

CF = RocksDB Column Family
```

---

## 3. Storage Estimates

| Data Type | Size Per Unit | Notes |
|-----------|---------------|-------|
| Models (Free tier) | ~120MB total | E1 + E6 (one-time download) |
| Models (Pro tier) | ~230MB total | All 4 models (one-time download) |
| Registry DB | ~1MB | Scales with number of collections |
| Per document page (Free) | ~30KB | 3 embeddings + chunk text + provenance |
| Per document page (Pro) | ~50KB | 6 embeddings + chunk text + provenance |
| 100-page collection (Free) | ~3MB | |
| 100-page collection (Pro) | ~5MB | |
| 1000-page collection (Pro) | ~50MB | |
| BM25 index per 1000 chunks | ~2MB | Inverted index |

**Example total disk usage:**
- Free tier, 3 collections of 100 pages each: 165MB (models) + 9MB (data) = **~175MB**
- Pro tier, 10 collections of 500 pages each: 370MB (models) + 250MB (data) = **~620MB**

---

## 4. RocksDB Configuration

### 4.1 Why RocksDB

| Requirement | RocksDB | SQLite | LMDB |
|-------------|---------|--------|------|
| Embedded (no server) | Yes | Yes | Yes |
| Column families (namespacing) | Yes | No (tables) | No |
| Prefix iteration | Yes | No | Limited |
| Bulk write performance | Excellent | Good | Good |
| Concurrent reads | Excellent | Limited (WAL) | Excellent |
| Rust crate quality | Good (rust-rocksdb) | Good (rusqlite) | Fair |
| Per-collection isolation | Separate DB instances | Separate files | Separate files |

RocksDB was chosen for: column families (clean separation of data types), prefix iteration (efficient collection listing), and bulk write performance (ingestion throughput).

### 4.2 Column Family Schema

Each collection database uses these column families:

```rust
pub const COLUMN_FAMILIES: &[&str] = &[
    "documents",        // Document metadata
    "chunks",           // Text chunk content
    "embeddings",       // All embedder vectors + chunk text + provenance per chunk
    "bm25_index",       // Inverted index for BM25
    "metadata",         // Collection-level metadata, stats

    // === Context Graph (relationships between documents, chunks, entities) ===
    "entities",         // Extracted entities (person, organization, date, amount, location, concept)
    "entity_index",     // Entity -> chunk mentions index
    "references",       // Cross-document references (shared entities, citations, hyperlinks)
    "doc_graph",        // Document-to-document relationships (similarity, reference links)
    "chunk_graph",      // Chunk-to-chunk relationships (similarity edges, co-reference)
    "knowledge_graph",  // Entity-to-entity relationships, entity-to-chunk mappings
    "collection_map",   // Collection-level summary: key actors, dates, document categories
];
```

### 4.3 Key Schema

```rust
// === Registry DB Keys ===

// Collection listing
"collection:{uuid}"                          -> bincode<Collection>
"schema_version"                             -> u32 (current: 1)

// Folder watches (auto-sync)
"watch:{uuid}"                               -> bincode<FolderWatch>
"watch_collection:{collection_uuid}:{watch_uuid}" -> watch_uuid  (index: watches by collection)

// === Collection DB Keys (per column family) ===

// documents CF
"doc:{uuid}"                       -> bincode<DocumentMetadata>

// chunks CF
"chunk:{uuid}"                     -> bincode<ChunkData>
"doc_chunks:{doc_uuid}:{seq}"      -> chunk_uuid  (index: chunks by document)

// embeddings CF -- each chunk stores all embedder vectors alongside text and provenance
// chunk_id -> { text, provenance, e1_vector, e6_vector, e12_vector, bm25_terms }
"emb:{chunk_uuid}"                 -> bincode<ChunkEmbeddingRecord>

// Legacy per-embedder keys (supported for migration)
"e1:{chunk_uuid}"                  -> [f32; 384] as bytes
"e6:{chunk_uuid}"                  -> bincode<SparseVec>
"e12:{chunk_uuid}"                 -> bincode<TokenEmbeddings>

// bm25_index CF
"term:{term}"                      -> bincode<PostingList>
"doc_len:{doc_uuid}"               -> u32 (document length in tokens)
"stats"                            -> bincode<Bm25Stats> (avg doc length, total docs)

// metadata CF
"collection_info"                  -> bincode<Collection>
"stats"                            -> bincode<CollectionStats>

// === CONTEXT GRAPH COLUMN FAMILIES ===

// entities CF
"entity:{type}:{normalized_name}"  -> bincode<Entity>
// type = person | organization | date | amount | location | concept

// entity_index CF (bidirectional)
"ent_chunks:{entity_key}"          -> bincode<Vec<EntityMention>>
"chunk_ents:{chunk_uuid}"          -> bincode<Vec<EntityRef>>

// references CF (cross-document references)
"ref:{reference_key}"              -> bincode<ReferenceRecord>
"ref_chunks:{reference_key}"       -> bincode<Vec<ReferenceMention>>
"chunk_refs:{chunk_uuid}"          -> bincode<Vec<ReferenceRef>>

// doc_graph CF
"doc_sim:{doc_a}:{doc_b}"         -> f32 (cosine similarity)
"doc_refs:{source_doc}:{target_doc}" -> bincode<DocReference>
"doc_entities:{doc_uuid}"         -> bincode<Vec<EntityRef>>
"doc_category:{category}:{doc_uuid}" -> doc_uuid

// chunk_graph CF
"chunk_sim:{chunk_a}:{chunk_b}"   -> f32 (stored only when > 0.7)
"chunk_coref:{chunk_a}:{chunk_b}" -> bincode<CoReference>  (shared entity co-reference)
"chunk_seq:{doc_uuid}:{seq}"      -> chunk_uuid

// knowledge_graph CF
"kg_ent_rel:{entity_a}:{rel_type}:{entity_b}" -> bincode<EntityRelationship>
"kg_ent_chunks:{entity_key}"       -> bincode<Vec<Uuid>>  (entity-to-chunk mappings)
"kg_chunk_ents:{chunk_uuid}"       -> bincode<Vec<String>> (chunk-to-entity mappings)

// collection_map CF (rebuilt after ingestion)
"key_actors"                       -> bincode<Vec<KeyActor>>
"key_dates"                        -> bincode<Vec<KeyDate>>
"key_topics"                       -> bincode<Vec<Topic>>
"doc_categories"                   -> bincode<HashMap<String, Vec<Uuid>>>
"reference_stats"                  -> bincode<Vec<ReferenceStat>>
"entity_stats"                     -> bincode<Vec<EntityStat>>
```

### 4.4 RocksDB Tuning for Consumer Hardware

```rust
pub fn rocks_options() -> rocksdb::Options {
    let mut opts = rocksdb::Options::default();

    // Create column families if missing
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);

    // Memory budget: ~64MB per open database
    // (allows multiple collections open simultaneously on 8GB machines)
    let mut block_cache = rocksdb::Cache::new_lru_cache(32 * 1024 * 1024); // 32MB
    let mut table_opts = rocksdb::BlockBasedOptions::default();
    table_opts.set_block_cache(&block_cache);
    table_opts.set_block_size(16 * 1024); // 16KB blocks
    opts.set_block_based_table_factory(&table_opts);

    // Write buffer: 16MB (reduces write amplification)
    opts.set_write_buffer_size(16 * 1024 * 1024);
    opts.set_max_write_buffer_number(2);

    // Compression: LZ4 for speed, Zstd for bottom level
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);

    // Limit background threads (save CPU for embedding)
    opts.set_max_background_jobs(2);
    opts.increase_parallelism(2);

    opts
}
```

---

## 5. Serialization Format

### 5.1 Bincode for Structs

All Rust structs stored via `bincode` for fast, compact serialization:

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct ChunkData {
    pub id: Uuid,
    pub document_id: Uuid,
    pub text: String,
    pub sequence: u32,              // Position within document (0-indexed)
    pub char_count: u32,
    pub provenance: Provenance,     // Full source trace -- see Section 5.2 Provenance Chain
    pub created_at: i64,            // Unix timestamp
    pub embedded_at: i64,           // Unix timestamp: last embedding computation
    pub embedder_versions: Vec<String>, // e.g., ["e1", "e6"] for Free tier
}

#[derive(Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub id: Uuid,
    pub name: String,                     // Original filename
    pub original_path: Option<String>,    // Absolute path to source file
    pub document_type: DocumentType,
    pub page_count: u32,
    pub chunk_count: u32,
    pub ingested_at: i64,
    pub updated_at: i64,
    pub file_hash: String,                // SHA256 (dedup + staleness detection)
    pub file_size_bytes: u64,
    pub extraction_method: ExtractionMethod,
    pub embedder_coverage: Vec<String>,   // e.g., ["e1", "e6"]
    pub entity_count: u32,
    pub reference_count: u32,
}

/// Unified embedding record: all embedder vectors stored alongside chunk text and provenance
#[derive(Serialize, Deserialize)]
pub struct ChunkEmbeddingRecord {
    pub chunk_id: Uuid,
    pub text: String,
    pub provenance: Provenance,
    pub e1_vector: Option<Vec<f32>>,       // 384D dense vector
    pub e6_vector: Option<SparseVec>,      // SPLADE sparse vector
    pub e12_vector: Option<TokenEmbeddings>, // ColBERT per-token embeddings
    pub bm25_terms: Option<Vec<String>>,   // Pre-extracted BM25 terms
}
```

### 5.2 The Provenance Chain (How Embeddings Trace Back to Source)

```
PROVENANCE CHAIN -- EVERY VECTOR TRACES TO ITS SOURCE
=================================================================================

Embedding Vector (e.g., key "e1:{chunk_uuid}")
    |
    +---> chunk_uuid ---> ChunkData (key "chunk:{uuid}")
                           |
                           +-- text: "Either party may terminate..."
                           +-- provenance: Provenance {
                           |       document_id:        "doc-abc"
                           |       source_file_path:   "/Users/alex/Projects/Alpha/Contract.pdf"
                           |       document_filename:  "Contract.pdf"
                           |       page_number:        12
                           |       paragraph_number:   8
                           |       line_number:        1
                           |       char_start:         2401
                           |       char_end:           4401
                           |       extraction_method:  Native
                           |       ocr_confidence:     None
                           |       chunk_index:        47
                           |       created_at:         1706367600
                           |       embedded_at:        1706367612
                           |   }
                           +-- created_at:     1706367600  (when chunk was created)
                           +-- embedded_at:    1706367612  (when embedding was computed)

There is NO embedding without a chunk. There is NO chunk without provenance.
There is NO provenance without a source document path and filename.

This chain MUST be maintained through all operations:
  - Ingestion: creates chunk + provenance + embeddings together (atomic)
  - Reindex: deletes old, creates new (preserves source_file_path)
  - Delete: removes all three (chunk, provenance, embeddings) together
  - Sync: detects changed files by source_file_path + SHA256 hash
```

### 5.3 Embeddings as Raw Bytes

Dense vectors stored as raw `f32` byte arrays for zero-copy reads:

```rust
/// Store embedding
pub fn store_embedding(
    db: &rocksdb::DB,
    embedder: &str,
    chunk_id: &Uuid,
    embedding: &[f32],
) -> Result<()> {
    let key = format!("{}:{}", embedder, chunk_id);
    let bytes: &[u8] = bytemuck::cast_slice(embedding);
    let cf = db.cf_handle("embeddings").unwrap();
    db.put_cf(&cf, key.as_bytes(), bytes)?;
    Ok(())
}

/// Read embedding (zero-copy when possible)
pub fn load_embedding(
    db: &rocksdb::DB,
    embedder: &str,
    chunk_id: &Uuid,
) -> Result<Vec<f32>> {
    let key = format!("{}:{}", embedder, chunk_id);
    let cf = db.cf_handle("embeddings").unwrap();
    let bytes = db.get_cf(&cf, key.as_bytes())?
        .ok_or(CaseTrackError::EmbeddingNotFound)?;
    let embedding: &[f32] = bytemuck::cast_slice(&bytes);
    Ok(embedding.to_vec())
}
```

### 5.3 Sparse Vectors (SPLADE)

```rust
#[derive(Serialize, Deserialize)]
pub struct SparseVec {
    pub indices: Vec<u32>,    // Token IDs with non-zero weights
    pub values: Vec<f32>,     // Corresponding weights
}

impl SparseVec {
    pub fn dot(&self, other: &SparseVec) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut sum = 0.0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
        }

        sum
    }
}
```

---

## 6. Data Versioning & Migration

### 6.1 Schema Version Tracking

```rust
const CURRENT_SCHEMA_VERSION: u32 = 1;

pub fn check_and_migrate(registry_path: &Path) -> Result<()> {
    let db = rocksdb::DB::open_default(registry_path)?;

    let version = match db.get(b"schema_version")? {
        Some(bytes) => u32::from_le_bytes(bytes.try_into().unwrap()),
        None => {
            // Fresh install
            db.put(b"schema_version", CURRENT_SCHEMA_VERSION.to_le_bytes())?;
            return Ok(());
        }
    };

    if version == CURRENT_SCHEMA_VERSION {
        return Ok(());
    }

    if version > CURRENT_SCHEMA_VERSION {
        return Err(CaseTrackError::FutureSchemaVersion {
            found: version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }

    // Run migrations sequentially
    tracing::info!("Migrating database from v{} to v{}", version, CURRENT_SCHEMA_VERSION);

    // Backup first
    let backup_path = registry_path.with_extension(format!("bak.v{}", version));
    fs::copy_dir_all(registry_path, &backup_path)?;

    for v in version..CURRENT_SCHEMA_VERSION {
        match v {
            0 => migrate_v0_to_v1(&db)?,
            // Future migrations go here
            _ => unreachable!(),
        }
    }

    db.put(b"schema_version", CURRENT_SCHEMA_VERSION.to_le_bytes())?;
    tracing::info!("Migration complete.");

    Ok(())
}
```

### 6.2 Migration Rules

1. Migrations are **idempotent** (safe to re-run)
2. Always **back up** before migrating
3. Migration failures are **fatal** (don't start with corrupt data)
4. Each migration is a separate function with clear documentation
5. Never delete user data during migration -- only restructure

---

## 7. Isolation Guarantees

### 7.1 Per-Customer Isolation

Every CaseTrack installation is **fully isolated per customer**:

- CaseTrack installs on each customer's machine independently
- Each customer has their own `~/Documents/CaseTrack/` directory
- No data is shared between customers -- there is no server, no cloud, no shared state
- For Team tier (5 seats), each seat is a separate installation on a separate machine with its own database
- Customer A's embeddings, vectors, chunks, and provenance records **never touch** Customer B's data
- There is no central database. Each customer IS their own database.

```
CUSTOMER ISOLATION
=================================================================================

Customer A (Sarah's MacBook)         Customer B (Mike's Windows PC)
~/Documents/CaseTrack/               C:\Users\Mike\Documents\CaseTrack\
|-- models/                          |-- models/
|-- registry.db                      |-- registry.db
+-- collections/                     +-- collections/
    |-- {sarah-collection-1}/            |-- {mike-collection-1}/
    +-- {sarah-collection-2}/            |-- {mike-collection-2}/
                                         +-- {mike-collection-3}/

ZERO shared state. ZERO shared databases. ZERO network communication.
Each installation is a completely independent system.
```

### 7.2 Per-Collection Isolation

Each collection is a **completely independent RocksDB instance**:

- Separate database, embeddings, and index files per collection
- No cross-collection queries, shared vectors, or embedding bleed
- Independent lifecycle: deleting Collection A has zero impact on Collection B
- Portable: copy a collection directory to another machine
- Cleanly deletable: `rm -rf collections/{uuid}/`

```rust
/// Opening a collection creates or loads its isolated database
pub struct CollectionHandle {
    db: rocksdb::DB,
    collection_id: Uuid,
    collection_dir: PathBuf,
}

impl CollectionHandle {
    pub fn open(collection_dir: &Path) -> Result<Self> {
        let db_path = collection_dir.join("collection.db");
        let mut opts = rocks_options();

        let cfs = COLUMN_FAMILIES.iter()
            .map(|name| rocksdb::ColumnFamilyDescriptor::new(*name, opts.clone()))
            .collect::<Vec<_>>();

        let db = rocksdb::DB::open_cf_descriptors(&opts, &db_path, cfs)?;

        Ok(Self {
            db,
            collection_id: Uuid::parse_str(
                collection_dir.file_name().unwrap().to_str().unwrap()
            )?,
            collection_dir: collection_dir.to_path_buf(),
        })
    }

    /// Delete this collection entirely
    pub fn destroy(self) -> Result<()> {
        let path = self.collection_dir.clone();
        drop(self); // Close DB handle first
        fs::remove_dir_all(&path)?;
        Ok(())
    }
}
```

---

## 8. Context Graph Data Models

Entities, references, and relationships extracted during ingestion and stored as graph edges for structured collection navigation.

### 8.1 Entity Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,                  // Canonical name
    pub entity_type: EntityType,
    pub aliases: Vec<String>,
    pub mention_count: u32,
    pub first_seen_doc: Uuid,
    pub first_seen_chunk: Uuid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EntityType {
    Person, Organization, Date, Amount, Location, Concept,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMention {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub char_start: u64,
    pub char_end: u64,
    pub context_snippet: String,       // ~100 chars around mention
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRef {
    pub entity_key: String,            // "person:john_smith"
    pub entity_type: EntityType,
    pub name: String,
}
```

### 8.2 Reference Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceRecord {
    pub reference_key: String,         // Normalized reference identifier
    pub reference_type: ReferenceType,
    pub display_name: String,          // Human-readable reference label
    pub mention_count: u32,
    pub source_documents: Vec<Uuid>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReferenceType {
    InternalCrossRef, ExternalDocument, Hyperlink,
    Standard, Specification, Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceMention {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub context_snippet: String,
    pub relationship: Option<ReferenceRelationship>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReferenceRelationship {
    Cites, Supports, Contradicts, Extends, Supersedes, Discusses,
}
```

### 8.3 Document Graph Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocRelationship {
    pub doc_a: Uuid,
    pub doc_b: Uuid,
    pub relationship_type: DocRelType,
    pub similarity_score: Option<f32>,  // E1 cosine similarity
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DocRelType {
    SharedReferences, SharedEntities, SemanticSimilar,
    ResponseTo, Amends, Attachment,
}
```

### 8.4 Collection Summary Model

```rust
/// High-level collection overview, rebuilt after each ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSummary {
    pub key_actors: Vec<KeyActor>,
    pub key_dates: Vec<KeyDate>,
    pub key_topics: Vec<Topic>,
    pub document_categories: HashMap<String, Vec<Uuid>>,
    pub top_references: Vec<ReferenceStat>,
    pub top_entities: Vec<EntityStat>,
    pub statistics: CollectionStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyActor {
    pub name: String,
    pub role: ActorRole,
    pub mention_count: u32,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActorRole {
    Author, Reviewer, Approver, Contributor,
    Owner, Stakeholder, Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDate {
    pub date: String,
    pub description: String,
    pub source_chunk: Uuid,
    pub source_document: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub name: String,
    pub mention_count: u32,
    pub relevant_documents: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceStat {
    pub reference_key: String,
    pub display_name: String,
    pub reference_count: u32,
    pub source_documents: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityStat {
    pub entity_key: String,
    pub name: String,
    pub entity_type: EntityType,
    pub mention_count: u32,
    pub document_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStatistics {
    pub total_documents: u32,
    pub total_pages: u32,
    pub total_chunks: u32,
    pub total_entities: u32,
    pub total_references: u32,
    pub storage_bytes: u64,
    pub document_type_breakdown: HashMap<String, u32>,
    pub embedder_coverage: HashMap<String, u32>,
}
```

---

## 9. Folder Watch & Auto-Sync Storage

Watch configurations persist in `~/Documents/CaseTrack/watches.json` (JSON for human readability) to survive server restarts.

### 9.1 Watch Registry

```rust
#[derive(Serialize, Deserialize)]
pub struct WatchRegistry {
    pub watches: Vec<FolderWatch>,
}

#[derive(Serialize, Deserialize)]
pub struct FolderWatch {
    pub id: Uuid,
    pub collection_id: Uuid,
    pub folder_path: String,           // Absolute path to watched folder
    pub recursive: bool,               // Watch subfolders (default: true)
    pub enabled: bool,                 // Can be paused
    pub created_at: i64,               // Unix timestamp
    pub last_sync_at: Option<i64>,     // Last successful sync timestamp
    pub schedule: SyncSchedule,        // When to auto-sync
    pub file_extensions: Option<Vec<String>>,  // Filter (None = all supported)
    pub auto_remove_deleted: bool,     // Remove docs whose source files are gone
}

#[derive(Serialize, Deserialize)]
pub enum SyncSchedule {
    OnChange,                   // OS file-change notifications
    Interval { hours: u32 },    // Fixed interval
    Daily { time: String },     // e.g., "02:00"
    Manual,                     // Only via sync_folder tool
}
```

### 9.2 Per-Document Sync Metadata

Sync uses `file_hash` and `original_path` from `DocumentMetadata` (see Section 5.1) to detect changes:

```
FOR each file in watched folder:
  1. Compute SHA256 of file
  2. Look up file by original_path in collection DB
  3. IF not found -> new file -> ingest
  4. IF found AND hash matches -> unchanged -> skip
  5. IF found AND hash differs -> modified -> reindex (delete old, re-ingest)

FOR each document in collection DB with original_path under watched folder:
  6. IF source file no longer exists on disk -> deleted
     IF auto_remove_deleted -> delete document from collection
     ELSE -> log warning, skip
```

---

## 10. Backup & Export

### 10.1 Collection Export

Collections can be exported as portable archives:

```
casetrack export --collection "Project Alpha" --output ~/Desktop/project-alpha.ctcollection
```

The `.ctcollection` file is a ZIP containing:
- `collection.db/` -- RocksDB snapshot
- `originals/` -- Original documents (if stored)
- `manifest.json` -- Collection metadata, schema version, embedder versions

### 10.2 Collection Import

```
casetrack import ~/Desktop/project-alpha.ctcollection
```

1. Validates schema version compatibility
2. Creates new collection UUID (avoids collisions)
3. Copies database and originals to `collections/` directory
4. Registers in collection registry

---

## 11. What's Stored Where (Summary)

| Data Type | Storage Location | Format | Size Per Unit |
|-----------|------------------|--------|---------------|
| Collection metadata | `registry.db` | bincode via RocksDB | ~500 bytes/collection |
| Document metadata | `collections/{id}/collection.db` documents CF | bincode | ~200 bytes/doc |
| Text chunks (2000 chars) | `collections/{id}/collection.db` chunks CF | bincode | ~2.5KB/chunk (text + provenance metadata) |
| E1 embeddings (384D) | `collections/{id}/collection.db` embeddings CF | f32 bytes | 1,536 bytes/chunk |
| E6 sparse vectors | `collections/{id}/collection.db` embeddings CF | bincode sparse | ~500 bytes/chunk |
| E12 token embeddings | `collections/{id}/collection.db` embeddings CF | bincode | ~8KB/chunk |
| BM25 inverted index | `collections/{id}/collection.db` bm25_index CF | bincode | ~2MB/1000 chunks |
| Provenance records | Embedded in chunk embedding records | bincode | ~300 bytes/chunk |
| Original documents | `collections/{id}/originals/` | original files | varies |
| ONNX models | `models/` | ONNX format | 35-110MB each |

---

*CaseTrack PRD v4.0.0 -- Document 4 of 10*
