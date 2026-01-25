# Code Embedding Separation Plan

## Problem Statement

Currently, the file watcher only handles markdown files. Code files are not watched, and when code is embedded (e.g., during benchmarks), it goes through the same 13-embedder pipeline as regular text content. This is suboptimal because:

1. **Code requires different chunking** - AST-based (functions, structs, traits) vs word-based (200 words, 50 overlap)
2. **Code uses different embedders** - E7 (Qodo-Embed) is the primary code embedder, not E1
3. **Code has different metadata** - language, line numbers, parent type, signature
4. **Retrieval patterns differ** - code search is often by signature/pattern, not semantic

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    File Watcher System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │ GitFileWatcher  │          │ CodeFileWatcher │             │
│   │ (Markdown)      │          │ (Code Files)    │             │
│   └────────┬────────┘          └────────┬────────┘             │
│            │                            │                       │
│            ▼                            ▼                       │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │ TextChunker     │          │ ASTChunker      │             │
│   │ (200w, 50 over) │          │ (tree-sitter)   │             │
│   └────────┬────────┘          └────────┬────────┘             │
│            │                            │                       │
│            ▼                            ▼                       │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │ 13-Embedder     │          │ Code Embedder   │             │
│   │ (Full Pipeline) │          │ (E7 + Subset)   │             │
│   └────────┬────────┘          └────────┬────────┘             │
│            │                            │                       │
│            ▼                            ▼                       │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │ TeleologicalDB  │          │ CodeDB          │             │
│   │ (Existing CFs)  │          │ (New CFs)       │             │
│   └─────────────────┘          └─────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: New Column Families for Code Storage

Add to `column_families.rs`:

```rust
// =============================================================================
// CODE EMBEDDING COLUMN FAMILIES (Code-specific storage)
// =============================================================================

/// Column family for code entity storage.
/// Key: UUID (16 bytes) → Value: CodeEntity serialized via bincode
pub const CF_CODE_ENTITIES: &str = "code_entities";

/// Column family for code E7 embeddings (1536D).
/// Key: UUID (16 bytes) → Value: Vec<f32> (1536 × 4 = 6144 bytes)
pub const CF_CODE_E7_EMBEDDINGS: &str = "code_e7_embeddings";

/// Column family for code file index.
/// Key: file_path bytes → Value: CodeFileIndexEntry serialized
pub const CF_CODE_FILE_INDEX: &str = "code_file_index";

/// Column family for code entity index by name.
/// Key: entity_name bytes → Value: Vec<Uuid>
pub const CF_CODE_NAME_INDEX: &str = "code_name_index";

/// Column family for code entity index by signature.
/// Key: signature_hash bytes → Value: Vec<Uuid>
pub const CF_CODE_SIGNATURE_INDEX: &str = "code_signature_index";
```

## Phase 2: Code Entity Types

Create `crates/context-graph-core/src/types/code_entity.rs`:

```rust
/// A code entity extracted from source files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEntity {
    /// Unique identifier
    pub id: Uuid,
    /// Entity type
    pub entity_type: CodeEntityType,
    /// Entity name (function name, struct name, etc.)
    pub name: String,
    /// Full code content
    pub code: String,
    /// Language
    pub language: CodeLanguage,
    /// File path
    pub file_path: String,
    /// Line number (1-indexed)
    pub line_start: usize,
    /// End line number
    pub line_end: usize,
    /// Module path (e.g., "context_graph_core::memory")
    pub module_path: Option<String>,
    /// Function signature (for functions/methods)
    pub signature: Option<String>,
    /// Parent type (for methods inside impl blocks)
    pub parent_type: Option<String>,
    /// Visibility (pub, pub(crate), private)
    pub visibility: Visibility,
    /// When this entity was last updated
    pub last_updated: DateTime<Utc>,
    /// Content hash for change detection
    pub content_hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeEntityType {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Impl,
    Const,
    Static,
    TypeAlias,
    Macro,
    Module,
    Import,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeLanguage {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    Cpp,
    C,
    SQL,
    Unknown,
}
```

## Phase 3: AST Chunker with tree-sitter

Create `crates/context-graph-core/src/memory/ast_chunker.rs`:

```rust
/// AST-based code chunker using tree-sitter.
pub struct ASTChunker {
    /// tree-sitter parsers by language
    parsers: HashMap<CodeLanguage, Parser>,
}

impl ASTChunker {
    /// Create a new AST chunker with supported languages.
    pub fn new() -> Result<Self, ChunkerError>;

    /// Parse a source file and extract code entities.
    pub fn chunk_file(
        &self,
        content: &str,
        file_path: &str,
        language: CodeLanguage,
    ) -> Result<Vec<CodeEntity>, ChunkerError>;

    /// Detect language from file extension.
    pub fn detect_language(file_path: &str) -> CodeLanguage;
}
```

## Phase 4: Code File Watcher

Create `crates/context-graph-core/src/memory/code_watcher.rs`:

```rust
/// Code file watcher for source code changes.
pub struct CodeFileWatcher {
    /// Code capture service for storing entities
    capture_service: Arc<CodeCaptureService>,
    /// AST chunker for parsing code
    chunker: ASTChunker,
    /// File content hashes for change detection
    file_hashes: Arc<RwLock<HashMap<PathBuf, String>>>,
    /// Session ID
    session_id: String,
    /// Paths being watched
    watch_paths: Vec<PathBuf>,
    /// Supported languages
    languages: HashSet<CodeLanguage>,
    /// Running state
    is_running: bool,
}

impl CodeFileWatcher {
    /// Create a new code file watcher.
    pub fn new(
        watch_paths: Vec<PathBuf>,
        capture_service: Arc<CodeCaptureService>,
        session_id: String,
    ) -> Result<Self, WatcherError>;

    /// Start watching and perform initial scan.
    pub async fn start(&mut self) -> Result<(), WatcherError>;

    /// Process pending changes.
    pub async fn process_events(&mut self) -> Result<usize, WatcherError>;

    /// Stop watching.
    pub fn stop(&mut self);
}
```

## Phase 5: Code Capture Service

Create `crates/context-graph-core/src/memory/code_capture.rs`:

```rust
/// Service for capturing code entities with E7 embeddings.
pub struct CodeCaptureService {
    /// Code storage
    store: Arc<CodeStore>,
    /// E7 code embedder
    embedder: Arc<CodeModel>,
    /// Optional: E1 for semantic search
    semantic_embedder: Option<Arc<dyn EmbeddingModel>>,
}

impl CodeCaptureService {
    /// Capture a code entity.
    pub async fn capture_entity(
        &self,
        entity: CodeEntity,
    ) -> Result<Uuid, CaptureError>;

    /// Delete entities for a file.
    pub async fn delete_by_file_path(
        &self,
        file_path: &str,
    ) -> Result<usize, CaptureError>;

    /// Search code by query.
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<CodeSearchResult>, CaptureError>;
}
```

## Phase 6: Code Store Implementation

Create `crates/context-graph-storage/src/code/mod.rs`:

```rust
/// Storage for code entities and embeddings.
pub struct CodeStore {
    db: Arc<DB>,
    // Column family handles
    cf_entities: Arc<BoundColumnFamily<'static>>,
    cf_embeddings: Arc<BoundColumnFamily<'static>>,
    cf_file_index: Arc<BoundColumnFamily<'static>>,
    cf_name_index: Arc<BoundColumnFamily<'static>>,
    cf_signature_index: Arc<BoundColumnFamily<'static>>,
}

impl CodeStore {
    /// Store a code entity with its E7 embedding.
    pub fn store(
        &self,
        entity: &CodeEntity,
        embedding: &[f32],
    ) -> Result<(), StorageError>;

    /// Get entity by ID.
    pub fn get(&self, id: Uuid) -> Result<Option<CodeEntity>, StorageError>;

    /// Get entities by file path.
    pub fn get_by_file(&self, file_path: &str) -> Result<Vec<CodeEntity>, StorageError>;

    /// Search by name prefix.
    pub fn search_by_name(&self, prefix: &str) -> Result<Vec<CodeEntity>, StorageError>;

    /// Get embedding for entity.
    pub fn get_embedding(&self, id: Uuid) -> Result<Option<Vec<f32>>, StorageError>;

    /// Delete all entities for a file.
    pub fn delete_file(&self, file_path: &str) -> Result<usize, StorageError>;

    /// Get file statistics.
    pub fn get_stats(&self) -> Result<CodeStats, StorageError>;
}
```

## Phase 7: MCP Tool Updates

Update `file_watcher_tools.rs` to expose code-specific operations:

```rust
// New tools for code
pub const TOOL_LIST_CODE_FILES: &str = "list_code_files";
pub const TOOL_GET_CODE_FILE_STATS: &str = "get_code_file_stats";
pub const TOOL_SEARCH_CODE: &str = "search_code";
pub const TOOL_GET_CODE_ENTITY: &str = "get_code_entity";
pub const TOOL_DELETE_CODE_FILE: &str = "delete_code_file_content";
pub const TOOL_RECONCILE_CODE_FILES: &str = "reconcile_code_files";
```

## Phase 8: Configuration

Add to `config/default.toml`:

```toml
[code_watcher]
enabled = true
languages = ["rust", "python", "typescript", "javascript", "go"]
watch_paths = ["./crates", "./src"]
exclude_patterns = ["**/target/**", "**/node_modules/**", "**/.git/**"]
min_entity_lines = 3
max_entity_lines = 500

[code_embedding]
# Use E7 as primary, optionally blend with E1
primary_embedder = "E7"
enable_semantic_blend = true
semantic_weight = 0.4  # When blending with E1
```

## Implementation Order

### Week 1: Storage Layer
1. Add new column families to `column_families.rs`
2. Create `CodeEntity` and related types
3. Implement `CodeStore` with basic CRUD

### Week 2: AST Chunking
1. Add tree-sitter dependency
2. Implement `ASTChunker` for Rust
3. Add language detection
4. Write tests with real Rust files

### Week 3: Watcher & Capture
1. Implement `CodeFileWatcher`
2. Implement `CodeCaptureService`
3. Integrate with E7 model
4. Write integration tests

### Week 4: MCP Integration
1. Add MCP tools for code operations
2. Update server to run both watchers
3. Add search_code tool
4. End-to-end testing

## Migration Path

Existing databases will work unchanged - the new column families are additive. The code watcher runs alongside the existing markdown watcher.

## Success Metrics

1. **Code coverage**: Watch all `.rs` files in crates/
2. **Entity extraction**: >95% of functions/structs detected
3. **Embedding accuracy**: E7 benchmark MRR > 0.55
4. **Latency**: Entity embedding < 100ms
5. **Storage efficiency**: < 10KB per entity (including embedding)

## Dependencies

```toml
# Cargo.toml additions
tree-sitter = "0.22"
tree-sitter-rust = "0.21"
tree-sitter-python = "0.21"
tree-sitter-typescript = "0.21"
tree-sitter-javascript = "0.21"
tree-sitter-go = "0.21"
```
