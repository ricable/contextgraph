# PRD 10: Technical Build Guide

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

> **LANGUAGE: RUST** -- This entire project is built in Rust. Every crate, every
> module, every line of product code is Rust. The final deliverable is a single
> statically-linked Rust binary per platform. No runtime dependencies. The only
> non-Rust code is `scripts/convert_models.py` (one-time build tool, not shipped).

---

## 1. Project Bootstrap

### 1.1 Create Fresh Project

```bash
mkdir casetrack && cd casetrack
cargo init --name casetrack
mkdir -p crates/casetrack-core
cd crates/casetrack-core && cargo init --lib --name casetrack-core && cd ../..
git init
echo -e "target/\n*.onnx\nmodels/" > .gitignore
```

### 1.2 Workspace Structure

```
casetrack/
|-- Cargo.toml                   # Workspace root
|-- Cargo.lock
|-- .github/workflows/
|   |-- ci.yml
|   +-- release.yml
|-- scripts/
|   |-- convert_models.py
|   |-- build_mcpb.sh
|   +-- install.sh
|-- crates/
|   |-- casetrack/               # Binary crate (MCP server entry point)
|   |   |-- Cargo.toml
|   |   +-- src/
|   |       |-- main.rs
|   |       |-- cli.rs
|   |       |-- server.rs
|   |       +-- format.rs
|   +-- casetrack-core/          # Library crate (all business logic)
|       |-- Cargo.toml
|       +-- src/
|           |-- lib.rs
|           |-- error.rs
|           |-- config.rs
|           |-- collection/      # registry, handle, model
|           |-- document/        # pdf, docx, xlsx, ocr, chunker, model
|           |-- embedding/       # engine, models, download, types
|           |-- search/          # engine, bm25, ranking, result
|           |-- provenance/      # citation formatting
|           |-- storage/         # rocks, schema
|           +-- license/         # validator (ed25519)
|-- tests/
|   |-- integration/
|   |   |-- test_collection_lifecycle.rs
|   |   |-- test_ingest_pdf.rs
|   |   |-- test_search.rs
|   |   +-- test_mcp_tools.rs
|   +-- fixtures/
|       |-- sample.pdf
|       |-- sample.docx
|       |-- sample.xlsx
|       +-- scanned.png
+-- docs/prd/
```

### 1.3 Workspace Cargo.toml

```toml
[workspace]
members = ["crates/casetrack", "crates/casetrack-core"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
license = "LicenseRef-Commercial"
repository = "https://github.com/casetrack-dev/casetrack"

[workspace.dependencies]
rmcp = { version = "0.13", features = ["server", "transport-io", "macros"] }
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
rocksdb = "0.22"
ort = { version = "2.0", features = ["download-binaries"] }
pdf-extract = "0.7"
lopdf = "0.32"
docx-rs = "0.4"
calamine = "0.24"
image = "0.25"
tesseract = { version = "0.14", optional = true }
hf-hub = "0.3"
reqwest = { version = "0.12", features = ["json", "rustls-tls"], default-features = false }
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
clap = { version = "4.4", features = ["derive"] }
thiserror = "2.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
ed25519-dalek = "2.1"
base64 = "0.22"
bytemuck = { version = "1.14", features = ["derive"] }
sha2 = "0.10"
sysinfo = "0.30"
walkdir = "2.4"
notify = "6.1"
semver = "1.0"
dirs = "5.0"
```

### 1.4 Crate Cargo.toml Files

**Binary crate** (`crates/casetrack/Cargo.toml`):

```toml
[package]
name = "casetrack"
version.workspace = true
edition.workspace = true

[[bin]]
name = "casetrack"
path = "src/main.rs"

[dependencies]
casetrack-core = { path = "../casetrack-core" }
rmcp.workspace = true
tokio.workspace = true
serde_json.workspace = true
clap.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
dirs.workspace = true
anyhow.workspace = true

[features]
default = ["ocr"]
ocr = ["casetrack-core/ocr"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"
```

**Core library** (`crates/casetrack-core/Cargo.toml`):

```toml
[package]
name = "casetrack-core"
version.workspace = true
edition.workspace = true

[dependencies]
rocksdb.workspace = true
ort.workspace = true
pdf-extract.workspace = true
lopdf.workspace = true
docx-rs.workspace = true
calamine.workspace = true
image.workspace = true
tesseract = { workspace = true, optional = true }
hf-hub.workspace = true
reqwest.workspace = true
serde.workspace = true
serde_json.workspace = true
bincode.workspace = true
uuid.workspace = true
chrono.workspace = true
thiserror.workspace = true
anyhow.workspace = true
tracing.workspace = true
ed25519-dalek.workspace = true
base64.workspace = true
bytemuck.workspace = true
sha2.workspace = true
sysinfo.workspace = true
walkdir.workspace = true
notify.workspace = true
semver.workspace = true
dirs.workspace = true

[features]
default = ["ocr"]
ocr = ["dep:tesseract"]
```

---

## 2. Entry Point

```rust
// crates/casetrack/src/main.rs
use clap::Parser;
use tracing_subscriber::EnvFilter;

mod cli;
mod server;
mod format;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::Args::parse();

    match &args.command {
        Some(cli::Command::SetupClaudeCode) => return casetrack_core::setup_claude_code(&args.data_dir()),
        Some(cli::Command::Update) => return casetrack_core::self_update().await,
        Some(cli::Command::Uninstall) => return casetrack_core::uninstall(),
        None => {}
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("casetrack=info")))
        .with_writer(std::io::stderr)
        .init();

    tracing::info!("CaseTrack v{} starting...", env!("CARGO_PKG_VERSION"));

    server::CaseTrackServer::start(casetrack_core::Config {
        data_dir: args.data_dir(),
        license_key: args.license.clone(),
    }).await
}
```

---

## 3. CLI Arguments

```rust
// crates/casetrack/src/cli.rs
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "casetrack", about = "Document intelligence MCP server for Claude", version)]
pub struct Args {
    #[arg(long, env = "CASETRACK_HOME")]
    pub data_dir: Option<PathBuf>,

    #[arg(long, env = "CASETRACK_LICENSE")]
    pub license: Option<String>,

    #[arg(long, value_enum)]
    pub memory_mode: Option<MemoryMode>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    SetupClaudeCode,
    Update,
    Uninstall,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum MemoryMode { Full, Standard, Constrained }
```

`Args::data_dir()` defaults to `~/Documents/CaseTrack/` via `dirs::document_dir()`.

---

## 4. Error Handling

### 4.1 Error Types

```rust
// crates/casetrack-core/src/error.rs
use thiserror::Error;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum CaseTrackError {
    // === Collection Errors ===
    #[error("Collection not found: {0}")]
    CollectionNotFound(Uuid),

    #[error("No active collection. Create or switch to a collection first.")]
    NoCollectionActive,

    #[error("Collection name not found: \"{0}\"")]
    CollectionNameNotFound(String),

    // === Document Errors ===
    #[error("Document not found: {0}")]
    DocumentNotFound(Uuid),

    #[error("File not found: {}", .0.display())]
    FileNotFound(PathBuf),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Duplicate document (SHA256 matches existing document ID: {0})")]
    DuplicateDocument(Uuid),

    // === PDF Errors ===
    #[error("PDF parse error for {}: {source}", .path.display())]
    PdfParseError { path: PathBuf, source: lopdf::Error },

    // === DOCX Errors ===
    #[error("DOCX parse error for {}: {source}", .path.display())]
    DocxParseError { path: PathBuf, source: String },

    // === XLSX Errors ===
    #[error("XLSX parse error for {}: {source}", .path.display())]
    XlsxParseError { path: PathBuf, source: String },

    // === OCR Errors ===
    #[error("OCR not available (build without OCR feature)")]
    OcrNotAvailable,

    #[error("OCR failed: {0}")]
    OcrFailed(String),

    // === Embedding Errors ===
    #[error("Embedder not loaded: {0:?}")]
    EmbedderNotLoaded(crate::embedding::EmbedderId),

    #[error("Model not downloaded: {0:?}. Run server with network access to download.")]
    ModelNotDownloaded(crate::embedding::EmbedderId),

    #[error("ONNX inference failed: {0}")]
    InferenceFailed(String),

    #[error("Embedding not found for chunk {0}")]
    EmbeddingNotFound(Uuid),

    // === Storage Errors ===
    #[error("Registry database failed to open: {source}")]
    RegistryOpenFailed { source: rocksdb::Error },

    #[error("Collection database failed to open at {}: {source}", .path.display())]
    CollectionDbOpenFailed { path: PathBuf, source: rocksdb::Error },

    #[error("Database schema version {found} is newer than supported version {supported}. Update CaseTrack.")]
    FutureSchemaVersion { found: u32, supported: u32 },

    #[error("BM25 index is empty. Ingest documents first.")]
    Bm25IndexEmpty,

    // === Search Errors ===
    #[error("Chunk not found: {0}")]
    ChunkNotFound(Uuid),

    // === License Errors ===
    #[error("Free tier limit: {resource} ({current}/{max}). Upgrade: https://casetrack.dev/upgrade")]
    FreeTierLimit { resource: String, current: u32, max: u32 },

    #[error("Invalid license key format")]
    InvalidLicenseFormat,

    // === System Errors ===
    #[error("Home directory not found")]
    NoHomeDir,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),
}

pub type Result<T> = std::result::Result<T, CaseTrackError>;
```

### 4.2 Error Design Principles

1. **Specific**: Every error tells you exactly what went wrong
2. **Actionable**: Every error tells you what to do about it
3. **No silent failures**: Every operation returns `Result<T>`
4. **User-facing errors include guidance**: "Create or switch to a collection first"
5. **Internal errors include report URL**: "Please report this at github.com/..."

---

## 5. Configuration

```rust
// crates/casetrack-core/src/config.rs
use std::path::PathBuf;

pub struct Config {
    pub data_dir: PathBuf,
    pub license_key: Option<String>,
}

/// Optional config file (~/Documents/CaseTrack/config.toml) -- zero-config by default
#[derive(serde::Deserialize, Default)]
pub struct ConfigFile {
    pub data_dir: Option<PathBuf>,
    pub license_key: Option<String>,
    pub ocr_language: Option<String>,
    pub copy_originals: Option<bool>,
    pub memory_mode: Option<String>,
    pub inference_threads: Option<u32>,
}
```

---

## 6. Logging

Logging goes to stderr (stdout is MCP transport). Controlled by `RUST_LOG` env var.

| Level | Usage |
|-------|-------|
| ERROR | Failures preventing operations (file not found, DB corruption) |
| WARN  | Degraded functionality (low memory, OCR disabled) |
| INFO  | Normal operations (server started, collection created, search completed) |
| DEBUG | Internal details (model loading times, RocksDB stats) |
| TRACE | Verbose (individual chunk embeddings, token counts) |

---

## 7. Cross-Platform Concerns

### 7.1 Path Handling

```rust
pub fn resolve_path(input: &str) -> PathBuf {
    let expanded = if input.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            home.join(&input[2..])
        } else {
            PathBuf::from(input)
        }
    } else {
        PathBuf::from(input)
    };
    expanded
}
```

### 7.2 Default Data Directory

| Platform | Default Path |
|----------|-------------|
| macOS | `~/Documents/CaseTrack/` |
| Windows | `C:\Users\{user}\Documents\CaseTrack\` |
| Linux | `~/Documents/CaseTrack/` (or `~/.local/share/casetrack/`) |

### 7.3 Platform Dependencies

| Component | macOS | Windows | Linux |
|-----------|-------|---------|-------|
| RocksDB | Works via `rust-rocksdb` | Requires MSVC build tools | Static link |
| Tesseract | Static link (vendored) | Bundle DLLs in installer | Static link or system pkg |
| ONNX Runtime | CoreML + CPU fallback | DirectML + CPU fallback | CPU only (CUDA optional) |

---

## 8. Security

### 8.1 Input Validation

```rust
pub fn validate_file_path(path: &Path, data_dir: &Path) -> Result<PathBuf> {
    let canonical = path.canonicalize()
        .map_err(|_| CaseTrackError::FileNotFound(path.to_path_buf()))?;
    Ok(canonical)
}

pub fn validate_write_path(path: &Path, data_dir: &Path) -> Result<PathBuf> {
    let canonical = path.canonicalize()
        .map_err(|_| CaseTrackError::FileNotFound(path.to_path_buf()))?;
    let data_canonical = data_dir.canonicalize()?;
    if !canonical.starts_with(&data_canonical) {
        return Err(CaseTrackError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("Write path must be within data directory: {}", data_dir.display()),
        )));
    }
    Ok(canonical)
}
```

### 8.2 License Key Security

- ed25519 signed keys, validated offline. Public key embedded in binary.
- Key format: `TIER-XXXXXX-XXXXXX-XXXXXX-SIG`
- Cached validation avoids repeated network calls. No user data sent.

### 8.3 No Network After Setup

After initial model download and license activation, CaseTrack makes zero network requests. Document processing, search, and storage are 100% local. Update checks are optional and non-blocking.

### 8.4 Data Privacy and Confidentiality

All documents and embeddings are stored locally on the user's machine. CaseTrack never transmits document content, metadata, or search queries to any external service. This ensures data privacy and confidentiality for sensitive business documents.

---

## 9. Testing Strategy

### 9.1 Unit Tests

Key test areas (every module has unit tests):

```rust
#[cfg(test)]
mod tests {
    // Chunking: 2000-char target, 200-char overlap, paragraph-aware
    fn test_chunk_respects_paragraph_boundaries() { ... }
    fn test_chunk_target_2000_chars() { ... }
    fn test_chunk_overlap_200_chars() { ... }
    fn test_chunk_min_400_chars() { ... }
    fn test_chunk_max_2200_chars() { ... }
    fn test_chunk_provenance_complete() { ... }

    // BM25
    fn test_bm25_basic_search() { ... }
    fn test_bm25_term_frequency() { ... }

    // PROVENANCE (MOST CRITICAL TEST SUITE)
    // Every chunk MUST have: file path, document name, page, paragraph, line, char offsets, timestamps.
    // Every embedding MUST link back to a chunk with valid provenance.
    // Every search result MUST include complete provenance.
    fn test_citation_format() { ... }
    fn test_short_citation() { ... }
    fn test_provenance_includes_file_path() { ... }
    fn test_provenance_includes_document_name() { ... }
    fn test_provenance_includes_page_number() { ... }
    fn test_provenance_includes_paragraph_range() { ... }
    fn test_provenance_includes_line_range() { ... }
    fn test_provenance_includes_char_offsets() { ... }
    fn test_provenance_includes_timestamps() { ... }
    fn test_provenance_round_trip() { ... }
    fn test_embedding_links_to_valid_chunk() { ... }
    fn test_no_orphaned_embeddings() { ... }
    fn test_no_chunk_without_provenance() { ... }

    // Document parsers
    fn test_pdf_extraction() { ... }
    fn test_docx_extraction() { ... }
    fn test_xlsx_extraction() { ... }

    // RRF, cosine similarity, license
    fn test_rrf_fusion() { ... }
    fn test_cosine_identical_vectors() { ... }
    fn test_cosine_orthogonal_vectors() { ... }
    fn test_free_tier_limits() { ... }
    fn test_valid_license_key() { ... }
}
```

### 9.2 Integration Tests

```rust
// tests/integration/test_collection_lifecycle.rs
#[tokio::test]
async fn test_create_list_switch_delete_collection() {
    let dir = tempdir().unwrap();
    let mut registry = CollectionRegistry::open(dir.path()).unwrap();

    let collection = registry.create_collection(CreateCollectionParams {
        name: "Project Alpha".to_string(),
        collection_id: None,
        collection_type: Some(CollectionType::Contract),
    }).unwrap();
    assert_eq!(collection.name, "Project Alpha");

    let collections = registry.list_collections().unwrap();
    assert_eq!(collections.len(), 1);

    let handle = registry.switch_collection(collection.id).unwrap();
    assert_eq!(registry.active_collection_id(), Some(collection.id));

    drop(handle);
    registry.delete_collection(collection.id).unwrap();
    assert_eq!(registry.list_collections().unwrap().len(), 0);
}

// tests/integration/test_search.rs
#[tokio::test]
async fn test_search_returns_relevant_results() {
    // Setup: create collection, ingest sample PDF with known content
    let results = search_engine.search(&collection_handle, "payment terms", 10, None).unwrap();

    assert!(!results.is_empty());
    assert!(results[0].score > 0.5);
    assert!(results[0].citation.contains("sample.pdf"));

    // Verify full provenance on every result
    for result in &results {
        assert!(!result.provenance.document_name.is_empty());
        assert!(!result.provenance.document_path.is_empty());
        assert!(result.provenance.page > 0);
        assert!(result.provenance.char_start < result.provenance.char_end);
    }
}

#[tokio::test]
async fn test_collection_isolation() {
    // Verify chunks from one collection never appear in another collection's search
    // Ingest into Collection A, search Collection B -- must return zero results
}
```

### 9.3 Test Fixtures

- `sample.pdf` -- 3-page PDF with known contract terms
- `sample.docx` -- Word document with headings, paragraphs, lists
- `sample.xlsx` -- Spreadsheet with financial data and multiple sheets
- `scanned.png` -- Image of typed text for OCR testing
- `empty.pdf` -- Edge case: empty PDF
- `large_paragraph.txt` -- Edge case: single paragraph >2000 characters

### 9.4 Running Tests

```bash
cargo test              # All tests
cargo test --lib        # Unit tests only (fast)
cargo test --test '*'   # Integration tests (needs fixtures)
RUST_LOG=debug cargo test -- --nocapture   # With logging
cargo test test_bm25_basic_search          # Specific test
```

---

## 10. CI/CD Pipeline

### 10.1 GitHub Actions CI

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
      - run: cargo build --release
      - run: cargo test
      - run: cargo clippy -- -D warnings
      - run: cargo fmt -- --check

  size-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --release
      - name: Check binary size (<50MB)
        run: |
          SIZE=$(stat -f%z target/release/casetrack 2>/dev/null || stat -c%s target/release/casetrack)
          echo "Binary size: $SIZE bytes ($(($SIZE / 1024 / 1024)) MB)"
          [ "$SIZE" -le 52428800 ] || exit 1
```

### 10.2 Release Pipeline

```yaml
name: Release
on:
  push: { tags: ['v*'] }
permissions: { contents: write }

jobs:
  build:
    strategy:
      matrix:
        include:
          - { target: x86_64-apple-darwin, os: macos-latest, name: casetrack-darwin-x64 }
          - { target: aarch64-apple-darwin, os: macos-latest, name: casetrack-darwin-arm64 }
          - { target: x86_64-pc-windows-msvc, os: windows-latest, name: casetrack-win32-x64.exe }
          - { target: x86_64-unknown-linux-gnu, os: ubuntu-latest, name: casetrack-linux-x64 }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { targets: "${{ matrix.target }}" }
      - run: cargo build --release --target ${{ matrix.target }}
      - shell: bash
        run: |
          if [ "${{ matrix.os }}" = "windows-latest" ]; then
            cp target/${{ matrix.target }}/release/casetrack.exe ${{ matrix.name }}
          else
            cp target/${{ matrix.target }}/release/casetrack ${{ matrix.name }}
          fi
      - uses: actions/upload-artifact@v4
        with: { name: "${{ matrix.name }}", path: "${{ matrix.name }}" }

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - uses: softprops/action-gh-release@v1
        with:
          files: |
            casetrack-darwin-x64/casetrack-darwin-x64
            casetrack-darwin-arm64/casetrack-darwin-arm64
            casetrack-win32-x64.exe/casetrack-win32-x64.exe
            casetrack-linux-x64/casetrack-linux-x64
          generate_release_notes: true

  mcpb:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
      - run: bash scripts/build_mcpb.sh
      - uses: softprops/action-gh-release@v1
        with: { files: casetrack.mcpb }
```

---

## 11. Monetization Implementation

### 11.1 Pricing Tiers

| Tier | Price | Collections | Docs/Collection | Embedders | Key Features |
|------|-------|-------------|-----------------|-----------|-------------|
| Free | $0 | 3 | 100 | 2 (E1, E6) | BM25, basic search, provenance |
| Pro | $29/mo | Unlimited | Unlimited | 3 (+ E12) | ColBERT rerank, auto-sync, priority support |

### 11.2 License Key System

See [PRD 07](PRD_07_COLLECTION_MANAGEMENT.md) for details. Key points: ed25519 offline validation, online activation via Lemon Squeezy on first use, 30-day cache, graceful degradation to Free tier on failure.

---

## 12. Implementation Roadmap

### Phase 1: Foundation

```
PROJECT SETUP
  [ ] Create workspace with crates/casetrack and crates/casetrack-core
  [ ] Configure Cargo.toml workspace dependencies
  [ ] Set up GitHub repo + CI (ci.yml)
  [ ] Implement error types (error.rs)
  [ ] Implement config + CLI parsing (config.rs, cli.rs)
  [ ] Set up tracing/logging to stderr

COLLECTION MANAGEMENT
  [ ] Implement Collection, CollectionType, CollectionStatus structs
  [ ] Implement CollectionRegistry (create, list, switch, delete)
  [ ] Implement CollectionHandle (open collection DB, column families)
  [ ] RocksDB configuration (rocks_options)
  [ ] Schema versioning (check_and_migrate)
  [ ] Unit tests for collection operations

MCP SERVER SKELETON
  [ ] Set up rmcp server with stdio transport
  [ ] Register create_collection, list_collections, switch_collection, delete_collection tools
  [ ] Test with Claude Code manually
```

### Phase 2: Document Ingestion

```
PDF PROCESSING
  [ ] Implement PdfProcessor (native text extraction)
  [ ] PDF metadata extraction
  [ ] Scanned page detection heuristic
  [ ] Page/paragraph/line detection

DOCX PROCESSING
  [ ] Implement DocxProcessor
  [ ] Paragraph and heading extraction
  [ ] Section break handling

XLSX PROCESSING
  [ ] Implement XlsxProcessor (calamine crate)
  [ ] Sheet enumeration and cell extraction
  [ ] Table structure preservation
  [ ] Header row detection

CHUNKING (2000-character chunks, 10% overlap -- see PRD 06)
  [ ] Implement DocumentChunker (2000-char target, 200-char overlap, paragraph-aware)
  [ ] Character counting (not token-based)
  [ ] Long paragraph splitting (>2200 chars)
  [ ] Provenance attachment per chunk (file path, document name, page, paragraph, line, char offsets)
  [ ] Chunk boundary validation (min 400 chars, max 2200 chars)

STORAGE (Per-collection isolated databases -- see PRD 04)
  [ ] Store chunks in RocksDB (one DB per collection)
  [ ] Store document metadata
  [ ] Store provenance records (full path, page, paragraph, line, char offsets per chunk)
  [ ] Duplicate detection (SHA256)
  [ ] ingest_document MCP tool
  [ ] list_documents, get_document, delete_document tools
  [ ] get_chunk, get_document_chunks, get_source_context provenance tools (see PRD 09)
```

### Phase 3: Embedding & Search

```
MODEL MANAGEMENT
  [ ] Model download via hf-hub (with retry)
  [ ] Model spec definitions (repo, files, sizes)
  [ ] First-run download flow
  [ ] Model existence checking

EMBEDDING ENGINE
  [ ] ONNX Runtime setup (Environment, Session)
  [ ] E1 (bge-small dense embedding)
  [ ] E6 (SPLADE sparse embedding)
  [ ] Batch embedding for ingestion
  [ ] Store embeddings in RocksDB

BM25 INDEX
  [ ] Tokenization (lowercase, stopword removal)
  [ ] Inverted index (posting lists in RocksDB)
  [ ] BM25 scoring formula
  [ ] Index update during ingestion

SEARCH ENGINE
  [ ] Stage 1: BM25 recall
  [ ] Stage 2: Semantic ranking (E1 + E6 via RRF)
  [ ] Cosine similarity, sparse dot product
  [ ] search_documents MCP tool
  [ ] Result formatting with citations
```

### Phase 4: Pro Features

```
PRO EMBEDDERS
  [ ] E12 (ColBERT token-level)
  [ ] Stage 3: ColBERT rerank (MaxSim)

LICENSE SYSTEM
  [ ] ed25519 key validation
  [ ] Online activation (Lemon Squeezy)
  [ ] Offline cache (30-day)
  [ ] Feature gating per tier
  [ ] Upgrade prompts in error messages

FOLDER INGESTION & SYNC
  [ ] ingest_folder tool (recursive directory walking via walkdir)
  [ ] SHA256 duplicate detection (skip already-ingested files)
  [ ] sync_folder tool (differential sync: new/changed/deleted detection)
  [ ] sync_folder dry_run mode (preview changes without applying)
  [ ] sync_folder remove_deleted option (remove docs whose source files are gone)
  [ ] File extension filtering
  [ ] Progress reporting (per-file status via stderr logging)
  [ ] Error collection and summary for batch operations

REINDEXING & EMBEDDING FRESHNESS
  [ ] reindex_document tool (delete old chunks/embeddings, re-extract, re-chunk, re-embed)
  [ ] reindex_collection tool (full rebuild of all documents in a collection)
  [ ] reparse=false mode (keep chunks, rebuild embeddings only -- fast tier upgrade path)
  [ ] skip_unchanged mode (only reindex docs whose source SHA256 changed or embeddings incomplete)
  [ ] get_index_status tool (health check: per-document embedder coverage, SHA256 staleness)
  [ ] Embedder coverage tracking (store which embedders were used per chunk)
  [ ] Automatic stale detection (compare stored SHA256 vs source file on disk)
  [ ] Force reindex flag (rebuild even if SHA256 matches)

AUTO-SYNC & FOLDER WATCHING (see PRD 09 Section 3)
  [ ] WatchManager struct (manages all active watches)
  [ ] notify crate integration (cross-platform OS file notifications)
      - inotify (Linux), FSEvents (macOS), ReadDirectoryChangesW (Windows)
  [ ] watches.json persistence (survives server restarts)
  [ ] FolderWatch config struct (folder_path, schedule, auto_remove, extensions)
  [ ] SyncSchedule enum (OnChange, Interval, Daily, Manual)
  [ ] Real-time event processing with 2-second debounce
  [ ] Event batching (Created -> ingest, Modified -> reindex, Deleted -> remove)
  [ ] Scheduled sync runner (tokio interval, checks every 60 seconds)
  [ ] watch_folder MCP tool
  [ ] unwatch_folder MCP tool
  [ ] list_watches MCP tool
  [ ] set_sync_schedule MCP tool
  [ ] Restore watches on server startup (WatchManager::init)
  [ ] Graceful shutdown (stop watchers, flush pending events)
```

### Phase 4b: Context Graph

```
ENTITY EXTRACTION (runs during ingestion, after chunking)
  [ ] Entity extraction pipeline (post-chunk processing step)
  [ ] Regex-based extractors:
      - Date patterns (deadlines, milestones, event dates)
      - Monetary amounts ("$1,250,000.00", "1.25 million dollars")
      - Location references (addresses, regions, countries)
  [ ] NER-based extractors:
      - Person names (stakeholders, contacts, signatories)
      - Organization names (companies, agencies, departments)
      - Concepts and topics (domain-specific terms)
  [ ] Entity deduplication (same entity across chunks/documents)
  [ ] Entity storage in `entities` and `entity_index` column families
  [ ] EntityMention records linking entities to chunks with char offsets

REFERENCE EXTRACTION & NETWORK
  [ ] Reference parser (regex for document references, standards, external sources)
  [ ] Reference normalization (canonical form for dedup)
  [ ] ReferenceRecord storage with source_doc, target, context
  [ ] Reference type classification (Document, Standard, Regulation, Report)
  [ ] Reference network storage in `references` column family
  [ ] Cross-document reference linking (Doc A references same source as Doc B)

KNOWLEDGE GRAPH
  [ ] Entity relationship graph construction
  [ ] Co-occurrence detection (entities appearing in same chunks)
  [ ] Cross-document entity linking
  [ ] Graph storage in `knowledge_graph` column family
  [ ] Graph traversal queries (shortest path, neighbors, clusters)

DOCUMENT GRAPH
  [ ] DocRelationship storage in `doc_graph` column family
  [ ] Relationship types: SharedReferences, SharedEntities, SemanticSimilar, VersionOf, Exhibits
  [ ] Automatic relationship detection during ingestion:
      - SharedReferences: documents referencing same external sources
      - SharedEntities: documents mentioning same entities
      - SemanticSimilar: E1 cosine > 0.75 between document-level embeddings
  [ ] Chunk similarity graph in `chunk_graph` column family
  [ ] Cross-chunk similarity edges (E1 cosine > 0.8 between chunks)

COLLECTION SUMMARY
  [ ] CollectionSummary builder (aggregates entities, references, relationships per collection)
  [ ] Stakeholder extraction and role classification (key people and organizations)
  [ ] Key date extraction and timeline construction
  [ ] Topic extraction from headings, content analysis
  [ ] Reference statistics (most-cited references in the collection)
  [ ] Entity statistics (most-mentioned entities)
  [ ] CollectionStatistics computation (doc count, chunk count, entity count, etc.)
  [ ] Collection summary storage in `collection_summary` column family
  [ ] Incremental collection summary updates (on ingest/delete/reindex)

CONTEXT GRAPH MCP TOOLS (18 tools -- see PRD 09 Section 2b)
  [ ] Collection Overview tools:
      - get_collection_summary (stakeholders, topics, key dates, key references)
      - get_collection_timeline (chronological events extracted from documents)
      - get_collection_statistics (counts, coverage, health metrics)
  [ ] Entity & Reference tools:
      - list_entities (filter by type, sort by mention count)
      - get_entity_mentions (all mentions of an entity across documents)
      - search_entity_relationships (entities connected via shared documents)
      - get_entity_graph (entity relationship visualization)
      - list_references (all referenced external sources with citation counts)
      - get_reference_citations (all documents citing a specific reference)
  [ ] Document Navigation tools:
      - get_document_structure (headings, sections, page count, entity/reference summary)
      - browse_pages (paginated page content with entities highlighted)
      - find_related_documents (documents related via references, entities, or semantics)
      - get_related_documents (knowledge-graph-first document discovery)
      - list_documents_by_type (filter by inferred document type)
      - traverse_chunks (sequential chunk navigation with prev/next)
  [ ] Advanced Search tools:
      - search_similar_chunks (find chunks semantically similar to a given chunk)
      - compare_documents (side-by-side entity, reference, and semantic comparison)
      - find_document_clusters (group documents by topic/entity similarity)
```

### Phase 5: OCR & Polish

```
OCR
  [ ] Tesseract integration
  [ ] Image preprocessing (grayscale, contrast)
  [ ] Scanned PDF detection + automatic OCR
  [ ] ingest_image support
  [ ] OCR confidence in provenance

MEMORY MANAGEMENT
  [ ] System RAM detection (sysinfo)
  [ ] Auto memory tier selection
  [ ] Lazy model loading
  [ ] Memory pressure handling (model unloading)

CROSS-PLATFORM
  [ ] Test on macOS (Intel + Apple Silicon)
  [ ] Test on Windows 10/11
  [ ] Test on Ubuntu
  [ ] Path handling (~ expansion, separators)
  [ ] CoreML / DirectML execution providers
```

### Phase 6: Distribution

```
DISTRIBUTION
  [ ] Install script (install.sh, install.ps1)
  [ ] --setup-claude-code command
  [ ] MCPB bundle creation script
  [ ] manifest.json
  [ ] Release pipeline (GitHub Actions)
  [ ] Cross-platform builds
  [ ] Binary signing (macOS notarization, Windows Authenticode)

UPDATE MECHANISM
  [ ] Version check on startup (non-blocking)
  [ ] --update self-update command
  [ ] --uninstall command
  [ ] Data migration on upgrade

DOCUMENTATION
  [ ] README.md
  [ ] CHANGELOG.md
  [ ] Landing page content
```

---

## 13. Success Metrics

### 13.1 Product Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Install success rate | >95% | Opt-in telemetry |
| Time to first search | <5 minutes | User testing |
| Search relevance (top 5) | >85% | Manual evaluation |
| Provenance accuracy | 100% | Automated tests |
| Crash rate | <0.1% | Error reporting |

### 13.2 Performance Metrics

| Metric | Free Tier | Pro Tier |
|--------|-----------|----------|
| Search latency (p95) | <150ms | <250ms |
| Ingestion speed | <1.5s/page | <1s/page |
| RAM usage (idle) | <500MB | <800MB |
| RAM usage (search) | <1.5GB | <2GB |
| Model download | <3 min | <5 min |
| Binary size | <30MB | <30MB |

### 13.3 Business Metrics

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Downloads | 10,000 | 50,000 | 200,000 |
| Free users | 5,000 | 25,000 | 100,000 |
| Pro conversions (2%) | 100 | 500 | 2,000 |
| ARR | $35K | $174K | $696K |

---

## Appendix A: File Size Estimates

| Component | Size |
|-----------|------|
| Binary (release, stripped) | ~15-25MB |
| MCPB bundle (all platforms) | ~50MB |
| Models (Free tier) | ~165MB |
| Models (Pro tier) | ~370MB |
| Collection database (per 100 docs) | ~5-50MB |
| Total install (Free, 1 collection) | ~200MB |
| Total install (Pro, 10 collections) | ~900MB |

## Appendix B: Comparison with Alternatives

| Feature | CaseTrack | Traditional SaaS | DIY RAG |
|---------|-----------|-------------------|---------|
| Price | $0-29/mo | $200-400/mo | Free |
| Install time | 2 min | N/A (SaaS) | Hours |
| Runs locally | Yes | No | Yes |
| No GPU required | Yes | N/A | Usually no |
| Claude integration | Native MCP | No | Manual |
| Provenance | Always | Sometimes | DIY |
| Document-optimized models | Yes | Varies | No |
| Privacy | 100% local | Cloud | Local |
| Offline capable | Yes | No | Yes |

---

*CaseTrack PRD v4.0.0 -- Document 10 of 10*
