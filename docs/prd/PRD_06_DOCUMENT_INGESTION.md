# PRD 06: Document Ingestion

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Supported Formats

| Format | Method | Quality | Rust Crate | Notes |
|--------|--------|---------|------------|-------|
| PDF (native text) | pdf-extract | Excellent | `pdf-extract`, `lopdf` | Direct text extraction |
| PDF (scanned) | Tesseract OCR | Good (>95%) | `tesseract` | Requires image rendering |
| DOCX | docx-rs | Excellent | `docx-rs` | Preserves structure |
| DOC (legacy) | Convert via LibreOffice | Good | CLI shelling | Optional, warns user |
| XLSX/XLS/ODS | calamine | Excellent | `calamine` | Pure Rust, reads all spreadsheet formats |
| Images (JPG/PNG/TIFF) | Tesseract OCR | Good | `tesseract`, `image` | Single page per image |
| TXT/RTF | Direct read | Excellent | `std::fs` | Plain text, no metadata |

---

## 2. Ingestion Pipeline

```
DOCUMENT INGESTION FLOW
=================================================================================

User: "Ingest ~/Downloads/Report.pdf"
                    |
                    v
+-----------------------------------------------------------------------+
| 1. VALIDATE                                                            |
|    - Check file exists and is readable                                |
|    - Detect file type (by extension + magic bytes)                    |
|    - Check file size (warn if >100MB)                                 |
|    - Check for duplicates (SHA256 hash comparison)                    |
|    Output: ValidatedFile { path, file_type, hash, size }             |
+-----------------------------------------------------------------------+
                    |
                    v
+-----------------------------------------------------------------------+
| 2. PARSE                                                               |
|    - Route to format-specific parser                                  |
|    - Extract text with position metadata                              |
|    - For scanned pages: detect and run OCR                            |
|    - For spreadsheets: extract sheets, rows, and cell data            |
|    - Extract document metadata (title, author, dates)                 |
|    Output: ParsedDocument { pages: Vec<Page>, metadata }              |
+-----------------------------------------------------------------------+
                    |
                    v
+-----------------------------------------------------------------------+
| 3. CHUNK (provenance is attached here -- THE MOST CRITICAL STEP)       |
|    - Split into 2000-character chunks                                  |
|    - 10% overlap (200 chars from end of previous chunk)                |
|    - Respect paragraph and sentence boundaries                         |
|    - Attach FULL provenance to EVERY chunk:                            |
|      * document_path: absolute file path on disk                       |
|      * document_name: original filename                                |
|      * page: page number (1-indexed)                                   |
|      * paragraph_start/end: which paragraphs this chunk spans          |
|      * line_start/end: which lines this chunk spans                    |
|      * char_start/end: exact character offsets within the page          |
|      * sheet_name: sheet name (for spreadsheets)                       |
|      * row_range: row range (for spreadsheets, e.g., rows 1-45)       |
|      * column_range: column range (for spreadsheets)                   |
|      * extraction_method: Native / OCR / Hybrid / Spreadsheet          |
|      * ocr_confidence: quality score for OCR-extracted text             |
|      * created_at: Unix timestamp of chunk creation                     |
|      * embedded_at: Unix timestamp (set after Step 4)                   |
|    A chunk without provenance MUST NOT be stored. Period.              |
|    Output: Vec<Chunk> with Provenance                                  |
+-----------------------------------------------------------------------+
                    |
                    v
+-----------------------------------------------------------------------+
| 4. EMBED                                                               |
|    - Run each chunk through active embedders (3-6 depending on tier) |
|    - Batch for efficiency (32 chunks at a time)                      |
|    - Build BM25 inverted index entries                                |
|    Output: Vec<ChunkWithEmbeddings>                                   |
+-----------------------------------------------------------------------+
                    |
                    v
+-----------------------------------------------------------------------+
| 5. STORE (provenance chain is sealed here)                             |
|    - Write chunks + provenance to collection RocksDB (chunks CF)      |
|      Each chunk stored with its FULL provenance inline                |
|    - Write embedding vectors to embeddings CF, keyed by chunk_id      |
|      chunk_id is the bridge: embedding → chunk → provenance → file    |
|    - Update embedded_at timestamp on each chunk                       |
|    - Write provenance records to provenance CF (prov:{chunk_uuid})    |
|    - Update BM25 inverted index                                       |
|    - Update document metadata (ingested_at, updated_at timestamps)    |
|    - Update collection stats                                          |
|    - Optionally copy original file to collection/originals/           |
|    Output: IngestResult { pages, chunks, duration, timestamps }       |
+-----------------------------------------------------------------------+
                    |
                    v
Response: "Ingested Report.pdf: 45 pages, 234 chunks, 12s"
```

### 2.1 Post-Chunk Processing: Entity Extraction & Knowledge Graph

After Step 5 (STORE), the pipeline extracts entities and builds the **Context Graph** (see [PRD 04 Section 8](PRD_04_STORAGE_ARCHITECTURE.md)).

**Step 6 -- EXTRACT ENTITIES**: For each chunk, run regex extractors (dates, amounts, percentages) and NER (persons, organizations, locations). Deduplicate within the document. Store Entity and EntityMention records in `entities` CF, update `entity_index`. Output: `Vec<Entity>`, `Vec<EntityMention>`.

**Step 7 -- BUILD DOCUMENT GRAPH EDGES**: Find shared entities (SharedEntities edges) across documents and compute document-level E1 similarity (SemanticSimilar edges). Cross-document entity matching links the same entity appearing in different documents. Store in `doc_graph` and `chunk_graph` CFs. Output: `Vec<DocRelationship>`.

**Step 8 -- UPDATE COLLECTION MAP**: Incrementally add new entities, key dates, entity statistics. Recompute CollectionStatistics. Output: Updated CollectionMap.

Complete response: `"Ingested Report.pdf: 45 pages, 234 chunks, 47 entities, 12s"`

#### Entity Types Extracted

| Type | Detection Method | Examples |
|------|-----------------|----------|
| Person | NER | "John Smith", "Sarah Chen" |
| Organization | NER | "Acme Corp", "Finance Department" |
| Date | Regex + NER | "January 15, 2024", "Q3 2024", "filed on 01/15/2024" |
| Amount | Regex | "$1,250,000.00", "1.25 million dollars", "15.7%" |
| Location | NER | "New York, NY", "123 Main Street", "United Kingdom" |
| Concept | NER | "supply chain optimization", "revenue forecast" |

#### Knowledge Graph Integration During Ingestion

After entity extraction, the pipeline builds knowledge graph connections:

| Step | Description |
|------|-------------|
| Entity-to-Chunk edges | Each extracted entity links to the chunk(s) where it appears |
| Cross-document entity matching | Same entity (e.g., "Acme Corp") across multiple documents creates shared-entity edges |
| Document relationship edges | Documents sharing 3+ entities are linked with SharedEntities relationship |
| Entity co-occurrence | Entities appearing in the same chunk are linked with co-occurrence edges |

---

## 3. PDF Processing

```rust
use lopdf::Document as PdfDocument;

pub struct PdfProcessor {
    ocr_enabled: bool,
    ocr_language: String,  // "eng" default
}

impl PdfProcessor {
    pub fn process(&self, path: &Path) -> Result<ParsedDocument> {
        let pdf = PdfDocument::load(path)
            .map_err(|e| CaseTrackError::PdfParseError {
                path: path.to_path_buf(),
                source: e,
            })?;

        let page_count = pdf.get_pages().len();
        let mut pages = Vec::with_capacity(page_count);
        let metadata = self.extract_pdf_metadata(&pdf)?;

        for page_num in 1..=page_count {
            // Try native text extraction first
            let native_text = pdf_extract::extract_text_from_page(&pdf, page_num)
                .unwrap_or_default();

            let trimmed = native_text.trim();

            if trimmed.is_empty() || self.looks_like_scanned(trimmed) {
                if self.ocr_enabled {
                    // Scanned page -- use OCR
                    let image = self.render_page_to_image(&pdf, page_num)?;
                    let ocr_result = self.run_ocr(&image)?;
                    pages.push(Page {
                        number: page_num as u32,
                        content: ocr_result.text,
                        paragraphs: self.detect_paragraphs(&ocr_result.text),
                        extraction_method: ExtractionMethod::Ocr,
                        ocr_confidence: Some(ocr_result.confidence),
                    });
                } else {
                    // OCR disabled -- store empty page with warning
                    tracing::warn!(
                        "Page {} appears scanned but OCR is disabled. Skipping.",
                        page_num
                    );
                    pages.push(Page {
                        number: page_num as u32,
                        content: String::new(),
                        paragraphs: vec![],
                        extraction_method: ExtractionMethod::Skipped,
                        ocr_confidence: None,
                    });
                }
            } else {
                pages.push(Page {
                    number: page_num as u32,
                    content: native_text,
                    paragraphs: self.detect_paragraphs(&native_text),
                    extraction_method: ExtractionMethod::Native,
                    ocr_confidence: None,
                });
            }
        }

        Ok(ParsedDocument {
            id: Uuid::new_v4(),
            filename: path.file_name().unwrap().to_string_lossy().to_string(),
            pages,
            metadata,
            file_hash: compute_sha256(path)?,
        })
    }

    /// Heuristic: if extracted text is mostly whitespace or control chars, it's scanned
    fn looks_like_scanned(&self, text: &str) -> bool {
        let alpha_ratio = text.chars().filter(|c| c.is_alphanumeric()).count() as f32
            / text.len().max(1) as f32;
        alpha_ratio < 0.3
    }

    fn extract_pdf_metadata(&self, pdf: &PdfDocument) -> Result<DocumentMetadataRaw> {
        // Extract from PDF info dictionary if present
        let info = pdf.trailer.get(b"Info")
            .and_then(|r| pdf.get_object(r.as_reference().ok()?).ok());

        Ok(DocumentMetadataRaw {
            title: self.get_pdf_string(&info, b"Title"),
            author: self.get_pdf_string(&info, b"Author"),
            created_date: self.get_pdf_string(&info, b"CreationDate"),
        })
    }
}
```

---

## 4. DOCX Processing

```rust
pub struct DocxProcessor;

impl DocxProcessor {
    pub fn process(&self, path: &Path) -> Result<ParsedDocument> {
        let docx = docx_rs::read_docx(&fs::read(path)?)
            .map_err(|e| CaseTrackError::DocxParseError {
                path: path.to_path_buf(),
                source: e,
            })?;

        let mut pages = vec![];
        let mut current_page = Page::new(1);
        let mut para_idx = 0;

        for element in &docx.document.children {
            match element {
                DocumentChild::Paragraph(para) => {
                    let text = self.extract_paragraph_text(para);
                    if !text.trim().is_empty() {
                        current_page.paragraphs.push(Paragraph {
                            index: para_idx,
                            text: text.clone(),
                            style: self.detect_style(para),
                        });
                        current_page.content.push_str(&text);
                        current_page.content.push('\n');
                        para_idx += 1;
                    }
                }
                DocumentChild::SectionProperty(sp) => {
                    // Section break = new page (approximate)
                    if !current_page.content.is_empty() {
                        pages.push(current_page);
                        current_page = Page::new(pages.len() as u32 + 1);
                    }
                }
                _ => {}
            }
        }

        // Don't forget the last page
        if !current_page.content.is_empty() {
            pages.push(current_page);
        }

        Ok(ParsedDocument {
            id: Uuid::new_v4(),
            filename: path.file_name().unwrap().to_string_lossy().to_string(),
            pages,
            metadata: DocumentMetadataRaw::default(),
            file_hash: compute_sha256(path)?,
        })
    }
}
```

---

## 5. XLSX/Excel Processing

```rust
use calamine::{open_workbook_auto, Reader, DataType};

pub struct XlsxProcessor;

impl XlsxProcessor {
    pub fn process(&self, path: &Path) -> Result<ParsedDocument> {
        let mut workbook = open_workbook_auto(path)
            .map_err(|e| CaseTrackError::SpreadsheetParseError {
                path: path.to_path_buf(),
                source: e,
            })?;

        let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
        let mut pages = Vec::with_capacity(sheet_names.len());

        for (sheet_idx, sheet_name) in sheet_names.iter().enumerate() {
            let range = workbook.worksheet_range(sheet_name)
                .map_err(|e| CaseTrackError::SpreadsheetParseError {
                    path: path.to_path_buf(),
                    source: e,
                })?;

            // Detect headers from first row
            let headers: Vec<String> = range.rows().next()
                .map(|row| row.iter().map(|cell| cell.to_string()).collect())
                .unwrap_or_default();

            let mut content = String::new();
            let mut paragraphs = Vec::new();
            let mut para_idx = 0;

            for (row_idx, row) in range.rows().enumerate() {
                let row_text: Vec<String> = row.iter()
                    .enumerate()
                    .filter(|(_, cell)| !cell.is_empty())
                    .map(|(col_idx, cell)| {
                        let header = headers.get(col_idx)
                            .filter(|h| !h.is_empty() && row_idx > 0);
                        match header {
                            Some(h) => format!("{}: {}", h, cell),
                            None => cell.to_string(),
                        }
                    })
                    .collect();

                if !row_text.is_empty() {
                    let line = row_text.join(" | ");
                    paragraphs.push(Paragraph {
                        index: para_idx,
                        text: line.clone(),
                        style: if row_idx == 0 { ParagraphStyle::Heading } else { ParagraphStyle::Body },
                    });
                    content.push_str(&line);
                    content.push('\n');
                    para_idx += 1;
                }
            }

            // Each sheet becomes a logical "page"
            pages.push(Page {
                number: (sheet_idx + 1) as u32,
                content,
                paragraphs,
                extraction_method: ExtractionMethod::Spreadsheet,
                ocr_confidence: None,
            });
        }

        Ok(ParsedDocument {
            id: Uuid::new_v4(),
            filename: path.file_name().unwrap().to_string_lossy().to_string(),
            pages,
            metadata: DocumentMetadataRaw::default(),
            file_hash: compute_sha256(path)?,
        })
    }
}
```

**Spreadsheet provenance**: Each chunk from a spreadsheet includes `sheet_name`, `row_range` (e.g., rows 1-45), and `column_range` in its provenance record, enabling precise traceability back to specific cells.

---

## 6. OCR (Tesseract)

### 6.1 Bundling Strategy

| Platform | Method |
|----------|--------|
| macOS | Statically linked via `leptonica-sys` + `tesseract-sys` |
| Windows | Tesseract DLLs in installer/MCPB bundle |
| Linux | Statically linked via musl build |

The `eng.traineddata` (~15MB) is bundled or downloaded on first OCR use.

### 6.2 OCR Pipeline

```rust
pub struct OcrEngine {
    tesseract: tesseract::Tesseract,
}

impl OcrEngine {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let tessdata = data_dir.join("models").join("tessdata");
        let tesseract = tesseract::Tesseract::new(tessdata.to_str().unwrap(), "eng")?;
        Ok(Self { tesseract })
    }

    pub fn recognize(&self, image: &image::DynamicImage) -> Result<OcrResult> {
        let processed = self.preprocess(image);
        let bytes = processed.to_luma8();

        let mut tess = self.tesseract.clone();
        tess.set_image(bytes.as_raw(), bytes.width() as i32, bytes.height() as i32, 1, bytes.width() as i32)?;

        Ok(OcrResult {
            text: tess.get_text()?,
            confidence: tess.mean_text_conf() as f32 / 100.0,
        })
    }

    fn preprocess(&self, image: &image::DynamicImage) -> image::DynamicImage {
        image.grayscale().adjust_contrast(1.5)
    }
}
```

---

## 7. Chunking Strategy

### 7.1 Chunking Rules (MANDATORY)

| Parameter | Value |
|-----------|-------|
| Target size | 2000 characters |
| Overlap | 10% = 200 characters (from end of previous chunk) |
| Min size | 400 characters (no tiny fragments) |
| Max size | 2200 characters (small overrun to avoid mid-sentence splits) |

Character-based (not token-based) for deterministic, reproducible chunking.

**Boundary priority**: (1) paragraph break, (2) sentence boundary, (3) word boundary. Never split mid-word. Chunks do NOT cross page boundaries.

### 7.2 Provenance Per Chunk (MANDATORY)

**Every chunk MUST store its complete provenance at creation time.** Fields: `document_id`, `document_name`, `document_path`, `page`, `paragraph_start/end`, `line_start/end`, `char_start/end`, `extraction_method`, `ocr_confidence`, `sheet_name` (spreadsheets), `row_range` (spreadsheets), `column_range` (spreadsheets), `chunk_index`.

Provenance is: (1) stored in RocksDB with chunk text and embeddings, (2) returned in every search result, (3) queryable via MCP tools, (4) immutable after creation. See [PRD 04 Section 5.2](PRD_04_STORAGE_ARCHITECTURE.md) for the canonical Provenance struct and storage layout.

### 7.3 Chunking Implementation

```rust
/// Chunker configuration: 2000 chars, 10% overlap
pub struct DocumentChunker {
    target_chars: usize,   // 2000
    max_chars: usize,      // 2200 (small overrun to avoid mid-sentence)
    min_chars: usize,      // 400 (don't emit tiny fragments)
    overlap_chars: usize,  // 200 (10% of target)
}

impl Default for DocumentChunker {
    fn default() -> Self {
        Self {
            target_chars: 2000,
            max_chars: 2200,
            min_chars: 400,
            overlap_chars: 200,  // 10% overlap
        }
    }
}

impl DocumentChunker {
    pub fn chunk(&self, doc: &ParsedDocument) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut chunk_seq: u32 = 0;

        for page in &doc.pages {
            if page.content.trim().is_empty() {
                continue;
            }

            // Track character offset within the page
            let mut page_char_offset: u64 = 0;

            let paragraphs = &page.paragraphs;
            let mut current_text = String::new();
            let mut current_start_para: u32 = 0;
            let mut current_start_line: u32 = 0;
            let mut current_char_start: u64 = 0;

            for (para_idx, paragraph) in paragraphs.iter().enumerate() {
                let para_chars = paragraph.text.len();

                // Single paragraph exceeds max? Split by sentence
                if para_chars > self.max_chars {
                    // Flush current chunk first
                    if current_text.len() >= self.min_chars {
                        chunks.push(self.make_chunk(
                            doc, page, &current_text, chunk_seq,
                            current_start_para, para_idx.saturating_sub(1) as u32,
                            current_start_line,
                            current_char_start,
                        ));
                        chunk_seq += 1;
                    }

                    // Split long paragraph by sentences
                    let sub_chunks = self.split_long_paragraph(
                        doc, page, paragraph, para_idx as u32,
                        page_char_offset, &mut chunk_seq,
                    );
                    chunks.extend(sub_chunks);

                    page_char_offset += para_chars as u64;
                    current_text.clear();
                    current_start_para = (para_idx + 1) as u32;
                    current_char_start = page_char_offset;
                    continue;
                }

                // Would adding this paragraph exceed 2000 chars?
                if current_text.len() + para_chars > self.target_chars
                    && current_text.len() >= self.min_chars
                {
                    // Emit current chunk
                    chunks.push(self.make_chunk(
                        doc, page, &current_text, chunk_seq,
                        current_start_para, para_idx.saturating_sub(1) as u32,
                        current_start_line,
                        current_char_start,
                    ));
                    chunk_seq += 1;

                    // Start new chunk with 200-char overlap from end of previous
                    let overlap = self.compute_overlap(&current_text);
                    current_text = overlap;
                    current_start_para = para_idx as u32;
                    current_char_start = page_char_offset.saturating_sub(self.overlap_chars as u64);
                }

                current_text.push_str(&paragraph.text);
                current_text.push('\n');
                page_char_offset += para_chars as u64 + 1; // +1 for newline
            }

            // Emit remaining text for this page
            if current_text.len() >= self.min_chars {
                chunks.push(self.make_chunk(
                    doc, page, &current_text, chunk_seq,
                    current_start_para, paragraphs.len().saturating_sub(1) as u32,
                    current_start_line,
                    current_char_start,
                ));
                chunk_seq += 1;
            }
        }

        chunks
    }

    fn make_chunk(
        &self,
        doc: &ParsedDocument,
        page: &Page,
        text: &str,
        sequence: u32,
        para_start: u32,
        para_end: u32,
        line_start: u32,
        char_start: u64,
    ) -> Chunk {
        let line_end = line_start + text.lines().count() as u32;
        let char_end = char_start + text.len() as u64;

        Chunk {
            id: Uuid::new_v4(),
            document_id: doc.id,
            text: text.to_string(),
            sequence,
            char_count: text.len() as u32,
            provenance: Provenance {
                document_id: doc.id,
                document_name: doc.filename.clone(),
                document_path: doc.original_path.clone(),
                page: page.number,
                paragraph_start: para_start,
                paragraph_end: para_end,
                line_start,
                line_end,
                char_start,
                char_end,
                extraction_method: page.extraction_method,
                ocr_confidence: page.ocr_confidence,
                chunk_index: sequence,
            },
        }
    }

    /// Take last 200 characters as overlap for next chunk
    fn compute_overlap(&self, text: &str) -> String {
        if text.len() <= self.overlap_chars {
            return text.to_string();
        }
        let start = text.len() - self.overlap_chars;
        // Find nearest word boundary after the cut point
        let boundary = text[start..].find(' ').map(|i| start + i + 1).unwrap_or(start);
        text[boundary..].to_string()
    }

    /// Split a paragraph longer than 2200 chars into sentence-bounded chunks
    fn split_long_paragraph(
        &self,
        doc: &ParsedDocument,
        page: &Page,
        paragraph: &Paragraph,
        para_idx: u32,
        char_offset: u64,
        chunk_seq: &mut u32,
    ) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let sentences = split_sentences(&paragraph.text);
        let mut current = String::new();
        let mut local_offset = 0u64;

        for sentence in &sentences {
            if current.len() + sentence.len() > self.target_chars && current.len() >= self.min_chars {
                chunks.push(self.make_chunk(
                    doc, page, &current, *chunk_seq,
                    para_idx, para_idx,
                    0, // line tracking within paragraph
                    char_offset + local_offset,
                ));
                *chunk_seq += 1;

                let overlap = self.compute_overlap(&current);
                local_offset += (current.len() - overlap.len()) as u64;
                current = overlap;
            }
            current.push_str(sentence);
        }

        if current.len() >= self.min_chars {
            chunks.push(self.make_chunk(
                doc, page, &current, *chunk_seq,
                para_idx, para_idx,
                0,
                char_offset + local_offset,
            ));
            *chunk_seq += 1;
        }

        chunks
    }
}

/// Split text into sentences (period/question/exclamation + space + uppercase)
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if (ch == '.' || ch == '?' || ch == '!') {
            // Check if next char is space + uppercase (sentence boundary)
            // Simplified: just split at sentence-ending punctuation
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}
```

---

## 8. Batch Ingestion (Pro Tier)

```rust
/// Ingest all supported files in a directory
pub async fn ingest_folder(
    collection: &mut CollectionHandle,
    engine: &EmbeddingEngine,
    folder: &Path,
    recursive: bool,
) -> Result<BatchIngestResult> {
    let files = discover_files(folder, recursive)?;
    let total = files.len();
    let mut results = Vec::new();
    let mut errors = Vec::new();

    for (idx, file) in files.iter().enumerate() {
        tracing::info!("[{}/{}] Ingesting: {}", idx + 1, total, file.display());

        match ingest_single_file(collection, engine, file).await {
            Ok(result) => results.push(result),
            Err(e) => {
                tracing::error!("Failed to ingest {}: {}", file.display(), e);
                errors.push(IngestError {
                    file: file.clone(),
                    error: e.to_string(),
                });
            }
        }
    }

    Ok(BatchIngestResult {
        total_files: total,
        succeeded: results.len(),
        failed: errors.len(),
        results,
        errors,
    })
}

fn discover_files(folder: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let supported = &[
        "pdf", "docx", "doc", "xlsx", "xls", "ods",
        "txt", "rtf", "jpg", "jpeg", "png", "tiff", "tif",
    ];

    let walker = if recursive {
        walkdir::WalkDir::new(folder)
    } else {
        walkdir::WalkDir::new(folder).max_depth(1)
    };

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if supported.contains(&ext.to_lowercase().as_str()) {
                    files.push(entry.into_path());
                }
            }
        }
    }

    files.sort(); // Deterministic order
    Ok(files)
}
```

---

## 9. Duplicate Detection

Check SHA256 hash against existing documents before ingesting. If duplicate found, return error with existing document ID and `--force` hint.

```rust
pub fn check_duplicate(collection: &CollectionHandle, file_hash: &str) -> Result<Option<Uuid>> {
    let cf = collection.db.cf_handle("documents").unwrap();
    for item in collection.db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
        let (_, value) = item?;
        let doc: DocumentMetadata = bincode::deserialize(&value)?;
        if doc.file_hash == file_hash {
            return Ok(Some(doc.id));
        }
    }
    Ok(None)
}
```

---

## 10. Data Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub id: Uuid,
    pub filename: String,
    pub pages: Vec<Page>,
    pub metadata: DocumentMetadataRaw,
    pub file_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub number: u32,
    pub content: String,
    pub paragraphs: Vec<Paragraph>,
    pub extraction_method: ExtractionMethod,
    pub ocr_confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paragraph {
    pub index: usize,
    pub text: String,
    pub style: ParagraphStyle,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ParagraphStyle {
    Body,
    Heading,
    ListItem,
    BlockQuote,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExtractionMethod {
    Native,       // Direct text extraction from PDF/DOCX
    Ocr,          // Tesseract OCR
    Spreadsheet,  // calamine spreadsheet extraction
    Skipped,      // OCR disabled, scanned page skipped
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub document_id: Uuid,
    pub text: String,
    pub sequence: u32,        // Position within document
    pub char_count: u32,      // Length in characters (target: 2000)
    pub provenance: Provenance, // FULL source location (MANDATORY)
}

#[derive(Debug, Serialize)]
pub struct IngestResult {
    pub document_id: Uuid,
    pub document_name: String,
    pub page_count: u32,
    pub chunk_count: u32,
    pub extraction_method: ExtractionMethod,
    pub ocr_pages: u32,
    pub duration_ms: u64,
}
```

---

*CaseTrack PRD v4.0.0 -- Document 6 of 10*
