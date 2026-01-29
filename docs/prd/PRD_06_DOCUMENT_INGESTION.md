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
| Images (JPG/PNG/TIFF) | Tesseract OCR | Good | `tesseract`, `image` | Single page per image |
| TXT/RTF | Direct read | Excellent | `std::fs` | Plain text, no metadata |

---

## 2. Ingestion Pipeline

```
DOCUMENT INGESTION FLOW
=================================================================================

User: "Ingest ~/Downloads/Complaint.pdf"
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
|    - Extract document metadata (title, author, dates)                 |
|    Output: ParsedDocument { pages: Vec<Page>, metadata }              |
+-----------------------------------------------------------------------+
                    |
                    v
+-----------------------------------------------------------------------+
| 3. CHUNK                                                               |
|    - Split into ~500 token chunks                                     |
|    - Respect paragraph and sentence boundaries                        |
|    - Attach provenance (doc, page, para, line, char offset)          |
|    - Add 50-token overlap between consecutive chunks                  |
|    Output: Vec<Chunk> with Provenance                                 |
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
| 5. STORE                                                               |
|    - Write chunks + embeddings to case RocksDB                        |
|    - Write provenance records                                         |
|    - Update BM25 inverted index                                       |
|    - Update document metadata and case stats                          |
|    - Optionally copy original file to case/originals/                 |
|    Output: IngestResult { pages, chunks, duration }                   |
+-----------------------------------------------------------------------+
                    |
                    v
Response: "Ingested Complaint.pdf: 45 pages, 234 chunks, 12s"
```

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

## 5. OCR (Tesseract)

### 5.1 Bundling Strategy

Tesseract is bundled with the CaseTrack binary:
- **macOS**: Statically linked via `leptonica-sys` and `tesseract-sys`
- **Windows**: Tesseract DLLs included in installer/MCPB bundle
- **Linux**: Statically linked via musl build

The `eng.traineddata` language model (~15MB) is included in the MCPB bundle or downloaded on first OCR use.

### 5.2 OCR Pipeline

```rust
pub struct OcrEngine {
    tesseract: tesseract::Tesseract,
}

impl OcrEngine {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let tessdata = data_dir.join("models").join("tessdata");
        let tesseract = tesseract::Tesseract::new(
            tessdata.to_str().unwrap(),
            "eng",
        )?;
        Ok(Self { tesseract })
    }

    pub fn recognize(&self, image: &image::DynamicImage) -> Result<OcrResult> {
        // Preprocess image for better OCR accuracy
        let processed = self.preprocess(image);

        // Convert to bytes
        let bytes = processed.to_luma8();

        let mut tess = self.tesseract.clone();
        tess.set_image(
            bytes.as_raw(),
            bytes.width() as i32,
            bytes.height() as i32,
            1,  // bytes per pixel
            bytes.width() as i32,  // bytes per line
        )?;

        let text = tess.get_text()?;
        let confidence = tess.mean_text_conf();

        Ok(OcrResult {
            text,
            confidence: confidence as f32 / 100.0,
        })
    }

    /// Image preprocessing for better OCR results
    fn preprocess(&self, image: &image::DynamicImage) -> image::DynamicImage {
        image
            .grayscale()          // Convert to grayscale
            .adjust_contrast(1.5) // Increase contrast
            // Binarization handled by Tesseract internally
    }
}
```

---

## 6. Chunking Strategy

### 6.1 Legal-Aware Chunking

```rust
pub struct LegalChunker {
    target_tokens: usize,  // 500
    max_tokens: usize,     // 1000
    min_tokens: usize,     // 100
    overlap_tokens: usize, // 50
}

impl LegalChunker {
    pub fn chunk(&self, doc: &ParsedDocument) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut chunk_seq = 0;

        for page in &doc.pages {
            if page.content.trim().is_empty() {
                continue;
            }

            let paragraphs = &page.paragraphs;
            let mut current_text = String::new();
            let mut current_start_para = 0;
            let mut current_start_line = 0;
            let mut current_token_count = 0;

            for (para_idx, paragraph) in paragraphs.iter().enumerate() {
                let para_tokens = count_tokens(&paragraph.text);

                // Single paragraph exceeds max? Split it
                if para_tokens > self.max_tokens {
                    // Flush current chunk first
                    if !current_text.is_empty() {
                        chunks.push(self.make_chunk(
                            doc, page, &current_text, chunk_seq,
                            current_start_para, para_idx.saturating_sub(1),
                            current_start_line,
                        ));
                        chunk_seq += 1;
                    }

                    // Split long paragraph by sentences
                    let sub_chunks = self.split_long_paragraph(
                        doc, page, paragraph, para_idx, &mut chunk_seq,
                    );
                    chunks.extend(sub_chunks);

                    current_text.clear();
                    current_token_count = 0;
                    current_start_para = para_idx + 1;
                    continue;
                }

                // Would adding this paragraph exceed target?
                if current_token_count + para_tokens > self.target_tokens
                    && !current_text.is_empty()
                    && current_token_count >= self.min_tokens
                {
                    // Emit current chunk
                    chunks.push(self.make_chunk(
                        doc, page, &current_text, chunk_seq,
                        current_start_para, para_idx.saturating_sub(1),
                        current_start_line,
                    ));
                    chunk_seq += 1;

                    // Start new chunk with overlap
                    let overlap = self.compute_overlap(&current_text);
                    current_text = overlap;
                    current_token_count = count_tokens(&current_text);
                    current_start_para = para_idx;
                }

                current_text.push_str(&paragraph.text);
                current_text.push('\n');
                current_token_count += para_tokens;
            }

            // Emit remaining text for this page
            if !current_text.is_empty() && count_tokens(&current_text) >= self.min_tokens {
                chunks.push(self.make_chunk(
                    doc, page, &current_text, chunk_seq,
                    current_start_para, paragraphs.len().saturating_sub(1),
                    current_start_line,
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
        para_start: usize,
        para_end: usize,
        line_start: usize,
    ) -> Chunk {
        let line_end = line_start + text.lines().count();

        Chunk {
            id: Uuid::new_v4(),
            document_id: doc.id,
            text: text.to_string(),
            sequence,
            token_count: count_tokens(text) as u32,
            provenance: Provenance {
                document_id: doc.id,
                document_name: doc.filename.clone(),
                document_path: None,
                page: page.number,
                paragraph_start: para_start as u32,
                paragraph_end: para_end as u32,
                line_start: line_start as u32,
                line_end: line_end as u32,
                char_start: 0,   // Computed during storage
                char_end: 0,
                extraction_method: page.extraction_method,
                ocr_confidence: page.ocr_confidence,
                bates_number: None,
            },
        }
    }

    fn compute_overlap(&self, text: &str) -> String {
        // Take last N tokens as overlap
        let words: Vec<&str> = text.split_whitespace().collect();
        let overlap_words = words.len().min(self.overlap_tokens);
        words[words.len() - overlap_words..].join(" ")
    }
}
```

### 6.2 Token Counting

Use a fast approximation (not full tokenizer) for chunking decisions:

```rust
/// Fast approximate token count (whitespace + punctuation splitting)
/// Full tokenizer is only used during embedding inference
pub fn count_tokens(text: &str) -> usize {
    // Approximate: 1 token ~ 4 characters for English text
    // More accurate than word count for legal text with long words
    (text.len() + 3) / 4
}
```

---

## 7. Batch Ingestion (Pro Tier)

```rust
/// Ingest all supported files in a directory
pub async fn ingest_folder(
    case: &mut CaseHandle,
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

        match ingest_single_file(case, engine, file).await {
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
    let supported = &["pdf", "docx", "doc", "txt", "rtf", "jpg", "jpeg", "png", "tiff", "tif"];

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

## 8. Duplicate Detection

Before ingesting, check if the document already exists in the case:

```rust
pub fn check_duplicate(case: &CaseHandle, file_hash: &str) -> Result<Option<Uuid>> {
    let cf = case.db.cf_handle("documents").unwrap();
    let iter = case.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);

    for item in iter {
        let (_, value) = item?;
        let doc: DocumentMetadata = bincode::deserialize(&value)?;
        if doc.file_hash == file_hash {
            return Ok(Some(doc.id));
        }
    }

    Ok(None)
}
```

If duplicate is found, return an error with the existing document ID:

```
"Document already ingested as 'Complaint.pdf' (ID: abc-123).
 Use --force to re-ingest."
```

---

## 9. Data Types

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
    Native,   // Direct text extraction from PDF/DOCX
    Ocr,      // Tesseract OCR
    Skipped,  // OCR disabled, scanned page skipped
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub document_id: Uuid,
    pub text: String,
    pub sequence: u32,
    pub token_count: u32,
    pub provenance: Provenance,
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
