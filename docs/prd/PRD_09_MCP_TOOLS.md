# PRD 09: MCP Tools

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Tool Overview

| Tool | Description | Tier | Requires Active Collection |
|------|-------------|------|---------------------------|
| `create_collection` | Create a new document collection | Free | No |
| `list_collections` | List all collections | Free | No |
| `switch_collection` | Switch active collection | Free | No |
| `delete_collection` | Delete a collection and all its data | Free | No |
| `get_collection_info` | Get details about active collection | Free | Yes |
| `ingest_document` | Ingest a PDF, DOCX, XLSX, or image | Free | Yes |
| `ingest_folder` | Ingest all supported files in a folder and subfolders | Free | Yes |
| `sync_folder` | Sync a folder -- ingest new/changed files, optionally remove deleted | Free | Yes |
| `list_documents` | List documents in active collection | Free | Yes |
| `get_document` | Get document details and stats | Free | Yes |
| `delete_document` | Remove a document from a collection | Free | Yes |
| `search_documents` | Search across all documents | Free (limited) | Yes |
| `find_entity` | Find mentions of an entity across documents | Pro | Yes |
| `get_chunk` | Get a specific chunk with full provenance | Free | Yes |
| `get_document_chunks` | List all chunks in a document with provenance | Free | Yes |
| `get_source_context` | Get surrounding text for a chunk (context window) | Free | Yes |
| `reindex_document` | Delete old embeddings/indexes for a document and rebuild from scratch | Free | Yes |
| `reindex_collection` | Rebuild all embeddings and indexes for the entire active collection | Free | Yes |
| `get_index_status` | Show embedding/index health for all documents in active collection | Free | Yes |
| `watch_folder` | Start watching a folder for file changes -- auto-sync on change or schedule | Free | Yes |
| `unwatch_folder` | Stop watching a folder | Free | Yes |
| `list_watches` | List all active folder watches and their sync status | Free | No |
| `set_sync_schedule` | Set the auto-sync schedule (on_change, hourly, daily, manual) | Free | Yes |
| `get_status` | Get server status and model info | Free | No |
| | | | |
| **--- Context Graph: Collection Overview ---** | | | |
| `get_collection_summary` | High-level collection briefing: key stakeholders, key dates, topics, document categories, top entities, key references, statistics | Free | Yes |
| `get_collection_timeline` | Chronological view of key dates and events extracted from documents | Free | Yes |
| `get_collection_statistics` | Document counts, page counts, chunk counts, entity counts, reference counts, embedder coverage | Free | Yes |
| | | | |
| **--- Context Graph: Entity & Reference Search ---** | | | |
| `list_entities` | List all extracted entities in the collection, grouped by type (person, org, date, amount, etc.) | Free | Yes |
| `get_entity_mentions` | Get all chunks mentioning a specific entity, with context snippets | Free | Yes |
| `search_entity_relationships` | Find chunks mentioning two or more entities together | Pro | Yes |
| `get_entity_graph` | Show entity relationships across documents in the collection | Pro | Yes |
| `list_references` | List all referenced external sources (documents, standards, regulations) with reference counts | Free | Yes |
| `get_reference_citations` | Get all chunks citing a specific reference, with context | Free | Yes |
| | | | |
| **--- Context Graph: Document Navigation ---** | | | |
| `get_document_structure` | Get headings, sections, and table of contents for a document | Free | Yes |
| `browse_pages` | Get all chunks from a specific page range within a document | Free | Yes |
| `find_related_documents` | Find documents similar to a given document (by shared entities, references, or semantic similarity) | Free | Yes |
| `get_related_documents` | Given a document, find related docs via knowledge graph (shared entities, references) | Free | Yes |
| `list_documents_by_type` | List documents filtered by type (contract, report, spreadsheet, etc.) | Free | Yes |
| `traverse_chunks` | Navigate forward/backward through chunks in a document from a starting point | Free | Yes |
| | | | |
| **--- Context Graph: Advanced Search ---** | | | |
| `search_similar_chunks` | Find chunks semantically similar to a given chunk across all documents | Free | Yes |
| `compare_documents` | Compare what two documents say about a topic (side-by-side search) | Pro | Yes |
| `find_document_clusters` | Group documents by theme/topic using semantic clustering | Pro | Yes |

---

## 2. Tool Specifications

> **PROVENANCE IN EVERY RESPONSE**: Every MCP tool that returns text from a document
> MUST include the full provenance chain: source document filename, file path on disk,
> page number, paragraph range, line range, character offsets, extraction method, OCR
> confidence (if applicable), and timestamps (when ingested, when last embedded).
> A tool response that returns document text without telling the user exactly where
> it came from is a **bug**. The AI must always be able to cite its sources.

### Common Error Patterns

All tools return errors in a consistent MCP format. The four common error types:

```json
// NoCollectionActive -- returned by any tool that requires an active collection
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "No active collection. Create or switch to a collection first:\n  - create_collection: Create a new collection\n  - switch_collection: Switch to an existing collection\n  - list_collections: See all collections"
  }]
}

// FileNotFound -- returned by ingest_document, reindex_document, etc.
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "File not found: /Users/sarah/Downloads/Contract.pdf\n\nCheck that the path is correct and the file exists."
  }]
}

// FreeTierLimit -- returned when a free tier quota is exceeded
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "Free tier allows 3 collections (you have 3). Delete a collection or upgrade to Pro for unlimited collections: https://casetrack.dev/upgrade"
  }]
}

// NotFound -- returned when a collection, document, or chunk ID is not found
{
  "isError": true,
  "content": [{
    "type": "text",
    "text": "Collection not found: \"Acme\". Did you mean:\n  - Acme Corp Partnership (ID: a1b2c3d4)\nUse the full name or ID."
  }]
}
```

Per-tool error examples are omitted below; all errors follow these patterns.

---

### 2.1 `create_collection`

```json
{
  "name": "create_collection",
  "description": "Create a new document collection. Creates an isolated database for this collection on your machine. Automatically switches to the new collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Collection name (e.g., 'Project Alpha', 'Acme Corp Partnership')"
      },
      "collection_id": {
        "type": "string",
        "description": "Optional identifier or reference number"
      },
      "collection_type": {
        "type": "string",
        "enum": ["project", "contract", "financial", "compliance", "research", "hr", "operations", "other"],
        "description": "Type of document collection"
      }
    },
    "required": ["name"]
  }
}
```

**Success Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Created collection \"Acme Corp Partnership\" (ID: a1b2c3d4).\nType: Contract\nThis is now your active collection.\n\nNext: Ingest documents with ingest_document."
  }]
}
```

---

### 2.2 `list_collections`

```json
{
  "name": "list_collections",
  "description": "List all collections. Shows name, type, status, document count, and which collection is active.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "status_filter": {
        "type": "string",
        "enum": ["active", "closed", "archived", "all"],
        "default": "active",
        "description": "Filter by collection status"
      }
    }
  }
}
```

---

### 2.3 `switch_collection`

```json
{
  "name": "switch_collection",
  "description": "Switch to a different collection. All subsequent operations (ingest, search) will use this collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "collection_name": {
        "type": "string",
        "description": "Collection name or ID to switch to"
      }
    },
    "required": ["collection_name"]
  }
}
```

---

### 2.4 `delete_collection`

```json
{
  "name": "delete_collection",
  "description": "Permanently delete a collection and all its documents, embeddings, and data. This cannot be undone.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "collection_name": {
        "type": "string",
        "description": "Collection name or ID to delete"
      },
      "confirm": {
        "type": "boolean",
        "description": "Must be true to confirm deletion",
        "default": false
      }
    },
    "required": ["collection_name", "confirm"]
  }
}
```

---

### 2.5 `get_collection_info`

```json
{
  "name": "get_collection_info",
  "description": "Get detailed information about the active collection including document list and storage usage.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### 2.6 `ingest_document`

```json
{
  "name": "ingest_document",
  "description": "Ingest a document (PDF, DOCX, XLSX, or image) into the active collection. Extracts text (with OCR for scans), chunks the text, computes embeddings, and indexes for search. All processing and storage happens locally on your machine.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Absolute path to the file on your computer"
      },
      "document_name": {
        "type": "string",
        "description": "Optional display name (defaults to filename)"
      },
      "document_type": {
        "type": "string",
        "enum": ["contract", "report", "spreadsheet", "presentation", "correspondence", "memo", "proposal", "invoice", "policy", "other"],
        "description": "Type of document"
      },
      "copy_original": {
        "type": "boolean",
        "default": false,
        "description": "Copy the original file into the collection folder"
      }
    },
    "required": ["file_path"]
  }
}
```

---

### 2.7 `ingest_folder`

```json
{
  "name": "ingest_folder",
  "description": "Ingest all supported documents in a folder and all subfolders. Walks the entire directory tree recursively. Automatically skips files already ingested (matched by SHA256 hash). Supported formats: PDF, DOCX, DOC, XLSX, TXT, RTF, JPG, PNG, TIFF. Each file is chunked into 2000-character segments with full provenance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Absolute path to folder containing documents. All subfolders are included automatically."
      },
      "recursive": {
        "type": "boolean",
        "default": true,
        "description": "Include subfolders (default: true). Set to false to only process the top-level folder."
      },
      "skip_existing": {
        "type": "boolean",
        "default": true,
        "description": "Skip files already ingested (matched by SHA256 hash). Set to false to re-ingest everything."
      },
      "document_type": {
        "type": "string",
        "enum": ["contract", "report", "spreadsheet", "presentation", "correspondence", "memo", "proposal", "invoice", "policy", "other"],
        "description": "Default document type for all files. If omitted, CaseTrack infers from file content."
      },
      "file_extensions": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Optional filter: only ingest files with these extensions (e.g., [\"pdf\", \"docx\", \"xlsx\"]). Default: all supported formats."
      }
    },
    "required": ["folder_path"]
  }
}
```

**Success Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Folder ingestion complete for Acme Corp Partnership\n\n  Folder:     ~/Projects/Acme/Documents/\n  Subfolders: 4 (Contracts/, Reports/, Financials/, Correspondence/)\n  Found:      47 supported files\n  New:        23 (ingested)\n  Skipped:    22 (already ingested, matching SHA256)\n  Failed:     2\n  Duration:   4 minutes 12 seconds\n\n  New documents ingested:\n  - Contracts/Vendor_Agreement.docx (45 pages, 234 chunks)\n  - Contracts/Service_Contract.pdf (12 pages, 67 chunks)\n  - Reports/Q3_Report.xlsx (8 pages, 42 chunks)\n  ... 20 more\n\n  Failures:\n  - Financials/corrupted.pdf: PDF parsing error (file may be corrupted)\n  - Reports/scan_2019.tiff: OCR failed (image too low resolution)\n\nAll 23 new documents are now searchable."
  }]
}
```

---

### 2.8 `sync_folder`

```json
{
  "name": "sync_folder",
  "description": "Sync a folder with the active collection. Compares files on disk against what is already ingested and: (1) ingests new files not yet in the collection, (2) re-ingests files that have changed since last ingestion (detected by SHA256 mismatch), (3) optionally removes documents whose source files no longer exist on disk. This is the easiest way to keep a collection up to date with a directory of documents -- just point it at the folder and run it whenever files change.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Absolute path to folder to sync. All subfolders are included."
      },
      "remove_deleted": {
        "type": "boolean",
        "default": false,
        "description": "If true, documents whose source files no longer exist on disk will be removed from the collection (chunks + embeddings deleted). Default: false (only add/update, never remove)."
      },
      "document_type": {
        "type": "string",
        "enum": ["contract", "report", "spreadsheet", "presentation", "correspondence", "memo", "proposal", "invoice", "policy", "other"],
        "description": "Default document type for newly ingested files."
      },
      "dry_run": {
        "type": "boolean",
        "default": false,
        "description": "If true, report what would change without actually ingesting or removing anything. Useful for previewing a sync."
      }
    },
    "required": ["folder_path"]
  }
}
```

---

### 2.9 `list_documents`

```json
{
  "name": "list_documents",
  "description": "List all documents in the active collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sort_by": {
        "type": "string",
        "enum": ["name", "date", "pages", "type"],
        "default": "date",
        "description": "Sort order"
      }
    }
  }
}
```

---

### 2.10 `get_document`

```json
{
  "name": "get_document",
  "description": "Get detailed information about a specific document including page count, extraction method, and chunk statistics.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.11 `delete_document`

```json
{
  "name": "delete_document",
  "description": "Remove a document and all its chunks, embeddings, and index entries from the active collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID to delete"
      },
      "confirm": {
        "type": "boolean",
        "default": false,
        "description": "Must be true to confirm deletion"
      }
    },
    "required": ["document_name", "confirm"]
  }
}
```

---

### 2.12 `search_documents`

```json
{
  "name": "search_documents",
  "description": "Search across all documents in the active collection using semantic and keyword search. Returns results with FULL provenance: source document filename, file path, page, paragraph, line numbers, character offsets, extraction method, timestamps. Every result is traceable to its exact source location.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query (e.g., 'Q3 revenue analysis', 'vendor payment terms', 'project timeline')"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 50,
        "description": "Number of results to return"
      },
      "document_filter": {
        "type": "string",
        "description": "Optional: restrict search to a specific document name or ID"
      }
    },
    "required": ["query"]
  }
}
```

**Success Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Search: \"payment terms\"\nCollection: Acme Corp Partnership | 5 documents, 1,051 chunks searched\nTime: 87ms | Tier: Pro (3-stage pipeline)\n\n--- Result 1 (score: 0.94) ---\nVendor_Agreement.docx, p. 12, para. 8, ll. 1-4\n\n\"Payment shall be made within thirty (30) days of receipt of invoice. Late payments shall accrue interest at a rate of 1.5% per month on the outstanding balance.\"\n\n--- Result 2 (score: 0.89) ---\nVendor_Agreement.docx, p. 13, para. 10, ll. 1-6\n\n\"In the event of early termination, the service provider shall be entitled to recover all outstanding fees, including accrued interest and reasonable costs of transition.\"\n\n--- Result 3 (score: 0.76) ---\nQ3_Report.xlsx, p. 8, para. 22, ll. 3-5\n\n\"Vendor payments exceeded budget by 12% in Q3, primarily due to accelerated delivery schedules under the revised service level agreement.\""
  }]
}
```

---

### 2.13 `find_entity`

```json
{
  "name": "find_entity",
  "description": "Find all mentions of an entity (person, organization, date, amount) across documents. Uses the entity index built during ingestion.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity": {
        "type": "string",
        "description": "Entity to find (e.g., 'John Smith', 'Acme Corp', '$1.2 million')"
      },
      "entity_type": {
        "type": "string",
        "enum": ["person", "organization", "date", "amount", "location", "concept", "any"],
        "default": "any",
        "description": "Type of entity to search for"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["entity"]
  }
}
```

---

### 2.14 `reindex_document`

```json
{
  "name": "reindex_document",
  "description": "Rebuild all embeddings, chunks, and search indexes for a single document. Deletes all existing chunks and embeddings for the document, re-extracts text from the original file, re-chunks into 2000-character segments, re-embeds with all active models, and rebuilds the BM25 index. Use this when: (1) a document's source file has been updated on disk, (2) you upgraded to Pro tier and want the document embedded with all 4 models, (3) embeddings seem stale or corrupt, (4) OCR results need refreshing. The original file path stored in provenance is used to re-read the source file.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID to reindex"
      },
      "force": {
        "type": "boolean",
        "default": false,
        "description": "If true, reindex even if the source file SHA256 has not changed. Default: only reindex if the file has changed."
      },
      "reparse": {
        "type": "boolean",
        "default": true,
        "description": "If true (default), re-extract text from the source file and re-chunk. If false, keep existing chunks but only rebuild embeddings and indexes (faster, useful after tier upgrade)."
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.15 `reindex_collection`

```json
{
  "name": "reindex_collection",
  "description": "Rebuild all embeddings, chunks, and search indexes for every document in the active collection. This is a full rebuild -- it deletes ALL existing chunks and embeddings, re-reads every source file, re-chunks, re-embeds with all active models, and rebuilds the entire BM25 index. Use this when: (1) upgrading from Free to Pro tier (re-embed everything with 4 models instead of 3), (2) after a CaseTrack update that changes chunking or embedding logic, (3) the collection index seems corrupted or stale, (4) you want a clean rebuild. WARNING: This can be slow for large collections (hundreds of documents).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "confirm": {
        "type": "boolean",
        "default": false,
        "description": "Must be true to confirm. This deletes and rebuilds ALL embeddings in the collection."
      },
      "reparse": {
        "type": "boolean",
        "default": true,
        "description": "If true (default), re-extract text from source files and re-chunk everything. If false, keep existing chunks but only rebuild embeddings and indexes (faster, useful after tier upgrade)."
      },
      "skip_unchanged": {
        "type": "boolean",
        "default": false,
        "description": "If true, skip documents whose source files have not changed (SHA256 match) and whose embeddings are complete for the current tier. Default: false (rebuild everything)."
      }
    },
    "required": ["confirm"]
  }
}
```

---

### 2.16 `get_index_status`

```json
{
  "name": "get_index_status",
  "description": "Show the embedding and index health status for all documents in the active collection. Reports which documents have complete embeddings for the current tier, which need reindexing (source file changed, missing embedder coverage, stale embeddings), and overall collection index health. Use this to diagnose issues or decide whether to run reindex_document or reindex_collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_filter": {
        "type": "string",
        "description": "Optional: check a specific document instead of all"
      }
    }
  }
}
```

---

### 2.17 `get_status`

```json
{
  "name": "get_status",
  "description": "Get CaseTrack server status including version, license tier, loaded models, and storage usage.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### 2.18 `get_chunk`

```json
{
  "name": "get_chunk",
  "description": "Get a specific chunk by ID with its full text, provenance (source file, page, paragraph, line, character offsets), and embedding status.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "chunk_id": {
        "type": "string",
        "description": "UUID of the chunk"
      }
    },
    "required": ["chunk_id"]
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Chunk abc-123 (2000 chars)\n\nText:\n\"Payment shall be made within thirty (30) days of receipt of invoice. Late payments shall accrue interest...\"\n\nProvenance:\n  Document:   Vendor_Agreement.docx\n  File Path:  /Users/sarah/Projects/Acme/Vendor_Agreement.docx\n  Page:       12\n  Paragraphs: 8-9\n  Lines:      1-14\n  Chars:      2401-4401 (within page)\n  Extraction: Native text\n  Chunk Index: 47 of 234\n\nEmbeddings: E1, E6, E12"
  }]
}
```

---

### 2.19 `get_document_chunks`

```json
{
  "name": "get_document_chunks",
  "description": "List all chunks in a document with their provenance. Shows where every piece of text came from: page, paragraph, line numbers, and character offsets. Use this to understand how a document was chunked and indexed.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      },
      "page_filter": {
        "type": "integer",
        "description": "Optional: only show chunks from this page number"
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.20 `get_source_context`

```json
{
  "name": "get_source_context",
  "description": "Get the surrounding context for a chunk -- the chunks immediately before and after it in the original document. Useful for understanding the full context around a search result.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "chunk_id": {
        "type": "string",
        "description": "UUID of the chunk to get context for"
      },
      "window": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "maximum": 5,
        "description": "Number of chunks before and after to include"
      }
    },
    "required": ["chunk_id"]
  }
}
```

---

### 2.21 `watch_folder`

```json
{
  "name": "watch_folder",
  "description": "Start watching a folder for file changes. When files are added, modified, or deleted in the watched folder (or any subfolder), CaseTrack automatically syncs the changes into the active collection -- new files are ingested, modified files are reindexed (old chunks/embeddings deleted, new ones created), and optionally deleted files are removed from the collection. Uses OS-level file notifications (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows) for instant detection. Also supports scheduled sync as a safety net (daily, hourly, or custom interval). Watch persists across server restarts.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Absolute path to the folder to watch. All subfolders are included."
      },
      "schedule": {
        "type": "string",
        "enum": ["on_change", "hourly", "daily", "every_6h", "every_12h", "manual"],
        "default": "on_change",
        "description": "When to sync: 'on_change' = real-time via OS file notifications (recommended), 'hourly'/'daily'/'every_6h'/'every_12h' = scheduled interval (runs in addition to on_change), 'manual' = only sync when you call sync_folder."
      },
      "auto_remove_deleted": {
        "type": "boolean",
        "default": false,
        "description": "If true, documents whose source files are deleted from disk will be automatically removed from the collection (chunks + embeddings deleted). Default: false (only add/update, never auto-remove)."
      },
      "file_extensions": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Optional filter: only watch files with these extensions (e.g., [\"pdf\", \"docx\", \"xlsx\"]). Default: all supported formats."
      }
    },
    "required": ["folder_path"]
  }
}
```

---

### 2.22 `unwatch_folder`

```json
{
  "name": "unwatch_folder",
  "description": "Stop watching a folder. Removes the watch but does NOT delete any documents already ingested from that folder. The collection data remains intact -- only the automatic sync is stopped.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Path to the folder to stop watching (or watch ID)"
      }
    },
    "required": ["folder_path"]
  }
}
```

---

### 2.23 `list_watches`

```json
{
  "name": "list_watches",
  "description": "List all active folder watches across all collections. Shows the watched folder, which collection it syncs to, the schedule, last sync time, and current status.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "collection_filter": {
        "type": "string",
        "description": "Optional: only show watches for a specific collection name or ID"
      }
    }
  }
}
```

---

### 2.24 `set_sync_schedule`

```json
{
  "name": "set_sync_schedule",
  "description": "Change the sync schedule for an existing folder watch. Controls how often CaseTrack checks for file changes and reindexes.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "folder_path": {
        "type": "string",
        "description": "Path to the watched folder (or watch ID)"
      },
      "schedule": {
        "type": "string",
        "enum": ["on_change", "hourly", "daily", "every_6h", "every_12h", "manual"],
        "description": "New schedule: 'on_change' = real-time OS notifications, 'hourly'/'daily' etc = interval-based, 'manual' = only when you call sync_folder"
      },
      "auto_remove_deleted": {
        "type": "boolean",
        "description": "Optionally update auto-remove behavior"
      }
    },
    "required": ["folder_path", "schedule"]
  }
}
```

---

## 2b. Context Graph Tool Specifications

The context graph tools give the AI structured navigation of the collection beyond flat search. They are built on the entity, reference, and document graph data extracted during ingestion (see PRD 04 Section 8).

### 2.25 `get_collection_summary`

```json
{
  "name": "get_collection_summary",
  "description": "Get a high-level briefing on the active collection. Returns: key stakeholders (people and organizations mentioned most), key dates and events, key topics, document breakdown by category, key references (most-referenced documents or external sources), most-mentioned entities, and collection statistics. This is the FIRST tool the AI should call when starting work on a collection -- it provides the structural overview needed to plan search strategy for 1000+ documents.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "COLLECTION SUMMARY: Acme Corp Partnership (Contract)\n\n  KEY STAKEHOLDERS:\n    Client:     Acme Corp (CEO: John Smith)\n    Vendor:     Summit Services LLC (CEO: Mary Jones)\n    Analysts:   Sarah Chen (Acme), Michael Brown (Summit)\n\n  KEY DATES:\n    2022-01-15  Contract signed (Vendor_Agreement.docx, p.1)\n    2023-06-01  Service level review (Q3_Report.xlsx, p.5)\n    2023-07-01  Renewal proposal submitted (Proposal.pdf, p.1)\n    2023-09-15  Budget approved (Budget.xlsx, p.1)\n    2024-01-10  Q1 deliverables deadline (Status_Report.docx, p.2)\n    2024-06-15  Partnership review date (Meeting_Notes.pdf, p.3)\n\n  KEY TOPICS:\n    1. Service level agreement compliance -- 23 documents, 187 chunks\n    2. Payment terms and schedules -- 18 documents, 145 chunks\n    3. Vendor performance metrics -- 8 documents, 42 chunks\n    4. Cost optimization -- 5 documents, 28 chunks\n\n  DOCUMENTS (47 total, 2,341 pages, 12,450 chunks):\n    Contracts:       5 docs (Vendor Agreement, Service Contract, Amendments...)\n    Reports:        20 docs (Q3 Report, Performance Reviews, Audits...)\n    Financials:     15 docs (Budgets, Invoices, Cost Analyses...)\n    Correspondence:  7 docs (Meeting Notes, Status Updates, Memos...)\n\n  KEY REFERENCES (most cited):\n    1. Master Service Agreement v2.1 -- 47 references across 15 documents\n    2. SLA Framework 2023 -- 23 references across 8 documents\n    3. Industry Benchmark Report -- 12 references across 6 documents\n\n  TOP ENTITIES:\n    Acme Corp -- 892 mentions in 45 documents\n    Summit Services LLC -- 756 mentions in 42 documents\n    John Smith -- 234 mentions in 28 documents\n    Service level agreement -- 187 mentions in 23 documents\n\n  EMBEDDINGS: 4/4 embedders (Pro tier), all 12,450 chunks fully embedded"
  }]
}
```

---

### 2.26 `get_collection_timeline`

```json
{
  "name": "get_collection_timeline",
  "description": "Get a chronological timeline of key dates and events extracted from documents in the active collection. Each event includes the date, description, and source document/chunk provenance. Use this to understand the narrative sequence of events.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "start_date": {
        "type": "string",
        "description": "Optional: filter events from this date (YYYY-MM-DD)"
      },
      "end_date": {
        "type": "string",
        "description": "Optional: filter events until this date (YYYY-MM-DD)"
      }
    }
  }
}
```

---

### 2.27 `get_collection_statistics`

```json
{
  "name": "get_collection_statistics",
  "description": "Get detailed statistics about the active collection: document counts by type, page/chunk totals, entity and reference counts, embedder coverage, storage usage. Useful for understanding collection scope and data quality.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

---

### 2.28 `list_entities`

```json
{
  "name": "list_entities",
  "description": "List all entities extracted from documents in the active collection, grouped by type. Shows name, type, mention count, and number of documents mentioning each entity. Entities include: persons, organizations, dates, monetary amounts, locations, and concepts.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_type": {
        "type": "string",
        "enum": ["person", "organization", "date", "amount", "location", "concept", "all"],
        "default": "all",
        "description": "Filter by entity type"
      },
      "sort_by": {
        "type": "string",
        "enum": ["mentions", "documents", "name"],
        "default": "mentions",
        "description": "Sort order"
      },
      "top_k": {
        "type": "integer",
        "default": 50,
        "maximum": 500,
        "description": "Maximum entities to return"
      }
    }
  }
}
```

---

### 2.29 `get_entity_mentions`

```json
{
  "name": "get_entity_mentions",
  "description": "Get all chunks that mention a specific entity, with context snippets showing how the entity is referenced. Uses the entity index built during ingestion. Supports fuzzy matching on entity name.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_name": {
        "type": "string",
        "description": "Name of the entity to find (e.g., 'John Smith', 'Acme Corp', 'payment terms')"
      },
      "entity_type": {
        "type": "string",
        "enum": ["person", "organization", "date", "amount", "location", "concept", "any"],
        "default": "any"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["entity_name"]
  }
}
```

**Response:**
```json
{
  "content": [{
    "type": "text",
    "text": "Mentions of \"John Smith\" (person) -- 234 total, showing top 20:\n\n  1. Proposal.pdf, p.2, para.3\n     \"John Smith, CEO of Acme Corp, presented the partnership proposal...\"\n\n  2. Meeting_Notes.pdf, p.15, para.8\n     \"Q: Mr. Smith, when did you first review the vendor performance report?\"\n     \"A: I received the summary from our VP on March 10, 2023...\"\n\n  3. Vendor_Agreement.docx, p.12, para.1 (signature block)\n     \"John Smith, Chief Executive Officer, Acme Corp\"\n\n  ... 17 more mentions"
  }]
}
```

---

### 2.30 `search_entity_relationships`

```json
{
  "name": "search_entity_relationships",
  "description": "Find chunks where two or more entities are mentioned together. Use this to trace relationships (who interacted with whom, what terms apply to which party). Pro tier only.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entities": {
        "type": "array",
        "items": { "type": "string" },
        "minItems": 2,
        "maxItems": 5,
        "description": "Entity names to find together (e.g., ['Acme Corp', 'Summit Services'])"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["entities"]
  }
}
```

---

### 2.31 `get_entity_graph`

```json
{
  "name": "get_entity_graph",
  "description": "Show entity relationships across documents in the active collection. Returns a graph of entities connected by co-occurrence in documents and chunks. Use this to understand how people, organizations, and concepts relate to each other across the collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_name": {
        "type": "string",
        "description": "Optional: center the graph on a specific entity. If omitted, returns the top entities by connectivity."
      },
      "depth": {
        "type": "integer",
        "default": 2,
        "minimum": 1,
        "maximum": 4,
        "description": "How many relationship hops to include from the center entity"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100,
        "description": "Maximum entities to include in the graph"
      }
    }
  }
}
```

---

### 2.32 `list_references`

```json
{
  "name": "list_references",
  "description": "List all referenced external sources (documents, standards, regulations, reports) cited in the active collection. Shows the reference, type, citation count, and number of citing documents. Use this to understand which external sources matter most in the collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "reference_type": {
        "type": "string",
        "enum": ["document", "standard", "regulation", "report", "all"],
        "default": "all"
      },
      "sort_by": {
        "type": "string",
        "enum": ["citations", "documents", "name"],
        "default": "citations"
      },
      "top_k": {
        "type": "integer",
        "default": 50,
        "maximum": 200
      }
    }
  }
}
```

---

### 2.33 `get_reference_citations`

```json
{
  "name": "get_reference_citations",
  "description": "Get all chunks that cite a specific reference. Shows the context of each citation. Use this to understand how a reference is used throughout the collection.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "reference": {
        "type": "string",
        "description": "The reference to look up (e.g., 'Master Service Agreement v2.1', 'ISO 27001')"
      },
      "top_k": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      }
    },
    "required": ["reference"]
  }
}
```

---

### 2.34 `get_document_structure`

```json
{
  "name": "get_document_structure",
  "description": "Get the structural outline of a document: headings, sections, numbered clauses, and their page/chunk locations. This gives the AI a table-of-contents view for navigation. Works best with structured documents (contracts, reports, policies).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.35 `browse_pages`

```json
{
  "name": "browse_pages",
  "description": "Get all chunks from a specific page range within a document. Use this to read through a section of a document sequentially. Returns chunks in order with full provenance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      },
      "start_page": {
        "type": "integer",
        "minimum": 1,
        "description": "First page to read"
      },
      "end_page": {
        "type": "integer",
        "minimum": 1,
        "description": "Last page to read"
      }
    },
    "required": ["document_name", "start_page", "end_page"]
  }
}
```

---

### 2.36 `find_related_documents`

```json
{
  "name": "find_related_documents",
  "description": "Find documents related to a given document. Relationships detected: shared entities, shared references, semantic similarity (E1 cosine), and version chains. Returns related documents ranked by relationship strength.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID to find relationships for"
      },
      "relationship_type": {
        "type": "string",
        "enum": ["all", "shared_entities", "shared_references", "semantic_similar", "version_chain"],
        "default": "all"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "maximum": 50
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.37 `get_related_documents`

```json
{
  "name": "get_related_documents",
  "description": "Given a document, find related docs via the knowledge graph. Uses shared entities, references, and semantic similarity to surface connections. This is a knowledge-graph-first approach compared to find_related_documents which also supports explicit relationship types.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_name": {
        "type": "string",
        "description": "Document name or ID"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "maximum": 50
      }
    },
    "required": ["document_name"]
  }
}
```

---

### 2.38 `list_documents_by_type`

```json
{
  "name": "list_documents_by_type",
  "description": "List all documents in the active collection filtered by document type (contract, report, spreadsheet, etc.). Includes page count, chunk count, and ingestion date.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_type": {
        "type": "string",
        "enum": ["contract", "report", "spreadsheet", "presentation", "correspondence", "memo", "proposal", "invoice", "policy", "other"],
        "description": "Type to filter by"
      }
    },
    "required": ["document_type"]
  }
}
```

---

### 2.39 `traverse_chunks`

```json
{
  "name": "traverse_chunks",
  "description": "Navigate forward or backward through chunks in a document from a starting point. Use this to read through a document sequentially from any position. Returns N chunks in document order with full provenance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "start_chunk_id": {
        "type": "string",
        "description": "UUID of the starting chunk"
      },
      "direction": {
        "type": "string",
        "enum": ["forward", "backward"],
        "default": "forward",
        "description": "Direction to traverse"
      },
      "count": {
        "type": "integer",
        "default": 5,
        "minimum": 1,
        "maximum": 20,
        "description": "Number of chunks to return"
      }
    },
    "required": ["start_chunk_id"]
  }
}
```

---

### 2.40 `search_similar_chunks`

```json
{
  "name": "search_similar_chunks",
  "description": "Find chunks across all documents that are semantically similar to a given chunk. Uses E1 cosine similarity. Use this to find related passages in other documents (e.g., 'find other places in the collection that discuss the same topic as this paragraph').",
  "inputSchema": {
    "type": "object",
    "properties": {
      "chunk_id": {
        "type": "string",
        "description": "UUID of the chunk to find similar content for"
      },
      "exclude_same_document": {
        "type": "boolean",
        "default": true,
        "description": "Exclude results from the same document (default: true, for cross-document discovery)"
      },
      "min_similarity": {
        "type": "number",
        "default": 0.6,
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Minimum cosine similarity threshold"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "maximum": 50
      }
    },
    "required": ["chunk_id"]
  }
}
```

---

### 2.41 `compare_documents`

```json
{
  "name": "compare_documents",
  "description": "Compare what two documents say about a specific topic. Searches both documents independently, then returns side-by-side results showing how each document addresses the topic. Pro tier only. Use this for: contract vs. proposal comparison, report A vs. report B, any 'what does X say vs. what does Y say' question.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "document_a": {
        "type": "string",
        "description": "First document name or ID"
      },
      "document_b": {
        "type": "string",
        "description": "Second document name or ID"
      },
      "topic": {
        "type": "string",
        "description": "Topic to compare (e.g., 'payment terms', 'delivery schedule', 'performance metrics')"
      },
      "top_k_per_document": {
        "type": "integer",
        "default": 5,
        "maximum": 20
      }
    },
    "required": ["document_a", "document_b", "topic"]
  }
}
```

---

### 2.42 `find_document_clusters`

```json
{
  "name": "find_document_clusters",
  "description": "Group all documents in the collection by theme or topic using semantic clustering. Returns clusters of related documents with a label describing what they share. Pro tier only. Use this to understand the structure of a large collection (100+ documents) at a glance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "strategy": {
        "type": "string",
        "enum": ["topical", "entity", "reference", "document_type"],
        "default": "topical",
        "description": "Clustering strategy: 'topical' = semantic similarity, 'entity' = shared people/orgs, 'reference' = shared external sources, 'document_type' = by type"
      },
      "max_clusters": {
        "type": "integer",
        "default": 10,
        "maximum": 20
      }
    }
  }
}
```

---

## 3. Background Watch System

The folder watch system runs as background tasks inside the MCP server process using the `notify` crate for cross-platform OS file notifications. Key data structures:

```rust
pub struct WatchManager {
    watches: Arc<RwLock<Vec<ActiveWatch>>>,
    fs_watcher: notify::RecommendedWatcher,
    event_tx: mpsc::Sender<FsEvent>,
}

struct ActiveWatch {
    config: FolderWatch,
    collection_handle: Arc<CollectionHandle>,
}

enum FsEventKind { Created, Modified, Deleted }
```

Behavior: On startup, `WatchManager::init` restores saved watches from `watches.json`, starts OS watchers, and spawns two background tasks -- an event processor (with 2-second debounce) and a scheduled sync runner (checks every 60 seconds). Events are batched: Created triggers ingest, Modified triggers reindex, Deleted triggers removal (if `auto_remove_deleted` is enabled).

For full implementation details (server initialization, tool registration, error handling), see [PRD 10: Technical Build Guide](PRD_10_TECHNICAL_BUILD.md).

---

## 4. Active Collection State

The server maintains an "active collection" that all document and search operations target. The server starts with no active collection; `create_collection` automatically switches to the new collection, and `switch_collection` explicitly changes it. Tools requiring a collection return a `NoCollectionActive` error if none is set. The active collection persists for the MCP session duration but not across sessions.

---

*CaseTrack PRD v4.0.0 -- Document 9 of 10*
