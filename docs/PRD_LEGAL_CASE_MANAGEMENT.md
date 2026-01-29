# Product Requirements Document: CaseTrack

## One-Click Legal Document Analysis for Claude Code & Claude Desktop

**Version**: 3.1.0
**Date**: 2026-01-28
**Status**: Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Vision](#2-product-vision)
3. [Target User & Hardware](#3-target-user--hardware)
4. [Distribution & Installation](#4-distribution--installation)
5. [Local Storage Architecture](#5-local-storage-architecture)
6. [7-Embedder Legal Stack](#6-7-embedder-legal-stack)
7. [Document Ingestion](#7-document-ingestion)
8. [Case Management](#8-case-management)
9. [Provenance System](#9-provenance-system)
10. [Search & Retrieval](#10-search--retrieval)
11. [MCP Tools](#11-mcp-tools)
12. [Monetization](#12-monetization)
13. [Technical Implementation](#13-technical-implementation)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Success Metrics](#15-success-metrics)

---

## 1. Executive Summary

### 1.1 What is CaseTrack?

CaseTrack is a **one-click installable MCP server** that plugs into **Claude Code** and **Claude Desktop**, giving Claude the ability to ingest, search, and analyze legal documents. Everything runs on the user's machine—**all embeddings, vectors, and databases are stored locally** on the user's device with zero cloud dependencies.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CASETRACK                                       │
│                                                                             │
│        "Install once. Everything runs on YOUR machine."                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHAT IT DOES                                                              │
│   ════════════                                                              │
│   • Ingests PDFs, DOCX, scanned images                                     │
│   • Embeds documents with 7 specialized legal embedders                    │
│   • Stores all vectors/embeddings in LOCAL database on YOUR device         │
│   • Provides semantic search with full source citations                    │
│                                                                             │
│   WHAT YOU GET                                                              │
│   ════════════                                                              │
│   • MCP server that plugs into Claude Code or Claude Desktop               │
│   • All 7 embedding models downloaded and ready to use                     │
│   • RocksDB database installed on your machine                             │
│   • Your data NEVER leaves your computer                                   │
│                                                                             │
│   RUNS ON                        REQUIRES                                   │
│   ────────                       ────────                                   │
│   • macOS (Intel + Apple Silicon)  • 8GB RAM minimum                       │
│   • Windows 10/11                  • 5GB storage                           │
│   • Linux (Ubuntu 20.04+)          • NO GPU needed                         │
│                                                                             │
│   STORAGE (All on YOUR device)                                              │
│   ════════════════════════════                                              │
│   • Embedding models: ~400MB                                               │
│   • Vector database: RocksDB (scales with documents)                       │
│   • Embeddings: ~50KB per document page                                    │
│   • Everything stored in ~/Documents/CaseTrack/                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Problem

Legal professionals waste hours searching through case documents:

- **Keyword search fails**: "breach of duty" won't find "violation of fiduciary obligation"
- **No AI integration**: Can't ask questions about documents in natural language
- **No provenance**: When you find something, you can't cite the exact source
- **Complex tools**: Existing legal tech requires IT departments and training

### 1.3 The Solution

CaseTrack solves this with:

1. **One-click install**: Single command or MCPB file—embedders and database included
2. **100% local storage**: All embeddings and vectors stored on YOUR device in RocksDB
3. **7 specialized embedders**: Semantic search that understands legal language
4. **Full provenance**: Every answer cites document, page, paragraph, line
5. **Claude Code + Desktop integration**: Works with both Claude Code CLI and Claude Desktop app
6. **Runs anywhere**: Works on an 8GB MacBook Air, no GPU needed

### 1.4 Key Metrics

| Metric | Target |
|--------|--------|
| Install time | < 2 minutes |
| First search after install | < 5 minutes |
| Search latency | < 200ms on any laptop |
| PDF ingestion | < 1 second per page |
| RAM usage | < 2GB peak |
| Model download | ~400MB one-time |

---

## 2. Product Vision

### 2.1 Vision Statement

> **Any attorney can ask Claude questions about their case documents and get accurate, cited answers—without IT support, cloud accounts, or technical knowledge.**

### 2.2 Design Principles

```
DESIGN PRINCIPLES
═════════════════════════════════════════════════════════════════════════════

1. ZERO CONFIGURATION
   User downloads file → double-clicks → starts using
   No terminal, no config files, no environment variables

2. RUNS ON ANYTHING
   8GB RAM laptop from 2020 should work fine
   No GPU required, ever
   Intel, AMD, Apple Silicon all supported

3. PRIVACY FIRST
   Documents never leave the device
   No telemetry, no analytics, no cloud
   Attorney-client privilege preserved

4. INSTANT VALUE
   First useful search within 5 minutes of download
   No training required
   Works like asking a research assistant

5. PROVENANCE ALWAYS
   Every answer includes exact source citation
   Document name, page, paragraph, line number
   One click to view original context

6. GRACEFUL DEGRADATION
   Low RAM? Use fewer models
   Slow CPU? Longer ingestion, same quality
   Free tier? Fewer features, still useful
```

### 2.3 What CaseTrack is NOT

- **Not a document management system**: Use Dropbox/OneDrive for storage
- **Not a practice management tool**: No billing, calendaring, or client management
- **Not e-discovery software**: Not built for litigation holds or productions
- **Not a cloud service**: Everything runs locally, we never see your data

---

## 3. Target User & Hardware

### 3.1 Primary Users

```
PRIMARY: SOLO PRACTITIONERS & SMALL FIRMS
═════════════════════════════════════════════════════════════════════════════

Profile:
• 1-10 attorney firm
• No dedicated IT staff
• Uses consumer hardware (MacBook, Windows laptop)
• Handles 5-50 active matters
• Documents stored in folders on local drive or cloud sync

Pain Points:
• Can't find documents they know exist
• Spend hours re-reading to find specific facts
• No budget for enterprise legal tech ($500+/seat/month)
• Frustrated by keyword search limitations

Why CaseTrack:
• Works on their existing laptop
• No IT support needed
• Affordable ($29/month or free tier)
• Immediate productivity boost
```

### 3.2 Minimum Hardware Requirements

```
MINIMUM REQUIREMENTS (Must Run)
═════════════════════════════════════════════════════════════════════════════

CPU:     Any 64-bit processor (2018 or newer recommended)
         • Intel Core i3 or better
         • AMD Ryzen 3 or better
         • Apple M1 or better

RAM:     8GB minimum
         • 16GB recommended for large cases

Storage: 5GB available
         • 400MB for models
         • 4.6GB for case data (scales with usage)

OS:      • macOS 11 (Big Sur) or later
         • Windows 10 (64-bit) or later
         • Ubuntu 20.04 or later

GPU:     NOT REQUIRED
         • Optional: Metal (macOS), CUDA (NVIDIA), DirectML (Windows)
         • GPU provides ~2x speedup if available
```

### 3.3 Performance by Hardware Tier

| Hardware | Ingest 50-page PDF | Search Latency | Concurrent Models |
|----------|-------------------|----------------|-------------------|
| **Entry** (M1 Air 8GB) | 45 seconds | 150ms | 3 |
| **Mid** (M2 Pro 16GB) | 25 seconds | 80ms | 5 |
| **High** (i7 32GB) | 20 seconds | 60ms | 7 |
| **With GPU** (RTX 3060) | 10 seconds | 30ms | 7 |

---

## 4. Distribution & Installation

### 4.1 Distribution Strategy

```
DISTRIBUTION CHANNELS
═════════════════════════════════════════════════════════════════════════════

FOR CLAUDE CODE (Recommended)
─────────────────────────────────
# macOS/Linux - One command:
curl -fsSL https://casetrack.legal/install.sh | sh

# Windows - PowerShell:
irm https://casetrack.legal/install.ps1 | iex

# Or via cargo:
cargo binstall casetrack   # Pre-compiled (fast)
cargo install casetrack    # From source

# Then add to Claude Code settings (~/.claude/settings.json):
{
  "mcpServers": {
    "casetrack": {
      "command": "casetrack",
      "args": ["--data-dir", "~/Documents/CaseTrack"]
    }
  }
}

FOR CLAUDE DESKTOP
─────────────────────────────────
• Download casetrack.mcpb from website
• Double-click or drag to Claude Desktop
• Click "Install" in dialog
• Done (same binary, just packaged for GUI install)

PACKAGE MANAGERS
─────────────────────────────────
macOS:     brew install casetrack
Windows:   winget install CaseTrack
Linux:     cargo binstall casetrack
```

### 4.2 MCPB Bundle Structure

The `.mcpb` file is a ZIP archive containing everything needed:

```
casetrack.mcpb (ZIP archive, ~50MB)
├── manifest.json           # MCP configuration
├── icon.png               # Extension icon (256x256)
├── server/
│   ├── casetrack-darwin-x64      # macOS Intel
│   ├── casetrack-darwin-arm64    # macOS Apple Silicon
│   ├── casetrack-win32-x64.exe   # Windows
│   └── casetrack-linux-x64       # Linux
└── resources/
    ├── tokenizer.json     # Shared tokenizer (~5MB)
    └── legal-vocab.txt    # Legal term expansions (~2MB)
```

### 4.3 Manifest Specification

```json
{
  "manifest_version": "1.0",
  "name": "casetrack",
  "version": "1.0.0",
  "display_name": "CaseTrack Legal Document Analysis",
  "description": "Ingest PDFs, Word docs, and scans. Search with AI. Every answer cites the source.",

  "author": {
    "name": "CaseTrack",
    "url": "https://casetrack.legal"
  },

  "server": {
    "type": "binary",
    "entry_point": "server/casetrack"
  },

  "compatibility": {
    "platforms": ["darwin", "win32", "linux"]
  },

  "user_config": [
    {
      "id": "data_dir",
      "name": "Data Location",
      "description": "Where to store cases and models",
      "type": "directory",
      "default": "${DOCUMENTS}/CaseTrack",
      "required": true
    },
    {
      "id": "license_key",
      "name": "License Key (Optional)",
      "description": "Leave blank for free tier. Purchase at casetrack.legal",
      "type": "string",
      "sensitive": true,
      "required": false
    }
  ],

  "mcp_config": {
    "command": "server/casetrack",
    "args": ["--data-dir", "${user_config.data_dir}"],
    "env": {
      "CASETRACK_LICENSE": "${user_config.license_key}",
      "CASETRACK_HOME": "${user_config.data_dir}"
    }
  },

  "platform_overrides": {
    "darwin-arm64": {
      "mcp_config": { "command": "server/casetrack-darwin-arm64" }
    },
    "darwin-x64": {
      "mcp_config": { "command": "server/casetrack-darwin-x64" }
    },
    "win32": {
      "mcp_config": { "command": "server/casetrack-win32-x64.exe" }
    }
  },

  "permissions": {
    "filesystem": {
      "read": ["${user_config.data_dir}"],
      "write": ["${user_config.data_dir}"]
    },
    "network": {
      "domains": ["huggingface.co"],
      "reason": "Download embedding models on first use"
    }
  },

  "icons": { "256": "icon.png" },
  "tools": { "_generated": true }
}
```

### 4.4 Installation Flow

```
INSTALLATION FLOW
═════════════════════════════════════════════════════════════════════════════

Step 1: Download (User Action)
────────────────────────────────
User visits casetrack.legal
Clicks "Download for Claude Desktop"
Browser downloads casetrack.mcpb (~50MB)

Step 2: Install (User Action)
────────────────────────────────
User double-clicks casetrack.mcpb
  OR drags to Claude Desktop window
  OR Settings → Extensions → Install from file

Step 3: Configure (Dialog)
────────────────────────────────
┌─────────────────────────────────────────────────────────┐
│ Install CaseTrack?                                      │
│                                                         │
│ CaseTrack lets you search legal documents with AI.      │
│ All processing happens on your computer.                │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Data Location                                       │ │
│ │ [~/Documents/CaseTrack                        ] [📁]│ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ License Key (optional - leave blank for free tier) │ │
│ │ [                                             ] [🔒]│ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ This extension will:                                    │
│ ✓ Read and write files in your Data Location           │
│ ✓ Download AI models from huggingface.co (~400MB)      │
│ ✗ NOT send your documents anywhere                     │
│                                                         │
│                          [Cancel]  [Install Extension]  │
└─────────────────────────────────────────────────────────┘

Step 4: First Run (Automatic)
────────────────────────────────
Claude Desktop starts CaseTrack server
Server detects missing models
Downloads models in background (~400MB, 2-5 min)
Shows progress notification in Claude Desktop

Step 5: Ready (Automatic)
────────────────────────────────
CaseTrack icon appears in Extensions panel
User can now use CaseTrack tools in conversation
```

### 4.5 Model Download Strategy

Models are NOT bundled (would make .mcpb too large). Instead, downloaded on first use:

```rust
/// Model download configuration
pub struct ModelConfig {
    /// Models to download (in priority order)
    pub models: Vec<ModelSpec>,

    /// Total expected download size
    pub total_size_mb: u32,  // ~400MB

    /// Cache directory
    pub cache_dir: PathBuf,
}

pub struct ModelSpec {
    pub id: &'static str,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub size_mb: u32,
    pub required: bool,  // false = only download for Pro tier
}

/// Models for the 7-embedder stack
pub const MODELS: &[ModelSpec] = &[
    // E1-LEGAL: Core semantic (REQUIRED)
    ModelSpec {
        id: "e1-legal",
        repo: "BAAI/bge-small-en-v1.5",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 65,
        required: true,
    },
    // E6-LEGAL: Keywords (REQUIRED)
    ModelSpec {
        id: "e6-legal",
        repo: "naver/splade-cocondenser-selfdistil",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 55,
        required: true,
    },
    // E7: Structured text (REQUIRED)
    ModelSpec {
        id: "e7",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 45,
        required: true,
    },
    // E8-LEGAL: Citations (PRO)
    ModelSpec {
        id: "e8-legal",
        repo: "casetrack/citation-minilm-v1",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 35,
        required: false,
    },
    // E11-LEGAL: Entities (PRO)
    ModelSpec {
        id: "e11-legal",
        repo: "nlpaueb/legal-bert-small-uncased",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 60,
        required: false,
    },
    // E12: ColBERT rerank (PRO)
    ModelSpec {
        id: "e12",
        repo: "colbert-ir/colbertv2.0-msmarco-passage",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 110,
        required: false,
    },
];

// E13 (BM25) requires no model download - pure algorithm
```

---

## 5. Local Storage Architecture

### 5.1 Core Principle: Everything on YOUR Machine

**CaseTrack stores ALL data locally on the user's device:**

- **Embedding models**: Downloaded once, stored in `~/Documents/CaseTrack/models/`
- **Vector embeddings**: Stored in RocksDB on your device
- **Document chunks**: Stored in RocksDB on your device
- **Case databases**: Each case is an isolated RocksDB instance
- **Original documents**: Optionally copied to your CaseTrack folder

**Nothing is sent to any cloud service. Ever.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOCAL STORAGE ARCHITECTURE                                │
│                   (All on YOUR Computer)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ~/Documents/CaseTrack/                      ← All CaseTrack data lives here│
│  │                                                                          │
│  ├── models/                                 ← Embedding models (~400MB)    │
│  │   ├── bge-small-en-v1.5/                   • Downloaded on first use    │
│  │   │   ├── model.onnx                       • Cached permanently         │
│  │   │   └── tokenizer.json                   • No re-download needed      │
│  │   ├── splade-distil/                                                    │
│  │   ├── minilm-l6/                                                        │
│  │   └── colbert-small/                                                    │
│  │                                                                          │
│  ├── registry.db/                            ← Case index (RocksDB)        │
│  │   └── [list of all your cases]                                          │
│  │                                                                          │
│  └── cases/                                  ← Per-case databases          │
│      │                                                                      │
│      ├── {case-uuid-1}/                      ← Case "Smith v. Jones"       │
│      │   ├── db/                               (Isolated RocksDB)          │
│      │   │   ├── documents      # Document metadata                        │
│      │   │   ├── chunks         # Text chunks                              │
│      │   │   ├── embeddings     # Vector embeddings (arrays of floats)    │
│      │   │   │   ├── e1_legal   # 384D vectors                            │
│      │   │   │   ├── e6_legal   # Sparse vectors                          │
│      │   │   │   ├── e7         # 384D vectors                            │
│      │   │   │   └── ...        # All 7 embedders                         │
│      │   │   ├── provenance     # Source location tracking                 │
│      │   │   └── bm25_index     # Inverted index for keyword search       │
│      │   └── originals/          # Original files (optional)               │
│      │       ├── Complaint.pdf                                             │
│      │       └── Contract.docx                                             │
│      │                                                                      │
│      ├── {case-uuid-2}/                      ← Case "Doe v. Corp"          │
│      │   └── ...                               (Completely isolated)       │
│      │                                                                      │
│      └── {case-uuid-N}/                      ← More cases...               │
│                                                                             │
│  STORAGE REQUIREMENTS:                                                      │
│  ══════════════════════                                                     │
│  • Models (one-time):     ~400MB                                           │
│  • Per document page:     ~50KB (embeddings + metadata)                    │
│  • 100-page case:         ~5MB                                             │
│  • 1000-page case:        ~50MB                                            │
│  • Registry overhead:     ~1MB                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CASETRACK SYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              CLAUDE CODE  or  CLAUDE DESKTOP                         │   │
│  │                                                                      │   │
│  │   User: "What does the contract say about termination?"             │   │
│  │                          │                                           │   │
│  │                          ▼ MCP Protocol (stdio)                      │   │
│  └──────────────────────────┼───────────────────────────────────────────┘   │
│                             │                                               │
│  ┌──────────────────────────┼───────────────────────────────────────────┐   │
│  │                    CASETRACK MCP SERVER                              │   │
│  │                    (Rust Binary on YOUR machine)                     │   │
│  │                          │                                           │   │
│  │   ┌──────────────────────▼──────────────────────────────────────┐   │   │
│  │   │                   MCP TOOL ROUTER                            │   │   │
│  │   │  search_case │ ingest_pdf │ create_case │ list_documents    │   │   │
│  │   └──────────────────────┬───────────────────────────────────────┘   │   │
│  │                          │                                           │   │
│  │   ┌──────────────────────▼──────────────────────────────────────┐   │   │
│  │   │            PROCESSING (All local, no network)                │   │   │
│  │   │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │   │   │
│  │   │  │  Document  │ │  Chunking  │ │  Embedding │              │   │   │
│  │   │  │  Parser    │ │  Engine    │ │  Engine    │              │   │   │
│  │   │  │ PDF/DOCX/  │ │ 500-token  │ │ 7 ONNX     │              │   │   │
│  │   │  │ OCR        │ │ chunks     │ │ models     │              │   │   │
│  │   │  └────────────┘ └────────────┘ └────────────┘              │   │   │
│  │   └──────────────────────┬───────────────────────────────────────┘   │   │
│  │                          │                                           │   │
│  │   ┌──────────────────────▼──────────────────────────────────────┐   │   │
│  │   │         LOCAL STORAGE (RocksDB on YOUR device)               │   │   │
│  │   │                                                              │   │   │
│  │   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │   │   │
│  │   │  │  Case A     │ │  Case B     │ │  Case C     │           │   │   │
│  │   │  │  (RocksDB)  │ │  (RocksDB)  │ │  (RocksDB)  │           │   │   │
│  │   │  │             │ │             │ │             │           │   │   │
│  │   │  │ documents   │ │ documents   │ │ documents   │           │   │   │
│  │   │  │ chunks      │ │ chunks      │ │ chunks      │           │   │   │
│  │   │  │ EMBEDDINGS  │ │ EMBEDDINGS  │ │ EMBEDDINGS  │           │   │   │
│  │   │  │ (vectors)   │ │ (vectors)   │ │ (vectors)   │           │   │   │
│  │   │  │ provenance  │ │ provenance  │ │ provenance  │           │   │   │
│  │   │  └─────────────┘ └─────────────┘ └─────────────┘           │   │   │
│  │   │                                                              │   │   │
│  │   │  ALL STORED IN: ~/Documents/CaseTrack/cases/                │   │   │
│  │   └──────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Technology Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Binary** | Rust | Single executable, no runtime dependencies |
| **MCP Transport** | stdio | Works with Claude Code and Claude Desktop |
| **MCP SDK** | rmcp | Official Rust MCP SDK |
| **Vector Storage** | RocksDB | Fast embedded database, stores on user's disk |
| **Embedding Inference** | ONNX Runtime | CPU-optimized, cross-platform |
| **PDF Parsing** | pdf-extract + lopdf | Pure Rust, no external deps |
| **DOCX Parsing** | docx-rs | Pure Rust |
| **OCR** | Tesseract (bundled) | Best open-source OCR |

### 5.4 What's Stored Where

| Data Type | Storage Location | Format |
|-----------|------------------|--------|
| **Case metadata** | `registry.db` | RocksDB |
| **Document text chunks** | `cases/{id}/db/chunks` | RocksDB (bincode) |
| **E1 embeddings (384D)** | `cases/{id}/db/embeddings/e1` | RocksDB (f32 arrays) |
| **E6 sparse vectors** | `cases/{id}/db/embeddings/e6` | RocksDB (sparse format) |
| **E7-E12 embeddings** | `cases/{id}/db/embeddings/e*` | RocksDB (f32 arrays) |
| **BM25 inverted index** | `cases/{id}/db/bm25_index` | RocksDB |
| **Provenance records** | `cases/{id}/db/provenance` | RocksDB (bincode) |
| **Original documents** | `cases/{id}/originals/` | Original files |
| **ONNX models** | `models/` | ONNX format |

### 5.5 Data Flow

```
DOCUMENT INGESTION FLOW
═════════════════════════════════════════════════════════════════════════════

User: "Ingest ~/Downloads/Complaint.pdf"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. PARSE                                                                 │
│    • Detect file type (PDF/DOCX/image)                                  │
│    • Extract text with position metadata                                │
│    • For scans: run OCR with Tesseract                                  │
│    Output: Page[] with Paragraph[] with Line[]                          │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. CHUNK                                                                 │
│    • Split into ~500 token chunks                                       │
│    • Respect paragraph boundaries                                       │
│    • Attach provenance (doc, page, para, line, char offset)            │
│    Output: Chunk[] with Provenance                                      │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. EMBED                                                                 │
│    • Run each chunk through active embedders (3-7 depending on tier)   │
│    • Batch for efficiency (32 chunks at a time)                        │
│    Output: Chunk[] with Embeddings[7]                                   │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. INDEX                                                                 │
│    • Store chunks + embeddings in RocksDB                               │
│    • Build BM25 inverted index for E13                                  │
│    • Update document metadata                                           │
│    Output: Searchable case database                                     │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
Response: "Ingested Complaint.pdf: 45 pages, 234 chunks"
```

---

## 6. 7-Embedder Legal Stack

### 6.1 Design Philosophy

The embedder stack is designed for **consumer hardware**:

- **7 embedders** (not 13-15): Reduced from research system for practical use
- **384D max**: Smaller dimensions = less RAM, faster search
- **ONNX format**: CPU-optimized, cross-platform
- **Quantized (INT8)**: 50% smaller, nearly same quality
- **No LLM inference**: Removed causal/reasoning embedders that need GPUs

### 6.2 Embedder Specifications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      7-EMBEDDER LEGAL STACK                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  E1-LEGAL: Semantic Similarity (PRIMARY)                                    │
│  ════════════════════════════════════════════════════════════════════════   │
│  Model:      bge-small-en-v1.5 (BAAI)                                      │
│  Dimension:  384                                                            │
│  Size:       65MB (INT8 ONNX)                                              │
│  Speed:      50ms/chunk (M1), 100ms/chunk (Intel i5)                       │
│  Purpose:    Core semantic search - "breach of duty" finds                 │
│              "violation of fiduciary obligation"                           │
│  Tier:       FREE                                                           │
│                                                                             │
│  E6-LEGAL: Keyword Expansion                                                │
│  ════════════════════════════════════════════════════════════════════════   │
│  Model:      SPLADE-cocondenser-selfdistil (Naver)                         │
│  Dimension:  Sparse (30K vocabulary)                                       │
│  Size:       55MB (INT8 ONNX)                                              │
│  Speed:      30ms/chunk                                                    │
│  Purpose:    Exact term matching + expansion                               │
│              "Daubert" → also matches "expert testimony", "Rule 702"       │
│  Tier:       FREE                                                           │
│                                                                             │
│  E7: Structured Text                                                        │
│  ════════════════════════════════════════════════════════════════════════   │
│  Model:      all-MiniLM-L6-v2 (Sentence Transformers)                      │
│  Dimension:  384                                                            │
│  Size:       45MB (INT8 ONNX)                                              │
│  Speed:      40ms/chunk                                                    │
│  Purpose:    Contracts, statutes, numbered clauses                         │
│              Understands "Section 4.2(a)" structure                        │
│  Tier:       FREE                                                           │
│                                                                             │
│  E8-LEGAL: Citation Relationships                                           │
│  ════════════════════════════════════════════════════════════════════════   │
│  Model:      MiniLM-L3-v2 fine-tuned on citation pairs                     │
│  Dimension:  256 (asymmetric: citing/cited)                                │
│  Size:       35MB (INT8 ONNX)                                              │
│  Speed:      25ms/chunk                                                    │
│  Purpose:    Find documents citing same authorities                        │
│              "Cases citing Miranda v. Arizona"                              │
│  Tier:       PRO                                                            │
│                                                                             │
│  E11-LEGAL: Legal Entities                                                  │
│  ════════════════════════════════════════════════════════════════════════   │
│  Model:      legal-bert-small-uncased (nlpaueb)                            │
│  Dimension:  384                                                            │
│  Size:       60MB (INT8 ONNX)                                              │
│  Speed:      45ms/chunk                                                    │
│  Purpose:    Find by party, court, statute, doctrine                       │
│              "Documents mentioning Judge Smith"                             │
│  Tier:       PRO                                                            │
│                                                                             │
│  E12: Precision Reranking                                                   │
│  ════════════════════════════════════════════════════════════════════════   │
│  Model:      ColBERT-v2-small                                              │
│  Dimension:  64 per token                                                  │
│  Size:       110MB (INT8 ONNX)                                             │
│  Speed:      100ms for top 50 candidates                                   │
│  Purpose:    Final reranking for exact phrase matches                      │
│              Ensures "breach of fiduciary duty" ranks higher than          │
│              "fiduciary duty was not breached"                             │
│  Tier:       PRO                                                            │
│                                                                             │
│  E13: Fast Recall (BM25)                                                    │
│  ════════════════════════════════════════════════════════════════════════   │
│  Model:      None (algorithmic - BM25/TF-IDF)                              │
│  Dimension:  N/A (inverted index)                                          │
│  Size:       ~2MB index per 1000 documents                                 │
│  Speed:      <5ms for any query                                            │
│  Purpose:    Fast initial candidate retrieval                              │
│              Stage 1 recall before neural ranking                          │
│  Tier:       FREE                                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TOTAL FOOTPRINT                                                            │
│  ════════════════════════════════════════════════════════════════════════   │
│  Free tier models:   ~165MB (E1 + E6 + E7 + E13)                          │
│  Pro tier models:    ~370MB (all 7)                                        │
│  RAM at runtime:     ~1.2GB (free), ~1.8GB (pro)                          │
│  Per-chunk latency:  ~120ms (free), ~200ms (pro)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 What Was Removed (vs. Research System)

| Removed | Original Purpose | Why Removed | Alternative |
|---------|------------------|-------------|-------------|
| E2-E4 Temporal | Document recency/sequence | Complex, minimal value for legal | Date metadata filter |
| E5 Causal | "X caused Y" reasoning | Requires LLM-scale compute | Keyword patterns |
| E9 HDC | Noise robustness | Experimental, adds RAM | BM25 fallback |
| E10 Intent | Same-goal matching | 400MB+ model, GPU preferred | E1 covers this |
| E14 SAILER | Legal doc structure | Research model, no ONNX | E7 handles structure |
| E15 Citation Net | Citation patterns | Requires graph training | E8 simpler version |

### 6.4 Embedder Implementation

```rust
use ort::{Session, Environment, GraphOptimizationLevel};

/// Consumer-optimized embedding engine
pub struct EmbeddingEngine {
    env: Arc<Environment>,
    models: HashMap<EmbedderId, Option<Session>>,
    tier: LicenseTier,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbedderId {
    E1Legal,    // Semantic (FREE)
    E6Legal,    // Keywords (FREE)
    E7,         // Structured (FREE)
    E8Legal,    // Citations (PRO)
    E11Legal,   // Entities (PRO)
    E12,        // ColBERT (PRO)
    // E13 is BM25, not a neural model
}

impl EmbeddingEngine {
    pub fn new(model_dir: &Path, tier: LicenseTier) -> Result<Self> {
        let env = Environment::builder()
            .with_name("casetrack")
            .with_execution_providers([
                // Use CoreML on macOS for ~2x speedup
                #[cfg(target_os = "macos")]
                ExecutionProvider::CoreML(Default::default()),
                // CPU fallback (always works)
                ExecutionProvider::CPU(CPUExecutionProvider::default()),
            ])
            .build()?;

        let mut models = HashMap::new();

        // Load models based on tier
        for id in Self::models_for_tier(tier) {
            let path = model_dir.join(id.model_filename());
            if path.exists() {
                let session = Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(2)?  // Limit threads for RAM
                    .with_model_from_file(&path)?;
                models.insert(id, Some(session));
            } else {
                models.insert(id, None);  // Will download on first use
            }
        }

        Ok(Self { env: Arc::new(env), models, tier })
    }

    fn models_for_tier(tier: LicenseTier) -> Vec<EmbedderId> {
        match tier {
            LicenseTier::Free => vec![
                EmbedderId::E1Legal,
                EmbedderId::E6Legal,
                EmbedderId::E7,
            ],
            LicenseTier::Pro | LicenseTier::Firm | LicenseTier::Enterprise => vec![
                EmbedderId::E1Legal,
                EmbedderId::E6Legal,
                EmbedderId::E7,
                EmbedderId::E8Legal,
                EmbedderId::E11Legal,
                EmbedderId::E12,
            ],
        }
    }

    /// Embed a chunk with all active models
    pub fn embed(&self, text: &str) -> Result<ChunkEmbeddings> {
        let mut embeddings = ChunkEmbeddings::new();

        for (id, session) in &self.models {
            if let Some(session) = session {
                let embedding = self.run_inference(session, text)?;
                embeddings.set(*id, embedding);
            }
        }

        Ok(embeddings)
    }

    fn run_inference(&self, session: &Session, text: &str) -> Result<Vec<f32>> {
        // Tokenize
        let tokens = self.tokenize(text)?;

        // Run model
        let outputs = session.run(inputs![
            "input_ids" => tokens.input_ids,
            "attention_mask" => tokens.attention_mask,
        ]?)?;

        // Mean pooling
        let embeddings = outputs["last_hidden_state"].extract_tensor::<f32>()?;
        Ok(mean_pool(&embeddings, &tokens.attention_mask))
    }
}

/// Embeddings for a single chunk
pub struct ChunkEmbeddings {
    pub e1_legal: Option<Vec<f32>>,   // 384D
    pub e6_legal: Option<SparseVec>,  // Sparse
    pub e7: Option<Vec<f32>>,         // 384D
    pub e8_legal: Option<Vec<f32>>,   // 256D
    pub e11_legal: Option<Vec<f32>>,  // 384D
    pub e12: Option<TokenEmbeddings>, // 64D per token
}
```

---

## 7. Document Ingestion

### 7.1 Supported Formats

| Format | Method | Quality | Notes |
|--------|--------|---------|-------|
| PDF (native text) | pdf-extract | Excellent | Direct text extraction |
| PDF (scanned) | Tesseract OCR | Good (>95%) | Requires image preprocessing |
| DOCX | docx-rs | Excellent | Preserves structure |
| DOC (legacy) | Convert via docx | Good | Requires conversion |
| Images (JPG/PNG/TIFF) | Tesseract OCR | Good | Single page per image |
| TXT/RTF | Direct read | Excellent | Plain text |

### 7.2 PDF Processing

```rust
pub struct PdfProcessor {
    ocr_enabled: bool,
    ocr_language: String,
}

impl PdfProcessor {
    pub fn process(&self, path: &Path) -> Result<ProcessedDocument> {
        let doc = lopdf::Document::load(path)?;
        let mut pages = Vec::new();

        for page_num in 1..=doc.get_pages().len() {
            // Try native text extraction first
            let text = pdf_extract::extract_text_from_page(&doc, page_num)?;

            if text.trim().is_empty() && self.ocr_enabled {
                // Scanned page - use OCR
                let image = self.render_page_to_image(&doc, page_num)?;
                let ocr_result = self.run_ocr(&image)?;
                pages.push(Page {
                    number: page_num as u32,
                    content: ocr_result.text,
                    extraction_method: ExtractionMethod::Ocr,
                    ocr_confidence: Some(ocr_result.confidence),
                });
            } else {
                pages.push(Page {
                    number: page_num as u32,
                    content: text,
                    extraction_method: ExtractionMethod::Native,
                    ocr_confidence: None,
                });
            }
        }

        Ok(ProcessedDocument {
            filename: path.file_name().unwrap().to_string_lossy().to_string(),
            pages,
            metadata: self.extract_metadata(&doc)?,
        })
    }
}
```

### 7.3 Chunking Strategy

```rust
pub struct LegalChunker {
    target_size: usize,  // 500 tokens
    max_size: usize,     // 1000 tokens
    min_size: usize,     // 100 tokens
    overlap: usize,      // 50 tokens
}

impl LegalChunker {
    pub fn chunk(&self, doc: &ProcessedDocument) -> Vec<Chunk> {
        let mut chunks = Vec::new();

        for page in &doc.pages {
            let paragraphs = self.split_paragraphs(&page.content);

            let mut current_chunk = String::new();
            let mut chunk_start_para = 0;
            let mut chunk_start_line = 0;

            for (para_idx, paragraph) in paragraphs.iter().enumerate() {
                let para_tokens = self.count_tokens(paragraph);

                if self.count_tokens(&current_chunk) + para_tokens > self.target_size
                   && !current_chunk.is_empty() {
                    // Emit current chunk
                    chunks.push(Chunk {
                        text: current_chunk.clone(),
                        provenance: Provenance {
                            document_id: doc.id,
                            page: page.number,
                            paragraph_start: chunk_start_para as u32,
                            paragraph_end: para_idx as u32,
                            line_start: chunk_start_line as u32,
                            // ... more fields
                        },
                    });

                    // Start new chunk with overlap
                    current_chunk = self.get_overlap(&current_chunk);
                    chunk_start_para = para_idx;
                }

                current_chunk.push_str(paragraph);
                current_chunk.push('\n');
            }

            // Emit final chunk for page
            if !current_chunk.is_empty() {
                chunks.push(Chunk { /* ... */ });
            }
        }

        chunks
    }
}
```

---

## 8. Case Management

### 8.1 Case Model

```rust
/// A legal case/matter containing documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    /// Unique identifier
    pub id: Uuid,

    /// Display name (e.g., "Smith v. Jones")
    pub name: String,

    /// Optional case/docket number
    pub case_number: Option<String>,

    /// Case type
    pub case_type: CaseType,

    /// Current status
    pub status: CaseStatus,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last modification
    pub updated_at: DateTime<Utc>,

    /// Document statistics
    pub stats: CaseStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseStats {
    pub document_count: u32,
    pub page_count: u32,
    pub chunk_count: u32,
    pub storage_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CaseType {
    Civil,
    Criminal,
    Family,
    Bankruptcy,
    Contract,
    Employment,
    PersonalInjury,
    RealEstate,
    IntellectualProperty,
    Immigration,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CaseStatus {
    Active,
    Closed,
    Archived,
}
```

### 8.2 Case Isolation

Each case has completely isolated storage:

```
~/Documents/CaseTrack/
├── registry.db                    # Case index (shared)
├── models/                        # Embedding models (shared)
│   ├── bge-small-en-v1.5/
│   ├── splade-distil/
│   └── ...
└── cases/
    ├── {case-uuid-1}/            # Case A (isolated)
    │   ├── case.db/              # RocksDB instance
    │   │   ├── documents         # Document metadata
    │   │   ├── chunks            # Chunk text
    │   │   ├── embeddings        # Vector storage
    │   │   ├── provenance        # Source tracking
    │   │   └── bm25_index        # Inverted index
    │   └── originals/            # Original files (optional)
    │       ├── Complaint.pdf
    │       └── ...
    ├── {case-uuid-2}/            # Case B (isolated)
    │   └── ...
    └── ...
```

### 8.3 Case Registry

```rust
/// Central registry managing all cases
pub struct CaseRegistry {
    db: rocksdb::DB,
    active_case: Option<Uuid>,
    data_dir: PathBuf,
}

impl CaseRegistry {
    pub fn create_case(&mut self, params: CreateCaseParams) -> Result<Case> {
        let id = Uuid::new_v4();
        let case_dir = self.data_dir.join("cases").join(id.to_string());

        // Create isolated database for this case
        fs::create_dir_all(&case_dir)?;

        let case = Case {
            id,
            name: params.name,
            case_number: params.case_number,
            case_type: params.case_type.unwrap_or(CaseType::Other),
            status: CaseStatus::Active,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            stats: CaseStats::default(),
        };

        // Store in registry
        let key = format!("case:{}", id);
        self.db.put(key.as_bytes(), bincode::serialize(&case)?)?;

        Ok(case)
    }

    pub fn switch_case(&mut self, case_id: Uuid) -> Result<CaseHandle> {
        // Validate case exists
        let case = self.get_case(case_id)?;

        // Open case database
        let case_dir = self.data_dir.join("cases").join(case_id.to_string());
        let handle = CaseHandle::open(&case_dir)?;

        self.active_case = Some(case_id);

        Ok(handle)
    }

    pub fn list_cases(&self) -> Result<Vec<Case>> {
        let mut cases = Vec::new();

        let iter = self.db.prefix_iterator(b"case:");
        for item in iter {
            let (_, value) = item?;
            let case: Case = bincode::deserialize(&value)?;
            cases.push(case);
        }

        cases.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        Ok(cases)
    }
}
```

---

## 9. Provenance System

### 9.1 Provenance Model

Every chunk tracks exactly where it came from:

```rust
/// Complete source tracking for a chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Source document
    pub document_id: Uuid,
    pub document_name: String,
    pub document_path: Option<PathBuf>,

    /// Location in document
    pub page: u32,
    pub paragraph_start: u32,
    pub paragraph_end: u32,
    pub line_start: u32,
    pub line_end: u32,

    /// Character offsets (for highlighting)
    pub char_start: u64,
    pub char_end: u64,

    /// Extraction metadata
    pub extraction_method: ExtractionMethod,
    pub ocr_confidence: Option<f32>,

    /// Optional Bates number
    pub bates_number: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExtractionMethod {
    Native,  // Direct text extraction
    Ocr,     // Tesseract OCR
}

impl Provenance {
    /// Generate a legal citation string
    pub fn cite(&self) -> String {
        let mut parts = vec![self.document_name.clone()];

        parts.push(format!("p. {}", self.page));

        if self.paragraph_start == self.paragraph_end {
            parts.push(format!("¶ {}", self.paragraph_start));
        } else {
            parts.push(format!("¶¶ {}-{}", self.paragraph_start, self.paragraph_end));
        }

        if let Some(bates) = &self.bates_number {
            parts.push(format!("({})", bates));
        }

        parts.join(", ")
    }

    /// Short citation for inline use
    pub fn cite_short(&self) -> String {
        if let Some(bates) = &self.bates_number {
            bates.clone()
        } else {
            format!("{}, p. {}",
                self.document_name.split('.').next().unwrap_or(&self.document_name),
                self.page
            )
        }
    }
}
```

### 9.2 Search Results with Provenance

```rust
/// A search result with full source citation
#[derive(Debug, Serialize)]
pub struct SearchResult {
    /// The matching text
    pub text: String,

    /// Relevance score (0.0 - 1.0)
    pub score: f32,

    /// Full provenance for citation
    pub provenance: Provenance,

    /// Pre-formatted citations
    pub citation: String,
    pub citation_short: String,

    /// Surrounding context
    pub context_before: Option<String>,
    pub context_after: Option<String>,
}

impl SearchResult {
    pub fn to_mcp_response(&self) -> serde_json::Value {
        json!({
            "text": self.text,
            "score": self.score,
            "citation": self.citation,
            "citation_short": self.citation_short,
            "source": {
                "document": self.provenance.document_name,
                "page": self.provenance.page,
                "paragraph": self.provenance.paragraph_start,
                "lines": format!("{}-{}", self.provenance.line_start, self.provenance.line_end),
                "bates": self.provenance.bates_number,
            },
            "context": {
                "before": self.context_before,
                "after": self.context_after,
            }
        })
    }
}
```

---

## 10. Search & Retrieval

### 10.1 Search Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4-STAGE SEARCH PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: "What does the contract say about early termination?"              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: BM25 RECALL                                    [<5ms]      │   │
│  │                                                                      │   │
│  │ • E13 inverted index lookup                                         │   │
│  │ • Terms: "contract", "early", "termination"                         │   │
│  │ • Fast lexical matching                                             │   │
│  │                                                                      │   │
│  │ Output: 500 candidate chunks                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: SEMANTIC RANKING                               [<80ms]     │   │
│  │                                                                      │   │
│  │ • E1-LEGAL: Semantic similarity (384D dense)                        │   │
│  │ • E6-LEGAL: Keyword expansion (sparse)                              │   │
│  │ • Score fusion via Reciprocal Rank Fusion (RRF)                     │   │
│  │                                                                      │   │
│  │ Output: 100 candidates, ranked                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: MULTI-SIGNAL BOOST                             [<30ms]     │   │
│  │ (PRO TIER ONLY)                                                     │   │
│  │                                                                      │   │
│  │ • E7: Boost structured text matches                                 │   │
│  │ • E8-LEGAL: Boost citation similarity                               │   │
│  │ • E11-LEGAL: Boost entity matches                                   │   │
│  │                                                                      │   │
│  │ Weights: 0.6 × semantic + 0.2 × structure + 0.1 × citation          │   │
│  │          + 0.1 × entity                                              │   │
│  │                                                                      │   │
│  │ Output: 50 candidates, re-ranked                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: COLBERT RERANK                                 [<100ms]    │   │
│  │ (PRO TIER ONLY)                                                     │   │
│  │                                                                      │   │
│  │ • E12: Token-level MaxSim scoring                                   │   │
│  │ • Ensures exact phrase matches rank highest                         │   │
│  │ • "early termination" > "termination that was early"                │   │
│  │                                                                      │   │
│  │ Output: Top K results with provenance                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LATENCY TARGETS                                                            │
│  ─────────────────                                                          │
│  Free tier (Stages 1-2):  <100ms                                           │
│  Pro tier (Stages 1-4):   <200ms                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Search Implementation

```rust
pub struct SearchEngine {
    embedder: EmbeddingEngine,
    bm25: Bm25Index,
    tier: LicenseTier,
}

impl SearchEngine {
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // Stage 1: BM25 recall
        let candidates = self.bm25.search(query, 500)?;

        // Stage 2: Semantic ranking
        let query_e1 = self.embedder.embed_query(query, EmbedderId::E1Legal)?;
        let query_e6 = self.embedder.embed_query(query, EmbedderId::E6Legal)?;

        let mut scored: Vec<(ChunkId, f32)> = candidates
            .into_iter()
            .map(|chunk_id| {
                let e1_score = cosine_similarity(&query_e1, &chunk.embeddings.e1_legal);
                let e6_score = sparse_dot(&query_e6, &chunk.embeddings.e6_legal);
                let rrf_score = rrf_fusion(&[e1_score, e6_score], &[1.0, 0.8]);
                (chunk_id, rrf_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(100);

        // Stage 3: Multi-signal boost (Pro only)
        if self.tier.is_pro() {
            scored = self.apply_pro_signals(query, scored)?;
        }

        // Stage 4: ColBERT rerank (Pro only)
        if self.tier.is_pro() {
            scored = self.colbert_rerank(query, scored)?;
        }

        // Build results with provenance
        let results = scored
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, score)| self.build_result(chunk_id, score))
            .collect::<Result<Vec<_>>>()?;

        Ok(results)
    }

    fn build_result(&self, chunk_id: ChunkId, score: f32) -> Result<SearchResult> {
        let chunk = self.storage.get_chunk(chunk_id)?;
        let provenance = self.storage.get_provenance(chunk_id)?;

        Ok(SearchResult {
            text: chunk.text,
            score,
            provenance: provenance.clone(),
            citation: provenance.cite(),
            citation_short: provenance.cite_short(),
            context_before: self.get_context_before(chunk_id)?,
            context_after: self.get_context_after(chunk_id)?,
        })
    }
}

/// Reciprocal Rank Fusion
fn rrf_fusion(scores: &[f32], weights: &[f32]) -> f32 {
    const K: f32 = 60.0;

    scores.iter()
        .zip(weights.iter())
        .map(|(score, weight)| weight / (K + (1.0 / score)))
        .sum()
}
```

---

## 11. MCP Tools

### 11.1 Tool Overview

| Tool | Description | Tier |
|------|-------------|------|
| `create_case` | Create a new case | Free |
| `list_cases` | List all cases | Free |
| `switch_case` | Switch active case | Free |
| `delete_case` | Delete a case | Free |
| `ingest_pdf` | Ingest a PDF document | Free |
| `ingest_docx` | Ingest a Word document | Free |
| `ingest_image` | Ingest image via OCR | Free |
| `ingest_folder` | Batch ingest folder | Pro |
| `list_documents` | List case documents | Free |
| `search_case` | Search current case | Free (limited) |
| `find_entity` | Find entity mentions | Pro |
| `get_document` | Get document details | Free |

### 11.2 Tool Specifications

#### `create_case`

```json
{
  "name": "create_case",
  "description": "Create a new legal case. All documents will be stored in this case's isolated database.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Case name (e.g., 'Smith v. Jones')"
      },
      "case_number": {
        "type": "string",
        "description": "Optional docket or case number"
      },
      "case_type": {
        "type": "string",
        "enum": ["civil", "criminal", "family", "bankruptcy", "contract", "employment", "personal_injury", "real_estate", "intellectual_property", "immigration", "other"]
      }
    },
    "required": ["name"]
  }
}
```

**Example:**
```
User: Create a new case called "Smith v. Jones Corp" for a contract dispute

Claude: [calls create_case with name="Smith v. Jones Corp", case_type="contract"]

Response: Created case "Smith v. Jones Corp" (ID: abc-123).
          This is now your active case.
```

#### `ingest_pdf`

```json
{
  "name": "ingest_pdf",
  "description": "Ingest a PDF document into the current case. Extracts text (with OCR for scans), chunks, and embeds for search.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to the PDF file"
      },
      "document_name": {
        "type": "string",
        "description": "Optional display name (defaults to filename)"
      },
      "document_type": {
        "type": "string",
        "enum": ["pleading", "motion", "brief", "contract", "exhibit", "correspondence", "deposition", "discovery", "other"]
      }
    },
    "required": ["file_path"]
  }
}
```

**Example:**
```
User: Add the complaint from ~/Downloads/Complaint.pdf

Claude: [calls ingest_pdf with file_path="~/Downloads/Complaint.pdf"]

Response: Ingested "Complaint.pdf"
          • 45 pages
          • 234 chunks
          • Processing time: 12 seconds
          • Method: Native text extraction
```

#### `search_case`

```json
{
  "name": "search_case",
  "description": "Search across all documents in the current case. Returns results with full source citations.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "maximum": 50,
        "description": "Number of results to return"
      },
      "document_filter": {
        "type": "string",
        "description": "Optional: only search specific document"
      }
    },
    "required": ["query"]
  }
}
```

**Example Response:**
```json
{
  "query": "early termination clause",
  "results": [
    {
      "text": "Either party may terminate this Agreement upon thirty (30) days written notice...",
      "score": 0.94,
      "citation": "Contract.pdf, p. 12, ¶ 8.1",
      "citation_short": "Contract, p. 12",
      "source": {
        "document": "Contract.pdf",
        "page": 12,
        "paragraph": 8,
        "lines": "1-4"
      }
    },
    {
      "text": "In the event of early termination, the non-breaching party shall be entitled to...",
      "score": 0.89,
      "citation": "Contract.pdf, p. 13, ¶ 8.3",
      "citation_short": "Contract, p. 13",
      "source": {
        "document": "Contract.pdf",
        "page": 13,
        "paragraph": 10,
        "lines": "1-6"
      }
    }
  ],
  "total_searched": 234,
  "search_time_ms": 87
}
```

---

## 12. Monetization

### 12.1 Pricing Tiers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRICING TIERS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FREE TIER                                                                  │
│  ══════════                                                                 │
│  Price: $0                                                                  │
│                                                                             │
│  Limits:                                                                    │
│  • 3 active cases                                                          │
│  • 100 documents per case                                                  │
│  • 1,000 pages total                                                       │
│                                                                             │
│  Features:                                                                  │
│  • 4 embedders (E1, E6, E7, E13)                                          │
│  • Basic semantic search                                                   │
│  • Full provenance tracking                                                │
│  • PDF, DOCX, image ingestion                                             │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  PRO TIER                                                                   │
│  ════════                                                                   │
│  Price: $29/month or $249/year (29% discount)                              │
│                                                                             │
│  Limits:                                                                    │
│  • Unlimited cases                                                         │
│  • Unlimited documents                                                     │
│  • Unlimited pages                                                         │
│                                                                             │
│  Features (everything in Free, plus):                                      │
│  • All 7 embedders (adds E8, E11, E12)                                    │
│  • ColBERT precision reranking                                            │
│  • Entity search                                                           │
│  • Citation matching                                                       │
│  • Batch folder ingestion                                                 │
│  • Priority email support                                                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  FIRM TIER                                                                  │
│  ═════════                                                                  │
│  Price: $99/month or $899/year (24% discount)                              │
│                                                                             │
│  Everything in Pro, plus:                                                  │
│  • 5 seats (same license key)                                             │
│  • Phone support                                                           │
│  • Bulk ingestion optimizations                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 License Key System

```rust
/// License validation (offline-first)
pub struct LicenseManager {
    cache_path: PathBuf,
    public_key: ed25519::PublicKey,  // Embedded in binary
}

impl LicenseManager {
    /// Validate license key
    pub fn validate(&self, key: Option<&str>) -> LicenseTier {
        // No key = free tier
        let Some(key) = key else {
            return LicenseTier::Free;
        };

        // Check cache first (offline validation)
        if let Ok(cached) = self.load_cached(key) {
            if !cached.is_expired() {
                return cached.tier;
            }
        }

        // Validate signature offline
        // Key format: TIER-XXXXXX-XXXXXX-XXXXXX-SIG
        match self.validate_signature(key) {
            Ok(tier) => {
                self.cache_license(key, tier);
                tier
            }
            Err(_) => {
                // Try online validation (first activation)
                self.validate_online(key).unwrap_or(LicenseTier::Free)
            }
        }
    }

    fn validate_signature(&self, key: &str) -> Result<LicenseTier> {
        let parts: Vec<&str> = key.split('-').collect();
        if parts.len() != 5 {
            return Err(LicenseError::InvalidFormat);
        }

        let tier = match parts[0] {
            "PRO" => LicenseTier::Pro,
            "FIRM" => LicenseTier::Firm,
            "ENT" => LicenseTier::Enterprise,
            _ => return Err(LicenseError::InvalidTier),
        };

        let payload = format!("{}-{}-{}-{}", parts[0], parts[1], parts[2], parts[3]);
        let signature = base64::decode(parts[4])?;

        if self.public_key.verify(payload.as_bytes(), &signature).is_ok() {
            Ok(tier)
        } else {
            Err(LicenseError::InvalidSignature)
        }
    }
}
```

### 12.3 Feature Gating

```rust
impl LicenseTier {
    pub fn max_cases(&self) -> Option<u32> {
        match self {
            Self::Free => Some(3),
            _ => None,
        }
    }

    pub fn max_docs_per_case(&self) -> Option<u32> {
        match self {
            Self::Free => Some(100),
            _ => None,
        }
    }

    pub fn available_embedders(&self) -> Vec<EmbedderId> {
        match self {
            Self::Free => vec![
                EmbedderId::E1Legal,
                EmbedderId::E6Legal,
                EmbedderId::E7,
                // E13 (BM25) always available
            ],
            _ => vec![
                EmbedderId::E1Legal,
                EmbedderId::E6Legal,
                EmbedderId::E7,
                EmbedderId::E8Legal,
                EmbedderId::E11Legal,
                EmbedderId::E12,
            ],
        }
    }

    pub fn can_batch_ingest(&self) -> bool {
        !matches!(self, Self::Free)
    }
}
```

### 12.4 Upgrade Flow

When users hit limits, guide them to upgrade:

```rust
pub fn check_case_limit(registry: &CaseRegistry, tier: LicenseTier) -> Result<()> {
    if let Some(max) = tier.max_cases() {
        let current = registry.count_cases()?;
        if current >= max {
            return Err(McpError::new(
                ErrorCode::InvalidRequest,
                format!(
                    "Free tier allows {} cases (you have {}). \
                     Upgrade to Pro for unlimited cases: https://casetrack.legal/upgrade",
                    max, current
                ),
            ));
        }
    }
    Ok(())
}
```

---

## 13. Technical Implementation

### 13.1 Crate Structure

```
crates/
├── casetrack/                    # Main binary crate
│   ├── src/
│   │   ├── main.rs              # Entry point, MCP server setup
│   │   ├── server.rs            # MCP tool implementations
│   │   └── cli.rs               # CLI argument parsing
│   └── Cargo.toml
│
├── casetrack-core/              # Core library
│   ├── src/
│   │   ├── lib.rs
│   │   ├── case/                # Case management
│   │   │   ├── mod.rs
│   │   │   ├── registry.rs
│   │   │   └── handle.rs
│   │   ├── document/            # Document processing
│   │   │   ├── mod.rs
│   │   │   ├── pdf.rs
│   │   │   ├── docx.rs
│   │   │   ├── ocr.rs
│   │   │   └── chunker.rs
│   │   ├── embedding/           # Embedding engine
│   │   │   ├── mod.rs
│   │   │   ├── engine.rs
│   │   │   ├── models.rs
│   │   │   └── download.rs
│   │   ├── search/              # Search & retrieval
│   │   │   ├── mod.rs
│   │   │   ├── engine.rs
│   │   │   ├── bm25.rs
│   │   │   └── ranking.rs
│   │   ├── provenance/          # Source tracking
│   │   │   ├── mod.rs
│   │   │   └── citation.rs
│   │   ├── storage/             # RocksDB layer
│   │   │   ├── mod.rs
│   │   │   └── schema.rs
│   │   └── license/             # License validation
│   │       ├── mod.rs
│   │       └── validator.rs
│   └── Cargo.toml
│
└── Cargo.toml                   # Workspace
```

### 13.2 Dependencies

```toml
[workspace.package]
version = "1.0.0"
edition = "2021"
license = "Commercial"

[workspace.dependencies]
# MCP
rmcp = { version = "0.13", features = ["server", "transport-io", "macros"] }

# Async runtime
tokio = { version = "1.35", features = ["full"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# Storage
rocksdb = "0.22"

# ML inference
ort = { version = "2.0", features = ["download-binaries"] }

# PDF processing
pdf-extract = "0.7"
lopdf = "0.31"

# DOCX processing
docx-rs = "0.4"

# OCR (optional feature)
tesseract = { version = "0.14", optional = true }

# Model download
hf-hub = "0.3"

# Utilities
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
thiserror = "1.0"
tracing = "0.1"
```

### 13.3 Build Configuration

```toml
# Cargo.toml for main binary

[package]
name = "casetrack"
version.workspace = true

[[bin]]
name = "casetrack"
path = "src/main.rs"

[features]
default = ["ocr"]
ocr = ["tesseract"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
```

---

## 14. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

```
WEEK 1: Project Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Create workspace structure
□ Set up CI/CD (GitHub Actions)
□ Implement case registry (create, list, switch, delete)
□ RocksDB storage layer
□ Basic MCP server with rmcp

WEEK 2: Document Processing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ PDF text extraction (native)
□ DOCX parsing
□ Chunking with provenance
□ ingest_pdf and ingest_docx tools

WEEK 3: Basic Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ BM25 index implementation
□ E1-LEGAL embedding (bge-small)
□ Model download via hf-hub
□ search_case tool (BM25 + E1 only)
□ Provenance in search results
```

### Phase 2: Full Embedder Stack (Weeks 4-6)

```
WEEK 4: Additional Embedders
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ E6-LEGAL (SPLADE keywords)
□ E7 (MiniLM structured)
□ ONNX Runtime integration
□ Model lazy loading

WEEK 5: Pro Embedders
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ E8-LEGAL (citation)
□ E11-LEGAL (entity)
□ E12 (ColBERT rerank)
□ 4-stage search pipeline

WEEK 6: OCR & Scanned Documents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Tesseract integration
□ Scanned PDF detection
□ Image preprocessing
□ ingest_image tool
```

### Phase 3: Distribution (Weeks 7-8)

```
WEEK 7: MCPB Bundle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ manifest.json specification
□ Cross-platform builds (cargo-dist)
□ Bundle creation script
□ Test with Claude Desktop

WEEK 8: Licensing & Monetization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ License key validation
□ Feature gating
□ Lemon Squeezy integration
□ Upgrade flow in error messages
```

### Phase 4: Polish (Weeks 9-10)

```
WEEK 9: Testing & Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Integration tests
□ Performance benchmarks
□ Memory optimization
□ Error handling polish

WEEK 10: Documentation & Launch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ User documentation
□ Landing page
□ Video demo
□ Product Hunt launch prep
```

---

## 15. Success Metrics

### 15.1 Product Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Install success rate | >95% | Telemetry (opt-in) |
| Time to first search | <5 minutes | User testing |
| Search relevance (top 5) | >85% | Manual evaluation |
| Provenance accuracy | 100% | Automated tests |
| Crash rate | <0.1% | Error reporting |

### 15.2 Performance Metrics

| Metric | Free Tier | Pro Tier |
|--------|-----------|----------|
| Search latency (p95) | <150ms | <250ms |
| Ingestion speed | <1.5s/page | <1s/page |
| RAM usage (idle) | <500MB | <800MB |
| RAM usage (search) | <1.5GB | <2GB |
| Model download | <3 min | <5 min |

### 15.3 Business Metrics

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Downloads | 10,000 | 50,000 | 200,000 |
| Free users | 5,000 | 25,000 | 100,000 |
| Pro conversions (2%) | 100 | 500 | 2,000 |
| Firm conversions | 20 | 100 | 400 |
| ARR | $58K | $318K | $1.37M |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Chunk** | A ~500 token segment of a document, the unit of search |
| **Embedder** | A model that converts text to a numerical vector |
| **MCPB** | MCP Bundle - a ZIP file format for distributing MCP servers |
| **MCP** | Model Context Protocol - standard for AI tool integration |
| **ONNX** | Open Neural Network Exchange - cross-platform ML format |
| **Provenance** | The source location (document, page, paragraph) of text |
| **RRF** | Reciprocal Rank Fusion - method to combine search rankings |

## Appendix B: File Size Estimates

| Component | Size |
|-----------|------|
| MCPB bundle (binaries + resources) | ~50MB |
| Models (Free tier) | ~165MB |
| Models (Pro tier) | ~370MB |
| Case database (per 100 docs) | ~50MB |
| Total install (Free, 1 case) | ~265MB |
| Total install (Pro, 10 cases) | ~920MB |

## Appendix C: Comparison with Alternatives

| Feature | CaseTrack | Casetext | Westlaw | DIY RAG |
|---------|-----------|----------|---------|---------|
| Price | $0-29/mo | $200/mo | $400/mo | Free |
| Install time | 2 min | N/A | N/A | Hours |
| Runs locally | Yes | No | No | Yes |
| No GPU required | Yes | N/A | N/A | Usually no |
| Claude integration | Native | No | No | Manual |
| Provenance | Always | Sometimes | Sometimes | DIY |
| Legal-specific | Yes | Yes | Yes | No |

---

*Document version: 3.1.0 | Last updated: 2026-01-28*
