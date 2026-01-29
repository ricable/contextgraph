# PRD 02: Target User & Hardware

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Primary Users

```
PRIMARY: SOLO PRACTITIONERS & SMALL FIRMS
=================================================================================

Profile:
  - 1-10 attorney firm
  - No dedicated IT staff
  - Uses consumer hardware (MacBook, Windows laptop)
  - Handles 5-50 active matters
  - Documents stored in folders on local drive or cloud sync (Dropbox, OneDrive)
  - Already uses Claude Code or Claude Desktop for other work

Pain Points:
  - Can't find documents they know exist
  - Spend hours re-reading to find specific facts
  - No budget for enterprise legal tech ($500+/seat/month)
  - Frustrated by keyword search limitations
  - Need to cite sources precisely for court filings

Why CaseTrack:
  - Works on their existing laptop
  - No IT support needed
  - Affordable ($29/month or free tier)
  - Immediate productivity boost
  - Integrates with Claude they already use
```

---

## 2. Secondary Users

```
SECONDARY: PARALEGALS & LEGAL ASSISTANTS
=================================================================================

Profile:
  - Support 1-5 attorneys
  - Manage document collections per case
  - Responsible for organizing and indexing case files
  - Often do initial research and fact-finding

Pain Points:
  - Manual document review is tedious and error-prone
  - No way to semantically search across hundreds of documents
  - Need to produce exact citations for attorney review

Why CaseTrack:
  - Batch ingest entire case folders at once
  - Search returns cited sources ready for attorney review
  - Reduces manual document review by 70%+
```

```
TERTIARY: LAW STUDENTS & RESEARCHERS
=================================================================================

Profile:
  - Studying case law, writing papers
  - Limited budget (free tier)
  - Comfortable with CLI tools

Pain Points:
  - Organizing research materials across multiple cases
  - Finding connections between documents
  - Citing sources accurately

Why CaseTrack:
  - Free tier handles 3 cases with full search
  - Provenance system generates proper citations
```

---

## 3. User Personas

### Persona A: Sarah (Solo Practitioner)

- **Age**: 42
- **Practice**: Family law, solo
- **Hardware**: MacBook Air M1, 8GB RAM
- **Tech comfort**: Uses email, Word, basic cloud storage
- **Current workflow**: Ctrl+F in PDFs, manual notes in Word
- **CaseTrack use**: Ingests all case documents, asks Claude "What did the respondent claim about custody in the deposition?"
- **Key need**: Just works, no setup friction

### Persona B: Mike (Small Firm Partner)

- **Age**: 55
- **Practice**: Contract/commercial litigation, 5-attorney firm
- **Hardware**: Windows 11 desktop, 16GB RAM
- **Tech comfort**: Moderate, uses practice management software
- **Current workflow**: Associates do manual document review
- **CaseTrack use**: Firm license, each attorney searches their own cases
- **Key need**: Works on Windows, multiple seats, worth $99/month

### Persona C: Alex (Paralegal)

- **Age**: 28
- **Practice**: Personal injury firm
- **Hardware**: Windows 10 laptop, 8GB RAM
- **Tech comfort**: High, uses multiple software tools daily
- **Current workflow**: Manually indexes documents in spreadsheets
- **CaseTrack use**: Batch ingests entire case folders, builds searchable case databases
- **Key need**: Fast ingestion, reliable OCR for scanned documents

---

## 4. Minimum Hardware Requirements

```
MINIMUM REQUIREMENTS (Must Run)
=================================================================================

CPU:     Any 64-bit processor (2018 or newer recommended)
         - Intel Core i3 or better
         - AMD Ryzen 3 or better
         - Apple M1 or better

RAM:     8GB minimum
         - 16GB recommended for large cases (1000+ pages)

Storage: 5GB available
         - 400MB for embedding models (one-time download)
         - 4.6GB for case data (scales with usage)
         - SSD strongly recommended (HDD works but slower ingestion)

OS:      - macOS 11 (Big Sur) or later
         - Windows 10 (64-bit) or later
         - Ubuntu 20.04 or later (other Linux distros likely work)

GPU:     NOT REQUIRED
         - Optional: Metal (macOS), CUDA (NVIDIA), DirectML (Windows)
         - GPU provides ~2x speedup for ingestion if available
         - Search latency unaffected (small batch sizes)

Network: Required ONLY for:
         - Initial model download (~400MB, one-time)
         - License activation (one-time, then cached offline)
         - Software updates (optional)
         ALL document processing is 100% offline

Prerequisites:
         - Claude Code or Claude Desktop installed
         - No other runtime dependencies (Rust binary is self-contained)
         - Tesseract OCR bundled with binary (no separate install)
```

---

## 5. Performance by Hardware Tier

### 5.1 Ingestion Performance

| Hardware | 50-page PDF | 500-page PDF | OCR (50 scanned pages) |
|----------|-------------|--------------|------------------------|
| **Entry** (M1 Air 8GB) | 45 seconds | 7 minutes | 3 minutes |
| **Mid** (M2 Pro 16GB) | 25 seconds | 4 minutes | 2 minutes |
| **High** (i7 32GB) | 20 seconds | 3 minutes | 90 seconds |
| **With GPU** (RTX 3060) | 10 seconds | 90 seconds | 45 seconds |

### 5.2 Search Performance

| Hardware | Free Tier (2-stage) | Pro Tier (4-stage) | Concurrent Models |
|----------|--------------------|--------------------|-------------------|
| **Entry** (M1 Air 8GB) | 100ms | 200ms | 3 (lazy loaded) |
| **Mid** (M2 Pro 16GB) | 60ms | 120ms | 5 |
| **High** (i7 32GB) | 40ms | 80ms | 7 (all loaded) |
| **With GPU** (RTX 3060) | 20ms | 50ms | 7 (all loaded) |

### 5.3 Memory Usage

| Scenario | RAM Usage |
|----------|-----------|
| Idle (server running, no models loaded) | ~50MB |
| Free tier (3 models loaded) | ~800MB |
| Pro tier (6 models loaded) | ~1.5GB |
| During ingestion (peak) | +300MB above baseline |
| During search (peak) | +100MB above baseline |

---

## 6. Supported Platforms

### 6.1 Build Targets

| Platform | Architecture | Binary Name | Status |
|----------|-------------|-------------|--------|
| macOS | x86_64 (Intel) | `casetrack-darwin-x64` | Supported |
| macOS | aarch64 (Apple Silicon) | `casetrack-darwin-arm64` | Supported |
| Windows | x86_64 | `casetrack-win32-x64.exe` | Supported |
| Linux | x86_64 | `casetrack-linux-x64` | Supported |
| Linux | aarch64 | `casetrack-linux-arm64` | Future |

### 6.2 Platform-Specific Notes

**macOS:**
- CoreML execution provider available for ~2x inference speedup
- Universal binary option (fat binary for Intel + Apple Silicon)
- Code signing required for Gatekeeper (`codesign --sign`)
- Notarization required for distribution outside App Store

**Windows:**
- DirectML execution provider available for GPU acceleration
- Binary should be signed with Authenticode certificate
- Windows Defender may flag unsigned binaries
- Long path support: use `\\?\` prefix or registry setting

**Linux:**
- CPU-only by default; CUDA available if NVIDIA drivers present
- Statically linked against musl for maximum compatibility
- AppImage format as alternative distribution

### 6.3 Claude Integration Compatibility

| Client | Transport | Config Location | Status |
|--------|-----------|-----------------|--------|
| Claude Code (CLI) | stdio | `~/.claude/settings.json` | Primary target |
| Claude Desktop (macOS) | stdio | `~/Library/Application Support/Claude/claude_desktop_config.json` | Supported |
| Claude Desktop (Windows) | stdio | `%APPDATA%\Claude\claude_desktop_config.json` | Supported |
| Claude Desktop (Linux) | stdio | `~/.config/Claude/claude_desktop_config.json` | Supported |

---

## 7. Graceful Degradation Strategy

CaseTrack adapts to available hardware:

```
DEGRADATION TIERS
=================================================================================

TIER 1: FULL (16GB+ RAM)
  - All models loaded in memory simultaneously
  - Zero model loading latency on search
  - Maximum ingestion throughput (parallel embedding)

TIER 2: STANDARD (8-16GB RAM)
  - Free models always loaded (E1, E6, E7 = ~800MB)
  - Pro models lazy-loaded on demand
  - ~200ms model load penalty on first Pro search
  - Models stay loaded after first use until memory pressure

TIER 3: CONSTRAINED (<8GB RAM)
  - Only E1 + E13 (BM25) always loaded (~400MB)
  - Other models loaded one at a time, unloaded after use
  - Ingestion uses sequential (not parallel) embedding
  - Search still works but with higher latency
  - Warning shown on startup: "Low memory mode active"

DETECTION:
  - On startup, check available RAM via sysinfo crate
  - Set tier automatically, log the decision
  - User can override via --memory-mode=full|standard|constrained
```

---

*CaseTrack PRD v4.0.0 -- Document 2 of 10*
