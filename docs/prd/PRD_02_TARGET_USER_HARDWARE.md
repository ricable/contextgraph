# PRD 02: Target User & Hardware

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Target Users

| Segment | Profile | Key Pain Point | Why CaseTrack |
|---------|---------|---------------|---------------|
| **Primary: Solo Professionals** (consultants, analysts, researchers) | No IT staff, consumer hardware, 5-50 active projects, already uses Claude | Can't semantically search documents; no budget for enterprise platforms ($500+/seat) | Works on existing laptop, no setup, $29/mo or free tier |
| **Secondary: Small Teams** (5-20 people) | Standard business hardware, shared document collections, need consistent search across team members | Manual document review is tedious; need exact citations for reporting and collaboration | Batch ingest folders, search returns cited sources, 70%+ review reduction |
| **Tertiary: Enterprise Departments** | Mixed hardware across the organization, limited IT support for individual tools | Organizing research, finding cross-document connections, accurate citations across large document sets | Free tier (3 collections), provenance generates proper citations |

---

## 2. User Personas

| Persona | Role / Domain | Hardware | CaseTrack Use | Key Need |
|---------|--------------|----------|---------------|----------|
| **Sarah** | Solo management consultant | MacBook Air M1, 8GB | Ingests client deliverables and research, asks Claude questions about project documents | Zero setup friction |
| **Mike** | Team lead at a market research firm (12 analysts) | Windows 11, 16GB | Team license, each analyst searches their own collections | Windows support, multi-seat, worth $99/mo |
| **Alex** | Operations coordinator, mid-size company | Windows 10, 8GB | Batch ingests policy documents and reports, builds searchable knowledge bases | Fast ingestion, reliable OCR |

---

## 3. Minimum Hardware Requirements

```
MINIMUM REQUIREMENTS (Must Run)
=================================================================================

CPU:     Any 64-bit processor (2018 or newer recommended)
         - Intel Core i3 or better
         - AMD Ryzen 3 or better
         - Apple M1 or better

RAM:     8GB minimum
         - 16GB recommended for large collections (1000+ pages)

Storage: 5GB available
         - 400MB for embedding models (one-time download)
         - 4.6GB for collection data (scales with usage)
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

## 4. Performance by Hardware Tier

### 4.1 Ingestion Performance

| Hardware | 50-page PDF | 500-page PDF | OCR (50 scanned pages) |
|----------|-------------|--------------|------------------------|
| **Entry** (M1 Air 8GB) | 45 seconds | 7 minutes | 3 minutes |
| **Mid** (M2 Pro 16GB) | 25 seconds | 4 minutes | 2 minutes |
| **High** (i7 32GB) | 20 seconds | 3 minutes | 90 seconds |
| **With GPU** (RTX 3060) | 10 seconds | 90 seconds | 45 seconds |

### 4.2 Search Performance

| Hardware | Free Tier (2-stage) | Pro Tier (3-stage) | Concurrent Models |
|----------|--------------------|--------------------|-------------------|
| **Entry** (M1 Air 8GB) | 100ms | 200ms | 2 (lazy loaded) |
| **Mid** (M2 Pro 16GB) | 60ms | 120ms | 3 |
| **High** (i7 32GB) | 40ms | 80ms | 3 (all loaded) |
| **With GPU** (RTX 3060) | 20ms | 50ms | 3 (all loaded) |

### 4.3 Memory Usage

| Scenario | RAM Usage |
|----------|-----------|
| Idle (server running, no models loaded) | ~50MB |
| Free tier (3 models loaded) | ~800MB |
| Pro tier (all models loaded) | ~1.5GB |
| During ingestion (peak) | +300MB above baseline |
| During search (peak) | +100MB above baseline |

---

## 5. Supported Platforms

### 5.1 Build Targets

| Platform | Architecture | Binary Name | Status |
|----------|-------------|-------------|--------|
| macOS | x86_64 (Intel) | `casetrack-darwin-x64` | Supported |
| macOS | aarch64 (Apple Silicon) | `casetrack-darwin-arm64` | Supported |
| Windows | x86_64 | `casetrack-win32-x64.exe` | Supported |
| Linux | x86_64 | `casetrack-linux-x64` | Supported |
| Linux | aarch64 | `casetrack-linux-arm64` | Future |

### 5.2 Platform-Specific Notes

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

### 5.3 Claude Integration Compatibility

| Client | Transport | Config Location | Status |
|--------|-----------|-----------------|--------|
| Claude Code (CLI) | stdio | `~/.claude/settings.json` | Primary target |
| Claude Desktop (macOS) | stdio | `~/Library/Application Support/Claude/claude_desktop_config.json` | Supported |
| Claude Desktop (Windows) | stdio | `%APPDATA%\Claude\claude_desktop_config.json` | Supported |
| Claude Desktop (Linux) | stdio | `~/.config/Claude/claude_desktop_config.json` | Supported |

---

## 6. Graceful Degradation Strategy

| Tier | RAM | Models Loaded | Behavior |
|------|-----|---------------|----------|
| **Full** | 16GB+ | All 3 neural models + BM25 simultaneously | Zero load latency, parallel embedding |
| **Standard** | 8-16GB | E1 + BM25 always (~400MB); others lazy-loaded | ~200ms first-use penalty; models stay loaded until memory pressure |
| **Constrained** | <8GB | E1 + BM25 only (~400MB); others loaded one-at-a-time | Sequential embedding, higher search latency, startup warning |

**Detection**: On startup, check available RAM via `sysinfo` crate. Set tier automatically, log the decision. User override: `--memory-mode=full|standard|constrained`.

---

*CaseTrack PRD v4.0.0 -- Document 2 of 10*
