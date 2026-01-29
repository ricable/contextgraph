# PRD 03: Distribution & Installation

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Distribution Channels

```
DISTRIBUTION CHANNELS (Priority Order)
=================================================================================

1. CLAUDE CODE (Primary - Recommended)
   ────────────────────────────────
   # macOS/Linux - One command:
   curl -fsSL https://casetrack.dev/install.sh | sh

   # Windows - PowerShell:
   irm https://casetrack.dev/install.ps1 | iex

   # Or via cargo:
   cargo binstall casetrack   # Pre-compiled binary (fast)
   cargo install casetrack    # From source (slow, needs Rust toolchain)

   # Then add to Claude Code settings (~/.claude/settings.json):
   {
     "mcpServers": {
       "casetrack": {
         "command": "casetrack",
         "args": ["--data-dir", "~/Documents/CaseTrack"]
       }
     }
   }

2. CLAUDE DESKTOP (Secondary)
   ────────────────────────────────
   - Download casetrack.mcpb from website
   - Double-click or drag to Claude Desktop window
   - Click "Install" in dialog
   - Done (same binary, just packaged for GUI install)

3. PACKAGE MANAGERS (Tertiary)
   ────────────────────────────────
   macOS:     brew install casetrack
   Windows:   winget install CaseTrack.CaseTrack
   Linux:     cargo binstall casetrack

4. GITHUB RELEASES (Developer)
   ────────────────────────────────
   - Pre-built binaries attached to GitHub releases
   - SHA256 checksums for verification
   - Source tarball for audit
```

---

## 2. Install Script Specification

### 2.1 macOS/Linux Install Script (`install.sh`)

```
Steps:
1. Detect platform (darwin/linux) and architecture (x86_64/arm64/aarch64)
2. Map to binary name: casetrack-{os}-{arch}
   Supported: darwin-arm64, darwin-x64, linux-x64, linux-arm64
3. Download binary from GitHub releases to ~/.local/bin/casetrack
4. Add ~/.local/bin to PATH via .zshrc or .bashrc (if not already present)
5. If ~/.claude/ exists, run: casetrack --setup-claude-code
6. Print success with next steps (restart terminal, try a command)
```

### 2.2 Windows Install Script (`install.ps1`)

```
Steps:
1. Require 64-bit Windows
2. Download binary from GitHub releases to %LOCALAPPDATA%\CaseTrack\casetrack.exe
3. Add install dir to user PATH
4. Run: casetrack.exe --setup-claude-code
5. Print success with next steps
```

---

## 3. Self-Setup CLI Command

```
casetrack --setup-claude-code
```

Reads or creates `~/.claude/settings.json`, merges `mcpServers.casetrack` entry (using the current binary path and configured data-dir), writes back with pretty JSON formatting.

---

## 4. MCPB Bundle Structure

`.mcpb` = ZIP archive (~50MB) for Claude Desktop GUI install. Contains platform binaries, manifest, icon, and shared resources (tokenizer, vocabulary).

### 4.1 Manifest Specification

```json
{
  "manifest_version": "1.0",
  "name": "casetrack",
  "version": "1.0.0",
  "display_name": "CaseTrack Document Intelligence",
  "description": "Ingest PDFs, Word docs, Excel spreadsheets, and scans. Search with AI. Every answer cites the source.",

  "author": {
    "name": "CaseTrack",
    "url": "https://casetrack.dev"
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
      "description": "Where to store collections and models on your computer",
      "type": "directory",
      "default": "${DOCUMENTS}/CaseTrack",
      "required": true
    },
    {
      "id": "license_key",
      "name": "License Key (Optional)",
      "description": "Leave blank for free tier. Purchase at casetrack.dev",
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

---

## 5. Installation Flow (Claude Desktop)

```
1. Download: User downloads casetrack.mcpb (~50MB) from casetrack.dev
2. Install:  Double-click .mcpb, drag to Claude Desktop, or Settings > Extensions > Install
3. Configure: Dialog prompts for data location and optional license key
   +-------------------------------------------------------+
   | Install CaseTrack?                                     |
   |                                                        |
   | CaseTrack lets you search documents with AI.           |
   | All processing happens on your computer.               |
   |                                                        |
   | Data Location:  [~/Documents/CaseTrack            ] [F]|
   | License Key:    [optional - blank for free tier   ] [L]|
   |                                                        |
   | [Y] Read and write files in your Data Location        |
   | [Y] Download AI models from huggingface.co (~400MB)   |
   | [N] NOT send your documents anywhere                  |
   |                                                        |
   |                         [Cancel]  [Install Extension]  |
   +-------------------------------------------------------+
4. First Run: Server starts, downloads missing models (~400MB) in background
5. Ready:     CaseTrack icon appears in Extensions panel
```

---

## 6. First-Run Experience

Initialization sequence on first launch:

```
1. Create directory structure: models/, collections/ (registry.db created by RocksDB::open)
2. Check for missing models based on tier; download any missing (with progress logging)
3. Open or create registry database
4. Validate license (offline-first)
5. Log ready state: tier + collection count
```

### 6.1 Model Download Strategy

Models are NOT bundled (would make `.mcpb` too large). Downloaded on first use:

```rust
pub struct ModelSpec {
    pub id: &'static str,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub size_mb: u32,
    pub required: bool,  // false = only download for Pro tier
}

pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "e1",
        repo: "BAAI/bge-small-en-v1.5",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 65,
        required: true,
    },
    ModelSpec {
        id: "e6",
        repo: "naver/splade-cocondenser-selfdistil",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 55,
        required: true,
    },
    ModelSpec {
        id: "e12",
        repo: "colbert-ir/colbertv2.0-msmarco-passage",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 110,
        required: false,
    },
];
// E13 (BM25) requires no model download -- pure algorithm
```

### 6.2 Download Resilience

- Skip files already downloaded with valid checksums
- Retry up to 3 attempts with exponential backoff (2s, 4s, 8s)
- Fatal error after 3 failures for any single file

---

## 7. Update Mechanism

### 7.1 Version Checking

Non-blocking check on startup (fire-and-forget via `tokio::spawn`):
- Queries GitHub releases API for latest version
- Compares via semver; logs update notice if newer version exists
- Silently ignores failures (offline, rate-limited)

### 7.2 Self-Update Command

```
casetrack --update
```

Downloads the latest binary and replaces itself:

1. Download new binary to temporary path
2. Verify SHA256 checksum
3. Replace current binary (platform-specific swap)
4. Print success message

On Windows, use the "rename on restart" pattern since running binaries can't be replaced directly.

### 7.3 Data Migration

When a new version introduces schema changes:

1. On startup, check `schema_version` in `registry.db`
2. If schema is older, run migration functions sequentially
3. Migrations are idempotent (safe to re-run)
4. Back up existing DB before migration (copy `registry.db` to `registry.db.bak.{version}`)

See [PRD 04: Storage Architecture](PRD_04_STORAGE_ARCHITECTURE.md) for schema versioning details.

---

## 8. Uninstallation

### 8.1 CLI Uninstall

```
casetrack --uninstall
```

This command:
1. Asks for confirmation ("This will remove CaseTrack. Your data will NOT be deleted.")
2. Removes the binary from PATH
3. Removes the Claude Code/Desktop configuration entry
4. Prints location of data directory for manual cleanup
5. Does NOT delete `~/Documents/CaseTrack/` (user's data is sacred)

### 8.2 Manual Uninstall

```
# Remove binary
rm ~/.local/bin/casetrack   # macOS/Linux
# OR delete %LOCALAPPDATA%\CaseTrack\ on Windows

# Remove Claude Code config (edit ~/.claude/settings.json, remove "casetrack" key)

# Optionally remove data (YOUR CHOICE -- this deletes all collections):
rm -rf ~/Documents/CaseTrack/
```

---

*CaseTrack PRD v4.0.0 -- Document 3 of 10*
