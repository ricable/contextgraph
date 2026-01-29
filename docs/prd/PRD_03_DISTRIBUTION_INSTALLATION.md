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
   curl -fsSL https://casetrack.legal/install.sh | sh

   # Windows - PowerShell:
   irm https://casetrack.legal/install.ps1 | iex

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

The install script must:

```bash
#!/bin/sh
set -e

# 1. Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')   # darwin, linux
ARCH=$(uname -m)                                # x86_64, arm64, aarch64

# 2. Map to binary name
case "${OS}-${ARCH}" in
  darwin-arm64)   BINARY="casetrack-darwin-arm64" ;;
  darwin-x86_64)  BINARY="casetrack-darwin-x64" ;;
  linux-x86_64)   BINARY="casetrack-linux-x64" ;;
  linux-aarch64)  BINARY="casetrack-linux-arm64" ;;
  *) echo "Unsupported platform: ${OS}-${ARCH}"; exit 1 ;;
esac

# 3. Download binary
VERSION="latest"
URL="https://github.com/casetrack-legal/casetrack/releases/${VERSION}/download/${BINARY}"
INSTALL_DIR="${HOME}/.local/bin"

mkdir -p "${INSTALL_DIR}"
curl -fsSL "${URL}" -o "${INSTALL_DIR}/casetrack"
chmod +x "${INSTALL_DIR}/casetrack"

# 4. Add to PATH if needed
if ! echo "${PATH}" | grep -q "${INSTALL_DIR}"; then
  SHELL_RC=""
  if [ -f "${HOME}/.zshrc" ]; then SHELL_RC="${HOME}/.zshrc"
  elif [ -f "${HOME}/.bashrc" ]; then SHELL_RC="${HOME}/.bashrc"
  fi
  if [ -n "${SHELL_RC}" ]; then
    echo "export PATH=\"${INSTALL_DIR}:\${PATH}\"" >> "${SHELL_RC}"
  fi
fi

# 5. Configure Claude Code (if installed)
CLAUDE_SETTINGS="${HOME}/.claude/settings.json"
if [ -d "${HOME}/.claude" ]; then
  # Merge MCP server config into existing settings
  casetrack --setup-claude-code
fi

# 6. Print success
echo ""
echo "CaseTrack installed successfully!"
echo ""
echo "  Binary: ${INSTALL_DIR}/casetrack"
echo "  Data:   ~/Documents/CaseTrack/"
echo ""
echo "Next steps:"
echo "  1. Restart your terminal (or run: source ${SHELL_RC})"
echo "  2. Open Claude Code and ask: 'Create a new case called Test'"
echo ""
```

### 2.2 Windows Install Script (`install.ps1`)

```powershell
# Detect architecture
$arch = if ([Environment]::Is64BitOperatingSystem) { "x64" } else {
    Write-Error "CaseTrack requires 64-bit Windows"; exit 1
}

# Download
$version = "latest"
$url = "https://github.com/casetrack-legal/casetrack/releases/$version/download/casetrack-win32-$arch.exe"
$installDir = "$env:LOCALAPPDATA\CaseTrack"

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Invoke-WebRequest -Uri $url -OutFile "$installDir\casetrack.exe"

# Add to PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$installDir;$currentPath", "User")
}

# Configure Claude Code
& "$installDir\casetrack.exe" --setup-claude-code

Write-Host ""
Write-Host "CaseTrack installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Binary: $installDir\casetrack.exe"
Write-Host "  Data:   $env:USERPROFILE\Documents\CaseTrack\"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Restart your terminal"
Write-Host "  2. Open Claude Code and ask: 'Create a new case called Test'"
```

---

## 3. Self-Setup CLI Command

The binary itself handles Claude Code configuration:

```
casetrack --setup-claude-code
```

This command:

1. Detects Claude Code settings file location:
   - `~/.claude/settings.json` (all platforms)
2. Reads existing settings (or creates empty `{}`)
3. Merges `mcpServers.casetrack` entry
4. Writes back with proper JSON formatting
5. Prints confirmation

```rust
/// CLI setup command for Claude Code integration
pub fn setup_claude_code(data_dir: &Path) -> Result<()> {
    let settings_path = dirs::home_dir()
        .ok_or(CaseTrackError::NoHomeDir)?
        .join(".claude")
        .join("settings.json");

    // Read existing settings or create new
    let mut settings: serde_json::Value = if settings_path.exists() {
        let content = fs::read_to_string(&settings_path)?;
        serde_json::from_str(&content)?
    } else {
        fs::create_dir_all(settings_path.parent().unwrap())?;
        json!({})
    };

    // Add MCP server config
    let mcp_servers = settings
        .as_object_mut()
        .unwrap()
        .entry("mcpServers")
        .or_insert(json!({}));

    mcp_servers["casetrack"] = json!({
        "command": std::env::current_exe()?.to_string_lossy(),
        "args": ["--data-dir", data_dir.to_string_lossy()]
    });

    // Write back
    let formatted = serde_json::to_string_pretty(&settings)?;
    fs::write(&settings_path, formatted)?;

    println!("Claude Code configured. CaseTrack will be available in your next session.");
    Ok(())
}
```

---

## 4. MCPB Bundle Structure

The `.mcpb` file is a ZIP archive for Claude Desktop GUI installation:

```
casetrack.mcpb (ZIP archive, ~50MB)
|-- manifest.json           # MCP configuration
|-- icon.png               # Extension icon (256x256)
|-- server/
|   |-- casetrack-darwin-x64      # macOS Intel
|   |-- casetrack-darwin-arm64    # macOS Apple Silicon
|   |-- casetrack-win32-x64.exe   # Windows
|   +-- casetrack-linux-x64       # Linux
+-- resources/
    |-- tokenizer.json     # Shared tokenizer (~5MB)
    +-- legal-vocab.txt    # Legal term expansions (~2MB)
```

### 4.1 Manifest Specification

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
      "description": "Where to store cases and models on your computer",
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

---

## 5. Installation Flow (Claude Desktop)

```
INSTALLATION FLOW
=================================================================================

Step 1: Download (User Action)
---------------------------------
User visits casetrack.legal
Clicks "Download for Claude Desktop"
Browser downloads casetrack.mcpb (~50MB)

Step 2: Install (User Action)
---------------------------------
User double-clicks casetrack.mcpb
  OR drags to Claude Desktop window
  OR Settings -> Extensions -> Install from file

Step 3: Configure (Dialog)
---------------------------------
+-------------------------------------------------------+
| Install CaseTrack?                                     |
|                                                        |
| CaseTrack lets you search legal documents with AI.     |
| All processing happens on your computer.               |
|                                                        |
| +----------------------------------------------------+ |
| | Data Location                                      | |
| | [~/Documents/CaseTrack                        ] [F]| |
| +----------------------------------------------------+ |
|                                                        |
| +----------------------------------------------------+ |
| | License Key (optional - leave blank for free tier)  | |
| | [                                             ] [L] | |
| +----------------------------------------------------+ |
|                                                        |
| This extension will:                                   |
| [Y] Read and write files in your Data Location        |
| [Y] Download AI models from huggingface.co (~400MB)   |
| [N] NOT send your documents anywhere                  |
|                                                        |
|                         [Cancel]  [Install Extension]  |
+-------------------------------------------------------+

Step 4: First Run (Automatic)
---------------------------------
Claude Desktop starts CaseTrack server
Server detects missing models
Downloads models in background (~400MB)
Shows progress notification in Claude Desktop

Step 5: Ready (Automatic)
---------------------------------
CaseTrack icon appears in Extensions panel
User can now use CaseTrack tools in conversation
```

---

## 6. First-Run Experience

On first launch, the server must handle model bootstrapping gracefully:

```rust
/// First-run initialization sequence
pub async fn initialize(config: &Config) -> Result<ServerState> {
    let data_dir = &config.data_dir;

    // Step 1: Create directory structure
    create_directory_structure(data_dir)?;

    // Step 2: Check and download models
    let model_manager = ModelManager::new(&data_dir.join("models"));
    let missing = model_manager.check_missing_models(config.tier)?;

    if !missing.is_empty() {
        // Report progress via MCP notification (if supported)
        // or via stderr logging
        tracing::info!(
            "First run detected. Downloading {} models ({} MB)...",
            missing.len(),
            missing.iter().map(|m| m.size_mb).sum::<u32>()
        );

        for model in &missing {
            tracing::info!("Downloading {}...", model.id);
            model_manager.download(model).await?;
            tracing::info!("Downloaded {} ({} MB)", model.id, model.size_mb);
        }

        tracing::info!("All models ready.");
    }

    // Step 3: Open or create registry database
    let registry = CaseRegistry::open(&data_dir.join("registry.db"))?;

    // Step 4: Validate license (offline-first)
    let tier = LicenseManager::new(data_dir)
        .validate(config.license_key.as_deref());

    tracing::info!("CaseTrack ready. Tier: {:?}, Cases: {}", tier, registry.count()?);

    Ok(ServerState { registry, model_manager, tier })
}

fn create_directory_structure(data_dir: &Path) -> Result<()> {
    fs::create_dir_all(data_dir.join("models"))?;
    fs::create_dir_all(data_dir.join("cases"))?;
    // registry.db is created by RocksDB::open
    Ok(())
}
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
        id: "e1-legal",
        repo: "BAAI/bge-small-en-v1.5",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 65,
        required: true,
    },
    ModelSpec {
        id: "e6-legal",
        repo: "naver/splade-cocondenser-selfdistil",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 55,
        required: true,
    },
    ModelSpec {
        id: "e7",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 45,
        required: true,
    },
    ModelSpec {
        id: "e8-legal",
        repo: "casetrack/citation-minilm-v1",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 35,
        required: false,
    },
    ModelSpec {
        id: "e11-legal",
        repo: "nlpaueb/legal-bert-small-uncased",
        files: &["model.onnx", "tokenizer.json"],
        size_mb: 60,
        required: false,
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

```rust
/// Download with retry and resume support
pub async fn download_model(&self, spec: &ModelSpec) -> Result<()> {
    let model_dir = self.cache_dir.join(spec.id);
    fs::create_dir_all(&model_dir)?;

    for file in spec.files {
        let dest = model_dir.join(file);

        // Skip if already downloaded and valid
        if dest.exists() && self.verify_checksum(&dest, spec, file)? {
            continue;
        }

        // Download with retry (3 attempts, exponential backoff)
        let mut attempts = 0;
        loop {
            attempts += 1;
            match self.download_file(spec.repo, file, &dest).await {
                Ok(()) => break,
                Err(e) if attempts < 3 => {
                    tracing::warn!(
                        "Download attempt {}/3 failed for {}/{}: {}. Retrying...",
                        attempts, spec.id, file, e
                    );
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempts))).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    Ok(())
}
```

---

## 7. Update Mechanism

### 7.1 Version Checking

CaseTrack checks for updates on startup (non-blocking, does not delay server start):

```rust
/// Non-blocking update check
pub async fn check_for_updates(current_version: &str) {
    // Fire and forget -- do not block server startup
    tokio::spawn(async move {
        let check = async {
            let url = "https://api.github.com/repos/casetrack-legal/casetrack/releases/latest";
            let resp: GithubRelease = reqwest::get(url).await?.json().await?;
            let latest = resp.tag_name.trim_start_matches('v');

            if semver::Version::parse(latest)? > semver::Version::parse(current_version)? {
                tracing::info!(
                    "Update available: v{} -> v{}. \
                     Run 'casetrack --update' or download from https://casetrack.legal",
                    current_version, latest
                );
            }
            Ok::<(), anyhow::Error>(())
        };

        if let Err(e) = check.await {
            // Silently ignore update check failures -- user is offline or rate limited
            tracing::debug!("Update check failed (non-critical): {}", e);
        }
    });
}
```

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
1. Asks for confirmation ("This will remove CaseTrack. Your case data will NOT be deleted.")
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

# Optionally remove data (YOUR CHOICE -- this deletes all cases):
rm -rf ~/Documents/CaseTrack/
```

---

*CaseTrack PRD v4.0.0 -- Document 3 of 10*
