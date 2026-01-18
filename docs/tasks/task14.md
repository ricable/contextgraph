# Task 14: Connect CLI to MCP Server with Warm-Loaded Models

## Metadata
- **Task ID**: TASK-GAP-014
- **Phase**: 4 (Integration)
- **Priority**: CRITICAL
- **Complexity**: Medium
- **Dependencies**: MCP TCP transport (already implemented)
- **Status**: ✅ COMPLETED (2026-01-18)

## Problem Statement

**ARCHITECTURAL BUG**: CLI commands use `StubMultiArrayProvider` (zeroed embeddings) instead of connecting to the MCP server which has warm-loaded GPU models.

### Current Broken Architecture:
```
Claude Code starts MCP server (TCP port 3000)
    ↓
MCP server warm-loads 13 embedding models (ProductionMultiArrayProvider)
    ↓
Hook triggers → calls CLI command (separate process)
    ↓
CLI uses StubMultiArrayProvider (WRONG - zeroed embeddings!)
    ↓
Memory stored with FAKE embeddings
```

### Required Architecture:
```
Claude Code starts MCP server (TCP port 3000)
    ↓
MCP server warm-loads 13 embedding models (ProductionMultiArrayProvider)
    ↓
Hook triggers → calls CLI command
    ↓
CLI connects to MCP server via TCP
    ↓
CLI calls MCP tool (store_memory/inject_context)
    ↓
MCP server uses warm-loaded models
    ↓
Memory stored with REAL embeddings
```

## Root Cause

In `crates/context-graph-cli/src/commands/memory/capture.rs` lines 306-309 and 385-388:
```rust
// WRONG: Creates new stub provider instead of connecting to MCP
let stub_provider = StubMultiArrayProvider::new();
let embedder = Arc::new(MultiArrayEmbeddingAdapter::new(stub_provider));
```

## Solution: CLI as MCP Client

CLI commands must connect to the running MCP server and call MCP tools instead of using local stub embeddings.

### MCP Server Already Supports TCP (server.rs:476-568)
- Listens on `config.mcp.bind_address:config.mcp.tcp_port` (default: `127.0.0.1:3000`)
- Uses newline-delimited JSON (NDJSON) - same as stdio
- Warm-loaded models available via `LazyMultiArrayProvider`

### Required Changes

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-cli/src/commands/memory/capture.rs` | Replace StubMultiArrayProvider with MCP client |
| `crates/context-graph-cli/src/commands/memory/inject.rs` | Replace stub with MCP client |
| `crates/context-graph-cli/Cargo.toml` | Add tokio TCP client deps |
| `.claude/hooks/stop.sh` | Update to use MCP-connected CLI |

## Implementation Steps

### Step 1: Create MCP Client Module

Create `crates/context-graph-cli/src/mcp_client.rs`:

```rust
//! MCP Client for CLI commands.
//!
//! Connects to running MCP server via TCP to use warm-loaded embedding models.
//! ELIMINATES StubMultiArrayProvider - all embeddings go through MCP server.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tracing::{debug, error, info};

const DEFAULT_MCP_HOST: &str = "127.0.0.1";
const DEFAULT_MCP_PORT: u16 = 3000;
const CONNECTION_TIMEOUT_MS: u64 = 5000;

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: &'static str,
    params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

pub struct McpClient {
    host: String,
    port: u16,
}

impl McpClient {
    pub fn new() -> Self {
        let host = std::env::var("CONTEXT_GRAPH_MCP_HOST")
            .unwrap_or_else(|_| DEFAULT_MCP_HOST.to_string());
        let port = std::env::var("CONTEXT_GRAPH_MCP_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(DEFAULT_MCP_PORT);

        Self { host, port }
    }

    /// Call store_memory MCP tool with content.
    /// Uses warm-loaded models on MCP server.
    pub async fn store_memory(
        &self,
        content: &str,
        importance: f64,
        modality: &str,
        tags: Option<Vec<String>>,
    ) -> Result<serde_json::Value> {
        let params = json!({
            "name": "store_memory",
            "arguments": {
                "content": content,
                "importance": importance,
                "modality": modality,
                "tags": tags.unwrap_or_default()
            }
        });

        self.call_tool(params).await
    }

    /// Call inject_context MCP tool.
    pub async fn inject_context(
        &self,
        content: &str,
        rationale: &str,
        importance: f64,
    ) -> Result<serde_json::Value> {
        let params = json!({
            "name": "inject_context",
            "arguments": {
                "content": content,
                "rationale": rationale,
                "importance": importance
            }
        });

        self.call_tool(params).await
    }

    async fn call_tool(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let addr = format!("{}:{}", self.host, self.port);
        debug!("Connecting to MCP server at {}", addr);

        let stream = tokio::time::timeout(
            std::time::Duration::from_millis(CONNECTION_TIMEOUT_MS),
            TcpStream::connect(&addr),
        )
        .await
        .context("Connection timeout")?
        .context("Failed to connect to MCP server")?;

        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);

        // Send tools/call request
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params,
        };

        let request_json = serde_json::to_string(&request)?;
        debug!("Sending: {}", request_json);

        writer.write_all(request_json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;

        // Read response
        let mut response_line = String::new();
        reader.read_line(&mut response_line).await?;
        debug!("Received: {}", response_line.trim());

        let response: JsonRpcResponse = serde_json::from_str(&response_line)?;

        if let Some(error) = response.error {
            error!("MCP error {}: {}", error.code, error.message);
            anyhow::bail!("MCP error {}: {}", error.code, error.message);
        }

        response.result.context("No result in response")
    }
}
```

### Step 2: Update capture.rs to Use MCP Client

Replace `handle_capture_response` in `capture.rs`:

```rust
pub async fn handle_capture_response(args: CaptureResponseArgs) -> i32 {
    // Step 1: Resolve content
    let content = resolve_content(args.content, "RESPONSE_SUMMARY", None);
    let Some(content) = content else {
        debug!("No content to capture, returning success");
        return CliExitCode::Success as i32;
    };

    // Step 2: Connect to MCP server and call store_memory
    let client = McpClient::new();

    match client.store_memory(
        &content,
        0.5, // default importance
        "text",
        Some(vec!["ClaudeResponse".to_string(), args.response_type]),
    ).await {
        Ok(result) => {
            info!("Memory stored via MCP: {:?}", result);
            CliExitCode::Success as i32
        }
        Err(e) => {
            error!("Failed to store memory via MCP: {}", e);
            eprintln!("ERROR: MCP call failed: {}", e);
            CliExitCode::Warning as i32
        }
    }
}
```

### Step 3: Remove StubMultiArrayProvider Usage

Delete these lines from `capture.rs`:
```rust
// DELETE THIS:
use context_graph_core::embeddings::StubMultiArrayProvider;

// DELETE THIS:
let stub_provider = StubMultiArrayProvider::new();
let embedder = Arc::new(MultiArrayEmbeddingAdapter::new(stub_provider));
```

### Step 4: Update CLI Module Structure

In `crates/context-graph-cli/src/main.rs`, add:
```rust
mod mcp_client;
pub use mcp_client::McpClient;
```

### Step 5: Update Hook Scripts

Update `.claude/hooks/stop.sh` to ensure MCP server is running:
```bash
#!/bin/bash
# Check MCP server is reachable before calling CLI
if ! nc -z 127.0.0.1 3000 2>/dev/null; then
    echo '{"success":false,"error":"MCP server not running on port 3000","exit_code":1}' >&2
    exit 1
fi

# Proceed with CLI call (now uses MCP client internally)
# ... rest of script
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_GRAPH_MCP_HOST` | `127.0.0.1` | MCP server hostname |
| `CONTEXT_GRAPH_MCP_PORT` | `3000` | MCP server TCP port |

## Testing Requirements

### Test 1: MCP Server Running Check
```bash
# Start MCP server in TCP mode
./target/release/context-graph-mcp --transport tcp --port 3000 &

# Verify listening
nc -z 127.0.0.1 3000 && echo "MCP server ready"
```

### Test 2: CLI Connects to MCP
```bash
export RESPONSE_SUMMARY="Test response content"
echo '{"session_id":"test"}' | ./.claude/hooks/stop.sh
# Should return: {"success":true,"stored":true,"exit_code":0}
```

### Test 3: Verify Real Embeddings Stored
```bash
# Query the stored memory
./target/release/context-graph-mcp --transport tcp --port 3000 &
# Send search_graph request via netcat
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"test"}}}' | nc 127.0.0.1 3000
# Should return memories with non-zero embeddings
```

### Test 4: Verify No Stub Usage
```bash
# Search codebase for stub usage - should find NOTHING in CLI
grep -r "StubMultiArrayProvider" crates/context-graph-cli/
# Expected: no matches
```

## Full State Verification

### Source of Truth
- **Primary**: MCP server logs showing embedding computation
- **Secondary**: RocksDB stored embeddings are non-zero
- **Tertiary**: CLI exit codes

### Verification Commands
```bash
# 1. Check embeddings are real (non-zero)
./target/release/context-graph-mcp --transport tcp &
sleep 5  # Wait for model warm-up

# 2. Store a memory via CLI
export RESPONSE_SUMMARY="Verification test content"
echo '{"session_id":"verify-001"}' | ./.claude/hooks/stop.sh

# 3. Search and verify embeddings exist
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"verification"}}}' | nc 127.0.0.1 3000

# 4. Check MCP server logs show embedding computation
# Should see: "Generated embeddings for content" NOT "Using stub embeddings"
```

## Definition of Done

- [x] `StubMultiArrayProvider` removed from CLI entirely
- [x] CLI connects to MCP server via TCP for all embedding operations
- [x] `McpClient` module created with `store_memory`, `inject_context`, `search_graph`, and `get_memetic_status` methods
- [x] Hook scripts verify MCP server is running before CLI calls
- [x] All tests pass (59 Task-14 specific tests passing)
- [x] `grep -r "StubMultiArrayProvider" crates/context-graph-cli/` returns no matches
- [x] Manual verification shows MCP server communication working

## Implementation Summary (Completed 2026-01-18)

### Files Created
| File | Description |
|------|-------------|
| `crates/context-graph-cli/src/mcp_client.rs` | TCP client for MCP server with JSON-RPC 2.0 protocol |
| `crates/context-graph-cli/src/mcp_helpers.rs` | Shared utilities: session ID resolution, error mapping, server checks |

### Files Modified
| File | Change |
|------|--------|
| `crates/context-graph-cli/src/main.rs` | Added `pub mod mcp_client` and `pub mod mcp_helpers` |
| `crates/context-graph-cli/src/commands/memory/capture.rs` | Rewrote to use McpClient.store_memory() |
| `crates/context-graph-cli/src/commands/memory/inject.rs` | Rewrote to use McpClient.search_graph() |

### Key Implementation Details

**MCP Client (mcp_client.rs)**
- Default port: 3100 (via `CONTEXT_GRAPH_MCP_HOST` and `CONTEXT_GRAPH_MCP_PORT` env vars)
- Connection timeout: 5000ms
- Request timeout: 30000ms
- Methods: `store_memory()`, `inject_context()`, `search_graph()`, `get_memetic_status()`, `is_server_running()`

**MCP Helpers (mcp_helpers.rs)**
- `resolve_session_id()`: CLI arg > CLAUDE_SESSION_ID env > "default"
- `mcp_error_to_exit_code()`: Maps McpClientError to exit codes (1=warning, 2=corruption)
- `require_mcp_server()`: Check server running, print error and return exit code if not

### Test Results
```
cargo test -p context-graph-cli -- mcp capture inject
19 MCP tests: PASSED
21 capture tests: PASSED
19 inject tests: PASSED
E2E memory capture test: PASSED
Integration topic state test: PASSED
```

### Manual Verification
1. MCP server not running → Exit 1 with "MCP server not running on 127.0.0.1:3100"
2. Empty content → Exit 0 (silent success, no MCP call)
3. CLI-to-MCP TCP communication → Working, stores memories with real embeddings

## Error Handling

| Scenario | Behavior |
|----------|----------|
| MCP server not running | Exit 1, error: "MCP server not running on port 3000" |
| Connection timeout (5s) | Exit 1, error: "Connection timeout" |
| MCP tool error | Exit 1, propagate MCP error message |
| Empty content | Exit 0, skip (no MCP call) |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code Session                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌─────────────────────────────────────┐  │
│  │  MCP Server  │     │         Hook Scripts                 │  │
│  │  (TCP:3000)  │◄────│  stop.sh, post_tool_use.sh, etc.    │  │
│  │              │     │              │                       │  │
│  │ ┌──────────┐ │     │              ▼                       │  │
│  │ │  Warm    │ │     │  ┌────────────────────────────┐     │  │
│  │ │ Loaded   │ │     │  │   context-graph-cli        │     │  │
│  │ │ Models   │ │     │  │                            │     │  │
│  │ │ (13 GPU) │ │     │  │  ┌──────────────────────┐ │     │  │
│  │ └──────────┘ │     │  │  │     McpClient        │ │     │  │
│  │              │◄────────│  │  (TCP connection)   │ │     │  │
│  │ ┌──────────┐ │     │  │  └──────────────────────┘ │     │  │
│  │ │ RocksDB  │ │     │  └────────────────────────────┘     │  │
│  │ │ Storage  │ │     └─────────────────────────────────────┘  │
│  │ └──────────┘ │                                               │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## References

- MCP TCP Transport: `crates/context-graph-mcp/src/server.rs:476-568`
- Warm Loading: `crates/context-graph-mcp/src/server.rs:174-202`
- LazyMultiArrayProvider: `crates/context-graph-mcp/src/adapters/lazy_provider.rs`
- Current Stub Usage (TO DELETE): `crates/context-graph-cli/src/commands/memory/capture.rs:306-309`
