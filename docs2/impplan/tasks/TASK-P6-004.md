# Task: TASK-P6-004 - Inject Brief Command (CLI Integration)

```xml
<task_spec id="TASK-P6-004" version="2.0">
<metadata>
  <title>Inject Brief CLI Command</title>
  <phase>6</phase>
  <sequence>46</sequence>
  <layer>surface</layer>
  <estimated_loc>120</estimated_loc>
  <last_audit>2026-01-17</last_audit>
  <dependencies>
    <dependency task="TASK-P6-001" status="COMPLETE">CLI infrastructure (main.rs, CliExitCode)</dependency>
    <dependency task="TASK-P6-003" status="COMPLETE">InjectContext command (pattern to follow)</dependency>
    <dependency task="TASK-P5-007" status="COMPLETE">InjectionPipeline.generate_brief_context() - IMPLEMENTED</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">InjectBriefArgs in inject.rs</artifact>
    <artifact type="function">handle_inject_brief() in inject.rs</artifact>
    <artifact type="enum_variant">InjectBrief in MemoryCommands</artifact>
  </produces>
</metadata>
```

## Executive Summary

**Core library is COMPLETE. Only CLI wiring remains.**

The `generate_brief_context()` method and all supporting infrastructure exist in `context-graph-core`. This task is ONLY about adding the CLI command (`inject-brief`) that calls the existing core functionality.

---

## Current State Audit (Verified 2026-01-17)

### IMPLEMENTED (DO NOT RECREATE)

| Component | Location | Status |
|-----------|----------|--------|
| `InjectionPipeline::generate_brief_context()` | `crates/context-graph-core/src/injection/pipeline.rs:296-349` | COMPLETE |
| `ContextFormatter::format_brief_context()` | `crates/context-graph-core/src/injection/formatter.rs:164-190` | COMPLETE |
| `BRIEF_BUDGET` constant | `crates/context-graph-core/src/injection/budget.rs:47` | = 200 tokens |
| `BRIEF_MAX_TOKENS` constant | `crates/context-graph-core/src/injection/formatter.rs:23` | = 200 tokens |
| `TokenBudget::with_total()` | `crates/context-graph-core/src/injection/budget.rs:85-113` | COMPLETE |
| `InjectionResult` struct | `crates/context-graph-core/src/injection/result.rs` | COMPLETE |
| All FSV tests for brief | `formatter.rs:410-457`, `pipeline.rs` | COMPLETE |

### MISSING (IMPLEMENT THESE)

| Component | Location | What to Do |
|-----------|----------|------------|
| `InjectBriefArgs` struct | `crates/context-graph-cli/src/commands/memory/inject.rs` | Create new struct |
| `handle_inject_brief()` | `crates/context-graph-cli/src/commands/memory/inject.rs` | Create handler function |
| `InjectBrief` enum variant | `crates/context-graph-cli/src/commands/memory/mod.rs:25` | Add to MemoryCommands |
| Match arm in handler | `crates/context-graph-cli/src/commands/memory/mod.rs:54` | Route to handler |

---

## File Paths (EXACT - Verified)

```
# CLI Layer - MODIFY THESE
crates/context-graph-cli/src/commands/memory/mod.rs      # Add InjectBrief variant
crates/context-graph-cli/src/commands/memory/inject.rs   # Add InjectBriefArgs + handle_inject_brief

# Core Layer - READ ONLY (already complete)
crates/context-graph-core/src/injection/pipeline.rs      # generate_brief_context() at line 296
crates/context-graph-core/src/injection/budget.rs        # BRIEF_BUDGET = 200 at line 47
crates/context-graph-core/src/injection/formatter.rs     # format_brief_context() at line 164
crates/context-graph-core/src/injection/mod.rs           # Exports BRIEF_BUDGET
```

---

## Implementation Specification

### 1. Add `InjectBriefArgs` Struct

**File**: `crates/context-graph-cli/src/commands/memory/inject.rs`
**Insert after**: `InjectContextArgs` struct (around line 53)

```rust
/// Arguments for inject-brief command.
///
/// Called by PreToolUse hook for quick context injection.
/// Reads TOOL_DESCRIPTION or TOOL_NAME from environment for query.
#[derive(Args)]
pub struct InjectBriefArgs {
    /// Query text (or use TOOL_DESCRIPTION/TOOL_NAME env var)
    pub query: Option<String>,

    /// Session ID (or use CLAUDE_SESSION_ID env var)
    #[arg(long)]
    pub session_id: Option<String>,

    /// Path to models directory
    #[arg(long, env = "CONTEXT_GRAPH_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// Path to data directory
    #[arg(long, env = "CONTEXT_GRAPH_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
}
```

**NOTE**: No `--budget` argument. Brief ALWAYS uses 200 tokens (BRIEF_BUDGET constant).

### 2. Add `handle_inject_brief()` Function

**File**: `crates/context-graph-cli/src/commands/memory/inject.rs`
**Insert after**: `handle_inject_context()` function

```rust
/// Handle inject-brief command.
///
/// Generates brief context for PreToolUse hook (<200 tokens).
/// Reads query from TOOL_DESCRIPTION, then TOOL_NAME, then CLI arg.
///
/// # Exit Codes
/// - 0: Success (including empty result)
/// - 1: Error (pipeline/storage failure)
/// - 2: Corruption (missing memory, stale index)
pub async fn handle_inject_brief(args: InjectBriefArgs) -> i32 {
    // Priority: CLI arg > TOOL_DESCRIPTION env > TOOL_NAME env
    let query = args
        .query
        .or_else(|| std::env::var("TOOL_DESCRIPTION").ok())
        .or_else(|| std::env::var("TOOL_NAME").ok())
        .filter(|q| !q.trim().is_empty());

    let Some(query) = query else {
        debug!("No query provided, returning empty brief context");
        return CliExitCode::Success as i32;
    };

    // Get session ID from arg or CLAUDE_SESSION_ID env var
    let session_id = args
        .session_id
        .or_else(|| std::env::var("CLAUDE_SESSION_ID").ok())
        .unwrap_or_else(|| {
            warn!("No session ID available, using 'default'");
            "default".to_string()
        });

    // Get paths from args or defaults
    let models_dir = args.models_dir.unwrap_or_else(|| PathBuf::from("./models"));
    let data_dir = args.data_dir.unwrap_or_else(|| PathBuf::from("./data"));

    info!(
        query_len = query.len(),
        session_id = %session_id,
        "Generating brief context"
    );

    // Initialize embedding provider
    let provider = match ProductionMultiArrayProvider::new(models_dir.clone(), GpuConfig::default()).await {
        Ok(p) => p,
        Err(e) => {
            error!(error = %e, models_dir = ?models_dir, "Failed to initialize embedding provider");
            eprintln!("ERROR: Failed to initialize embeddings: {}", e);
            return CliExitCode::Warning as i32;
        }
    };

    // Embed the query (all 13 embedders)
    let embedding_output = match provider.embed_all(&query).await {
        Ok(output) => output,
        Err(e) => {
            error!(error = %e, query_len = query.len(), "Failed to embed query");
            eprintln!("ERROR: Failed to embed query: {}", e);
            return CliExitCode::Warning as i32;
        }
    };

    let query_fingerprint = embedding_output.fingerprint;

    // Initialize storage
    let memories_path = data_dir.join("memories");
    let store = match MemoryStore::new(&memories_path) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            error!(error = %e, path = ?memories_path, "Failed to open memory store");
            eprintln!("ERROR: Failed to open memory store: {}", e);
            return CliExitCode::Warning as i32;
        }
    };

    // Create retriever and pipeline with brief budget (200 tokens)
    let retriever = SimilarityRetriever::with_defaults(store.clone());
    let pipeline = InjectionPipeline::new(retriever, store);

    // Generate BRIEF context (uses BRIEF_BUDGET internally)
    let result = match pipeline.generate_brief_context(&query_fingerprint, &session_id) {
        Ok(r) => r,
        Err(InjectionError::MemoryNotFound(id)) => {
            error!(memory_id = %id, "Memory not found (stale index?)");
            eprintln!("CORRUPTION: Memory {} not found - index may be stale", id);
            return CliExitCode::Blocking as i32;
        }
        Err(e) => {
            error!(error = %e, "Brief pipeline failed");
            eprintln!("ERROR: Brief context generation failed: {}", e);
            return CliExitCode::Warning as i32;
        }
    };

    // Output result to stdout
    if result.is_empty() {
        debug!("No relevant brief context found");
        // Empty stdout = no injection (exit 0)
    } else {
        info!(
            memories = result.memory_count(),
            tokens = result.tokens_used,
            "Brief context generated"
        );
        // Print brief context to stdout for hook capture
        print!("{}", result.formatted_context);
    }

    CliExitCode::Success as i32
}
```

### 3. Add Enum Variant and Handler

**File**: `crates/context-graph-cli/src/commands/memory/mod.rs`

Add variant to `MemoryCommands` enum (around line 45):

```rust
#[derive(Subcommand)]
pub enum MemoryCommands {
    /// Inject relevant context from memory store
    InjectContext(inject::InjectContextArgs),

    /// Inject brief context for PreToolUse hook
    ///
    /// Generates compact context (<200 tokens) for tool execution.
    /// Uses TOOL_DESCRIPTION or TOOL_NAME environment variable as query.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # With environment variable (typical hook usage)
    /// TOOL_DESCRIPTION="Writing file" context-graph-cli memory inject-brief
    ///
    /// # With explicit query
    /// context-graph-cli memory inject-brief --query "Editing code"
    /// ```
    InjectBrief(inject::InjectBriefArgs),
}
```

Add match arm in `handle_memory_command()` (around line 54):

```rust
pub async fn handle_memory_command(cmd: MemoryCommands) -> i32 {
    match cmd {
        MemoryCommands::InjectContext(args) => inject::handle_inject_context(args).await,
        MemoryCommands::InjectBrief(args) => inject::handle_inject_brief(args).await,
    }
}
```

---

## Required Imports

Add to `inject.rs` if not already present:

```rust
use context_graph_core::injection::BRIEF_BUDGET;  // For documentation/reference
```

Note: The actual BRIEF_BUDGET enforcement happens in `InjectionPipeline::generate_brief_context()` which internally calls `TokenBudget::with_total(BRIEF_BUDGET)`.

---

## Environment Variable Precedence

For `inject-brief` command:

| Priority | Source | Variable |
|----------|--------|----------|
| 1 (highest) | CLI argument | `--query "text"` |
| 2 | Environment | `TOOL_DESCRIPTION` |
| 3 | Environment | `TOOL_NAME` |
| 4 | Fallback | Empty (returns exit 0) |

This differs from `inject-context` which uses `USER_PROMPT`. The brief command is specifically for PreToolUse hook which provides tool info.

---

## Tests to Add

**File**: `crates/context-graph-cli/src/commands/memory/inject.rs`
**Insert in tests module** (after existing tests):

```rust
// =========================================================================
// inject-brief Tests
// =========================================================================

#[test]
fn test_brief_tool_description_env() {
    let _lock = GLOBAL_IDENTITY_LOCK.lock();

    std::env::set_var("TOOL_DESCRIPTION", "Running cargo test");
    std::env::remove_var("TOOL_NAME");

    let args = InjectBriefArgs {
        query: None,
        session_id: None,
        models_dir: None,
        data_dir: None,
    };

    let query = args.query
        .clone()
        .or_else(|| std::env::var("TOOL_DESCRIPTION").ok())
        .or_else(|| std::env::var("TOOL_NAME").ok())
        .filter(|q| !q.trim().is_empty());

    assert_eq!(query, Some("Running cargo test".to_string()));
    std::env::remove_var("TOOL_DESCRIPTION");
    println!("[PASS] TOOL_DESCRIPTION env var read");
}

#[test]
fn test_brief_tool_name_fallback() {
    let _lock = GLOBAL_IDENTITY_LOCK.lock();

    std::env::remove_var("TOOL_DESCRIPTION");
    std::env::set_var("TOOL_NAME", "Bash");

    let args = InjectBriefArgs {
        query: None,
        session_id: None,
        models_dir: None,
        data_dir: None,
    };

    let query = args.query
        .clone()
        .or_else(|| std::env::var("TOOL_DESCRIPTION").ok())
        .or_else(|| std::env::var("TOOL_NAME").ok())
        .filter(|q| !q.trim().is_empty());

    assert_eq!(query, Some("Bash".to_string()));
    std::env::remove_var("TOOL_NAME");
    println!("[PASS] TOOL_NAME fallback works");
}

#[test]
fn test_brief_arg_overrides_env() {
    let _lock = GLOBAL_IDENTITY_LOCK.lock();

    std::env::set_var("TOOL_DESCRIPTION", "env tool");

    let args = InjectBriefArgs {
        query: Some("arg query".to_string()),
        session_id: None,
        models_dir: None,
        data_dir: None,
    };

    let query = args.query
        .clone()
        .or_else(|| std::env::var("TOOL_DESCRIPTION").ok())
        .or_else(|| std::env::var("TOOL_NAME").ok())
        .filter(|q| !q.trim().is_empty());

    assert_eq!(query, Some("arg query".to_string()));
    std::env::remove_var("TOOL_DESCRIPTION");
    println!("[PASS] CLI arg overrides env for brief");
}

#[test]
fn test_brief_no_query_returns_none() {
    let _lock = GLOBAL_IDENTITY_LOCK.lock();

    std::env::remove_var("TOOL_DESCRIPTION");
    std::env::remove_var("TOOL_NAME");

    let args = InjectBriefArgs {
        query: None,
        session_id: None,
        models_dir: None,
        data_dir: None,
    };

    let query = args.query
        .clone()
        .or_else(|| std::env::var("TOOL_DESCRIPTION").ok())
        .or_else(|| std::env::var("TOOL_NAME").ok())
        .filter(|q| !q.trim().is_empty());

    assert!(query.is_none());
    println!("[PASS] No query/env returns None (exit 0)");
}

#[test]
fn test_brief_budget_constant() {
    use context_graph_core::injection::BRIEF_BUDGET;
    assert_eq!(BRIEF_BUDGET, 200, "BRIEF_BUDGET must be 200 per constitution");
    println!("[PASS] BRIEF_BUDGET = 200");
}
```

---

## Definition of Done

| ID | Criterion | Verification Method |
|----|-----------|---------------------|
| DOD-1 | `inject-brief` outputs brief format to stdout | Output starts with "Related:" or is empty |
| DOD-2 | Output fits within 200 token budget | Word count × 1.3 ≤ 200 |
| DOD-3 | TOOL_DESCRIPTION env var is read | `TOOL_DESCRIPTION=test ./context-graph-cli memory inject-brief` works |
| DOD-4 | TOOL_NAME fallback works | `TOOL_NAME=Bash ./context-graph-cli memory inject-brief` works |
| DOD-5 | Command completes in <400ms | `time` command shows fast execution |
| DOD-6 | Exit code 0 on success/empty | Verify with `echo $?` |
| DOD-7 | Exit code 1 on error | Simulate error, verify exit code |
| DOD-8 | All tests pass | `cargo test commands::inject --package context-graph-cli` |

---

## Performance Requirements

| Metric | Requirement | Rationale |
|--------|-------------|-----------|
| Total execution | <400ms | PreToolUse hook timeout is 500ms |
| Output tokens | ≤200 | BRIEF_BUDGET constant |
| Memory retrieval | <10ms | Quick lookup, 10 candidates max |

---

## Validation Commands

```bash
# Build
cargo build --package context-graph-cli

# Run tests
cargo test commands::inject --package context-graph-cli

# Manual test with query arg
./target/debug/context-graph-cli memory inject-brief --query "test"

# Manual test with TOOL_DESCRIPTION env
TOOL_DESCRIPTION="Writing file" ./target/debug/context-graph-cli memory inject-brief

# Manual test with TOOL_NAME fallback
TOOL_NAME="Bash" ./target/debug/context-graph-cli memory inject-brief

# Verify performance
time ./target/debug/context-graph-cli memory inject-brief --query "test"

# Verify exit code
./target/debug/context-graph-cli memory inject-brief --query "test"; echo "Exit: $?"
```

---

## Full State Verification (FSV) Protocol

After implementing, you MUST perform these verification steps:

### 1. Source of Truth Identification

| What | Source of Truth | How to Verify |
|------|-----------------|---------------|
| CLI command exists | Binary accepts `memory inject-brief` | `./target/debug/context-graph-cli memory inject-brief --help` |
| Output format | stdout | Capture and inspect output |
| Exit codes | Shell `$?` | `echo $?` after command |
| Token budget | BRIEF_BUDGET constant | Check tokens in output |

### 2. Execute & Inspect Checklist

After running the command:

```bash
# Capture output to file for inspection
TOOL_DESCRIPTION="Running cargo test" ./target/debug/context-graph-cli memory inject-brief > /tmp/brief_output.txt 2>&1

# Verify output format
cat /tmp/brief_output.txt  # Should be "Related: ..." or empty

# Count tokens (approximate)
wc -w /tmp/brief_output.txt  # word_count × 1.3 should be ≤ 200

# Verify exit code
echo "Exit code: $?"
```

### 3. Edge Case Audit (MANDATORY - 3 Cases)

For EACH case, print system state BEFORE and AFTER:

#### Case 1: Empty Input (no query, no env vars)
```bash
echo "=== EDGE CASE 1: Empty Input ==="
echo "BEFORE: TOOL_DESCRIPTION=$(env | grep TOOL_DESCRIPTION || echo 'unset')"
echo "BEFORE: TOOL_NAME=$(env | grep TOOL_NAME || echo 'unset')"

unset TOOL_DESCRIPTION TOOL_NAME
./target/debug/context-graph-cli memory inject-brief
EXIT_CODE=$?

echo "AFTER: Exit code = $EXIT_CODE"
echo "AFTER: stdout = (should be empty)"
echo "EXPECTED: Exit 0, empty output"
[ $EXIT_CODE -eq 0 ] && echo "[PASS]" || echo "[FAIL]"
```

#### Case 2: Maximum Content (very long tool description)
```bash
echo "=== EDGE CASE 2: Long Tool Description ==="
LONG_DESC=$(python3 -c "print('word ' * 100)")
echo "BEFORE: TOOL_DESCRIPTION length = $(echo -n "$LONG_DESC" | wc -c)"

TOOL_DESCRIPTION="$LONG_DESC" ./target/debug/context-graph-cli memory inject-brief > /tmp/long_output.txt
EXIT_CODE=$?

WORD_COUNT=$(wc -w < /tmp/long_output.txt)
TOKEN_EST=$((WORD_COUNT * 13 / 10))

echo "AFTER: Exit code = $EXIT_CODE"
echo "AFTER: Output word count = $WORD_COUNT"
echo "AFTER: Est. tokens = $TOKEN_EST"
echo "EXPECTED: Exit 0, tokens ≤ 200"
[ $EXIT_CODE -eq 0 ] && [ $TOKEN_EST -le 200 ] && echo "[PASS]" || echo "[FAIL]"
```

#### Case 3: Invalid Data Directory
```bash
echo "=== EDGE CASE 3: Invalid Data Directory ==="
echo "BEFORE: data_dir = /nonexistent/path"

TOOL_DESCRIPTION="test" ./target/debug/context-graph-cli memory inject-brief --data-dir /nonexistent/path 2>/tmp/error.txt
EXIT_CODE=$?

echo "AFTER: Exit code = $EXIT_CODE"
echo "AFTER: stderr = $(cat /tmp/error.txt)"
echo "EXPECTED: Exit 1 (error), error message on stderr"
[ $EXIT_CODE -eq 1 ] && echo "[PASS]" || echo "[FAIL]"
```

### 4. Evidence of Success Log

Create a verification log showing actual data:

```bash
echo "=== FSV EVIDENCE LOG ==="
echo "Timestamp: $(date -Iseconds)"
echo ""

echo "1. COMMAND EXISTS:"
./target/debug/context-graph-cli memory inject-brief --help 2>&1 | head -5

echo ""
echo "2. HAPPY PATH OUTPUT:"
TOOL_DESCRIPTION="Writing code" ./target/debug/context-graph-cli memory inject-brief 2>&1

echo ""
echo "3. EXIT CODE VERIFICATION:"
TOOL_DESCRIPTION="test" ./target/debug/context-graph-cli memory inject-brief > /dev/null 2>&1
echo "Exit code: $?"

echo ""
echo "4. TEST SUITE:"
cargo test commands::inject --package context-graph-cli -- --nocapture 2>&1 | tail -20
```

---

## Data Storage Verification

This command does NOT write to storage. It only READS from:
- RocksDB store at `{data_dir}/memories/`

To verify memories exist for retrieval:
```bash
# Check if memories directory exists and has data
ls -la ./data/memories/

# If empty, you'll get empty output (exit 0, no stdout)
# This is correct behavior, not an error
```

---

## Common Pitfalls

1. **Wrong command path**: Use `memory inject-brief`, not `inject-brief`
2. **Wrong env var**: Use `TOOL_DESCRIPTION` not `USER_PROMPT`
3. **Budget override**: Do NOT add --budget flag; brief is always 200 tokens
4. **Missing import**: Add `use crate::error::CliExitCode` if not present
5. **Test isolation**: Use `GLOBAL_IDENTITY_LOCK` for env var tests

---

## Constitution Compliance

- **ARCH-09**: Topic threshold = 2.5 (enforced in core pipeline)
- **ARCH-10**: Divergence uses SEMANTIC only (enforced in core)
- **AP-14**: No .unwrap() in library code (use ? operator)
- **AP-26**: Exit codes: 0=success, 1=error, 2=corruption
- **AP-53**: Hook logic in shell scripts calling CLI (this enables that)

</task_spec>
```
