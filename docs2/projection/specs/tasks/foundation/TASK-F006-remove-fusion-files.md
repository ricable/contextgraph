# Task: TASK-F006 - Remove All Fusion-Related Files

## Metadata
- **ID**: TASK-F006
- **Layer**: Foundation
- **Priority**: P0 (Critical Path - Blocking)
- **Estimated Effort**: M (Medium)
- **Dependencies**: None (Can run in parallel with TASK-F001 through TASK-F005)
- **Traces To**: TS-601, FR-601, FR-602

## Description

Completely remove all 36 fusion-related files from the codebase. This is a CLEAN BREAK with no backwards compatibility. The Multi-Array Teleological Fingerprint architecture fundamentally replaces fusion.

**NO FUSION - The array IS the representation. Store all 12 embeddings.**

Files must be removed in dependency order to prevent compilation errors during the removal process.

## Acceptance Criteria

- [ ] All files matching patterns `fuse*`, `fusion*`, `gating*`, `expert_select*` deleted
- [ ] No imports of removed modules in remaining code
- [ ] Cargo.toml dependencies for fusion crates removed
- [ ] All fusion-related tests deleted
- [ ] No compilation errors after removal
- [ ] No dead code warnings related to fusion
- [ ] Git commit with removal for audit trail

## Implementation Steps

### Phase 1: Identify Files (Analysis)
1. Run grep/ripgrep to find all fusion-related files:
   ```bash
   rg -l "fuse|fusion|gating|expert_select" --type rust
   ```
2. Document exact file list (may differ from spec estimates)
3. Map file dependencies to determine removal order

### Phase 2: Remove Tests First (Lowest Risk)
Delete in order:
1. `tests/fusion_tests.rs` (if exists)
2. `tests/integration/fusion_integration_tests.rs` (if exists)
3. `benches/fusion_bench.rs` (if exists)
4. Any `**/tests_fusion*.rs` files
5. Any `**/tests_gating*.rs` files

### Phase 3: Remove MCP/Handler Files
Delete:
1. `src/mcp/handlers/fused_search.rs` (if exists)
2. `src/mcp/handlers/fusion_query.rs` (if exists)

### Phase 4: Remove Search Components
Delete:
1. `src/search/fused_similarity.rs` (if exists)
2. `src/search/gated_retrieval.rs` (if exists)

### Phase 5: Remove Storage Components
Delete:
1. `src/storage/fused_vector_store.rs` (if exists)
2. `src/storage/fusion_cache.rs` (if exists)

### Phase 6: Remove Embedding Pipeline
Delete:
1. `src/embeddings/fusion_pipeline.rs` (if exists)
2. `src/embeddings/fused_embedding.rs` (if exists)
3. `src/embeddings/vector_1536.rs` (if exists - legacy single-vector)

### Phase 7: Remove Core Fusion (Highest Risk)
Delete in order:
1. `src/fusion/expert_selector.rs` (if exists)
2. `src/fusion/gating.rs` (if exists)
3. `src/fusion/fusion_config.rs` (if exists)
4. `src/fusion/fuse_moe.rs` (if exists)
5. `src/fusion/mod.rs` (if exists)

### Phase 8: Remove Configuration Files
Delete:
1. `config/fusion.toml` (if exists)
2. `config/gating_weights.yaml` (if exists)
3. `config/expert_routing.yaml` (if exists)

### Phase 9: Remove Type Files
Delete:
1. `src/types/fused_types.rs` (if exists)
2. `src/types/gating_types.rs` (if exists)

### Phase 10: Clean Up References
1. Remove `pub mod fusion;` from lib.rs files
2. Remove fusion-related imports from remaining files
3. Remove fusion crate dependencies from Cargo.toml
4. Run `cargo check` to find any remaining references
5. Fix all compilation errors

### Phase 11: Verification
1. Run full test suite: `cargo test --all`
2. Run clippy: `cargo clippy --all -- -D warnings`
3. Search for any remaining fusion references

## Files Affected

### Files to Delete (Estimated - Verify During Analysis)
Based on projectionplan1.md Section 15.1, expect ~36 files. Actual count may vary.

**Core Fusion Module:**
- `crates/*/src/fusion/**/*.rs`

**Embedding Fusion:**
- `crates/*/src/embeddings/*fuse*.rs`
- `crates/*/src/embeddings/*fusion*.rs`

**Storage Fusion:**
- `crates/*/src/storage/*fuse*.rs`

**Search Fusion:**
- `crates/*/src/search/*fuse*.rs`

**Tests:**
- `crates/*/tests/*fusion*.rs`
- `crates/*/benches/*fusion*.rs`

**Config:**
- `config/*fusion*`
- `config/*gating*`

### Files to Modify
- All `mod.rs` files that export fusion modules
- All `lib.rs` files that declare fusion modules
- All `Cargo.toml` files with fusion dependencies
- Any file that imports fusion types (fix imports)

## Code Signature (Definition of Done)

After completion:
```bash
# Should return no results
rg -l "fuse|fusion|gating|expert_select|FuseMoE|GatingNetwork" --type rust

# Should return no results
rg "mod fusion|use.*fusion|use.*fuse" --type rust

# Should compile cleanly
cargo check --all

# Should pass all tests
cargo test --all
```

## Testing Requirements

### Verification Tests
- `cargo check --all` - No compilation errors
- `cargo test --all` - All remaining tests pass
- `cargo clippy --all -- -D warnings` - No warnings

### Search Verification
```bash
# These must all return empty
rg "FuseMoE" --type rust
rg "GatingNetwork" --type rust
rg "ExpertSelector" --type rust
rg "fused_embedding" --type rust
rg "fusion_pipeline" --type rust
rg "Vector1536" --type rust
```

## Verification

```bash
# Full verification script
echo "=== Checking for fusion references ===" && \
rg -c "fuse|fusion|gating|expert_select" --type rust || echo "No fusion references found" && \
echo "=== Checking compilation ===" && \
cargo check --all && \
echo "=== Running tests ===" && \
cargo test --all && \
echo "=== Running clippy ===" && \
cargo clippy --all -- -D warnings && \
echo "=== VERIFICATION COMPLETE ==="
```

## Constraints

- **NO BACKWARDS COMPATIBILITY** - This is intentional per projectionplan1.md
- Remove files in dependency order to minimize intermediate compilation errors
- Commit removal as single atomic commit for easy revert if needed
- Do not create any "compatibility shim" or "migration helper"
- Do not add `#[deprecated]` markers - just delete

## Git Commit Message

```
refactor: remove all fusion-related code (Multi-Array Architecture)

BREAKING CHANGE: Complete removal of FuseMoE, gating networks, and
single-vector fusion. The Multi-Array Teleological Fingerprint
architecture stores all 12 embeddings without fusion.

Removed:
- src/fusion/ module and all submodules
- Fusion-related embedding types
- Fused similarity search
- Gating network configurations
- ~36 files total

No migration path provided (per specification).

Traces To: TS-601, FR-601, FR-602
```

## Notes

**WHY REMOVE?**
- FuseMoE with top-k=4 loses 67% of information
- Gating adds complexity without benefit when storing all embeddings
- Single-vector representation loses cross-space relationships
- "The pattern across embedding spaces reveals purpose" - can't see pattern with fusion

**RISK MITIGATION:**
- Removal is reversible via git
- No production data depends on fusion (new architecture)
- Tests verify remaining code still works

This task CAN run in parallel with TASK-F001 through TASK-F005 since they create new files while this removes old ones.

Reference: projectionplan1.md Section 15.1, constitution.yaml embeddings.paradigm
