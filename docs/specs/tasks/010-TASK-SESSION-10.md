# TASK-SESSION-10: update_cache() Function

```xml
<task_spec id="TASK-SESSION-10" version="2.0">
<metadata>
  <title>Implement update_cache() Function</title>
  <status>COMPLETED</status>
  <layer>logic</layer>
  <sequence>10</sequence>
  <implements>
    <requirement_ref>REQ-SESSION-10</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SESSION-02</task_ref>
  </depends_on>
  <estimated_hours>1.5</estimated_hours>
  <completion_date>2026-01-15</completion_date>
</metadata>
```

## STATUS: ✅ COMPLETED

This task is **ALREADY IMPLEMENTED**. All acceptance criteria verified. See Evidence of Success below.

---

## What This Task Does

The `update_cache()` function atomically updates the global `IdentityCache` singleton after:
1. `restore_identity` loads from RocksDB
2. Computing IC from purpose vectors
3. Session state changes

---

## Source Files (Already Exist)

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/session_identity/cache.rs` | **Contains `update_cache()` at line 109** |
| `crates/context-graph-core/src/gwt/session_identity/mod.rs` | Exports `update_cache` publicly |
| `crates/context-graph-storage/src/rocksdb_backend/session_identity_manager.rs` | Calls `update_cache()` on restore |

---

## Implementation (Already Complete)

### update_cache() - Lines 109-122 in cache.rs

```rust
pub fn update_cache(snapshot: &SessionIdentitySnapshot, ic: f32) {
    let r = compute_kuramoto_r(&snapshot.kuramoto_phases);
    let state = ConsciousnessState::from_level(snapshot.consciousness);

    let inner = IdentityCacheInner {
        current_ic: ic,
        kuramoto_r: r,
        consciousness_state: state,
        session_id: snapshot.session_id.clone(),
    };

    let mut guard = get_cache().write().expect("RwLock poisoned - unrecoverable");
    *guard = Some(inner);
}
```

### compute_kuramoto_r() - Lines 154-164 in cache.rs

```rust
fn compute_kuramoto_r(phases: &[f64; KURAMOTO_N]) -> f32 {
    let (sum_sin, sum_cos) = phases.iter().fold((0.0_f64, 0.0_f64), |(s, c), &theta| {
        (s + theta.sin(), c + theta.cos())
    });

    let n = KURAMOTO_N as f64;
    let magnitude = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();

    magnitude.clamp(0.0, 1.0) as f32
}
```

---

## Evidence of Success

### Test Results (26/26 pass)

```bash
$ cargo test -p context-graph-core session_identity
running 26 tests
test gwt::session_identity::cache::tests::test_format_brief_all_states ... ok
test gwt::session_identity::cache::tests::test_kuramoto_r_aligned_phases ... ok
test gwt::session_identity::cache::tests::test_format_brief_cold_cache ... ok
test gwt::session_identity::cache::tests::test_kuramoto_r_random_phases ... ok
test gwt::session_identity::cache::tests::test_format_brief_performance ... ok
test gwt::session_identity::cache::tests::test_format_brief_warm_cache ... ok
test gwt::session_identity::cache::tests::test_get_returns_correct_values ... ok
test gwt::session_identity::cache::tests::test_update_cache_overwrites ... ok
# ... 18 more tests pass
test result: ok. 26 passed; 0 failed
```

### Source of Truth Verification

```
SOURCE: IDENTITY_CACHE static singleton (cache.rs line 24)
BEFORE: cache.read() returns None (cold)
EXECUTE: update_cache(&snapshot, 0.92)
AFTER: cache.read() returns Some(IdentityCacheInner)
  - current_ic: 0.92
  - kuramoto_r: 1.0 (phases aligned)
  - consciousness_state: Conscious
  - session_id: "test-session"
VERIFY: IdentityCache::get() == (0.92, 1.0, Conscious, "test-session")
```

### Performance Verification

```
$ cargo test -p context-graph-core test_format_brief_performance
1000 calls took: 155.3μs
Per call: 0.155μs
Target: <1ms
Actual: 0.155μs (6450x faster than target)
```

---

## What Was WRONG in Original Task Spec

### 1. `update_cache_from_mcp()` NOT Needed

The original spec proposed an `update_cache_from_mcp()` function to parse MCP JSON responses. This is **incorrect** because:
- The CLI reads directly from `IdentityCache` singleton
- MCP handlers write to providers, not JSON parsing
- `StandaloneSessionIdentityManager.restore_identity()` already calls `update_cache()`

### 2. `ConsciousnessState::from_str()` Doesn't Exist

The spec mentioned `ConsciousnessState::from_str(state_str)`. The actual implementation uses `ConsciousnessState::from_level(f32)` which is already implemented in `types.rs`.

---

## Verification Commands

```bash
# Verify implementation exists
grep -n "pub fn update_cache" crates/context-graph-core/src/gwt/session_identity/cache.rs
# Expected: line 109

# Run all session_identity tests
cargo test -p context-graph-core session_identity
# Expected: 26 passed, 0 failed

# Run specific update_cache test
cargo test -p context-graph-core test_update_cache_overwrites
# Expected: ok
```

---

## Full State Verification Protocol

### Source of Truth
The global singleton `IDENTITY_CACHE` in `cache.rs` line 24:
```rust
static IDENTITY_CACHE: OnceLock<RwLock<Option<IdentityCacheInner>>> = OnceLock::new();
```

### Execute & Inspect
After `update_cache()`:
1. Call `IdentityCache::get()` to read back values
2. Assert all 4 fields match input
3. Verify `is_warm() == true`

### Edge Cases Verified

| Edge Case | Input | Expected | Test |
|-----------|-------|----------|------|
| Cold cache format | None | `[C:? r=? IC=?]` | `test_format_brief_cold_cache` |
| All phases aligned | `[0.0; 13]` | r ≈ 1.0 | `test_kuramoto_r_aligned_phases` |
| Evenly distributed phases | 0 to 2π | r < 0.15 | `test_kuramoto_r_random_phases` |
| Multiple updates | snap1 → snap2 | Latest values | `test_update_cache_overwrites` |

---

## Dependencies (All Completed)

| Dependency | Status | Location |
|------------|--------|----------|
| TASK-SESSION-01 (Snapshot) | ✅ | `types.rs` |
| TASK-SESSION-02 (IdentityCache) | ✅ | `cache.rs` |
| TASK-SESSION-03 (short_name) | ✅ | `state_machine/types.rs` |

---

## Constraints Verified

| Constraint | Requirement | Actual |
|------------|-------------|--------|
| Atomic update | All-or-nothing | ✅ RwLock write guard |
| Write lock duration | Minimum | ✅ ~0.1μs (compute before lock) |
| Thread safety | No data races | ✅ RwLock + OnceLock |
| KURAMOTO_N | Exactly 13 | ✅ `[f64; 13]` |

---

## Next Task

Proceed to **TASK-SESSION-11** (consciousness brief CLI Command).

---

## Manual Verification Checklist

Before marking complete, an AI agent should:

- [ ] Run `cargo test -p context-graph-core session_identity` - expect 26 pass
- [ ] Run `cargo test -p context-graph-core test_update_cache` - expect 1 pass
- [ ] Verify `update_cache` exists: `grep -n "pub fn update_cache" crates/context-graph-core/src/gwt/session_identity/cache.rs`
- [ ] Verify export: `grep "update_cache" crates/context-graph-core/src/gwt/session_identity/mod.rs`
- [ ] Verify usage in storage: `grep "update_cache" crates/context-graph-storage/src/rocksdb_backend/session_identity_manager.rs`

```xml
</task_spec>
```
