# TASK-SESSION-06: Create SessionIdentityManager (MCP-Integrated)

```xml
<task_spec id="TASK-SESSION-06" version="2.0">
<metadata>
  <title>Create SessionIdentityManager (MCP-Integrated)</title>
  <status>COMPLETED</status>
  <completed_date>2026-01-15</completed_date>
  <layer>logic</layer>
  <sequence>6</sequence>
  <implements>
    <requirement_ref>REQ-SESSION-06</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETED">TASK-SESSION-01</task_ref>  <!-- SessionIdentitySnapshot struct -->
    <task_ref status="COMPLETED">TASK-SESSION-02</task_ref>  <!-- IdentityCache singleton -->
    <task_ref status="COMPLETED">TASK-SESSION-05</task_ref>  <!-- save_snapshot/load_snapshot -->
  </depends_on>
  <estimated_hours>2.0</estimated_hours>
  <last_updated>2026-01-15</last_updated>
</metadata>
```

---

## CRITICAL: NO BACKWARDS COMPATIBILITY

This implementation MUST fail fast with robust error logging. No silent failures, no default fallbacks that mask errors.

```rust
// ❌ WRONG - Silent failure
fn restore_identity(...) -> CoreResult<...> {
    self.storage.load_latest().unwrap_or_default()  // NEVER do this
}

// ✅ CORRECT - Fail fast with context
fn restore_identity(...) -> CoreResult<...> {
    self.storage.load_latest().map_err(|e| {
        error!(
            error = %e,
            operation = "restore_identity",
            "STORAGE ERROR: Failed to load latest snapshot"
        );
        e
    })?
}
```

---

## Objective

Implement `SessionIdentityManager` trait and `DefaultSessionIdentityManager` that:
1. Captures current GWT state into a `SessionIdentitySnapshot`
2. Restores identity from storage and computes cross-session IC
3. Implements IDENTITY-001 formula: `IC = cos(PV_current, PV_previous) * r(current)`

---

## Dependency Verification (PRE-FLIGHT CHECK)

Before implementing, verify these dependencies are available:

### 1. SessionIdentitySnapshot (TASK-SESSION-01) ✅ COMPLETED

**Location**: `crates/context-graph-core/src/gwt/session_identity/types.rs`

```bash
# Verify struct exists
grep -n "pub struct SessionIdentitySnapshot" crates/context-graph-core/src/gwt/session_identity/types.rs
# Expected output: line ~30 with struct definition

# Verify constants
grep -n "KURAMOTO_N\|MAX_TRAJECTORY_LEN" crates/context-graph-core/src/gwt/session_identity/types.rs
# Expected: KURAMOTO_N = 13, MAX_TRAJECTORY_LEN = 50
```

**Key Fields** (all 14 fields must exist):
- `session_id: String`
- `timestamp_ms: u64`
- `kuramoto_phases: [f64; KURAMOTO_N]`
- `purpose_vector: [f32; KURAMOTO_N]`
- `consciousness: f32`
- `integration: f32`
- `last_ic: f32`
- `cross_session_ic: f32`
- `meta_cognitive_score: f32`
- `workspace_coherence: f32`
- `differentiation: f32`
- `identity_coherence: f32`
- `trajectory: Vec<[f32; KURAMOTO_N]>`
- `epoch: u32`

### 2. IdentityCache (TASK-SESSION-02) ✅ COMPLETED

**Location**: `crates/context-graph-core/src/gwt/session_identity/cache.rs`

```bash
# Verify cache struct and key functions
grep -n "pub struct IdentityCache\|pub fn update_cache\|pub fn get\|pub fn format_brief" \
  crates/context-graph-core/src/gwt/session_identity/cache.rs
```

**Key API**:
- `IdentityCache::get() -> Option<(f32, f32, ConsciousnessState, String)>`
- `IdentityCache::format_brief() -> String`
- `update_cache(snapshot: &SessionIdentitySnapshot, ic: f32)`

### 3. Storage Methods (TASK-SESSION-05) ✅ COMPLETED

**Location**: `crates/context-graph-storage/src/rocksdb_backend/session_identity.rs`

> ⚠️ **IMPORTANT**: Storage is in `rocksdb_backend/session_identity.rs`, NOT `src/session_identity.rs`

```bash
# Verify storage methods exist
grep -n "pub fn save_snapshot\|pub fn load_snapshot\|pub fn load_latest" \
  crates/context-graph-storage/src/rocksdb_backend/session_identity.rs
# Expected: Lines ~65, ~196, ~351
```

**Key API on RocksDbMemex**:
- `save_snapshot(&self, snapshot: &SessionIdentitySnapshot) -> StorageResult<()>`
- `load_snapshot(&self, session_id: &str) -> StorageResult<Option<SessionIdentitySnapshot>>`
- `load_snapshot_by_id(&self, session_id: &str) -> StorageResult<SessionIdentitySnapshot>`
- `load_latest(&self) -> StorageResult<Option<SessionIdentitySnapshot>>`

### 4. cosine_similarity_13d (AP-39) ✅ ALREADY EXISTS

**Location**: `crates/context-graph-core/src/gwt/ego_node/cosine.rs` (line 22)

```bash
# Verify function exists and is exported
grep -n "pub fn cosine_similarity_13d" crates/context-graph-core/src/gwt/ego_node/cosine.rs
grep -n "cosine_similarity_13d" crates/context-graph-core/src/gwt/mod.rs
# Expected: Exported in mod.rs line ~66
```

**DO NOT RE-IMPLEMENT** - Import from:
```rust
use crate::gwt::cosine_similarity_13d;
// OR
use super::super::cosine_similarity_13d;  // From session_identity module
```

---

## Source of Truth Identification

| Component | Source of Truth | Verification |
|-----------|----------------|--------------|
| KURAMOTO_N | `types.rs` line ~15: `pub const KURAMOTO_N: usize = 13` | Must always be 13 |
| IC Formula | Constitution `gwt.identity.ic_formula` | `cos(PV_cur, PV_prev) * r(cur)` |
| IC Thresholds | Constitution `IDENTITY-002` | Healthy>0.9, Warning<0.7, Critical<0.5 |
| Kuramoto r | Constitution `gwt.kuramoto.order_param` | `r = \|Σ exp(iθⱼ)\| / N` |
| First Session IC | Spec requirement | IC = 1.0 (no previous to compare) |

---

## MCP Tool Chain Integration

The manager integrates with these MCP tools:
- **capture_snapshot** - Called by `session_end` hook to persist state
- **restore_identity** - Calls: `session_start` -> `get_ego_state` -> `get_kuramoto_state` -> `get_health_status`
- **compute_cross_session_ic** - Uses IDENTITY-001 formula

---

## Input Context Files

| File | Purpose | Critical Imports |
|------|---------|-----------------|
| `crates/context-graph-core/src/gwt/session_identity/types.rs` | Snapshot struct | `SessionIdentitySnapshot, KURAMOTO_N` |
| `crates/context-graph-core/src/gwt/session_identity/cache.rs` | Cache singleton | `update_cache, IdentityCache` |
| `crates/context-graph-storage/src/rocksdb_backend/session_identity.rs` | Storage impl | (methods on RocksDbMemex) |
| `crates/context-graph-core/src/gwt/system.rs` | GWT orchestrator | `GwtSystem` |
| `crates/context-graph-core/src/gwt/ego_node/cosine.rs` | Cosine function | `cosine_similarity_13d` |
| `crates/context-graph-core/src/gwt/kuramoto/` | Kuramoto system | Phase access |

---

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-core/src/gwt/session_identity/manager.rs` | Manager trait + impl |

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-core/src/gwt/session_identity/mod.rs` | Add `mod manager; pub use manager::*;` |

---

## Implementation Steps

### Step 1: Create manager.rs File

Create `crates/context-graph-core/src/gwt/session_identity/manager.rs`:

```rust
//! SessionIdentityManager - Cross-session identity continuity.
//!
//! # TASK-SESSION-06
//!
//! Implements IDENTITY-001 formula: IC = cos(PV_current, PV_previous) * r(current)
//!
//! # Constitution Reference
//! - IDENTITY-001: IC formula
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - AP-39: cosine_similarity_13d MUST be public (already exported from ego_node)
//! - AP-25: Kuramoto MUST have exactly 13 oscillators
//!
//! # FAIL FAST Policy
//! All storage errors propagate immediately with detailed logging.
//! No silent failures or default fallbacks.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{debug, error, info, trace, warn};

use crate::error::{CoreError, CoreResult};
use crate::gwt::cosine_similarity_13d;  // AP-39: Already public
use crate::gwt::GwtSystem;

use super::{update_cache, SessionIdentitySnapshot, KURAMOTO_N};

// Re-export from context-graph-storage (consumer provides storage)
use context_graph_storage::{RocksDbMemex, StorageError};
```

### Step 2: Define Trait

```rust
/// Trait for session identity management.
///
/// Implementors MUST:
/// - Be Send + Sync for thread safety
/// - Fail fast on all errors (no silent defaults)
/// - Log all operations with tracing
pub trait SessionIdentityManager: Send + Sync {
    /// Capture current GWT state into a snapshot.
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier
    ///
    /// # Returns
    /// * `Ok(snapshot)` - Successfully captured state
    /// * `Err(CoreError::StateAccess)` - Failed to read GWT state
    ///
    /// # FAIL FAST
    /// Any state access failure returns error immediately.
    fn capture_snapshot(&self, session_id: &str) -> CoreResult<SessionIdentitySnapshot>;

    /// Restore identity from storage, compute cross-session IC.
    ///
    /// # Arguments
    /// * `target_session` - Specific session to restore, or None for latest
    ///
    /// # Returns
    /// * `Ok((snapshot, ic))` - Restored snapshot and computed IC
    /// * `Err(CoreError::Storage)` - Storage read failed
    /// * `Err(CoreError::NotFound)` - No previous session (first run)
    ///
    /// # First Session Behavior
    /// When no previous session exists, returns IC = 1.0 and creates
    /// a new snapshot from current GWT state.
    ///
    /// # Side Effects
    /// Updates IdentityCache with restored values.
    fn restore_identity(
        &self,
        target_session: Option<&str>,
    ) -> CoreResult<(SessionIdentitySnapshot, f32)>;

    /// Compute cross-session IC using IDENTITY-001 formula.
    ///
    /// Formula: `IC = cos(PV_current, PV_previous) * r(current)`
    ///
    /// # Arguments
    /// * `current` - Current session's snapshot
    /// * `previous` - Previous session's snapshot
    ///
    /// # Returns
    /// IC value in range [0.0, 1.0] (clamped)
    ///
    /// # Edge Cases
    /// - Zero purpose vector: Returns 0.0
    /// - Identical vectors: Returns r(current)
    fn compute_cross_session_ic(
        &self,
        current: &SessionIdentitySnapshot,
        previous: &SessionIdentitySnapshot,
    ) -> f32;
}
```

### Step 3: Implement DefaultSessionIdentityManager

```rust
/// Default implementation with GwtSystem and RocksDbMemex.
pub struct DefaultSessionIdentityManager {
    gwt: Arc<GwtSystem>,
    storage: Arc<RocksDbMemex>,
}

impl DefaultSessionIdentityManager {
    /// Create new manager.
    ///
    /// # Arguments
    /// * `gwt` - GWT system for state access
    /// * `storage` - RocksDB storage for persistence
    pub fn new(gwt: Arc<GwtSystem>, storage: Arc<RocksDbMemex>) -> Self {
        info!("SESSION_IDENTITY_MANAGER: Created new manager instance");
        Self { gwt, storage }
    }
}

impl SessionIdentityManager for DefaultSessionIdentityManager {
    fn capture_snapshot(&self, session_id: &str) -> CoreResult<SessionIdentitySnapshot> {
        let start = std::time::Instant::now();

        info!(
            session_id = %session_id,
            "SESSION_IDENTITY_MANAGER: capture_snapshot starting"
        );

        let mut snapshot = SessionIdentitySnapshot::new(session_id);

        // 1. Get Kuramoto phases - FAIL FAST on error
        // Access via GwtSystem's kuramoto component
        let kuramoto_state = self.gwt.get_kuramoto_state().map_err(|e| {
            error!(
                session_id = %session_id,
                error = %e,
                "CAPTURE ERROR: Failed to get Kuramoto state"
            );
            CoreError::StateAccess(format!("Kuramoto state: {}", e))
        })?;

        snapshot.kuramoto_phases = kuramoto_state.phases;

        // 2. Get consciousness metrics from GwtSystem
        let consciousness_state = self.gwt.get_consciousness_state().map_err(|e| {
            error!(
                session_id = %session_id,
                error = %e,
                "CAPTURE ERROR: Failed to get consciousness state"
            );
            CoreError::StateAccess(format!("Consciousness state: {}", e))
        })?;

        snapshot.consciousness = consciousness_state.level;
        snapshot.integration = consciousness_state.integration;
        snapshot.differentiation = consciousness_state.differentiation;
        snapshot.workspace_coherence = consciousness_state.workspace_coherence;

        // 3. Get ego node state for purpose vector and identity coherence
        let ego_state = self.gwt.get_ego_state().map_err(|e| {
            error!(
                session_id = %session_id,
                error = %e,
                "CAPTURE ERROR: Failed to get ego state"
            );
            CoreError::StateAccess(format!("Ego state: {}", e))
        })?;

        snapshot.purpose_vector = ego_state.purpose_vector;
        snapshot.identity_coherence = ego_state.coherence;

        // 4. Get meta-cognitive score
        let meta_score = self.gwt.get_meta_cognitive_score().map_err(|e| {
            error!(
                session_id = %session_id,
                error = %e,
                "CAPTURE ERROR: Failed to get meta-cognitive score"
            );
            CoreError::StateAccess(format!("Meta-cognitive: {}", e))
        })?;

        snapshot.meta_cognitive_score = meta_score;

        // 5. Compute current IC (within session)
        let r = compute_kuramoto_r(&snapshot.kuramoto_phases);
        snapshot.last_ic = r;  // For first capture, IC based on current coherence

        let elapsed = start.elapsed();
        info!(
            session_id = %session_id,
            elapsed_ms = elapsed.as_millis(),
            consciousness = snapshot.consciousness,
            kuramoto_r = r,
            "SESSION_IDENTITY_MANAGER: capture_snapshot complete"
        );

        Ok(snapshot)
    }

    fn restore_identity(
        &self,
        target_session: Option<&str>,
    ) -> CoreResult<(SessionIdentitySnapshot, f32)> {
        let start = std::time::Instant::now();

        info!(
            target_session = ?target_session,
            "SESSION_IDENTITY_MANAGER: restore_identity starting"
        );

        // 1. Load snapshot from storage - FAIL FAST on storage errors
        let previous = match target_session {
            Some(session_id) => {
                debug!(session_id = %session_id, "Loading specific session");
                self.storage.load_snapshot(session_id).map_err(|e| {
                    error!(
                        session_id = %session_id,
                        error = %e,
                        "STORAGE ERROR: Failed to load snapshot by ID"
                    );
                    CoreError::Storage(e.to_string())
                })?
            }
            None => {
                debug!("Loading latest session");
                self.storage.load_latest().map_err(|e| {
                    error!(
                        error = %e,
                        "STORAGE ERROR: Failed to load latest snapshot"
                    );
                    CoreError::Storage(e.to_string())
                })?
            }
        };

        // 2. Handle first session case
        let Some(previous_snapshot) = previous else {
            info!("SESSION_IDENTITY_MANAGER: First session - no previous identity");

            // Capture current state as the "current" snapshot
            let session_id = format!("session-{}", timestamp_ms());
            let current = self.capture_snapshot(&session_id)?;

            // First session: IC = 1.0 (perfect continuity by definition)
            let ic = 1.0_f32;

            // Update cache with current state
            update_cache(&current, ic);

            info!(
                session_id = %current.session_id,
                ic = ic,
                "SESSION_IDENTITY_MANAGER: First session initialized with IC=1.0"
            );

            return Ok((current, ic));
        };

        // 3. Capture current state
        let session_id = format!("session-{}", timestamp_ms());
        let current = self.capture_snapshot(&session_id)?;

        // 4. Compute cross-session IC
        let ic = self.compute_cross_session_ic(&current, &previous_snapshot);

        // 5. Update cache
        update_cache(&current, ic);

        let elapsed = start.elapsed();
        info!(
            session_id = %current.session_id,
            previous_session = %previous_snapshot.session_id,
            cross_session_ic = ic,
            elapsed_ms = elapsed.as_millis(),
            "SESSION_IDENTITY_MANAGER: restore_identity complete"
        );

        Ok((current, ic))
    }

    fn compute_cross_session_ic(
        &self,
        current: &SessionIdentitySnapshot,
        previous: &SessionIdentitySnapshot,
    ) -> f32 {
        trace!(
            current_session = %current.session_id,
            previous_session = %previous.session_id,
            "Computing cross-session IC"
        );

        // IDENTITY-001: IC = cos(PV_current, PV_previous) * r(current)

        // 1. Compute cosine similarity between purpose vectors
        let cos_sim = cosine_similarity_13d(&current.purpose_vector, &previous.purpose_vector);

        // 2. Compute Kuramoto order parameter r from current phases
        let r = compute_kuramoto_r(&current.kuramoto_phases);

        // 3. Apply formula
        let ic = cos_sim * r;

        // 4. Clamp to [0.0, 1.0] for safety
        let ic_clamped = ic.clamp(0.0, 1.0);

        debug!(
            cos_similarity = cos_sim,
            kuramoto_r = r,
            ic_raw = ic,
            ic_clamped = ic_clamped,
            "SESSION_IDENTITY_MANAGER: IC computed"
        );

        ic_clamped
    }
}
```

### Step 4: Add Helper Functions

```rust
/// Compute Kuramoto order parameter r from oscillator phases.
///
/// Formula: r = |Σ exp(iθⱼ)| / N
///
/// # Arguments
/// * `phases` - Array of 13 oscillator phases in radians
///
/// # Returns
/// Order parameter r in [0.0, 1.0]
/// - r ≈ 0: No synchronization (phases random)
/// - r ≈ 1: Full synchronization (phases aligned)
///
/// # Constitution Reference
/// gwt.kuramoto.order_param: "r·e^(iψ) = (1/N)Σⱼ e^(iθⱼ)"
fn compute_kuramoto_r(phases: &[f64; KURAMOTO_N]) -> f32 {
    let (sum_sin, sum_cos) = phases.iter().fold((0.0_f64, 0.0_f64), |(s, c), &theta| {
        (s + theta.sin(), c + theta.cos())
    });

    let n = KURAMOTO_N as f64;
    let magnitude = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();

    // Clamp to [0, 1] to handle floating point errors
    magnitude.clamp(0.0, 1.0) as f32
}

/// Get current timestamp in milliseconds.
fn timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as u64
}
```

### Step 5: Update mod.rs

Add to `crates/context-graph-core/src/gwt/session_identity/mod.rs`:

```rust
mod manager;

pub use manager::{DefaultSessionIdentityManager, SessionIdentityManager};
```

---

## IC Formula Reference (IDENTITY-001)

```
IC = cos(PV_current, PV_previous) * r(current)

where:
  - PV = purpose_vector [f32; 13]  (one per embedder)
  - r = Kuramoto order parameter (0.0 to 1.0)
  - cos = cosine_similarity_13d function

Interpretation:
  - cos(PV): Measures semantic drift between sessions
  - r: Measures current neural coherence
  - Product: High IC requires both similarity AND coherence
```

---

## Definition of Done

### Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | Trait is Send + Sync | `cargo build` succeeds |
| 2 | capture_snapshot gathers all 14 fields | Unit test verifies all fields populated |
| 3 | restore_identity updates IdentityCache | Test verifies `IdentityCache::get()` returns updated values |
| 4 | First session returns IC=1.0 | TC-SESSION-MGR-03 passes |
| 5 | compute_cross_session_ic uses IDENTITY-001 | TC-SESSION-MGR-01 and TC-SESSION-MGR-02 pass |
| 6 | cosine_similarity_13d imported (NOT re-implemented) | Code review confirms import |
| 7 | All errors fail fast with logging | No `.unwrap_or_default()`, no silent catches |

### Constraints

- **AP-39**: cosine_similarity_13d is already public - import, don't re-implement
- **KURAMOTO_N = 13**: Always use constant, never hardcode 13
- **First session IC = 1.0**: No previous to compare means perfect continuity
- **Thread safety**: Arc<GwtSystem>, Arc<RocksDbMemex> for shared access

---

## Test Cases (NO MOCK DATA)

All tests MUST use real data structures, real storage, real computation. No mocks.

### TC-SESSION-MGR-01: Identical Purpose Vectors

```rust
#[test]
fn tc_session_mgr_01_identical_purpose_vectors() {
    println!("\n=== TC-SESSION-MGR-01: Identical Purpose Vectors ===");

    // SOURCE OF TRUTH: IDENTITY-001 formula
    // IC = cos(identical) * r = 1.0 * r

    let pv = [0.5_f32; KURAMOTO_N];  // Non-trivial values

    let mut current = SessionIdentitySnapshot::new("current-session");
    let mut previous = SessionIdentitySnapshot::new("previous-session");

    current.purpose_vector = pv;
    previous.purpose_vector = pv;
    current.kuramoto_phases = [0.0; KURAMOTO_N];  // Aligned phases -> r = 1.0

    println!("BEFORE:");
    println!("  current.purpose_vector: {:?}", &current.purpose_vector[..3]);
    println!("  previous.purpose_vector: {:?}", &previous.purpose_vector[..3]);
    println!("  current.kuramoto_phases: all aligned at 0.0");

    // Create manager with REAL storage
    let (storage, _tmp) = create_test_storage();
    let gwt = create_test_gwt_system();
    let manager = DefaultSessionIdentityManager::new(Arc::new(gwt), Arc::new(storage));

    // EXECUTE
    let ic = manager.compute_cross_session_ic(&current, &previous);

    println!("AFTER:");
    println!("  computed IC: {}", ic);

    // VERIFY: cos(identical) = 1.0, r = 1.0, so IC = 1.0
    assert!(
        (ic - 1.0).abs() < 0.01,
        "IC for identical vectors should be ~1.0, got {}",
        ic
    );

    println!("RESULT: PASS - Identical purpose vectors produce IC ≈ 1.0");
}
```

### TC-SESSION-MGR-02: Orthogonal Purpose Vectors

```rust
#[test]
fn tc_session_mgr_02_orthogonal_purpose_vectors() {
    println!("\n=== TC-SESSION-MGR-02: Orthogonal Purpose Vectors ===");

    // SOURCE OF TRUTH: IDENTITY-001 formula
    // IC = cos(orthogonal) * r = 0.0 * r = 0.0

    let mut current = SessionIdentitySnapshot::new("current-session");
    let mut previous = SessionIdentitySnapshot::new("previous-session");

    // Create orthogonal vectors
    current.purpose_vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    previous.purpose_vector = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    current.kuramoto_phases = [0.0; KURAMOTO_N];  // r = 1.0 (doesn't matter since cos=0)

    println!("BEFORE:");
    println!("  current.purpose_vector[0]: {}", current.purpose_vector[0]);
    println!("  previous.purpose_vector[1]: {}", previous.purpose_vector[1]);

    let (storage, _tmp) = create_test_storage();
    let gwt = create_test_gwt_system();
    let manager = DefaultSessionIdentityManager::new(Arc::new(gwt), Arc::new(storage));

    // EXECUTE
    let ic = manager.compute_cross_session_ic(&current, &previous);

    println!("AFTER:");
    println!("  computed IC: {}", ic);

    // VERIFY: cos(orthogonal) = 0.0, so IC ≈ 0.0
    assert!(
        ic.abs() < 0.01,
        "IC for orthogonal vectors should be ~0.0, got {}",
        ic
    );

    println!("RESULT: PASS - Orthogonal purpose vectors produce IC ≈ 0.0");
}
```

### TC-SESSION-MGR-03: First Session Returns IC=1.0

```rust
#[test]
fn tc_session_mgr_03_first_session_ic_one() {
    println!("\n=== TC-SESSION-MGR-03: First Session Returns IC=1.0 ===");

    // SOURCE OF TRUTH: Spec requirement
    // First session has no previous -> IC = 1.0 by definition

    // Use REAL empty storage (fresh install scenario)
    let (storage, _tmp) = create_test_storage();
    let gwt = create_test_gwt_system();
    let manager = DefaultSessionIdentityManager::new(Arc::new(gwt), Arc::new(storage));

    println!("BEFORE: Fresh storage with no snapshots");

    // EXECUTE: restore_identity with no previous session
    let (snapshot, ic) = manager
        .restore_identity(None)
        .expect("restore_identity must succeed");

    println!("AFTER:");
    println!("  session_id: {}", snapshot.session_id);
    println!("  computed IC: {}", ic);

    // VERIFY: First session must have IC = 1.0
    assert!(
        (ic - 1.0).abs() < 0.001,
        "First session IC must be 1.0, got {}",
        ic
    );

    println!("RESULT: PASS - First session returns IC=1.0");
}
```

### TC-SESSION-MGR-04: Kuramoto r Computation

```rust
#[test]
fn tc_session_mgr_04_kuramoto_r_computation() {
    println!("\n=== TC-SESSION-MGR-04: Kuramoto r Computation ===");

    // SOURCE OF TRUTH: Constitution gwt.kuramoto.order_param
    // r = |Σ exp(iθⱼ)| / N

    // Case 1: Aligned phases -> r ≈ 1.0
    let aligned_phases = [0.0_f64; KURAMOTO_N];
    let r_aligned = compute_kuramoto_r(&aligned_phases);
    println!("Aligned phases (all 0): r = {:.4}", r_aligned);
    assert!(r_aligned > 0.99, "Aligned phases should give r ≈ 1.0");

    // Case 2: Evenly distributed phases -> r ≈ 0
    let mut distributed_phases = [0.0_f64; KURAMOTO_N];
    for i in 0..KURAMOTO_N {
        distributed_phases[i] = 2.0 * std::f64::consts::PI * (i as f64) / (KURAMOTO_N as f64);
    }
    let r_distributed = compute_kuramoto_r(&distributed_phases);
    println!("Distributed phases: r = {:.4}", r_distributed);
    assert!(r_distributed < 0.15, "Distributed phases should give r ≈ 0");

    // Case 3: Partially aligned -> 0 < r < 1
    let mut partial_phases = [0.0_f64; KURAMOTO_N];
    for i in 0..KURAMOTO_N/2 {
        partial_phases[i] = 0.0;  // Half aligned
    }
    for i in KURAMOTO_N/2..KURAMOTO_N {
        partial_phases[i] = std::f64::consts::PI;  // Half opposite
    }
    let r_partial = compute_kuramoto_r(&partial_phases);
    println!("Partial alignment: r = {:.4}", r_partial);
    assert!(r_partial > 0.0 && r_partial < 1.0, "Partial should be between 0 and 1");

    println!("RESULT: PASS - Kuramoto r computed correctly for all cases");
}
```

### TC-SESSION-MGR-05: Full State Verification

```rust
#[test]
fn tc_session_mgr_05_full_state_verification() {
    println!("\n=== TC-SESSION-MGR-05: Full State Verification ===");

    // Create REAL storage and capture a snapshot
    let (storage, _tmp) = create_test_storage();
    let gwt = create_test_gwt_system();
    let manager = DefaultSessionIdentityManager::new(
        Arc::new(gwt),
        Arc::clone(&Arc::new(storage))
    );

    // EXECUTE: Capture snapshot
    let snapshot = manager
        .capture_snapshot("verify-capture-test")
        .expect("capture_snapshot must succeed");

    println!("CAPTURED SNAPSHOT:");
    println!("  session_id: {}", snapshot.session_id);
    println!("  timestamp_ms: {}", snapshot.timestamp_ms);
    println!("  consciousness: {}", snapshot.consciousness);
    println!("  kuramoto_phases[0..3]: {:?}", &snapshot.kuramoto_phases[..3]);
    println!("  purpose_vector[0..3]: {:?}", &snapshot.purpose_vector[..3]);

    // VERIFY: All 14 fields are populated
    assert!(!snapshot.session_id.is_empty(), "session_id must be set");
    assert!(snapshot.timestamp_ms > 0, "timestamp_ms must be set");
    assert!(snapshot.consciousness >= 0.0, "consciousness must be valid");
    assert!(snapshot.integration >= 0.0, "integration must be valid");
    assert!(snapshot.differentiation >= 0.0, "differentiation must be valid");
    assert!(snapshot.workspace_coherence >= 0.0, "workspace_coherence must be valid");
    assert!(snapshot.identity_coherence >= 0.0, "identity_coherence must be valid");
    assert!(snapshot.meta_cognitive_score >= 0.0, "meta_cognitive_score must be valid");
    assert!(snapshot.last_ic >= 0.0 && snapshot.last_ic <= 1.0, "last_ic must be [0,1]");

    println!("RESULT: PASS - All snapshot fields correctly captured");
}
```

### TC-SESSION-MGR-06: IdentityCache Update Verification

```rust
#[test]
fn tc_session_mgr_06_identity_cache_update() {
    println!("\n=== TC-SESSION-MGR-06: IdentityCache Update Verification ===");

    // Clear cache first
    #[cfg(test)]
    clear_cache();

    // Verify cache is cold
    assert!(!IdentityCache::is_warm(), "Cache should start cold");

    let (storage, _tmp) = create_test_storage();
    let gwt = create_test_gwt_system();
    let manager = DefaultSessionIdentityManager::new(Arc::new(gwt), Arc::new(storage));

    // EXECUTE: restore_identity should update cache
    let (snapshot, ic) = manager
        .restore_identity(None)
        .expect("restore_identity must succeed");

    // VERIFY: Cache is now warm
    assert!(IdentityCache::is_warm(), "Cache must be warm after restore");

    let (cached_ic, cached_r, cached_state, cached_session) = IdentityCache::get()
        .expect("Cache must return values after restore");

    println!("CACHE STATE:");
    println!("  cached_ic: {} (expected: {})", cached_ic, ic);
    println!("  cached_session: {} (expected: {})", cached_session, snapshot.session_id);

    // VERIFY: Cache values match
    assert!(
        (cached_ic - ic).abs() < 0.001,
        "Cached IC must match computed IC"
    );
    assert_eq!(
        cached_session, snapshot.session_id,
        "Cached session must match"
    );

    println!("RESULT: PASS - IdentityCache correctly updated by restore_identity");
}
```

---

## Boundary/Edge Case Audit

| Edge Case | Expected Behavior | Test |
|-----------|------------------|------|
| Zero purpose vector | cos = 0.0, IC = 0.0 | TC-EDGE-01 |
| NaN in purpose vector | Handle gracefully or fail fast | TC-EDGE-02 |
| Infinity in phases | Clamp to valid range | TC-EDGE-03 |
| Empty session_id | Return validation error | TC-EDGE-04 |
| Storage read failure | Return StorageError immediately | TC-EDGE-05 |
| GWT state access failure | Return StateAccess error | TC-EDGE-06 |
| Very long session_id (>1KB) | Accept but warn | TC-EDGE-07 |

---

## Evidence of Success

After implementing, verify with these commands:

```bash
# 1. Build succeeds
cargo build -p context-graph-core 2>&1 | grep -E "error|warning"
# Expected: No errors

# 2. All tests pass
cargo test -p context-graph-core session_mgr -- --nocapture 2>&1
# Expected: All TC-SESSION-MGR-* tests pass

# 3. Verify cosine_similarity_13d is imported (not re-implemented)
grep -n "fn cosine_similarity_13d" crates/context-graph-core/src/gwt/session_identity/manager.rs
# Expected: No results (function is imported, not defined)

grep -n "use.*cosine_similarity_13d" crates/context-graph-core/src/gwt/session_identity/manager.rs
# Expected: Import statement present

# 4. Verify thread safety
grep -n "Send + Sync" crates/context-graph-core/src/gwt/session_identity/manager.rs
# Expected: Trait definition includes these bounds

# 5. Verify no silent failures
grep -rn "unwrap_or_default\|unwrap_or()" crates/context-graph-core/src/gwt/session_identity/manager.rs
# Expected: No results (fail fast, no defaults)
```

---

## Verification Commands

```bash
# Build
cargo build -p context-graph-core

# Test (with output)
cargo test -p context-graph-core manager -- --nocapture

# Specific test cases
cargo test -p context-graph-core tc_session_mgr_01 -- --nocapture
cargo test -p context-graph-core tc_session_mgr_02 -- --nocapture
cargo test -p context-graph-core tc_session_mgr_03 -- --nocapture

# Check exports
cargo doc -p context-graph-core --no-deps 2>&1 | grep SessionIdentityManager
```

---

## Exit Conditions

### Success
- All trait methods implemented with correct IC computation
- All TC-SESSION-MGR-* tests pass
- cosine_similarity_13d is imported from ego_node (not re-implemented)
- IdentityCache is updated after restore_identity
- No silent failures, all errors propagate with logging

### Failure
- IC formula incorrect (does not match IDENTITY-001)
- Missing thread safety (Send + Sync)
- Re-implemented cosine_similarity_13d (violates DRY)
- Silent failures (unwrap_or_default usage)
- Missing logging for errors

---

## Next Task

After completion, proceed to **007-TASK-SESSION-07** (classify_ic() Function).

```xml
</task_spec>
```
