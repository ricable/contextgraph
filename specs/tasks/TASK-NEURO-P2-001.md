# TASK-NEURO-P2-001: Implement Direct Dopamine Feedback Loop

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-NEURO-P2-001 |
| **Title** | Implement Direct Dopamine Feedback from Steering Subsystem |
| **Status** | Ready |
| **Priority** | P2 (Minor Refinement) |
| **Layer** | logic |
| **Sequence** | 1 |
| **Estimated Complexity** | Low |
| **Estimated Duration** | 2-4 hours |
| **Implements** | SPEC-NEURO-001 |
| **Depends On** | None (foundation layer already exists) |

---

## 1. Context

This task implements the Direct Dopamine Feedback Loop as specified in SPEC-NEURO-001. The goal is to create a direct path from steering feedback to dopamine modulation, bypassing the indirect edge weight -> workspace entry chain.

**Current State**:
- `DopamineModulator` exists with `on_workspace_entry()` and `on_negative_event()`
- `NeuromodulationManager` coordinates all neuromodulators
- `SteeringSystem` computes feedback with reward in [-1, 1]
- No direct connection between steering and dopamine

**Target State**:
- `DopamineModulator.on_goal_progress(delta)` directly modulates DA
- `NeuromodulationManager.on_goal_progress(delta)` forwards to dopamine
- MCP steering handler invokes `on_goal_progress()` after computing feedback

---

## 2. Input Context Files

The agent MUST read these files before implementation:

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/neuromod/dopamine.rs` | Dopamine modulator implementation - add `on_goal_progress()` |
| `crates/context-graph-core/src/neuromod/state.rs` | Neuromodulation manager - add forwarding method |
| `crates/context-graph-core/src/neuromod/mod.rs` | Module exports - verify public API |
| `crates/context-graph-mcp/src/handlers/steering.rs` | MCP handler - integration point |
| `crates/context-graph-mcp/src/handlers/mod.rs` | Handlers struct - verify neuromod access |
| `specs/functional/SPEC-NEURO-001.md` | Functional specification |

---

## 3. Scope

### 3.1 In Scope

1. Add `DA_GOAL_SENSITIVITY` constant (0.1) to `dopamine.rs`
2. Implement `DopamineModulator::on_goal_progress(delta: f32)` method
3. Implement `NeuromodulationManager::on_goal_progress(delta: f32)` forwarding method
4. Integrate `on_goal_progress()` call in `call_get_steering_feedback()` MCP handler
5. Add unit tests for dopamine goal progress
6. Add unit tests for manager goal progress forwarding
7. Add debug logging for goal progress events

### 3.2 Out of Scope

- Modifying `on_workspace_entry()` behavior
- Changing existing decay mechanics
- Adding configuration file support (constant only for now)
- Integration tests (separate task if needed)
- MCP tool schema changes

---

## 4. Definition of Done

### 4.1 Required Signatures

The following exact signatures MUST be produced:

**File: `crates/context-graph-core/src/neuromod/dopamine.rs`**

```rust
/// Dopamine adjustment sensitivity for goal progress events.
/// Default: 0.1 means maximum reward (+1.0) increases DA by 0.1
pub const DA_GOAL_SENSITIVITY: f32 = 0.1;

impl DopamineModulator {
    /// Handle goal progress event from steering subsystem.
    ///
    /// Adjusts dopamine based on goal achievement delta:
    /// - Positive delta (goal progress): DA increases
    /// - Negative delta (goal regression): DA decreases
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta, typically from SteeringReward.value [-1, 1]
    ///
    /// # Effects
    /// * DA adjusted by delta * DA_GOAL_SENSITIVITY
    /// * Clamped to [DA_MIN, DA_MAX]
    /// * Updates last_trigger if adjustment is non-zero
    pub fn on_goal_progress(&mut self, delta: f32)
}
```

**File: `crates/context-graph-core/src/neuromod/state.rs`**

```rust
impl NeuromodulationManager {
    /// Handle goal progress from steering subsystem.
    ///
    /// Propagates goal achievement/regression to dopamine modulator.
    /// This provides direct neurochemical response to steering feedback.
    ///
    /// # Arguments
    /// * `delta` - Goal progress delta, typically SteeringReward.value [-1, 1]
    pub fn on_goal_progress(&mut self, delta: f32)
}
```

### 4.2 Constraints

- [ ] `DA_GOAL_SENSITIVITY` MUST be 0.1
- [ ] `on_goal_progress()` MUST clamp result to [DA_MIN, DA_MAX]
- [ ] `on_goal_progress()` MUST update `last_trigger` when adjustment is non-zero
- [ ] `on_goal_progress()` MUST log at DEBUG level with: delta, adjustment, new_value
- [ ] `on_goal_progress()` MUST skip adjustment when delta is NaN or epsilon-small
- [ ] No heap allocations in `on_goal_progress()` hot path
- [ ] All existing tests MUST continue to pass
- [ ] New tests MUST achieve 100% coverage of new code paths

### 4.3 Test Requirements

**Unit Tests (in `dopamine.rs`)**:

```rust
#[test]
fn test_dopamine_on_goal_progress_positive() {
    let mut modulator = DopamineModulator::new();
    let initial = modulator.value();
    modulator.on_goal_progress(0.5);
    let expected = initial + 0.5 * DA_GOAL_SENSITIVITY;
    assert!((modulator.value() - expected).abs() < f32::EPSILON);
}

#[test]
fn test_dopamine_on_goal_progress_negative() {
    let mut modulator = DopamineModulator::new();
    let initial = modulator.value();
    modulator.on_goal_progress(-0.5);
    let expected = initial - 0.5 * DA_GOAL_SENSITIVITY;
    assert!((modulator.value() - expected).abs() < f32::EPSILON);
}

#[test]
fn test_dopamine_on_goal_progress_ceiling_clamp() {
    let mut modulator = DopamineModulator::new();
    modulator.set_value(DA_MAX);
    modulator.on_goal_progress(1.0);
    assert!((modulator.value() - DA_MAX).abs() < f32::EPSILON);
}

#[test]
fn test_dopamine_on_goal_progress_floor_clamp() {
    let mut modulator = DopamineModulator::new();
    modulator.set_value(DA_MIN);
    modulator.on_goal_progress(-1.0);
    assert!((modulator.value() - DA_MIN).abs() < f32::EPSILON);
}

#[test]
fn test_dopamine_on_goal_progress_zero_delta() {
    let mut modulator = DopamineModulator::new();
    let initial = modulator.value();
    modulator.on_goal_progress(0.0);
    assert!((modulator.value() - initial).abs() < f32::EPSILON);
}

#[test]
fn test_dopamine_on_goal_progress_updates_trigger() {
    let mut modulator = DopamineModulator::new();
    assert!(modulator.level().last_trigger.is_none());
    modulator.on_goal_progress(0.5);
    assert!(modulator.level().last_trigger.is_some());
}
```

**Unit Tests (in `state.rs`)**:

```rust
#[test]
fn test_manager_on_goal_progress() {
    let mut manager = NeuromodulationManager::new();
    let initial = manager.get_hopfield_beta();
    manager.on_goal_progress(0.8);
    let expected = initial + 0.8 * dopamine::DA_GOAL_SENSITIVITY;
    assert!((manager.get_hopfield_beta() - expected).abs() < f32::EPSILON);
}

#[test]
fn test_manager_on_goal_progress_negative() {
    let mut manager = NeuromodulationManager::new();
    let initial = manager.get_hopfield_beta();
    manager.on_goal_progress(-0.6);
    let expected = initial - 0.6 * dopamine::DA_GOAL_SENSITIVITY;
    assert!((manager.get_hopfield_beta() - expected).abs() < f32::EPSILON);
}
```

---

## 5. Files to Modify

| File Path | Action | Description |
|-----------|--------|-------------|
| `crates/context-graph-core/src/neuromod/dopamine.rs` | MODIFY | Add `DA_GOAL_SENSITIVITY` constant and `on_goal_progress()` method |
| `crates/context-graph-core/src/neuromod/state.rs` | MODIFY | Add `on_goal_progress()` forwarding method to manager |
| `crates/context-graph-core/src/neuromod/mod.rs` | MODIFY | Export `DA_GOAL_SENSITIVITY` constant |

---

## 6. Files to Create

None - all changes are modifications to existing files.

---

## 7. Pseudo-Code

### 7.1 dopamine.rs Additions

```rust
// After existing constants (around line 35)
/// Dopamine adjustment sensitivity for goal progress events
pub const DA_GOAL_SENSITIVITY: f32 = 0.1;

// Add method to DopamineModulator impl block (around line 95)
impl DopamineModulator {
    // ... existing methods ...

    /// Handle goal progress event from steering subsystem
    pub fn on_goal_progress(&mut self, delta: f32) {
        // Guard against NaN
        if delta.is_nan() {
            tracing::warn!("on_goal_progress received NaN delta, skipping");
            return;
        }

        // Calculate adjustment
        let adjustment = delta * DA_GOAL_SENSITIVITY;

        // Skip if adjustment is effectively zero
        if adjustment.abs() <= f32::EPSILON {
            return;
        }

        // Apply adjustment with clamping
        let old_value = self.level.value;
        self.level.value = (self.level.value + adjustment).clamp(DA_MIN, DA_MAX);

        // Update trigger timestamp
        self.level.last_trigger = Some(Utc::now());

        // Log the adjustment
        tracing::debug!(
            delta = delta,
            adjustment = adjustment,
            old_value = old_value,
            new_value = self.level.value,
            "Dopamine adjusted on goal progress"
        );
    }
}
```

### 7.2 state.rs Additions

```rust
impl NeuromodulationManager {
    // ... existing methods ...

    /// Handle goal progress from steering subsystem.
    ///
    /// Propagates goal achievement/regression to dopamine modulator.
    pub fn on_goal_progress(&mut self, delta: f32) {
        self.dopamine.on_goal_progress(delta);
    }
}
```

### 7.3 mod.rs Additions

```rust
// Update the dopamine re-exports
pub use dopamine::{
    DopamineLevel, DopamineModulator,
    DA_BASELINE, DA_MAX, DA_MIN,
    DA_GOAL_SENSITIVITY,  // ADD THIS
};
```

---

## 8. Validation Criteria

### 8.1 Automated Validation

| Command | Expected Result |
|---------|-----------------|
| `cargo build -p context-graph-core` | Success, no warnings |
| `cargo test -p context-graph-core neuromod` | All tests pass |
| `cargo clippy -p context-graph-core` | No warnings |
| `cargo doc -p context-graph-core --no-deps` | Docs generate successfully |

### 8.2 Manual Validation

1. **Positive Progress Test**:
   - Call `on_goal_progress(1.0)` on fresh modulator
   - Verify DA increased from 3.0 to 3.1

2. **Negative Progress Test**:
   - Call `on_goal_progress(-1.0)` on fresh modulator
   - Verify DA decreased from 3.0 to 2.9

3. **Bounds Test**:
   - Set DA to 5.0, call `on_goal_progress(1.0)`
   - Verify DA stays at 5.0 (clamped)

4. **Manager Forwarding Test**:
   - Call `manager.on_goal_progress(0.5)`
   - Verify `manager.get_hopfield_beta()` reflects change

---

## 9. Test Commands

```bash
# Build the crate
cargo build -p context-graph-core

# Run all neuromod tests
cargo test -p context-graph-core neuromod

# Run specific dopamine tests
cargo test -p context-graph-core dopamine

# Run with verbose output to see debug logs
RUST_LOG=debug cargo test -p context-graph-core test_dopamine_on_goal_progress -- --nocapture

# Check for clippy warnings
cargo clippy -p context-graph-core -- -D warnings

# Verify documentation builds
cargo doc -p context-graph-core --no-deps
```

---

## 10. Rollback Plan

If implementation causes issues:

1. Revert changes to `dopamine.rs` (remove constant and method)
2. Revert changes to `state.rs` (remove forwarding method)
3. Revert changes to `mod.rs` (remove export)
4. Run `cargo test` to verify rollback

No database migrations or configuration changes are involved.

---

## 11. Dependencies and Blockers

### 11.1 Technical Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| `chrono::Utc` | Available | Already imported in dopamine.rs |
| `tracing` | Available | Already imported in dopamine.rs |
| `DopamineModulator` | Exists | Target for modification |
| `NeuromodulationManager` | Exists | Target for modification |

### 11.2 Blockers

None identified. All required infrastructure exists.

---

## 12. Implementation Checklist

- [ ] Read all input context files
- [ ] Add `DA_GOAL_SENSITIVITY` constant to `dopamine.rs`
- [ ] Implement `DopamineModulator::on_goal_progress()` method
- [ ] Add unit tests for `on_goal_progress()` in `dopamine.rs`
- [ ] Implement `NeuromodulationManager::on_goal_progress()` forwarding
- [ ] Add unit tests for manager forwarding in `state.rs`
- [ ] Update exports in `mod.rs`
- [ ] Run `cargo build -p context-graph-core`
- [ ] Run `cargo test -p context-graph-core neuromod`
- [ ] Run `cargo clippy -p context-graph-core`
- [ ] Verify all existing tests still pass
- [ ] Update task status to COMPLETED

---

## 13. Notes for Implementation Agent

### 13.1 Code Style

- Follow existing patterns in `dopamine.rs` for method structure
- Use `tracing::debug!` macro with structured logging fields
- Document public API with rustdoc comments
- Include `# Arguments`, `# Effects` sections in doc comments

### 13.2 Test Style

- Follow existing test patterns in `dopamine.rs`
- Use descriptive test names with `test_dopamine_on_goal_progress_*` prefix
- Assert with `f32::EPSILON` for floating point comparisons
- Include edge cases (zero, NaN, bounds)

### 13.3 Integration Point (Future Task)

The MCP steering handler integration is OUT OF SCOPE for this task. A follow-up task will:
1. Add `NeuromodulationManager` to `Handlers` struct (if not present)
2. Call `on_goal_progress()` in `call_get_steering_feedback()`

This task focuses solely on the core neuromodulation layer.

---

## 14. Traceability

| Requirement | Implemented By | Test Coverage |
|-------------|----------------|---------------|
| FR-NEURO-001-01 | `NeuromodulationManager::on_goal_progress()` | `test_manager_on_goal_progress*` |
| FR-NEURO-001-02 | `DopamineModulator::on_goal_progress()` | `test_dopamine_on_goal_progress*` |
| FR-NEURO-001-03 | Out of scope (MCP integration) | - |
| FR-NEURO-001-04 | `DA_GOAL_SENSITIVITY` constant | Implicit in all tests |
| NFR-NEURO-001-01 | No allocations, simple math | Performance inherent |
| NFR-NEURO-001-02 | Existing tests still pass | Regression suite |
| NFR-NEURO-001-03 | `tracing::debug!` calls | Log verification |

---

## Appendix A: Related Code References

### A.1 Existing `on_workspace_entry()` Pattern

```rust
// From dopamine.rs lines 89-98 - follow this pattern
pub fn on_workspace_entry(&mut self) {
    self.level.value = (self.level.value + DA_WORKSPACE_INCREMENT).clamp(DA_MIN, DA_MAX);
    self.level.last_trigger = Some(Utc::now());
    tracing::debug!(
        "Dopamine increased on workspace entry: value={:.3}",
        self.level.value
    );
}
```

### A.2 Existing `on_negative_event()` Pattern

```rust
// From dopamine.rs lines 100-108 - follow this pattern
pub fn on_negative_event(&mut self, magnitude: f32) {
    let delta = magnitude.abs() * 0.1;
    self.level.value = (self.level.value - delta).clamp(DA_MIN, DA_MAX);
    tracing::debug!(
        "Dopamine decreased on negative event: value={:.3}",
        self.level.value
    );
}
```

### A.3 Manager Event Forwarding Pattern

```rust
// From state.rs lines 258-261 - follow this pattern
pub fn on_workspace_entry(&mut self) {
    self.dopamine.on_workspace_entry();
}
```
