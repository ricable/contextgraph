# TASK-HOOKS-001: Create HookEventType Enum

```xml
<task_spec id="TASK-HOOKS-001" version="1.0">
<metadata>
  <title>Create HookEventType Enum</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>1</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-01</requirement_ref>
    <requirement_ref>REQ-HOOKS-02</requirement_ref>
    <requirement_ref>REQ-HOOKS-03</requirement_ref>
    <requirement_ref>REQ-HOOKS-04</requirement_ref>
    <requirement_ref>REQ-HOOKS-05</requirement_ref>
  </implements>
  <depends_on>
    <!-- No dependencies - first foundation task -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <estimated_hours>0.5</estimated_hours>
</metadata>

<context>
This task creates the HookEventType enum that defines all five Claude Code native hook events.
This is a foundational type that all hook handling code depends on. The enum defines:
- SessionStart (5000ms timeout)
- PreToolUse (100ms timeout - FAST PATH)
- PostToolUse (3000ms timeout)
- UserPromptSubmit (2000ms timeout)
- SessionEnd (30000ms timeout)

Constitution Reference: IDENTITY-002 (timeouts must respect Claude Code limits)
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#section-2.2</file>
  <file purpose="existing_cli_structure">crates/context-graph-cli/src/commands/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>context-graph-cli crate exists</check>
  <check>serde, serde_json are workspace dependencies</check>
</prerequisites>

<scope>
  <in_scope>
    - Create HookEventType enum with 5 variants
    - Implement timeout_ms() const method
    - Add serde serialization with snake_case rename
    - Add comprehensive documentation
    - Create unit tests
  </in_scope>
  <out_of_scope>
    - HookInput/HookOutput structs (TASK-HOOKS-002)
    - HookPayload variants (TASK-HOOKS-003)
    - CLI argument types (TASK-HOOKS-004)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/types.rs">
/// Hook event types matching Claude Code native hooks
/// Implements REQ-HOOKS-01 through REQ-HOOKS-05
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEventType {
    /// Session initialization (timeout: 5000ms)
    SessionStart,
    /// Before tool execution (timeout: 100ms) - FAST PATH
    PreToolUse,
    /// After tool execution (timeout: 3000ms)
    PostToolUse,
    /// User prompt submitted (timeout: 2000ms)
    UserPromptSubmit,
    /// Session termination (timeout: 30000ms)
    SessionEnd,
}

impl HookEventType {
    /// Get timeout in milliseconds for this hook type
    /// Constitution Reference: IDENTITY-002
    pub const fn timeout_ms(&self) -> u64;
}
    </signature>
  </signatures>
  <constraints>
    - Enum variants MUST match Claude Code hook names exactly
    - Timeouts MUST match specification: 5000, 100, 3000, 2000, 30000
    - serde rename_all snake_case for JSON serialization
    - NO additional dependencies beyond serde
  </constraints>
  <verification>
    - cargo test --package context-graph-cli hook_event_type
    - cargo doc shows HookEventType with all 5 variants documented
  </verification>
</definition_of_done>

<pseudo_code>
1. Create directory structure:
   crates/context-graph-cli/src/commands/hooks/

2. Create types.rs with HookEventType enum:
   - SessionStart variant
   - PreToolUse variant
   - PostToolUse variant
   - UserPromptSubmit variant
   - SessionEnd variant

3. Implement timeout_ms() const method:
   match self:
     PreToolUse => 100       // Fast path - no DB access
     UserPromptSubmit => 2000
     PostToolUse => 3000
     SessionStart => 5000
     SessionEnd => 30000

4. Add tests:
   - test_timeout_values: verify each timeout matches spec
   - test_serialization: verify snake_case JSON serialization
   - test_all_variants_defined: verify 5 variants exist
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/types.rs">HookEventType enum and related types (start of module)</file>
</files_to_create>

<files_to_modify>
  <!-- None for this task - module registration happens in TASK-HOOKS-011 -->
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli hook_event_type</command>
  <command>cargo doc --package context-graph-cli --no-deps</command>
</test_commands>
</task_spec>
```

## Implementation

### Step 1: Create Directory

```bash
mkdir -p crates/context-graph-cli/src/commands/hooks
```

### Step 2: Create types.rs

```rust
// crates/context-graph-cli/src/commands/hooks/types.rs
//! Hook types for Claude Code native integration
//!
//! # Architecture
//! This module defines the data types for hook input/output that match
//! Claude Code's native hook system specification.
//!
//! # Constitution References
//! - IDENTITY-002: IC thresholds and timeout requirements
//! - GWT-003: Identity continuity tracking
//! - AP-25: Kuramoto N=13

use serde::{Deserialize, Serialize};

/// Hook event types matching Claude Code native hooks
/// Implements REQ-HOOKS-01 through REQ-HOOKS-05
///
/// # Timeouts
/// Each hook type has a specific timeout defined by Claude Code:
/// - `SessionStart`: 5000ms - Session initialization
/// - `PreToolUse`: 100ms - FAST PATH, must not block tool execution
/// - `PostToolUse`: 3000ms - Post-execution identity verification
/// - `UserPromptSubmit`: 2000ms - Context injection
/// - `SessionEnd`: 30000ms - Final persistence and cleanup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEventType {
    /// Session initialization (timeout: 5000ms)
    /// Triggered: startup, resume, /clear
    SessionStart,

    /// Before tool execution (timeout: 100ms) - FAST PATH
    /// CRITICAL: Must not access database, uses cached state only
    PreToolUse,

    /// After tool execution (timeout: 3000ms)
    /// Updates IC and trajectory based on tool result
    PostToolUse,

    /// User prompt submitted (timeout: 2000ms)
    /// Injects relevant context from memory
    UserPromptSubmit,

    /// Session termination (timeout: 30000ms)
    /// Persists final snapshot and optional consolidation
    SessionEnd,
}

impl HookEventType {
    /// Get timeout in milliseconds for this hook type
    /// Constitution Reference: IDENTITY-002
    ///
    /// # Returns
    /// Timeout value in milliseconds
    ///
    /// # Performance Note
    /// PreToolUse has the strictest timeout (100ms) and must use
    /// cached state only - no database access allowed.
    #[inline]
    pub const fn timeout_ms(&self) -> u64 {
        match self {
            Self::PreToolUse => 100,        // Fast path - no DB access
            Self::UserPromptSubmit => 2000, // Context injection
            Self::PostToolUse => 3000,      // IC update + trajectory
            Self::SessionStart => 5000,     // Load/create snapshot
            Self::SessionEnd => 30000,      // Final persist + consolidation
        }
    }

    /// Check if this hook type is time-critical (fast path)
    ///
    /// # Returns
    /// true if timeout is under 500ms
    #[inline]
    pub const fn is_fast_path(&self) -> bool {
        self.timeout_ms() < 500
    }

    /// Get human-readable description of this hook type
    pub const fn description(&self) -> &'static str {
        match self {
            Self::SessionStart => "Session initialization and identity restoration",
            Self::PreToolUse => "Pre-tool consciousness brief injection",
            Self::PostToolUse => "Post-tool identity continuity verification",
            Self::UserPromptSubmit => "User prompt context injection",
            Self::SessionEnd => "Session persistence and consolidation",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_values_match_spec() {
        // Verify timeouts match TECH-HOOKS.md specification exactly
        assert_eq!(HookEventType::SessionStart.timeout_ms(), 5000);
        assert_eq!(HookEventType::PreToolUse.timeout_ms(), 100);
        assert_eq!(HookEventType::PostToolUse.timeout_ms(), 3000);
        assert_eq!(HookEventType::UserPromptSubmit.timeout_ms(), 2000);
        assert_eq!(HookEventType::SessionEnd.timeout_ms(), 30000);
    }

    #[test]
    fn test_serialization_snake_case() {
        // Verify snake_case serialization for Claude Code compatibility
        let json = serde_json::to_string(&HookEventType::SessionStart).unwrap();
        assert_eq!(json, r#""session_start""#);

        let json = serde_json::to_string(&HookEventType::PreToolUse).unwrap();
        assert_eq!(json, r#""pre_tool_use""#);

        let json = serde_json::to_string(&HookEventType::PostToolUse).unwrap();
        assert_eq!(json, r#""post_tool_use""#);

        let json = serde_json::to_string(&HookEventType::UserPromptSubmit).unwrap();
        assert_eq!(json, r#""user_prompt_submit""#);

        let json = serde_json::to_string(&HookEventType::SessionEnd).unwrap();
        assert_eq!(json, r#""session_end""#);
    }

    #[test]
    fn test_deserialization_snake_case() {
        let hook: HookEventType = serde_json::from_str(r#""session_start""#).unwrap();
        assert_eq!(hook, HookEventType::SessionStart);

        let hook: HookEventType = serde_json::from_str(r#""pre_tool_use""#).unwrap();
        assert_eq!(hook, HookEventType::PreToolUse);
    }

    #[test]
    fn test_all_five_variants_exist() {
        // Verify all 5 hook types are defined
        let variants = [
            HookEventType::SessionStart,
            HookEventType::PreToolUse,
            HookEventType::PostToolUse,
            HookEventType::UserPromptSubmit,
            HookEventType::SessionEnd,
        ];
        assert_eq!(variants.len(), 5);
    }

    #[test]
    fn test_fast_path_detection() {
        // Only PreToolUse should be fast path (< 500ms)
        assert!(HookEventType::PreToolUse.is_fast_path());
        assert!(!HookEventType::SessionStart.is_fast_path());
        assert!(!HookEventType::PostToolUse.is_fast_path());
        assert!(!HookEventType::UserPromptSubmit.is_fast_path());
        assert!(!HookEventType::SessionEnd.is_fast_path());
    }

    #[test]
    fn test_copy_clone_traits() {
        let hook = HookEventType::SessionStart;
        let copied = hook; // Copy
        let cloned = hook.clone();
        assert_eq!(hook, copied);
        assert_eq!(hook, cloned);
    }
}
```

## Verification Checklist

- [ ] HookEventType enum has exactly 5 variants
- [ ] timeout_ms() returns correct values per TECH-HOOKS.md
- [ ] JSON serialization uses snake_case
- [ ] All tests pass
- [ ] Documentation visible in cargo doc
