# TASK-P1-003: Session and SessionStatus Types

```xml
<task_spec id="TASK-P1-003" version="1.0">
<metadata>
  <title>Session and SessionStatus Types</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>8</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-06</requirement_ref>
  </implements>
  <depends_on>
    <!-- Can run parallel to P1-001, P1-002 -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Foundation task defining session tracking types. Sessions group memories
and track their lifecycle from start to end.

The Session struct tracks when a session started, ended, its status,
and how many memories were captured during the session.
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#data_models</file>
</input_context_files>

<prerequisites>
  <check>crates/context-graph-core/src/memory directory exists</check>
</prerequisites>

<scope>
  <in_scope>
    - Create Session struct
    - Create SessionStatus enum
    - Derive necessary traits
    - Export from session module
  </in_scope>
  <out_of_scope>
    - SessionManager implementation (TASK-P1-006)
    - Session storage
    - Session lifecycle logic
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/session.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct Session {
          pub id: String,
          pub started_at: DateTime&lt;Utc&gt;,
          pub ended_at: Option&lt;DateTime&lt;Utc&gt;&gt;,
          pub status: SessionStatus,
          pub memory_count: u32,
      }

      #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
      pub enum SessionStatus {
          Active,
          Completed,
          Abandoned,
      }
    </signature>
  </signatures>

  <constraints>
    - Session.id is a UUID string (not Uuid type for simplicity)
    - ended_at is None while session is Active
    - memory_count tracks memories in this session
    - Both types must be Serialize/Deserialize
  </constraints>

  <verification>
    - cargo check passes
    - Types exported from memory module
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/memory/session.rs

use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Status of a memory capture session
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// Session is currently active
    Active,
    /// Session ended normally
    Completed,
    /// Session ended without proper closure
    Abandoned,
}

/// A memory capture session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session identifier (UUID string)
    pub id: String,
    /// When the session started
    pub started_at: DateTime&lt;Utc&gt;,
    /// When the session ended (None if still active)
    pub ended_at: Option&lt;DateTime&lt;Utc&gt;&gt;,
    /// Current session status
    pub status: SessionStatus,
    /// Number of memories captured in this session
    pub memory_count: u32,
}

impl Session {
    /// Create a new active session
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            started_at: Utc::now(),
            ended_at: None,
            status: SessionStatus::Active,
            memory_count: 0,
        }
    }

    /// Create session with specific ID (for restoration)
    pub fn with_id(id: String) -> Self {
        Self {
            id,
            started_at: Utc::now(),
            ended_at: None,
            status: SessionStatus::Active,
            memory_count: 0,
        }
    }

    /// Check if session is currently active
    pub fn is_active(&amp;self) -> bool {
        self.status == SessionStatus::Active
    }

    /// Mark session as completed
    pub fn complete(&amp;mut self) {
        self.ended_at = Some(Utc::now());
        self.status = SessionStatus::Completed;
    }

    /// Mark session as abandoned
    pub fn abandon(&amp;mut self) {
        self.ended_at = Some(Utc::now());
        self.status = SessionStatus::Abandoned;
    }

    /// Increment memory count
    pub fn increment_memory_count(&amp;mut self) {
        self.memory_count += 1;
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/session.rs">Session and SessionStatus types</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/mod.rs">Add pub mod session and re-export types</file>
</files_to_modify>

<validation_criteria>
  <criterion>Session has all 5 fields per spec</criterion>
  <criterion>SessionStatus has 3 variants</criterion>
  <criterion>Both types derive Serialize, Deserialize</criterion>
  <criterion>Session has new(), complete(), abandon() methods</criterion>
  <criterion>cargo check passes</criterion>
</validation_criteria>

<test_commands>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Verify exports">grep -r "pub struct Session" crates/context-graph-core/src/memory/</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create session.rs in memory directory
- [ ] Implement SessionStatus enum
- [ ] Implement Session struct
- [ ] Add helper methods (new, complete, abandon, etc.)
- [ ] Update mod.rs with module declaration and re-exports
- [ ] Verify with cargo check
- [ ] Proceed to TASK-P1-004
