# TASK-P1-001: Memory Struct and MemorySource Enum

```xml
<task_spec id="TASK-P1-001" version="1.0">
<metadata>
  <title>Memory Struct and MemorySource Enum</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>6</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-01</requirement_ref>
    <requirement_ref>REQ-P1-02</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P0-005</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Foundation task for the Memory Capture system. Creates the core Memory struct
and MemorySource enum that all subsequent memory-related tasks depend on.

The Memory struct is the primary data unit. The MemorySource enum discriminates
between hook descriptions, Claude responses, and MD file chunks.
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#data_models</file>
  <file purpose="existing_structure">crates/context-graph-core/src/lib.rs</file>
</input_context_files>

<prerequisites>
  <check>Phase 0 complete (North Star removed)</check>
  <check>crates/context-graph-core exists</check>
  <check>uuid and chrono crates available</check>
</prerequisites>

<scope>
  <in_scope>
    - Create Memory struct with all fields
    - Create MemorySource enum with variants
    - Create HookType enum
    - Create ResponseType enum
    - Derive Serialize, Deserialize for storage
    - Add module to crate lib.rs
  </in_scope>
  <out_of_scope>
    - ChunkMetadata (TASK-P1-002)
    - Session types (TASK-P1-003)
    - TeleologicalArray reference (Phase 2)
    - Storage implementation (TASK-P1-005)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/mod.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct Memory {
          pub id: Uuid,
          pub content: String,
          pub source: MemorySource,
          pub created_at: DateTime&lt;Utc&gt;,
          pub session_id: String,
          pub teleological_array: TeleologicalArray,
          pub chunk_metadata: Option&lt;ChunkMetadata&gt;,
          pub word_count: u32,
      }
    </signature>
    <signature file="crates/context-graph-core/src/memory/source.rs">
      #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
      pub enum MemorySource {
          HookDescription { hook_type: HookType, tool_name: Option&lt;String&gt; },
          ClaudeResponse { response_type: ResponseType },
          MDFileChunk { file_path: String, chunk_index: u32, total_chunks: u32 },
      }

      #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
      pub enum HookType {
          SessionStart,
          UserPromptSubmit,
          PreToolUse,
          PostToolUse,
          Stop,
          SessionEnd,
      }

      #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
      pub enum ResponseType {
          SessionSummary,
          StopResponse,
          SignificantResponse,
      }
    </signature>
  </signatures>

  <constraints>
    - Use Uuid from uuid crate
    - Use DateTime&lt;Utc&gt; from chrono crate
    - All types must derive Serialize, Deserialize
    - TeleologicalArray can be a placeholder type for now
    - ChunkMetadata can be forward-declared or use Option
  </constraints>

  <verification>
    - cargo check --package context-graph-core succeeds
    - All types are exported from memory module
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/memory/mod.rs

pub mod source;

// Re-exports
pub use source::{MemorySource, HookType, ResponseType};

// Forward declare or import from Phase 2
use crate::embedding::TeleologicalArray;  // Placeholder if Phase 2 not done
use crate::memory::chunker::ChunkMetadata; // Forward ref

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Uuid,
    pub content: String,
    pub source: MemorySource,
    pub created_at: DateTime&lt;Utc&gt;,
    pub session_id: String,
    pub teleological_array: TeleologicalArray,
    pub chunk_metadata: Option&lt;ChunkMetadata&gt;,
    pub word_count: u32,
}

impl Memory {
    pub fn new(
        content: String,
        source: MemorySource,
        session_id: String,
        teleological_array: TeleologicalArray,
        chunk_metadata: Option&lt;ChunkMetadata&gt;,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.clone(),
            source,
            created_at: Utc::now(),
            session_id,
            teleological_array,
            chunk_metadata,
            word_count: content.split_whitespace().count() as u32,
        }
    }
}

---
File: crates/context-graph-core/src/memory/source.rs

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemorySource {
    HookDescription { hook_type: HookType, tool_name: Option&lt;String&gt; },
    ClaudeResponse { response_type: ResponseType },
    MDFileChunk { file_path: String, chunk_index: u32, total_chunks: u32 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum HookType {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    Stop,
    SessionEnd,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ResponseType {
    SessionSummary,
    StopResponse,
    SignificantResponse,
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/mod.rs">Memory struct and module re-exports</file>
  <file path="crates/context-graph-core/src/memory/source.rs">MemorySource, HookType, ResponseType enums</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/lib.rs">Add pub mod memory declaration</file>
</files_to_modify>

<validation_criteria>
  <criterion>Memory struct has all 8 fields per spec</criterion>
  <criterion>MemorySource has 3 variants with correct fields</criterion>
  <criterion>HookType has 6 variants</criterion>
  <criterion>ResponseType has 3 variants</criterion>
  <criterion>All types derive Serialize, Deserialize</criterion>
  <criterion>cargo check passes</criterion>
</validation_criteria>

<test_commands>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Verify exports">grep -r "pub struct Memory" crates/context-graph-core/src/memory/</command>
</test_commands>

<notes>
  <note category="placeholder_types">
    TeleologicalArray may not exist yet (Phase 2). Create a placeholder:
    pub type TeleologicalArray = Vec&lt;f32&gt;;  // Placeholder until Phase 2
    This allows Phase 1 to proceed independently.
  </note>
  <note category="forward_reference">
    ChunkMetadata is defined in P1-002. Use forward declaration or
    define a minimal version here and expand in P1-002.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Create memory directory in crates/context-graph-core/src/
- [ ] Create source.rs with MemorySource, HookType, ResponseType
- [ ] Create mod.rs with Memory struct
- [ ] Add placeholder for TeleologicalArray if needed
- [ ] Add pub mod memory to lib.rs
- [ ] Verify with cargo check
- [ ] Proceed to TASK-P1-002
