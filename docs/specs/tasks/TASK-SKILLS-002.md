# TASK-SKILLS-002: Create SubagentDefinition Type

```xml
<task_spec id="TASK-SKILLS-002" version="1.0">
<metadata>
  <title>Create SubagentDefinition and Related Subagent Types</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>2</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-11</requirement_ref>
    <requirement_ref>REQ-SKILLS-12</requirement_ref>
    <requirement_ref>REQ-SKILLS-20</requirement_ref>
  </implements>
  <depends_on>
    <!-- No dependencies on other SKILLS tasks -->
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.5</estimated_hours>
</metadata>

<context>
This task creates the data types for subagent definitions. Subagents are isolated context
agents spawned via Claude Code's Task tool. Each subagent is defined in a markdown file
at .claude/agents/{name}.md. Types include SubagentDefinition (parsed agent file),
SubagentModel, ContextGraphSubagent enum (our 4 custom subagents), BuiltinSubagentType
(Claude's 3 built-in types), and TaskToolParams for Task tool invocation.

Constitution Reference: claude_code.subagents
Technical Spec Reference: TECH-SKILLS Section 2.2
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-2.2</file>
  <file purpose="functional_spec">docs/specs/functional/SPEC-SKILLS.md#requirements</file>
  <file purpose="skills_module">crates/context-graph-cli/src/skills/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-001 completed (skills module exists)</check>
  <check>serde, serde_json are workspace dependencies</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SubagentDefinition struct
    - Create SubagentModel enum
    - Create BuiltinSubagentType enum
    - Create ContextGraphSubagent enum with 4 variants
    - Create TaskToolParams struct
    - Create SubagentSpawnResult struct
    - Add unit tests for all types
  </in_scope>
  <out_of_scope>
    - SubagentError type (TASK-SKILLS-004)
    - SubagentSpawner implementation (TASK-SKILLS-010)
    - Actual agent markdown files (TASK-SKILLS-014)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/subagent_types.rs">
/// Subagent definition loaded from .claude/agents/*.md
/// Implements REQ-SKILLS-11, REQ-SKILLS-12
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentDefinition {
    pub name: String,
    pub description: String,
    pub model: SubagentModel,
    pub tools: Vec&lt;String&gt;,
    pub instructions: String,
    pub definition_path: PathBuf,
}

/// Model selection for subagents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SubagentModel {
    Haiku,
    #[default]
    Sonnet,
    Opus,
}

/// Built-in subagent types (Task tool defaults)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinSubagentType {
    Explore,
    Plan,
    GeneralPurpose,
}

/// Context Graph custom subagent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContextGraphSubagent {
    IdentityGuardian,
    MemorySpecialist,
    ConsciousnessExplorer,
    DreamAgent,
}

/// Task tool invocation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskToolParams {
    pub prompt: String,
    pub subagent_type: String,
    pub description: String,
    pub model: Option&lt;SubagentModel&gt;,
    pub run_in_background: Option&lt;bool&gt;,
    pub resume: Option&lt;String&gt;,
}

/// Result of spawning a subagent
#[derive(Debug, Clone)]
pub struct SubagentSpawnResult {
    pub success: bool,
    pub subagent_type: String,
    pub summary: Option&lt;String&gt;,
    pub duration_ms: u64,
    pub background: bool,
    pub error: Option&lt;String&gt;,
}
    </signature>
  </signatures>
  <constraints>
    - SubagentModel default is Sonnet (most capable balanced option)
    - ContextGraphSubagent uses kebab-case serialization
    - memory-specialist uses Haiku (fast), others use Sonnet
    - NO any type anywhere
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli subagent
  </verification>
</definition_of_done>

<pseudo_code>
1. Create subagent_types.rs in skills module

2. Create SubagentModel enum:
   - Haiku (fast, for memory-specialist)
   - Sonnet (default, balanced)
   - Opus (most capable)
   - Implement Default -> Sonnet

3. Create BuiltinSubagentType enum:
   - Explore (haiku, read-only)
   - Plan (sonnet, read-only)
   - GeneralPurpose (sonnet, read/write)
   - Implement model() and is_read_only() methods

4. Create ContextGraphSubagent enum:
   - IdentityGuardian (sonnet)
   - MemorySpecialist (haiku, target <500ms)
   - ConsciousnessExplorer (sonnet)
   - DreamAgent (sonnet)
   - Implement model(), definition_filename(), target_latency_ms()

5. Create SubagentDefinition struct:
   - name: String
   - description: String
   - model: SubagentModel
   - tools: Vec&lt;String&gt;
   - instructions: String
   - definition_path: PathBuf

6. Create TaskToolParams struct:
   - prompt: String (required)
   - subagent_type: String (required)
   - description: String (required)
   - model: Option&lt;SubagentModel&gt;
   - run_in_background: Option&lt;bool&gt;
   - resume: Option&lt;String&gt;

7. Create SubagentSpawnResult struct:
   - success: bool
   - subagent_type: String
   - summary: Option&lt;String&gt;
   - duration_ms: u64
   - background: bool
   - error: Option&lt;String&gt;

8. Add tests:
   - test_subagent_model_default
   - test_context_graph_subagent_models
   - test_context_graph_subagent_filenames
   - test_builtin_subagent_readonly
   - test_task_tool_params_serialization
   - test_subagent_spawn_result
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/subagent_types.rs">SubagentDefinition, SubagentModel, ContextGraphSubagent, TaskToolParams types</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Add pub mod subagent_types;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli subagent</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Context Graph Subagent Configuration

| Subagent | Model | Target Latency | Purpose |
|----------|-------|----------------|---------|
| identity-guardian | Sonnet | - | IC monitoring and protection |
| memory-specialist | Haiku | <500ms | Fast memory operations |
| consciousness-explorer | Sonnet | - | GWT debugging |
| dream-agent | Sonnet | - | NREM/REM execution |

### Built-in Subagent Types (Claude Code defaults)

| Type | Model | Read-Only | Purpose |
|------|-------|-----------|---------|
| Explore | Haiku | Yes | Fast codebase search |
| Plan | Sonnet | Yes | Analysis and planning |
| GeneralPurpose | Sonnet | No | Complex multi-step tasks |

### TaskToolParams Required Fields

```rust
// Minimum required invocation
TaskToolParams {
    prompt: "Search for authentication patterns".into(),
    subagent_type: "memory-specialist".into(),
    description: "Find auth-related memories".into(),
    model: None,           // Uses subagent default
    run_in_background: None, // Default: false
    resume: None,          // For background task resumption
}
```

## Verification Checklist

- [ ] SubagentModel has 3 variants with Sonnet as default
- [ ] BuiltinSubagentType has 3 variants with correct model() and is_read_only()
- [ ] ContextGraphSubagent has 4 variants with correct model() and definition_filename()
- [ ] MemorySpecialist target_latency_ms returns Some(500)
- [ ] TaskToolParams has all required fields
- [ ] SubagentSpawnResult captures execution outcome
- [ ] All tests pass
