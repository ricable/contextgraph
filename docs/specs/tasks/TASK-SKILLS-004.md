# TASK-SKILLS-004: Create SkillError and SubagentError Enums

```xml
<task_spec id="TASK-SKILLS-004" version="1.0">
<metadata>
  <title>Create SkillError, SubagentError, and TriggerError Enums</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>4</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-01</requirement_ref>
    <requirement_ref>REQ-SKILLS-02</requirement_ref>
    <requirement_ref>REQ-SKILLS-17</requirement_ref>
    <requirement_ref>REQ-SKILLS-18</requirement_ref>
    <requirement_ref>REQ-SKILLS-19</requirement_ref>
  </implements>
  <depends_on>
    <!-- No dependencies - error types are standalone -->
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
</metadata>

<context>
This task creates the error types for the skills and subagents system. SkillError covers
skill loading and validation errors (not found, parse error, missing field, invalid name,
tool restriction violations, path traversal). SubagentError covers subagent spawning errors
(not found, spawn blocked, background MCP blocked, timeout). TriggerError covers auto-invocation
errors (conflicts, blocked). All errors use thiserror for derive macros.

Technical Spec Reference: TECH-SKILLS Section 2.3
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-2.3</file>
  <file purpose="existing_error_pattern">crates/context-graph-cli/src/error.rs</file>
</input_context_files>

<prerequisites>
  <check>thiserror is workspace dependency</check>
  <check>skills module exists</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SkillError enum with 9 variants
    - Create SubagentError enum with 6 variants
    - Create TriggerError enum with 2 variants
    - Implement thiserror Error derive
    - Add unit tests for error messages
  </in_scope>
  <out_of_scope>
    - Error handling logic (in loader/spawner implementations)
    - Error recovery strategies (in higher-level code)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/errors.rs">
use thiserror::Error;

/// Skill system error types
/// Implements error states ERR-SKILLS-01 through ERR-SKILLS-09
#[derive(Debug, Error)]
pub enum SkillError {
    #[error("Skill '{skill_name}' not found at path: {path}")]
    SkillNotFound { skill_name: String, path: String },

    #[error("Failed to parse skill '{skill_name}': Invalid YAML frontmatter. {details}")]
    SkillParseError { skill_name: String, details: String },

    #[error("Skill '{path}' missing required field '{field}'. Required fields: name, description.")]
    MissingRequiredField { path: String, field: String },

    #[error("Invalid skill name '{name}': {reason}")]
    InvalidSkillName { name: String, reason: String },

    #[error("Tool '{tool}' not allowed for skill '{skill}'. Allowed: {allowed:?}")]
    ToolRestrictionViolation { skill: String, tool: String, allowed: Vec&lt;String&gt; },

    #[error("Skill '{skill}' references unknown tool '{tool}'. Tool not available in current MCP configuration.")]
    ToolNotFound { skill: String, tool: String },

    #[error("Resource not found for skill '{skill}': {path}")]
    ResourceNotFound { skill: String, path: String },

    #[error("Security error: Resource path '{path}' attempts traversal outside skill directory.")]
    PathTraversal { path: String },

    #[error("Description for skill '{skill}' exceeds 1024 character limit ({length} chars).")]
    DescriptionTooLong { skill: String, length: usize },
}

/// Subagent error types
#[derive(Debug, Error)]
pub enum SubagentError {
    #[error("Subagent type '{subagent_type}' not found. Expected file: .claude/agents/{subagent_type}.md")]
    SubagentNotFound { subagent_type: String },

    #[error("Subagent '{id}' cannot spawn subagents. Subagent spawning is prohibited to prevent recursion.")]
    SpawnBlocked { id: String },

    #[error("MCP tool '{tool}' unavailable in background mode. Use foreground execution for MCP access.")]
    BackgroundMcpBlocked { tool: String },

    #[error("Failed to spawn subagent '{subagent_type}': {reason}")]
    SpawnFailed { subagent_type: String, reason: String },

    #[error("Subagent '{subagent_type}' timed out after {timeout_ms}ms")]
    ExecutionTimeout { subagent_type: String, timeout_ms: u64 },

    #[error("MCP tool '{tool}' failed: {details}")]
    McpToolFailed { tool: String, details: String },
}

/// Trigger matching error types
#[derive(Debug, Error)]
pub enum TriggerError {
    #[error("Trigger conflict: Skills {skills:?} all matched with confidence {confidence}. User clarification needed.")]
    TriggerMatchConflict { skills: Vec&lt;String&gt;, confidence: f32 },

    #[error("Skill '{skill}' has auto-invocation disabled. Use explicit /{skill} command.")]
    AutoInvocationBlocked { skill: String },
}
    </signature>
  </signatures>
  <constraints>
    - All errors must have descriptive, actionable messages
    - Use thiserror Error derive
    - Include relevant context in error variants
    - Follow existing CLI error patterns
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli skill_error
    - cargo test --package context-graph-cli subagent_error
  </verification>
</definition_of_done>

<pseudo_code>
1. Create errors.rs in skills module

2. Create SkillError enum with thiserror:
   - SkillNotFound { skill_name, path }
   - SkillParseError { skill_name, details }
   - MissingRequiredField { path, field }
   - InvalidSkillName { name, reason }
   - ToolRestrictionViolation { skill, tool, allowed }
   - ToolNotFound { skill, tool }
   - ResourceNotFound { skill, path }
   - PathTraversal { path }
   - DescriptionTooLong { skill, length }

3. Create SubagentError enum with thiserror:
   - SubagentNotFound { subagent_type }
   - SpawnBlocked { id }
   - BackgroundMcpBlocked { tool }
   - SpawnFailed { subagent_type, reason }
   - ExecutionTimeout { subagent_type, timeout_ms }
   - McpToolFailed { tool, details }

4. Create TriggerError enum with thiserror:
   - TriggerMatchConflict { skills, confidence }
   - AutoInvocationBlocked { skill }

5. Add tests:
   - test_skill_not_found_message
   - test_skill_parse_error_message
   - test_missing_field_message
   - test_invalid_name_message
   - test_tool_restriction_message
   - test_path_traversal_message
   - test_subagent_not_found_message
   - test_spawn_blocked_message
   - test_background_mcp_blocked_message
   - test_trigger_conflict_message
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/errors.rs">SkillError, SubagentError, TriggerError enums</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Add pub mod errors;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli skill_error</command>
  <command>cargo test --package context-graph-cli subagent_error</command>
  <command>cargo test --package context-graph-cli trigger_error</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### SkillError Variants and Causes

| Variant | Cause | Example Message |
|---------|-------|-----------------|
| SkillNotFound | Directory/SKILL.md missing | "Skill 'foo' not found at path: .claude/skills/foo" |
| SkillParseError | Invalid YAML frontmatter | "Failed to parse skill 'foo': Invalid YAML frontmatter. Unexpected key 'bar'" |
| MissingRequiredField | name or description missing | "Skill '.claude/skills/foo/SKILL.md' missing required field 'name'" |
| InvalidSkillName | Name validation failed | "Invalid skill name 'Foo_Bar': Must be lowercase with hyphens only" |
| ToolRestrictionViolation | Tool not in allowed list | "Tool 'Write' not allowed for skill 'read-only'. Allowed: [Read, Grep]" |
| ToolNotFound | MCP tool doesn't exist | "Skill 'foo' references unknown tool 'mcp__bar__baz'" |
| ResourceNotFound | Resource file missing | "Resource not found for skill 'foo': references/doc.md" |
| PathTraversal | Escape attempt detected | "Security error: Resource path '../../../etc/passwd' attempts traversal" |
| DescriptionTooLong | >1024 chars | "Description for skill 'verbose' exceeds 1024 character limit (1500 chars)" |

### SubagentError Variants and Causes

| Variant | Cause | Example Message |
|---------|-------|-----------------|
| SubagentNotFound | .md file missing | "Subagent type 'custom' not found. Expected: .claude/agents/custom.md" |
| SpawnBlocked | Subagent spawning subagent | "Subagent 'identity-guardian' cannot spawn subagents. Prohibited." |
| BackgroundMcpBlocked | MCP in background mode | "MCP tool 'get_consciousness_state' unavailable in background mode" |
| SpawnFailed | General spawn failure | "Failed to spawn subagent 'dream-agent': Permission denied" |
| ExecutionTimeout | Subagent took too long | "Subagent 'dream-agent' timed out after 30000ms" |
| McpToolFailed | MCP call error | "MCP tool 'trigger_dream' failed: Database connection lost" |

### TriggerError Variants and Causes

| Variant | Cause | Example Message |
|---------|-------|-----------------|
| TriggerMatchConflict | Multiple equal matches | "Trigger conflict: Skills ['memory-inject', 'semantic-search'] all matched with confidence 0.8" |
| AutoInvocationBlocked | disable-model-invocation: true | "Skill 'admin' has auto-invocation disabled. Use explicit /admin command" |

## Verification Checklist

- [ ] SkillError has 9 variants with descriptive messages
- [ ] SubagentError has 6 variants with descriptive messages
- [ ] TriggerError has 2 variants with descriptive messages
- [ ] All errors include relevant context fields
- [ ] thiserror Error derive works correctly
- [ ] Error Display format is human-readable
- [ ] All tests pass
