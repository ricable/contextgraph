# TASK-SKILLS-009: Implement ToolRestrictor for MCP Tool Filtering

```xml
<task_spec id="TASK-SKILLS-009" version="1.0">
<metadata>
  <title>Implement ToolRestrictor for MCP Tool Filtering</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>9</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-02</requirement_ref>
    <requirement_ref>REQ-SKILLS-19</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-001</task_ref>
    <task_ref>TASK-SKILLS-004</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.5</estimated_hours>
</metadata>

<context>
This task implements the ToolRestrictor that enforces tool restrictions for skills and
subagents. Skills define allowed tools via the allowed-tools frontmatter field. Subagents
running in background mode have MCP tools blocked (only Read, Grep, Glob allowed).
The restrictor provides check() method that returns success or ToolRestrictionViolation.

Technical Spec Reference: TECH-SKILLS Section 3.4
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-3.4</file>
  <file purpose="types">crates/context-graph-cli/src/skills/types.rs</file>
  <file purpose="errors">crates/context-graph-cli/src/skills/errors.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-001 completed (SkillDefinition with allowed_tools_set)</check>
  <check>TASK-SKILLS-004 completed (ToolRestrictionViolation error)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ToolRestrictor struct
    - Implement apply_skill() for skill restrictions
    - Implement apply_background_mode() for subagent restrictions
    - Implement check() for tool validation
    - Implement is_mcp_tool() helper
    - Implement is_read_only_tool() helper
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - Actual MCP tool invocation interception
    - PreToolUse hook integration (Phase 3)
    - SubagentSpawner (TASK-SKILLS-010)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/restrictor.rs">
use crate::skills::{
    errors::SkillError,
    types::SkillDefinition,
};
use std::collections::HashSet;

/// Enforces tool restrictions per skill/subagent
pub struct ToolRestrictor {
    active_restrictions: HashSet&lt;String&gt;,
    active_skill: Option&lt;String&gt;,
    background_mode: bool,
}

impl ToolRestrictor {
    /// Create new restrictor with no active restrictions
    pub fn new() -> Self;

    /// Apply restrictions from skill definition
    pub fn apply_skill(&mut self, skill: &SkillDefinition);

    /// Apply restrictions for subagent in background mode
    pub fn apply_background_mode(&mut self);

    /// Clear all restrictions
    pub fn clear(&mut self);

    /// Check if tool invocation is allowed
    pub fn check(&self, tool_name: &str) -> Result&lt;(), SkillError&gt;;

    /// Check if tool is MCP tool
    pub fn is_mcp_tool(tool_name: &str) -> bool;

    /// Check if tool is read-only (allowed in background)
    pub fn is_read_only_tool(tool_name: &str) -> bool;

    /// Get current active restrictions
    pub fn active_restrictions(&self) -> &HashSet&lt;String&gt;;

    /// Check if any restrictions are active
    pub fn has_restrictions(&self) -> bool;

    /// Check if in background mode
    pub fn is_background_mode(&self) -> bool;
}

impl Default for ToolRestrictor {
    fn default() -> Self { Self::new() }
}
    </signature>
  </signatures>
  <constraints>
    - Empty allowed-tools means all tools allowed
    - Background mode blocks all MCP tools
    - Read, Grep, Glob always allowed in background
    - Tool names matched exactly or via scope prefix
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli tool_restrictor
  </verification>
</definition_of_done>

<pseudo_code>
1. Create restrictor.rs in skills module

2. Implement ToolRestrictor::new():
   fn new() -> Self:
       Self {
           active_restrictions: HashSet::new(),
           active_skill: None,
           background_mode: false,
       }

3. Implement apply_skill():
   fn apply_skill(&mut self, skill: &SkillDefinition):
       self.active_skill = Some(skill.frontmatter.name.clone());
       self.active_restrictions = skill.allowed_tools_set.clone();
       self.background_mode = false;

4. Implement apply_background_mode():
   fn apply_background_mode(&mut self):
       self.background_mode = true;
       // In background mode, only read-only tools allowed
       self.active_restrictions = ["Read", "Grep", "Glob"]
           .iter()
           .map(|s| s.to_string())
           .collect();

5. Implement clear():
   fn clear(&mut self):
       self.active_restrictions.clear();
       self.active_skill = None;
       self.background_mode = false;

6. Implement check():
   fn check(&self, tool_name: &str) -> Result&lt;(), SkillError&gt;:
       // No restrictions = all allowed
       if self.active_restrictions.is_empty() && !self.background_mode:
           return Ok(());

       // Background mode: block MCP tools
       if self.background_mode && Self::is_mcp_tool(tool_name):
           return Err(SkillError::ToolRestrictionViolation {
               skill: self.active_skill.clone().unwrap_or_default(),
               tool: tool_name.to_string(),
               allowed: self.active_restrictions.iter().cloned().collect(),
           });

       // Check exact match
       if self.active_restrictions.contains(tool_name):
           return Ok(());

       // Check scoped match (e.g., Bash(git:*) matches Bash(git:status))
       for allowed in &self.active_restrictions:
           if allowed.contains('(') && allowed.contains('*'):
               let prefix = allowed.split('(').next().unwrap_or("");
               if tool_name.starts_with(prefix) && tool_name.contains('('):
                   return Ok(());

       // Tool not allowed
       Err(SkillError::ToolRestrictionViolation {
           skill: self.active_skill.clone().unwrap_or("(none)".to_string()),
           tool: tool_name.to_string(),
           allowed: self.active_restrictions.iter().cloned().collect(),
       })

7. Implement helper methods:
   fn is_mcp_tool(tool_name: &str) -> bool:
       tool_name.starts_with("mcp__")

   fn is_read_only_tool(tool_name: &str) -> bool:
       matches!(tool_name, "Read" | "Grep" | "Glob")

8. Add tests:
   - test_no_restrictions_allows_all
   - test_skill_restrictions_enforced
   - test_scoped_bash_matching
   - test_background_mode_blocks_mcp
   - test_background_mode_allows_readonly
   - test_clear_removes_restrictions
   - test_is_mcp_tool
   - test_is_read_only_tool
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/restrictor.rs">ToolRestrictor implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Add pub mod restrictor;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli tool_restrictor</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Tool Restriction Matrix

| Tool Category | Skills (no restriction) | Skills (with allowed-tools) | Subagent (FG) | Subagent (BG) |
|---------------|------------------------|----------------------------|---------------|---------------|
| MCP Tools | Allowed | Per allowed-tools | Per definition | BLOCKED |
| Read | Allowed | Per allowed-tools | Allowed | Allowed |
| Grep | Allowed | Per allowed-tools | Per definition | Allowed |
| Glob | Allowed | Per allowed-tools | Per definition | Allowed |
| Write | Allowed | Per allowed-tools | Per definition | BLOCKED |
| Edit | Allowed | Per allowed-tools | Per definition | BLOCKED |
| Bash | Allowed | Per allowed-tools | Per definition | BLOCKED |
| Task | Allowed | Allowed | BLOCKED | BLOCKED |

### Scoped Tool Matching

```rust
// allowed-tools: "Bash(git:*)"

restrictor.check("Bash(git:status)")  // Ok - matches prefix
restrictor.check("Bash(git:commit)")  // Ok - matches prefix
restrictor.check("Bash(rm:-rf)")      // Err - different prefix
restrictor.check("Bash")              // Err - not scoped
```

### MCP Tool Detection

```rust
// MCP tools follow pattern: mcp__{server}__{tool}
ToolRestrictor::is_mcp_tool("mcp__context-graph__get_consciousness_state") // true
ToolRestrictor::is_mcp_tool("mcp__memory__store")                          // true
ToolRestrictor::is_mcp_tool("Read")                                         // false
ToolRestrictor::is_mcp_tool("Bash")                                         // false
```

### Background Mode Behavior

```rust
// Subagent spawned with run_in_background: true
restrictor.apply_background_mode();

restrictor.check("Read")                                    // Ok
restrictor.check("Grep")                                    // Ok
restrictor.check("Glob")                                    // Ok
restrictor.check("Write")                                   // Err - not read-only
restrictor.check("mcp__context-graph__get_consciousness")   // Err - MCP blocked
```

### Usage Flow

```rust
// 1. Create restrictor
let mut restrictor = ToolRestrictor::new();

// 2. Apply skill restrictions when skill activates
let skill = registry.get("consciousness")?;
restrictor.apply_skill(&skill);

// 3. Check before each tool call (PreToolUse hook)
match restrictor.check("mcp__context-graph__get_consciousness_state") {
    Ok(()) => {
        // Proceed with tool call
    }
    Err(SkillError::ToolRestrictionViolation { skill, tool, allowed }) => {
        // Block tool call, return error to LLM
        eprintln!("Tool {} not allowed for skill {}", tool, skill);
    }
}

// 4. Clear when skill deactivates
restrictor.clear();
```

## Verification Checklist

- [ ] No restrictions allows all tools
- [ ] Skill restrictions enforce allowed-tools
- [ ] Scoped Bash matching works (Bash(git:*))
- [ ] Background mode blocks MCP tools
- [ ] Background mode allows Read, Grep, Glob
- [ ] is_mcp_tool correctly identifies MCP tools
- [ ] is_read_only_tool identifies Read, Grep, Glob
- [ ] clear() removes all restrictions
- [ ] Error includes skill name, tool, and allowed list
- [ ] All tests pass
