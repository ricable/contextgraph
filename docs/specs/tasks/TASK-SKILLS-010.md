# TASK-SKILLS-010: Implement SubagentSpawner via Task Tool

```xml
<task_spec id="TASK-SKILLS-010" version="1.0">
<metadata>
  <title>Implement SubagentSpawner via Task Tool</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>10</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-11</requirement_ref>
    <requirement_ref>REQ-SKILLS-17</requirement_ref>
    <requirement_ref>REQ-SKILLS-18</requirement_ref>
    <requirement_ref>REQ-SKILLS-19</requirement_ref>
    <requirement_ref>REQ-SKILLS-20</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-002</task_ref>
    <task_ref>TASK-SKILLS-004</task_ref>
    <task_ref>TASK-SKILLS-009</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
This task implements the SubagentSpawner that spawns subagents via Claude Code's Task tool.
Subagents are isolated context agents defined in .claude/agents/*.md files. The spawner
enforces constraints: subagents cannot spawn other subagents (prevents recursion), and
background subagents cannot use MCP tools. It also provides spawn helpers for Context Graph's
4 custom subagent types.

Technical Spec Reference: TECH-SKILLS Section 3.3
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-3.3</file>
  <file purpose="subagent_types">crates/context-graph-cli/src/skills/subagent_types.rs</file>
  <file purpose="errors">crates/context-graph-cli/src/skills/errors.rs</file>
  <file purpose="restrictor">crates/context-graph-cli/src/skills/restrictor.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-002 completed (SubagentDefinition, TaskToolParams exist)</check>
  <check>TASK-SKILLS-004 completed (SubagentError exists)</check>
  <check>TASK-SKILLS-009 completed (ToolRestrictor exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SubagentSpawner struct
    - Implement load_definition() to parse agent .md files
    - Implement spawn() for generic subagent spawning
    - Implement spawn_context_graph() for our 4 subagent types
    - Implement subagent context tracking (no-spawn enforcement)
    - Implement MCP tool blocking for background mode
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - Actual Task tool invocation (Claude Code handles this)
    - Agent markdown files (TASK-SKILLS-014)
    - Integration with Task tool output
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/spawner.rs">
use crate::skills::{
    errors::SubagentError,
    subagent_types::{
        ContextGraphSubagent, SubagentDefinition, SubagentModel, SubagentSpawnResult, TaskToolParams,
    },
};
use std::path::PathBuf;

/// Spawns subagents via Task tool
/// Implements REQ-SKILLS-17, REQ-SKILLS-18, REQ-SKILLS-19, REQ-SKILLS-20
pub struct SubagentSpawner {
    agents_dir: PathBuf,
    current_subagent: Option&lt;String&gt;,
}

impl SubagentSpawner {
    /// Create new spawner for given agents directory
    pub fn new(agents_dir: impl Into&lt;PathBuf&gt;) -> Self;

    /// Load subagent definition from file
    pub fn load_definition(&self, subagent_type: &str) -> Result&lt;SubagentDefinition, SubagentError&gt;;

    /// Spawn a subagent with given parameters
    pub fn spawn(&mut self, params: TaskToolParams) -> Result&lt;SubagentSpawnResult, SubagentError&gt;;

    /// Spawn a Context Graph subagent by type
    pub fn spawn_context_graph(
        &mut self,
        subagent: ContextGraphSubagent,
        prompt: &str,
        background: bool,
    ) -> Result&lt;SubagentSpawnResult, SubagentError&gt;;

    /// Check if currently executing within a subagent context
    pub fn is_in_subagent_context(&self) -> bool;

    /// Set current subagent context
    pub fn enter_subagent_context(&mut self, subagent_type: &str);

    /// Clear current subagent context
    pub fn exit_subagent_context(&mut self);

    /// Check if MCP tools allowed for current context
    pub fn is_mcp_allowed(&self, tool_name: &str, background: bool) -> bool;

    /// Get list of read-only tools allowed in background mode
    pub fn background_allowed_tools() -> Vec&lt;&'static str&gt;;

    /// Parse agent markdown file into SubagentDefinition
    fn parse_agent_markdown(&self, content: &str, name: &str, path: PathBuf) -> Result&lt;SubagentDefinition, SubagentError&gt;;
}
    </signature>
  </signatures>
  <constraints>
    - Subagents CANNOT spawn other subagents (SpawnBlocked error)
    - Background subagents CANNOT use MCP tools (BackgroundMcpBlocked error)
    - Read, Grep, Glob always allowed in background
    - Agent files parsed for ## sections: Role, Model, Allowed Tools, Protocol
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli subagent_spawner
  </verification>
</definition_of_done>

<pseudo_code>
1. Create spawner.rs in skills module

2. Implement SubagentSpawner::new():
   fn new(agents_dir: impl Into&lt;PathBuf&gt;) -> Self:
       Self {
           agents_dir: agents_dir.into(),
           current_subagent: None,
       }

3. Implement load_definition():
   fn load_definition(&self, subagent_type: &str) -> Result&lt;SubagentDefinition, SubagentError&gt;:
       let path = self.agents_dir.join(format!("{}.md", subagent_type));
       if !path.exists():
           return Err(SubagentError::SubagentNotFound { subagent_type: subagent_type.into() });

       let content = std::fs::read_to_string(&path)?;
       self.parse_agent_markdown(&content, subagent_type, path)

4. Implement parse_agent_markdown():
   fn parse_agent_markdown(&self, content: &str, name: &str, path: PathBuf) -> Result&lt;...&gt;:
       // Parse ## sections
       let mut model = SubagentModel::Sonnet;  // default
       let mut tools = Vec::new();
       let mut description = String::new();

       for line in content.lines():
           if line.starts_with("## Model"):
               // Parse next line for model
           if line.starts_with("## Allowed Tools"):
               // Parse tool list until next ##
           if line.starts_with("## Role"):
               // Parse description until next ##

       Ok(SubagentDefinition {
           name: name.to_string(),
           description,
           model,
           tools,
           instructions: content.to_string(),
           definition_path: path,
       })

5. Implement spawn():
   fn spawn(&mut self, params: TaskToolParams) -> Result&lt;SubagentSpawnResult, SubagentError&gt;:
       // 1. Check not in subagent context
       if self.is_in_subagent_context():
           return Err(SubagentError::SpawnBlocked {
               id: self.current_subagent.clone().unwrap(),
           });

       // 2. Load definition
       let definition = self.load_definition(&params.subagent_type)?;

       // 3. Check background MCP constraint
       let background = params.run_in_background.unwrap_or(false);
       if background:
           for tool in &definition.tools:
               if tool.starts_with("mcp__"):
                   return Err(SubagentError::BackgroundMcpBlocked { tool: tool.clone() });

       // 4. Build result (actual spawn is done by Claude Code Task tool)
       Ok(SubagentSpawnResult {
           success: true,
           subagent_type: params.subagent_type,
           summary: None,  // Filled by Task tool response
           duration_ms: 0,  // Measured by Task tool
           background,
           error: None,
       })

6. Implement spawn_context_graph():
   fn spawn_context_graph(
       &mut self,
       subagent: ContextGraphSubagent,
       prompt: &str,
       background: bool,
   ) -> Result&lt;SubagentSpawnResult, SubagentError&gt;:
       let params = TaskToolParams {
           prompt: prompt.to_string(),
           subagent_type: subagent.definition_filename().trim_end_matches(".md").to_string(),
           description: format!("{} subagent", subagent.definition_filename()),
           model: Some(subagent.model()),
           run_in_background: Some(background),
           resume: None,
       };
       self.spawn(params)

7. Implement context tracking:
   fn is_in_subagent_context(&self) -> bool:
       self.current_subagent.is_some()

   fn enter_subagent_context(&mut self, subagent_type: &str):
       self.current_subagent = Some(subagent_type.to_string());

   fn exit_subagent_context(&mut self):
       self.current_subagent = None;

8. Implement MCP checking:
   fn is_mcp_allowed(&self, tool_name: &str, background: bool) -> bool:
       if background && tool_name.starts_with("mcp__"):
           return false;
       true

   fn background_allowed_tools() -> Vec&lt;&'static str&gt;:
       vec!["Read", "Grep", "Glob"]

9. Add tests:
   - test_load_definition_success
   - test_load_definition_not_found
   - test_spawn_blocked_in_subagent_context
   - test_spawn_background_mcp_blocked
   - test_spawn_context_graph_identity_guardian
   - test_spawn_context_graph_memory_specialist
   - test_is_mcp_allowed
   - test_background_allowed_tools
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/spawner.rs">SubagentSpawner implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Add pub mod spawner;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli subagent_spawner</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Agent Markdown Format

```markdown
# Identity Guardian Subagent

## Role
Monitors identity continuity and triggers dreams on crisis.

## Model
sonnet

## Allowed Tools
- mcp__context-graph__get_identity_continuity
- mcp__context-graph__get_ego_state
- mcp__context-graph__trigger_dream
- Read

## Protocol
1. Check IC at start of task
2. Monitor after each memory operation
3. Trigger dream if IC < 0.5 (critical)
4. Report IC changes > 0.1

## Thresholds
| Status | Range | Color | Action |
|--------|-------|-------|--------|
| healthy | >= 0.9 | Green | Continue |
| warning | 0.7-0.9 | Yellow | Monitor |
| degraded | 0.5-0.7 | Orange | Alert |
| critical | < 0.5 | Red | TRIGGER DREAM |

## Output Format
Return summary with IC status, changes detected, and actions taken.
```

### Context Graph Subagents

| Subagent | Model | File | Target Latency |
|----------|-------|------|----------------|
| identity-guardian | Sonnet | identity-guardian.md | - |
| memory-specialist | Haiku | memory-specialist.md | <500ms |
| consciousness-explorer | Sonnet | consciousness-explorer.md | - |
| dream-agent | Sonnet | dream-agent.md | - |

### Spawn Constraints

```rust
// 1. Subagent cannot spawn subagent
spawner.enter_subagent_context("identity-guardian");
spawner.spawn(...) // Err(SpawnBlocked)

// 2. Background cannot use MCP
spawner.spawn(TaskToolParams {
    run_in_background: Some(true),
    ...
}) // Err(BackgroundMcpBlocked) if agent has MCP tools
```

### Integration with Task Tool

```rust
// This spawner prepares the parameters and validates constraints.
// Actual Task tool invocation happens in Claude Code:

// 1. Validate and prepare
let result = spawner.spawn(params)?;

// 2. Claude Code invokes Task tool with:
Task({
    prompt: params.prompt,
    subagent_type: params.subagent_type,
    description: params.description,
    model: params.model,
    run_in_background: params.run_in_background,
})

// 3. Result summary returned to spawner
result.summary = task_tool_response.summary;
result.duration_ms = task_tool_response.duration_ms;
```

## Verification Checklist

- [ ] SubagentSpawner loads agent .md files correctly
- [ ] parse_agent_markdown extracts Role, Model, Allowed Tools
- [ ] spawn() blocks when in subagent context
- [ ] spawn() blocks background mode with MCP tools
- [ ] spawn_context_graph() uses correct model per subagent type
- [ ] enter_subagent_context/exit_subagent_context work
- [ ] is_mcp_allowed returns false for background MCP
- [ ] background_allowed_tools returns Read, Grep, Glob
- [ ] All tests pass
