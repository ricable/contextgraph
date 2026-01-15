# TASK-SKILLS-016: Create E2E Tests for Skill Invocation

```xml
<task_spec id="TASK-SKILLS-016" version="1.0">
<metadata>
  <title>Create E2E Tests for Skill Invocation</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>16</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-31</requirement_ref>
    <requirement_ref>REQ-SKILLS-32</requirement_ref>
    <requirement_ref>REQ-SKILLS-33</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-015</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
This task creates end-to-end tests that verify the complete skill invocation flow
from user input to tool restriction enforcement. Tests simulate the UserPromptSubmit
hook flow, skill activation, and PreToolUse hook enforcement. These tests validate
that the skills system works correctly in a production-like environment.

Technical Spec Reference: TECH-SKILLS Section 5
</context>

<input_context_files>
  <file purpose="integration_tests">crates/context-graph-cli/tests/skills_integration.rs</file>
  <file purpose="loader">crates/context-graph-cli/src/skills/loader.rs</file>
  <file purpose="registry">crates/context-graph-cli/src/skills/registry.rs</file>
  <file purpose="matcher">crates/context-graph-cli/src/skills/matcher.rs</file>
  <file purpose="restrictor">crates/context-graph-cli/src/skills/restrictor.rs</file>
  <file purpose="spawner">crates/context-graph-cli/src/skills/spawner.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-015 completed (integration tests exist)</check>
  <check>All skill and agent files created</check>
</prerequisites>

<scope>
  <in_scope>
    - Create E2E test module
    - Test UserPromptSubmit simulation (trigger detection)
    - Test skill activation and instruction loading
    - Test PreToolUse simulation (restriction enforcement)
    - Test subagent spawn simulation
    - Test background mode restrictions
    - Test error handling and recovery
    - Verify token budgets at each level
  </in_scope>
  <out_of_scope>
    - Actual Claude Code hook integration
    - Actual MCP tool invocation
    - Performance benchmarking (separate task)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/tests/skills_e2e.rs">
use context_graph_cli::skills::{
    loader::SkillLoader,
    registry::{SkillRegistry, SkillSource},
    matcher::TriggerMatcher,
    restrictor::ToolRestrictor,
    spawner::SubagentSpawner,
    types::{ProgressiveDisclosureLevel, ContextGraphSubagent, TaskToolParams},
    errors::{SkillError, SubagentError, TriggerError},
};
use std::path::PathBuf;

/// Simulates full skill invocation flow
struct SkillsE2EHarness {
    registry: SkillRegistry,
    matcher: TriggerMatcher,
    restrictor: ToolRestrictor,
    spawner: SubagentSpawner,
}

impl SkillsE2EHarness {
    /// Create new test harness
    fn new() -> Result&lt;Self, SkillError&gt;;

    /// Simulate UserPromptSubmit hook
    fn on_user_prompt(&mut self, prompt: &str) -> Result&lt;Option&lt;String&gt;, TriggerError&gt;;

    /// Simulate skill activation
    fn activate_skill(&mut self, name: &str) -> Result&lt;String, SkillError&gt;;

    /// Simulate PreToolUse hook
    fn on_pre_tool_use(&self, tool_name: &str) -> Result&lt;(), SkillError&gt;;

    /// Simulate subagent spawn
    fn spawn_subagent(&mut self, params: TaskToolParams) -> Result&lt;(), SubagentError&gt;;

    /// Deactivate current skill
    fn deactivate_skill(&mut self);
}

/// Test: User prompt triggers consciousness skill
#[test]
fn test_e2e_consciousness_skill_trigger();

/// Test: User prompt triggers memory skill
#[test]
fn test_e2e_memory_skill_trigger();

/// Test: Skill activation loads instructions
#[test]
fn test_e2e_skill_activation();

/// Test: Tool restriction blocks disallowed tools
#[test]
fn test_e2e_tool_restriction();

/// Test: Subagent spawn blocked in subagent context
#[test]
fn test_e2e_subagent_spawn_blocked();

/// Test: Background mode blocks MCP tools
#[test]
fn test_e2e_background_mcp_blocked();

/// Test: Token budgets respected at each level
#[test]
fn test_e2e_token_budgets();

/// Test: Error recovery flow
#[test]
fn test_e2e_error_recovery();

/// Test: Complete consciousness workflow
#[test]
fn test_e2e_complete_consciousness_workflow();

/// Test: Complete memory workflow
#[test]
fn test_e2e_complete_memory_workflow();

/// Test: Complete dream workflow
#[test]
fn test_e2e_complete_dream_workflow();
    </signature>
  </signatures>
  <constraints>
    - NO mock data - use real fixture files
    - Simulate hooks, don't require Claude Code
    - Tests must be self-contained
    - Tests must verify complete workflows
    - All tests must pass with --test-threads=1
  </constraints>
  <verification>
    - cargo test --package context-graph-cli --test skills_e2e
    - cargo test --package context-graph-cli --test skills_e2e -- --test-threads=1
  </verification>
</definition_of_done>

<pseudo_code>
1. Create tests/skills_e2e.rs

2. Implement SkillsE2EHarness:
   struct SkillsE2EHarness {
       registry: SkillRegistry,
       matcher: TriggerMatcher,
       restrictor: ToolRestrictor,
       spawner: SubagentSpawner,
   }

   impl SkillsE2EHarness {
       fn new() -> Result&lt;Self, SkillError&gt; {
           let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
               .join("../../.claude");

           let mut registry = SkillRegistry::new(fixture_dir.parent().unwrap())?;
           registry.discover_all()?;

           let mut matcher = TriggerMatcher::new();
           matcher.register_triggers(&registry.get_triggers());

           let restrictor = ToolRestrictor::new();
           let spawner = SubagentSpawner::new(fixture_dir.join("agents"));

           Ok(Self { registry, matcher, restrictor, spawner })
       }

       fn on_user_prompt(&mut self, prompt: &str) -> Result&lt;Option&lt;String&gt;, TriggerError&gt; {
           self.matcher.match_input(prompt)
       }

       fn activate_skill(&mut self, name: &str) -> Result&lt;String, SkillError&gt; {
           let result = self.registry.load_instructions(name)?;
           if let Some(skill) = &result.skill {
               self.restrictor.apply_skill(skill);
           }
           result.instructions.ok_or_else(|| SkillError::SkillNotFound { ... })
       }

       fn on_pre_tool_use(&self, tool_name: &str) -> Result&lt;(), SkillError&gt; {
           self.restrictor.check(tool_name)
       }

       fn spawn_subagent(&mut self, params: TaskToolParams) -> Result&lt;(), SubagentError&gt; {
           self.spawner.spawn(params).map(|_| ())
       }

       fn deactivate_skill(&mut self) {
           self.restrictor.clear();
       }
   }

3. Implement test_e2e_consciousness_skill_trigger():
   let mut harness = SkillsE2EHarness::new()?;

   // User asks about consciousness
   let skill = harness.on_user_prompt("check my consciousness state")?;
   assert_eq!(skill, Some("consciousness".to_string()));

   // Activate the skill
   let instructions = harness.activate_skill("consciousness")?;
   assert!(instructions.contains("## Overview"));

   // Verify allowed tool
   assert!(harness.on_pre_tool_use("mcp__context-graph__get_consciousness_state").is_ok());

   // Verify blocked tool
   assert!(harness.on_pre_tool_use("Write").is_err());

4. Implement test_e2e_tool_restriction():
   let mut harness = SkillsE2EHarness::new()?;

   // No skill active - all tools allowed
   assert!(harness.on_pre_tool_use("Write").is_ok());

   // Activate consciousness skill
   harness.activate_skill("consciousness")?;

   // Consciousness tools allowed
   assert!(harness.on_pre_tool_use("mcp__context-graph__get_consciousness_state").is_ok());
   assert!(harness.on_pre_tool_use("Read").is_ok());

   // Other tools blocked
   assert!(harness.on_pre_tool_use("Write").is_err());
   assert!(harness.on_pre_tool_use("mcp__context-graph__inject_memory").is_err());

   // Deactivate - all tools allowed again
   harness.deactivate_skill();
   assert!(harness.on_pre_tool_use("Write").is_ok());

5. Implement test_e2e_subagent_spawn_blocked():
   let mut harness = SkillsE2EHarness::new()?;

   // First spawn succeeds
   let params = TaskToolParams {
       prompt: "check IC".to_string(),
       subagent_type: "identity-guardian".to_string(),
       ...
   };
   assert!(harness.spawn_subagent(params.clone()).is_ok());

   // Enter subagent context
   harness.spawner.enter_subagent_context("identity-guardian");

   // Second spawn blocked
   assert!(matches!(
       harness.spawn_subagent(params),
       Err(SubagentError::SpawnBlocked { .. })
   ));

6. Implement test_e2e_token_budgets():
   let mut harness = SkillsE2EHarness::new()?;

   // Level 1: Metadata only (~100 tokens)
   let level1 = harness.registry.get("consciousness").unwrap();
   assert!(level1.estimate_tokens() < 200);

   // Level 2: Full instructions (<5k tokens)
   let result = harness.registry.load_instructions("consciousness")?;
   let instructions = result.instructions.unwrap();
   let estimated = instructions.len() / 4; // ~4 chars per token
   assert!(estimated < 5000);

7. Implement test_e2e_complete_consciousness_workflow():
   let mut harness = SkillsE2EHarness::new()?;

   // 1. User prompt triggers skill
   let skill = harness.on_user_prompt("what is my identity continuity?")?;
   assert_eq!(skill, Some("consciousness".to_string()));

   // 2. Skill activates
   let instructions = harness.activate_skill("consciousness")?;
   assert!(instructions.contains("Identity Continuity"));

   // 3. Tool calls are validated
   assert!(harness.on_pre_tool_use("mcp__context-graph__get_identity_continuity").is_ok());

   // 4. If IC low, spawn identity guardian
   let params = TaskToolParams {
       prompt: "monitor IC and alert on changes".to_string(),
       subagent_type: "identity-guardian".to_string(),
       description: "IC monitoring".to_string(),
       model: Some(SubagentModel::Sonnet),
       run_in_background: Some(true),
       resume: None,
   };

   // Background with MCP should fail
   assert!(matches!(
       harness.spawn_subagent(params),
       Err(SubagentError::BackgroundMcpBlocked { .. })
   ));

   // 5. Skill deactivates
   harness.deactivate_skill();
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/tests/skills_e2e.rs">E2E tests</file>
</files_to_create>

<test_commands>
  <command>cargo test --package context-graph-cli --test skills_e2e</command>
  <command>cargo test --package context-graph-cli --test skills_e2e -- --test-threads=1</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Test Harness Design

The SkillsE2EHarness simulates the Claude Code hook integration:

```rust
struct SkillsE2EHarness {
    registry: SkillRegistry,    // Skill discovery and loading
    matcher: TriggerMatcher,    // User prompt matching
    restrictor: ToolRestrictor, // Tool enforcement
    spawner: SubagentSpawner,   // Subagent management
}
```

### Hook Simulation

| Hook | Method | Purpose |
|------|--------|---------|
| UserPromptSubmit | on_user_prompt() | Detect skill triggers |
| (skill activation) | activate_skill() | Load instructions, apply restrictions |
| PreToolUse | on_pre_tool_use() | Enforce tool restrictions |
| (subagent) | spawn_subagent() | Spawn subagent with validation |

### Complete Workflow Tests

Each workflow test covers:

1. **Trigger**: User prompt triggers skill
2. **Activation**: Skill instructions loaded
3. **Restriction**: Tools validated
4. **Subagent** (optional): Spawn subagent
5. **Deactivation**: Skill deactivated

### Token Budget Verification

```rust
// Level 1: ~100 tokens (metadata only)
// Level 2: <5,000 tokens (full instructions)
// Level 3: unlimited (resources on demand)

fn estimate_tokens(text: &str) -> usize {
    text.len() / 4 // ~4 chars per token approximation
}
```

### Error Scenarios Tested

| Error | Test | Expected |
|-------|------|----------|
| SpawnBlocked | test_e2e_subagent_spawn_blocked | Subagent cannot spawn |
| BackgroundMcpBlocked | test_e2e_background_mcp_blocked | MCP tools blocked |
| ToolRestrictionViolation | test_e2e_tool_restriction | Non-allowed tool blocked |
| TriggerMatchConflict | (in integration tests) | Equal confidence conflict |

## Verification Checklist

- [ ] SkillsE2EHarness compiles and initializes
- [ ] on_user_prompt() correctly triggers skills
- [ ] activate_skill() loads instructions and applies restrictions
- [ ] on_pre_tool_use() enforces restrictions
- [ ] spawn_subagent() validates constraints
- [ ] Token budgets verified at each level
- [ ] All workflow tests pass
- [ ] Error recovery tests pass
- [ ] Tests run with --test-threads=1
- [ ] No mock data used
