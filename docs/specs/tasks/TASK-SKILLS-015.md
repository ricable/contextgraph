# TASK-SKILLS-015: Create Integration Tests for Skills System

```xml
<task_spec id="TASK-SKILLS-015" version="1.0">
<metadata>
  <title>Create Integration Tests for Skills System</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>15</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-28</requirement_ref>
    <requirement_ref>REQ-SKILLS-29</requirement_ref>
    <requirement_ref>REQ-SKILLS-30</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-006</task_ref>
    <task_ref>TASK-SKILLS-007</task_ref>
    <task_ref>TASK-SKILLS-008</task_ref>
    <task_ref>TASK-SKILLS-009</task_ref>
    <task_ref>TASK-SKILLS-010</task_ref>
    <task_ref>TASK-SKILLS-011</task_ref>
    <task_ref>TASK-SKILLS-012</task_ref>
    <task_ref>TASK-SKILLS-013</task_ref>
    <task_ref>TASK-SKILLS-014</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>2.5</estimated_hours>
</metadata>

<context>
This task creates integration tests for the skills system, testing the interaction
between SkillLoader, SkillRegistry, TriggerMatcher, ToolRestrictor, and SubagentSpawner.
Tests use real SKILL.md files and agent markdown files created in previous tasks.
NO mock data - all tests use actual fixture files.

Technical Spec Reference: TECH-SKILLS Section 5
</context>

<input_context_files>
  <file purpose="loader">crates/context-graph-cli/src/skills/loader.rs</file>
  <file purpose="registry">crates/context-graph-cli/src/skills/registry.rs</file>
  <file purpose="matcher">crates/context-graph-cli/src/skills/matcher.rs</file>
  <file purpose="restrictor">crates/context-graph-cli/src/skills/restrictor.rs</file>
  <file purpose="spawner">crates/context-graph-cli/src/skills/spawner.rs</file>
  <file purpose="skills">.claude/skills/</file>
  <file purpose="agents">.claude/agents/</file>
</input_context_files>

<prerequisites>
  <check>All logic layer tasks completed (006-010)</check>
  <check>All skill SKILL.md files created (011-013)</check>
  <check>All subagent files created (014)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create integration test module
    - Test SkillLoader with real SKILL.md files
    - Test SkillRegistry discovery with real directories
    - Test TriggerMatcher with extracted keywords
    - Test ToolRestrictor with skill restrictions
    - Test SubagentSpawner with real agent files
    - Test full workflow: discover -> trigger -> load -> restrict
    - Use real fixture files, NO mocks
  </in_scope>
  <out_of_scope>
    - E2E tests with Claude Code Task tool (TASK-SKILLS-016)
    - MCP tool invocation tests
    - Performance benchmarks
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/tests/skills_integration.rs">
use context_graph_cli::skills::{
    loader::SkillLoader,
    registry::{SkillRegistry, SkillSource},
    matcher::TriggerMatcher,
    restrictor::ToolRestrictor,
    spawner::SubagentSpawner,
    types::{ProgressiveDisclosureLevel, ContextGraphSubagent},
    errors::{SkillError, SubagentError},
};
use std::path::PathBuf;

/// Test fixture directory
fn fixture_dir() -> PathBuf;

/// Test SkillLoader with real consciousness skill
#[test]
fn test_loader_consciousness_skill();

/// Test SkillLoader with all 5 skills
#[test]
fn test_loader_all_skills();

/// Test SkillRegistry discovery
#[test]
fn test_registry_discover_all();

/// Test SkillRegistry precedence
#[test]
fn test_registry_precedence();

/// Test TriggerMatcher with real keywords
#[test]
fn test_matcher_consciousness_trigger();

/// Test TriggerMatcher conflict detection
#[test]
fn test_matcher_conflict_detection();

/// Test ToolRestrictor with consciousness skill
#[test]
fn test_restrictor_consciousness_tools();

/// Test ToolRestrictor background mode
#[test]
fn test_restrictor_background_mode();

/// Test SubagentSpawner with identity-guardian
#[test]
fn test_spawner_identity_guardian();

/// Test SubagentSpawner spawn blocking
#[test]
fn test_spawner_spawn_blocked();

/// Test SubagentSpawner background MCP blocking
#[test]
fn test_spawner_background_mcp_blocked();

/// Test full workflow: discover -> trigger -> load -> restrict
#[test]
fn test_full_skill_workflow();

/// Test full subagent workflow
#[test]
fn test_full_subagent_workflow();
    </signature>
  </signatures>
  <constraints>
    - NO mock data - use real fixture files
    - Tests must be deterministic
    - Tests must clean up any created state
    - Tests must not depend on external services
    - All tests must pass with --test-threads=1
  </constraints>
  <verification>
    - cargo test --package context-graph-cli --test skills_integration
    - cargo test --package context-graph-cli --test skills_integration -- --test-threads=1
  </verification>
</definition_of_done>

<pseudo_code>
1. Create tests/skills_integration.rs

2. Create fixture helper:
   fn fixture_dir() -> PathBuf:
       PathBuf::from(env!("CARGO_MANIFEST_DIR"))
           .join("../../.claude")

3. Implement test_loader_consciousness_skill():
   let loader = SkillLoader::new(fixture_dir().join("skills"))?;
   let result = loader.load_metadata("consciousness")?;
   assert!(result.skill.is_some());
   assert_eq!(result.level, ProgressiveDisclosureLevel::Metadata);
   let skill = result.skill.unwrap();
   assert_eq!(skill.frontmatter.name, "consciousness");
   assert!(skill.keywords.contains(&"consciousness".to_string()));

4. Implement test_loader_all_skills():
   let loader = SkillLoader::new(fixture_dir().join("skills"))?;
   for name in ["consciousness", "memory-inject", "semantic-search",
                "dream-consolidation", "curation"]:
       let result = loader.load_metadata(name)?;
       assert!(result.skill.is_some());

5. Implement test_registry_discover_all():
   let registry = SkillRegistry::new(fixture_dir().parent().unwrap())?;
   let count = registry.discover_all()?;
   assert!(count >= 5); // At least our 5 skills

6. Implement test_matcher_consciousness_trigger():
   let registry = SkillRegistry::new(fixture_dir().parent().unwrap())?;
   registry.discover_all()?;
   let mut matcher = TriggerMatcher::new();
   let skills = registry.list_all().iter()
       .filter_map(|n| registry.get(n))
       .cloned()
       .collect();
   matcher.register_triggers(&skills);

   let result = matcher.match_input("check consciousness state")?;
   assert_eq!(result, Some("consciousness".to_string()));

7. Implement test_restrictor_consciousness_tools():
   let loader = SkillLoader::new(fixture_dir().join("skills"))?;
   let result = loader.load_metadata("consciousness")?;
   let skill = result.skill.unwrap();

   let mut restrictor = ToolRestrictor::new();
   restrictor.apply_skill(&skill);

   // Allowed tools should pass
   assert!(restrictor.check("mcp__context-graph__get_consciousness_state").is_ok());
   assert!(restrictor.check("Read").is_ok());

   // Non-allowed tools should fail
   assert!(restrictor.check("Write").is_err());

8. Implement test_spawner_identity_guardian():
   let spawner = SubagentSpawner::new(fixture_dir().join("agents"));
   let definition = spawner.load_definition("identity-guardian")?;
   assert_eq!(definition.name, "identity-guardian");
   assert_eq!(definition.model, SubagentModel::Sonnet);

9. Implement test_spawner_spawn_blocked():
   let mut spawner = SubagentSpawner::new(fixture_dir().join("agents"));
   spawner.enter_subagent_context("identity-guardian");

   let result = spawner.spawn(TaskToolParams { ... });
   assert!(matches!(result, Err(SubagentError::SpawnBlocked { .. })));

10. Implement test_full_skill_workflow():
    // 1. Create registry and discover
    let registry = SkillRegistry::new(fixture_dir().parent().unwrap())?;
    registry.discover_all()?;

    // 2. Create matcher and register triggers
    let mut matcher = TriggerMatcher::new();
    matcher.register_triggers(&...);

    // 3. Match user input
    let skill_name = matcher.match_input("check consciousness")?.unwrap();

    // 4. Load full instructions
    let result = registry.load_instructions(&skill_name)?;
    assert!(result.instructions.is_some());

    // 5. Apply restrictions
    let mut restrictor = ToolRestrictor::new();
    restrictor.apply_skill(result.skill.as_ref().unwrap());

    // 6. Verify tool allowed
    assert!(restrictor.check("mcp__context-graph__get_consciousness_state").is_ok());
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/tests/skills_integration.rs">Integration tests</file>
</files_to_create>

<test_commands>
  <command>cargo test --package context-graph-cli --test skills_integration</command>
  <command>cargo test --package context-graph-cli --test skills_integration -- --test-threads=1</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| Loader | 2 | Verify SKILL.md parsing |
| Registry | 2 | Verify discovery and precedence |
| Matcher | 2 | Verify trigger matching |
| Restrictor | 2 | Verify tool restrictions |
| Spawner | 3 | Verify subagent spawning |
| Workflow | 2 | Verify full integration |

### Fixture Directory Structure

```
.claude/
├── skills/
│   ├── consciousness/
│   │   └── SKILL.md
│   ├── memory-inject/
│   │   └── SKILL.md
│   ├── semantic-search/
│   │   └── SKILL.md
│   ├── dream-consolidation/
│   │   └── SKILL.md
│   └── curation/
│       └── SKILL.md
└── agents/
    ├── identity-guardian.md
    ├── memory-specialist.md
    ├── consciousness-explorer.md
    └── dream-agent.md
```

### NO Mock Data Rule

All tests use real files:
- Real SKILL.md files from .claude/skills/
- Real agent files from .claude/agents/
- Real frontmatter parsing
- Real keyword extraction

### Test Isolation

```rust
// Each test creates its own instances
// No shared state between tests
// Tests can run in any order
// Tests clean up any created files
```

### Error Testing

```rust
// Test expected errors
#[test]
fn test_spawner_spawn_blocked() {
    let mut spawner = SubagentSpawner::new(...);
    spawner.enter_subagent_context("identity-guardian");

    let result = spawner.spawn(...);
    assert!(matches!(
        result,
        Err(SubagentError::SpawnBlocked { id }) if id == "identity-guardian"
    ));
}
```

## Verification Checklist

- [ ] All tests use real fixture files
- [ ] No mock data in any test
- [ ] Tests are deterministic
- [ ] Tests clean up state
- [ ] Tests run with --test-threads=1
- [ ] Loader tests verify all 5 skills
- [ ] Registry tests verify discovery
- [ ] Matcher tests verify triggering
- [ ] Restrictor tests verify restrictions
- [ ] Spawner tests verify subagent loading
- [ ] Workflow tests verify full integration
- [ ] All tests pass
