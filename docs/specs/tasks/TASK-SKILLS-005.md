# TASK-SKILLS-005: Create ProgressiveDisclosureLevel Enum

```xml
<task_spec id="TASK-SKILLS-005" version="1.0">
<metadata>
  <title>Create ProgressiveDisclosureLevel Enum</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>5</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-03</requirement_ref>
  </implements>
  <depends_on>
    <!-- No dependencies - this is a core type -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <estimated_hours>0.5</estimated_hours>
</metadata>

<context>
This task creates the ProgressiveDisclosureLevel enum that defines the three levels of
skill content loading. Skills use progressive disclosure to minimize context overhead:
Level 1 (Metadata) loads only name + description (~100 tokens), Level 2 (Instructions)
loads the full SKILL.md body (<5k tokens), and Level 3 (Resources) loads bundled files
on-demand (unlimited).

Constitution Reference: claude_code.skills.progressive_disclosure
Technical Spec Reference: TECH-SKILLS Section 2.1
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-1.2</file>
  <file purpose="functional_spec">docs/specs/functional/SPEC-SKILLS.md#req-skills-03</file>
</input_context_files>

<prerequisites>
  <check>skills module exists</check>
  <check>serde is workspace dependency</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ProgressiveDisclosureLevel enum with 3 variants
    - Implement token_budget() method
    - Implement description() method
    - Add serde serialization
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - SkillLoader progressive loading logic (TASK-SKILLS-006)
    - Token counting implementation (handled by estimate_tokens)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/types.rs">
/// Progressive disclosure level for skill content loading
/// Implements REQ-SKILLS-03
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProgressiveDisclosureLevel {
    /// Level 1: name + description only (~100 tokens)
    #[default]
    Metadata,
    /// Level 2: Full SKILL.md body (&lt;5k tokens)
    Instructions,
    /// Level 3: Bundled resources (unlimited)
    Resources,
}

impl ProgressiveDisclosureLevel {
    /// Get maximum token budget for this level
    pub fn token_budget(&self) -> Option&lt;usize&gt;;

    /// Get human-readable description of this level
    pub fn description(&self) -> &'static str;

    /// Check if this level requires the previous level to be loaded
    pub fn requires_previous(&self) -> bool;

    /// Get the previous level, if any
    pub fn previous(&self) -> Option&lt;Self&gt;;
}
    </signature>
  </signatures>
  <constraints>
    - Metadata is the default level (loaded at startup)
    - token_budget returns None for Resources (unlimited)
    - Levels are sequential: Metadata -> Instructions -> Resources
    - Resources requires Instructions loaded first
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli progressive_disclosure
  </verification>
</definition_of_done>

<pseudo_code>
1. Add ProgressiveDisclosureLevel enum to types.rs:
   - Metadata (default, Level 1)
   - Instructions (Level 2)
   - Resources (Level 3)

2. Implement token_budget():
   match self:
       Metadata => Some(100)
       Instructions => Some(5000)
       Resources => None  // Unlimited

3. Implement description():
   match self:
       Metadata => "name + description only (~100 tokens)"
       Instructions => "Full SKILL.md body (<5k tokens)"
       Resources => "Bundled resources (unlimited)"

4. Implement requires_previous():
   match self:
       Metadata => false  // First level
       Instructions => true  // Requires Metadata
       Resources => true  // Requires Instructions

5. Implement previous():
   match self:
       Metadata => None
       Instructions => Some(Metadata)
       Resources => Some(Instructions)

6. Add tests:
   - test_default_is_metadata
   - test_token_budget_metadata
   - test_token_budget_instructions
   - test_token_budget_resources_unlimited
   - test_requires_previous
   - test_previous_level
   - test_serialization
</pseudo_code>

<files_to_create>
  <!-- Additions go to existing types.rs -->
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/types.rs">Add ProgressiveDisclosureLevel enum</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli progressive_disclosure</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Progressive Disclosure Levels

| Level | Content | Token Budget | Loaded When |
|-------|---------|--------------|-------------|
| Metadata | name + description | ~100 | At startup (all skills) |
| Instructions | SKILL.md body | <5000 | On trigger/command |
| Resources | Bundled files | Unlimited | On demand |

### Level Transitions

```
                    Startup
                       |
                       v
    +------------------------------------------+
    |              Level 1: Metadata           |
    |         name + description only          |
    |              (~100 tokens)               |
    +------------------------------------------+
                       |
            Keyword trigger or /skill
                       |
                       v
    +------------------------------------------+
    |           Level 2: Instructions          |
    |            Full SKILL.md body            |
    |              (<5k tokens)                |
    +------------------------------------------+
                       |
              Resource referenced
                       |
                       v
    +------------------------------------------+
    |            Level 3: Resources            |
    |       Bundled files loaded on-demand     |
    |              (unlimited)                 |
    +------------------------------------------+
```

### State Machine Properties

- **Metadata**: Always loaded, minimal overhead
- **Instructions**: Only loaded when skill activates
- **Resources**: Only loaded when explicitly referenced

### Token Budget Enforcement

```rust
// Example usage in SkillLoader
let level = ProgressiveDisclosureLevel::Instructions;
if let Some(budget) = level.token_budget() {
    let tokens = skill.estimate_tokens(level);
    if tokens > budget {
        warn!("Skill {} exceeds token budget: {} > {}",
              skill.frontmatter.name, tokens, budget);
    }
}
```

## Verification Checklist

- [ ] Enum has 3 variants: Metadata, Instructions, Resources
- [ ] Metadata is the default (startup level)
- [ ] token_budget returns Some(100) for Metadata
- [ ] token_budget returns Some(5000) for Instructions
- [ ] token_budget returns None for Resources
- [ ] requires_previous returns false for Metadata
- [ ] previous() returns correct level chain
- [ ] Serde serialization works correctly
- [ ] All tests pass
