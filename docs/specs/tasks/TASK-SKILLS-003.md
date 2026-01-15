# TASK-SKILLS-003: Create SkillLoadResult and Related Types

```xml
<task_spec id="TASK-SKILLS-003" version="1.0">
<metadata>
  <title>Create SkillLoadResult and SkillTrigger Types</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>3</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-03</requirement_ref>
    <requirement_ref>REQ-SKILLS-10</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-001</task_ref>
    <task_ref>TASK-SKILLS-005</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
</metadata>

<context>
This task creates the result types for skill loading operations. SkillLoadResult captures
the outcome of loading a skill at a given disclosure level, including the skill definition,
loaded instructions, tool restrictions, and any errors. SkillTrigger defines the configuration
for auto-invoking skills based on keyword matching.

Technical Spec Reference: TECH-SKILLS Section 2.1
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-2.1</file>
  <file purpose="skills_types">crates/context-graph-cli/src/skills/types.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-001 completed (SkillDefinition exists)</check>
  <check>TASK-SKILLS-005 completed (ProgressiveDisclosureLevel exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SkillLoadResult struct
    - Create SkillTrigger struct
    - Add estimate_tokens method to SkillDefinition
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - SkillLoader implementation (TASK-SKILLS-006)
    - TriggerMatcher implementation (TASK-SKILLS-008)
    - SkillRegistry implementation (TASK-SKILLS-007)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/types.rs">
/// Result of loading a skill
/// Implements REQ-SKILLS-03
#[derive(Debug, Clone)]
pub struct SkillLoadResult {
    /// Whether skill was successfully loaded
    pub loaded: bool,
    /// Current disclosure level
    pub level: ProgressiveDisclosureLevel,
    /// Skill definition (if loaded)
    pub skill: Option&lt;SkillDefinition&gt;,
    /// Instructions text (Level 2+)
    pub instructions: Option&lt;String&gt;,
    /// Tool restrictions parsed from frontmatter
    pub tool_restrictions: HashSet&lt;String&gt;,
    /// Error message if loading failed
    pub error: Option&lt;String&gt;,
}

/// Trigger configuration for auto-invoking skills
/// Implements US-SKILLS-01 through US-SKILLS-05
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTrigger {
    /// Skill name to trigger
    pub skill_name: String,
    /// Keywords that trigger this skill
    pub keywords: Vec&lt;String&gt;,
    /// Minimum confidence score for trigger (0.0-1.0)
    pub min_confidence: f32,
    /// Whether user must confirm before loading
    pub require_confirmation: bool,
}

impl SkillDefinition {
    /// Estimate token count for current disclosure level
    pub fn estimate_tokens(&self, level: ProgressiveDisclosureLevel) -> usize;
}

impl SkillLoadResult {
    /// Create successful result
    pub fn success(skill: SkillDefinition, level: ProgressiveDisclosureLevel) -> Self;

    /// Create failure result
    pub fn failure(error: String) -> Self;
}
    </signature>
  </signatures>
  <constraints>
    - Token estimation: ~4 chars per token
    - Level 1 (Metadata): name + description only, ~100 tokens
    - Level 2 (Instructions): body content, <5k tokens
    - Level 3 (Resources): unlimited, 0 returned (cannot estimate)
    - Default min_confidence: 0.5
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli skill_load_result
    - cargo test --package context-graph-cli skill_trigger
  </verification>
</definition_of_done>

<pseudo_code>
1. Add SkillLoadResult struct to types.rs:
   - loaded: bool
   - level: ProgressiveDisclosureLevel
   - skill: Option&lt;SkillDefinition&gt;
   - instructions: Option&lt;String&gt;
   - tool_restrictions: HashSet&lt;String&gt;
   - error: Option&lt;String&gt;

2. Implement SkillLoadResult constructors:
   fn success(skill, level) -> Self:
       Self {
           loaded: true,
           level,
           tool_restrictions: skill.allowed_tools_set.clone(),
           skill: Some(skill),
           instructions: None, // Set by loader for Level 2+
           error: None,
       }

   fn failure(error: String) -> Self:
       Self {
           loaded: false,
           level: ProgressiveDisclosureLevel::Metadata,
           skill: None,
           instructions: None,
           tool_restrictions: HashSet::new(),
           error: Some(error),
       }

3. Add SkillTrigger struct:
   - skill_name: String
   - keywords: Vec&lt;String&gt;
   - min_confidence: f32 (default 0.5)
   - require_confirmation: bool (default false)

4. Implement Default for SkillTrigger:
   fn default() -> Self:
       Self {
           skill_name: String::new(),
           keywords: Vec::new(),
           min_confidence: 0.5,
           require_confirmation: false,
       }

5. Implement estimate_tokens on SkillDefinition:
   fn estimate_tokens(&self, level: ProgressiveDisclosureLevel) -> usize:
       match level:
           Metadata => (name.len() + description.len()) / 4
           Instructions => body.len() / 4
           Resources => 0  // Cannot estimate, varies by resource

6. Add tests:
   - test_skill_load_result_success
   - test_skill_load_result_failure
   - test_skill_trigger_default
   - test_estimate_tokens_metadata
   - test_estimate_tokens_instructions
   - test_estimate_tokens_resources_zero
</pseudo_code>

<files_to_create>
  <!-- All additions go to existing types.rs -->
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/types.rs">Add SkillLoadResult, SkillTrigger, and estimate_tokens</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli skill_load</command>
  <command>cargo test --package context-graph-cli skill_trigger</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Token Estimation Rules

| Level | Content | Calculation | Target |
|-------|---------|-------------|--------|
| Metadata | name + description | (name.len + desc.len) / 4 | ~100 tokens |
| Instructions | SKILL.md body | body.len / 4 | <5000 tokens |
| Resources | External files | 0 (cannot estimate) | Unlimited |

### SkillTrigger Configuration Examples

```rust
// Consciousness skill trigger
SkillTrigger {
    skill_name: "consciousness".into(),
    keywords: vec![
        "consciousness".into(),
        "awareness".into(),
        "identity".into(),
        "coherence".into(),
        "kuramoto".into(),
        "GWT".into(),
    ],
    min_confidence: 0.5,
    require_confirmation: false,
}

// Memory-inject skill trigger
SkillTrigger {
    skill_name: "memory-inject".into(),
    keywords: vec![
        "memory".into(),
        "context".into(),
        "inject".into(),
        "retrieve".into(),
        "recall".into(),
        "background".into(),
    ],
    min_confidence: 0.5,
    require_confirmation: false,
}
```

### SkillLoadResult States

| State | loaded | skill | error |
|-------|--------|-------|-------|
| Success | true | Some(_) | None |
| Not Found | false | None | Some("Skill not found...") |
| Parse Error | false | None | Some("Invalid YAML...") |
| Missing Field | false | None | Some("Missing required field...") |

## Verification Checklist

- [ ] SkillLoadResult has all required fields
- [ ] SkillLoadResult::success correctly initializes with skill
- [ ] SkillLoadResult::failure correctly sets error
- [ ] SkillTrigger has default min_confidence of 0.5
- [ ] estimate_tokens returns ~100 for metadata level
- [ ] estimate_tokens returns body.len/4 for instructions
- [ ] estimate_tokens returns 0 for resources
- [ ] All tests pass
