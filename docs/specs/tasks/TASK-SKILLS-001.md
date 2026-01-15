# TASK-SKILLS-001: Create SkillDefinition and SkillFrontmatter Types

```xml
<task_spec id="TASK-SKILLS-001" version="1.0">
<metadata>
  <title>Create SkillDefinition and SkillFrontmatter Types</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>1</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-01</requirement_ref>
    <requirement_ref>REQ-SKILLS-02</requirement_ref>
  </implements>
  <depends_on>
    <!-- No dependencies - first foundation task -->
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.5</estimated_hours>
</metadata>

<context>
This task creates the core data types for skill definitions. Skills are YAML-defined prompt
extensions in SKILL.md files. The SkillFrontmatter struct captures the YAML header fields
(name, description, allowed-tools, model, etc.), while SkillDefinition holds the complete
parsed skill including frontmatter, body, base directory, and extracted keywords.

Constitution Reference: claude_code.skills
Technical Spec Reference: TECH-SKILLS Section 2.1
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-2.1</file>
  <file purpose="functional_spec">docs/specs/functional/SPEC-SKILLS.md#requirements</file>
  <file purpose="existing_cli_structure">crates/context-graph-cli/src/lib.rs</file>
</input_context_files>

<prerequisites>
  <check>context-graph-cli crate exists</check>
  <check>serde, serde_json, serde_yaml are workspace dependencies</check>
</prerequisites>

<scope>
  <in_scope>
    - Create skills module directory structure
    - Create SkillFrontmatter struct with YAML parsing
    - Create SkillModel enum (haiku, sonnet, opus, inherit)
    - Create SkillDefinition struct with helper methods
    - Add serde serialization/deserialization
    - Create unit tests for all types
  </in_scope>
  <out_of_scope>
    - ProgressiveDisclosureLevel enum (TASK-SKILLS-005)
    - SkillLoadResult type (TASK-SKILLS-003)
    - SkillError types (TASK-SKILLS-004)
    - SkillLoader implementation (TASK-SKILLS-006)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/types.rs">
/// YAML frontmatter structure for SKILL.md files
/// Implements REQ-SKILLS-02
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillFrontmatter {
    /// Skill name (max 64 chars, lowercase with hyphens)
    pub name: String,
    /// Skill description (max 1024 chars, WHAT/WHEN/keywords format)
    pub description: String,
    /// Comma-separated list of allowed tools (optional)
    #[serde(rename = "allowed-tools")]
    pub allowed_tools: Option&lt;String&gt;,
    /// Model to use: haiku|sonnet|opus|inherit (optional)
    pub model: Option&lt;SkillModel&gt;,
    /// Semantic version (optional)
    pub version: Option&lt;String&gt;,
    /// Block auto-invocation by context (optional)
    #[serde(rename = "disable-model-invocation")]
    pub disable_model_invocation: Option&lt;bool&gt;,
    /// Show in /skill menu (optional)
    #[serde(rename = "user-invocable")]
    pub user_invocable: Option&lt;bool&gt;,
}

/// Model selection for skills
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SkillModel {
    Haiku,
    Sonnet,
    Opus,
    #[default]
    Inherit,
}

/// Complete skill definition with parsed content
/// Implements REQ-SKILLS-01, REQ-SKILLS-02
#[derive(Debug, Clone)]
pub struct SkillDefinition {
    pub frontmatter: SkillFrontmatter,
    pub body: String,
    pub base_dir: PathBuf,
    pub allowed_tools_set: HashSet&lt;String&gt;,
    pub keywords: Vec&lt;String&gt;,
}

impl SkillDefinition {
    pub fn parse_allowed_tools(&mut self);
    pub fn extract_keywords(&mut self);
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool;
}
    </signature>
  </signatures>
  <constraints>
    - Skill name max 64 chars, lowercase a-z, digits 0-9, hyphens only
    - Description max 1024 chars
    - serde rename for kebab-case YAML fields
    - NO any type anywhere
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli skill_definition
  </verification>
</definition_of_done>

<pseudo_code>
1. Create directory structure:
   crates/context-graph-cli/src/skills/
   crates/context-graph-cli/src/skills/mod.rs
   crates/context-graph-cli/src/skills/types.rs

2. Create SkillModel enum:
   - Haiku variant
   - Sonnet variant
   - Opus variant
   - Inherit variant (default)
   - Implement Default trait

3. Create SkillFrontmatter struct:
   - name: String (required)
   - description: String (required)
   - allowed_tools: Option&lt;String&gt; with serde rename
   - model: Option&lt;SkillModel&gt;
   - version: Option&lt;String&gt;
   - disable_model_invocation: Option&lt;bool&gt; with serde rename
   - user_invocable: Option&lt;bool&gt; with serde rename

4. Create SkillDefinition struct:
   - frontmatter: SkillFrontmatter
   - body: String
   - base_dir: PathBuf
   - allowed_tools_set: HashSet&lt;String&gt;
   - keywords: Vec&lt;String&gt;

5. Implement SkillDefinition methods:
   - parse_allowed_tools(): Split comma-separated string into set
   - extract_keywords(): Parse "Keywords:" from description
   - is_tool_allowed(): Check exact match or scoped match (Bash(git:*))

6. Add tests:
   - test_skill_model_default
   - test_skill_model_serialization
   - test_frontmatter_parsing
   - test_allowed_tools_parsing
   - test_keyword_extraction
   - test_tool_allowed_check
   - test_scoped_tool_matching
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Skills module root with exports</file>
  <file path="crates/context-graph-cli/src/skills/types.rs">SkillDefinition, SkillFrontmatter, SkillModel types</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/lib.rs">Add pub mod skills;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli skill_definition</command>
  <command>cargo test --package context-graph-cli skill_model</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### SkillFrontmatter Fields

| Field | YAML Key | Type | Required | Notes |
|-------|----------|------|----------|-------|
| name | name | String | Yes | Max 64 chars, lowercase, hyphens |
| description | description | String | Yes | Max 1024 chars |
| allowed_tools | allowed-tools | Option<String> | No | Comma-separated |
| model | model | Option<SkillModel> | No | Default: inherit |
| version | version | Option<String> | No | SemVer format |
| disable_model_invocation | disable-model-invocation | Option<bool> | No | Default: false |
| user_invocable | user-invocable | Option<bool> | No | Default: true |

### Tool Matching Logic

```rust
// Exact match
"Read" matches allowed "Read"

// Scoped Bash match
"Bash(git:status)" matches allowed "Bash(git:*)"
"Bash(rm:-rf)" does NOT match allowed "Bash(git:*)"

// Empty allowed_tools means all tools allowed
allowed_tools: None -> all tools allowed
```

## Verification Checklist

- [ ] SkillModel has 4 variants with correct serde rename
- [ ] SkillFrontmatter has all YAML fields with correct serde renames
- [ ] SkillDefinition has all required fields
- [ ] parse_allowed_tools correctly splits comma-separated string
- [ ] extract_keywords finds "Keywords:" and parses list
- [ ] is_tool_allowed handles exact and scoped matches
- [ ] All tests pass
