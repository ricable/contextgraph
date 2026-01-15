# TASK-SKILLS-006: Implement SkillLoader with Progressive Disclosure

```xml
<task_spec id="TASK-SKILLS-006" version="1.0">
<metadata>
  <title>Implement SkillLoader with Progressive Disclosure</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>6</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-01</requirement_ref>
    <requirement_ref>REQ-SKILLS-02</requirement_ref>
    <requirement_ref>REQ-SKILLS-03</requirement_ref>
    <requirement_ref>REQ-SKILLS-26</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-001</task_ref>
    <task_ref>TASK-SKILLS-003</task_ref>
    <task_ref>TASK-SKILLS-004</task_ref>
    <task_ref>TASK-SKILLS-005</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>2.5</estimated_hours>
</metadata>

<context>
This task implements the SkillLoader that loads SKILL.md files with progressive disclosure.
The loader supports three levels: Level 1 (Metadata) loads only frontmatter, Level 2
(Instructions) loads the full body, and Level 3 (Resources) loads bundled files on-demand.
The loader validates skill names, parses YAML frontmatter, extracts keywords, and resolves
{baseDir} placeholders in resource paths.

Technical Spec Reference: TECH-SKILLS Section 3.1
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-3.1</file>
  <file purpose="types">crates/context-graph-cli/src/skills/types.rs</file>
  <file purpose="errors">crates/context-graph-cli/src/skills/errors.rs</file>
</input_context_files>

<prerequisites>
  <check>All foundation types exist (001-005)</check>
  <check>serde_yaml is workspace dependency</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SkillLoader struct
    - Implement load_metadata() for Level 1
    - Implement load_instructions() for Level 2
    - Implement load_resource() for Level 3
    - Implement parse_frontmatter()
    - Implement validate_skill_name()
    - Implement validate_description()
    - Implement path traversal checking
    - Implement {baseDir} placeholder resolution
    - Add comprehensive unit tests
  </in_scope>
  <out_of_scope>
    - SkillRegistry (TASK-SKILLS-007)
    - TriggerMatcher (TASK-SKILLS-008)
    - Actual SKILL.md files (TASK-SKILLS-011-013)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/loader.rs">
use crate::skills::{
    errors::SkillError,
    types::{ProgressiveDisclosureLevel, SkillDefinition, SkillFrontmatter, SkillLoadResult},
};
use std::path::{Path, PathBuf};

/// Loads SKILL.md files with progressive disclosure
/// Implements REQ-SKILLS-01, REQ-SKILLS-02, REQ-SKILLS-03
pub struct SkillLoader {
    skills_dir: PathBuf,
}

impl SkillLoader {
    /// Create new loader for given skills directory
    pub fn new(skills_dir: impl AsRef&lt;Path&gt;) -> Result&lt;Self, SkillError&gt;;

    /// Load skill metadata only (Level 1, ~100 tokens)
    pub fn load_metadata(&self, skill_name: &str) -> Result&lt;SkillLoadResult, SkillError&gt;;

    /// Load skill instructions (Level 2, &lt;5k tokens)
    pub fn load_instructions(&self, skill_name: &str) -> Result&lt;SkillLoadResult, SkillError&gt;;

    /// Load skill resource (Level 3, unlimited)
    pub fn load_resource(&self, skill_name: &str, resource_path: &str) -> Result&lt;String, SkillError&gt;;

    /// Parse YAML frontmatter from SKILL.md content
    fn parse_frontmatter(&self, content: &str, skill_name: &str) -> Result&lt;(SkillFrontmatter, String), SkillError&gt;;

    /// Validate skill name
    fn validate_skill_name(&self, name: &str) -> Result&lt;(), SkillError&gt;;

    /// Validate description length
    fn validate_description(&self, skill_name: &str, description: &str) -> Result&lt;(), SkillError&gt;;

    /// Resolve {baseDir} placeholders
    fn resolve_base_dir(&self, skill_name: &str, path: &str) -> PathBuf;

    /// Check for path traversal attempts
    fn check_path_traversal(&self, skill_name: &str, path: &str) -> Result&lt;(), SkillError&gt;;

    /// Get path to skill directory
    fn skill_path(&self, skill_name: &str) -> PathBuf;

    /// Get path to SKILL.md file
    fn skill_md_path(&self, skill_name: &str) -> PathBuf;
}
    </signature>
  </signatures>
  <constraints>
    - Skill names: max 64 chars, lowercase a-z, digits 0-9, hyphens only
    - Skill names cannot contain "anthropic" or "claude"
    - Description max 1024 characters
    - YAML frontmatter must be between --- delimiters
    - Path traversal (../) must be rejected
    - {baseDir} resolved to skill directory absolute path
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli skill_loader
  </verification>
</definition_of_done>

<pseudo_code>
1. Create loader.rs in skills module

2. Implement SkillLoader::new():
   fn new(skills_dir: impl AsRef&lt;Path&gt;) -> Result&lt;Self, SkillError&gt;:
       let path = skills_dir.as_ref().canonicalize()?;
       Ok(Self { skills_dir: path })

3. Implement load_metadata():
   fn load_metadata(&self, skill_name: &str) -> Result&lt;SkillLoadResult, SkillError&gt;:
       // 1. Validate skill name
       self.validate_skill_name(skill_name)?;

       // 2. Check SKILL.md exists
       let skill_md = self.skill_md_path(skill_name);
       if !skill_md.exists():
           return Err(SkillError::SkillNotFound { ... });

       // 3. Read file content
       let content = std::fs::read_to_string(&skill_md)?;

       // 4. Parse frontmatter only
       let (frontmatter, body) = self.parse_frontmatter(&content, skill_name)?;

       // 5. Validate description
       self.validate_description(skill_name, &frontmatter.description)?;

       // 6. Build SkillDefinition
       let mut skill = SkillDefinition {
           frontmatter,
           body,
           base_dir: self.skill_path(skill_name),
           allowed_tools_set: HashSet::new(),
           keywords: Vec::new(),
       };
       skill.parse_allowed_tools();
       skill.extract_keywords();

       Ok(SkillLoadResult::success(skill, ProgressiveDisclosureLevel::Metadata))

4. Implement load_instructions():
   fn load_instructions(&self, skill_name: &str) -> Result&lt;SkillLoadResult, SkillError&gt;:
       // Load metadata first
       let mut result = self.load_metadata(skill_name)?;
       result.level = ProgressiveDisclosureLevel::Instructions;
       result.instructions = result.skill.as_ref().map(|s| s.body.clone());
       Ok(result)

5. Implement load_resource():
   fn load_resource(&self, skill_name: &str, resource_path: &str) -> Result&lt;String, SkillError&gt;:
       // 1. Check path traversal
       self.check_path_traversal(skill_name, resource_path)?;

       // 2. Resolve {baseDir}
       let full_path = self.resolve_base_dir(skill_name, resource_path);

       // 3. Check file exists
       if !full_path.exists():
           return Err(SkillError::ResourceNotFound { ... });

       // 4. Read and return
       std::fs::read_to_string(&full_path).map_err(...)

6. Implement parse_frontmatter():
   fn parse_frontmatter(&self, content: &str, skill_name: &str) -> Result&lt;...&gt;:
       // Find --- delimiters
       let parts: Vec&lt;&str&gt; = content.splitn(3, "---").collect();
       if parts.len() < 3:
           return Err(SkillError::SkillParseError { ... });

       // Parse YAML
       let frontmatter: SkillFrontmatter = serde_yaml::from_str(parts[1])?;
       let body = parts[2].trim().to_string();

       Ok((frontmatter, body))

7. Implement validate_skill_name():
   fn validate_skill_name(&self, name: &str) -> Result&lt;(), SkillError&gt;:
       // Check length
       if name.len() > 64:
           return Err(InvalidSkillName { reason: "exceeds 64 chars" });

       // Check pattern: lowercase, digits, hyphens
       let valid = name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
       if !valid:
           return Err(InvalidSkillName { reason: "must be lowercase with hyphens" });

       // Check reserved words
       if name.contains("anthropic") || name.contains("claude"):
           return Err(InvalidSkillName { reason: "cannot contain reserved words" });

       Ok(())

8. Implement check_path_traversal():
   fn check_path_traversal(&self, skill_name: &str, path: &str) -> Result&lt;(), SkillError&gt;:
       if path.contains(".."):
           return Err(SkillError::PathTraversal { path: path.to_string() });
       Ok(())

9. Add tests:
   - test_load_metadata_valid_skill
   - test_load_metadata_not_found
   - test_load_instructions
   - test_load_resource
   - test_resource_path_traversal_blocked
   - test_invalid_skill_name_uppercase
   - test_invalid_skill_name_reserved
   - test_description_too_long
   - test_parse_frontmatter_valid
   - test_parse_frontmatter_invalid_yaml
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/loader.rs">SkillLoader implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Add pub mod loader;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli skill_loader</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### YAML Frontmatter Format

```yaml
---
name: consciousness
description: |
  Access Context Graph consciousness state.
  When to use: querying system awareness.
  Keywords: consciousness, awareness, identity
allowed-tools: Read,Grep,mcp__context-graph__get_consciousness_state
model: sonnet
version: 1.0.0
---
# Skill body starts here
```

### Skill Name Validation Rules

| Rule | Regex | Valid | Invalid |
|------|-------|-------|---------|
| Max 64 chars | `.{1,64}` | `my-skill` | `very-long-name-that-...` |
| Lowercase | `[a-z0-9-]+` | `memory-inject` | `Memory-Inject` |
| Hyphens only | `^[a-z][a-z0-9-]*$` | `dream-consolidation` | `dream_consolidation` |
| No reserved | - | `my-assistant` | `claude-helper` |

### Path Traversal Protection

```rust
// Blocked
"../../../etc/passwd" -> Err(PathTraversal)
"references/../../../secrets" -> Err(PathTraversal)

// Allowed
"references/doc.md" -> Ok
"scripts/helper.py" -> Ok
```

### {baseDir} Resolution

```rust
// Input: "{baseDir}/scripts/helper.py"
// skill_name: "consciousness"
// skills_dir: "/project/.claude/skills"
// Output: "/project/.claude/skills/consciousness/scripts/helper.py"
```

## Verification Checklist

- [ ] SkillLoader::new validates skills_dir exists
- [ ] load_metadata returns Level 1 result with ~100 token estimate
- [ ] load_instructions returns Level 2 result with body
- [ ] load_resource resolves {baseDir} and returns content
- [ ] Path traversal is blocked with clear error
- [ ] Invalid skill names are rejected
- [ ] Description >1024 chars is rejected
- [ ] YAML parse errors include helpful details
- [ ] All tests pass with real fixture files
