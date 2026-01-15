# TASK-SKILLS-007: Implement SkillRegistry with Discovery Precedence

```xml
<task_spec id="TASK-SKILLS-007" version="1.0">
<metadata>
  <title>Implement SkillRegistry with Discovery Precedence</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>7</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-04</requirement_ref>
    <requirement_ref>REQ-SKILLS-31</requirement_ref>
    <requirement_ref>REQ-SKILLS-34</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-001</task_ref>
    <task_ref>TASK-SKILLS-004</task_ref>
    <task_ref>TASK-SKILLS-006</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
This task implements the SkillRegistry that discovers and manages skills from multiple
sources. Skills can come from project (.claude/skills/), personal (~/.claude/skills/),
or plugins. Project skills have highest precedence, then personal, then plugins.
The registry discovers all skills at startup with Level 1 metadata and provides
lookup/listing capabilities.

Technical Spec Reference: TECH-SKILLS Section 3.2
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-3.2</file>
  <file purpose="loader">crates/context-graph-cli/src/skills/loader.rs</file>
  <file purpose="types">crates/context-graph-cli/src/skills/types.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-006 completed (SkillLoader exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SkillRegistry struct
    - Implement skill discovery from multiple sources
    - Implement precedence resolution (project > personal > plugin)
    - Implement get/list operations
    - Implement register/unregister for dynamic updates
    - Implement load_instructions for on-demand Level 2 loading
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - TriggerMatcher (TASK-SKILLS-008)
    - Plugin system integration (future phase)
    - Actual SKILL.md files (TASK-SKILLS-011-013)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/registry.rs">
use crate::skills::{
    errors::SkillError,
    loader::SkillLoader,
    types::{SkillDefinition, SkillLoadResult, SkillTrigger},
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Source of a skill definition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillSource {
    Project,   // .claude/skills/ (highest precedence)
    Personal,  // ~/.claude/skills/
    Plugin,    // installed plugin skills/
}

/// Manages skill discovery and registration
/// Implements REQ-SKILLS-34 (precedence)
pub struct SkillRegistry {
    project_skills: HashMap&lt;String, SkillDefinition&gt;,
    personal_skills: HashMap&lt;String, SkillDefinition&gt;,
    plugin_skills: HashMap&lt;String, SkillDefinition&gt;,
    loader: SkillLoader,
}

impl SkillRegistry {
    /// Create new registry and discover all skills
    pub fn new(project_dir: impl Into&lt;PathBuf&gt;) -> Result&lt;Self, SkillError&gt;;

    /// Discover all skills at startup (Level 1 metadata only)
    pub fn discover_all(&mut self) -> Result&lt;usize, SkillError&gt;;

    /// Get skill by name with precedence resolution
    pub fn get(&self, name: &str) -> Option&lt;&SkillDefinition&gt;;

    /// Get all registered skill names
    pub fn list_all(&self) -> Vec&lt;String&gt;;

    /// Get skills filtered by source
    pub fn list_by_source(&self, source: SkillSource) -> Vec&lt;&SkillDefinition&gt;;

    /// Register a skill manually
    pub fn register(&mut self, skill: SkillDefinition, source: SkillSource);

    /// Unregister a skill by name
    pub fn unregister(&mut self, name: &str, source: SkillSource) -> bool;

    /// Load full instructions for a skill (Level 2)
    pub fn load_instructions(&mut self, name: &str) -> Result&lt;SkillLoadResult, SkillError&gt;;

    /// Get all trigger configurations for registered skills
    pub fn get_triggers(&self) -> Vec&lt;SkillTrigger&gt;;

    /// Check if skill exists (any source)
    pub fn exists(&self, name: &str) -> bool;

    /// Get skill count by source
    pub fn count_by_source(&self, source: SkillSource) -> usize;
}
    </signature>
  </signatures>
  <constraints>
    - Precedence: Project > Personal > Plugin
    - Discovery loads Level 1 (metadata) only
    - Same-name skills resolved by precedence
    - Plugin skills placeholder for future integration
    - NO backwards compatibility code
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli skill_registry
  </verification>
</definition_of_done>

<pseudo_code>
1. Create registry.rs in skills module

2. Create SkillSource enum:
   - Project (highest precedence)
   - Personal
   - Plugin (lowest precedence)

3. Implement SkillRegistry::new():
   fn new(project_dir: impl Into&lt;PathBuf&gt;) -> Result&lt;Self, SkillError&gt;:
       let project_dir = project_dir.into();
       let project_skills_dir = project_dir.join(".claude/skills");
       let loader = SkillLoader::new(&project_skills_dir)?;

       Ok(Self {
           project_skills: HashMap::new(),
           personal_skills: HashMap::new(),
           plugin_skills: HashMap::new(),
           loader,
       })

4. Implement discover_all():
   fn discover_all(&mut self) -> Result&lt;usize, SkillError&gt;:
       let mut count = 0;

       // 1. Discover project skills
       let project_dir = self.loader.skills_dir();
       if project_dir.exists():
           for entry in std::fs::read_dir(project_dir)?:
               let skill_name = entry.file_name();
               if let Ok(result) = self.loader.load_metadata(&skill_name):
                   if let Some(skill) = result.skill:
                       self.project_skills.insert(skill.frontmatter.name.clone(), skill);
                       count += 1;

       // 2. Discover personal skills
       if let Some(home) = dirs::home_dir():
           let personal_dir = home.join(".claude/skills");
           // Similar discovery...

       // 3. Plugin skills (future)
       // self.discover_plugin_skills()?;

       Ok(count)

5. Implement get() with precedence:
   fn get(&self, name: &str) -> Option&lt;&SkillDefinition&gt;:
       // Check project first (highest precedence)
       if let Some(skill) = self.project_skills.get(name):
           return Some(skill);

       // Check personal second
       if let Some(skill) = self.personal_skills.get(name):
           return Some(skill);

       // Check plugin last
       self.plugin_skills.get(name)

6. Implement list_all():
   fn list_all(&self) -> Vec&lt;String&gt;:
       let mut names = HashSet::new();
       names.extend(self.project_skills.keys().cloned());
       names.extend(self.personal_skills.keys().cloned());
       names.extend(self.plugin_skills.keys().cloned());
       names.into_iter().collect()

7. Implement get_triggers():
   fn get_triggers(&self) -> Vec&lt;SkillTrigger&gt;:
       self.list_all()
           .iter()
           .filter_map(|name| self.get(name))
           .filter(|skill| !skill.frontmatter.disable_model_invocation.unwrap_or(false))
           .map(|skill| SkillTrigger {
               skill_name: skill.frontmatter.name.clone(),
               keywords: skill.keywords.clone(),
               min_confidence: 0.5,
               require_confirmation: false,
           })
           .collect()

8. Add tests:
   - test_registry_new
   - test_discover_all_counts
   - test_get_precedence_project_first
   - test_get_precedence_personal_second
   - test_list_all_unique
   - test_register_manual
   - test_unregister
   - test_load_instructions_level2
   - test_get_triggers
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/registry.rs">SkillRegistry implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Add pub mod registry;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli skill_registry</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Discovery Order and Precedence

```
1. Project skills: .claude/skills/ (HIGHEST)
   └── Skills specific to this project
   └── Override personal/plugin skills

2. Personal skills: ~/.claude/skills/ (MEDIUM)
   └── User's personal skill collection
   └── Override plugin skills

3. Plugin skills: (plugin_dir)/skills/ (LOWEST)
   └── Skills from installed plugins
   └── May be overridden by project/personal
```

### Precedence Resolution Example

```rust
// Project has: consciousness, memory-inject
// Personal has: consciousness, custom-skill
// Plugin has: consciousness, plugin-skill

registry.get("consciousness")
// Returns: Project's consciousness (highest precedence)

registry.get("custom-skill")
// Returns: Personal's custom-skill

registry.get("plugin-skill")
// Returns: Plugin's plugin-skill

registry.list_all()
// Returns: [consciousness, memory-inject, custom-skill, plugin-skill]
// Note: Only unique names, project version of consciousness
```

### Directory Structure

```
project/
├── .claude/
│   └── skills/
│       ├── consciousness/
│       │   └── SKILL.md
│       └── memory-inject/
│           └── SKILL.md

~/.claude/
└── skills/
    └── custom-skill/
        └── SKILL.md
```

### Startup Behavior

```rust
// At startup, discover all skills with Level 1 (metadata only)
let mut registry = SkillRegistry::new(&project_dir)?;
let count = registry.discover_all()?;
// count = total unique skills discovered

// Later, when skill is triggered, load Level 2
let result = registry.load_instructions("consciousness")?;
// result.instructions contains full SKILL.md body
```

## Verification Checklist

- [ ] SkillRegistry discovers project skills
- [ ] SkillRegistry discovers personal skills
- [ ] Precedence: project > personal > plugin
- [ ] get() returns highest-precedence skill
- [ ] list_all() returns unique names only
- [ ] register() adds to correct source map
- [ ] unregister() removes from correct source map
- [ ] load_instructions() returns Level 2 content
- [ ] get_triggers() excludes disabled skills
- [ ] All tests pass
