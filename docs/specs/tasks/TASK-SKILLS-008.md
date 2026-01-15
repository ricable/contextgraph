# TASK-SKILLS-008: Implement TriggerMatcher for Auto-Invocation

```xml
<task_spec id="TASK-SKILLS-008" version="1.0">
<metadata>
  <title>Implement TriggerMatcher for Skill Auto-Invocation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>8</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-10</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-001</task_ref>
    <task_ref>TASK-SKILLS-003</task_ref>
    <task_ref>TASK-SKILLS-004</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.5</estimated_hours>
</metadata>

<context>
This task implements the TriggerMatcher that matches user input against skill keywords
to determine which skill (if any) should be auto-invoked. Skills define keywords in their
description (after "Keywords:"), and the matcher calculates confidence scores based on
keyword matches. When multiple skills match, the highest confidence wins. Equal confidence
triggers a conflict error requiring user clarification.

Technical Spec Reference: TECH-SKILLS Section 3.5
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-3.5</file>
  <file purpose="types">crates/context-graph-cli/src/skills/types.rs</file>
  <file purpose="errors">crates/context-graph-cli/src/skills/errors.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-003 completed (SkillTrigger exists)</check>
  <check>TASK-SKILLS-004 completed (TriggerError exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create TriggerMatcher struct
    - Implement register_triggers() from skill definitions
    - Implement match_input() for keyword matching
    - Implement calculate_confidence() scoring
    - Implement get_all_matches() for debugging
    - Implement conflict detection
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - SkillRegistry integration (handled by caller)
    - UserPromptSubmit hook (Phase 3 integration)
    - Actual triggering of skill loading
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/skills/matcher.rs">
use crate::skills::{
    errors::TriggerError,
    types::{SkillDefinition, SkillTrigger},
};

/// Matches context to auto-trigger skills
pub struct TriggerMatcher {
    triggers: Vec&lt;SkillTrigger&gt;,
    min_confidence: f32,
}

impl TriggerMatcher {
    /// Create new matcher with default confidence threshold (0.5)
    pub fn new() -> Self;

    /// Create matcher with custom confidence threshold
    pub fn with_confidence(min_confidence: f32) -> Self;

    /// Register triggers from skill definitions
    pub fn register_triggers(&mut self, skills: &[SkillDefinition]);

    /// Match user input against registered triggers
    /// Returns matched skill name if confident match found
    pub fn match_input(&self, input: &str) -> Result&lt;Option&lt;String&gt;, TriggerError&gt;;

    /// Calculate match confidence for a skill
    fn calculate_confidence(&self, trigger: &SkillTrigger, input: &str) -> f32;

    /// Get all potential matches above threshold
    pub fn get_all_matches(&self, input: &str) -> Vec&lt;(String, f32)&gt;;

    /// Check for trigger conflicts
    fn check_conflicts(&self, matches: &[(String, f32)]) -> Result&lt;(), TriggerError&gt;;

    /// Get number of registered triggers
    pub fn trigger_count(&self) -> usize;

    /// Clear all triggers
    pub fn clear(&mut self);
}

impl Default for TriggerMatcher {
    fn default() -> Self { Self::new() }
}
    </signature>
  </signatures>
  <constraints>
    - Default min_confidence: 0.5
    - Confidence = matched_keywords / total_keywords
    - Case-insensitive matching
    - Equal top confidence triggers conflict error
    - disable-model-invocation skills excluded
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli trigger_matcher
  </verification>
</definition_of_done>

<pseudo_code>
1. Create matcher.rs in skills module

2. Implement TriggerMatcher::new():
   fn new() -> Self:
       Self {
           triggers: Vec::new(),
           min_confidence: 0.5,
       }

3. Implement with_confidence():
   fn with_confidence(min_confidence: f32) -> Self:
       Self {
           triggers: Vec::new(),
           min_confidence,
       }

4. Implement register_triggers():
   fn register_triggers(&mut self, skills: &[SkillDefinition]):
       for skill in skills:
           // Skip if auto-invocation disabled
           if skill.frontmatter.disable_model_invocation.unwrap_or(false):
               continue;

           let trigger = SkillTrigger {
               skill_name: skill.frontmatter.name.clone(),
               keywords: skill.keywords.clone(),
               min_confidence: 0.5,
               require_confirmation: false,
           };
           self.triggers.push(trigger);

5. Implement calculate_confidence():
   fn calculate_confidence(&self, trigger: &SkillTrigger, input: &str) -> f32:
       if trigger.keywords.is_empty():
           return 0.0;

       let input_lower = input.to_lowercase();
       let matched = trigger.keywords.iter()
           .filter(|kw| input_lower.contains(&kw.to_lowercase()))
           .count();

       matched as f32 / trigger.keywords.len() as f32

6. Implement get_all_matches():
   fn get_all_matches(&self, input: &str) -> Vec&lt;(String, f32)&gt;:
       let mut matches: Vec&lt;_&gt; = self.triggers.iter()
           .map(|t| (t.skill_name.clone(), self.calculate_confidence(t, input)))
           .filter(|(_, conf)| *conf >= self.min_confidence)
           .collect();

       // Sort by confidence descending
       matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
       matches

7. Implement check_conflicts():
   fn check_conflicts(&self, matches: &[(String, f32)]) -> Result&lt;(), TriggerError&gt;:
       if matches.len() >= 2:
           let top_conf = matches[0].1;
           let tied: Vec&lt;_&gt; = matches.iter()
               .take_while(|(_, c)| *c == top_conf)
               .map(|(n, _)| n.clone())
               .collect();

           if tied.len() > 1:
               return Err(TriggerError::TriggerMatchConflict {
                   skills: tied,
                   confidence: top_conf,
               });
       Ok(())

8. Implement match_input():
   fn match_input(&self, input: &str) -> Result&lt;Option&lt;String&gt;, TriggerError&gt;:
       let matches = self.get_all_matches(input);

       if matches.is_empty():
           return Ok(None);

       self.check_conflicts(&matches)?;

       Ok(Some(matches[0].0.clone()))

9. Add tests:
   - test_match_single_keyword
   - test_match_multiple_keywords
   - test_confidence_calculation
   - test_below_threshold_no_match
   - test_conflict_detection
   - test_highest_confidence_wins
   - test_case_insensitive
   - test_disabled_skill_excluded
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/skills/matcher.rs">TriggerMatcher implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/skills/mod.rs">Add pub mod matcher;</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli trigger_matcher</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Confidence Calculation

```rust
// Keywords: ["consciousness", "awareness", "identity", "coherence"]
// Input: "check consciousness and identity"

matched = 2  // consciousness, identity
total = 4
confidence = 2/4 = 0.5

// Meets threshold (0.5), skill triggers
```

### Conflict Scenarios

```rust
// Skill A: keywords=["memory", "search"]
// Skill B: keywords=["memory", "inject"]
// Input: "memory operation"

// A confidence: 1/2 = 0.5
// B confidence: 1/2 = 0.5
// Result: TriggerMatchConflict (both 0.5)
```

### Priority Resolution

```rust
// Skill A: keywords=["memory", "search", "find"]
// Skill B: keywords=["memory", "inject"]
// Input: "memory search find"

// A confidence: 3/3 = 1.0
// B confidence: 1/2 = 0.5
// Result: Skill A wins (higher confidence)
```

### Context Graph Skill Keywords

| Skill | Keywords |
|-------|----------|
| consciousness | consciousness, awareness, identity, coherence, kuramoto, GWT |
| memory-inject | memory, context, inject, retrieve, recall, background |
| semantic-search | search, find, query, lookup, semantic, causal |
| dream-consolidation | dream, consolidate, nrem, rem, blind spots, entropy |
| curation | curate, merge, forget, annotate, prune, duplicate |

### Usage Flow

```rust
// 1. Build matcher from registry
let mut matcher = TriggerMatcher::new();
let skills: Vec<_> = registry.list_all()
    .iter()
    .filter_map(|n| registry.get(n))
    .cloned()
    .collect();
matcher.register_triggers(&skills);

// 2. Match user input
let user_input = "check consciousness state";
match matcher.match_input(user_input) {
    Ok(Some(skill_name)) => {
        // Load and activate skill
        let result = registry.load_instructions(&skill_name)?;
        // ...
    }
    Ok(None) => {
        // No skill matched, proceed normally
    }
    Err(TriggerError::TriggerMatchConflict { skills, confidence }) => {
        // Ask user to clarify
        println!("Multiple skills matched: {:?}", skills);
    }
}
```

## Verification Checklist

- [ ] TriggerMatcher::new() uses 0.5 default threshold
- [ ] with_confidence() allows custom threshold
- [ ] register_triggers() skips disabled skills
- [ ] calculate_confidence() returns correct ratio
- [ ] Case-insensitive matching works
- [ ] Below-threshold matches return None
- [ ] Equal confidence triggers conflict error
- [ ] Highest confidence wins when unambiguous
- [ ] All tests pass
