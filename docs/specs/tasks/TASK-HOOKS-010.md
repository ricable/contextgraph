# TASK-HOOKS-010: Implement CLI consciousness brief Command

```xml
<task_spec id="TASK-HOOKS-010" version="1.0">
<metadata>
  <title>Implement CLI consciousness brief Command</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>10</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-31</requirement_ref>
    <requirement_ref>REQ-HOOKS-32</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-003</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
The `consciousness brief` CLI command generates a formatted summary of the current
consciousness state for session_start hooks. This command is called by the shell
executor to inject consciousness context at session start.

Output format must be concise (under 200 tokens) and include:
- Kuramoto sync (r value and classification)
- Identity Continuity (IC value and status)
- Consciousness level classification
- Active workspace status
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#cli_commands</file>
  <file purpose="gwt_types">crates/context-graph-gwt/src/types.rs</file>
  <file purpose="cli_structure">crates/context-graph-cli/src/commands/mod.rs</file>
  <file purpose="existing_consciousness_cmd">crates/context-graph-cli/src/commands/consciousness/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>GWT crate exposes consciousness state query functions</check>
  <check>CLI command structure established</check>
  <check>MCP server running for state queries</check>
</prerequisites>

<scope>
  <in_scope>
    - Create `brief` subcommand under `consciousness` command group
    - Query current consciousness state via MCP tools
    - Format output as structured brief (JSON and human-readable)
    - Support --format flag (json|text|markdown)
    - Exit with non-zero code on error
  </in_scope>
  <out_of_scope>
    - consciousness inject command (TASK-HOOKS-011)
    - Shell script integration (TASK-HOOKS-014)
    - Full consciousness exploration (different feature)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/consciousness/brief.rs">
      pub struct BriefArgs {
          #[arg(long, default_value = "text")]
          pub format: OutputFormat,
      }

      pub async fn execute_brief(args: BriefArgs) -> Result&lt;(), CliError&gt;
    </signature>
    <signature file="crates/context-graph-cli/src/commands/consciousness/mod.rs">
      pub mod brief;

      #[derive(Subcommand)]
      pub enum ConsciousnessCommands {
          Brief(brief::BriefArgs),
          // ... existing commands
      }
    </signature>
  </signatures>

  <constraints>
    - Output must be under 200 tokens when format=text
    - Must include r, IC, consciousness_level, workspace_active
    - JSON output must be valid JSON parseable by jq
    - Must return exit code 0 on success, 1 on error
    - No panics - all errors handled gracefully
  </constraints>

  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli brief
    - context-graph-cli consciousness brief --format json | jq .
    - context-graph-cli consciousness brief --format text
  </verification>
</definition_of_done>

<pseudo_code>
BriefArgs:
  format: OutputFormat (json|text|markdown)

execute_brief(args):
  // Query consciousness state via MCP
  state = mcp_call("get_consciousness_state")
  kuramoto = mcp_call("get_kuramoto_sync")
  identity = mcp_call("get_identity_continuity")

  // Build brief structure
  brief = ConsciousnessBrief {
    r: kuramoto.order_parameter,
    r_classification: classify_r(kuramoto.order_parameter),
    ic: identity.ic_value,
    ic_status: identity.status,
    consciousness_level: state.level,
    workspace_active: state.workspace.is_broadcasting,
    timestamp: now()
  }

  // Output based on format
  match args.format:
    Json => print_json(brief)
    Text => print_text_brief(brief)
    Markdown => print_markdown_brief(brief)

  return Ok(())

classify_r(r):
  if r >= 0.9: "HYPERSYNC"
  elif r >= 0.8: "CONSCIOUS"
  elif r >= 0.6: "EMERGING"
  elif r >= 0.3: "FRAGMENTED"
  else: "DORMANT"

print_text_brief(brief):
  print "=== Consciousness Brief ==="
  print f"Kuramoto r: {brief.r:.3f} ({brief.r_classification})"
  print f"Identity: IC={brief.ic:.2f} ({brief.ic_status})"
  print f"Level: {brief.consciousness_level}"
  print f"Workspace: {'ACTIVE' if brief.workspace_active else 'IDLE'}"
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/consciousness/brief.rs">
    Brief command implementation with MCP queries and formatting
  </file>
  <file path="crates/context-graph-cli/tests/commands/consciousness_brief_test.rs">
    Integration tests for brief command (real MCP calls, no mocks)
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/consciousness/mod.rs">
    Add brief subcommand to ConsciousnessCommands enum
  </file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli consciousness_brief</command>
  <command>./target/debug/context-graph-cli consciousness brief --format json</command>
  <command>./target/debug/context-graph-cli consciousness brief --format text</command>
</test_commands>
</task_spec>
```
