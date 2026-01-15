# TASK-HOOKS-011: Implement CLI consciousness inject Command

```xml
<task_spec id="TASK-HOOKS-011" version="1.0">
<metadata>
  <title>Implement CLI consciousness inject Command</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>11</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-33</requirement_ref>
    <requirement_ref>REQ-HOOKS-34</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-010</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.5</estimated_hours>
</metadata>

<context>
The `consciousness inject` CLI command injects context from the knowledge graph
into the current session. Called by pre_tool_use hooks to provide relevant memories
before tool execution.

Supports two modes:
1. Automatic: Query semantically relevant memories based on tool context
2. Manual: Inject specific node IDs
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#cli_commands</file>
  <file purpose="inject_context_tool">docs2/mcptools.md#inject_context</file>
  <file purpose="search_graph_tool">docs2/mcptools.md#search_graph</file>
  <file purpose="cli_structure">crates/context-graph-cli/src/commands/consciousness/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>MCP server provides inject_context and search_graph tools</check>
  <check>consciousness brief command exists (TASK-HOOKS-010)</check>
  <check>Knowledge graph has stored memories to inject</check>
</prerequisites>

<scope>
  <in_scope>
    - Create `inject` subcommand under `consciousness` command group
    - Support --query flag for semantic search injection
    - Support --node-ids flag for explicit node injection
    - Support --max-tokens flag to limit injection size
    - Output injected context summary to stdout
    - Return injected content as structured output
  </in_scope>
  <out_of_scope>
    - Dream consolidation triggers (different feature)
    - Curation operations (different feature)
    - Shell script integration (TASK-HOOKS-014)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/consciousness/inject.rs">
      pub struct InjectArgs {
          #[arg(long)]
          pub query: Option&lt;String&gt;,

          #[arg(long, value_delimiter = ',')]
          pub node_ids: Option&lt;Vec&lt;String&gt;&gt;,

          #[arg(long, default_value = "500")]
          pub max_tokens: u32,

          #[arg(long, default_value = "text")]
          pub format: OutputFormat,
      }

      pub async fn execute_inject(args: InjectArgs) -> Result&lt;InjectResult, CliError&gt;
    </signature>
  </signatures>

  <constraints>
    - Must use inject_context MCP tool for actual injection
    - Output must respect max_tokens limit
    - Must handle empty search results gracefully
    - JSON output must include injected_nodes, total_tokens, relevance_scores
    - No panics - all errors handled gracefully
  </constraints>

  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli inject
    - context-graph-cli consciousness inject --query "test query" --format json
    - context-graph-cli consciousness inject --node-ids "node1,node2" --max-tokens 200
  </verification>
</definition_of_done>

<pseudo_code>
InjectArgs:
  query: Option[String]
  node_ids: Option[Vec[String]]
  max_tokens: u32
  format: OutputFormat

execute_inject(args):
  // Validate args - need either query or node_ids
  if args.query.is_none() && args.node_ids.is_none():
    return Err(CliError::MissingArgument("query or node_ids required"))

  nodes_to_inject = []

  if args.query.is_some():
    // Search for relevant nodes
    search_result = mcp_call("search_graph", {
      query: args.query,
      limit: 10
    })
    nodes_to_inject = search_result.nodes

  if args.node_ids.is_some():
    // Add explicit nodes
    for node_id in args.node_ids:
      node = mcp_call("get_node", { id: node_id })
      nodes_to_inject.push(node)

  // Inject context with token limit
  inject_result = mcp_call("inject_context", {
    nodes: nodes_to_inject,
    max_tokens: args.max_tokens
  })

  // Build result
  result = InjectResult {
    injected_nodes: inject_result.node_ids,
    total_tokens: inject_result.token_count,
    content_summary: inject_result.summary,
    relevance_scores: inject_result.scores
  }

  // Output based on format
  match args.format:
    Json => print_json(result)
    Text => print_text_inject(result)

  return Ok(result)
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/consciousness/inject.rs">
    Inject command implementation with MCP queries and injection logic
  </file>
  <file path="crates/context-graph-cli/tests/commands/consciousness_inject_test.rs">
    Integration tests for inject command (real MCP calls, no mocks)
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/consciousness/mod.rs">
    Add inject subcommand to ConsciousnessCommands enum
  </file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli consciousness_inject</command>
  <command>./target/debug/context-graph-cli consciousness inject --query "test" --format json</command>
</test_commands>
</task_spec>
```
