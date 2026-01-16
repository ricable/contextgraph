# Functional Specification: Phase 5 - Injection Pipeline

```xml
<functional_spec id="SPEC-PHASE5" version="1.0">
<metadata>
  <title>Injection Pipeline</title>
  <status>approved</status>
  <owner>Context Graph Team</owner>
  <created>2026-01-16</created>
  <last_updated>2026-01-16</last_updated>
  <implements>impplan.md Part 3</implements>
  <depends_on>
    <spec_ref>SPEC-PHASE0</spec_ref>
    <spec_ref>SPEC-PHASE1</spec_ref>
    <spec_ref>SPEC-PHASE2</spec_ref>
    <spec_ref>SPEC-PHASE3</spec_ref>
    <spec_ref>SPEC-PHASE4</spec_ref>
  </depends_on>
  <related_specs>
    <spec_ref>SPEC-PHASE6</spec_ref>
  </related_specs>
</metadata>

<overview>
Implement the context injection pipeline that surfaces relevant memories and divergence alerts to Claude during hook execution. The pipeline:
1. Retrieves relevant memories using multi-space similarity (Phase 3)
2. Ranks by priority algorithm (relevance × recency × diversity)
3. Formats output with token budgeting
4. Includes divergence alerts when activity shifts detected

**Problem Solved**: Currently no automatic context injection exists. Claude must manually search for context. This pipeline automatically surfaces relevant memories and alerts.

**Who Benefits**: Claude instances that receive targeted context at each prompt; users who get better responses due to richer context.
</overview>

<user_stories>
<story id="US-P5-01" priority="must-have">
  <narrative>
    As a Claude instance
    I want to receive relevant context at UserPromptSubmit
    So that I have background information for the current query
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P5-01-01">
      <given>A user prompt about authentication</given>
      <when>UserPromptSubmit hook fires</when>
      <then>Relevant memories about authentication are injected</then>
    </criterion>
    <criterion id="AC-P5-01-02">
      <given>Injected context</given>
      <when>Measuring token count</when>
      <then>Total tokens &lt;= 1150 (within budget)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P5-02" priority="must-have">
  <narrative>
    As a Claude instance
    I want divergence alerts surfaced prominently
    So that I know when I'm working on something different from recent context
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P5-02-01">
      <given>Current query with low similarity to recent memories in E1</given>
      <when>Injection pipeline runs</when>
      <then>Divergence alert appears at TOP of injected context</then>
    </criterion>
    <criterion id="AC-P5-02-02">
      <given>No divergence detected</given>
      <when>Injection pipeline runs</when>
      <then>No divergence section in output</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P5-03" priority="must-have">
  <narrative>
    As a priority algorithm
    I want to rank memories by relevance, recency, and diversity
    So that the most useful memories appear first
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P5-03-01">
      <given>Memory A (high relevance, 1 hour ago) and Memory B (medium relevance, 5 minutes ago)</given>
      <when>Computing priority</when>
      <then>Both factors contribute to final ranking</then>
    </criterion>
    <criterion id="AC-P5-03-02">
      <given>Memory with weighted_agreement >= 5.0 vs memory with weighted_agreement &lt; 2.5</given>
      <when>Computing priority</when>
      <then>High weighted_agreement memory gets diversity bonus (1.5x)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P5-04" priority="must-have">
  <narrative>
    As an output formatter
    I want to produce structured markdown
    So that context is easy to scan and understand
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P5-04-01">
      <given>Retrieved memories and divergence alerts</given>
      <when>Formatting output</when>
      <then>Output follows specified markdown template with sections</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
<requirement id="REQ-P5-01" story_ref="US-P5-01" priority="must">
  <description>Implement priority ranking algorithm</description>
  <rationale>Need to surface most useful memories first</rationale>
  <formula>
    priority = relevance_score × recency_factor × space_diversity_bonus

    relevance_score = Σ (weight_i × max(0, similarity_i - threshold_i))  // From Phase 3

    recency_factor:
      &lt; 1 hour:   1.3x
      &lt; 1 day:    1.2x
      &lt; 7 days:   1.1x
      &lt; 30 days:  1.0x
      &lt; 90 days:  0.9x
      &gt;= 90 days: 0.8x

    weighted_diversity_bonus:
      weighted_agreement >= 5.0: 1.5x
      weighted_agreement >= 2.5: 1.2x
      weighted_agreement < 2.5: 1.0x
  </formula>
</requirement>

<requirement id="REQ-P5-02" story_ref="US-P5-01" priority="must">
  <description>Implement token budgeting per injection category</description>
  <rationale>Context window is limited; must allocate wisely</rationale>
  <budgets>
    | Priority | Category | Budget | Description |
    |----------|----------|--------|-------------|
    | 1 | DivergenceAlert | 200 | Activity shift warnings |
    | 2 | HighRelevanceCluster | 400 | weighted_agreement >= 2.5 |
    | 3 | SingleSpaceMatch | 300 | Similar in 1-2 semantic spaces |
    | 4 | RecentSession | 200 | Last session summary |
    | 5 | TemporalEnrichment | 50 | Time correlation badges |
    | **Total** | | **1150** | |
  </budgets>
  <behavior>
    1. Allocate tokens in priority order
    2. If category underfills, redistribute to next category
    3. Truncate individual memories to fit, preferring to include more memories partially than fewer completely
  </behavior>
</requirement>

<requirement id="REQ-P5-03" story_ref="US-P5-02" priority="must">
  <description>Implement divergence alert generation and formatting</description>
  <rationale>Divergence must be prominently surfaced</rationale>
  <format>
    ```
    ⚠️ DIVERGENCE DETECTED
    Recent activity in [space_name]: "[recent_memory_summary]"
    Current appears different - similarity: [value]
    ```
  </format>
  <placement>TOP of injected context, before any memories</placement>
</requirement>

<requirement id="REQ-P5-04" story_ref="US-P5-04" priority="must">
  <description>Implement markdown formatter for injected context</description>
  <rationale>Structured output is easier to parse</rationale>
  <template>
    ```markdown
    ## Relevant Context

    ### ⚠️ Note: Activity Shift Detected
    [Divergence alerts if any - ONLY if divergence detected]

    ### Recent Related Work
    [High-relevance memories from clusters - weighted_agreement >= 2.5]

    ### Potentially Related
    [Single-space matches - weighted_agreement &lt; 2.5]

    ### Session Context
    [Last session summary if available]

    ### Temporal Hints
    [Time correlation badges - subtle hints only]
    ```
  </template>
</requirement>

<requirement id="REQ-P5-09" story_ref="US-P5-01" priority="should">
  <description>Implement temporal enrichment badges</description>
  <rationale>Temporal embedders (E2-E4) provide valuable time correlation context</rationale>
  <behavior>
    Temporal embedders (E2-E4) provide enrichment badges:
    - "Same time of day as previous work on X"
    - "Continuation from yesterday's session"
    - "Recent activity (within last hour)"

    These appear as subtle hints, NOT as primary context.
    Budget: ~50 tokens max.
  </behavior>
  <placement>BOTTOM of injected context, after session summary</placement>
  <examples>
    - "You typically work on authentication in the morning"
    - "Continues from yesterday's session on user management"
    - "Recent activity: edited auth.rs 45 minutes ago"
  </examples>
</requirement>

<requirement id="REQ-P5-05" story_ref="US-P5-01" priority="must">
  <description>Implement InjectionPipeline orchestrator</description>
  <rationale>Need unified interface for context injection</rationale>
  <methods>
    - inject_context(query: &amp;str, session_id: &amp;str) -&gt; InjectionResult
    - set_token_budget(budget: TokenBudget)
    - configure_recency_weights(weights: RecencyWeights)
  </methods>
  <pipeline>
    1. Embed query with all 13 embedders
    2. Retrieve candidate memories (top 100 by any-space similarity)
    3. Detect divergence against recent (last 2h / session)
    4. Compute priority scores for candidates
    5. Sort by priority descending
    6. Allocate to categories with token budgeting
    7. Format output markdown
    8. Return InjectionResult
  </pipeline>
</requirement>

<requirement id="REQ-P5-06" story_ref="US-P5-01" priority="must">
  <description>Define InjectionResult structure</description>
  <rationale>Need structured output for hooks to consume</rationale>
  <schema>
    InjectionResult {
      formatted_context: String,       // Markdown output
      total_tokens: usize,
      divergence_alerts: Vec&lt;DivergenceAlert&gt;,
      high_relevance_count: usize,     // weighted_agreement >= 2.5
      single_space_count: usize,       // weighted_agreement &lt; 2.5
      session_summary_included: bool,
      temporal_badges: Vec&lt;String&gt;,    // Time correlation hints
      memories_included: Vec&lt;MemoryId&gt;,
      latency_ms: u64,
    }
  </schema>
</requirement>

<requirement id="REQ-P5-07" story_ref="US-P5-01" priority="must">
  <description>Implement memory summarization for token efficiency</description>
  <rationale>Long memories must be truncated to fit budget</rationale>
  <behavior>
    - Memories &gt;100 tokens: Truncate to first 80 tokens + "..."
    - Prefer including more memories partially than fewer completely
    - Never truncate mid-word or mid-sentence when possible
  </behavior>
</requirement>

<requirement id="REQ-P5-08" story_ref="US-P5-03" priority="must">
  <description>Define RecencyFactor enum and weights</description>
  <rationale>Configurable recency weighting</rationale>
  <schema>
    RecencyFactor {
      VeryRecent,   // &lt;1 hour,   1.3x
      Recent,       // &lt;1 day,    1.2x
      ThisWeek,     // &lt;7 days,   1.1x
      ThisMonth,    // &lt;30 days,  1.0x
      Older,        // &lt;90 days,  0.9x
      Stale,        // &gt;=90 days, 0.8x
    }
  </schema>
  <method>
    fn get_recency_factor(created_at: DateTime&lt;Utc&gt;) -&gt; f32
  </method>
</requirement>
</requirements>

<edge_cases>
<edge_case id="EC-P5-01" req_ref="REQ-P5-01">
  <scenario>No relevant memories found (new system, empty DB)</scenario>
  <expected_behavior>Return empty context with message: "No relevant memories found. This appears to be a new topic."</expected_behavior>
</edge_case>

<edge_case id="EC-P5-02" req_ref="REQ-P5-02">
  <scenario>Token budget exceeded by single high-priority memory</scenario>
  <expected_behavior>Truncate that memory to fit. Log: "Truncated memory [id] from [N] to [M] tokens"</expected_behavior>
</edge_case>

<edge_case id="EC-P5-03" req_ref="REQ-P5-03">
  <scenario>Multiple divergence alerts (different spaces)</scenario>
  <expected_behavior>Include up to 3 most significant alerts (lowest similarity). Group under single "⚠️ Activity Shift Detected" header.</expected_behavior>
</edge_case>

<edge_case id="EC-P5-04" req_ref="REQ-P5-05">
  <scenario>Embedding fails for query</scenario>
  <expected_behavior>Return error: "Failed to embed query: [error]". Do NOT return partial/invalid context.</expected_behavior>
</edge_case>

<edge_case id="EC-P5-05" req_ref="REQ-P5-02">
  <scenario>All memories are from a single category</scenario>
  <expected_behavior>Redistribute unused budget. If only high-relevance exists, use full 900 tokens for them (skipping single-space budget).</expected_behavior>
</edge_case>

<edge_case id="EC-P5-06" req_ref="REQ-P5-07">
  <scenario>Memory content is already very short (&lt;10 tokens)</scenario>
  <expected_behavior>Include in full, no truncation. Count towards budget normally.</expected_behavior>
</edge_case>
</edge_cases>

<error_states>
<error id="ERR-P5-01" http_code="500">
  <condition>Database query fails during retrieval</condition>
  <message>Failed to retrieve memories: [database error]</message>
  <recovery>Return error to hook. Hook should fail gracefully with no context rather than crash.</recovery>
</error>

<error id="ERR-P5-02" http_code="500">
  <condition>Token counting produces invalid result</condition>
  <message>Token counting failed: [tokenizer error]</message>
  <recovery>Use character-based estimation (4 chars = 1 token). Log warning.</recovery>
</error>

<error id="ERR-P5-03" http_code="408">
  <condition>Injection pipeline exceeds timeout (2000ms for UserPromptSubmit)</condition>
  <message>Context injection timed out after 2000ms</message>
  <recovery>Return partial results if available, or empty context. Log timeout with diagnostic info.</recovery>
</error>
</error_states>

<test_plan>
<test_case id="TC-P5-01" type="unit" req_ref="REQ-P5-01">
  <description>Priority correctly combines relevance, recency, diversity</description>
  <inputs>["Memory with relevance=0.5, created 30min ago, 4-space match"]</inputs>
  <expected>priority = 0.5 × 1.3 × 1.2 = 0.78</expected>
</test_case>

<test_case id="TC-P5-02" type="unit" req_ref="REQ-P5-02">
  <description>Token budget enforced per category</description>
  <inputs>["5 high-relevance memories totaling 600 tokens"]</inputs>
  <expected>Category capped at 400 tokens, remaining memories truncated or excluded</expected>
</test_case>

<test_case id="TC-P5-03" type="unit" req_ref="REQ-P5-03">
  <description>Divergence alerts formatted correctly</description>
  <inputs>["DivergenceAlert for E1 with similarity=0.15"]</inputs>
  <expected>Output contains "⚠️ DIVERGENCE DETECTED" and similarity value</expected>
</test_case>

<test_case id="TC-P5-04" type="unit" req_ref="REQ-P5-04">
  <description>Markdown format matches template</description>
  <inputs>["2 divergence alerts, 3 high-relevance, 2 single-space"]</inputs>
  <expected>Output has all sections in correct order</expected>
</test_case>

<test_case id="TC-P5-05" type="integration" req_ref="REQ-P5-05">
  <description>Full pipeline completes in &lt;2000ms</description>
  <inputs>["Query about authentication", "1000 memories in DB"]</inputs>
  <expected>latency_ms &lt; 2000</expected>
</test_case>

<test_case id="TC-P5-06" type="unit" req_ref="REQ-P5-07">
  <description>Memory truncation preserves meaning</description>
  <inputs>["150-token memory"]</inputs>
  <expected>Truncated to ~80 tokens at sentence boundary + "..."</expected>
</test_case>

<test_case id="TC-P5-07" type="unit" req_ref="REQ-P5-08">
  <description>Recency factor computed correctly</description>
  <inputs>["Memory from 3 hours ago"]</inputs>
  <expected>recency_factor = 1.2 (within 1 day)</expected>
</test_case>

<test_case id="TC-P5-08" type="integration" req_ref="REQ-P5-06">
  <description>InjectionResult contains all metadata</description>
  <inputs>["Run full pipeline"]</inputs>
  <expected>Result has formatted_context, total_tokens, divergence_alerts, counts, temporal_badges, latency</expected>
</test_case>

<test_case id="TC-P5-09" type="unit" req_ref="REQ-P5-09">
  <description>Temporal enrichment badges generated from E2-E4 embedders</description>
  <inputs>["Memory from same time of day", "Memory from yesterday's session"]</inputs>
  <expected>Badges generated: "Same time of day as previous work on X", "Continuation from yesterday's session"</expected>
</test_case>

<test_case id="TC-P5-10" type="unit" req_ref="REQ-P5-09">
  <description>Temporal enrichment respects 50 token budget</description>
  <inputs>["5 potential temporal badges totaling 100 tokens"]</inputs>
  <expected>Only top badges included, total &lt;= 50 tokens</expected>
</test_case>

<test_case id="TC-P5-11" type="unit" req_ref="REQ-P5-01">
  <description>Weighted diversity bonus calculated correctly</description>
  <inputs>["Memory with weighted_agreement=5.2", "Memory with weighted_agreement=3.0", "Memory with weighted_agreement=1.5"]</inputs>
  <expected>Bonuses: 1.5x, 1.2x, 1.0x respectively</expected>
</test_case>
</test_plan>

<validation_criteria>
  <criterion>Priority algorithm combines relevance x recency x weighted_diversity_bonus</criterion>
  <criterion>Token budget enforced (~1150 total)</criterion>
  <criterion>Divergence alerts appear at TOP when detected</criterion>
  <criterion>Markdown format matches specified template</criterion>
  <criterion>Pipeline completes in &lt;2000ms</criterion>
  <criterion>Memory truncation preserves readability</criterion>
  <criterion>InjectionResult contains complete metadata</criterion>
  <criterion>Temporal enrichment badges appear at BOTTOM as subtle hints</criterion>
</validation_criteria>
</functional_spec>
```

## Output Format Example

```markdown
## Relevant Context

### ⚠️ Note: Activity Shift Detected
Recent activity in Semantic space: "Discussing authentication implementation with JWT tokens"
Current appears different - similarity: 0.23
This may indicate a context switch to a new topic.

### Recent Related Work
- **[2 hours ago]** Implemented password hashing using bcrypt with cost factor 12. Added validation for minimum 12 character passwords...
- **[Yesterday]** Created User entity with fields: id (UUID), email (unique), passwordHash, emailVerified (default false)...
- **[3 days ago]** Designed authentication flow: login -> validate -> issue JWT -> store refresh token...

### Potentially Related
- **[1 week ago]** Note about session management: Sessions stored in Redis with 24h TTL...

### Session Context
Last session focused on implementing user registration endpoint with email verification.

### Temporal Hints
- Same time of day as previous work on authentication
- Continuation from yesterday's session
```

## Token Budget Allocation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Total Budget: 1150 tokens                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 1: Divergence Alerts (~200 tokens)                    │
│  ┌─────────────────────────────────────────┐                    │
│  │ If alerts exist: allocate up to 200     │                    │
│  │ If no alerts: redistribute to P2        │                    │
│  └─────────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│  Priority 2: High-Relevance Clusters (~400 tokens)              │
│  ┌─────────────────────────────────────────┐                    │
│  │ weighted_agreement >= 2.5, sorted by    │                    │
│  │ priority. Truncate memories if needed   │                    │
│  └─────────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│  Priority 3: Single-Space Matches (~300 tokens)                 │
│  ┌─────────────────────────────────────────┐                    │
│  │ weighted_agreement < 2.5, sorted by     │                    │
│  │ priority. Only if budget remains        │                    │
│  └─────────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│  Priority 4: Session Context (~200 tokens)                      │
│  ┌─────────────────────────────────────────┐                    │
│  │ Last session summary                    │                    │
│  │ Only if budget remains                  │                    │
│  └─────────────────────────────────────────┘                    │
│                         │                                       │
│                         ▼                                       │
│  Priority 5: Temporal Enrichment (~50 tokens)                   │
│  ┌─────────────────────────────────────────┐                    │
│  │ Time correlation badges (E2-E4)         │                    │
│  │ Subtle hints, not primary context       │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Priority Calculation Example

**Memory A**:
- Relevance score: 0.45 (matches in E1 and E7)
- Created: 30 minutes ago
- Weighted agreement: 1.8 (below 2.5 threshold)

**Priority A** = 0.45 x 1.3 (very recent) x 1.0 (weighted_agreement < 2.5) = **0.585**

**Memory B**:
- Relevance score: 0.35 (matches in E1, E5, E7, E10, E13)
- Created: 3 days ago
- Weighted agreement: 5.2 (high cross-space agreement)

**Priority B** = 0.35 x 1.1 (this week) x 1.5 (weighted_agreement >= 5.0) = **0.578**

**Result**: Memory A ranks slightly higher despite lower relevance, due to recency.

## Weighted Diversity Bonus Tiers

| Weighted Agreement | Bonus | Rationale |
|--------------------|-------|-----------|
| >= 5.0 | 1.5x | Strong cross-space agreement indicates high confidence |
| >= 2.5 | 1.2x | Moderate agreement, classified as HighRelevanceCluster |
| < 2.5 | 1.0x | Low agreement, classified as SingleSpaceMatch |
