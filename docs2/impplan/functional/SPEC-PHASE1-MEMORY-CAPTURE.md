# Functional Specification: Phase 1 - Memory Capture System

```xml
<functional_spec id="SPEC-PHASE1" version="1.0">
<metadata>
  <title>Memory Capture System</title>
  <status>approved</status>
  <owner>Context Graph Team</owner>
  <created>2026-01-16</created>
  <last_updated>2026-01-16</last_updated>
  <implements>impplan.md Part 1</implements>
  <depends_on>
    <spec_ref>SPEC-PHASE0</spec_ref>
  </depends_on>
  <related_specs>
    <spec_ref>SPEC-PHASE2</spec_ref>
    <spec_ref>SPEC-PHASE6</spec_ref>
  </related_specs>
</metadata>

<overview>
Implement a unified memory capture system that collects content from three sources:
1. **Hook Descriptions**: Claude's description of what it's doing at every hook event
2. **Claude Responses**: End-of-session answers and significant responses
3. **MD File Chunks**: Content from created/modified .md files in the project directory

Each captured memory is stored with all 13 embeddings (TeleologicalArray), source type tracking, and optional chunk metadata for file-based memories.

**Problem Solved**: Currently the system lacks automatic memory capture from Claude's activities. Memories must be manually injected. This phase enables autonomous memory collection.

**Who Benefits**: Claude instances that automatically build context from their activities; users who get better context injection from accumulated memories.
</overview>

<user_stories>
<story id="US-P1-01" priority="must-have">
  <narrative>
    As a Claude instance
    I want my tool descriptions automatically captured as memories
    So that my activities are recorded for future context retrieval
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P1-01-01">
      <given>A PreToolUse hook fires with a tool description</given>
      <when>The hook completes</when>
      <then>A new Memory is created with source=HookDescription</then>
    </criterion>
    <criterion id="AC-P1-01-02">
      <given>A PostToolUse hook fires with tool output summary</given>
      <when>The hook completes</when>
      <then>A new Memory is created with source=HookDescription and includes output context</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P1-02" priority="must-have">
  <narrative>
    As a Claude instance
    I want my session responses captured at session end
    So that significant outputs are preserved for future reference
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P1-02-01">
      <given>A SessionEnd hook fires with response summary</given>
      <when>The hook processes the response</when>
      <then>A new Memory is created with source=ClaudeResponse</then>
    </criterion>
    <criterion id="AC-P1-02-02">
      <given>A Stop hook fires with Claude's final response</given>
      <when>The hook completes</when>
      <then>A new Memory is created with source=ClaudeResponse</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P1-03" priority="must-have">
  <narrative>
    As a knowledge graph
    I want to automatically capture content from .md files
    So that project documentation becomes part of my memory
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P1-03-01">
      <given>A new .md file is created in the watched directory</given>
      <when>The file watcher detects the creation</when>
      <then>The file is chunked and each chunk becomes a Memory with source=MDFileChunk</then>
    </criterion>
    <criterion id="AC-P1-03-02">
      <given>An existing .md file is modified</given>
      <when>The file watcher detects the modification</when>
      <then>The file is re-chunked and memories are updated (old chunks for that file removed, new chunks added)</then>
    </criterion>
    <criterion id="AC-P1-03-03">
      <given>A 1000-word .md file</given>
      <when>The chunker processes it</when>
      <then>Approximately 5-6 chunks are created (200 words each, 50-word overlap)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P1-04" priority="must-have">
  <narrative>
    As a memory
    I want to have a clear source type and session association
    So that retrieval can filter by source and session
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P1-04-01">
      <given>Any captured memory</given>
      <when>Inspecting its fields</when>
      <then>It has a valid source enum (HookDescription, ClaudeResponse, or MDFileChunk)</then>
    </criterion>
    <criterion id="AC-P1-04-02">
      <given>A memory captured during a session</given>
      <when>Inspecting its fields</when>
      <then>It has the correct session_id matching the current session</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
<requirement id="REQ-P1-01" story_ref="US-P1-04" priority="must">
  <description>Define new Memory struct with source tracking and all 13 embeddings</description>
  <rationale>Current MemoryNode only has single 1536D embedding; need full teleological array plus source</rationale>
  <schema>
    Memory {
      id: UUID,
      content: String,
      source: MemorySource (enum: HookDescription | ClaudeResponse | MDFileChunk),
      created_at: DateTime&lt;Utc&gt;,
      session_id: String,
      teleological_array: TeleologicalArray,  // All 13 embeddings
      chunk_metadata: Option&lt;ChunkMetadata&gt;,
      importance: f32,
      access_count: u64,
      last_accessed: DateTime&lt;Utc&gt;
    }
  </schema>
</requirement>

<requirement id="REQ-P1-02" story_ref="US-P1-04" priority="must">
  <description>Define MemorySource enum for source type tracking</description>
  <rationale>Need to distinguish between hook captures, response captures, and file chunks</rationale>
  <schema>
    MemorySource {
      HookDescription { hook_type: HookEventType, tool_name: Option&lt;String&gt; },
      ClaudeResponse { response_type: ResponseType },
      MDFileChunk { file_path: String }
    }
  </schema>
</requirement>

<requirement id="REQ-P1-03" story_ref="US-P1-03" priority="must">
  <description>Define ChunkMetadata for file-based memories</description>
  <rationale>Need to track which file and position each chunk came from</rationale>
  <schema>
    ChunkMetadata {
      file_path: String,
      chunk_index: u32,
      total_chunks: u32,
      word_offset: u32,
      file_hash: String  // SHA-256 of file content for change detection
    }
  </schema>
</requirement>

<requirement id="REQ-P1-04" story_ref="US-P1-01" priority="must">
  <description>Implement HookDescriptionCapture for PreToolUse and PostToolUse hooks</description>
  <rationale>Tool descriptions contain valuable context about Claude's activities</rationale>
  <behavior>
    1. Extract tool description from hook input
    2. Generate all 13 embeddings via MultiArrayProvider
    3. Create Memory with source=HookDescription
    4. Store Memory to RocksDB
  </behavior>
</requirement>

<requirement id="REQ-P1-05" story_ref="US-P1-02" priority="must">
  <description>Implement ClaudeResponseCapture for Stop and SessionEnd hooks</description>
  <rationale>Claude's responses contain synthesized knowledge worth preserving</rationale>
  <behavior>
    1. Extract response summary from hook input
    2. If response &gt; 500 words, chunk using same chunker as MD files
    3. Generate all 13 embeddings for each chunk
    4. Create Memory(s) with source=ClaudeResponse
    5. Store Memory(s) to RocksDB
  </behavior>
</requirement>

<requirement id="REQ-P1-06" story_ref="US-P1-03" priority="must">
  <description>Implement MDFileWatcher for file system monitoring</description>
  <rationale>MD files in the project should be automatically indexed</rationale>
  <behavior>
    1. Watch configured directory for .md file create/modify events
    2. On event, read file content
    3. Compute file hash (SHA-256)
    4. If hash changed from stored value, re-chunk
    5. Remove old chunks for that file_path
    6. Create new Memory(s) for each chunk with source=MDFileChunk
    7. Store Memory(s) to RocksDB
  </behavior>
</requirement>

<requirement id="REQ-P1-07" story_ref="US-P1-03" priority="must">
  <description>Implement TextChunker with 200-word chunks and 50-word overlap</description>
  <rationale>Chunking enables fine-grained retrieval and fits context windows</rationale>
  <params>
    chunk_size: 200 words
    overlap: 50 words (25%)
    preserve_sentences: true (avoid mid-sentence splits when possible)
  </params>
  <behavior>
    1. Split text into words
    2. Create chunks of 200 words with 50-word overlap
    3. Adjust chunk boundaries to nearest sentence end when within 20 words
    4. Return Vec&lt;TextChunk&gt; with offset tracking
  </behavior>
</requirement>

<requirement id="REQ-P1-08" story_ref="US-P1-04" priority="must">
  <description>Implement MemoryStore trait for Memory persistence</description>
  <rationale>Need CRUD operations for the new Memory type</rationale>
  <methods>
    - store(memory: Memory) -> Result&lt;MemoryId&gt;
    - store_batch(memories: Vec&lt;Memory&gt;) -> Result&lt;Vec&lt;MemoryId&gt;&gt;
    - get(id: MemoryId) -> Result&lt;Option&lt;Memory&gt;&gt;
    - delete(id: MemoryId) -> Result&lt;()&gt;
    - delete_by_file_path(path: &amp;str) -> Result&lt;u64&gt;  // For re-chunking
    - find_by_session(session_id: &amp;str) -> Result&lt;Vec&lt;Memory&gt;&gt;
    - find_by_source(source: MemorySource) -> Result&lt;Vec&lt;Memory&gt;&gt;
  </methods>
</requirement>
</requirements>

<edge_cases>
<edge_case id="EC-P1-01" req_ref="REQ-P1-04">
  <scenario>Hook fires but description is empty or only whitespace</scenario>
  <expected_behavior>Memory is NOT created. Warning logged: "Skipping empty hook description for [hook_type]"</expected_behavior>
</edge_case>

<edge_case id="EC-P1-02" req_ref="REQ-P1-05">
  <scenario>Claude response is extremely long (&gt;10000 words)</scenario>
  <expected_behavior>Response is chunked into ~50 chunks. Each chunk stored as separate Memory with sequential chunk_index. All chunks share same response_id for grouping.</expected_behavior>
</edge_case>

<edge_case id="EC-P1-03" req_ref="REQ-P1-06">
  <scenario>MD file is binary or contains invalid UTF-8</scenario>
  <expected_behavior>File is skipped with error log: "Skipping non-UTF8 file: [path]". System continues processing other files.</expected_behavior>
</edge_case>

<edge_case id="EC-P1-04" req_ref="REQ-P1-06">
  <scenario>MD file is extremely large (&gt;1MB)</scenario>
  <expected_behavior>File is processed but with warning: "Large file [path] will create [N] chunks". Processing continues normally. No arbitrary size limit.</expected_behavior>
</edge_case>

<edge_case id="EC-P1-05" req_ref="REQ-P1-07">
  <scenario>Text has very long sentences (&gt;200 words)</scenario>
  <expected_behavior>Chunk boundary placed at 200 words regardless of sentence boundary. Log warning: "Chunk split mid-sentence at word [N]"</expected_behavior>
</edge_case>

<edge_case id="EC-P1-06" req_ref="REQ-P1-06">
  <scenario>Watched directory does not exist</scenario>
  <expected_behavior>Error returned on watcher initialization: "Watch directory does not exist: [path]". System MUST NOT silently fail.</expected_behavior>
</edge_case>

<edge_case id="EC-P1-07" req_ref="REQ-P1-06">
  <scenario>File is deleted while being processed</scenario>
  <expected_behavior>Race condition handled gracefully. If file disappears mid-read, error logged and file skipped. Existing chunks for that path retained until next successful read.</expected_behavior>
</edge_case>

<edge_case id="EC-P1-08" req_ref="REQ-P1-08">
  <scenario>Storage is full or RocksDB write fails</scenario>
  <expected_behavior>Error propagated to caller: "Failed to store memory: [RocksDB error]". Memory is NOT silently lost. Retry logic at caller's discretion.</expected_behavior>
</edge_case>
</edge_cases>

<error_states>
<error id="ERR-P1-01" http_code="500">
  <condition>Embedding generation fails for any of 13 embedders</condition>
  <message>Failed to generate embeddings for memory: [embedder] failed with [error]</message>
  <recovery>Memory is NOT stored. Error logged with full context. Caller should retry or skip this content.</recovery>
</error>

<error id="ERR-P1-02" http_code="500">
  <condition>File watcher inotify/kqueue limit reached</condition>
  <message>File watcher limit reached. Cannot watch additional files.</message>
  <recovery>Log error with instructions to increase inotify_max_user_watches. Continue watching existing files.</recovery>
</error>

<error id="ERR-P1-03" http_code="400">
  <condition>Session ID not provided for hook capture</condition>
  <message>session_id is required for memory capture</message>
  <recovery>Caller must provide valid session_id</recovery>
</error>
</error_states>

<test_plan>
<test_case id="TC-P1-01" type="unit" req_ref="REQ-P1-01">
  <description>Memory struct has all required fields</description>
  <inputs>["Create Memory with all fields"]</inputs>
  <expected>Memory compiles and serializes correctly</expected>
</test_case>

<test_case id="TC-P1-02" type="unit" req_ref="REQ-P1-07">
  <description>Chunker produces correct chunk count</description>
  <inputs>["400-word text"]</inputs>
  <expected>3 chunks: words 0-199, 150-349, 300-399 (with overlap)</expected>
</test_case>

<test_case id="TC-P1-03" type="unit" req_ref="REQ-P1-07">
  <description>Chunker respects sentence boundaries when possible</description>
  <inputs>["Text with sentence ending at word 195"]</inputs>
  <expected>Chunk boundary at word 195, not word 200</expected>
</test_case>

<test_case id="TC-P1-04" type="integration" req_ref="REQ-P1-04">
  <description>Hook description capture stores memory</description>
  <inputs>["PreToolUse hook with description='Reading file config.yaml'"]</inputs>
  <expected>Memory stored with source=HookDescription, content matches, 13 embeddings present</expected>
</test_case>

<test_case id="TC-P1-05" type="integration" req_ref="REQ-P1-06">
  <description>MD file creation triggers memory capture</description>
  <inputs>["Create test.md with 500 words in watched directory"]</inputs>
  <expected>3 Memory entries created with source=MDFileChunk, chunk_metadata populated</expected>
</test_case>

<test_case id="TC-P1-06" type="integration" req_ref="REQ-P1-06">
  <description>MD file modification re-chunks</description>
  <inputs>["Modify test.md to have 1000 words"]</inputs>
  <expected>Old 3 chunks deleted, new 6 chunks created</expected>
</test_case>

<test_case id="TC-P1-07" type="integration" req_ref="REQ-P1-08">
  <description>Memory can be retrieved by session_id</description>
  <inputs>["Store 5 memories with session_id='test-session'"]</inputs>
  <expected>find_by_session returns all 5 memories</expected>
</test_case>

<test_case id="TC-P1-08" type="unit" req_ref="REQ-P1-07">
  <description>Empty text produces no chunks</description>
  <inputs>["Empty string"]</inputs>
  <expected>Empty Vec returned, no error</expected>
</test_case>

<test_case id="TC-P1-09" type="integration" req_ref="REQ-P1-04">
  <description>Hook capture generates all 13 embeddings</description>
  <inputs>["Capture hook description"]</inputs>
  <expected>Memory.teleological_array has non-empty values for all 13 embedders</expected>
</test_case>
</test_plan>

<validation_criteria>
  <criterion>Memory struct matches schema with source enum, session_id, teleological_array</criterion>
  <criterion>Chunker produces 200-word chunks with 50-word overlap</criterion>
  <criterion>Hook descriptions captured on PreToolUse and PostToolUse</criterion>
  <criterion>Claude responses captured on Stop and SessionEnd</criterion>
  <criterion>MD file watcher detects create/modify events</criterion>
  <criterion>All 13 embeddings generated for every captured memory</criterion>
  <criterion>Memories can be queried by session_id and source type</criterion>
</validation_criteria>
</functional_spec>
```

## Memory Schema Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Memory                                                          │
├─────────────────────────────────────────────────────────────────┤
│ id: UUID                                                        │
│ content: String                                                 │
│ source: MemorySource                                            │
│   ├─ HookDescription { hook_type, tool_name }                   │
│   ├─ ClaudeResponse { response_type }                           │
│   └─ MDFileChunk { file_path }                                  │
│ created_at: DateTime<Utc>                                       │
│ session_id: String                                              │
│ teleological_array: TeleologicalArray [E1..E13]                 │
│ chunk_metadata: Option<ChunkMetadata>                           │
│   ├─ file_path: String                                          │
│   ├─ chunk_index: u32                                           │
│   ├─ total_chunks: u32                                          │
│   ├─ word_offset: u32                                           │
│   └─ file_hash: String                                          │
│ importance: f32                                                 │
│ access_count: u64                                               │
│ last_accessed: DateTime<Utc>                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Hook Lifecycle Integration

| Hook | Captures | CLI Command | Timeout |
|------|----------|-------------|---------|
| SessionStart | - | `session start` | 5000ms |
| UserPromptSubmit | Prompt text | `inject-context` | 2000ms |
| PreToolUse | Tool description | `capture-memory --source hook` | 500ms |
| PostToolUse | Tool description + output summary | `capture-memory --source hook` | 3000ms |
| Stop | Claude's response summary | `capture-memory --source response` | 3000ms |
| SessionEnd | Session summary | `session end` | 30000ms |

## Chunking Example

**Input**: 450-word document

**Output** (200-word chunks, 50-word overlap):
```
Chunk 0: words 0-199   (200 words)
Chunk 1: words 150-349 (200 words) - 50-word overlap with chunk 0
Chunk 2: words 300-449 (150 words) - 50-word overlap with chunk 1
```

Total: 3 chunks for 450 words
