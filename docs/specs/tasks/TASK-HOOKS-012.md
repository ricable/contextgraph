# TASK-HOOKS-012: Implement Session Identity Snapshot Persistence

```xml
<task_spec id="TASK-HOOKS-012" version="1.0">
<metadata>
  <title>Implement Session Identity Snapshot Persistence</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>12</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-09</requirement_ref>
    <requirement_ref>REQ-HOOKS-10</requirement_ref>
    <requirement_ref>REQ-HOOKS-11</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-006</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>3.0</estimated_hours>
</metadata>

<context>
Session identity must persist across Claude Code sessions. When session_end fires,
the current SessionIdentitySnapshot is serialized and stored. This enables identity
continuity across conversation boundaries.

Storage location: .claude/identity/snapshots/{session_id}.json
Index file: .claude/identity/snapshots/index.json (tracks latest, session history)
</context>

<input_context_files>
  <file purpose="snapshot_struct">docs/specs/technical/TECH-HOOKS.md#data_models</file>
  <file purpose="storage_types">crates/context-graph-storage/src/types.rs</file>
  <file purpose="session_identity">crates/context-graph-gwt/src/identity/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>SessionIdentitySnapshot struct exists (TASK-HOOKS-001)</check>
  <check>session_end hook handler exists (TASK-HOOKS-007)</check>
  <check>.claude/ directory structure established</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SnapshotStore trait for snapshot persistence
    - Implement FileSnapshotStore for .claude/identity/snapshots/
    - Create snapshot index management (latest pointer, history)
    - Implement serialize/deserialize for SessionIdentitySnapshot
    - Handle concurrent write protection (file locking)
    - Support snapshot rotation (keep last N snapshots)
  </in_scope>
  <out_of_scope>
    - Snapshot restoration (TASK-HOOKS-013)
    - Remote/cloud snapshot storage (future feature)
    - Snapshot encryption (future feature)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/identity/snapshot_store.rs">
      pub trait SnapshotStore: Send + Sync {
          async fn save(&amp;self, snapshot: &amp;SessionIdentitySnapshot) -> Result&lt;SnapshotId, StoreError&gt;;
          async fn load(&amp;self, id: &amp;SnapshotId) -> Result&lt;SessionIdentitySnapshot, StoreError&gt;;
          async fn load_latest(&amp;self) -> Result&lt;Option&lt;SessionIdentitySnapshot&gt;, StoreError&gt;;
          async fn list(&amp;self, limit: usize) -> Result&lt;Vec&lt;SnapshotMetadata&gt;, StoreError&gt;;
          async fn delete(&amp;self, id: &amp;SnapshotId) -> Result&lt;(), StoreError&gt;;
      }

      pub struct FileSnapshotStore {
          base_path: PathBuf,
          max_snapshots: usize,
      }

      impl FileSnapshotStore {
          pub fn new(base_path: PathBuf, max_snapshots: usize) -> Self;
      }
    </signature>
    <signature file="crates/context-graph-cli/src/identity/types.rs">
      pub struct SnapshotId(pub String);

      pub struct SnapshotMetadata {
          pub id: SnapshotId,
          pub session_id: String,
          pub created_at: DateTime&lt;Utc&gt;,
          pub ic_value: f64,
          pub file_size_bytes: u64,
      }

      pub struct SnapshotIndex {
          pub latest: Option&lt;SnapshotId&gt;,
          pub snapshots: Vec&lt;SnapshotMetadata&gt;,
          pub version: u32,
      }
    </signature>
  </signatures>

  <constraints>
    - Snapshots must be valid JSON (serde_json)
    - File operations must use atomic writes (write to temp, rename)
    - Index file must be updated atomically with snapshot
    - Must handle disk full errors gracefully
    - Must validate snapshot integrity on save
    - Maximum 50 snapshots retained (configurable)
  </constraints>

  <verification>
    - cargo test --package context-graph-cli snapshot_store
    - Verify atomic write behavior under concurrent access
    - Verify snapshot rotation when limit exceeded
    - Verify index consistency after save/delete operations
  </verification>
</definition_of_done>

<pseudo_code>
FileSnapshotStore:
  base_path: PathBuf
  max_snapshots: usize

  save(snapshot):
    // Generate snapshot ID from session_id + timestamp
    id = SnapshotId(format!("{}-{}", snapshot.session_id, timestamp_ms()))

    // Serialize snapshot to JSON
    json = serde_json::to_string_pretty(snapshot)?

    // Atomic write: temp file -> rename
    temp_path = base_path / "temp" / format!("{}.json.tmp", id)
    final_path = base_path / format!("{}.json", id)

    write_file(temp_path, json)?
    rename(temp_path, final_path)?

    // Update index atomically
    index = load_index()?
    index.latest = Some(id.clone())
    index.snapshots.push(SnapshotMetadata {
      id: id.clone(),
      session_id: snapshot.session_id,
      created_at: now(),
      ic_value: snapshot.identity_continuity,
      file_size_bytes: json.len()
    })

    // Rotate old snapshots if needed
    if index.snapshots.len() > max_snapshots:
      old = index.snapshots.remove(0)
      delete_file(base_path / format!("{}.json", old.id))?

    save_index(index)?
    return Ok(id)

  load_latest():
    index = load_index()?
    if index.latest.is_none():
      return Ok(None)
    return load(index.latest.unwrap())

  load(id):
    path = base_path / format!("{}.json", id)
    json = read_file(path)?
    snapshot = serde_json::from_str(json)?
    return Ok(snapshot)
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/identity/snapshot_store.rs">
    SnapshotStore trait and FileSnapshotStore implementation
  </file>
  <file path="crates/context-graph-cli/src/identity/types.rs">
    SnapshotId, SnapshotMetadata, SnapshotIndex types
  </file>
  <file path="crates/context-graph-cli/tests/identity/snapshot_store_test.rs">
    Integration tests for snapshot persistence (real file operations)
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/identity/mod.rs">
    Export snapshot_store and types modules
  </file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli snapshot_store</command>
</test_commands>
</task_spec>
```
