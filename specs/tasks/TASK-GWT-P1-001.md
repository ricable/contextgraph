# TASK-GWT-P1-001: Implement SELF_EGO_NODE Persistence Layer

## Metadata
| Field | Value |
|-------|-------|
| **Task ID** | TASK-GWT-P1-001 |
| **Title** | Wire SELF_EGO_NODE Persistence to RocksDB |
| **Status** | Completed |
| **Priority** | P1 |
| **Layer** | Foundation (Layer 1) |
| **Parent Spec** | SPEC-GWT-001 |
| **Estimated Effort** | 4 hours |
| **Created** | 2026-01-11 |

---

## 1. Input Context Files

| File | Purpose | Key Sections |
|------|---------|--------------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | SelfEgoNode struct with Serde annotations | Lines 1-743, SelfEgoNode struct |
| `crates/context-graph-core/src/gwt/mod.rs` | GwtSystem orchestration | Lines 125-303, GwtSystem impl |
| `crates/context-graph-core/src/storage/mod.rs` | RocksDB storage layer | Column family definitions |
| `docs2/constitution.yaml` | SELF_EGO_NODE requirements | Lines 366-370, self_ego_node section |

---

## 2. Problem Statement

The `SelfEgoNode` struct has Serde `Serialize`/`Deserialize` annotations but lacks a persistence layer to RocksDB. Per constitution.yaml, the SELF_EGO_NODE must persist across sessions to maintain identity continuity (IC).

Current state:
- `SelfEgoNode`: Has `#[derive(Serialize, Deserialize)]`
- RocksDB: Has column family infrastructure
- **GAP**: No `CF_EGO_NODE` column family, no save/load methods

---

## 3. Definition of Done

### 3.1 Required Signatures

```rust
// In storage/mod.rs or storage/column_families.rs
pub const CF_EGO_NODE: &str = "ego_node";

// In gwt/ego_node.rs
impl SelfEgoNode {
    /// Persist to RocksDB
    pub async fn persist(&self, db: &RocksDbHandle) -> CoreResult<()>;

    /// Load from RocksDB, returns None if not found
    pub async fn load(db: &RocksDbHandle) -> CoreResult<Option<Self>>;

    /// Check if persisted state exists
    pub async fn exists(db: &RocksDbHandle) -> CoreResult<bool>;
}

// In gwt/mod.rs GwtSystem
impl GwtSystem {
    /// Load or create SELF_EGO_NODE on initialization
    pub async fn load_or_create_ego_node(db: &RocksDbHandle) -> CoreResult<SelfEgoNode>;

    /// Persist current ego node state
    pub async fn persist_ego_node(&self, db: &RocksDbHandle) -> CoreResult<()>;
}
```

### 3.2 Required Tests

```rust
#[tokio::test]
async fn test_ego_node_persistence_roundtrip() {
    // Create ego node, persist, load, verify equality
}

#[tokio::test]
async fn test_ego_node_load_missing_returns_none() {
    // Load from empty DB returns None
}

#[tokio::test]
async fn test_ego_node_survives_restart() {
    // Persist, drop, recreate, load - data matches
}

#[tokio::test]
async fn test_identity_continuity_across_persist() {
    // IC calculation works correctly after load
}
```

---

## 4. Files to Create/Modify

| File | Action | Changes |
|------|--------|---------|
| `crates/context-graph-core/src/storage/column_families.rs` | Modify | Add `CF_EGO_NODE` constant |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Modify | Add `persist()`, `load()`, `exists()` methods |
| `crates/context-graph-core/src/gwt/mod.rs` | Modify | Add `load_or_create_ego_node()`, wire persistence on init |
| `crates/context-graph-core/src/gwt/persistence.rs` | Create | Dedicated persistence helpers (optional) |
| `crates/context-graph-core/tests/gwt_persistence_tests.rs` | Create | Integration tests for persistence |

---

## 5. Implementation Steps

### Step 1: Add Column Family
```rust
// storage/column_families.rs
pub const CF_EGO_NODE: &str = "ego_node";

// Add to column family list
pub fn all_column_families() -> Vec<&'static str> {
    vec![
        CF_MEMORIES,
        CF_EMBEDDINGS,
        CF_EGO_NODE,  // New
        // ...
    ]
}
```

### Step 2: Implement Persistence Methods
```rust
// gwt/ego_node.rs
use bincode;

const EGO_NODE_KEY: &[u8] = b"SELF_EGO_NODE";

impl SelfEgoNode {
    pub async fn persist(&self, db: &RocksDbHandle) -> CoreResult<()> {
        let cf = db.cf_handle(CF_EGO_NODE)?;
        let bytes = bincode::serialize(self)
            .map_err(|e| CoreError::Serialization(e.to_string()))?;
        db.put_cf(&cf, EGO_NODE_KEY, &bytes)?;
        tracing::debug!("Persisted SELF_EGO_NODE (IC={:.3})", self.identity_continuity());
        Ok(())
    }

    pub async fn load(db: &RocksDbHandle) -> CoreResult<Option<Self>> {
        let cf = db.cf_handle(CF_EGO_NODE)?;
        match db.get_cf(&cf, EGO_NODE_KEY)? {
            Some(bytes) => {
                let node: Self = bincode::deserialize(&bytes)
                    .map_err(|e| CoreError::Deserialization(e.to_string()))?;
                tracing::info!("Loaded SELF_EGO_NODE (IC={:.3})", node.identity_continuity());
                Ok(Some(node))
            }
            None => Ok(None),
        }
    }

    pub async fn exists(db: &RocksDbHandle) -> CoreResult<bool> {
        let cf = db.cf_handle(CF_EGO_NODE)?;
        Ok(db.get_cf(&cf, EGO_NODE_KEY)?.is_some())
    }
}
```

### Step 3: Wire into GwtSystem Initialization
```rust
// gwt/mod.rs
impl GwtSystem {
    pub async fn new(db: Option<&RocksDbHandle>) -> CoreResult<Self> {
        let ego_node = if let Some(db) = db {
            Self::load_or_create_ego_node(db).await?
        } else {
            SelfEgoNode::new()
        };

        Ok(Self {
            ego_node: Arc::new(RwLock::new(ego_node)),
            // ... other fields
        })
    }

    pub async fn load_or_create_ego_node(db: &RocksDbHandle) -> CoreResult<SelfEgoNode> {
        match SelfEgoNode::load(db).await? {
            Some(node) => {
                tracing::info!("Restored SELF_EGO_NODE from persistence");
                Ok(node)
            }
            None => {
                tracing::info!("Creating new SELF_EGO_NODE");
                Ok(SelfEgoNode::new())
            }
        }
    }

    pub async fn persist_ego_node(&self, db: &RocksDbHandle) -> CoreResult<()> {
        let node = self.ego_node.read().await;
        node.persist(db).await
    }
}
```

---

## 6. Validation Criteria

| Criterion | Test | Expected |
|-----------|------|----------|
| Column family exists | `db.cf_handle(CF_EGO_NODE)` | Returns valid handle |
| Persist succeeds | `ego_node.persist(db).await` | Ok(()) |
| Load returns data | `SelfEgoNode::load(db).await` | Some(node) with correct fields |
| Missing returns None | Load from empty DB | Ok(None) |
| IC preserved | Load, check IC | IC matches pre-persist value |
| Survives restart | Persist, drop DB, reopen, load | Data matches |

### Verification Commands

```bash
# Run persistence tests
cargo test --package context-graph-core ego_node_persist --no-fail-fast

# Run integration tests
cargo test --package context-graph-core --test gwt_persistence_tests

# Verify column family creation
cargo test --package context-graph-core storage::column_families
```

---

## 7. Dependencies

### Upstream
- RocksDB crate (0.21+)
- bincode crate (1.3+)
- serde crate (1.0+)

### Downstream
- TASK-GWT-P1-002 depends on this for full event wiring
- TASK-GWT-P1-003 depends on this for integration tests

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Schema migration needed | Low | Medium | Use versioned serialization |
| DB corruption | Low | High | Implement graceful degradation |
| Concurrent access | Medium | Medium | Use RwLock, atomic operations |

---

## 9. Notes

- The `SelfEgoNode` already has full Serde support, so serialization is straightforward
- Consider adding a schema version field for future migrations
- Persistence should be triggered after every `process_action_awareness()` call
- On IC < 0.5, persist before triggering IdentityCritical event
