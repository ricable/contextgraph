# Task Specification: Meta-Learning Event Log Implementation

**Task ID:** TASK-METAUTL-P0-004
**Version:** 2.0.0
**Status:** NOT STARTED - Ready for Implementation
**Layer:** Surface (Layer 3)
**Sequence:** 4
**Priority:** P0 (Critical)
**Estimated Complexity:** Medium

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description |
|----------------|-------------|
| REQ-METAUTL-010 | Log all meta-learning events with full context |
| REQ-METAUTL-011 | Event log SHALL support time-range queries |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| TASK-METAUTL-P0-001 | Core types (MetaLearningEvent) | ✅ COMPLETE |
| TASK-METAUTL-P0-002 | Lambda adjustment (event source) | ❌ NOT STARTED |
| TASK-METAUTL-P0-003 | Escalation (event source) | ❌ NOT STARTED |

### 1.3 Blocked By

- ~~TASK-METAUTL-P0-003 (all event sources must exist)~~
- **NOTE**: This task CAN proceed independently since MetaLearningEvent exists.
  Event sources (TASK-002, TASK-003) can emit events to the log later.

### 1.4 Implementation Note

**Timestamp Migration Required**: The existing `MetaLearningEvent` in `types.rs` uses `std::time::Instant`
for timestamps, which cannot be serialized to JSON. This task MUST:
1. Add a serializable timestamp field (`DateTime<Utc>`) alongside or replacing `Instant`
2. OR create a separate `SerializableMetaLearningEvent` for logging purposes
3. Ensure backwards compatibility with existing code using `Instant`

---

## 2. Context

This task implements the meta-learning event log for introspection and debugging. The log captures all self-correction events including:

- Lambda adjustments (MetaLearningEventType::LambdaAdjustment)
- Escalation triggers (MetaLearningEventType::BayesianEscalation)
- Accuracy alerts (MetaLearningEventType::AccuracyAlert)
- Accuracy recovery (MetaLearningEventType::AccuracyRecovery)
- Weight clamping (MetaLearningEventType::WeightClamped)

The log supports time-range queries and provides the foundation for the `get_meta_learning_log` MCP tool.

### 2.1 Current Implementation State

**What Exists:**
- `MetaLearningEvent` struct in `handlers/core/types.rs`
- `MetaLearningEventType` enum with 5 variants
- `Domain` enum for domain-specific filtering
- Factory methods: `lambda_adjustment()`, `bayesian_escalation()`, `weight_clamped()`

**What's Missing:**
- `MetaLearningEventLog` struct (this task)
- `MetaLearningLogger` trait (this task)
- Time-range queries, FIFO eviction, JSON serialization

---

## 3. Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/core/types.rs` | MetaLearningEvent, MetaLearningEventType, Domain |
| `crates/context-graph-mcp/src/handlers/core/meta_utl_tracker.rs` | MetaUtlTracker (event producer) |
| `crates/context-graph-mcp/src/handlers/core/mod.rs` | Module declarations |
| `specs/functional/SPEC-METAUTL-001.md` | Logging requirements |
| `docs2/constitution.yaml` | NORTH-016 bounds for context |

---

## 4. Scope

### 4.1 In Scope

- Create `MetaLearningEventLog` struct
- Implement `MetaLearningLogger` trait
- Implement time-range query
- Implement FIFO eviction with configurable retention
- Implement event filtering by type
- Implement JSON serialization for persistence
- Unit tests for log operations

### 4.2 Out of Scope

- Database persistence (future enhancement)
- MCP tool wiring (TASK-METAUTL-P0-005)
- Integration with correction/escalation (TASK-METAUTL-P0-006)

---

## 5. Prerequisites

| Check | Description | Status |
|-------|-------------|--------|
| [x] | TASK-METAUTL-P0-001 completed | ✅ Done |
| [ ] | TASK-METAUTL-P0-002 completed | ⏳ Not required (event source) |
| [ ] | TASK-METAUTL-P0-003 completed | ⏳ Not required (event source) |
| [x] | MetaLearningEvent type exists | ✅ In types.rs |
| [x] | MetaLearningEventType enum exists | ✅ In types.rs |
| [x] | Domain enum exists | ✅ In types.rs |
| [ ] | chrono crate available | ✅ Already in MCP crate |

---

## 6. Definition of Done

### 6.1 Required Signatures

#### File: `crates/context-graph-mcp/src/handlers/core/event_log.rs`

```rust
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::types::{MetaLearningEvent, MetaLearningEventType, Domain};

// NOTE: MetaLearningEvent uses Instant for runtime, but we need DateTime<Utc> for JSON.
// Create a SerializableMetaLearningEvent wrapper for persistence.

/// Default maximum events in memory
pub const DEFAULT_MAX_EVENTS: usize = 1000;

/// Default retention period in days
pub const DEFAULT_RETENTION_DAYS: u32 = 7;

/// Trait for meta-learning event logging
pub trait MetaLearningLogger {
    /// Log a meta-learning event
    fn log_event(&mut self, event: MetaLearningEvent) -> UtlResult<()>;

    /// Query events by time range
    fn query_by_time(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&MetaLearningEvent>;

    /// Query events by type
    fn query_by_type(&self, event_type: MetaLearningEventType) -> Vec<&MetaLearningEvent>;

    /// Query events by domain
    fn query_by_domain(&self, domain: Domain) -> Vec<&MetaLearningEvent>;

    /// Get recent events
    fn recent_events(&self, count: usize) -> Vec<&MetaLearningEvent>;

    /// Get total event count
    fn event_count(&self) -> usize;

    /// Clear all events
    fn clear(&mut self);
}

/// Query parameters for event log
#[derive(Debug, Clone, Default)]
pub struct EventLogQuery {
    /// Start time filter (inclusive)
    pub start_time: Option<DateTime<Utc>>,
    /// End time filter (inclusive)
    pub end_time: Option<DateTime<Utc>>,
    /// Event type filter
    pub event_type: Option<MetaLearningEventType>,
    /// Domain filter
    pub domain: Option<Domain>,
    /// Maximum events to return
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

impl EventLogQuery {
    /// Create new query builder
    pub fn new() -> Self;

    /// Set time range
    pub fn time_range(self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self;

    /// Set event type filter
    pub fn event_type(self, event_type: MetaLearningEventType) -> Self;

    /// Set domain filter
    pub fn domain(self, domain: Domain) -> Self;

    /// Set limit
    pub fn limit(self, limit: usize) -> Self;

    /// Set offset
    pub fn offset(self, offset: usize) -> Self;

    /// Check if event matches query
    pub fn matches(&self, event: &MetaLearningEvent) -> bool;
}

/// In-memory meta-learning event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningEventLog {
    /// Event buffer (FIFO)
    events: VecDeque<MetaLearningEvent>,
    /// Maximum events to keep in memory
    max_events: usize,
    /// Retention period
    retention_days: u32,
    /// Total events logged (including evicted)
    total_logged: u64,
    /// Total events evicted
    total_evicted: u64,
}

impl MetaLearningEventLog {
    /// Create new event log with default settings
    pub fn new() -> Self;

    /// Create with custom settings
    pub fn with_config(max_events: usize, retention_days: u32) -> Self;

    /// Execute a query
    pub fn query(&self, query: &EventLogQuery) -> Vec<&MetaLearningEvent>;

    /// Get events matching a predicate
    pub fn filter<F>(&self, predicate: F) -> Vec<&MetaLearningEvent>
    where
        F: Fn(&MetaLearningEvent) -> bool;

    /// Get statistics
    pub fn stats(&self) -> EventLogStats;

    /// Get events by accuracy range
    pub fn by_accuracy_range(&self, min: f32, max: f32) -> Vec<&MetaLearningEvent>;

    /// Get escalation events
    pub fn escalation_events(&self) -> Vec<&MetaLearningEvent>;

    /// Check if any events in last N minutes
    pub fn has_recent_events(&self, minutes: i64) -> bool;

    /// Export to JSON
    pub fn to_json(&self) -> UtlResult<String>;

    /// Import from JSON
    pub fn from_json(json: &str) -> UtlResult<Self>;

    /// Evict old events beyond retention period
    fn evict_old_events(&mut self);

    /// Evict events if over max limit (FIFO)
    fn evict_over_limit(&mut self);
}

impl MetaLearningLogger for MetaLearningEventLog {
    fn log_event(&mut self, event: MetaLearningEvent) -> UtlResult<()>;
    fn query_by_time(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&MetaLearningEvent>;
    fn query_by_type(&self, event_type: MetaLearningEventType) -> Vec<&MetaLearningEvent>;
    fn query_by_domain(&self, domain: Domain) -> Vec<&MetaLearningEvent>;
    fn recent_events(&self, count: usize) -> Vec<&MetaLearningEvent>;
    fn event_count(&self) -> usize;
    fn clear(&mut self);
}

impl Default for MetaLearningEventLog {
    fn default() -> Self;
}

/// Event log statistics
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EventLogStats {
    /// Current event count
    pub current_count: usize,
    /// Total events ever logged
    pub total_logged: u64,
    /// Total events evicted
    pub total_evicted: u64,
    /// Events by type
    pub by_type: EventTypeCount,
    /// Average accuracy in events
    pub avg_accuracy: f32,
    /// Oldest event timestamp
    pub oldest_event: Option<DateTime<Utc>>,
    /// Newest event timestamp
    pub newest_event: Option<DateTime<Utc>>,
}

/// Count of events by type
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EventTypeCount {
    pub lambda_adjustment: usize,
    pub bayesian_escalation: usize,
    pub accuracy_alert: usize,
    pub self_healing: usize,
    pub human_escalation: usize,
}
```

### 6.2 Constraints

- Events MUST be stored in chronological order
- FIFO eviction MUST remove oldest events first
- Time queries MUST be inclusive on both ends
- Pagination (offset/limit) MUST work correctly
- JSON export/import MUST preserve all fields
- Empty log queries MUST return empty vec, not error
- Stats computation MUST handle empty log

### 6.3 Verification Commands

```bash
# Type check
cargo check -p context-graph-mcp

# Unit tests
cargo test -p context-graph-mcp handlers::core::event_log

# JSON serialization tests
cargo test -p context-graph-mcp test_event_log_json

# Clippy
cargo clippy -p context-graph-mcp -- -D warnings

# FSV: Verify event log state after operations
cargo test -p context-graph-mcp test_event_log_fsv -- --nocapture
```

---

## 7. Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/core/event_log.rs` | Event log implementation |

---

## 8. Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-mcp/src/handlers/core/mod.rs` | Add `pub mod event_log;` |
| `crates/context-graph-mcp/src/handlers/core/types.rs` | Add `created_at: DateTime<Utc>` to MetaLearningEvent |
| `crates/context-graph-mcp/src/lib.rs` | Re-export event log types if needed |

---

## 9. Pseudo-Code

### 9.1 log_event Implementation

```
FUNCTION log_event(event: MetaLearningEvent) -> UtlResult<()>:
    // Evict old events first (by retention)
    self.evict_old_events()

    // Add new event
    self.events.push_back(event)
    self.total_logged += 1

    // Evict if over limit (FIFO)
    self.evict_over_limit()

    RETURN Ok(())
```

### 9.2 query Implementation

```
FUNCTION query(query: &EventLogQuery) -> Vec<&MetaLearningEvent>:
    LET results = Vec::new()
    LET count = 0
    LET skipped = 0

    FOR event IN self.events.iter():
        // Apply filters
        IF NOT query.matches(event):
            CONTINUE

        // Apply offset
        IF let Some(offset) = query.offset:
            IF skipped < offset:
                skipped += 1
                CONTINUE

        // Apply limit
        IF let Some(limit) = query.limit:
            IF count >= limit:
                BREAK

        results.push(event)
        count += 1

    RETURN results
```

### 9.3 evict_old_events Implementation

```
FUNCTION evict_old_events():
    LET cutoff = Utc::now() - Duration::days(self.retention_days as i64)

    WHILE let Some(oldest) = self.events.front():
        IF oldest.timestamp < cutoff:
            self.events.pop_front()
            self.total_evicted += 1
        ELSE:
            BREAK  // Events are chronological, so stop
```

---

## 10. Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| Events stored chronologically | Insert out-of-order, verify sorted |
| FIFO eviction works | Add max+1 events, verify oldest removed |
| Time range query correct | Query subset, verify bounds |
| Type filter works | Insert multiple types, filter |
| Pagination works | Query with offset/limit |
| JSON roundtrip preserves data | Serialize/deserialize, compare |
| Stats correct | Verify counts match insertions |

---

## 11. Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_event(event_type: MetaLearningEventType) -> MetaLearningEvent {
        MetaLearningEvent::lambda_adjustment(
            0.25,
            (0.5, 0.5),
            (0.45, 0.55),
            0.75,
            Some(Domain::Code),
        )
    }

    #[test]
    fn test_log_event() {
        let mut log = MetaLearningEventLog::new();
        let event = create_test_event(MetaLearningEventType::LambdaAdjustment);

        log.log_event(event).unwrap();
        assert_eq!(log.event_count(), 1);
    }

    #[test]
    fn test_fifo_eviction() {
        let mut log = MetaLearningEventLog::with_config(5, 7);

        for i in 0..10 {
            let mut event = create_test_event(MetaLearningEventType::LambdaAdjustment);
            event.prediction_error = i as f32 * 0.1;
            log.log_event(event).unwrap();
        }

        assert_eq!(log.event_count(), 5);

        // First event should be i=5 (0-4 evicted)
        let recent = log.recent_events(5);
        assert!((recent[0].prediction_error - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_query_by_time() {
        let mut log = MetaLearningEventLog::new();

        // Create events at different times
        let now = Utc::now();
        let mut event1 = create_test_event(MetaLearningEventType::LambdaAdjustment);
        event1.timestamp = now - Duration::hours(2);
        log.log_event(event1).unwrap();

        let mut event2 = create_test_event(MetaLearningEventType::LambdaAdjustment);
        event2.timestamp = now - Duration::hours(1);
        log.log_event(event2).unwrap();

        let mut event3 = create_test_event(MetaLearningEventType::LambdaAdjustment);
        event3.timestamp = now;
        log.log_event(event3).unwrap();

        // Query middle hour
        let start = now - Duration::minutes(90);
        let end = now - Duration::minutes(30);
        let results = log.query_by_time(start, end);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_by_type() {
        let mut log = MetaLearningEventLog::new();

        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment)).unwrap();
        log.log_event(MetaLearningEvent::escalation(0.6, None)).unwrap();
        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment)).unwrap();

        let adjustments = log.query_by_type(MetaLearningEventType::LambdaAdjustment);
        assert_eq!(adjustments.len(), 2);

        let escalations = log.query_by_type(MetaLearningEventType::BayesianEscalation);
        assert_eq!(escalations.len(), 1);
    }

    #[test]
    fn test_query_by_domain() {
        let mut log = MetaLearningEventLog::new();

        let mut event1 = create_test_event(MetaLearningEventType::LambdaAdjustment);
        event1.domain = Some(Domain::Code);
        log.log_event(event1).unwrap();

        let mut event2 = create_test_event(MetaLearningEventType::LambdaAdjustment);
        event2.domain = Some(Domain::Medical);
        log.log_event(event2).unwrap();

        let code_events = log.query_by_domain(Domain::Code);
        assert_eq!(code_events.len(), 1);
    }

    #[test]
    fn test_pagination() {
        let mut log = MetaLearningEventLog::new();

        for _ in 0..20 {
            log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment)).unwrap();
        }

        let query = EventLogQuery::new().limit(5).offset(10);
        let results = log.query(&query);

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut log = MetaLearningEventLog::new();
        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment)).unwrap();
        log.log_event(MetaLearningEvent::escalation(0.5, Some(Domain::Medical))).unwrap();

        let json = log.to_json().unwrap();
        let restored = MetaLearningEventLog::from_json(&json).unwrap();

        assert_eq!(restored.event_count(), 2);
    }

    #[test]
    fn test_stats() {
        let mut log = MetaLearningEventLog::new();

        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment)).unwrap();
        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment)).unwrap();
        log.log_event(MetaLearningEvent::escalation(0.5, None)).unwrap();

        let stats = log.stats();
        assert_eq!(stats.current_count, 3);
        assert_eq!(stats.by_type.lambda_adjustment, 2);
        assert_eq!(stats.by_type.bayesian_escalation, 1);
    }

    #[test]
    fn test_clear() {
        let mut log = MetaLearningEventLog::new();

        for _ in 0..10 {
            log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment)).unwrap();
        }

        log.clear();
        assert_eq!(log.event_count(), 0);
    }

    #[test]
    fn test_empty_log_queries() {
        let log = MetaLearningEventLog::new();

        assert!(log.recent_events(10).is_empty());
        assert!(log.query_by_type(MetaLearningEventType::LambdaAdjustment).is_empty());
        assert_eq!(log.stats().current_count, 0);
    }
}
```

---

## 12. Rollback Plan

If this task fails validation:

1. Revert `crates/context-graph-mcp/src/handlers/core/event_log.rs`
2. Remove mod declaration from `handlers/core/mod.rs`
3. Previous tasks remain unaffected
4. Document failure in task notes

---

## 13. Source of Truth

| State | Location | Type |
|-------|----------|------|
| Event buffer | `MetaLearningEventLog.events` | `VecDeque<MetaLearningEvent>` |
| Event count | `MetaLearningEventLog.events.len()` | Runtime count |
| Total logged | `MetaLearningEventLog.total_logged` | `u64` counter |
| Total evicted | `MetaLearningEventLog.total_evicted` | `u64` counter |
| Configuration | `MetaLearningEventLog.max_events`, `retention_days` | Struct fields |

**FSV Verification**: After any mutation (log_event, clear, evict), verify state by:
1. Calling `event_count()` and comparing to `events.len()`
2. Calling `stats()` and verifying counts match insertions
3. For JSON roundtrip: deserialize and compare all fields

---

## 14. FSV Requirements

### 14.1 Full State Verification Pattern

```rust
/// FSV: Execute & Inspect pattern for event log operations
#[cfg(test)]
fn fsv_verify_log_event(log: &MetaLearningEventLog, expected_count: usize) {
    // 1. INSPECT: Read the actual state (not return value)
    let actual_count = log.event_count();
    let stats = log.stats();

    // 2. VERIFY: Compare against expected
    assert_eq!(actual_count, expected_count,
        "FSV: event_count mismatch. Expected {}, got {}", expected_count, actual_count);
    assert_eq!(stats.current_count, expected_count,
        "FSV: stats.current_count mismatch");

    // 3. INVARIANTS: Check log invariants
    assert!(stats.total_logged >= stats.current_count as u64,
        "FSV: total_logged must be >= current_count");
    assert_eq!(
        stats.total_logged - stats.current_count as u64,
        stats.total_evicted,
        "FSV: total_evicted calculation wrong"
    );
}
```

### 14.2 Edge Case Audit (3 Cases)

#### Edge Case 1: FIFO Eviction at Max Capacity

```rust
#[test]
fn fsv_edge_case_fifo_eviction() {
    // BEFORE STATE
    let mut log = MetaLearningEventLog::with_config(3, 7); // max 3 events
    println!("BEFORE: count={}, total_logged={}", log.event_count(), log.stats().total_logged);

    // ACTION: Add 5 events (should evict 2)
    for i in 0..5 {
        let mut e = create_test_event();
        e.description = Some(format!("event_{}", i));
        log.log_event(e).unwrap();
    }

    // AFTER STATE (FSV)
    let stats = log.stats();
    println!("AFTER: count={}, total_logged={}, total_evicted={}",
        stats.current_count, stats.total_logged, stats.total_evicted);

    // VERIFY
    assert_eq!(stats.current_count, 3, "Should have exactly 3 events");
    assert_eq!(stats.total_logged, 5, "Should have logged 5 total");
    assert_eq!(stats.total_evicted, 2, "Should have evicted 2 events");

    // Verify oldest remaining is event_2 (events 0,1 evicted)
    let oldest = log.recent_events(3).first().unwrap();
    assert!(oldest.description.as_ref().unwrap().contains("event_2"));
}
```

#### Edge Case 2: Empty Log Query

```rust
#[test]
fn fsv_edge_case_empty_log() {
    let log = MetaLearningEventLog::new();

    // BEFORE STATE
    println!("BEFORE: count={}", log.event_count());

    // ACTION: Query empty log (should not panic, return empty vec)
    let by_time = log.query_by_time(Utc::now() - Duration::hours(1), Utc::now());
    let by_type = log.query_by_type(MetaLearningEventType::LambdaAdjustment);
    let recent = log.recent_events(10);
    let stats = log.stats();

    // AFTER STATE (FSV)
    println!("AFTER: query_by_time={}, query_by_type={}, recent={}",
        by_time.len(), by_type.len(), recent.len());

    // VERIFY: All should be empty, not error
    assert!(by_time.is_empty(), "FSV: Empty log should return empty vec for time query");
    assert!(by_type.is_empty(), "FSV: Empty log should return empty vec for type query");
    assert!(recent.is_empty(), "FSV: Empty log should return empty vec for recent");
    assert_eq!(stats.current_count, 0, "FSV: Stats should show 0 events");
    assert!(stats.oldest_event.is_none(), "FSV: oldest_event should be None");
}
```

#### Edge Case 3: JSON Roundtrip Preserves All Fields

```rust
#[test]
fn fsv_edge_case_json_roundtrip() {
    // BEFORE STATE
    let mut log = MetaLearningEventLog::new();
    log.log_event(MetaLearningEvent::lambda_adjustment(5, 0.3, 0.35)).unwrap();
    log.log_event(MetaLearningEvent::bayesian_escalation(7)).unwrap();
    let before_stats = log.stats();
    let before_json = log.to_json().unwrap();
    println!("BEFORE: count={}, json_len={}", before_stats.current_count, before_json.len());

    // ACTION: Serialize and deserialize
    let restored = MetaLearningEventLog::from_json(&before_json).unwrap();

    // AFTER STATE (FSV)
    let after_stats = restored.stats();
    println!("AFTER: count={}, total_logged={}", after_stats.current_count, after_stats.total_logged);

    // VERIFY: All fields preserved
    assert_eq!(before_stats.current_count, after_stats.current_count, "FSV: count mismatch");
    assert_eq!(before_stats.total_logged, after_stats.total_logged, "FSV: total_logged mismatch");
    assert_eq!(before_stats.by_type.lambda_adjustment, after_stats.by_type.lambda_adjustment);
    assert_eq!(before_stats.by_type.bayesian_escalation, after_stats.by_type.bayesian_escalation);

    // Verify events have correct data
    let events = restored.recent_events(2);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event_type, MetaLearningEventType::LambdaAdjustment);
    assert_eq!(events[1].event_type, MetaLearningEventType::BayesianEscalation);
}
```

### 14.3 Evidence of Success

When tests pass, output should show:

```
BEFORE: count=0, total_logged=0
AFTER: count=3, total_logged=5, total_evicted=2
✓ FSV: FIFO eviction verified

BEFORE: count=0
AFTER: query_by_time=0, query_by_type=0, recent=0
✓ FSV: Empty log queries verified

BEFORE: count=2, json_len=1847
AFTER: count=2, total_logged=2
✓ FSV: JSON roundtrip verified
```

---

## 15. Fail-Fast Error Handling

```rust
impl MetaLearningEventLog {
    pub fn log_event(&mut self, event: MetaLearningEvent) -> Result<(), EventLogError> {
        // FAIL-FAST: Validate event before accepting
        if event.timestamp > Utc::now() + Duration::seconds(60) {
            return Err(EventLogError::FutureTimestamp {
                timestamp: event.timestamp,
                now: Utc::now(),
            });
        }

        // Proceed with logging...
        self.evict_old_events();
        self.events.push_back(event);
        self.total_logged += 1;
        self.evict_over_limit();

        Ok(())
    }

    pub fn from_json(json: &str) -> Result<Self, EventLogError> {
        // FAIL-FAST: Reject invalid JSON immediately
        serde_json::from_str(json).map_err(|e| EventLogError::InvalidJson {
            message: e.to_string(),
            json_preview: json.chars().take(100).collect(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EventLogError {
    #[error("Event timestamp {timestamp:?} is in the future (now: {now:?})")]
    FutureTimestamp {
        timestamp: DateTime<Utc>,
        now: DateTime<Utc>,
    },

    #[error("Invalid JSON: {message}. Preview: {json_preview}")]
    InvalidJson {
        message: String,
        json_preview: String,
    },
}
```

---

## 16. Notes

- In-memory storage is Phase 1; database persistence is Phase 2
- VecDeque enables efficient FIFO operations
- Stats computation is O(n) but called infrequently
- JSON export enables backup and debugging
- Event timestamps use UTC for consistency
- **Architecture Decision**: Event log lives in MCP crate alongside MetaLearningEvent types
  for direct integration with MCP tool handlers

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
| 2.0.0 | 2026-01-12 | AI Agent | Updated paths to MCP crate, added FSV sections, Source of Truth, Edge Cases, Fail-Fast error handling |
