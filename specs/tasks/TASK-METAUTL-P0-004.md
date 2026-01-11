# Task Specification: Meta-Learning Event Log Implementation

**Task ID:** TASK-METAUTL-P0-004
**Version:** 1.0.0
**Status:** Ready
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
| TASK-METAUTL-P0-001 | Core types (MetaLearningEvent) | Must complete first |
| TASK-METAUTL-P0-002 | Lambda adjustment (event source) | Must complete first |
| TASK-METAUTL-P0-003 | Escalation (event source) | Must complete first |

### 1.3 Blocked By

- TASK-METAUTL-P0-003 (all event sources must exist)

---

## 2. Context

This task implements the meta-learning event log for introspection and debugging. The log captures all self-correction events including:

- Lambda adjustments
- Escalation triggers
- Accuracy alerts
- Human escalation requests

The log supports time-range queries and provides the foundation for the `get_meta_learning_log` MCP tool.

---

## 3. Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-utl/src/meta/types.rs` | MetaLearningEvent, MetaLearningEventType |
| `crates/context-graph-utl/src/meta/correction.rs` | Event generation patterns |
| `crates/context-graph-utl/src/meta/escalation.rs` | Escalation events |
| `specs/functional/SPEC-METAUTL-001.md` | Logging requirements |

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

| Check | Description |
|-------|-------------|
| [ ] | TASK-METAUTL-P0-001 completed |
| [ ] | TASK-METAUTL-P0-002 completed |
| [ ] | TASK-METAUTL-P0-003 completed |
| [ ] | MetaLearningEvent type exists |

---

## 6. Definition of Done

### 6.1 Required Signatures

#### File: `crates/context-graph-utl/src/meta/event_log.rs`

```rust
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::error::{UtlError, UtlResult};
use super::types::{MetaLearningEvent, MetaLearningEventType, Domain};

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
cargo check -p context-graph-utl

# Unit tests
cargo test -p context-graph-utl meta::event_log

# JSON serialization tests
cargo test -p context-graph-utl test_event_log_json

# Clippy
cargo clippy -p context-graph-utl -- -D warnings
```

---

## 7. Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-utl/src/meta/event_log.rs` | Event log implementation |
| `crates/context-graph-utl/src/meta/tests_event_log.rs` | Unit tests |

---

## 8. Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-utl/src/meta/mod.rs` | Add `pub mod event_log;` |
| `crates/context-graph-utl/src/lib.rs` | Re-export event log types |

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

1. Revert `crates/context-graph-utl/src/meta/event_log.rs`
2. Remove mod declaration
3. Previous tasks remain unaffected
4. Document failure in task notes

---

## 13. Notes

- In-memory storage is Phase 1; database persistence is Phase 2
- VecDeque enables efficient FIFO operations
- Stats computation is O(n) but called infrequently
- JSON export enables backup and debugging
- Event timestamps use UTC for consistency

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
