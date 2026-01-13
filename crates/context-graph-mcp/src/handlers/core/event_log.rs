//! Meta-Learning Event Log.
//!
//! TASK-METAUTL-P0-004: Implements event logging for all meta-learning events
//! including lambda adjustments, escalations, alerts, recoveries, and weight clamping.
//!
//! # Features
//!
//! - In-memory storage with FIFO eviction when capacity is exceeded
//! - Time-based retention policy (evict events older than retention_days)
//! - Time-range queries with inclusive bounds
//! - Filtering by event type and domain
//! - Pagination support (offset/limit)
//! - JSON serialization for persistence
//! - Statistics computation (event counts, accuracy averages)
//!
//! # Constitution Reference
//!
//! - REQ-METAUTL-011: Event log SHALL capture all meta-learning state changes
//! - REQ-METAUTL-012: Event log SHALL support time-range queries
//! - REQ-METAUTL-013: Event log SHALL implement FIFO eviction policy

// Allow dead_code until integration in TASK-METAUTL-P0-005/006
#![allow(dead_code)]

use std::collections::VecDeque;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use context_graph_utl::error::{UtlError, UtlResult};

use super::types::{Domain, MetaLearningEvent, MetaLearningEventType};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum number of events to store before FIFO eviction.
/// TASK-METAUTL-P0-004: Constitution-mandated default capacity.
pub const DEFAULT_MAX_EVENTS: usize = 1000;

/// Default retention period in days.
/// TASK-METAUTL-P0-004: Events older than this are eligible for eviction.
pub const DEFAULT_RETENTION_DAYS: u32 = 7;

// ============================================================================
// EventLogQuery
// ============================================================================

/// Query parameters for filtering events in the log.
///
/// Supports time-range queries, event type filtering, domain filtering,
/// and pagination via offset/limit.
///
/// # Example
///
/// ```ignore
/// let query = EventLogQuery::new()
///     .time_range(start, end)
///     .event_type(MetaLearningEventType::LambdaAdjustment)
///     .domain(Domain::Code)
///     .limit(50)
///     .offset(10);
/// ```
#[derive(Debug, Clone, Default)]
pub struct EventLogQuery {
    /// Start time for time-range query (inclusive)
    pub start_time: Option<DateTime<Utc>>,
    /// End time for time-range query (inclusive)
    pub end_time: Option<DateTime<Utc>>,
    /// Filter by event type
    pub event_type: Option<MetaLearningEventType>,
    /// Filter by domain
    pub domain: Option<Domain>,
    /// Maximum number of results to return
    pub limit: Option<usize>,
    /// Number of results to skip
    pub offset: Option<usize>,
}

impl EventLogQuery {
    /// Create a new empty query.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the time range for the query (inclusive bounds).
    pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Set the start time only.
    pub fn start_time(mut self, start: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self
    }

    /// Set the end time only.
    pub fn end_time(mut self, end: DateTime<Utc>) -> Self {
        self.end_time = Some(end);
        self
    }

    /// Filter by event type.
    pub fn event_type(mut self, event_type: MetaLearningEventType) -> Self {
        self.event_type = Some(event_type);
        self
    }

    /// Filter by domain.
    pub fn domain(mut self, domain: Domain) -> Self {
        self.domain = Some(domain);
        self
    }

    /// Set the maximum number of results.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the number of results to skip.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Check if an event matches this query's filters.
    ///
    /// Does NOT apply pagination (offset/limit) - that's handled by the log.
    pub fn matches(&self, event: &MetaLearningEvent) -> bool {
        // Check time range
        if let Some(start) = self.start_time {
            if event.created_at < start {
                return false;
            }
        }
        if let Some(end) = self.end_time {
            if event.created_at > end {
                return false;
            }
        }

        // Check event type
        if let Some(expected_type) = self.event_type {
            if event.event_type != expected_type {
                return false;
            }
        }

        // Check domain
        if let Some(expected_domain) = self.domain {
            if event.domain != expected_domain {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// EventTypeCount
// ============================================================================

/// Count of events by type.
///
/// Used in EventLogStats for reporting event distribution.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventTypeCount {
    /// Number of lambda adjustment events
    pub lambda_adjustment: usize,
    /// Number of bayesian escalation events
    pub bayesian_escalation: usize,
    /// Number of accuracy alert events
    pub accuracy_alert: usize,
    /// Number of accuracy recovery events
    pub accuracy_recovery: usize,
    /// Number of weight clamped events
    pub weight_clamped: usize,
}

impl EventTypeCount {
    /// Increment the count for a specific event type.
    pub fn increment(&mut self, event_type: MetaLearningEventType) {
        match event_type {
            MetaLearningEventType::LambdaAdjustment => self.lambda_adjustment += 1,
            MetaLearningEventType::BayesianEscalation => self.bayesian_escalation += 1,
            MetaLearningEventType::AccuracyAlert => self.accuracy_alert += 1,
            MetaLearningEventType::AccuracyRecovery => self.accuracy_recovery += 1,
            MetaLearningEventType::WeightClamped => self.weight_clamped += 1,
        }
    }

    /// Get the total count of all events.
    pub fn total(&self) -> usize {
        self.lambda_adjustment
            + self.bayesian_escalation
            + self.accuracy_alert
            + self.accuracy_recovery
            + self.weight_clamped
    }
}

// ============================================================================
// EventLogStats
// ============================================================================

/// Statistics about the event log.
///
/// Provides summary information about events, including counts by type,
/// average accuracy, and time bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLogStats {
    /// Current number of events in the log
    pub current_count: usize,
    /// Total events logged since creation (includes evicted)
    pub total_logged: u64,
    /// Total events evicted (due to capacity or retention)
    pub total_evicted: u64,
    /// Counts broken down by event type
    pub by_type: EventTypeCount,
    /// Average accuracy across events that have accuracy data
    pub avg_accuracy: f32,
    /// Timestamp of the oldest event in the log
    pub oldest_event: Option<DateTime<Utc>>,
    /// Timestamp of the newest event in the log
    pub newest_event: Option<DateTime<Utc>>,
}

impl Default for EventLogStats {
    fn default() -> Self {
        Self {
            current_count: 0,
            total_logged: 0,
            total_evicted: 0,
            by_type: EventTypeCount::default(),
            avg_accuracy: 0.0,
            oldest_event: None,
            newest_event: None,
        }
    }
}

// ============================================================================
// MetaLearningLogger Trait
// ============================================================================

/// Trait for logging meta-learning events.
///
/// Provides the interface for event storage, querying, and retrieval.
/// Implementors must provide FIFO eviction and time-based retention.
pub trait MetaLearningLogger {
    /// Log a new event.
    ///
    /// May trigger FIFO eviction if capacity is exceeded.
    fn log_event(&mut self, event: MetaLearningEvent) -> UtlResult<()>;

    /// Query events by time range (inclusive).
    fn query_by_time(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&MetaLearningEvent>;

    /// Query events by type.
    fn query_by_type(&self, event_type: MetaLearningEventType) -> Vec<&MetaLearningEvent>;

    /// Query events by domain.
    fn query_by_domain(&self, domain: Domain) -> Vec<&MetaLearningEvent>;

    /// Get the most recent events.
    fn recent_events(&self, count: usize) -> Vec<&MetaLearningEvent>;

    /// Get the total number of events currently in the log.
    fn event_count(&self) -> usize;

    /// Clear all events from the log.
    fn clear(&mut self);
}

// ============================================================================
// MetaLearningEventLog
// ============================================================================

/// In-memory event log for meta-learning events.
///
/// Stores events in chronological order with FIFO eviction when capacity
/// is exceeded. Also supports time-based retention eviction.
///
/// # Thread Safety
///
/// This struct is NOT thread-safe. Wrap in `Arc<Mutex<_>>` or `parking_lot::Mutex`
/// for concurrent access.
///
/// # Example
///
/// ```ignore
/// let mut log = MetaLearningEventLog::new();
/// log.log_event(MetaLearningEvent::lambda_adjustment(0, 0.5, 0.6))?;
///
/// let recent = log.recent_events(10);
/// let stats = log.stats();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningEventLog {
    /// Events stored in chronological order (oldest first).
    events: VecDeque<MetaLearningEvent>,
    /// Maximum number of events to store before FIFO eviction.
    max_events: usize,
    /// Retention period in days. Events older than this are evicted.
    retention_days: u32,
    /// Total events logged since creation.
    total_logged: u64,
    /// Total events evicted (capacity or retention).
    total_evicted: u64,
}

impl Default for MetaLearningEventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaLearningEventLog {
    /// Create a new event log with default configuration.
    pub fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(DEFAULT_MAX_EVENTS),
            max_events: DEFAULT_MAX_EVENTS,
            retention_days: DEFAULT_RETENTION_DAYS,
            total_logged: 0,
            total_evicted: 0,
        }
    }

    /// Create a new event log with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `max_events` - Maximum events before FIFO eviction
    /// * `retention_days` - Days to retain events
    pub fn with_config(max_events: usize, retention_days: u32) -> Self {
        Self {
            events: VecDeque::with_capacity(max_events.min(10000)), // Cap initial allocation
            max_events,
            retention_days,
            total_logged: 0,
            total_evicted: 0,
        }
    }

    /// Query events using a flexible query object.
    ///
    /// Applies all filters in the query and handles pagination.
    pub fn query(&self, query: &EventLogQuery) -> Vec<&MetaLearningEvent> {
        let mut results: Vec<&MetaLearningEvent> = self
            .events
            .iter()
            .filter(|e| query.matches(e))
            .collect();

        // Apply pagination
        let offset = query.offset.unwrap_or(0);
        if offset >= results.len() {
            return Vec::new();
        }

        results = results.into_iter().skip(offset).collect();

        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        results
    }

    /// Filter events using a custom predicate.
    pub fn filter<F>(&self, predicate: F) -> Vec<&MetaLearningEvent>
    where
        F: Fn(&MetaLearningEvent) -> bool,
    {
        self.events.iter().filter(|e| predicate(e)).collect()
    }

    /// Compute statistics about the event log.
    pub fn stats(&self) -> EventLogStats {
        let mut by_type = EventTypeCount::default();
        let mut accuracy_sum: f64 = 0.0;
        let mut accuracy_count: usize = 0;

        for event in &self.events {
            by_type.increment(event.event_type);
            if let Some(acc) = event.accuracy {
                accuracy_sum += acc as f64;
                accuracy_count += 1;
            }
        }

        let avg_accuracy = if accuracy_count > 0 {
            (accuracy_sum / accuracy_count as f64) as f32
        } else {
            0.0
        };

        EventLogStats {
            current_count: self.events.len(),
            total_logged: self.total_logged,
            total_evicted: self.total_evicted,
            by_type,
            avg_accuracy,
            oldest_event: self.events.front().map(|e| e.created_at),
            newest_event: self.events.back().map(|e| e.created_at),
        }
    }

    /// Query events within an accuracy range.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum accuracy (inclusive)
    /// * `max` - Maximum accuracy (inclusive)
    pub fn by_accuracy_range(&self, min: f32, max: f32) -> Vec<&MetaLearningEvent> {
        self.events
            .iter()
            .filter(|e| {
                if let Some(acc) = e.accuracy {
                    acc >= min && acc <= max
                } else {
                    false
                }
            })
            .collect()
    }

    /// Get all escalation events.
    pub fn escalation_events(&self) -> Vec<&MetaLearningEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == MetaLearningEventType::BayesianEscalation)
            .collect()
    }

    /// Check if there are recent events within the specified time window.
    ///
    /// # Arguments
    ///
    /// * `minutes` - Number of minutes to look back
    pub fn has_recent_events(&self, minutes: i64) -> bool {
        let cutoff = Utc::now() - Duration::minutes(minutes);
        self.events.iter().any(|e| e.created_at >= cutoff)
    }

    /// Serialize the event log to JSON.
    pub fn to_json(&self) -> UtlResult<String> {
        serde_json::to_string(self).map_err(|e| UtlError::SerializationError(e.to_string()))
    }

    /// Deserialize an event log from JSON.
    pub fn from_json(json: &str) -> UtlResult<Self> {
        serde_json::from_str(json).map_err(|e| UtlError::SerializationError(e.to_string()))
    }

    /// Evict events older than retention_days.
    ///
    /// Called automatically by log_event, but can be called manually.
    pub fn evict_old_events(&mut self) {
        let cutoff = Utc::now() - Duration::days(self.retention_days as i64);
        let mut evicted = 0;

        // Events are in chronological order, so we can stop once we find one within retention
        while let Some(front) = self.events.front() {
            if front.created_at < cutoff {
                self.events.pop_front();
                evicted += 1;
            } else {
                break;
            }
        }

        self.total_evicted += evicted;
    }

    /// Evict oldest events when over capacity (FIFO).
    ///
    /// Called automatically by log_event, but can be called manually.
    pub fn evict_over_limit(&mut self) {
        while self.events.len() >= self.max_events {
            self.events.pop_front();
            self.total_evicted += 1;
        }
    }

    /// Get the maximum capacity.
    pub fn max_events(&self) -> usize {
        self.max_events
    }

    /// Get the retention period in days.
    pub fn retention_days(&self) -> u32 {
        self.retention_days
    }

    /// Get total events logged since creation.
    pub fn total_logged(&self) -> u64 {
        self.total_logged
    }

    /// Get total events evicted.
    pub fn total_evicted(&self) -> u64 {
        self.total_evicted
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl MetaLearningLogger for MetaLearningEventLog {
    fn log_event(&mut self, event: MetaLearningEvent) -> UtlResult<()> {
        // First, evict old events based on retention policy
        self.evict_old_events();

        // Then, ensure we have capacity (FIFO eviction)
        self.evict_over_limit();

        // Add the new event
        self.events.push_back(event);
        self.total_logged += 1;

        Ok(())
    }

    fn query_by_time(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&MetaLearningEvent> {
        self.events
            .iter()
            .filter(|e| e.created_at >= start && e.created_at <= end)
            .collect()
    }

    fn query_by_type(&self, event_type: MetaLearningEventType) -> Vec<&MetaLearningEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    fn query_by_domain(&self, domain: Domain) -> Vec<&MetaLearningEvent> {
        self.events.iter().filter(|e| e.domain == domain).collect()
    }

    fn recent_events(&self, count: usize) -> Vec<&MetaLearningEvent> {
        // Return most recent events (from the back of the deque)
        let len = self.events.len();
        if count >= len {
            self.events.iter().collect()
        } else {
            self.events.iter().skip(len - count).collect()
        }
    }

    fn event_count(&self) -> usize {
        self.events.len()
    }

    fn clear(&mut self) {
        let cleared = self.events.len();
        self.events.clear();
        self.total_evicted += cleared as u64;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;

    fn create_test_event(event_type: MetaLearningEventType) -> MetaLearningEvent {
        match event_type {
            MetaLearningEventType::LambdaAdjustment => {
                MetaLearningEvent::lambda_adjustment(0, 0.5, 0.6)
            }
            MetaLearningEventType::BayesianEscalation => {
                MetaLearningEvent::bayesian_escalation(10)
            }
            MetaLearningEventType::AccuracyAlert => {
                MetaLearningEvent::accuracy_alert(0.5, 0.7)
            }
            MetaLearningEventType::AccuracyRecovery => {
                MetaLearningEvent::accuracy_recovery(0.5, 0.8)
            }
            MetaLearningEventType::WeightClamped => {
                MetaLearningEvent::weight_clamped(0, 0.95, 0.9)
            }
        }
    }

    #[test]
    fn test_log_event() {
        let mut log = MetaLearningEventLog::new();
        assert_eq!(log.event_count(), 0);

        let event = create_test_event(MetaLearningEventType::LambdaAdjustment);
        log.log_event(event).unwrap();

        assert_eq!(log.event_count(), 1);
        assert_eq!(log.total_logged(), 1);
        assert_eq!(log.total_evicted(), 0);
    }

    #[test]
    fn test_fifo_eviction() {
        let mut log = MetaLearningEventLog::with_config(5, 7);

        // Log 7 events (exceeds capacity of 5)
        for i in 0..7 {
            let mut event = create_test_event(MetaLearningEventType::LambdaAdjustment);
            event.embedder_index = Some(i);
            log.log_event(event).unwrap();
        }

        // Should have 5 events (oldest 2 evicted)
        assert_eq!(log.event_count(), 5);
        assert_eq!(log.total_logged(), 7);
        assert_eq!(log.total_evicted(), 2);

        // Verify oldest events were evicted (FIFO)
        let events: Vec<_> = log.recent_events(10);
        assert_eq!(events[0].embedder_index, Some(2)); // First remaining
        assert_eq!(events[4].embedder_index, Some(6)); // Last added
    }

    #[test]
    fn test_query_by_time() {
        let mut log = MetaLearningEventLog::new();

        let now = Utc::now();
        let event1 = create_test_event(MetaLearningEventType::LambdaAdjustment);
        log.log_event(event1).unwrap();

        // Small delay to ensure different timestamps
        thread::sleep(StdDuration::from_millis(10));

        let event2 = create_test_event(MetaLearningEventType::AccuracyAlert);
        log.log_event(event2).unwrap();

        // Query all events
        let results = log.query_by_time(now - Duration::seconds(1), now + Duration::seconds(1));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_type() {
        let mut log = MetaLearningEventLog::new();

        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::AccuracyAlert))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();

        let results = log.query_by_type(MetaLearningEventType::LambdaAdjustment);
        assert_eq!(results.len(), 2);

        let results = log.query_by_type(MetaLearningEventType::AccuracyAlert);
        assert_eq!(results.len(), 1);

        let results = log.query_by_type(MetaLearningEventType::BayesianEscalation);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_query_by_domain() {
        let mut log = MetaLearningEventLog::new();

        let event1 = create_test_event(MetaLearningEventType::LambdaAdjustment)
            .with_domain(Domain::Code);
        let event2 = create_test_event(MetaLearningEventType::LambdaAdjustment)
            .with_domain(Domain::Medical);
        let event3 = create_test_event(MetaLearningEventType::LambdaAdjustment)
            .with_domain(Domain::Code);

        log.log_event(event1).unwrap();
        log.log_event(event2).unwrap();
        log.log_event(event3).unwrap();

        let results = log.query_by_domain(Domain::Code);
        assert_eq!(results.len(), 2);

        let results = log.query_by_domain(Domain::Medical);
        assert_eq!(results.len(), 1);

        let results = log.query_by_domain(Domain::Legal);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_pagination() {
        let mut log = MetaLearningEventLog::new();

        // Log 10 events
        for i in 0..10 {
            let mut event = create_test_event(MetaLearningEventType::LambdaAdjustment);
            event.embedder_index = Some(i);
            log.log_event(event).unwrap();
        }

        // Query with limit
        let query = EventLogQuery::new().limit(3);
        let results = log.query(&query);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].embedder_index, Some(0));

        // Query with offset
        let query = EventLogQuery::new().offset(5);
        let results = log.query(&query);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].embedder_index, Some(5));

        // Query with both
        let query = EventLogQuery::new().offset(2).limit(3);
        let results = log.query(&query);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].embedder_index, Some(2));
        assert_eq!(results[2].embedder_index, Some(4));

        // Offset beyond length
        let query = EventLogQuery::new().offset(100);
        let results = log.query(&query);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut log = MetaLearningEventLog::with_config(100, 7);

        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();
        log.log_event(
            create_test_event(MetaLearningEventType::AccuracyAlert).with_domain(Domain::Code),
        )
        .unwrap();

        let json = log.to_json().unwrap();
        let restored = MetaLearningEventLog::from_json(&json).unwrap();

        assert_eq!(restored.event_count(), 2);
        assert_eq!(restored.total_logged(), 2);
        assert_eq!(restored.max_events(), 100);
        assert_eq!(restored.retention_days(), 7);

        // Verify event data survived
        let events: Vec<_> = restored.recent_events(10);
        assert_eq!(events[0].event_type, MetaLearningEventType::LambdaAdjustment);
        assert_eq!(events[1].event_type, MetaLearningEventType::AccuracyAlert);
        assert_eq!(events[1].domain, Domain::Code);
    }

    #[test]
    fn test_stats() {
        let mut log = MetaLearningEventLog::new();

        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::AccuracyAlert))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::BayesianEscalation))
            .unwrap();

        let stats = log.stats();
        assert_eq!(stats.current_count, 4);
        assert_eq!(stats.total_logged, 4);
        assert_eq!(stats.by_type.lambda_adjustment, 2);
        assert_eq!(stats.by_type.accuracy_alert, 1);
        assert_eq!(stats.by_type.bayesian_escalation, 1);
        assert_eq!(stats.by_type.weight_clamped, 0);
        assert!(stats.oldest_event.is_some());
        assert!(stats.newest_event.is_some());
    }

    #[test]
    fn test_stats_accuracy() {
        let mut log = MetaLearningEventLog::new();

        // Events with accuracy
        log.log_event(
            create_test_event(MetaLearningEventType::LambdaAdjustment).with_accuracy(0.8),
        )
        .unwrap();
        log.log_event(
            create_test_event(MetaLearningEventType::LambdaAdjustment).with_accuracy(0.6),
        )
        .unwrap();
        // Event without accuracy
        log.log_event(create_test_event(MetaLearningEventType::BayesianEscalation))
            .unwrap();

        let stats = log.stats();
        // Average of 0.8 and 0.6 = 0.7
        assert!((stats.avg_accuracy - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_clear() {
        let mut log = MetaLearningEventLog::new();

        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::AccuracyAlert))
            .unwrap();

        assert_eq!(log.event_count(), 2);

        log.clear();

        assert_eq!(log.event_count(), 0);
        assert_eq!(log.total_logged(), 2);
        assert_eq!(log.total_evicted(), 2);
        assert!(log.is_empty());
    }

    #[test]
    fn test_empty_log_queries() {
        let log = MetaLearningEventLog::new();

        let now = Utc::now();
        let results = log.query_by_time(now - Duration::hours(1), now);
        assert!(results.is_empty());

        let results = log.query_by_type(MetaLearningEventType::LambdaAdjustment);
        assert!(results.is_empty());

        let results = log.query_by_domain(Domain::Code);
        assert!(results.is_empty());

        let results = log.recent_events(10);
        assert!(results.is_empty());

        let results = log.by_accuracy_range(0.0, 1.0);
        assert!(results.is_empty());

        let results = log.escalation_events();
        assert!(results.is_empty());

        // Stats should work on empty log
        let stats = log.stats();
        assert_eq!(stats.current_count, 0);
        assert_eq!(stats.avg_accuracy, 0.0);
        assert!(stats.oldest_event.is_none());
        assert!(stats.newest_event.is_none());
    }

    #[test]
    fn test_recent_events() {
        let mut log = MetaLearningEventLog::new();

        for i in 0..10 {
            let mut event = create_test_event(MetaLearningEventType::LambdaAdjustment);
            event.embedder_index = Some(i);
            log.log_event(event).unwrap();
        }

        // Get last 3
        let results = log.recent_events(3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].embedder_index, Some(7));
        assert_eq!(results[1].embedder_index, Some(8));
        assert_eq!(results[2].embedder_index, Some(9));

        // Request more than available
        let results = log.recent_events(100);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_has_recent_events() {
        let mut log = MetaLearningEventLog::new();

        // Empty log has no recent events
        assert!(!log.has_recent_events(5));

        // Add an event
        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();

        // Should have recent events within last 5 minutes
        assert!(log.has_recent_events(5));
    }

    #[test]
    fn test_by_accuracy_range() {
        let mut log = MetaLearningEventLog::new();

        log.log_event(
            create_test_event(MetaLearningEventType::LambdaAdjustment).with_accuracy(0.5),
        )
        .unwrap();
        log.log_event(
            create_test_event(MetaLearningEventType::LambdaAdjustment).with_accuracy(0.7),
        )
        .unwrap();
        log.log_event(
            create_test_event(MetaLearningEventType::LambdaAdjustment).with_accuracy(0.9),
        )
        .unwrap();
        // No accuracy
        log.log_event(create_test_event(MetaLearningEventType::BayesianEscalation))
            .unwrap();

        let results = log.by_accuracy_range(0.6, 0.8);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].accuracy, Some(0.7));

        let results = log.by_accuracy_range(0.0, 1.0);
        assert_eq!(results.len(), 3); // Only those with accuracy
    }

    #[test]
    fn test_escalation_events() {
        let mut log = MetaLearningEventLog::new();

        log.log_event(create_test_event(MetaLearningEventType::LambdaAdjustment))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::BayesianEscalation))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::AccuracyAlert))
            .unwrap();
        log.log_event(create_test_event(MetaLearningEventType::BayesianEscalation))
            .unwrap();

        let results = log.escalation_events();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_event_type_count() {
        let mut count = EventTypeCount::default();

        count.increment(MetaLearningEventType::LambdaAdjustment);
        count.increment(MetaLearningEventType::LambdaAdjustment);
        count.increment(MetaLearningEventType::AccuracyAlert);

        assert_eq!(count.lambda_adjustment, 2);
        assert_eq!(count.accuracy_alert, 1);
        assert_eq!(count.total(), 3);
    }

    #[test]
    fn test_query_builder() {
        let query = EventLogQuery::new()
            .event_type(MetaLearningEventType::LambdaAdjustment)
            .domain(Domain::Code)
            .limit(10)
            .offset(5);

        assert_eq!(query.event_type, Some(MetaLearningEventType::LambdaAdjustment));
        assert_eq!(query.domain, Some(Domain::Code));
        assert_eq!(query.limit, Some(10));
        assert_eq!(query.offset, Some(5));
    }

    #[test]
    fn test_query_matches() {
        let event = create_test_event(MetaLearningEventType::LambdaAdjustment)
            .with_domain(Domain::Code)
            .with_accuracy(0.8);

        // Empty query matches everything
        let query = EventLogQuery::new();
        assert!(query.matches(&event));

        // Matching type
        let query = EventLogQuery::new().event_type(MetaLearningEventType::LambdaAdjustment);
        assert!(query.matches(&event));

        // Non-matching type
        let query = EventLogQuery::new().event_type(MetaLearningEventType::AccuracyAlert);
        assert!(!query.matches(&event));

        // Matching domain
        let query = EventLogQuery::new().domain(Domain::Code);
        assert!(query.matches(&event));

        // Non-matching domain
        let query = EventLogQuery::new().domain(Domain::Medical);
        assert!(!query.matches(&event));

        // Combined filters
        let query = EventLogQuery::new()
            .event_type(MetaLearningEventType::LambdaAdjustment)
            .domain(Domain::Code);
        assert!(query.matches(&event));
    }

    #[test]
    fn test_filter_custom() {
        let mut log = MetaLearningEventLog::new();

        for i in 0..10 {
            let mut event = create_test_event(MetaLearningEventType::LambdaAdjustment);
            event.embedder_index = Some(i);
            log.log_event(event).unwrap();
        }

        // Custom filter: even embedder indices only
        let results = log.filter(|e| e.embedder_index.map_or(false, |i| i % 2 == 0));
        assert_eq!(results.len(), 5);
    }

    // ========================================================================
    // FSV Tests
    // ========================================================================

    #[test]
    fn test_fsv_eviction_policy() {
        // FSV: Verify FIFO eviction maintains chronological order
        let mut log = MetaLearningEventLog::with_config(3, 7);

        // BEFORE: Empty log
        println!("FSV: Testing FIFO eviction policy");
        println!("BEFORE: event_count={}, total_logged={}", log.event_count(), log.total_logged());

        // Add 5 events (capacity is 3)
        for i in 0..5 {
            let mut event = create_test_event(MetaLearningEventType::LambdaAdjustment);
            event.embedder_index = Some(i);
            log.log_event(event).unwrap();
        }

        // AFTER
        println!(
            "AFTER: event_count={}, total_logged={}, total_evicted={}",
            log.event_count(),
            log.total_logged(),
            log.total_evicted()
        );

        // FSV Assertions
        assert_eq!(log.event_count(), 3, "FSV FAIL: Should have exactly 3 events");
        assert_eq!(log.total_evicted(), 2, "FSV FAIL: Should have evicted 2 events");

        // Verify FIFO: oldest events (0, 1) should be gone, (2, 3, 4) remain
        let events: Vec<_> = log.recent_events(10);
        assert_eq!(
            events[0].embedder_index,
            Some(2),
            "FSV FAIL: First remaining should be index 2"
        );
        assert_eq!(
            events[2].embedder_index,
            Some(4),
            "FSV FAIL: Last should be index 4"
        );
    }

    #[test]
    fn test_fsv_empty_log_safety() {
        // FSV: All operations should work on empty log without panic
        let log = MetaLearningEventLog::new();

        println!("FSV: Testing empty log safety");

        // All of these should return empty results, not panic
        let _ = log.query(&EventLogQuery::new());
        let _ = log.query_by_time(Utc::now(), Utc::now());
        let _ = log.query_by_type(MetaLearningEventType::LambdaAdjustment);
        let _ = log.query_by_domain(Domain::Code);
        let _ = log.recent_events(100);
        let _ = log.by_accuracy_range(0.0, 1.0);
        let _ = log.escalation_events();
        let _ = log.has_recent_events(60);
        let _ = log.stats();
        let _ = log.to_json();

        println!("FSV: All empty log operations completed without panic");
    }

    #[test]
    fn test_fsv_json_roundtrip_preserves_data() {
        // FSV: JSON serialization must preserve all fields
        let mut log = MetaLearningEventLog::with_config(50, 14);

        let event = create_test_event(MetaLearningEventType::AccuracyAlert)
            .with_domain(Domain::Medical)
            .with_accuracy(0.65);
        log.log_event(event).unwrap();

        println!("FSV: Testing JSON roundtrip preservation");

        // Serialize
        let json = log.to_json().unwrap();
        println!("JSON length: {} bytes", json.len());

        // Deserialize
        let restored = MetaLearningEventLog::from_json(&json).unwrap();

        // FSV Assertions
        assert_eq!(
            restored.max_events(),
            50,
            "FSV FAIL: max_events not preserved"
        );
        assert_eq!(
            restored.retention_days(),
            14,
            "FSV FAIL: retention_days not preserved"
        );
        assert_eq!(
            restored.event_count(),
            1,
            "FSV FAIL: event count not preserved"
        );

        let events: Vec<_> = restored.recent_events(1);
        assert_eq!(
            events[0].event_type,
            MetaLearningEventType::AccuracyAlert,
            "FSV FAIL: event_type not preserved"
        );
        assert_eq!(
            events[0].domain,
            Domain::Medical,
            "FSV FAIL: domain not preserved"
        );
        assert_eq!(
            events[0].accuracy,
            Some(0.65),
            "FSV FAIL: accuracy not preserved"
        );
    }

    #[test]
    fn test_fsv_time_range_inclusive() {
        // FSV: Time range queries must be inclusive on both bounds
        let mut log = MetaLearningEventLog::new();

        let event = create_test_event(MetaLearningEventType::LambdaAdjustment);
        let event_time = event.created_at;
        log.log_event(event).unwrap();

        println!("FSV: Testing time range inclusivity");

        // Query with exact bounds
        let results = log.query_by_time(event_time, event_time);
        assert_eq!(
            results.len(),
            1,
            "FSV FAIL: Exact time bounds should include the event"
        );

        // Query with bounds just inside
        let results = log.query_by_time(event_time - Duration::milliseconds(1), event_time);
        assert_eq!(
            results.len(),
            1,
            "FSV FAIL: Event at end bound should be included"
        );

        let results = log.query_by_time(event_time, event_time + Duration::milliseconds(1));
        assert_eq!(
            results.len(),
            1,
            "FSV FAIL: Event at start bound should be included"
        );
    }

    #[test]
    fn test_fsv_pagination_correctness() {
        // FSV: Pagination must work correctly with offset and limit
        let mut log = MetaLearningEventLog::new();

        for i in 0..20 {
            let mut event = create_test_event(MetaLearningEventType::LambdaAdjustment);
            event.embedder_index = Some(i);
            log.log_event(event).unwrap();
        }

        println!("FSV: Testing pagination correctness");

        // Page 1: items 0-4
        let query = EventLogQuery::new().offset(0).limit(5);
        let page1 = log.query(&query);
        assert_eq!(page1.len(), 5, "FSV FAIL: Page 1 should have 5 items");
        assert_eq!(
            page1[0].embedder_index,
            Some(0),
            "FSV FAIL: Page 1 first item"
        );

        // Page 2: items 5-9
        let query = EventLogQuery::new().offset(5).limit(5);
        let page2 = log.query(&query);
        assert_eq!(page2.len(), 5, "FSV FAIL: Page 2 should have 5 items");
        assert_eq!(
            page2[0].embedder_index,
            Some(5),
            "FSV FAIL: Page 2 first item"
        );

        // Last page with partial results
        let query = EventLogQuery::new().offset(18).limit(5);
        let last_page = log.query(&query);
        assert_eq!(
            last_page.len(),
            2,
            "FSV FAIL: Last page should have 2 items"
        );
        assert_eq!(
            last_page[0].embedder_index,
            Some(18),
            "FSV FAIL: Last page first item"
        );

        // Offset past end
        let query = EventLogQuery::new().offset(100);
        let empty = log.query(&query);
        assert!(empty.is_empty(), "FSV FAIL: Offset past end should be empty");
    }
}
