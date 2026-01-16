# TASK-P4-009: TopicStabilityTracker

```xml
<task_spec id="TASK-P4-009" version="1.0">
<metadata>
  <title>TopicStabilityTracker Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>35</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P4-002</task_ref>
    <task_ref>TASK-P4-008</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements TopicStabilityTracker which monitors topic churn over time and
triggers dream consolidation when high entropy is detected. Maintains snapshots
of topic state and computes churn rates between snapshots.

This component integrates with the dream consolidation system to trigger
memory reorganization when topics become unstable.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#component_contracts</file>
  <file purpose="topic">crates/context-graph-core/src/clustering/topic.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-002 complete (Topic types exist)</check>
  <check>TASK-P4-008 complete (TopicSynthesizer exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement TopicStabilityTracker struct
    - Implement track_churn method
    - Implement check_dream_trigger method
    - Implement take_snapshot method
    - Maintain snapshot history (24 hours)
    - Compute topic added/removed counts
  </in_scope>
  <out_of_scope>
    - Dream execution (Phase 6)
    - Topic persistence
    - Long-term trend analysis
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/stability.rs">
      pub struct TopicSnapshot {
          pub timestamp: DateTime&lt;Utc&gt;,
          pub topic_ids: Vec&lt;Uuid&gt;,
          pub topic_profiles: Vec&lt;TopicProfile&gt;,
          pub total_members: usize,
      }

      pub struct TopicStabilityTracker {
          snapshots: VecDeque&lt;TopicSnapshot&gt;,
          current_churn: f32,
          high_entropy_start: Option&lt;DateTime&lt;Utc&gt;&gt;,
          churn_threshold: f32,
          entropy_threshold: f32,
          entropy_duration_secs: u64,
      }

      impl TopicStabilityTracker {
          pub fn new() -> Self;
          pub fn track_churn(&amp;mut self) -> f32;
          pub fn check_dream_trigger(&amp;self, entropy: f32) -> bool;
          pub fn take_snapshot(&amp;mut self, topics: &amp;[Topic]);
          pub fn get_churn_history(&amp;self) -> Vec&lt;(DateTime&lt;Utc&gt;, f32)&gt;;
          fn cleanup_old_snapshots(&amp;mut self);
      }
    </signature>
  </signatures>

  <constraints>
    - Keep last 24 hours of snapshots
    - churn = (topics_added + topics_removed) / total_topics
    - Dream trigger: (entropy > 0.7 AND churn > 0.5) OR (entropy > 0.7 for 5+ min)
  </constraints>

  <verification>
    - track_churn computes correct rate
    - check_dream_trigger fires correctly
    - Old snapshots cleaned up
    - Churn history accessible
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/stability.rs

use std::collections::{VecDeque, HashSet};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use super::topic::{Topic, TopicProfile};

/// Snapshot of topic state at a point in time
#[derive(Debug, Clone)]
pub struct TopicSnapshot {
    /// When this snapshot was taken
    pub timestamp: DateTime&lt;Utc&gt;,
    /// Topic IDs at this time
    pub topic_ids: Vec&lt;Uuid&gt;,
    /// Topic profiles at this time
    pub topic_profiles: Vec&lt;TopicProfile&gt;,
    /// Total members across all topics
    pub total_members: usize,
}

impl TopicSnapshot {
    /// Create a snapshot from current topics
    pub fn from_topics(topics: &amp;[Topic]) -> Self {
        let topic_ids: Vec&lt;Uuid&gt; = topics.iter().map(|t| t.id).collect();
        let topic_profiles: Vec&lt;TopicProfile&gt; = topics.iter().map(|t| t.profile.clone()).collect();
        let total_members: usize = topics.iter().map(|t| t.member_count()).sum();

        Self {
            timestamp: Utc::now(),
            topic_ids,
            topic_profiles,
            total_members,
        }
    }
}

/// Configuration for stability tracking
const DEFAULT_CHURN_THRESHOLD: f32 = 0.5;
const DEFAULT_ENTROPY_THRESHOLD: f32 = 0.7;
const DEFAULT_ENTROPY_DURATION_SECS: u64 = 5 * 60; // 5 minutes
const SNAPSHOT_RETENTION_HOURS: i64 = 24;

/// Tracks topic stability and triggers dream consolidation
pub struct TopicStabilityTracker {
    /// Historical snapshots
    snapshots: VecDeque&lt;TopicSnapshot&gt;,
    /// Current churn rate
    current_churn: f32,
    /// When high entropy started (for duration check)
    high_entropy_start: Option&lt;DateTime&lt;Utc&gt;&gt;,
    /// Churn threshold for dream trigger
    churn_threshold: f32,
    /// Entropy threshold for dream trigger
    entropy_threshold: f32,
    /// Duration of high entropy needed for trigger (seconds)
    entropy_duration_secs: u64,
    /// History of churn calculations
    churn_history: VecDeque&lt;(DateTime&lt;Utc&gt;, f32)&gt;,
}

impl Default for TopicStabilityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicStabilityTracker {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            snapshots: VecDeque::new(),
            current_churn: 0.0,
            high_entropy_start: None,
            churn_threshold: DEFAULT_CHURN_THRESHOLD,
            entropy_threshold: DEFAULT_ENTROPY_THRESHOLD,
            entropy_duration_secs: DEFAULT_ENTROPY_DURATION_SECS,
            churn_history: VecDeque::new(),
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(
        churn_threshold: f32,
        entropy_threshold: f32,
        entropy_duration_secs: u64,
    ) -> Self {
        Self {
            snapshots: VecDeque::new(),
            current_churn: 0.0,
            high_entropy_start: None,
            churn_threshold,
            entropy_threshold,
            entropy_duration_secs,
            churn_history: VecDeque::new(),
        }
    }

    /// Take a snapshot of current topic state
    pub fn take_snapshot(&amp;mut self, topics: &amp;[Topic]) {
        let snapshot = TopicSnapshot::from_topics(topics);
        self.snapshots.push_back(snapshot);
        self.cleanup_old_snapshots();
    }

    /// Track churn by comparing to snapshot from 1 hour ago
    pub fn track_churn(&amp;mut self) -> f32 {
        let now = Utc::now();
        let one_hour_ago = now - Duration::hours(1);

        // Find closest snapshot to 1 hour ago
        let old_snapshot = self.snapshots.iter()
            .filter(|s| s.timestamp <= one_hour_ago)
            .last()
            .cloned();

        let current_snapshot = self.snapshots.back().cloned();

        let churn = match (old_snapshot, current_snapshot) {
            (Some(old), Some(current)) => {
                self.compute_churn(&amp;old, &amp;current)
            }
            _ => 0.0,
        };

        self.current_churn = churn;
        self.churn_history.push_back((now, churn));

        // Keep last 24 hours of churn history
        while let Some((timestamp, _)) = self.churn_history.front() {
            if now - *timestamp > Duration::hours(24) {
                self.churn_history.pop_front();
            } else {
                break;
            }
        }

        churn
    }

    /// Compute churn between two snapshots
    fn compute_churn(&amp;self, old: &amp;TopicSnapshot, current: &amp;TopicSnapshot) -> f32 {
        let old_ids: HashSet&lt;_&gt; = old.topic_ids.iter().collect();
        let current_ids: HashSet&lt;_&gt; = current.topic_ids.iter().collect();

        let added = current_ids.difference(&amp;old_ids).count();
        let removed = old_ids.difference(&amp;current_ids).count();
        let total = old_ids.union(&amp;current_ids).count();

        if total == 0 {
            return 0.0;
        }

        (added + removed) as f32 / total as f32
    }

    /// Check if dream consolidation should be triggered
    pub fn check_dream_trigger(&amp;mut self, entropy: f32) -> bool {
        let now = Utc::now();

        // Track high entropy duration
        if entropy > self.entropy_threshold {
            if self.high_entropy_start.is_none() {
                self.high_entropy_start = Some(now);
            }
        } else {
            self.high_entropy_start = None;
        }

        // Condition 1: High entropy AND high churn
        if entropy > self.entropy_threshold &amp;&amp; self.current_churn > self.churn_threshold {
            return true;
        }

        // Condition 2: High entropy for sustained duration
        if let Some(start) = self.high_entropy_start {
            let duration = (now - start).num_seconds() as u64;
            if duration >= self.entropy_duration_secs {
                return true;
            }
        }

        false
    }

    /// Get current churn rate
    pub fn current_churn(&amp;self) -> f32 {
        self.current_churn
    }

    /// Get churn history
    pub fn get_churn_history(&amp;self) -> Vec&lt;(DateTime&lt;Utc&gt;, f32)&gt; {
        self.churn_history.iter().cloned().collect()
    }

    /// Get snapshot count
    pub fn snapshot_count(&amp;self) -> usize {
        self.snapshots.len()
    }

    /// Get latest snapshot
    pub fn latest_snapshot(&amp;self) -> Option&lt;&amp;TopicSnapshot&gt; {
        self.snapshots.back()
    }

    /// Clean up snapshots older than 24 hours
    fn cleanup_old_snapshots(&amp;mut self) {
        let cutoff = Utc::now() - Duration::hours(SNAPSHOT_RETENTION_HOURS);

        while let Some(snapshot) = self.snapshots.front() {
            if snapshot.timestamp < cutoff {
                self.snapshots.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get average churn over a time period
    pub fn average_churn(&amp;self, hours: i64) -> f32 {
        let cutoff = Utc::now() - Duration::hours(hours);

        let relevant: Vec&lt;f32&gt; = self.churn_history
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, c)| *c)
            .collect();

        if relevant.is_empty() {
            return 0.0;
        }

        relevant.iter().sum::<f32>() / relevant.len() as f32
    }

    /// Check if system is stable (low churn over time)
    pub fn is_stable(&amp;self) -> bool {
        let avg_churn = self.average_churn(6); // Last 6 hours
        avg_churn < 0.2
    }

    /// Get topic count change
    pub fn topic_count_change(&amp;self) -> (i32, i32) {
        if self.snapshots.len() < 2 {
            return (0, 0);
        }

        let current = self.snapshots.back().map(|s| s.topic_ids.len()).unwrap_or(0);
        let oldest = self.snapshots.front().map(|s| s.topic_ids.len()).unwrap_or(0);

        let added = (current as i32 - oldest as i32).max(0);
        let removed = (oldest as i32 - current as i32).max(0);

        (added, removed)
    }

    /// Reset high entropy tracking (after dream completes)
    pub fn reset_entropy_tracking(&amp;mut self) {
        self.high_entropy_start = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::topic::TopicProfile;
    use std::collections::HashMap;

    fn create_test_topics(count: usize) -> Vec&lt;Topic&gt; {
        (0..count)
            .map(|_| {
                let profile = TopicProfile::new([0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                Topic::new(profile, HashMap::new(), vec![Uuid::new_v4()])
            })
            .collect()
    }

    #[test]
    fn test_snapshot_creation() {
        let topics = create_test_topics(5);
        let snapshot = TopicSnapshot::from_topics(&amp;topics);

        assert_eq!(snapshot.topic_ids.len(), 5);
        assert_eq!(snapshot.topic_profiles.len(), 5);
    }

    #[test]
    fn test_take_snapshot() {
        let mut tracker = TopicStabilityTracker::new();
        let topics = create_test_topics(5);

        tracker.take_snapshot(&amp;topics);
        assert_eq!(tracker.snapshot_count(), 1);

        tracker.take_snapshot(&amp;topics);
        assert_eq!(tracker.snapshot_count(), 2);
    }

    #[test]
    fn test_churn_calculation() {
        let tracker = TopicStabilityTracker::new();

        let old = TopicSnapshot {
            timestamp: Utc::now() - Duration::hours(2),
            topic_ids: vec![Uuid::new_v4(), Uuid::new_v4()],
            topic_profiles: vec![],
            total_members: 10,
        };

        // One topic removed, one added, one unchanged
        let current = TopicSnapshot {
            timestamp: Utc::now(),
            topic_ids: vec![old.topic_ids[0], Uuid::new_v4()],
            topic_profiles: vec![],
            total_members: 10,
        };

        // 1 removed + 1 added = 2 changes, 3 total topics (2 old + 1 new unique)
        let churn = tracker.compute_churn(&amp;old, &amp;current);
        assert!(churn > 0.0);
    }

    #[test]
    fn test_dream_trigger_high_entropy_and_churn() {
        let mut tracker = TopicStabilityTracker::with_thresholds(0.5, 0.7, 300);
        tracker.current_churn = 0.6;

        // High entropy + high churn = trigger
        assert!(tracker.check_dream_trigger(0.8));
    }

    #[test]
    fn test_dream_trigger_low_entropy() {
        let mut tracker = TopicStabilityTracker::new();
        tracker.current_churn = 0.6;

        // Low entropy = no trigger even with high churn
        assert!(!tracker.check_dream_trigger(0.3));
    }

    #[test]
    fn test_churn_history() {
        let mut tracker = TopicStabilityTracker::new();

        // Add two snapshots with different topics
        let topics1 = create_test_topics(5);
        tracker.take_snapshot(&amp;topics1);

        // Force a snapshot timestamp in the past for churn calculation
        if let Some(snapshot) = tracker.snapshots.back_mut() {
            snapshot.timestamp = Utc::now() - Duration::hours(2);
        }

        let topics2 = create_test_topics(3);
        tracker.take_snapshot(&amp;topics2);

        tracker.track_churn();

        let history = tracker.get_churn_history();
        assert!(!history.is_empty());
    }

    #[test]
    fn test_is_stable() {
        let tracker = TopicStabilityTracker::new();

        // With no history, should be considered stable (no evidence of instability)
        assert!(tracker.is_stable());
    }

    #[test]
    fn test_reset_entropy_tracking() {
        let mut tracker = TopicStabilityTracker::new();

        // Trigger high entropy tracking
        tracker.check_dream_trigger(0.8);
        assert!(tracker.high_entropy_start.is_some());

        // Reset
        tracker.reset_entropy_tracking();
        assert!(tracker.high_entropy_start.is_none());
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/stability.rs">TopicStabilityTracker implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">Add pub mod stability and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>take_snapshot captures topic state</criterion>
  <criterion>track_churn computes correct rate</criterion>
  <criterion>check_dream_trigger fires on high entropy + high churn</criterion>
  <criterion>check_dream_trigger fires on sustained high entropy (5+ min)</criterion>
  <criterion>Old snapshots cleaned up after 24 hours</criterion>
  <criterion>Churn history accessible</criterion>
  <criterion>reset_entropy_tracking clears state</criterion>
</validation_criteria>

<test_commands>
  <command description="Run stability tests">cargo test --package context-graph-core stability</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create stability.rs
- [ ] Implement TopicSnapshot struct
- [ ] Implement TopicStabilityTracker struct
- [ ] Implement take_snapshot method
- [ ] Implement track_churn method
- [ ] Implement compute_churn helper
- [ ] Implement check_dream_trigger
- [ ] Implement cleanup_old_snapshots
- [ ] Implement churn history tracking
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Phase 4 complete!
