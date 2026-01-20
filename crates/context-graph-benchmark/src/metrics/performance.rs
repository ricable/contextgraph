//! Performance metrics: latency, throughput, memory.
//!
//! These metrics track system performance characteristics.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// Performance metrics for a benchmark run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Latency percentiles (microseconds).
    pub latency_us: LatencyStats,

    /// Throughput (operations per second).
    pub throughput_ops_sec: f64,

    /// Memory usage (bytes).
    pub memory: MemoryStats,

    /// Index build time (microseconds).
    pub index_build_time_us: u64,

    /// Number of operations measured.
    pub operation_count: usize,
}

/// Latency statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Minimum latency (microseconds).
    pub min: u64,

    /// Maximum latency (microseconds).
    pub max: u64,

    /// Mean latency (microseconds).
    pub mean: f64,

    /// Standard deviation (microseconds).
    pub std_dev: f64,

    /// Percentiles (key = percentile * 100, e.g., 50 for p50).
    pub percentiles: BTreeMap<u32, u64>,
}

impl LatencyStats {
    /// Create from a list of latency measurements (in microseconds).
    pub fn from_measurements(measurements: &[u64]) -> Self {
        if measurements.is_empty() {
            return Self::default();
        }

        let mut sorted = measurements.to_vec();
        sorted.sort_unstable();

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let sum: u64 = sorted.iter().sum();
        let mean = sum as f64 / sorted.len() as f64;

        let variance: f64 = sorted
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / sorted.len() as f64;
        let std_dev = variance.sqrt();

        let mut percentiles = BTreeMap::new();
        for p in [50, 75, 90, 95, 99] {
            let idx = (p as f64 / 100.0 * (sorted.len() - 1) as f64).round() as usize;
            percentiles.insert(p, sorted[idx.min(sorted.len() - 1)]);
        }

        Self {
            min,
            max,
            mean,
            std_dev,
            percentiles,
        }
    }

    /// Get p50 (median).
    pub fn p50(&self) -> u64 {
        self.percentiles.get(&50).copied().unwrap_or(0)
    }

    /// Get p95.
    pub fn p95(&self) -> u64 {
        self.percentiles.get(&95).copied().unwrap_or(0)
    }

    /// Get p99.
    pub fn p99(&self) -> u64 {
        self.percentiles.get(&99).copied().unwrap_or(0)
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Index memory usage (bytes).
    pub index_bytes: usize,

    /// Corpus memory usage (bytes).
    pub corpus_bytes: usize,

    /// Peak memory usage (bytes).
    pub peak_bytes: usize,

    /// Memory per document (bytes).
    pub per_document_bytes: f64,
}

impl MemoryStats {
    /// Total memory usage.
    pub fn total(&self) -> usize {
        self.index_bytes + self.corpus_bytes
    }
}

impl PerformanceMetrics {
    /// Create from raw measurements.
    pub fn from_measurements(
        latencies_us: &[u64],
        total_duration: Duration,
        memory: MemoryStats,
        index_build_time: Duration,
    ) -> Self {
        let operation_count = latencies_us.len();
        let throughput_ops_sec = if total_duration.as_secs_f64() > 0.0 {
            operation_count as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            latency_us: LatencyStats::from_measurements(latencies_us),
            throughput_ops_sec,
            memory,
            index_build_time_us: index_build_time.as_micros() as u64,
            operation_count,
        }
    }

    /// Normalized performance score (0-1, higher is better).
    ///
    /// Based on targets from constitution.yaml.
    pub fn normalized_score(&self) -> f64 {
        use crate::config::targets;

        // Score based on p95 latency vs target
        let latency_score = if self.latency_us.p95() == 0 {
            1.0
        } else {
            let target_us = targets::SEARCH_LATENCY_MS as f64 * 1000.0;
            (target_us / self.latency_us.p95() as f64).min(1.0)
        };

        // Score based on throughput (assume 1000 ops/sec is good baseline)
        let throughput_score = (self.throughput_ops_sec / 1000.0).min(1.0);

        // Weighted combination (70% latency, 30% throughput)
        0.7 * latency_score + 0.3 * throughput_score
    }

    /// Check if performance meets targets.
    pub fn meets_targets(&self, latency_target_ms: u64) -> bool {
        let latency_target_us = latency_target_ms * 1000;
        self.latency_us.p95() <= latency_target_us
    }
}

/// Latency tracker for accumulating measurements during benchmark.
#[derive(Debug, Clone, Default)]
pub struct LatencyTracker {
    measurements: Vec<u64>,
}

impl LatencyTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            measurements: Vec::with_capacity(capacity),
        }
    }

    /// Record a latency measurement (microseconds).
    pub fn record(&mut self, latency_us: u64) {
        self.measurements.push(latency_us);
    }

    /// Record a duration.
    pub fn record_duration(&mut self, duration: Duration) {
        self.measurements.push(duration.as_micros() as u64);
    }

    /// Get all measurements.
    pub fn measurements(&self) -> &[u64] {
        &self.measurements
    }

    /// Compute latency stats.
    pub fn stats(&self) -> LatencyStats {
        LatencyStats::from_measurements(&self.measurements)
    }

    /// Clear all measurements.
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

/// Memory tracker for monitoring memory usage.
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    initial_bytes: usize,
    peak_bytes: usize,
    current_bytes: usize,
}

impl MemoryTracker {
    /// Create a new tracker with initial memory baseline.
    pub fn new(initial_bytes: usize) -> Self {
        Self {
            initial_bytes,
            peak_bytes: initial_bytes,
            current_bytes: initial_bytes,
        }
    }

    /// Update current memory usage.
    pub fn update(&mut self, current_bytes: usize) {
        self.current_bytes = current_bytes;
        if current_bytes > self.peak_bytes {
            self.peak_bytes = current_bytes;
        }
    }

    /// Get delta from initial memory.
    pub fn delta(&self) -> usize {
        self.current_bytes.saturating_sub(self.initial_bytes)
    }

    /// Get peak delta from initial memory.
    pub fn peak_delta(&self) -> usize {
        self.peak_bytes.saturating_sub(self.initial_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats() {
        let measurements = vec![100, 200, 300, 400, 500];
        let stats = LatencyStats::from_measurements(&measurements);

        assert_eq!(stats.min, 100);
        assert_eq!(stats.max, 500);
        assert!((stats.mean - 300.0).abs() < 0.01);
        assert_eq!(stats.p50(), 300);
    }

    #[test]
    fn test_latency_percentiles() {
        let measurements: Vec<u64> = (1..=100).collect();
        let stats = LatencyStats::from_measurements(&measurements);

        // With 100 values (1-100), percentile idx = (p/100 * 99).round()
        // p50: (50/100 * 99).round() = 50, sorted[50] = 51
        // p95: (95/100 * 99).round() = 94, sorted[94] = 95
        // p99: (99/100 * 99).round() = 98, sorted[98] = 99
        assert_eq!(stats.p50(), 51);
        assert_eq!(stats.p95(), 95);
        assert_eq!(stats.p99(), 99);
    }

    #[test]
    fn test_latency_tracker() {
        let mut tracker = LatencyTracker::new();
        tracker.record(100);
        tracker.record(200);
        tracker.record(300);

        let stats = tracker.stats();
        assert_eq!(stats.min, 100);
        assert_eq!(stats.max, 300);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new(1000);
        tracker.update(1500);
        tracker.update(1200);

        assert_eq!(tracker.delta(), 200);
        assert_eq!(tracker.peak_delta(), 500);
    }
}
