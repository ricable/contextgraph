//! Teleological Drift Detection with Per-Embedder Analysis (TASK-LOGIC-010)
//!
//! This module implements drift detection across all 13 embedders, providing
//! granular insight into which semantic dimensions are drifting from established goals.
//!
//! # Architecture
//!
//! From constitution.yaml (ARCH-02): "Compare Only Compatible Embedding Types (Apples-to-Apples)"
//! - E1 compares with E1, E5 with E5, NEVER cross-embedder
//! - Uses TeleologicalComparator for all comparisons
//!
//! # Design Philosophy
//!
//! Per-embedder drift detection provides:
//! 1. **Granular insight**: Know exactly which embedder is drifting
//! 2. **Early warning**: Detect drift in specific dimensions before overall alignment degrades
//! 3. **Actionable recommendations**: Embedder-specific suggestions for correction
//! 4. **Trend analysis**: Predict when drift will become critical
//!
//! # Example
//!
//! ```ignore
//! use context_graph_core::autonomous::drift::{TeleologicalDriftDetector, DriftThresholds};
//! use context_graph_core::teleological::TeleologicalComparator;
//!
//! let comparator = TeleologicalComparator::new();
//! let detector = TeleologicalDriftDetector::new(comparator);
//!
//! let result = detector.check_drift(&memories, &goal, &comparison_type)?;
//! println!("Overall drift level: {:?}", result.overall_drift.drift_level);
//! for info in &result.most_drifted_embedders {
//!     println!("{:?}: {:?}", info.embedder, info.drift_level);
//! }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;

use crate::teleological::{
    Embedder, MatrixSearchConfig, SearchStrategy, TeleologicalComparator,
};
use crate::types::SemanticFingerprint;

// ============================================
// BACKWARD COMPATIBILITY TYPES
// (Preserved from original drift.rs for NORTH-010/011 services)
// ============================================

/// Drift detection configuration (legacy, for NORTH-010/011 services).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Enable continuous monitoring
    pub monitoring: DriftMonitoring,

    /// Alert threshold (alignment drop to trigger alert)
    pub alert_threshold: f32, // default: 0.05

    /// Enable auto-correction
    pub auto_correct: bool,

    /// Severe drift threshold (requires user intervention)
    pub severe_threshold: f32, // default: 0.10

    /// Rolling window size in days
    pub window_days: u32, // default: 7
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            monitoring: DriftMonitoring::Continuous,
            alert_threshold: 0.05,
            auto_correct: true,
            severe_threshold: 0.10,
            window_days: 7,
        }
    }
}

/// Monitoring mode for drift detection (legacy).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DriftMonitoring {
    /// Continuous real-time monitoring
    Continuous,
    /// Periodic checks at specified interval
    Periodic { interval_hours: u32 },
    /// Manual checks only
    Manual,
}

/// Drift severity levels (legacy, 4 levels for NORTH-010/011 services).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DriftSeverity {
    /// No significant drift detected
    None,
    /// Mild drift (< alert_threshold)
    Mild,
    /// Moderate drift (>= alert_threshold, < severe_threshold)
    Moderate,
    /// Severe drift (>= severe_threshold)
    Severe,
}

/// A single drift data point for history (legacy).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftDataPoint {
    /// Mean alignment score at this point
    pub alignment_mean: f32,
    /// Number of new memories added
    pub new_memories_count: u32,
    /// Timestamp of this data point
    pub timestamp: DateTime<Utc>,
}

/// Current drift state (legacy, for NORTH-010/011 services).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftState {
    /// Rolling mean alignment (window-based)
    pub rolling_mean: f32,

    /// Baseline alignment (established mean)
    pub baseline: f32,

    /// Current drift magnitude (baseline - rolling_mean)
    pub drift: f32,

    /// Drift severity
    pub severity: DriftSeverity,

    /// Trend direction
    pub trend: DriftTrend,

    /// Last check timestamp
    pub checked_at: DateTime<Utc>,

    /// Historical data points
    pub history: VecDeque<DriftDataPoint>,
}

impl Default for DriftState {
    fn default() -> Self {
        Self {
            rolling_mean: 0.75,
            baseline: 0.75,
            drift: 0.0,
            severity: DriftSeverity::None,
            trend: DriftTrend::Stable,
            checked_at: Utc::now(),
            history: VecDeque::with_capacity(168), // 7 days * 24 hours
        }
    }
}

impl DriftState {
    /// Create a new DriftState with a specific baseline
    pub fn with_baseline(baseline: f32) -> Self {
        Self {
            rolling_mean: baseline,
            baseline,
            drift: 0.0,
            severity: DriftSeverity::None,
            trend: DriftTrend::Stable,
            checked_at: Utc::now(),
            history: VecDeque::with_capacity(168),
        }
    }

    /// Add a new data point and update rolling statistics
    pub fn add_data_point(&mut self, mean_alignment: f32, new_memories: u32, config: &DriftConfig) {
        let point = DriftDataPoint {
            alignment_mean: mean_alignment,
            new_memories_count: new_memories,
            timestamp: Utc::now(),
        };

        self.history.push_back(point);

        // Keep only window_days worth of data (assuming hourly points)
        let max_points = (config.window_days * 24) as usize;
        while self.history.len() > max_points {
            self.history.pop_front();
        }

        self.update_rolling_mean();
        self.update_severity(config);
        self.checked_at = Utc::now();
    }

    /// Update the rolling mean from history
    fn update_rolling_mean(&mut self) {
        if self.history.is_empty() {
            return;
        }

        let sum: f32 = self.history.iter().map(|p| p.alignment_mean).sum();
        let old_mean = self.rolling_mean;
        self.rolling_mean = sum / self.history.len() as f32;
        self.drift = self.baseline - self.rolling_mean;

        // Determine trend based on change in rolling mean
        let delta = self.rolling_mean - old_mean;
        self.trend = if delta.abs() < 0.01 {
            DriftTrend::Stable
        } else if delta > 0.0 {
            DriftTrend::Improving
        } else {
            DriftTrend::Declining
        };
    }

    /// Update severity classification based on drift magnitude
    fn update_severity(&mut self, config: &DriftConfig) {
        self.severity = if self.drift.abs() >= config.severe_threshold {
            DriftSeverity::Severe
        } else if self.drift.abs() >= config.alert_threshold {
            DriftSeverity::Moderate
        } else if self.drift.abs() > 0.01 {
            DriftSeverity::Mild
        } else {
            DriftSeverity::None
        };
    }

    /// Reset baseline to current rolling mean
    pub fn reset_baseline(&mut self) {
        self.baseline = self.rolling_mean;
        self.drift = 0.0;
        self.severity = DriftSeverity::None;
    }

    /// Check if drift requires attention (moderate or severe)
    pub fn requires_attention(&self) -> bool {
        matches!(
            self.severity,
            DriftSeverity::Moderate | DriftSeverity::Severe
        )
    }

    /// Check if drift is severe enough to require user intervention
    pub fn requires_intervention(&self) -> bool {
        matches!(self.severity, DriftSeverity::Severe)
    }

    /// Get the number of data points in history
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get total new memories across all data points in history
    pub fn total_new_memories(&self) -> u64 {
        self.history
            .iter()
            .map(|p| p.new_memories_count as u64)
            .sum()
    }
}

// ============================================
// TELEOLOGICAL DRIFT TYPES (TASK-LOGIC-010)
// ============================================

/// Number of embedders in the system.
const NUM_EMBEDDERS: usize = 13;

/// Minimum samples required for trend analysis.
const MIN_TREND_SAMPLES: usize = 3;

/// Maximum most-drifted embedders to return.
const MAX_MOST_DRIFTED: usize = 5;

// ============================================
// CORE TYPES
// ============================================

/// Drift severity levels (5 levels, ordered worst-to-best for Ord).
///
/// Uses ordering where Critical < High < Medium < Low < None, so that
/// sorting in ascending order puts worst drift first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriftLevel {
    /// Critical drift: similarity < 0.40
    Critical,
    /// High drift: similarity >= 0.40, < 0.55
    High,
    /// Medium drift: similarity >= 0.55, < 0.70
    Medium,
    /// Low drift: similarity >= 0.70, < 0.85
    Low,
    /// No significant drift: similarity >= 0.85
    None,
}

impl DriftLevel {
    /// Classify a similarity score to a drift level using the given thresholds.
    #[inline]
    pub fn from_similarity(similarity: f32, thresholds: &DriftThresholds) -> Self {
        if similarity >= thresholds.none_min {
            DriftLevel::None
        } else if similarity >= thresholds.low_min {
            DriftLevel::Low
        } else if similarity >= thresholds.medium_min {
            DriftLevel::Medium
        } else if similarity >= thresholds.high_min {
            DriftLevel::High
        } else {
            DriftLevel::Critical
        }
    }

    /// Check if this level indicates drift occurred.
    #[inline]
    pub fn has_drifted(self) -> bool {
        self != DriftLevel::None
    }

    /// Check if this level requires recommendations.
    #[inline]
    pub fn needs_recommendation(self) -> bool {
        matches!(self, DriftLevel::Critical | DriftLevel::High | DriftLevel::Medium)
    }
}

/// Drift trend direction computed from history via linear regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftTrend {
    /// Alignment is improving (positive slope in similarity)
    Improving,
    /// Alignment is stable (|slope| < 0.01)
    Stable,
    /// Alignment is worsening (negative slope in similarity) - new name for TASK-LOGIC-010
    Worsening,
    /// Alignment is declining (legacy name, same as Worsening)
    Declining,
}

/// Configuration for drift thresholds.
///
/// Thresholds define similarity score boundaries for each drift level.
/// Must satisfy: none_min > low_min > medium_min > high_min
#[derive(Debug, Clone)]
pub struct DriftThresholds {
    /// Minimum similarity for DriftLevel::None (default: 0.85)
    pub none_min: f32,
    /// Minimum similarity for DriftLevel::Low (default: 0.70)
    pub low_min: f32,
    /// Minimum similarity for DriftLevel::Medium (default: 0.55)
    pub medium_min: f32,
    /// Minimum similarity for DriftLevel::High (default: 0.40)
    pub high_min: f32,
    // Below high_min = DriftLevel::Critical
}

impl Default for DriftThresholds {
    fn default() -> Self {
        Self {
            none_min: 0.85,
            low_min: 0.70,
            medium_min: 0.55,
            high_min: 0.40,
        }
    }
}

impl DriftThresholds {
    /// Validate that thresholds are in proper order.
    ///
    /// Returns an error if thresholds are not strictly decreasing.
    pub fn validate(&self) -> Result<(), DriftError> {
        if self.none_min <= self.low_min
            || self.low_min <= self.medium_min
            || self.medium_min <= self.high_min
            || self.high_min <= 0.0
            || self.none_min > 1.0
        {
            return Err(DriftError::InvalidThresholds {
                reason: format!(
                    "Thresholds must be: 1.0 >= none ({}) > low ({}) > medium ({}) > high ({}) > 0.0",
                    self.none_min, self.low_min, self.medium_min, self.high_min
                ),
            });
        }
        Ok(())
    }
}

// ============================================
// ERROR TYPES
// ============================================

/// Error types for drift detection. All errors are fatal per FAIL FAST.
#[derive(Debug, Clone)]
pub enum DriftError {
    /// No memories provided for analysis
    EmptyMemories,
    /// Goal has invalid embeddings (NaN, Inf, or missing)
    InvalidGoal { reason: String },
    /// Comparison failed for a specific embedder
    ComparisonFailed { embedder: Embedder, reason: String },
    /// Invalid threshold configuration
    InvalidThresholds { reason: String },
    /// Comparison validation error
    ComparisonValidationFailed { reason: String },
}

impl std::fmt::Display for DriftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriftError::EmptyMemories => write!(f, "No memories provided for drift analysis"),
            DriftError::InvalidGoal { reason } => write!(f, "Invalid goal: {}", reason),
            DriftError::ComparisonFailed { embedder, reason } => {
                write!(f, "Comparison failed for {:?}: {}", embedder, reason)
            }
            DriftError::InvalidThresholds { reason } => {
                write!(f, "Invalid thresholds: {}", reason)
            }
            DriftError::ComparisonValidationFailed { reason } => {
                write!(f, "Comparison validation failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for DriftError {}

// ============================================
// RESULT TYPES
// ============================================

/// Result of drift analysis.
#[derive(Debug)]
pub struct DriftResult {
    /// Overall drift assessment
    pub overall_drift: OverallDrift,
    /// Per-embedder drift breakdown for all 13 embedders
    pub per_embedder_drift: PerEmbedderDrift,
    /// Most drifted embedders, sorted worst first (max 5)
    pub most_drifted_embedders: Vec<EmbedderDriftInfo>,
    /// Recommendations for addressing drift (only for Medium+ drift)
    pub recommendations: Vec<DriftRecommendation>,
    /// Trend analysis if history available
    pub trend: Option<TrendAnalysis>,
    /// Number of memories analyzed
    pub analyzed_count: usize,
    /// Timestamp of analysis
    pub timestamp: DateTime<Utc>,
}

/// Overall drift assessment.
#[derive(Debug)]
pub struct OverallDrift {
    /// Whether any drift was detected (drift_level > None)
    pub has_drifted: bool,
    /// Drift score: 1.0 - similarity (0.0 = no drift, 1.0 = total drift)
    pub drift_score: f32,
    /// Classified drift severity
    pub drift_level: DriftLevel,
    /// Raw similarity score (0.0 to 1.0)
    pub similarity: f32,
}

/// Per-embedder drift breakdown for all 13 embedders.
#[derive(Debug)]
pub struct PerEmbedderDrift {
    /// Drift info for each embedder, indexed by Embedder::index()
    pub embedder_drift: [EmbedderDriftInfo; NUM_EMBEDDERS],
}

/// Drift info for a single embedder.
#[derive(Debug, Clone)]
pub struct EmbedderDriftInfo {
    /// The embedder this info is for
    pub embedder: Embedder,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Classified drift level
    pub drift_level: DriftLevel,
    /// Drift score: 1.0 - similarity
    pub drift_score: f32,
}

impl EmbedderDriftInfo {
    /// Create a new embedder drift info.
    fn new(embedder: Embedder, similarity: f32, thresholds: &DriftThresholds) -> Self {
        let clamped = similarity.clamp(0.0, 1.0);
        Self {
            embedder,
            similarity: clamped,
            drift_level: DriftLevel::from_similarity(clamped, thresholds),
            drift_score: 1.0 - clamped,
        }
    }
}

// ============================================
// RECOMMENDATION TYPES
// ============================================

/// Recommendation for addressing drift in a specific embedder.
#[derive(Debug)]
pub struct DriftRecommendation {
    /// The embedder with drift
    pub embedder: Embedder,
    /// Description of the issue
    pub issue: String,
    /// Suggested action
    pub suggestion: String,
    /// Priority based on drift severity
    pub priority: RecommendationPriority,
}

/// Priority levels for recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Low priority (shouldn't occur in recommendations)
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

impl From<DriftLevel> for RecommendationPriority {
    fn from(level: DriftLevel) -> Self {
        match level {
            DriftLevel::Critical => RecommendationPriority::Critical,
            DriftLevel::High => RecommendationPriority::High,
            DriftLevel::Medium => RecommendationPriority::Medium,
            DriftLevel::Low => RecommendationPriority::Low,
            DriftLevel::None => RecommendationPriority::Low,
        }
    }
}

// ============================================
// TREND ANALYSIS
// ============================================

/// Trend analysis over time using linear regression.
#[derive(Debug)]
pub struct TrendAnalysis {
    /// Direction of the trend
    pub direction: DriftTrend,
    /// Velocity: absolute value of slope
    pub velocity: f32,
    /// Number of history samples used
    pub samples: usize,
    /// Projected time until critical drift (if worsening)
    pub projected_critical_in: Option<String>,
}

// ============================================
// HISTORY TYPES
// ============================================

/// History of drift measurements for trend analysis.
#[derive(Debug, Default)]
pub struct DriftHistory {
    /// Entries keyed by goal_id
    entries: HashMap<String, Vec<DriftHistoryEntry>>,
    /// Maximum entries to keep per goal
    max_entries_per_goal: usize,
}

impl DriftHistory {
    /// Create a new history with specified max entries per goal.
    pub fn new(max_entries_per_goal: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries_per_goal,
        }
    }

    /// Add an entry for a goal.
    pub fn add(&mut self, goal_id: &str, entry: DriftHistoryEntry) {
        let entries = self.entries.entry(goal_id.to_string()).or_default();
        entries.push(entry);

        // Trim to max entries
        while entries.len() > self.max_entries_per_goal {
            entries.remove(0);
        }
    }

    /// Get entries for a goal.
    pub fn get(&self, goal_id: &str) -> Option<&Vec<DriftHistoryEntry>> {
        self.entries.get(goal_id)
    }

    /// Clear entries for a goal.
    pub fn clear(&mut self, goal_id: &str) {
        self.entries.remove(goal_id);
    }

    /// Get entry count for a goal.
    pub fn len(&self, goal_id: &str) -> usize {
        self.entries.get(goal_id).map(|e| e.len()).unwrap_or(0)
    }
}

/// Single history entry with per-embedder breakdown.
#[derive(Debug, Clone)]
pub struct DriftHistoryEntry {
    /// Timestamp of this measurement
    pub timestamp: DateTime<Utc>,
    /// Overall similarity score
    pub overall_similarity: f32,
    /// Per-embedder similarity scores
    pub per_embedder: [f32; NUM_EMBEDDERS],
    /// Number of memories analyzed
    pub memories_analyzed: usize,
}

// ============================================
// TELEOLOGICAL DRIFT DETECTOR
// ============================================

/// Teleological drift detector using per-embedder array comparison.
///
/// Uses TeleologicalComparator from TASK-LOGIC-004 for apples-to-apples
/// comparison across all 13 embedders.
#[derive(Debug)]
pub struct TeleologicalDriftDetector {
    /// The comparator for fingerprint comparison (reserved for future per-embedder comparisons)
    #[allow(dead_code)]
    comparator: TeleologicalComparator,
    /// History for trend analysis
    history: DriftHistory,
    /// Thresholds for drift classification
    thresholds: DriftThresholds,
}

impl TeleologicalDriftDetector {
    /// Create a new detector with default thresholds.
    pub fn new(comparator: TeleologicalComparator) -> Self {
        Self {
            comparator,
            history: DriftHistory::new(100),
            thresholds: DriftThresholds::default(),
        }
    }

    /// Create a detector with custom thresholds.
    ///
    /// # Errors
    ///
    /// Returns error if thresholds are invalid.
    pub fn with_thresholds(
        comparator: TeleologicalComparator,
        thresholds: DriftThresholds,
    ) -> Result<Self, DriftError> {
        thresholds.validate()?;
        Ok(Self {
            comparator,
            history: DriftHistory::new(100),
            thresholds,
        })
    }

    /// Check drift of memories against a goal (stateless).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `memories` is empty (FAIL FAST)
    /// - `goal` has invalid embeddings (NaN, Inf)
    /// - Comparison fails
    pub fn check_drift(
        &self,
        memories: &[SemanticFingerprint],
        goal: &SemanticFingerprint,
        strategy: SearchStrategy,
    ) -> Result<DriftResult, DriftError> {
        // FAIL FAST: Empty memories
        if memories.is_empty() {
            return Err(DriftError::EmptyMemories);
        }

        // FAIL FAST: Validate goal
        self.validate_fingerprint(goal)?;

        // Compare all memories against the goal and aggregate
        let per_embedder_sims = self.compute_per_embedder_similarities(memories, goal, strategy)?;

        // Build result
        self.build_result(per_embedder_sims, memories.len(), None)
    }

    /// Check drift and update history for trend analysis (stateful).
    ///
    /// # Errors
    ///
    /// Same as `check_drift`.
    pub fn check_drift_with_history(
        &mut self,
        memories: &[SemanticFingerprint],
        goal: &SemanticFingerprint,
        goal_id: &str,
        strategy: SearchStrategy,
    ) -> Result<DriftResult, DriftError> {
        // FAIL FAST: Empty memories
        if memories.is_empty() {
            return Err(DriftError::EmptyMemories);
        }

        // FAIL FAST: Validate goal
        self.validate_fingerprint(goal)?;

        // Compare all memories against the goal and aggregate
        let per_embedder_sims = self.compute_per_embedder_similarities(memories, goal, strategy)?;

        // Record history
        let overall_sim = per_embedder_sims.iter().sum::<f32>() / NUM_EMBEDDERS as f32;
        let entry = DriftHistoryEntry {
            timestamp: Utc::now(),
            overall_similarity: overall_sim,
            per_embedder: per_embedder_sims,
            memories_analyzed: memories.len(),
        };
        self.history.add(goal_id, entry);

        // Compute trend
        let trend = self.compute_trend(goal_id);

        // Build result
        self.build_result(per_embedder_sims, memories.len(), trend)
    }

    /// Get drift trend for a goal from history.
    ///
    /// Returns None if fewer than 3 history samples.
    pub fn get_trend(&self, goal_id: &str) -> Option<TrendAnalysis> {
        self.compute_trend(goal_id)
    }

    /// Validate that a fingerprint has valid embeddings (no NaN/Inf).
    fn validate_fingerprint(&self, fp: &SemanticFingerprint) -> Result<(), DriftError> {
        for embedder in Embedder::all() {
            if let Some(slice) = fp.get_embedding(embedder.index()) {
                match slice {
                    crate::types::EmbeddingSlice::Dense(values) => {
                        for (i, &v) in values.iter().enumerate() {
                            if v.is_nan() {
                                return Err(DriftError::InvalidGoal {
                                    reason: format!(
                                        "NaN at index {} in {:?}",
                                        i,
                                        embedder
                                    ),
                                });
                            }
                            if v.is_infinite() {
                                return Err(DriftError::InvalidGoal {
                                    reason: format!(
                                        "Infinity at index {} in {:?}",
                                        i,
                                        embedder
                                    ),
                                });
                            }
                        }
                    }
                    crate::types::EmbeddingSlice::Sparse(sv) => {
                        for &v in &sv.values {
                            if v.is_nan() || v.is_infinite() {
                                return Err(DriftError::InvalidGoal {
                                    reason: format!(
                                        "Invalid value in sparse vector for {:?}",
                                        embedder
                                    ),
                                });
                            }
                        }
                    }
                    crate::types::EmbeddingSlice::TokenLevel(tokens) => {
                        for (t_idx, token) in tokens.iter().enumerate() {
                            for (i, &v) in token.iter().enumerate() {
                                if v.is_nan() || v.is_infinite() {
                                    return Err(DriftError::InvalidGoal {
                                        reason: format!(
                                            "Invalid value at token {} index {} in {:?}",
                                            t_idx, i, embedder
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute per-embedder similarities by averaging across all memories.
    fn compute_per_embedder_similarities(
        &self,
        memories: &[SemanticFingerprint],
        goal: &SemanticFingerprint,
        strategy: SearchStrategy,
    ) -> Result<[f32; NUM_EMBEDDERS], DriftError> {
        // Accumulate per-embedder scores
        let mut sums = [0.0f32; NUM_EMBEDDERS];
        let mut counts = [0usize; NUM_EMBEDDERS];

        for memory in memories {
            // Use comparator with the specified strategy
            let config = MatrixSearchConfig {
                strategy,
                ..MatrixSearchConfig::default()
            };
            let comparator = TeleologicalComparator::with_config(config);

            let result = comparator.compare(memory, goal).map_err(|e| {
                DriftError::ComparisonValidationFailed {
                    reason: format!("{:?}", e),
                }
            })?;

            // Aggregate per-embedder scores
            for (idx, score) in result.per_embedder.iter().enumerate() {
                if let Some(s) = score {
                    sums[idx] += s;
                    counts[idx] += 1;
                }
            }
        }

        // Compute averages
        let mut averages = [0.0f32; NUM_EMBEDDERS];
        for idx in 0..NUM_EMBEDDERS {
            if counts[idx] > 0 {
                averages[idx] = sums[idx] / counts[idx] as f32;
            }
        }

        Ok(averages)
    }

    /// Build the drift result from per-embedder similarities.
    fn build_result(
        &self,
        per_embedder_sims: [f32; NUM_EMBEDDERS],
        analyzed_count: usize,
        trend: Option<TrendAnalysis>,
    ) -> Result<DriftResult, DriftError> {
        // Build per-embedder drift info
        let embedder_drift: [EmbedderDriftInfo; NUM_EMBEDDERS] = std::array::from_fn(|idx| {
            let embedder = Embedder::from_index(idx).expect("valid index");
            EmbedderDriftInfo::new(embedder, per_embedder_sims[idx], &self.thresholds)
        });

        // Compute overall similarity (average)
        let overall_similarity = per_embedder_sims.iter().sum::<f32>() / NUM_EMBEDDERS as f32;
        let overall_drift_level = DriftLevel::from_similarity(overall_similarity, &self.thresholds);

        // Find most drifted embedders (sorted worst first)
        let mut sorted: Vec<EmbedderDriftInfo> = embedder_drift.to_vec();
        sorted.sort_by(|a, b| a.drift_level.cmp(&b.drift_level));
        let most_drifted: Vec<EmbedderDriftInfo> = sorted
            .into_iter()
            .filter(|info| info.drift_level.has_drifted())
            .take(MAX_MOST_DRIFTED)
            .collect();

        // Generate recommendations for Medium+ drift
        let recommendations = self.generate_recommendations(&embedder_drift);

        Ok(DriftResult {
            overall_drift: OverallDrift {
                has_drifted: overall_drift_level.has_drifted(),
                drift_score: 1.0 - overall_similarity,
                drift_level: overall_drift_level,
                similarity: overall_similarity,
            },
            per_embedder_drift: PerEmbedderDrift { embedder_drift },
            most_drifted_embedders: most_drifted,
            recommendations,
            trend,
            analyzed_count,
            timestamp: Utc::now(),
        })
    }

    /// Generate recommendations based on drift analysis.
    fn generate_recommendations(
        &self,
        embedder_drift: &[EmbedderDriftInfo; NUM_EMBEDDERS],
    ) -> Vec<DriftRecommendation> {
        let mut recommendations = Vec::new();

        for info in embedder_drift {
            // Only generate recommendations for Medium or worse drift
            if !info.drift_level.needs_recommendation() {
                continue;
            }

            let (issue, suggestion) = self.get_embedder_recommendation(info);

            recommendations.push(DriftRecommendation {
                embedder: info.embedder,
                issue,
                suggestion,
                priority: RecommendationPriority::from(info.drift_level),
            });
        }

        // Sort by priority (Critical first)
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        recommendations
    }

    /// Get embedder-specific recommendation text.
    fn get_embedder_recommendation(&self, info: &EmbedderDriftInfo) -> (String, String) {
        let severity = match info.drift_level {
            DriftLevel::Critical => "critical",
            DriftLevel::High => "high",
            DriftLevel::Medium => "moderate",
            _ => "minor",
        };

        let (issue, suggestion) = match info.embedder {
            Embedder::Semantic => (
                format!("Semantic meaning drift at {} level (sim: {:.2})", severity, info.similarity),
                "Review core semantic content alignment with goals".to_string(),
            ),
            Embedder::TemporalRecent => (
                format!("Recent temporal context drift at {} level (sim: {:.2})", severity, info.similarity),
                "Ensure recent memories are being captured appropriately".to_string(),
            ),
            Embedder::TemporalPeriodic => (
                format!("Periodic pattern drift at {} level (sim: {:.2})", severity, info.similarity),
                "Check cyclical patterns are being maintained".to_string(),
            ),
            Embedder::TemporalPositional => (
                format!("Positional temporal drift at {} level (sim: {:.2})", severity, info.similarity),
                "Review sequence ordering and positional context".to_string(),
            ),
            Embedder::Causal => (
                format!("Causal reasoning drift at {} level (sim: {:.2})", severity, info.similarity),
                "Strengthen cause-effect relationship tracking".to_string(),
            ),
            Embedder::Sparse => (
                format!("Lexical/keyword drift at {} level (sim: {:.2})", severity, info.similarity),
                "Review keyword relevance and lexical alignment".to_string(),
            ),
            Embedder::Code => (
                format!("Code structure drift at {} level (sim: {:.2})", severity, info.similarity),
                "Ensure code-related memories align with technical goals".to_string(),
            ),
            Embedder::Graph => (
                format!("Relational graph drift at {} level (sim: {:.2})", severity, info.similarity),
                "Review entity relationship structures".to_string(),
            ),
            Embedder::Hdc => (
                format!("Hyperdimensional pattern drift at {} level (sim: {:.2})", severity, info.similarity),
                "Check holographic pattern consistency".to_string(),
            ),
            Embedder::Multimodal => (
                format!("Multimodal drift at {} level (sim: {:.2})", severity, info.similarity),
                "Review cross-modal content alignment".to_string(),
            ),
            Embedder::Entity => (
                format!("Entity recognition drift at {} level (sim: {:.2})", severity, info.similarity),
                "Ensure named entities are consistently identified".to_string(),
            ),
            Embedder::LateInteraction => (
                format!("Token-level precision drift at {} level (sim: {:.2})", severity, info.similarity),
                "Review fine-grained token matching patterns".to_string(),
            ),
            Embedder::KeywordSplade => (
                format!("Keyword expansion drift at {} level (sim: {:.2})", severity, info.similarity),
                "Check learned keyword expansion coverage".to_string(),
            ),
        };

        (issue, suggestion)
    }

    /// Compute trend from history using linear regression.
    fn compute_trend(&self, goal_id: &str) -> Option<TrendAnalysis> {
        let entries = self.history.get(goal_id)?;

        if entries.len() < MIN_TREND_SAMPLES {
            return None;
        }

        // Simple linear regression on overall similarity
        let n = entries.len() as f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_xy = 0.0f32;
        let mut sum_xx = 0.0f32;

        for (i, entry) in entries.iter().enumerate() {
            let x = i as f32;
            let y = entry.overall_similarity;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        // slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return Some(TrendAnalysis {
                direction: DriftTrend::Stable,
                velocity: 0.0,
                samples: entries.len(),
                projected_critical_in: None,
            });
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        let direction = if slope.abs() < 0.01 {
            DriftTrend::Stable
        } else if slope > 0.0 {
            DriftTrend::Improving
        } else {
            DriftTrend::Worsening
        };

        // Project time to critical (similarity < 0.40) if worsening
        let projected_critical_in = if direction == DriftTrend::Worsening {
            let current_sim = entries.last()?.overall_similarity;
            let critical_threshold = self.thresholds.high_min; // 0.40

            if current_sim > critical_threshold && slope < 0.0 {
                // Steps until critical = (current_sim - critical_threshold) / |slope|
                let steps = (current_sim - critical_threshold) / slope.abs();
                // Assuming ~1 measurement per check, convert to human-readable
                Some(format!("{:.1} checks at current rate", steps))
            } else {
                None
            }
        } else {
            None
        };

        Some(TrendAnalysis {
            direction,
            velocity: slope.abs(),
            samples: entries.len(),
            projected_critical_in,
        })
    }
}

// ============================================
// TESTS
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::fingerprint::SparseVector;

    // ============================================
    // TEST FIXTURE HELPERS
    // ============================================

    /// Create a test goal with aligned values.
    fn create_test_goal() -> SemanticFingerprint {
        create_fingerprint_with_value(1.0)
    }

    /// Create a test memory with high similarity to the goal.
    fn create_test_memory() -> SemanticFingerprint {
        create_fingerprint_with_value(1.0)
    }

    /// Create a memory with specified drift (lower similarity).
    fn create_drifted_memory(base_similarity: f32) -> SemanticFingerprint {
        // Create a fingerprint that will produce approximately base_similarity
        // when compared to the test goal
        create_fingerprint_with_value(base_similarity)
    }

    /// Create a heavily drifted memory.
    fn create_heavily_drifted_memory() -> SemanticFingerprint {
        create_drifted_memory(0.3)
    }

    /// Create a goal with NaN values.
    fn create_goal_with_nan() -> SemanticFingerprint {
        let mut fp = create_test_goal();
        if !fp.e1_semantic.is_empty() {
            fp.e1_semantic[0] = f32::NAN;
        }
        fp
    }

    /// Create a goal with Inf values.
    fn create_goal_with_inf() -> SemanticFingerprint {
        let mut fp = create_test_goal();
        if !fp.e1_semantic.is_empty() {
            fp.e1_semantic[0] = f32::INFINITY;
        }
        fp
    }

    /// Create a normalized fingerprint with given base value.
    fn create_fingerprint_with_value(val: f32) -> SemanticFingerprint {
        use crate::types::fingerprint::{
            E10_DIM, E11_DIM, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM, E7_DIM, E8_DIM, E9_DIM,
        };

        let create_normalized_vec = |dim: usize, v: f32| -> Vec<f32> {
            let mut vec = vec![v; dim];
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            vec
        };

        SemanticFingerprint {
            e1_semantic: create_normalized_vec(E1_DIM, val),
            e2_temporal_recent: create_normalized_vec(E2_DIM, val),
            e3_temporal_periodic: create_normalized_vec(E3_DIM, val),
            e4_temporal_positional: create_normalized_vec(E4_DIM, val),
            e5_causal: create_normalized_vec(E5_DIM, val),
            e6_sparse: SparseVector::empty(),
            e7_code: create_normalized_vec(E7_DIM, val),
            e8_graph: create_normalized_vec(E8_DIM, val),
            e9_hdc: create_normalized_vec(E9_DIM, val),
            e10_multimodal: create_normalized_vec(E10_DIM, val),
            e11_entity: create_normalized_vec(E11_DIM, val),
            e12_late_interaction: vec![vec![val / 128.0_f32.sqrt(); 128]],
            e13_splade: SparseVector::empty(),
        }
    }

    // ============================================
    // DRIFT LEVEL CLASSIFICATION TESTS
    // ============================================

    #[test]
    fn test_drift_level_from_similarity_none() {
        let thresholds = DriftThresholds::default();
        assert_eq!(
            DriftLevel::from_similarity(0.90, &thresholds),
            DriftLevel::None
        );
        assert_eq!(
            DriftLevel::from_similarity(0.85, &thresholds),
            DriftLevel::None
        );
    }

    #[test]
    fn test_drift_level_from_similarity_low() {
        let thresholds = DriftThresholds::default();
        assert_eq!(
            DriftLevel::from_similarity(0.84, &thresholds),
            DriftLevel::Low
        );
        assert_eq!(
            DriftLevel::from_similarity(0.70, &thresholds),
            DriftLevel::Low
        );
    }

    #[test]
    fn test_drift_level_from_similarity_medium() {
        let thresholds = DriftThresholds::default();
        assert_eq!(
            DriftLevel::from_similarity(0.69, &thresholds),
            DriftLevel::Medium
        );
        assert_eq!(
            DriftLevel::from_similarity(0.55, &thresholds),
            DriftLevel::Medium
        );
    }

    #[test]
    fn test_drift_level_from_similarity_high() {
        let thresholds = DriftThresholds::default();
        assert_eq!(
            DriftLevel::from_similarity(0.54, &thresholds),
            DriftLevel::High
        );
        assert_eq!(
            DriftLevel::from_similarity(0.40, &thresholds),
            DriftLevel::High
        );
    }

    #[test]
    fn test_drift_level_from_similarity_critical() {
        let thresholds = DriftThresholds::default();
        assert_eq!(
            DriftLevel::from_similarity(0.39, &thresholds),
            DriftLevel::Critical
        );
        assert_eq!(
            DriftLevel::from_similarity(0.0, &thresholds),
            DriftLevel::Critical
        );
    }

    #[test]
    fn test_drift_level_ordering() {
        // Critical < High < Medium < Low < None (for sorting worst first)
        assert!(DriftLevel::Critical < DriftLevel::High);
        assert!(DriftLevel::High < DriftLevel::Medium);
        assert!(DriftLevel::Medium < DriftLevel::Low);
        assert!(DriftLevel::Low < DriftLevel::None);
    }

    // ============================================
    // FAIL FAST TESTS
    // ============================================

    #[test]
    fn test_fail_fast_empty_memories() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();

        let result = detector.check_drift(&[], &goal, SearchStrategy::Cosine);

        assert!(matches!(result, Err(DriftError::EmptyMemories)));
    }

    #[test]
    fn test_fail_fast_invalid_goal_nan() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_goal_with_nan();
        let memories = vec![create_test_memory()];

        let result = detector.check_drift(&memories, &goal, SearchStrategy::Cosine);

        assert!(matches!(result, Err(DriftError::InvalidGoal { .. })));
    }

    #[test]
    fn test_fail_fast_invalid_goal_inf() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_goal_with_inf();
        let memories = vec![create_test_memory()];

        let result = detector.check_drift(&memories, &goal, SearchStrategy::Cosine);

        assert!(matches!(result, Err(DriftError::InvalidGoal { .. })));
    }

    // ============================================
    // PER-EMBEDDER ANALYSIS TESTS
    // ============================================

    #[test]
    fn test_per_embedder_breakdown_all_13() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        // Must have exactly 13 embedder entries
        assert_eq!(result.per_embedder_drift.embedder_drift.len(), 13);

        // Each embedder must be present
        for embedder in Embedder::all() {
            let found = result
                .per_embedder_drift
                .embedder_drift
                .iter()
                .any(|e| e.embedder == embedder);
            assert!(found, "Missing embedder: {:?}", embedder);
        }
    }

    #[test]
    fn test_per_embedder_similarity_valid_range() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        for info in &result.per_embedder_drift.embedder_drift {
            assert!(
                info.similarity >= 0.0,
                "Similarity < 0 for {:?}",
                info.embedder
            );
            assert!(
                info.similarity <= 1.0,
                "Similarity > 1 for {:?}",
                info.embedder
            );
            assert!(
                !info.similarity.is_nan(),
                "NaN similarity for {:?}",
                info.embedder
            );
        }
    }

    #[test]
    fn test_drift_score_equals_one_minus_similarity() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        let expected_drift = 1.0 - result.overall_drift.similarity;
        assert!(
            (result.overall_drift.drift_score - expected_drift).abs() < 0.0001,
            "Drift score {} != 1 - similarity {}",
            result.overall_drift.drift_score,
            expected_drift
        );
    }

    // ============================================
    // MOST DRIFTED EMBEDDERS TESTS
    // ============================================

    #[test]
    fn test_most_drifted_sorted_worst_first() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_drifted_memory(0.6)];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        // Verify descending order by drift level (worst first)
        for window in result.most_drifted_embedders.windows(2) {
            assert!(
                window[0].drift_level <= window[1].drift_level,
                "Not sorted: {:?} should come before {:?}",
                window[0].drift_level,
                window[1].drift_level
            );
        }
    }

    #[test]
    fn test_most_drifted_max_five() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_heavily_drifted_memory()];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        assert!(
            result.most_drifted_embedders.len() <= 5,
            "Should return at most 5 drifted embedders"
        );
    }

    // ============================================
    // TREND ANALYSIS TESTS
    // ============================================

    #[test]
    fn test_trend_requires_minimum_samples() {
        let comparator = TeleologicalComparator::new();
        let mut detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        // Add only 2 samples
        for _ in 0..2 {
            let _ = detector.check_drift_with_history(
                &memories,
                &goal,
                "goal-1",
                SearchStrategy::Cosine,
            );
        }

        let trend = detector.get_trend("goal-1");
        assert!(trend.is_none(), "Trend should require >= 3 samples");
    }

    #[test]
    fn test_trend_available_with_enough_samples() {
        let comparator = TeleologicalComparator::new();
        let mut detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        // Add 5 samples
        for _ in 0..5 {
            let _ = detector.check_drift_with_history(
                &memories,
                &goal,
                "goal-1",
                SearchStrategy::Cosine,
            );
        }

        let trend = detector.get_trend("goal-1");
        assert!(trend.is_some(), "Trend should be available with 5 samples");
        assert_eq!(trend.unwrap().samples, 5);
    }

    #[test]
    fn test_trend_direction_stable_for_identical() {
        let comparator = TeleologicalComparator::new();
        let mut detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        // Add identical samples (should be stable)
        for _ in 0..5 {
            let _ = detector.check_drift_with_history(
                &memories,
                &goal,
                "goal-1",
                SearchStrategy::Cosine,
            );
        }

        let trend = detector.get_trend("goal-1").unwrap();
        assert_eq!(
            trend.direction,
            DriftTrend::Stable,
            "Identical samples should show stable trend"
        );
    }

    // ============================================
    // RECOMMENDATIONS TESTS
    // ============================================

    #[test]
    fn test_recommendations_only_for_medium_plus() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        for rec in &result.recommendations {
            // Find the corresponding embedder info
            let info = result
                .per_embedder_drift
                .embedder_drift
                .iter()
                .find(|e| e.embedder == rec.embedder)
                .unwrap();

            assert!(
                info.drift_level.needs_recommendation(),
                "Recommendation for {:?} with level {:?} (should be Medium or worse)",
                rec.embedder,
                info.drift_level
            );
        }
    }

    #[test]
    fn test_recommendations_priority_matches_drift_level() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_heavily_drifted_memory()];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        for rec in &result.recommendations {
            let info = result
                .per_embedder_drift
                .embedder_drift
                .iter()
                .find(|e| e.embedder == rec.embedder)
                .unwrap();

            let expected_priority = RecommendationPriority::from(info.drift_level);
            assert_eq!(
                rec.priority, expected_priority,
                "Priority mismatch for {:?}",
                rec.embedder
            );
        }
    }

    // ============================================
    // HISTORY TESTS
    // ============================================

    #[test]
    fn test_history_per_goal_isolation() {
        let comparator = TeleologicalComparator::new();
        let mut detector = TeleologicalDriftDetector::new(comparator);
        let goal1 = create_test_goal();
        let goal2 = create_test_goal();
        let memories = vec![create_test_memory()];

        // Add to goal-1
        for _ in 0..3 {
            let _ = detector.check_drift_with_history(
                &memories,
                &goal1,
                "goal-1",
                SearchStrategy::Cosine,
            );
        }

        // Add to goal-2
        for _ in 0..5 {
            let _ = detector.check_drift_with_history(
                &memories,
                &goal2,
                "goal-2",
                SearchStrategy::Cosine,
            );
        }

        let trend1 = detector.get_trend("goal-1");
        let trend2 = detector.get_trend("goal-2");

        assert_eq!(trend1.unwrap().samples, 3);
        assert_eq!(trend2.unwrap().samples, 5);
    }

    #[test]
    fn test_history_entry_has_per_embedder_array() {
        let mut history = DriftHistory::new(100);

        let entry = DriftHistoryEntry {
            timestamp: Utc::now(),
            overall_similarity: 0.75,
            per_embedder: [
                0.8, 0.7, 0.6, 0.5, 0.9, 0.85, 0.75, 0.65, 0.55, 0.45, 0.95, 0.88, 0.72,
            ],
            memories_analyzed: 10,
        };

        history.add("test-goal", entry);

        let entries = history.get("test-goal").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].per_embedder.len(), 13);
    }

    // ============================================
    // CUSTOM THRESHOLDS TESTS
    // ============================================

    #[test]
    fn test_custom_thresholds() {
        let custom = DriftThresholds {
            none_min: 0.90,
            low_min: 0.80,
            medium_min: 0.70,
            high_min: 0.60,
        };

        // With custom thresholds, 0.85 is Low (not None)
        assert_eq!(DriftLevel::from_similarity(0.85, &custom), DriftLevel::Low);

        // With default, 0.85 is None
        assert_eq!(
            DriftLevel::from_similarity(0.85, &DriftThresholds::default()),
            DriftLevel::None
        );
    }

    #[test]
    fn test_invalid_thresholds_rejected() {
        let invalid = DriftThresholds {
            none_min: 0.70, // Less than low_min!
            low_min: 0.80,
            medium_min: 0.55,
            high_min: 0.40,
        };

        assert!(invalid.validate().is_err());
    }

    // ============================================
    // SINGLE MEMORY TEST
    // ============================================

    #[test]
    fn test_single_memory_analysis() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();

        assert_eq!(result.analyzed_count, 1);
    }

    // ============================================
    // TIMESTAMP TESTS
    // ============================================

    #[test]
    fn test_result_has_recent_timestamp() {
        let comparator = TeleologicalComparator::new();
        let detector = TeleologicalDriftDetector::new(comparator);
        let goal = create_test_goal();
        let memories = vec![create_test_memory()];

        let before = Utc::now();
        let result = detector
            .check_drift(&memories, &goal, SearchStrategy::Cosine)
            .unwrap();
        let after = Utc::now();

        assert!(result.timestamp >= before);
        assert!(result.timestamp <= after);
    }
}
