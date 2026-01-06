//! Default implementation of JohariTransitionManager using TeleologicalMemoryStore.
//!
//! This module provides `DefaultJohariManager`, a concrete implementation that:
//! - Uses TeleologicalMemoryStore for persistence
//! - Validates all transitions via JohariQuadrant state machine
//! - Implements blind spot discovery algorithm
//! - Supports batch operations with all-or-nothing semantics

use std::sync::Arc;

use async_trait::async_trait;

use crate::traits::{TeleologicalMemoryStore, TeleologicalSearchOptions};
use crate::types::fingerprint::{JohariFingerprint, SemanticFingerprint, NUM_EMBEDDERS};
use crate::types::{JohariQuadrant, JohariTransition, TransitionTrigger};

use super::error::JohariError;
use super::external_signal::{BlindSpotCandidate, ExternalSignal};
use super::manager::{
    ClassificationContext, JohariTransitionManager, MemoryId, QuadrantPattern, TimeRange,
};
use super::stats::TransitionStats;

/// Default implementation using TeleologicalMemoryStore.
///
/// This implementation:
/// - Stores transitions implicitly in JohariFingerprint state
/// - Uses existing classification methods from JohariFingerprint
/// - Validates transitions using JohariQuadrant::can_transition_to()
pub struct DefaultJohariManager<S: TeleologicalMemoryStore> {
    /// The storage backend for teleological fingerprints
    store: Arc<S>,

    /// Threshold for blind spot detection (default: 0.5)
    blind_spot_threshold: f32,

    /// Maximum transition history per memory (default: 100)
    #[allow(dead_code)]
    max_history_per_memory: usize,
}

impl<S: TeleologicalMemoryStore> DefaultJohariManager<S> {
    /// Create a new DefaultJohariManager with the given store.
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            blind_spot_threshold: 0.5,
            max_history_per_memory: 100,
        }
    }

    /// Set the blind spot detection threshold.
    ///
    /// Signal strengths above this threshold will be considered blind spot candidates.
    pub fn with_blind_spot_threshold(mut self, threshold: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be [0,1], got {}",
            threshold
        );
        self.blind_spot_threshold = threshold;
        self
    }
}

/// Set quadrant weights based on the quadrant.
///
/// Sets 100% weight to the specified quadrant (hard classification).
fn set_quadrant_weights(
    johari: &mut JohariFingerprint,
    embedder_idx: usize,
    quadrant: JohariQuadrant,
) {
    match quadrant {
        JohariQuadrant::Open => johari.set_quadrant(embedder_idx, 1.0, 0.0, 0.0, 0.0, 1.0),
        JohariQuadrant::Hidden => johari.set_quadrant(embedder_idx, 0.0, 1.0, 0.0, 0.0, 1.0),
        JohariQuadrant::Blind => johari.set_quadrant(embedder_idx, 0.0, 0.0, 1.0, 0.0, 1.0),
        JohariQuadrant::Unknown => johari.set_quadrant(embedder_idx, 0.0, 0.0, 0.0, 1.0, 1.0),
    }
}

/// Dynamic (type-erased) version of DefaultJohariManager.
///
/// Use this when working with `Arc<dyn TeleologicalMemoryStore>` trait objects,
/// such as in the MCP Handlers struct.
///
/// TASK-S004: Required for johari/* handlers in context-graph-mcp.
pub struct DynDefaultJohariManager {
    /// The storage backend for teleological fingerprints (trait object)
    store: Arc<dyn TeleologicalMemoryStore>,

    /// Threshold for blind spot detection (default: 0.5)
    blind_spot_threshold: f32,

    /// Maximum transition history per memory (default: 100)
    #[allow(dead_code)]
    max_history_per_memory: usize,
}

impl DynDefaultJohariManager {
    /// Create a new DynDefaultJohariManager with a trait object store.
    pub fn new(store: Arc<dyn TeleologicalMemoryStore>) -> Self {
        Self {
            store,
            blind_spot_threshold: 0.5,
            max_history_per_memory: 100,
        }
    }

    /// Set the blind spot detection threshold.
    pub fn with_blind_spot_threshold(mut self, threshold: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be [0,1], got {}",
            threshold
        );
        self.blind_spot_threshold = threshold;
        self
    }
}

#[async_trait]
impl<S: TeleologicalMemoryStore + 'static> JohariTransitionManager for DefaultJohariManager<S> {
    async fn classify(
        &self,
        _semantic: &SemanticFingerprint,
        context: &ClassificationContext,
    ) -> Result<JohariFingerprint, JohariError> {
        let mut fingerprint = JohariFingerprint::zeroed();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let delta_s = context.delta_s[embedder_idx];
            let delta_c = context.delta_c[embedder_idx];

            // Use existing classification logic from JohariFingerprint
            let quadrant = JohariFingerprint::classify_quadrant(delta_s, delta_c);

            // Apply disclosure intent override: if hidden intent, force Hidden
            let final_quadrant = if !context.disclosure_intent[embedder_idx]
                && quadrant == JohariQuadrant::Open
            {
                JohariQuadrant::Hidden
            } else {
                quadrant
            };

            // Set hard classification (100% weight to one quadrant)
            set_quadrant_weights(&mut fingerprint, embedder_idx, final_quadrant);
        }

        Ok(fingerprint)
    }

    async fn transition(
        &self,
        memory_id: MemoryId,
        embedder_idx: usize,
        to_quadrant: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Result<JohariFingerprint, JohariError> {
        // Validate embedder index (FAIL FAST)
        if embedder_idx >= NUM_EMBEDDERS {
            return Err(JohariError::InvalidEmbedderIndex(embedder_idx));
        }

        // Retrieve current state
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let mut johari = current.johari.clone();
        let current_quadrant = johari.dominant_quadrant(embedder_idx);

        // Validate transition using existing state machine
        if !current_quadrant.can_transition_to(to_quadrant) {
            return Err(JohariError::InvalidTransition {
                from: current_quadrant,
                to: to_quadrant,
                embedder_idx,
            });
        }

        // Validate trigger is valid for this transition
        let valid_transition = current_quadrant.transition_to(to_quadrant, trigger);
        if valid_transition.is_err() {
            return Err(JohariError::InvalidTrigger {
                from: current_quadrant,
                to: to_quadrant,
                trigger,
            });
        }

        // Apply transition (set 100% weight to new quadrant)
        set_quadrant_weights(&mut johari, embedder_idx, to_quadrant);

        // Update stored fingerprint
        let mut updated = current;
        updated.johari = johari.clone();

        self.store
            .update(updated)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        Ok(johari)
    }

    async fn transition_batch(
        &self,
        memory_id: MemoryId,
        transitions: Vec<(usize, JohariQuadrant, TransitionTrigger)>,
    ) -> Result<JohariFingerprint, JohariError> {
        // Retrieve current state
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let mut johari = current.johari.clone();

        // Validate ALL transitions first (all-or-nothing)
        for (idx, (embedder_idx, to_quadrant, trigger)) in transitions.iter().enumerate() {
            // Check embedder index bounds
            if *embedder_idx >= NUM_EMBEDDERS {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!("Invalid embedder index: {}", embedder_idx),
                });
            }

            let current_quadrant = johari.dominant_quadrant(*embedder_idx);

            // Check transition validity
            if !current_quadrant.can_transition_to(*to_quadrant) {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!(
                        "Invalid transition {:?} → {:?} for embedder {}",
                        current_quadrant, to_quadrant, embedder_idx
                    ),
                });
            }

            // Check trigger validity
            if current_quadrant
                .transition_to(*to_quadrant, *trigger)
                .is_err()
            {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!(
                        "Invalid trigger {:?} for {:?} → {:?}",
                        trigger, current_quadrant, to_quadrant
                    ),
                });
            }
        }

        // Apply all transitions (all validated)
        for (embedder_idx, to_quadrant, _trigger) in transitions {
            set_quadrant_weights(&mut johari, embedder_idx, to_quadrant);
        }

        // Persist
        let mut updated = current;
        updated.johari = johari.clone();

        self.store
            .update(updated)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        Ok(johari)
    }

    async fn find_by_quadrant(
        &self,
        pattern: QuadrantPattern,
        limit: usize,
    ) -> Result<Vec<(MemoryId, JohariFingerprint)>, JohariError> {
        // Scan using semantic search with empty query
        let empty_query = SemanticFingerprint::zeroed();
        let options = TeleologicalSearchOptions::quick(limit * 10);

        let results = self
            .store
            .search_semantic(&empty_query, options)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        let matches: Vec<_> = results
            .into_iter()
            .filter(|r| matches_pattern(&r.fingerprint.johari, &pattern))
            .take(limit)
            .map(|r| (r.fingerprint.id, r.fingerprint.johari))
            .collect();

        Ok(matches)
    }

    async fn discover_blind_spots(
        &self,
        memory_id: MemoryId,
        external_signals: &[ExternalSignal],
    ) -> Result<Vec<BlindSpotCandidate>, JohariError> {
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let johari = &current.johari;
        let mut candidates = Vec::new();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let current_quadrant = johari.dominant_quadrant(embedder_idx);

            // Only consider Unknown or Hidden as potential blind spots
            if current_quadrant != JohariQuadrant::Unknown
                && current_quadrant != JohariQuadrant::Hidden
            {
                continue;
            }

            // Aggregate signal strength for this embedder
            let mut signal_strength = 0.0f32;
            let mut sources = Vec::new();

            for signal in external_signals {
                if signal.embedder_idx == embedder_idx {
                    signal_strength += signal.strength;
                    sources.push(signal.source.clone());
                }
            }

            if signal_strength > self.blind_spot_threshold {
                candidates.push(BlindSpotCandidate::new(
                    embedder_idx,
                    current_quadrant,
                    signal_strength,
                    sources,
                ));
            }
        }

        // Sort by signal strength descending
        candidates.sort_by(|a, b| {
            b.signal_strength
                .partial_cmp(&a.signal_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(candidates)
    }

    async fn get_transition_stats(
        &self,
        _time_range: TimeRange,
    ) -> Result<TransitionStats, JohariError> {
        // In a full implementation, this would query a transitions log table
        // For now, return empty stats (transitions aren't persisted to a log yet)
        Ok(TransitionStats::default())
    }

    async fn get_transition_history(
        &self,
        _memory_id: MemoryId,
        _limit: usize,
    ) -> Result<Vec<JohariTransition>, JohariError> {
        // In a full implementation, this would query stored transitions
        // For now, return empty (transitions aren't persisted to history yet)
        Ok(Vec::new())
    }
}

/// Implementation of JohariTransitionManager for DynDefaultJohariManager.
///
/// This is identical to the generic implementation but uses trait object store.
/// TASK-S004: Required for MCP handlers that use `Arc<dyn TeleologicalMemoryStore>`.
#[async_trait]
impl JohariTransitionManager for DynDefaultJohariManager {
    async fn classify(
        &self,
        _semantic: &SemanticFingerprint,
        context: &ClassificationContext,
    ) -> Result<JohariFingerprint, JohariError> {
        let mut fingerprint = JohariFingerprint::zeroed();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let delta_s = context.delta_s[embedder_idx];
            let delta_c = context.delta_c[embedder_idx];

            // Use existing classification logic from JohariFingerprint
            let quadrant = JohariFingerprint::classify_quadrant(delta_s, delta_c);

            // Apply disclosure intent override: if hidden intent, force Hidden
            let final_quadrant = if !context.disclosure_intent[embedder_idx]
                && quadrant == JohariQuadrant::Open
            {
                JohariQuadrant::Hidden
            } else {
                quadrant
            };

            // Set hard classification (100% weight to one quadrant)
            set_quadrant_weights(&mut fingerprint, embedder_idx, final_quadrant);
        }

        Ok(fingerprint)
    }

    async fn transition(
        &self,
        memory_id: MemoryId,
        embedder_idx: usize,
        to_quadrant: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Result<JohariFingerprint, JohariError> {
        // Validate embedder index (FAIL FAST)
        if embedder_idx >= NUM_EMBEDDERS {
            return Err(JohariError::InvalidEmbedderIndex(embedder_idx));
        }

        // Retrieve current state
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let mut johari = current.johari.clone();
        let current_quadrant = johari.dominant_quadrant(embedder_idx);

        // Validate transition using existing state machine
        if !current_quadrant.can_transition_to(to_quadrant) {
            return Err(JohariError::InvalidTransition {
                from: current_quadrant,
                to: to_quadrant,
                embedder_idx,
            });
        }

        // Validate trigger is valid for this transition
        let valid_transition = current_quadrant.transition_to(to_quadrant, trigger);
        if valid_transition.is_err() {
            return Err(JohariError::InvalidTrigger {
                from: current_quadrant,
                to: to_quadrant,
                trigger,
            });
        }

        // Apply transition (set 100% weight to new quadrant)
        set_quadrant_weights(&mut johari, embedder_idx, to_quadrant);

        // Update stored fingerprint
        let mut updated = current;
        updated.johari = johari.clone();

        self.store
            .update(updated)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        Ok(johari)
    }

    async fn transition_batch(
        &self,
        memory_id: MemoryId,
        transitions: Vec<(usize, JohariQuadrant, TransitionTrigger)>,
    ) -> Result<JohariFingerprint, JohariError> {
        // Retrieve current state
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let mut johari = current.johari.clone();

        // Validate ALL transitions first (all-or-nothing)
        for (idx, (embedder_idx, to_quadrant, trigger)) in transitions.iter().enumerate() {
            // Check embedder index bounds
            if *embedder_idx >= NUM_EMBEDDERS {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!("Invalid embedder index: {}", embedder_idx),
                });
            }

            let current_quadrant = johari.dominant_quadrant(*embedder_idx);

            // Check transition validity
            if !current_quadrant.can_transition_to(*to_quadrant) {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!(
                        "Invalid transition {:?} → {:?} for embedder {}",
                        current_quadrant, to_quadrant, embedder_idx
                    ),
                });
            }

            // Check trigger validity
            if current_quadrant
                .transition_to(*to_quadrant, *trigger)
                .is_err()
            {
                return Err(JohariError::BatchValidationFailed {
                    idx,
                    reason: format!(
                        "Invalid trigger {:?} for {:?} → {:?}",
                        trigger, current_quadrant, to_quadrant
                    ),
                });
            }
        }

        // Apply all transitions (all validated)
        for (embedder_idx, to_quadrant, _trigger) in transitions {
            set_quadrant_weights(&mut johari, embedder_idx, to_quadrant);
        }

        // Persist
        let mut updated = current;
        updated.johari = johari.clone();

        self.store
            .update(updated)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        Ok(johari)
    }

    async fn find_by_quadrant(
        &self,
        pattern: QuadrantPattern,
        limit: usize,
    ) -> Result<Vec<(MemoryId, JohariFingerprint)>, JohariError> {
        // Scan using semantic search with empty query
        let empty_query = SemanticFingerprint::zeroed();
        let options = TeleologicalSearchOptions::quick(limit * 10);

        let results = self
            .store
            .search_semantic(&empty_query, options)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?;

        let matches: Vec<_> = results
            .into_iter()
            .filter(|r| matches_pattern(&r.fingerprint.johari, &pattern))
            .take(limit)
            .map(|r| (r.fingerprint.id, r.fingerprint.johari))
            .collect();

        Ok(matches)
    }

    async fn discover_blind_spots(
        &self,
        memory_id: MemoryId,
        external_signals: &[ExternalSignal],
    ) -> Result<Vec<BlindSpotCandidate>, JohariError> {
        let current = self
            .store
            .retrieve(memory_id)
            .await
            .map_err(|e| JohariError::StorageError(e.to_string()))?
            .ok_or(JohariError::NotFound(memory_id))?;

        let johari = &current.johari;
        let mut candidates = Vec::new();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let current_quadrant = johari.dominant_quadrant(embedder_idx);

            // Only consider Unknown or Hidden as potential blind spots
            if current_quadrant != JohariQuadrant::Unknown
                && current_quadrant != JohariQuadrant::Hidden
            {
                continue;
            }

            // Aggregate signal strength for this embedder
            let mut signal_strength = 0.0f32;
            let mut sources = Vec::new();

            for signal in external_signals {
                if signal.embedder_idx == embedder_idx {
                    signal_strength += signal.strength;
                    sources.push(signal.source.clone());
                }
            }

            if signal_strength > self.blind_spot_threshold {
                candidates.push(BlindSpotCandidate::new(
                    embedder_idx,
                    current_quadrant,
                    signal_strength,
                    sources,
                ));
            }
        }

        // Sort by signal strength descending
        candidates.sort_by(|a, b| {
            b.signal_strength
                .partial_cmp(&a.signal_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(candidates)
    }

    async fn get_transition_stats(
        &self,
        _time_range: TimeRange,
    ) -> Result<TransitionStats, JohariError> {
        // In a full implementation, this would query a transitions log table
        // For now, return empty stats (transitions aren't persisted to a log yet)
        Ok(TransitionStats::default())
    }

    async fn get_transition_history(
        &self,
        _memory_id: MemoryId,
        _limit: usize,
    ) -> Result<Vec<JohariTransition>, JohariError> {
        // In a full implementation, this would query stored transitions
        // For now, return empty (transitions aren't persisted to history yet)
        Ok(Vec::new())
    }
}

/// Check if a JohariFingerprint matches a QuadrantPattern.
fn matches_pattern(johari: &JohariFingerprint, pattern: &QuadrantPattern) -> bool {
    match pattern {
        QuadrantPattern::AllIn(target) => {
            (0..NUM_EMBEDDERS).all(|i| johari.dominant_quadrant(i) == *target)
        }
        QuadrantPattern::AtLeast { quadrant, count } => {
            (0..NUM_EMBEDDERS)
                .filter(|&i| johari.dominant_quadrant(i) == *quadrant)
                .count()
                >= *count
        }
        QuadrantPattern::Exact(expected) => {
            (0..NUM_EMBEDDERS).all(|i| johari.dominant_quadrant(i) == expected[i])
        }
        QuadrantPattern::Mixed {
            min_open,
            max_unknown,
        } => {
            let open_count = (0..NUM_EMBEDDERS)
                .filter(|&i| johari.dominant_quadrant(i) == JohariQuadrant::Open)
                .count();
            let unknown_count = (0..NUM_EMBEDDERS)
                .filter(|&i| johari.dominant_quadrant(i) == JohariQuadrant::Unknown)
                .count();
            open_count >= *min_open && unknown_count <= *max_unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stubs::InMemoryTeleologicalStore;
    use crate::types::fingerprint::{PurposeVector, TeleologicalFingerprint};

    fn create_test_store() -> Arc<InMemoryTeleologicalStore> {
        Arc::new(InMemoryTeleologicalStore::new())
    }

    fn create_test_fingerprint() -> TeleologicalFingerprint {
        TeleologicalFingerprint::new(
            SemanticFingerprint::zeroed(),
            PurposeVector::default(),
            JohariFingerprint::zeroed(),
            [0u8; 32],
        )
    }

    #[tokio::test]
    async fn test_classify_from_utl_state() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store);
        let semantic = SemanticFingerprint::zeroed();

        // UTL state: low entropy, high coherence → Open
        let context = ClassificationContext {
            delta_s: [0.3; NUM_EMBEDDERS],
            delta_c: [0.7; NUM_EMBEDDERS],
            disclosure_intent: [true; NUM_EMBEDDERS],
            access_counts: [0; NUM_EMBEDDERS],
        };

        let result = manager.classify(&semantic, &context).await.unwrap();

        // Verify: All embedders should be Open
        for i in 0..NUM_EMBEDDERS {
            assert_eq!(
                result.dominant_quadrant(i),
                JohariQuadrant::Open,
                "Embedder {} should be Open, got {:?}",
                i,
                result.dominant_quadrant(i)
            );
        }

        println!("[VERIFIED] test_classify_from_utl_state: All embedders correctly classified as Open");
    }

    #[tokio::test]
    async fn test_classify_all_quadrants() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store);
        let semantic = SemanticFingerprint::zeroed();

        // Test all four quadrant classifications
        let test_cases = [
            (0.3, 0.7, JohariQuadrant::Open),   // Low S, High C
            (0.3, 0.3, JohariQuadrant::Hidden), // Low S, Low C
            (0.7, 0.3, JohariQuadrant::Blind),  // High S, Low C
            (0.7, 0.7, JohariQuadrant::Unknown), // High S, High C
        ];

        for (delta_s, delta_c, expected) in test_cases {
            let context = ClassificationContext::uniform(delta_s, delta_c);
            let result = manager.classify(&semantic, &context).await.unwrap();

            for i in 0..NUM_EMBEDDERS {
                assert_eq!(
                    result.dominant_quadrant(i),
                    expected,
                    "ΔS={}, ΔC={} should classify as {:?}",
                    delta_s,
                    delta_c,
                    expected
                );
            }
        }

        println!("[VERIFIED] test_classify_all_quadrants: All quadrant classifications correct");
    }

    #[tokio::test]
    async fn test_transition_valid() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store.clone());

        // Store a fingerprint with Hidden quadrant for E1
        let mut fp = create_test_fingerprint();
        fp.johari.set_quadrant(0, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
        let id = store.store(fp).await.unwrap();

        // Transition Hidden → Open via ExplicitShare
        let result = manager
            .transition(id, 0, JohariQuadrant::Open, TransitionTrigger::ExplicitShare)
            .await
            .unwrap();

        assert_eq!(result.dominant_quadrant(0), JohariQuadrant::Open);

        // [VERIFY] Read back from store to confirm persistence
        let stored = store.retrieve(id).await.unwrap().unwrap();
        assert_eq!(
            stored.johari.dominant_quadrant(0),
            JohariQuadrant::Open,
            "[VERIFICATION FAILED] Transition not persisted to store"
        );

        println!("[VERIFIED] test_transition_valid: Hidden→Open transition succeeded and persisted");
    }

    #[tokio::test]
    async fn test_transition_invalid_returns_error() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store.clone());

        // Store with Open quadrant
        let mut fp = create_test_fingerprint();
        fp.johari.set_quadrant(0, 1.0, 0.0, 0.0, 0.0, 1.0); // Open
        let id = store.store(fp).await.unwrap();

        // Attempt invalid transition: Open → Blind (not allowed)
        let result = manager
            .transition(
                id,
                0,
                JohariQuadrant::Blind,
                TransitionTrigger::ExternalObservation,
            )
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            JohariError::InvalidTransition {
                from,
                to,
                embedder_idx,
            } => {
                assert_eq!(from, JohariQuadrant::Open);
                assert_eq!(to, JohariQuadrant::Blind);
                assert_eq!(embedder_idx, 0);
            }
            e => panic!("Expected InvalidTransition, got {:?}", e),
        }

        println!("[VERIFIED] test_transition_invalid_returns_error: Invalid transition correctly rejected");
    }

    #[tokio::test]
    async fn test_discover_blind_spots() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store.clone()).with_blind_spot_threshold(0.5);

        // Store with Unknown in E5 (causal)
        let mut fp = create_test_fingerprint();
        fp.johari.set_quadrant(5, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
        let id = store.store(fp).await.unwrap();

        // External signals referencing E5
        let signals = vec![
            ExternalSignal::new("user_feedback", 5, 0.4),
            ExternalSignal::new("dream_layer", 5, 0.3),
        ];

        let candidates = manager.discover_blind_spots(id, &signals).await.unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].embedder_idx, 5);
        assert_eq!(candidates[0].current_quadrant, JohariQuadrant::Unknown);
        assert_eq!(candidates[0].suggested_transition, JohariQuadrant::Blind);
        assert!((candidates[0].signal_strength - 0.7).abs() < 0.01);

        println!("[VERIFIED] test_discover_blind_spots: Blind spot correctly discovered for E5");
    }

    #[tokio::test]
    async fn test_batch_transition_all_or_nothing() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store.clone());

        // Store with multiple Unknown embedders
        let mut fp = create_test_fingerprint();
        for i in 0..5 {
            fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
        }
        let id = store.store(fp).await.unwrap();

        // Batch with one invalid transition (invalid embedder index)
        let transitions_invalid_idx = vec![
            (0, JohariQuadrant::Open, TransitionTrigger::DreamConsolidation),
            (99, JohariQuadrant::Open, TransitionTrigger::DreamConsolidation), // Invalid index
        ];

        let result = manager.transition_batch(id, transitions_invalid_idx).await;
        assert!(result.is_err());

        // [VERIFY] Original state unchanged
        let stored = store.retrieve(id).await.unwrap().unwrap();
        assert_eq!(
            stored.johari.dominant_quadrant(0),
            JohariQuadrant::Unknown,
            "[VERIFICATION FAILED] Batch rollback didn't preserve original state"
        );

        println!("[VERIFIED] test_batch_transition_all_or_nothing: Failed batch preserved original state");
    }

    #[tokio::test]
    async fn test_batch_transition_success() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store.clone());

        // Store with multiple Unknown embedders
        let mut fp = create_test_fingerprint();
        for i in 0..5 {
            fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
        }
        let id = store.store(fp).await.unwrap();

        // Valid batch transitions
        let transitions = vec![
            (0, JohariQuadrant::Open, TransitionTrigger::DreamConsolidation),
            (1, JohariQuadrant::Hidden, TransitionTrigger::DreamConsolidation),
            (2, JohariQuadrant::Blind, TransitionTrigger::ExternalObservation),
        ];

        let result = manager.transition_batch(id, transitions).await.unwrap();

        assert_eq!(result.dominant_quadrant(0), JohariQuadrant::Open);
        assert_eq!(result.dominant_quadrant(1), JohariQuadrant::Hidden);
        assert_eq!(result.dominant_quadrant(2), JohariQuadrant::Blind);

        // [VERIFY] Changes persisted
        let stored = store.retrieve(id).await.unwrap().unwrap();
        assert_eq!(stored.johari.dominant_quadrant(0), JohariQuadrant::Open);
        assert_eq!(stored.johari.dominant_quadrant(1), JohariQuadrant::Hidden);
        assert_eq!(stored.johari.dominant_quadrant(2), JohariQuadrant::Blind);

        println!("[VERIFIED] test_batch_transition_success: All batch transitions applied and persisted");
    }

    // ==================== EDGE CASE TESTS (Required by TASK-L004) ====================

    #[tokio::test]
    async fn edge_case_1_empty_signals() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store.clone());

        let fp = create_test_fingerprint();
        let id = store.store(fp).await.unwrap();

        // State BEFORE
        let before = store.retrieve(id).await.unwrap().unwrap();
        println!("[STATE BEFORE] Johari: {:?}", before.johari.quadrants);

        // Action with empty signals
        let result = manager.discover_blind_spots(id, &[]).await.unwrap();

        assert!(result.is_empty(), "Should return empty vec for no signals");

        // State AFTER
        let after = store.retrieve(id).await.unwrap().unwrap();
        println!("[STATE AFTER] Johari: {:?}", after.johari.quadrants);

        // [VERIFY] State unchanged
        assert_eq!(
            before.johari.quadrants, after.johari.quadrants,
            "[EDGE CASE 1 FAILED] State changed with empty signals"
        );

        println!("[EDGE CASE 1 PASSED] Empty signals correctly handled");
    }

    #[tokio::test]
    async fn edge_case_2_boundary_classification() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store);

        let semantic = SemanticFingerprint::zeroed();
        let context = ClassificationContext {
            delta_s: [0.5; NUM_EMBEDDERS], // Exactly at threshold
            delta_c: [0.5; NUM_EMBEDDERS], // Exactly at threshold
            disclosure_intent: [true; NUM_EMBEDDERS],
            access_counts: [0; NUM_EMBEDDERS],
        };

        // State BEFORE
        println!(
            "[STATE BEFORE] UTL: ΔS={:?}, ΔC={:?}",
            context.delta_s, context.delta_c
        );

        let result = manager.classify(&semantic, &context).await.unwrap();

        // State AFTER
        println!("[STATE AFTER] Classification:");
        for i in 0..NUM_EMBEDDERS {
            println!("  E{}: {:?}", i + 1, result.dominant_quadrant(i));
        }

        // Per spec: entropy < 0.5 is low (0.5 is NOT < 0.5, so high entropy)
        // Per spec: coherence > 0.5 is high (0.5 is NOT > 0.5, so low coherence)
        // High S + Low C = Blind
        for i in 0..NUM_EMBEDDERS {
            assert_eq!(
                result.dominant_quadrant(i),
                JohariQuadrant::Blind,
                "[EDGE CASE 2 FAILED] Boundary (0.5, 0.5) should classify as Blind"
            );
        }

        println!("[EDGE CASE 2 PASSED] Boundary classification correct (Blind)");
    }

    #[tokio::test]
    async fn edge_case_3_max_embedder_index() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store.clone());

        let mut fp = create_test_fingerprint();
        // Set all to Unknown so transitions are valid
        for i in 0..NUM_EMBEDDERS {
            fp.johari.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0);
        }
        let id = store.store(fp).await.unwrap();

        // Index 12 should work (valid max)
        let valid_result = manager
            .transition(
                id,
                12,
                JohariQuadrant::Open,
                TransitionTrigger::DreamConsolidation,
            )
            .await;

        assert!(
            valid_result.is_ok(),
            "Embedder index 12 should be valid: {:?}",
            valid_result
        );

        // Index 13 should fail
        let invalid_result = manager
            .transition(
                id,
                13,
                JohariQuadrant::Open,
                TransitionTrigger::DreamConsolidation,
            )
            .await;

        assert!(invalid_result.is_err());
        match invalid_result {
            Err(JohariError::InvalidEmbedderIndex(idx)) => {
                assert_eq!(idx, 13);
                println!("[EDGE CASE 3 PASSED] Invalid embedder index correctly rejected");
            }
            other => panic!("Expected InvalidEmbedderIndex, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_disclosure_intent_overrides_open() {
        let store = create_test_store();
        let manager = DefaultJohariManager::new(store);
        let semantic = SemanticFingerprint::zeroed();

        // UTL state would classify as Open, but disclosure is false
        let context = ClassificationContext {
            delta_s: [0.3; NUM_EMBEDDERS],
            delta_c: [0.7; NUM_EMBEDDERS],
            disclosure_intent: [false; NUM_EMBEDDERS], // Force hidden
            access_counts: [0; NUM_EMBEDDERS],
        };

        let result = manager.classify(&semantic, &context).await.unwrap();

        // Should be Hidden despite Open-friendly UTL values
        for i in 0..NUM_EMBEDDERS {
            assert_eq!(
                result.dominant_quadrant(i),
                JohariQuadrant::Hidden,
                "Disclosure=false should force Hidden"
            );
        }

        println!("[VERIFIED] test_disclosure_intent_overrides_open: Disclosure intent correctly overrides");
    }

    #[test]
    fn test_matches_pattern_all_in() {
        let mut johari = JohariFingerprint::zeroed();
        for i in 0..NUM_EMBEDDERS {
            johari.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0); // All Open
        }

        assert!(matches_pattern(
            &johari,
            &QuadrantPattern::AllIn(JohariQuadrant::Open)
        ));
        assert!(!matches_pattern(
            &johari,
            &QuadrantPattern::AllIn(JohariQuadrant::Hidden)
        ));

        println!("[VERIFIED] test_matches_pattern_all_in: Pattern matching works correctly");
    }

    #[test]
    fn test_matches_pattern_at_least() {
        let mut johari = JohariFingerprint::zeroed();
        // Set 5 embedders to Open
        for i in 0..5 {
            johari.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0);
        }

        assert!(matches_pattern(
            &johari,
            &QuadrantPattern::AtLeast {
                quadrant: JohariQuadrant::Open,
                count: 5
            }
        ));
        assert!(!matches_pattern(
            &johari,
            &QuadrantPattern::AtLeast {
                quadrant: JohariQuadrant::Open,
                count: 6
            }
        ));

        println!("[VERIFIED] test_matches_pattern_at_least: AtLeast pattern matching works correctly");
    }
}
