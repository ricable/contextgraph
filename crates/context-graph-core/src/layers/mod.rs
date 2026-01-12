//! Real implementations of bio-nervous system layers.
//!
//! These are PRODUCTION implementations that replace the stubs.
//! Each layer implements the `NervousLayer` trait with real processing logic.
//!
//! # Layers
//!
//! - [`SensingLayer`] - L1 Multi-modal input processing with PII scrubbing
//! - [`ReflexLayer`] - L2 Modern Hopfield Network cache for instant responses
//! - [`MemoryLayer`] - L3 Modern Hopfield Network associative memory with decay scoring
//! - [`LearningLayer`] - L4 UTL-driven weight optimization with consolidation triggers
//! - [`CoherenceLayer`] - L5 Kuramoto sync and Global Workspace broadcast
//!
//! # Constitution Compliance
//!
//! Per SEC-01/SEC-02: All input is validated and sanitized via PII scrubber.
//! Per AP-007: No mock data, no fallbacks - errors fail fast.
//! Per AP-009: NaN/Infinity rejected in UTL computations.
//! Per Perf: Reflex layer latency <100us, Memory layer <1ms, Learning/Coherence <10ms.

mod coherence;
mod learning;
mod memory;
mod reflex;
mod sensing;

#[cfg(test)]
mod tests_full_state_verification;

pub use coherence::{
    CoherenceLayer, ConsciousnessState, GlobalWorkspace, GwtThresholds, KuramotoNetwork,
    KuramotoOscillator, INTEGRATION_STEPS, KURAMOTO_DT, KURAMOTO_K, KURAMOTO_N,
};
// Re-export deprecated constants with warning suppression for backwards compatibility
#[allow(deprecated)]
pub use coherence::{FRAGMENTATION_THRESHOLD, GW_THRESHOLD, HYPERSYNC_THRESHOLD};
pub use learning::{
    LearningLayer, UtlWeightComputer, WeightDelta, DEFAULT_CONSOLIDATION_THRESHOLD,
    DEFAULT_LEARNING_RATE, GRADIENT_CLIP, TARGET_FREQUENCY_HZ,
};
pub use memory::{
    AssociativeMemory, MemoryContent, MemoryLayer, ScoredMemory, StoredMemory,
    DECAY_HALF_LIFE_HOURS, DEFAULT_MAX_RETRIEVE, DEFAULT_MHN_BETA, MEMORY_PATTERN_DIM,
    MIN_MEMORY_SIMILARITY,
};
pub use reflex::{
    CacheStats, CachedResponse, ModernHopfieldCache, ReflexLayer, DEFAULT_BETA,
    DEFAULT_CACHE_CAPACITY, MIN_HIT_SIMILARITY, PATTERN_DIM,
};
pub use sensing::{PiiPattern, PiiScrubber, ScrubbedContent, SensingLayer, SensingMetrics};
