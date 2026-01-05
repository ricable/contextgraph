//! ConsolidationPhase enum and associated methods.
//!
//! Defines memory consolidation phases based on sleep-inspired dynamics.

use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

/// Memory consolidation phases based on sleep-inspired dynamics.
///
/// Each phase has different characteristics for memory processing:
/// - **NREM**: Non-REM phase for replay and tight coupling
/// - **REM**: REM phase for exploring attractor dynamics
/// - **Wake**: Normal waking operation
///
/// # Constitution Reference
///
/// - NREM: Replay + tight coupling (recency_bias: 0.8)
/// - REM: Explore attractors (temp: 2.0)
/// - Wake: Normal operation (balanced processing)
///
/// # Example
///
/// ```
/// use context_graph_utl::phase::ConsolidationPhase;
///
/// let phase = ConsolidationPhase::NREM;
///
/// // Get phase-specific parameters
/// assert_eq!(phase.recency_bias(), 0.8);
/// assert!(phase.is_consolidation_phase());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ConsolidationPhase {
    /// Non-REM sleep phase: Memory replay with tight coupling.
    ///
    /// Characteristics:
    /// - High recency bias (0.8)
    /// - Strong coupling strength
    /// - Focus on recent memory consolidation
    NREM,

    /// REM sleep phase: Attractor exploration.
    ///
    /// Characteristics:
    /// - High temperature (2.0) for exploration
    /// - Loose coupling
    /// - Creative association and pattern discovery
    REM,

    /// Waking phase: Normal operation.
    ///
    /// Characteristics:
    /// - Balanced processing
    /// - Standard temperature (1.0)
    /// - Active learning and interaction
    #[default]
    Wake,
}

impl ConsolidationPhase {
    /// Get the recency bias for this phase.
    ///
    /// Higher values prioritize recent memories during replay.
    ///
    /// # Returns
    ///
    /// Recency bias in `[0, 1]`:
    /// - NREM: 0.8 (strong recent memory bias)
    /// - REM: 0.4 (moderate bias, more exploration)
    /// - Wake: 0.5 (balanced)
    #[inline]
    pub fn recency_bias(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.8,
            ConsolidationPhase::REM => 0.4,
            ConsolidationPhase::Wake => 0.5,
        }
    }

    /// Get the temperature parameter for this phase.
    ///
    /// Temperature controls exploration vs. exploitation:
    /// - High temperature: More random/exploratory
    /// - Low temperature: More deterministic/focused
    ///
    /// # Returns
    ///
    /// Temperature parameter:
    /// - NREM: 0.5 (focused replay)
    /// - REM: 2.0 (high exploration)
    /// - Wake: 1.0 (balanced)
    #[inline]
    pub fn temperature(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.5,
            ConsolidationPhase::REM => 2.0,
            ConsolidationPhase::Wake => 1.0,
        }
    }

    /// Get the coupling strength for this phase.
    ///
    /// Controls how strongly different memory systems are coupled.
    ///
    /// # Returns
    ///
    /// Coupling strength in `[0, 1]`:
    /// - NREM: 0.9 (tight coupling for replay)
    /// - REM: 0.3 (loose coupling for exploration)
    /// - Wake: 0.6 (moderate coupling)
    #[inline]
    pub fn coupling_strength(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.9,
            ConsolidationPhase::REM => 0.3,
            ConsolidationPhase::Wake => 0.6,
        }
    }

    /// Get the learning rate modifier for this phase.
    ///
    /// # Returns
    ///
    /// Learning rate multiplier:
    /// - NREM: 0.3 (reduced active learning)
    /// - REM: 0.5 (moderate learning from associations)
    /// - Wake: 1.0 (full learning rate)
    #[inline]
    pub fn learning_rate_modifier(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.3,
            ConsolidationPhase::REM => 0.5,
            ConsolidationPhase::Wake => 1.0,
        }
    }

    /// Check if this is a consolidation phase (NREM or REM).
    ///
    /// # Returns
    ///
    /// `true` if NREM or REM, `false` if Wake.
    #[inline]
    pub fn is_consolidation_phase(&self) -> bool {
        matches!(self, ConsolidationPhase::NREM | ConsolidationPhase::REM)
    }

    /// Check if this is the waking phase.
    #[inline]
    pub fn is_wake(&self) -> bool {
        matches!(self, ConsolidationPhase::Wake)
    }

    /// Get the default phase angle for this consolidation state.
    ///
    /// # Returns
    ///
    /// Phase angle in `[0, pi]`:
    /// - NREM: 0 (synchronized, cos = 1)
    /// - REM: pi (anti-phase, cos = -1)
    /// - Wake: pi/2 (orthogonal, cos = 0)
    #[inline]
    pub fn default_phase_angle(&self) -> f32 {
        match self {
            ConsolidationPhase::NREM => 0.0,
            ConsolidationPhase::REM => PI,
            ConsolidationPhase::Wake => PI / 2.0,
        }
    }

    /// Get a human-readable name for this phase.
    pub fn name(&self) -> &'static str {
        match self {
            ConsolidationPhase::NREM => "NREM",
            ConsolidationPhase::REM => "REM",
            ConsolidationPhase::Wake => "Wake",
        }
    }

    /// Get a description of this phase.
    pub fn description(&self) -> &'static str {
        match self {
            ConsolidationPhase::NREM => "Memory replay with tight coupling",
            ConsolidationPhase::REM => "Attractor exploration with loose coupling",
            ConsolidationPhase::Wake => "Normal waking operation",
        }
    }
}

impl std::fmt::Display for ConsolidationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
