//! Coherence (ΔC) computation module.
//!
//! Implements coherence/understanding tracking:
//! - Rolling window buffer for temporal coherence
//! - Structural coherence from graph relationships
//! - Combined coherence calculation
//!
//! # Constitution Reference
//!
//! The coherence component (ΔC) measures how well information integrates with existing knowledge:
//! - Range: `[0, 1]` where higher values indicate better integration
//! - Used in UTL formula: `L = f((ΔS × ΔC) · wₑ · cos φ)`
//!
//! # Example
//!
//! ```ignore
//! use context_graph_utl::coherence::{CoherenceTracker, WindowConfig};
//! use context_graph_utl::config::CoherenceConfig;
//!
//! let config = CoherenceConfig::default();
//! let mut tracker = CoherenceTracker::new(&config);
//!
//! // Add embeddings to the rolling window
//! tracker.update(&[0.1, 0.2, 0.3, 0.4]);
//! tracker.update(&[0.15, 0.25, 0.35, 0.25]);
//!
//! // Compute coherence for a new embedding
//! let current = vec![0.12, 0.22, 0.32, 0.34];
//! let history = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.15, 0.25, 0.35, 0.25]];
//! let coherence = tracker.compute_coherence(&current, &history);
//! assert!(coherence >= 0.0 && coherence <= 1.0);
//! ```

mod structural;
mod tracker;
mod window;

pub use structural::{compute_structural_coherence, StructuralCoherenceCalculator};
pub use tracker::CoherenceTracker;
pub use window::{RollingWindow, WindowConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CoherenceConfig;

    #[test]
    fn test_module_exports() {
        // Verify all expected types are accessible
        let config = CoherenceConfig::default();
        let _tracker = CoherenceTracker::new(&config);
        let _window: RollingWindow<f32> = RollingWindow::new(10);
        let _window_config = WindowConfig::default();
        let _structural_calc = StructuralCoherenceCalculator::default();
    }

    #[test]
    fn test_standalone_functions() {
        let neighbor_embeddings = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.15, 0.25, 0.35, 0.25]];
        let node_id = uuid::Uuid::new_v4();

        let coherence = compute_structural_coherence(node_id, &neighbor_embeddings);
        assert!((0.0..=1.0).contains(&coherence));
    }

    #[test]
    fn test_end_to_end_coherence() {
        let config = CoherenceConfig::default();
        let mut tracker = CoherenceTracker::new(&config);

        // Build up history
        let embeddings = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.12, 0.22, 0.28, 0.38],
            vec![0.11, 0.21, 0.29, 0.39],
        ];

        for emb in &embeddings {
            tracker.update(emb);
        }

        // Test coherence computation
        let current = vec![0.13, 0.23, 0.27, 0.37];
        let coherence = tracker.compute_coherence(&current, &embeddings);

        // High coherence expected since embeddings are similar
        assert!((0.0..=1.0).contains(&coherence));
        assert!(
            coherence > 0.5,
            "Expected high coherence for similar embeddings, got {}",
            coherence
        );
    }
}
