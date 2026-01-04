//! Surprise (Delta-S) computation module.
//!
//! Implements surprise/novelty detection using:
//! - KL divergence for distribution comparison
//! - Embedding distance for semantic novelty
//! - Combined surprise calculation
//!
//! # Constitution Reference
//!
//! The surprise component (Delta-S) measures the novelty or unexpectedness of information:
//! - Range: `[0, 1]` where higher values indicate more surprising/novel information
//! - Used in UTL formula: `L = f((Delta-S x Delta-C) * w_e * cos(phi))`
//!
//! # Example
//!
//! ```ignore
//! use context_graph_utl::surprise::SurpriseCalculator;
//! use context_graph_utl::config::SurpriseConfig;
//!
//! let config = SurpriseConfig::default();
//! let calculator = SurpriseCalculator::new(&config);
//!
//! let current = vec![0.1, 0.2, 0.3, 0.4];
//! let history = vec![
//!     vec![0.15, 0.25, 0.35, 0.25],
//!     vec![0.12, 0.22, 0.32, 0.34],
//! ];
//!
//! let surprise = calculator.compute_surprise(&current, &history);
//! assert!(surprise >= 0.0 && surprise <= 1.0);
//! ```

mod calculator;
mod embedding_distance;
mod kl_divergence;

pub use calculator::SurpriseCalculator;
pub use embedding_distance::{
    compute_cosine_distance, compute_embedding_surprise, EmbeddingDistanceCalculator,
};
pub use kl_divergence::{compute_kl_divergence, KlDivergenceCalculator};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SurpriseConfig;

    #[test]
    fn test_module_exports() {
        // Verify all expected types are accessible
        let config = SurpriseConfig::default();
        let _calculator = SurpriseCalculator::new(&config);
        let _kl_calc = KlDivergenceCalculator::default();
        let _emb_calc = EmbeddingDistanceCalculator::default();
    }

    #[test]
    fn test_standalone_functions() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.1, 0.2, 0.3, 0.4];

        let kl = compute_kl_divergence(&a, &b, 1e-10);
        assert!(kl >= 0.0);

        let dist = compute_cosine_distance(&a, &b);
        assert!(dist >= 0.0 && dist <= 1.0);
    }
}
