//! Parameter tuning modules for embedder optimization.
//!
//! This module provides grid search and optimization utilities
//! for tuning embedder-specific parameters.
//!
//! # Available Tuners
//!
//! - **E7Tuner**: Grid search for E7 code embedding parameters (e7_blend, min_score, fetch_multiplier)
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_benchmark::tuning::{E7TuningConfig, E7Tuner, SimulatedScoreProvider};
//!
//! let config = E7TuningConfig::default();
//! let tuner = E7Tuner::new(config);
//! let ground_truth = vec![]; // Load from data/e7_ground_truth/queries.jsonl
//! let score_provider = SimulatedScoreProvider::new(42);
//! let results = tuner.run_grid_search(&ground_truth, &score_provider);
//! println!("Best params: {:?}", results.best_params);
//! ```

pub mod e7_tuning;

pub use e7_tuning::{
    E7ParamResult, E7Params, E7Tuner, E7TuningConfig, E7TuningResults, ParameterSensitivity,
    QueryTypeResult, ScoreProvider, SimulatedScoreProvider, TuningSummary,
};
