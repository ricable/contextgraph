//! NORTH-016: Weight Adjuster Service
//!
//! Optimizes section weights based on performance feedback using gradient descent
//! with momentum. This service learns optimal goal weights by adjusting them
//! based on alignment performance metrics.
//!
//! # Algorithm
//!
//! Uses momentum-based gradient descent:
//! - velocity[t] = momentum * velocity[t-1] + lr * gradient
//! - weight[t] = weight[t-1] - velocity[t]
//! - Weights are clamped to configured bounds

mod adjuster;
mod types;

// Re-export all public items for backwards compatibility
pub use adjuster::WeightAdjuster;
pub use types::{AdjustmentReason, AdjustmentReport, WeightAdjusterConfig};

#[cfg(test)]
mod tests {
    mod adjuster_tests;
    mod config_tests;
    mod integration_tests;
}
