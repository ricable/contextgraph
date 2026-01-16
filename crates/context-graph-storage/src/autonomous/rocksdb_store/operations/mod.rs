//! Database operations for RocksDbAutonomousStore.
//!
//! Each submodule contains operations for a specific data type or category.
//!
//! TASK-P0-004: Removed drift and goal modules (obsolete after North Star removal).

pub mod consolidation;
pub mod curation;
// TASK-P0-004: drift module removed - old drift detection replaced by topic_stability.churn_rate
// TASK-P0-004: goal module removed - manual goals forbidden by ARCH-03
pub mod lineage;
pub mod maintenance;
pub mod singleton;
