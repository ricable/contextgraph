//! Real dataset loading and processing for large-scale benchmarks.
//!
//! This module loads pre-processed datasets (e.g., Wikipedia chunks)
//! and generates fingerprints for benchmark testing.

pub mod embedder;
pub mod loader;
pub mod runner;

pub use embedder::RealDataEmbedder;
pub use loader::{ChunkRecord, DatasetLoader, DatasetMetadata, RealDataset};
pub use runner::{RealDataBenchConfig, RealDataBenchRunner, RealDataResults};
