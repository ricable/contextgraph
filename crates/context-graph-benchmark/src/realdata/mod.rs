//! Real dataset loading and processing for large-scale benchmarks.
//!
//! This module loads pre-processed datasets (e.g., Wikipedia chunks)
//! and generates fingerprints for benchmark testing.
//!
//! ## Modules
//!
//! - `loader` - JSONL dataset loading from chunks.jsonl + metadata.json
//! - `embedder` - 13-embedder fingerprint generation (GPU)
//! - `runner` - Basic real data benchmark runner
//! - `config` - Unified benchmark configuration
//! - `temporal_injector` - Temporal metadata injection for E2/E3/E4
//! - `ground_truth` - Per-embedder ground truth generation
//! - `results` - Unified benchmark result structures
//! - `report` - Markdown report generation

pub mod config;
pub mod embedder;
pub mod ground_truth;
pub mod loader;
pub mod report;
pub mod results;
pub mod runner;
pub mod temporal_injector;

pub use config::{EmbedderName, FusionStrategy, TemporalInjectionConfig, UnifiedBenchmarkConfig};
pub use embedder::RealDataEmbedder;
pub use ground_truth::{GroundTruthGenerator, QueryGroundTruth, UnifiedGroundTruth};
pub use loader::{ChunkRecord, DatasetLoader, DatasetMetadata, RealDataset};
pub use report::ReportGenerator;
pub use results::{EmbedderResults, UnifiedBenchmarkResults};
pub use runner::{RealDataBenchConfig, RealDataBenchRunner, RealDataResults};
pub use temporal_injector::{InjectedTemporalMetadata, TemporalMetadataInjector, TemporalSession};
