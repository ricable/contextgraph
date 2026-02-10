//! Causal embedding benchmarking system.
//!
//! 8-phase benchmark for evaluating E5 causal embedder quality:
//! 1. Query intent detection accuracy
//! 2. E5 embedding quality (spread, anisotropy, standalone accuracy)
//! 3. Direction modifier verification
//! 4. Ablation analysis (with/without E5)
//! 5. Causal gate effectiveness (TPR/TNR)
//! 6. End-to-end retrieval accuracy (MRR, NDCG)
//! 7. Cross-domain generalization (held-out domains)
//! 8. Performance profiling (latency, throughput)

pub mod comparison;
pub mod data_loader;
pub mod metrics;
pub mod phases;
pub mod provider;
pub mod report;
