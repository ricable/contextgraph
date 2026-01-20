//! Report generation for benchmark results.
//!
//! Supports JSON for CI integration and Markdown for documentation.

pub mod json;
pub mod markdown;

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::runners::BenchmarkResults;

/// Report format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// JSON for CI integration.
    Json,
    /// Markdown tables for documentation.
    Markdown,
    /// Both formats.
    Both,
}

/// Benchmark report generator.
pub struct BenchmarkReport {
    results: BenchmarkResults,
}

impl BenchmarkReport {
    /// Create a new report from benchmark results.
    pub fn new(results: BenchmarkResults) -> Self {
        Self { results }
    }

    /// Generate report in specified format.
    pub fn generate(&self, format: ReportFormat) -> ReportOutput {
        match format {
            ReportFormat::Json => ReportOutput {
                json: Some(json::generate_json(&self.results)),
                markdown: None,
            },
            ReportFormat::Markdown => ReportOutput {
                json: None,
                markdown: Some(markdown::generate_markdown(&self.results)),
            },
            ReportFormat::Both => ReportOutput {
                json: Some(json::generate_json(&self.results)),
                markdown: Some(markdown::generate_markdown(&self.results)),
            },
        }
    }

    /// Write report to file(s).
    pub fn write_to_file(&self, format: ReportFormat, base_path: &Path) -> std::io::Result<()> {
        let output = self.generate(format);

        if let Some(json_content) = output.json {
            let json_path = base_path.with_extension("json");
            std::fs::write(&json_path, json_content)?;
        }

        if let Some(md_content) = output.markdown {
            let md_path = base_path.with_extension("md");
            std::fs::write(&md_path, md_content)?;
        }

        Ok(())
    }

    /// Get results reference.
    pub fn results(&self) -> &BenchmarkResults {
        &self.results
    }
}

/// Generated report output.
#[derive(Debug, Clone)]
pub struct ReportOutput {
    /// JSON report content (if generated).
    pub json: Option<String>,
    /// Markdown report content (if generated).
    pub markdown: Option<String>,
}

/// Summary data for JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Timestamp of benchmark run.
    pub timestamp: String,
    /// Configuration used.
    pub tiers_tested: Vec<String>,
    /// Random seed used.
    pub seed: u64,
    /// Winner of comparison.
    pub winner: String,
    /// Scaling advantage factor.
    pub scaling_advantage_factor: Option<f64>,
    /// Overall improvement percentage.
    pub overall_improvement_pct: f64,
    /// Memory overhead factor.
    pub memory_overhead_factor: Option<f64>,
}

impl From<&BenchmarkResults> for BenchmarkSummary {
    fn from(results: &BenchmarkResults) -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            tiers_tested: results
                .config
                .tiers
                .iter()
                .map(|t| t.tier.to_string())
                .collect(),
            seed: results.config.seed,
            winner: results.comparison.winner.clone(),
            scaling_advantage_factor: results.comparison.scaling_advantage_factor,
            overall_improvement_pct: results.comparison.overall_improvement * 100.0,
            memory_overhead_factor: results.comparison.memory_overhead_factor,
        }
    }
}
