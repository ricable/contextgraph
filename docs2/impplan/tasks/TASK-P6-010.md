# Task: TASK-P6-010 - Performance Validation

```xml
<task_spec id="TASK-P6-010" version="1.0">
<metadata>
  <title>Performance Validation</title>
  <phase>6</phase>
  <sequence>52</sequence>
  <layer>surface</layer>
  <estimated_loc>200</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-009">Integration tests (validates functionality first)</dependency>
  </dependencies>
  <produces>
    <artifact type="test">performance_tests.rs</artifact>
    <artifact type="benchmark">cli_benchmarks.rs</artifact>
  </produces>
</metadata>

<context>
  <background>
    Performance validation ensures the system meets timeout constraints for
    all Claude Code hooks. Each hook has a specific timeout (500ms to 30s)
    and the CLI must complete well within these limits.
  </background>
  <business_value>
    Prevents timeout errors in production that would break the user experience.
    Identifies performance bottlenecks before they become problems.
  </business_value>
  <technical_context>
    Tests measure latency for each CLI command under various memory counts.
    Validates that inject-brief stays under 400ms even with large databases.
    Uses criterion for benchmark reproducibility.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">All CLI commands implemented and tested</prerequisite>
  <prerequisite type="crate">criterion = "0.5" for benchmarks</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>Timeout compliance validation for all hooks</item>
    <item>Latency benchmarks for each CLI command</item>
    <item>Scalability tests with increasing memory counts</item>
    <item>Cold start vs warm cache comparison</item>
  </includes>
  <excludes>
    <item>Functional correctness tests (TASK-P6-009)</item>
    <item>Stress testing / load testing</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>inject-brief completes in &lt;400ms (hook timeout 500ms)</description>
    <verification>Benchmark P95 latency &lt; 400ms</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>inject-context completes in &lt;4500ms (hook timeout 5000ms)</description>
    <verification>Benchmark P95 latency &lt; 4500ms</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>capture-memory completes in &lt;2700ms (hook timeout 3000ms)</description>
    <verification>Benchmark P95 latency &lt; 2700ms</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Performance degrades gracefully with memory count</description>
    <verification>Latency increase &lt;2x when memories increase 10x</verification>
  </criterion>

  <signatures>
    <signature name="benchmark_inject_brief">
      <code>
fn benchmark_inject_brief(c: &amp;mut Criterion)
      </code>
    </signature>
    <signature name="benchmark_inject_context">
      <code>
fn benchmark_inject_context(c: &amp;mut Criterion)
      </code>
    </signature>
    <signature name="benchmark_capture_memory">
      <code>
fn benchmark_capture_memory(c: &amp;mut Criterion)
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="timeout">Per TECH-PHASE6 timeout strategy table</constraint>
    <constraint type="measurement">Use P95 latency for safety margin</constraint>
    <constraint type="environment">Tests run on representative hardware</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-cli/benches/cli_benchmarks.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::process::Command;
use tempfile::TempDir;

/// Timeout budgets from TECH-PHASE6 (hook timeout - safety margin)
const INJECT_BRIEF_BUDGET_MS: u64 = 400;      // Hook: 500ms, Safety: 100ms
const INJECT_CONTEXT_BUDGET_MS: u64 = 4500;   // Hook: 5000ms, Safety: 500ms
const CAPTURE_MEMORY_BUDGET_MS: u64 = 2700;   // Hook: 3000ms, Safety: 300ms
const SESSION_START_BUDGET_MS: u64 = 4500;    // Hook: 5000ms, Safety: 500ms
const SESSION_END_BUDGET_MS: u64 = 28000;     // Hook: 30000ms, Safety: 2000ms

struct BenchmarkRunner {
    temp_dir: TempDir,
    db_path: String,
    session_id: Option<String>,
}

impl BenchmarkRunner {
    fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("bench_db").to_string_lossy().to_string();

        Self {
            temp_dir,
            db_path,
            session_id: None,
        }
    }

    fn setup_session(&mut self) {
        let output = Command::new("./target/release/context-graph-cli")
            .args(["session", "start", "--db-path", &self.db_path])
            .env("HOME", self.temp_dir.path())
            .output()
            .unwrap();

        self.session_id = Some(
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        );
    }

    fn seed_memories(&self, count: usize) {
        let session_id = self.session_id.as_ref().unwrap();

        for i in 0..count {
            let content = format!(
                "Benchmark memory {}: Implemented feature {} with {} components",
                i, i % 10, i % 5
            );

            Command::new("./target/release/context-graph-cli")
                .args(["capture-memory", "--source", "hook", "--content", &content, "--db-path", &self.db_path])
                .env("CLAUDE_SESSION_ID", session_id)
                .env("HOME", self.temp_dir.path())
                .output()
                .unwrap();
        }
    }

    fn run_inject_brief(&self) -> std::time::Duration {
        let session_id = self.session_id.as_ref().unwrap();
        let start = std::time::Instant::now();

        Command::new("./target/release/context-graph-cli")
            .args(["inject-brief", "--query", "benchmark query", "--db-path", &self.db_path])
            .env("CLAUDE_SESSION_ID", session_id)
            .env("HOME", self.temp_dir.path())
            .output()
            .unwrap();

        start.elapsed()
    }

    fn run_inject_context(&self) -> std::time::Duration {
        let session_id = self.session_id.as_ref().unwrap();
        let start = std::time::Instant::now();

        Command::new("./target/release/context-graph-cli")
            .args(["inject-context", "--query", "benchmark query for context injection", "--db-path", &self.db_path])
            .env("CLAUDE_SESSION_ID", session_id)
            .env("HOME", self.temp_dir.path())
            .output()
            .unwrap();

        start.elapsed()
    }

    fn run_capture_memory(&self) -> std::time::Duration {
        let session_id = self.session_id.as_ref().unwrap();
        let content = format!("Benchmark capture at {:?}", std::time::Instant::now());
        let start = std::time::Instant::now();

        Command::new("./target/release/context-graph-cli")
            .args(["capture-memory", "--source", "hook", "--content", &content, "--db-path", &self.db_path])
            .env("CLAUDE_SESSION_ID", session_id)
            .env("HOME", self.temp_dir.path())
            .output()
            .unwrap();

        start.elapsed()
    }
}

fn benchmark_inject_brief(c: &mut Criterion) {
    let mut group = c.benchmark_group("inject_brief");

    for memory_count in [0, 10, 100, 500].iter() {
        let mut runner = BenchmarkRunner::new();
        runner.setup_session();
        runner.seed_memories(*memory_count);

        group.bench_with_input(
            BenchmarkId::new("memories", memory_count),
            memory_count,
            |b, _| {
                b.iter(|| runner.run_inject_brief());
            },
        );

        // Verify within budget
        let duration = runner.run_inject_brief();
        assert!(
            duration.as_millis() < INJECT_BRIEF_BUDGET_MS as u128,
            "inject-brief with {} memories took {}ms, budget is {}ms",
            memory_count,
            duration.as_millis(),
            INJECT_BRIEF_BUDGET_MS
        );
    }

    group.finish();
}

fn benchmark_inject_context(c: &mut Criterion) {
    let mut group = c.benchmark_group("inject_context");

    for memory_count in [0, 10, 100, 500].iter() {
        let mut runner = BenchmarkRunner::new();
        runner.setup_session();
        runner.seed_memories(*memory_count);

        group.bench_with_input(
            BenchmarkId::new("memories", memory_count),
            memory_count,
            |b, _| {
                b.iter(|| runner.run_inject_context());
            },
        );

        // Verify within budget
        let duration = runner.run_inject_context();
        assert!(
            duration.as_millis() < INJECT_CONTEXT_BUDGET_MS as u128,
            "inject-context with {} memories took {}ms, budget is {}ms",
            memory_count,
            duration.as_millis(),
            INJECT_CONTEXT_BUDGET_MS
        );
    }

    group.finish();
}

fn benchmark_capture_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture_memory");

    for memory_count in [0, 10, 100, 500].iter() {
        let mut runner = BenchmarkRunner::new();
        runner.setup_session();
        runner.seed_memories(*memory_count);

        group.bench_with_input(
            BenchmarkId::new("memories", memory_count),
            memory_count,
            |b, _| {
                b.iter(|| runner.run_capture_memory());
            },
        );

        // Verify within budget
        let duration = runner.run_capture_memory();
        assert!(
            duration.as_millis() < CAPTURE_MEMORY_BUDGET_MS as u128,
            "capture-memory with {} memories took {}ms, budget is {}ms",
            memory_count,
            duration.as_millis(),
            CAPTURE_MEMORY_BUDGET_MS
        );
    }

    group.finish();
}

fn benchmark_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    // Measure latency growth as memories increase
    let memory_counts = vec![10, 100, 1000];
    let mut latencies = Vec::new();

    for memory_count in &memory_counts {
        let mut runner = BenchmarkRunner::new();
        runner.setup_session();
        runner.seed_memories(*memory_count);

        let duration = runner.run_inject_context();
        latencies.push(duration.as_millis());

        group.bench_with_input(
            BenchmarkId::new("inject_context", memory_count),
            memory_count,
            |b, _| {
                b.iter(|| runner.run_inject_context());
            },
        );
    }

    // Verify graceful degradation: 10x memories should be <2x latency increase
    if latencies.len() >= 2 {
        let ratio = latencies[1] as f64 / latencies[0] as f64;
        assert!(
            ratio < 2.0,
            "Latency should not more than 2x when memories 10x: ratio was {:.2}",
            ratio
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_inject_brief,
    benchmark_inject_context,
    benchmark_capture_memory,
    benchmark_scalability
);
criterion_main!(benches);

// Performance validation tests (not benchmarks)
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_inject_brief_timeout_compliance() {
        let mut runner = BenchmarkRunner::new();
        runner.setup_session();
        runner.seed_memories(100);

        // Run 10 iterations and check all are within budget
        for _ in 0..10 {
            let duration = runner.run_inject_brief();
            assert!(
                duration.as_millis() < INJECT_BRIEF_BUDGET_MS as u128,
                "inject-brief exceeded budget: {}ms",
                duration.as_millis()
            );
        }
    }

    #[test]
    fn test_inject_context_timeout_compliance() {
        let mut runner = BenchmarkRunner::new();
        runner.setup_session();
        runner.seed_memories(100);

        for _ in 0..5 {
            let duration = runner.run_inject_context();
            assert!(
                duration.as_millis() < INJECT_CONTEXT_BUDGET_MS as u128,
                "inject-context exceeded budget: {}ms",
                duration.as_millis()
            );
        }
    }

    #[test]
    fn test_cold_start_performance() {
        // First run after database creation (cold start)
        let mut runner = BenchmarkRunner::new();
        runner.setup_session();

        let cold_start = runner.run_inject_context();

        // Subsequent runs (warm cache)
        let warm_run = runner.run_inject_context();

        // Cold start should not be dramatically worse
        let ratio = cold_start.as_millis() as f64 / warm_run.as_millis().max(1) as f64;
        assert!(
            ratio < 5.0,
            "Cold start should not be >5x slower than warm: ratio was {:.2}",
            ratio
        );
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/benches/cli_benchmarks.rs">
    Criterion benchmarks for CLI performance
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/Cargo.toml">
    Add criterion dev-dependency and [[bench]] section
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="benchmark">cargo bench -- all benchmarks complete</criterion>
  <criterion type="test">Performance tests pass</criterion>
  <criterion type="compliance">All operations within timeout budgets</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --release --package context-graph-cli</command>
  <command>cargo bench --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli performance_tests</command>
</test_commands>
</task_spec>
```
