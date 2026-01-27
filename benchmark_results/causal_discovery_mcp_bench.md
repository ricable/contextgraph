# Causal Discovery MCP Tool Benchmark Report

**Generated:** 2026-01-27T12:25:05.786958407+00:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Max Memories | 20 |
| Max Pairs | 10 |
| Min Confidence | 0.7 |
| Iterations | 2 |

## trigger_causal_discovery Performance

| Metric | Dry Run | Full Execution | Target |
|--------|---------|----------------|--------|
| Avg Latency | 0.0ms | 0.0ms | <5000ms |
| P95 Latency | 0.0ms | 0.0ms | - |

## get_causal_discovery_status Performance

| Metric | Value | Target |
|--------|-------|--------|
| Avg Latency | 0.0ms | <10ms |
| P95 Latency | 0.0ms | - |
| Response Size | 390 bytes | - |

## Performance Targets

| Target | Status |
|--------|--------|
| Dry Run < 50ms | PASS |
| Get Status < 10ms | PASS |
| Full Execution < 5000ms | PASS |
| LLM Inference < 500ms/pair | PASS |

## Recommendations

- All performance targets met!
