# E5 Causal Benchmark with Real HuggingFace Data

**Generated:** 2026-01-21T19:22:55.250113204+00:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Chunks | 2000 |
| Documents | 1250 |
| Topics | 11 |
| E5 Coverage | 100.0% |
| Asymmetric E5 | true |

## Direction Detection

| Metric | Value |
|--------|-------|
| Total Samples | 300 |
| Cause Detected | 31 |
| Effect Detected | 49 |
| Detection Rate | 26.7% |

## Asymmetric Retrieval

| Metric | Value | Target |
|--------|-------|--------|
| MRR Cause→Effect | 0.9783 | - |
| MRR Effect→Cause | 0.9783 | - |
| MRR Symmetric (E1) | 1.0000 | - |
| **Asymmetry Ratio** | **1.00** | ~1.5 |
| Improvement over E1 | -2.2% | >0% |

## COPA-Style Reasoning

| Metric | Value | Target |
|--------|-------|--------|
| **E5 Asymmetric Accuracy** | **72.0%** | >70% |
| E1 Symmetric Accuracy | 96.0% | - |
| Random Baseline | 50.0% | - |
| Improvement over E1 | -25.0% | >0% |

## E5 Contribution Analysis

| Metric | Value | Target |
|--------|-------|--------|
| MRR with E5 | 1.0000 | - |
| MRR without E5 | 1.0000 | - |
| **E5 Contribution** | **0.0%** | >5% |

## Recommendations

- Asymmetry ratio below target (1.5). Consider tuning E5 direction modifiers.
- E5 contribution below 5%. Consider increasing E5 weight in fusion formula.
- E5 asymmetric performing worse than E1 symmetric. Check E5 embedding quality.

