# E5 Causal Benchmark with Real HuggingFace Data

**Generated:** 2026-01-21T18:13:42.209837231+00:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Chunks | 5000 |
| Documents | 1250 |
| Topics | 11 |
| E5 Coverage | 100.0% |
| Asymmetric E5 | true |

## Direction Detection

| Metric | Value |
|--------|-------|
| Total Samples | 500 |
| Cause Detected | 47 |
| Effect Detected | 62 |
| Detection Rate | 21.8% |

## Asymmetric Retrieval

| Metric | Value | Target |
|--------|-------|--------|
| MRR Cause→Effect | 0.2640 | - |
| MRR Effect→Cause | 0.2640 | - |
| MRR Symmetric (E1) | 0.4596 | - |
| **Asymmetry Ratio** | **1.00** | ~1.5 |
| Improvement over E1 | -42.6% | >0% |

## COPA-Style Reasoning

| Metric | Value | Target |
|--------|-------|--------|
| **E5 Asymmetric Accuracy** | **51.0%** | >70% |
| E1 Symmetric Accuracy | 57.0% | - |
| Random Baseline | 50.0% | - |
| Improvement over E1 | -10.5% | >0% |

## E5 Contribution Analysis

| Metric | Value | Target |
|--------|-------|--------|
| MRR with E5 | 0.4564 | - |
| MRR without E5 | 0.4596 | - |
| **E5 Contribution** | **-0.7%** | >5% |

## Recommendations

- Asymmetry ratio below target (1.5). Consider tuning E5 direction modifiers.
- COPA accuracy below 70%. E5 embedder may need domain-specific fine-tuning.
- E5 contribution below 5%. Consider increasing E5 weight in fusion formula.
- E5 asymmetric performing worse than E1 symmetric. Check E5 embedding quality.

