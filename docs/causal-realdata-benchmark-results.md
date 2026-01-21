# E5 Causal Benchmark with Real HuggingFace Data

**Generated:** 2026-01-21T17:58:04.224424705+00:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Chunks | 100 |
| Documents | 1250 |
| Topics | 11 |
| E5 Coverage | 100.0% |
| Asymmetric E5 | true |

## Direction Detection

| Metric | Value |
|--------|-------|
| Total Samples | 50 |
| Cause Detected | 8 |
| Effect Detected | 7 |
| Detection Rate | 30.0% |

## Asymmetric Retrieval

| Metric | Value | Target |
|--------|-------|--------|
| MRR Cause→Effect | 0.8228 | - |
| MRR Effect→Cause | 0.8228 | - |
| MRR Symmetric (E1) | 0.7559 | - |
| **Asymmetry Ratio** | **1.00** | ~1.5 |
| Improvement over E1 | 8.9% | >0% |

## COPA-Style Reasoning

| Metric | Value | Target |
|--------|-------|--------|
| **E5 Asymmetric Accuracy** | **40.0%** | >70% |
| E1 Symmetric Accuracy | 95.0% | - |
| Random Baseline | 50.0% | - |
| Improvement over E1 | -57.9% | >0% |

## E5 Contribution Analysis

| Metric | Value | Target |
|--------|-------|--------|
| MRR with E5 | 0.8866 | - |
| MRR without E5 | 0.8449 | - |
| **E5 Contribution** | **4.9%** | >5% |

## Recommendations

- Asymmetry ratio below target (1.5). Consider tuning E5 direction modifiers.
- COPA accuracy below 70%. E5 embedder may need domain-specific fine-tuning.
- E5 contribution below 5%. Consider increasing E5 weight in fusion formula.

