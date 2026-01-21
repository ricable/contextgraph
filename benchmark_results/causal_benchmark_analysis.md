# Causal Benchmark Suite - Comprehensive Analysis Report

**Generated:** 2026-01-21T18:14:00 UTC
**Benchmarks Executed:** Synthetic (causal_bench) + Real-data (causal-realdata-bench)

---

## Executive Summary

| Metric | Baseline | Target | Synthetic | Real-Data (5K) | Status |
|--------|----------|--------|-----------|----------------|--------|
| Direction Detection | 73% | >85% | **95.33%** | 21.8% | **PASS** (synthetic) |
| Asymmetry Ratio | ~1.0 | 1.2-1.5 | **1.50** | 1.00 | **PASS** (synthetic) |
| COPA Accuracy | 53% | >70% | 40.00% | 51.0% | FAIL |
| Chain Traversal | 69% | >80% | **75.26%** | N/A | PARTIAL |
| Kendall's Tau | 0.61 | >0.8 | 0.57 | N/A | FAIL |
| E5 Contribution | <5% | >5% | 3.22% | -0.7% | FAIL |

### Overall Assessment

The causal benchmark suite reveals a **significant gap between synthetic and real-world performance**. While the synthetic benchmark shows strong direction detection (95.33%) and perfect asymmetry ratio (1.50), these gains do not translate to real HuggingFace data when using simulated embeddings.

---

## Detailed Analysis

### 1. Direction Detection

**Synthetic Performance (PASS)**

| Metric | Value |
|--------|-------|
| Accuracy | 95.33% |
| Cause Precision | 1.00 |
| Cause Recall | 1.00 |
| Effect Precision | 1.00 |
| Effect Recall | 0.86 |
| Direction F1 | 0.96 |

**Confusion Matrix:**

| Actual \ Predicted | Cause | Effect | Unknown |
|-------------------|-------|--------|----------|
| Cause | 100 | 0 | 0 |
| Effect | 0 | 43 | 7 |
| Unknown | 0 | 0 | 50 |

**Finding:** The `detect_causal_query_intent()` function correctly identifies cause-seeking queries with perfect precision/recall. Effect detection has slight false-negative rate (7 misclassified as Unknown).

**Real-Data Performance (POOR)**

| Metric | Value |
|--------|-------|
| Detection Rate | 21.8% |
| Cause Detected | 47/500 |
| Effect Detected | 62/500 |
| Unknown | 391/500 |

**Finding:** Real arXiv scientific text contains mostly technical/mathematical content that doesn't match causal linguistic patterns ("because", "causes", "leads to", "results in"). The low detection rate (21.8%) reflects the domain mismatch - academic papers use formal language with few explicit causal markers.

**Recommendation:** Add domain-specific causal patterns for scientific text:
- "We hypothesize that X affects Y"
- "The mechanism by which X influences Y"
- "X is attributed to Y"
- "A consequence of X is Y"

---

### 2. Asymmetric Retrieval

**Synthetic Performance (PASS)**

| Metric | Value |
|--------|-------|
| Cause→Effect MRR | 0.9797 |
| Effect→Cause MRR | 0.9800 |
| Asymmetry Ratio | 1.50 |
| Direction Modifier Effectiveness | 1.00 |
| Intervention Overlap Correlation | 0.063 |

**Finding:** The synthetic benchmark confirms the asymmetric retrieval formula works correctly:
```
sim = base_cos × direction_mod × (0.7 + 0.3 × intervention_overlap)
```
Direction modifiers (cause→effect = 1.2, effect→cause = 0.8) produce the expected 1.5 asymmetry ratio.

**Real-Data Performance (FAIL)**

| Metric | Value |
|--------|-------|
| Cause→Effect MRR | 0.2640 |
| Effect→Cause MRR | 0.2640 |
| MRR Symmetric (E1) | 0.4596 |
| Asymmetry Ratio | 1.00 |
| Improvement over E1 | -42.6% |

**Root Cause Analysis:**

1. **Synthetic embeddings don't capture real asymmetry:** When using `--synthetic` mode, E5 embeddings are random vectors that don't reflect actual cause/effect semantics. The `as_cause` and `as_effect` vectors are identical random projections.

2. **Low direction detection rate:** With only 21.8% of queries having detected direction, most queries fall back to symmetric similarity (no direction modifier applied).

3. **E1 outperforms E5 on random vectors:** E1's simpler semantic similarity outperforms E5's asymmetric formula when the asymmetric vectors are meaningless.

**Recommendation:** Run GPU benchmark with real embeddings to get accurate asymmetry measurements. The synthetic mode is only useful for validating algorithm correctness, not measuring actual performance.

---

### 3. COPA Reasoning

**Synthetic Performance (FAIL)**

| Metric | Value |
|--------|-------|
| Overall Accuracy | 40.00% |
| Effect Questions | 45.24% |
| Cause Questions | 36.21% |

**Real-Data Performance (FAIL)**

| Metric | Value |
|--------|-------|
| E5 Asymmetric Accuracy | 51.0% |
| E1 Symmetric Accuracy | 57.0% |
| Random Baseline | 50.0% |
| Improvement over E1 | -10.5% |

**Analysis:**

Both benchmarks show COPA accuracy near random chance (50%), indicating the causal reasoning component is not working effectively. The E5 asymmetric approach actually performs *worse* than simple E1 symmetric similarity.

**Root Causes:**

1. **Synthetic COPA questions lack semantic grounding:** Generated questions may not have semantically coherent alternatives.

2. **Intervention overlap correlation is weak (0.063):** The shared interventions between cause and effect don't correlate well with actual causal relationships.

3. **E5 embeddings not trained for causal inference:** The pretrained E5 model wasn't fine-tuned on causal reasoning tasks.

**Recommendation:** Consider fine-tuning E5 on COPA-style datasets (e.g., SuperGLUE COPA, XCOPA) or using a purpose-built causal model.

---

### 4. Chain Traversal & Causal Ordering

**Synthetic Performance (PARTIAL)**

| Metric | Value | Target |
|--------|-------|--------|
| Chain Traversal | 75.26% | >80% |
| Kendall's Tau | 0.5711 | >0.8 |
| Counterfactual | 50.00% | - |

**Analysis:**

- Chain traversal at 75.26% shows the multi-hop reasoning works, but not at target level
- Kendall's tau at 0.57 indicates moderate-good but not excellent causal ordering
- Counterfactual reasoning at 50% (random chance) suggests this component needs work

**Recommendation:** Implement temporal decay/attenuation for longer chains. Current implementation may not properly weight chain length.

---

### 5. E5 Contribution Ablation

**Synthetic Ablation Results:**

| Configuration | Score | vs Baseline |
|---------------|-------|-------------|
| Symmetric Baseline | 1.000 | - |
| Direction Modifiers Only | 1.000 | 0.00% |
| Intervention Overlap Only | 0.841 | -15.9% |
| Full E5 | 0.826 | -17.4% |
| Without E5 (E1 only) | 0.800 | - |
| **E5 Contribution** | **3.22%** | - |

**Real-Data E5 Contribution:**

| Metric | Value |
|--------|-------|
| MRR with E5 | 0.4564 |
| MRR without E5 | 0.4596 |
| E5 Contribution | -0.7% |

**Critical Finding:**

E5's contribution is marginal (3.22% synthetic) to negative (-0.7% real-data). The asymmetric embedding approach is not providing meaningful lift over symmetric E1 similarity.

**Root Causes:**

1. **Intervention overlap hurts more than helps:** Adding intervention overlap reduces performance by 15.9%
2. **Direction modifiers are neutral:** They don't improve or hurt when applied correctly
3. **E5 weights in fusion may be too low:** The 15% E5 weight in the standard profile may not be sufficient

**Recommendations:**

1. Increase E5 weight in fusion formula from 15% to 25-30%
2. Re-evaluate intervention overlap calculation - currently near-zero correlation
3. Consider E5-only mode for explicitly causal queries

---

## Feature Contribution Breakdown

From synthetic benchmark:

| Feature | Contribution |
|---------|-------------|
| Direction Detection | 0.958 |
| Asymmetric Similarity | 0.990 |
| Intervention Overlap | 0.063 |
| Causal Reasoning | 0.600 |

**Interpretation:**
- Direction detection and asymmetric similarity are working well
- Intervention overlap is nearly useless (0.063)
- Causal reasoning (COPA/chains) needs significant improvement

---

## Performance Timing

| Benchmark | Duration |
|-----------|----------|
| Synthetic (500 pairs) | 236ms |
| Real-data (5000 chunks, synthetic embeddings) | 2.41s |

Embedding throughput: 2072 chunks/sec with synthetic embeddings.

---

## Comparison: Synthetic vs Previous Real-Data Run

A prior smaller real-data run (100 chunks) showed:

| Metric | 100 Chunks | 5000 Chunks | Trend |
|--------|------------|-------------|-------|
| Detection Rate | 30.0% | 21.8% | Worse |
| MRR Cause→Effect | 0.823 | 0.264 | Much worse |
| MRR E1 | 0.756 | 0.460 | Worse |
| E5 Contribution | 4.9% | -0.7% | Much worse |

**Analysis:** Performance degrades significantly as dataset size increases. This suggests:
1. Overfitting on small samples gives artificially good results
2. The model doesn't generalize to diverse scientific text
3. More data exposes weaknesses in the approach

---

## Recommendations Summary

### High Priority

1. **Run GPU benchmark** with real embeddings to measure true asymmetry
2. **Add scientific causal patterns** to direction detection
3. **Increase E5 weight** in fusion formula from 15% to 25%

### Medium Priority

4. **Remove intervention overlap** from formula (currently hurts performance)
5. **Fine-tune E5** on causal reasoning datasets (COPA, XCOPA)
6. **Implement proper chain attenuation** for multi-hop reasoning

### Low Priority

7. Add counterfactual reasoning training data
8. Investigate why E5 contribution is negative on real data
9. Consider domain-specific E5 variants (E5-science, E5-code)

---

## Conclusion

The causal benchmark suite validates that the algorithmic components (direction detection, asymmetric formula) work correctly on synthetic data. However, **real-world performance is significantly below targets**, primarily due to:

1. Domain mismatch between causal patterns and scientific text
2. Synthetic embeddings not capturing true asymmetric semantics
3. Intervention overlap providing negative contribution
4. E5 weights too low in fusion formula

**Next Steps:** Run the GPU benchmark (`--features real-embeddings`) to measure performance with actual E5 asymmetric embeddings. This will provide accurate asymmetry ratios and reveal whether the approach can work on real data with proper embeddings.

---

*Report generated by E5 Causal Benchmark Suite*
