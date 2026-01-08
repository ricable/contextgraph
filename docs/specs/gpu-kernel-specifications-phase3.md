# GPU Kernel Implementation Specifications - Phase 3

## Document Metadata
- **Agent**: P3-2 (Architecture)
- **Phase**: 3 - Planning
- **Date**: 2026-01-07
- **Dependencies**: P3-1 (Infrastructure Investigation)

## Executive Summary

This document specifies implementation details for four missing GPU kernels required by the context-graph-cuda crate:

| Kernel | Purpose | Target Latency | Priority |
|--------|---------|----------------|----------|
| hnsw.cu | GPU-accelerated HNSW graph traversal | <5ms for 100K search | 1 (Critical) |
| hopfield.cu | Modern Hopfield pattern retrieval | <1ms per retrieval | 2 (High) |
| kuramoto.cu | Oscillator phase synchronization | <1ms per step | 3 (Medium) |
| neuromod.cu | Neurotransmitter weight updates | <200us per update | 4 (High) |

## Common Architecture Patterns

### Target Hardware
- **GPU**: RTX 5090 (Compute Capability 12.0)
- **CUDA**: 13.1+
- **Architecture**: Blackwell (170 SMs, 21760 CUDA cores)
- **Memory**: 32GB GDDR7, 1.8 TB/s bandwidth

### Standard Block Configuration
```c
constexpr int BLOCK_DIM_X = 32;  // Warp size for coalesced access
constexpr int BLOCK_DIM_Y = 8;   // Typical tile size
// Block: 32 x 8 = 256 threads (optimal for Blackwell)
```

### FFI Pattern (Established)
```c
extern "C" int launch_<kernel_name>(
    /* device pointers */,
    /* dimensions */,
    void* stream  // cudaStream_t
);

extern "C" void get_<kernel_name>_config(
    int* block_dim_x,
    int* block_dim_y,
    /* kernel-specific params */
);
```

### Rust FFI Integration
```rust
#[link(name = "<kernel_name>", kind = "static")]
extern "C" {
    pub fn launch_<kernel_name>(...) -> c_int;
    pub fn get_<kernel_name>_config(...);
}
```

---

## KERNEL 1: hnsw.cu - GPU Graph Traversal

### Purpose
GPU-accelerated HNSW approximate nearest neighbor search for the 12 HNSW indexes (E1-E5, E7-E11, E1Matryoshka128, PurposeVector).

### Constitution Reference
- `perf.latency.faiss_1M_k100`: <2ms
- `perf.latency.hnsw_search`: <5ms for 100K vectors

### Algorithm Overview
The GPU HNSW search uses a parallel beam search strategy:
1. Multiple entry points explored in parallel
2. Neighbor expansion with coalesced memory access
3. GPU-side priority queue for candidate management
4. Distance computation vectorized across warps

### Interface

```c
/**
 * GPU-accelerated HNSW search.
 *
 * @param d_query        Query vector(s) [n_queries][dim] - device memory
 * @param d_vectors      Database vectors [n_vectors][dim] - device memory
 * @param d_neighbors    CSR neighbor adjacency: offsets[n_vectors+1], indices[]
 * @param d_offsets      CSR row offsets [n_vectors + 1]
 * @param d_results      Output: k nearest neighbor indices [n_queries][k]
 * @param d_distances    Output: k nearest distances [n_queries][k]
 * @param n_queries      Number of query vectors
 * @param n_vectors      Number of database vectors
 * @param dim            Vector dimension (128, 256, 384, 512, 768, 1024, 10000)
 * @param k              Number of nearest neighbors to return
 * @param ef_search      Search expansion factor (typically 2*k to 10*k)
 * @param max_level      Maximum layer in HNSW graph
 * @param entry_point    Entry point node index
 * @param stream         CUDA stream
 * @return               0 on success, CUDA error code on failure
 */
extern "C" int launch_hnsw_search(
    const float* d_query,
    const float* d_vectors,
    const int* d_neighbors,
    const int* d_offsets,
    int* d_results,
    float* d_distances,
    int n_queries,
    int n_vectors,
    int dim,
    int k,
    int ef_search,
    int max_level,
    int entry_point,
    void* stream
);

/**
 * Batch HNSW insert (rebuild neighbors on GPU).
 * Used during index construction.
 */
extern "C" int launch_hnsw_insert_batch(
    const float* d_new_vectors,    // [n_new][dim]
    float* d_vectors,              // [n_existing + n_new][dim]
    int* d_neighbors,              // Updated neighbor lists
    int* d_offsets,                // Updated CSR offsets
    int n_new,
    int n_existing,
    int dim,
    int m,                         // Max neighbors per node
    int ef_construction,
    void* stream
);
```

### Kernel Design

```c
// Block configuration for HNSW traversal
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;  // 256

// Each warp processes one query
// Threads within warp collaboratively compute distances to neighbors
__global__ void hnsw_search_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ vectors,
    const int* __restrict__ neighbors,
    const int* __restrict__ offsets,
    int* __restrict__ results,
    float* __restrict__ distances,
    int n_queries, int n_vectors, int dim,
    int k, int ef_search, int entry_point
) {
    // Shared memory for candidate list per warp
    __shared__ int s_candidates[WARPS_PER_BLOCK][MAX_EF];
    __shared__ float s_distances[WARPS_PER_BLOCK][MAX_EF];
    __shared__ int s_candidate_count[WARPS_PER_BLOCK];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (query_idx >= n_queries) return;

    // Load query into registers (distributed across lanes)
    float query_reg[MAX_DIM_PER_LANE];  // dim/32 elements per lane
    load_query_distributed(queries + query_idx * dim, query_reg, dim, lane_id);

    // Initialize with entry point
    if (lane_id == 0) {
        s_candidates[warp_id][0] = entry_point;
        s_candidate_count[warp_id] = 1;
    }
    __syncwarp();

    // Greedy search: expand best candidate, insert better neighbors
    int visited_count = 0;
    while (/* candidates to explore */) {
        // Get best unvisited candidate
        int current = pop_best_candidate(s_candidates, s_distances,
                                          s_candidate_count, warp_id);

        // Load neighbors (coalesced)
        int n_start = offsets[current];
        int n_end = offsets[current + 1];

        // Compute distances to all neighbors (parallel across warp)
        for (int n = n_start + lane_id; n < n_end; n += WARP_SIZE) {
            int neighbor_id = neighbors[n];
            float dist = compute_distance_distributed(
                query_reg, vectors + neighbor_id * dim, dim
            );

            // Insert into candidate list if better
            try_insert_candidate(s_candidates, s_distances,
                                  s_candidate_count, neighbor_id, dist,
                                  ef_search, warp_id, lane_id);
        }
    }

    // Write top-k results
    write_topk_results(results + query_idx * k, distances + query_idx * k,
                       s_candidates, s_distances, k, warp_id, lane_id);
}
```

### Memory Layout

```
Database Vectors (row-major, 128-byte aligned):
+------------------+------------------+-----+
| vec[0][0..dim-1] | vec[1][0..dim-1] | ... |
+------------------+------------------+-----+

Neighbor Lists (CSR format):
offsets:  [0, 32, 64, 80, ...]  // Start index for each node's neighbors
indices:  [5, 12, 3, ..., 7, 2, 9, ..., ...]  // Actual neighbor IDs

Query/Result Layout:
queries:   [n_queries][dim]
results:   [n_queries][k]     // Neighbor indices
distances: [n_queries][k]     // Corresponding distances
```

### Shared Memory Usage
```
Per block (8 warps):
- Candidates:  8 * 256 * 4 = 8KB (ef_search=256 max)
- Distances:   8 * 256 * 4 = 8KB
- Counters:    8 * 4 = 32 bytes
Total: ~16KB per block
```

### Performance Target
- 100K vectors, dim=1024, k=100, ef=200: **<5ms**
- 1M vectors, dim=1024, k=100, ef=200: **<20ms**

### Rust FFI Integration

```rust
// In crates/context-graph-cuda/src/hnsw.rs

#[link(name = "hnsw", kind = "static")]
extern "C" {
    pub fn launch_hnsw_search(
        d_query: *const f32,
        d_vectors: *const f32,
        d_neighbors: *const i32,
        d_offsets: *const i32,
        d_results: *mut i32,
        d_distances: *mut f32,
        n_queries: i32,
        n_vectors: i32,
        dim: i32,
        k: i32,
        ef_search: i32,
        max_level: i32,
        entry_point: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// GPU HNSW search configuration
pub struct HnswGpuConfig {
    pub dim: usize,
    pub k: usize,
    pub ef_search: usize,
    pub max_level: usize,
}
```

### Build Integration

```rust
// In build.rs, add:
compile_kernel("kernels/hnsw.cu", "hnsw");
```

---

## KERNEL 2: hopfield.cu - Modern Hopfield Network

### Purpose
GPU-accelerated pattern retrieval using Modern Hopfield Networks with exponential capacity. Used in L3_Memory layer for associative memory.

### Constitution Reference
- `perf.latency.hopfield`: <1ms
- `layers.L3_Memory.capacity`: 2^768 patterns
- `memory_math.hopfield.formula`: E = -Sum_i log(Sum_j exp(x_i^T * xi_j))
- `neuromod.Dopamine.param`: hopfield.beta (sharpness)

### Algorithm Overview

Modern Hopfield retrieval:
1. Compute attention scores: `a_j = exp(beta * x^T * xi_j)`
2. Normalize: `alpha_j = a_j / Sum_k a_k`
3. Retrieve: `y = Sum_j alpha_j * xi_j`

The key insight is that higher beta leads to sharper retrieval (more similar to nearest neighbor).

### Interface

```c
/**
 * Modern Hopfield pattern retrieval.
 *
 * Retrieves patterns from stored memory using attention-like mechanism.
 * beta controls sharpness: higher = more like 1-NN, lower = more averaging.
 *
 * @param d_queries      Query vectors [n_queries][dim]
 * @param d_patterns     Stored patterns [n_patterns][dim]
 * @param d_values       Values associated with patterns [n_patterns][value_dim]
 *                       (if NULL, returns weighted sum of patterns themselves)
 * @param d_output       Retrieved output [n_queries][value_dim or dim]
 * @param d_attention    Optional: attention weights [n_queries][n_patterns]
 * @param n_queries      Number of queries
 * @param n_patterns     Number of stored patterns
 * @param dim            Pattern dimension
 * @param value_dim      Value dimension (0 = use patterns as values)
 * @param beta           Inverse temperature (sharpness), range [1, 5]
 * @param stream         CUDA stream
 * @return               0 on success, CUDA error code on failure
 */
extern "C" int launch_hopfield_retrieve(
    const float* d_queries,
    const float* d_patterns,
    const float* d_values,        // nullable
    float* d_output,
    float* d_attention,           // nullable, for debugging
    int n_queries,
    int n_patterns,
    int dim,
    int value_dim,
    float beta,
    void* stream
);

/**
 * Hopfield energy computation (for monitoring/debugging).
 * E = -Sum_i log(Sum_j exp(x_i^T * xi_j))
 */
extern "C" int launch_hopfield_energy(
    const float* d_queries,
    const float* d_patterns,
    float* d_energies,            // [n_queries]
    int n_queries,
    int n_patterns,
    int dim,
    float beta,
    void* stream
);

/**
 * Hopfield pattern storage update (for learning).
 * Updates pattern matrix using self-referential rule:
 * M = M(alpha*I - eta*k*k^T) + eta*v_hat*k^T
 */
extern "C" int launch_hopfield_update(
    float* d_patterns,            // [n_patterns][dim] - updated in place
    const float* d_new_pattern,   // [1][dim] - new pattern to store
    const float* d_new_value,     // [1][value_dim] - associated value (nullable)
    int n_patterns,
    int dim,
    int value_dim,
    float alpha,                  // Retention factor
    float eta,                    // Learning rate
    void* stream
);
```

### Kernel Design

```c
// Hopfield retrieval kernel
// Uses softmax-like attention over stored patterns

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_PATTERNS_SHARED = 32;  // Cache patterns in shared mem

__global__ void hopfield_retrieve_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ patterns,
    const float* __restrict__ values,      // nullable
    float* __restrict__ output,
    float* __restrict__ attention,         // nullable
    int n_queries, int n_patterns, int dim, int value_dim,
    float beta
) {
    // Shared memory for partial sums and pattern caching
    __shared__ float s_max[BLOCK_SIZE / WARP_SIZE];  // For numerically stable softmax
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_pattern_cache[MAX_PATTERNS_SHARED][MAX_DIM];

    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (query_idx >= n_queries) return;

    // Load query into registers
    float query_reg[MAX_DIM / BLOCK_SIZE + 1];
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        query_reg[d / BLOCK_SIZE] = queries[query_idx * dim + d];
    }

    // Phase 1: Compute max dot product (for numerical stability)
    float local_max = -INFINITY;
    for (int p = 0; p < n_patterns; p++) {
        float dot = 0.0f;
        for (int d = tid; d < dim; d += BLOCK_SIZE) {
            dot += query_reg[d / BLOCK_SIZE] * patterns[p * dim + d];
        }
        // Warp reduction
        dot = warp_reduce_sum(dot);
        if (tid % WARP_SIZE == 0) {
            local_max = fmaxf(local_max, beta * dot);
        }
    }
    // Block reduction for max
    float global_max = block_reduce_max(local_max, s_max);

    // Phase 2: Compute sum of exp(beta * dot - max) and accumulate weighted output
    float local_sum = 0.0f;
    float output_accum[MAX_OUTPUT_DIM / BLOCK_SIZE + 1] = {0};

    for (int p = 0; p < n_patterns; p++) {
        // Compute dot product
        float dot = 0.0f;
        for (int d = tid; d < dim; d += BLOCK_SIZE) {
            dot += query_reg[d / BLOCK_SIZE] * patterns[p * dim + d];
        }
        dot = warp_reduce_sum(dot);

        // Compute exp(beta * dot - max)
        float score = __expf(beta * dot - global_max);
        local_sum += score;

        // Accumulate weighted value/pattern
        const float* value_ptr = (values != nullptr) ?
            values + p * value_dim : patterns + p * dim;
        int out_dim = (values != nullptr) ? value_dim : dim;

        for (int d = tid; d < out_dim; d += BLOCK_SIZE) {
            output_accum[d / BLOCK_SIZE] += score * value_ptr[d];
        }

        // Store attention weight if requested
        if (attention != nullptr && tid == 0) {
            attention[query_idx * n_patterns + p] = score;  // Will normalize later
        }
    }

    // Normalize output by sum
    float global_sum = block_reduce_sum(local_sum, s_sum);
    float inv_sum = 1.0f / (global_sum + EPS);

    int out_dim = (values != nullptr) ? value_dim : dim;
    for (int d = tid; d < out_dim; d += BLOCK_SIZE) {
        output[query_idx * out_dim + d] = output_accum[d / BLOCK_SIZE] * inv_sum;
    }

    // Normalize attention weights if needed
    if (attention != nullptr && tid == 0) {
        for (int p = 0; p < n_patterns; p++) {
            attention[query_idx * n_patterns + p] *= inv_sum;
        }
    }
}
```

### Memory Layout

```
Patterns (stored memories):
[n_patterns][dim] - row-major, 128-byte aligned

Values (associated with patterns):
[n_patterns][value_dim] - row-major, or NULL to use patterns

Query/Output:
queries: [n_queries][dim]
output:  [n_queries][value_dim or dim]

Attention (optional debug output):
[n_queries][n_patterns] - attention weights
```

### Performance Target
- 1000 patterns, dim=1024: **<1ms**
- 10000 patterns, dim=1024: **<5ms**

### Neuromodulation Integration
The `beta` parameter is controlled by Dopamine:
- Low dopamine (beta=1): Fuzzy, averaging retrieval
- High dopamine (beta=5): Sharp, near-exact retrieval

---

## KERNEL 3: kuramoto.cu - Oscillator Phase Updates

### Purpose
GPU-accelerated Kuramoto oscillator network for 13-embedding phase synchronization. Used in L5_Coherence layer for Global Workspace consciousness model.

### Constitution Reference
- `gwt.kuramoto`: Phase coupling formula
- `perf.latency.gwt_cycle`: <10ms (includes Kuramoto)
- Target: <1ms for phase update

### Algorithm

The Kuramoto model:
```
dtheta_i/dt = omega_i + (K/N) * Sum_j sin(theta_j - theta_i)
```

Order parameter (synchronization measure):
```
r * exp(i*psi) = (1/N) * Sum_j exp(i*theta_j)
```

### Interface

```c
/**
 * Kuramoto oscillator phase update.
 *
 * Updates phases of N oscillators using Kuramoto dynamics.
 * Uses 4th-order Runge-Kutta for numerical stability.
 *
 * @param d_phases           Current phases [n_oscillators] (updated in place)
 * @param d_natural_freqs    Natural frequencies [n_oscillators]
 * @param d_coupling_matrix  Optional: non-uniform coupling [n_osc][n_osc], NULL for uniform K
 * @param n_oscillators      Number of oscillators (13 for embeddings)
 * @param coupling_strength  Global coupling K (used if coupling_matrix is NULL)
 * @param dt                 Time step in seconds
 * @param stream             CUDA stream
 * @return                   0 on success
 */
extern "C" int launch_kuramoto_step(
    float* d_phases,
    const float* d_natural_freqs,
    const float* d_coupling_matrix,   // nullable
    int n_oscillators,
    float coupling_strength,
    float dt,
    void* stream
);

/**
 * Compute Kuramoto order parameter.
 *
 * @param d_phases       Oscillator phases [n_oscillators]
 * @param d_order_r      Output: order parameter magnitude [1]
 * @param d_order_psi    Output: mean phase [1]
 * @param n_oscillators  Number of oscillators
 * @param stream         CUDA stream
 * @return               0 on success
 */
extern "C" int launch_kuramoto_order_parameter(
    const float* d_phases,
    float* d_order_r,
    float* d_order_psi,
    int n_oscillators,
    void* stream
);

/**
 * Batch Kuramoto simulation (multiple steps).
 * More efficient than calling step() repeatedly.
 *
 * @param d_phases           Initial phases [n_oscillators]
 * @param d_phase_history    Output: phase history [n_steps][n_oscillators]
 * @param d_order_history    Output: order parameter history [n_steps][2] (r, psi)
 * @param d_natural_freqs    Natural frequencies [n_oscillators]
 * @param n_oscillators      Number of oscillators
 * @param n_steps            Number of simulation steps
 * @param coupling_strength  Global coupling K
 * @param dt                 Time step
 * @param stream             CUDA stream
 * @return                   0 on success
 */
extern "C" int launch_kuramoto_simulate(
    float* d_phases,
    float* d_phase_history,
    float* d_order_history,
    const float* d_natural_freqs,
    int n_oscillators,
    int n_steps,
    float coupling_strength,
    float dt,
    void* stream
);
```

### Kernel Design

For 13 oscillators, a single warp is sufficient:

```c
// Kuramoto kernel for small N (N <= 32)
// Entire computation fits in a single warp

__global__ void kuramoto_step_kernel(
    float* phases,
    const float* natural_freqs,
    const float* coupling_matrix,      // nullable
    int n_oscillators,
    float K,
    float dt
) {
    const int tid = threadIdx.x;

    if (tid >= n_oscillators) return;

    // Load phase into register
    float phase = phases[tid];
    float omega = natural_freqs[tid];

    // RK4 integration
    float k1, k2, k3, k4;

    // k1 = f(t, y)
    k1 = compute_dphase(phase, phases, coupling_matrix, n_oscillators, K, omega, tid);

    // k2 = f(t + dt/2, y + k1*dt/2)
    float phase_mid1 = phase + k1 * dt * 0.5f;
    k2 = compute_dphase(phase_mid1, phases, coupling_matrix, n_oscillators, K, omega, tid);

    // k3 = f(t + dt/2, y + k2*dt/2)
    float phase_mid2 = phase + k2 * dt * 0.5f;
    k3 = compute_dphase(phase_mid2, phases, coupling_matrix, n_oscillators, K, omega, tid);

    // k4 = f(t + dt, y + k3*dt)
    float phase_end = phase + k3 * dt;
    k4 = compute_dphase(phase_end, phases, coupling_matrix, n_oscillators, K, omega, tid);

    // Update: y_new = y + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    float new_phase = phase + (k1 + 2.0f*k2 + 2.0f*k3 + k4) * dt / 6.0f;

    // Wrap to [0, 2*pi]
    new_phase = fmodf(new_phase + 2.0f * M_PI, 2.0f * M_PI);

    phases[tid] = new_phase;
}

__device__ float compute_dphase(
    float my_phase,
    const float* all_phases,
    const float* coupling_matrix,
    int n, float K, float omega, int my_idx
) {
    float coupling_sum = 0.0f;

    for (int j = 0; j < n; j++) {
        if (j == my_idx) continue;

        float other_phase = all_phases[j];
        float coupling = (coupling_matrix != nullptr) ?
            coupling_matrix[my_idx * n + j] : K;

        coupling_sum += coupling * sinf(other_phase - my_phase);
    }

    return omega + coupling_sum / (float)n;
}

// Order parameter kernel
__global__ void kuramoto_order_kernel(
    const float* phases,
    float* order_r,
    float* order_psi,
    int n_oscillators
) {
    __shared__ float s_cos_sum;
    __shared__ float s_sin_sum;

    const int tid = threadIdx.x;

    // Each thread handles one oscillator
    float cos_val = (tid < n_oscillators) ? cosf(phases[tid]) : 0.0f;
    float sin_val = (tid < n_oscillators) ? sinf(phases[tid]) : 0.0f;

    // Warp reduction
    cos_val = warp_reduce_sum(cos_val);
    sin_val = warp_reduce_sum(sin_val);

    if (tid == 0) {
        float n = (float)n_oscillators;
        float avg_cos = cos_val / n;
        float avg_sin = sin_val / n;

        // r = sqrt(avg_cos^2 + avg_sin^2)
        *order_r = sqrtf(avg_cos * avg_cos + avg_sin * avg_sin);

        // psi = atan2(avg_sin, avg_cos)
        *order_psi = atan2f(avg_sin, avg_cos);
    }
}
```

### Memory Layout

```
Phases: [n_oscillators] = [13] floats (52 bytes)
Natural frequencies: [13] floats (52 bytes)
Coupling matrix (optional): [13][13] = 169 floats (676 bytes)

Phase history (simulation): [n_steps][13]
Order history: [n_steps][2] (r, psi per step)
```

### Performance Target
- 13 oscillators, single step: **<10us**
- 13 oscillators, 1000 steps: **<1ms**

### Consciousness State Detection
- r >= 0.8: CONSCIOUS (synchronized)
- 0.5 <= r < 0.8: EMERGING
- r < 0.5: FRAGMENTED

---

## KERNEL 4: neuromod.cu - Neurotransmitter Modulation

### Purpose
GPU-accelerated neurotransmitter weight updates for modulating Hopfield beta, similarity weights, attention temperature, and learning rate.

### Constitution Reference
- `perf.latency.neuromod_update`: <200us
- `neuromod.Dopamine`: hopfield.beta [1,5]
- `neuromod.Serotonin`: similarity.space_weights [0,1]
- `neuromod.Noradrenaline`: attention.temp [0.5,2]
- `neuromod.Acetylcholine`: utl.lr [0.001,0.002]

### Algorithm

Neurotransmitter dynamics (simplified Marblestone model):
```
d[NT]/dt = release_rate * stimulation - decay_rate * [NT]
```

Effect mapping:
```
hopfield_beta = 1.0 + 4.0 * sigmoid(dopamine_level - 0.5)
space_weights[i] = base_weight[i] * serotonin_level
attention_temp = 0.5 + 1.5 * noradrenaline_level
learning_rate = 0.001 + 0.001 * acetylcholine_level
```

### Interface

```c
/**
 * Update neurotransmitter levels based on stimulation.
 *
 * @param d_nt_levels       Current NT levels [4] (DA, 5HT, NE, ACh)
 * @param d_stimulation     Stimulation inputs [4]
 * @param d_release_rates   Release rate constants [4]
 * @param d_decay_rates     Decay rate constants [4]
 * @param dt                Time step
 * @param stream            CUDA stream
 * @return                  0 on success
 */
extern "C" int launch_neuromod_update(
    float* d_nt_levels,
    const float* d_stimulation,
    const float* d_release_rates,
    const float* d_decay_rates,
    float dt,
    void* stream
);

/**
 * Compute derived parameters from NT levels.
 *
 * @param d_nt_levels           Current NT levels [4]
 * @param d_hopfield_beta       Output: Hopfield beta
 * @param d_space_weights       Output: 13 space weights [13]
 * @param d_base_space_weights  Input: base weights [13]
 * @param d_attention_temp      Output: attention temperature
 * @param d_learning_rate       Output: UTL learning rate
 * @param stream                CUDA stream
 * @return                      0 on success
 */
extern "C" int launch_neuromod_compute_params(
    const float* d_nt_levels,
    float* d_hopfield_beta,
    float* d_space_weights,
    const float* d_base_space_weights,
    float* d_attention_temp,
    float* d_learning_rate,
    void* stream
);

/**
 * Batch update: combines update + compute in one kernel.
 * More efficient for typical usage pattern.
 *
 * @param d_nt_levels           NT levels [4] (updated in place)
 * @param d_stimulation         Stimulation inputs [4]
 * @param d_release_rates       Release rates [4]
 * @param d_decay_rates         Decay rates [4]
 * @param d_hopfield_beta       Output: beta
 * @param d_space_weights       Output: weights [13]
 * @param d_base_space_weights  Input: base weights [13]
 * @param d_attention_temp      Output: temp
 * @param d_learning_rate       Output: lr
 * @param dt                    Time step
 * @param stream                CUDA stream
 * @return                      0 on success
 */
extern "C" int launch_neuromod_batch_update(
    float* d_nt_levels,
    const float* d_stimulation,
    const float* d_release_rates,
    const float* d_decay_rates,
    float* d_hopfield_beta,
    float* d_space_weights,
    const float* d_base_space_weights,
    float* d_attention_temp,
    float* d_learning_rate,
    float dt,
    void* stream
);

/**
 * Reward-based NT update (for RL integration).
 *
 * Dopamine specifically responds to reward prediction error:
 * delta_DA = reward - expected_reward
 *
 * @param d_nt_levels      NT levels [4]
 * @param reward           Actual reward
 * @param expected_reward  Predicted reward
 * @param stream           CUDA stream
 * @return                 0 on success
 */
extern "C" int launch_neuromod_reward_update(
    float* d_nt_levels,
    float reward,
    float expected_reward,
    void* stream
);
```

### Kernel Design

```c
// Neurotransmitter indices
constexpr int NT_DOPAMINE = 0;
constexpr int NT_SEROTONIN = 1;
constexpr int NT_NORADRENALINE = 2;
constexpr int NT_ACETYLCHOLINE = 3;
constexpr int NUM_NT = 4;

// Number of embedding spaces for weight modulation
constexpr int NUM_SPACES = 13;

__global__ void neuromod_batch_kernel(
    float* nt_levels,
    const float* stimulation,
    const float* release_rates,
    const float* decay_rates,
    float* hopfield_beta,
    float* space_weights,
    const float* base_space_weights,
    float* attention_temp,
    float* learning_rate,
    float dt
) {
    const int tid = threadIdx.x;

    // Phase 1: Update NT levels (4 threads)
    if (tid < NUM_NT) {
        float level = nt_levels[tid];
        float stim = stimulation[tid];
        float release = release_rates[tid];
        float decay = decay_rates[tid];

        // d[NT]/dt = release * stim - decay * [NT]
        float d_level = release * stim - decay * level;
        level += d_level * dt;

        // Clamp to valid range [0, 1]
        level = fmaxf(0.0f, fminf(1.0f, level));

        nt_levels[tid] = level;
    }
    __syncthreads();

    // Phase 2: Compute derived parameters
    // Only thread 0 computes scalar outputs
    if (tid == 0) {
        float da = nt_levels[NT_DOPAMINE];
        float sht = nt_levels[NT_SEROTONIN];
        float ne = nt_levels[NT_NORADRENALINE];
        float ach = nt_levels[NT_ACETYLCHOLINE];

        // Hopfield beta: 1 + 4 * sigmoid(DA - 0.5)
        float sigmoid_da = 1.0f / (1.0f + expf(-(da - 0.5f) * 6.0f));
        *hopfield_beta = 1.0f + 4.0f * sigmoid_da;

        // Attention temperature: 0.5 + 1.5 * NE
        *attention_temp = 0.5f + 1.5f * ne;

        // Learning rate: 0.001 + 0.001 * ACh
        *learning_rate = 0.001f + 0.001f * ach;
    }

    // Phase 3: Compute space weights (13 threads)
    if (tid < NUM_SPACES) {
        float sht = nt_levels[NT_SEROTONIN];
        float base = base_space_weights[tid];

        // Serotonin modulates how much weight each space gets
        // Higher serotonin = more spaces considered (weights closer to base)
        // Lower serotonin = focus on top spaces (weights more differentiated)
        float modulation = 0.5f + 0.5f * sht;  // [0.5, 1.0]
        space_weights[tid] = base * modulation;
    }
}

// Reward-based dopamine update kernel
__global__ void neuromod_reward_kernel(
    float* nt_levels,
    float reward,
    float expected_reward
) {
    // Only one thread needed
    if (threadIdx.x != 0) return;

    // Reward prediction error
    float rpe = reward - expected_reward;

    // Update dopamine based on RPE
    // Positive RPE -> increase DA
    // Negative RPE -> decrease DA
    float da = nt_levels[NT_DOPAMINE];
    float da_update = 0.0f;

    if (rpe > 0.0f) {
        // Positive surprise: DA burst
        da_update = rpe * 0.2f;
    } else {
        // Negative surprise: DA dip
        da_update = rpe * 0.1f;
    }

    da = fmaxf(0.0f, fminf(1.0f, da + da_update));
    nt_levels[NT_DOPAMINE] = da;
}
```

### Memory Layout

```
NT levels: [4] floats (16 bytes)
- [0] = Dopamine
- [1] = Serotonin
- [2] = Noradrenaline
- [3] = Acetylcholine

Space weights: [13] floats (52 bytes)

Scalar outputs: 4 bytes each
- hopfield_beta
- attention_temp
- learning_rate
```

### Performance Target
- Single update: **<50us**
- Batch update with param computation: **<200us**

---

## Implementation Priority

1. **hnsw.cu** (Critical)
   - Largest performance impact
   - Required for all 12 HNSW indexes
   - Most complex implementation

2. **neuromod.cu** (High)
   - Required for bio-nervous layer integration
   - Relatively simple kernel
   - P4-1 dependency for bio-layers

3. **hopfield.cu** (High)
   - Required for L3_Memory layer
   - Central to associative memory system
   - Connected to neuromod (beta parameter)

4. **kuramoto.cu** (Medium)
   - Required for L5_Coherence layer
   - Small data size (13 oscillators)
   - Could use CPU as fallback initially

---

## P4-1 Integration Notes (GPU Dependencies for Bio-Layers)

The bio-nervous layer (P4-1) will depend on these GPU kernels:

1. **L2_Reflex + L3_Memory**: Uses `hopfield.cu` for pattern caching and retrieval
   - Requires `hopfield_beta` from `neuromod.cu`

2. **L4_Learning**: Uses `neuromod.cu` for neurotransmitter-based learning rate modulation
   - `learning_rate` output feeds UTL optimizer

3. **L5_Coherence**: Uses `kuramoto.cu` for phase synchronization
   - Order parameter `r` determines consciousness state

4. **Cross-layer**: HNSW search via `hnsw.cu` underpins vector similarity in all layers

The bio-layer implementation should:
- Initialize CUDA context once at startup
- Maintain device memory buffers for NT levels, phases, patterns
- Call GPU kernels via async streams for non-blocking operation
- Implement fallback CPU paths (with warnings) for development

---

## Build System Updates

Add to `crates/context-graph-cuda/build.rs`:

```rust
#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    // ... existing code ...

    // Add new kernels
    compile_kernel("kernels/hnsw.cu", "hnsw");
    compile_kernel("kernels/hopfield.cu", "hopfield");
    compile_kernel("kernels/kuramoto.cu", "kuramoto");
    compile_kernel("kernels/neuromod.cu", "neuromod");
}
```

Update `Cargo.toml`:
```toml
[features]
cuda = []
hnsw-gpu = ["cuda"]
hopfield-gpu = ["cuda"]
kuramoto-gpu = ["cuda"]
neuromod-gpu = ["cuda"]
full-gpu = ["hnsw-gpu", "hopfield-gpu", "kuramoto-gpu", "neuromod-gpu"]
```
