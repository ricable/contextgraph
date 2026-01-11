# Sherlock Holmes Forensic Investigation Report: Dream Layer and Memory Consolidation

```
================================================================================
                        CASE FILE: DREAM-LAYER-2026-001
================================================================================
```

**Case ID:** DREAM-LAYER-2026-001
**Date:** 2026-01-10
**Investigator:** Sherlock Holmes (Forensic Code Investigation Agent)
**Subject:** Dream Layer Implementation for Memory Consolidation and Blind Spot Discovery

---

## EXECUTIVE SUMMARY

*"The game is afoot!"*

After exhaustive forensic examination of the codebase, I present my findings on the dream layer implementation. The investigation reveals a **well-architected but partially implemented** system with robust constitution compliance for implemented components, but with critical stub implementations that prevent full dream functionality.

**VERDICT: PARTIALLY GUILTY - IMPLEMENTATION INCOMPLETE**

The dream layer architecture is sound and constitutionally compliant, but several core behaviors remain stub implementations awaiting completion by "Agent 2" as documented in code comments.

---

## 1. EVIDENCE COLLECTED

### 1.1 File Inventory

| File Path | Purpose | Status |
|-----------|---------|--------|
| `/crates/context-graph-core/src/dream/mod.rs` | Dream layer module root | COMPLETE |
| `/crates/context-graph-core/src/dream/controller.rs` | DreamController orchestrator | COMPLETE (structure) |
| `/crates/context-graph-core/src/dream/nrem.rs` | NREM phase handler | PARTIAL (stub processing) |
| `/crates/context-graph-core/src/dream/rem.rs` | REM phase handler | PARTIAL (stub processing) |
| `/crates/context-graph-core/src/dream/scheduler.rs` | Dream trigger scheduler | COMPLETE |
| `/crates/context-graph-core/src/dream/amortized.rs` | Marblestone shortcut learner | COMPLETE (logic, stub storage) |
| `/crates/context-graph-mcp/src/handlers/dream.rs` | MCP dream tool handlers | COMPLETE |
| `/crates/context-graph-utl/src/phase/consolidation/` | Consolidation phase detection | COMPLETE |

---

## 2. PRD REQUIREMENT ANALYSIS

### 2.1 Trigger Conditions

**PRD Requirement:**
- Activity < 0.15 for 10min
- OR Entropy > 0.7 for 5+min

**EVIDENCE - Constitution Reference (lines 391-394):**
```yaml
dream:
  trigger: { activity: "<0.15", idle: "10min" }
```

**EVIDENCE - Implementation (`scheduler.rs` lines 73-77):**
```rust
/// Activity threshold below which dream may trigger (Constitution: 0.15)
activity_threshold: f32,

/// Duration of low activity required (Constitution: 10 minutes)
idle_duration_trigger: Duration,
```

**EVIDENCE - Constants (`mod.rs` lines 73-77):**
```rust
/// Activity threshold for dream trigger (Constitution: 0.15)
pub const ACTIVITY_THRESHOLD: f32 = 0.15;

/// Idle duration before dream trigger (Constitution: 10 minutes)
pub const IDLE_DURATION_TRIGGER: Duration = Duration::from_secs(600);
```

| Condition | Constitution | Implementation | Verdict |
|-----------|--------------|----------------|---------|
| Activity < 0.15 for 10min | YES | YES - `DreamScheduler` | INNOCENT |
| Entropy > 0.7 for 5+min | DOCUMENTED | **NO** - Not implemented | **GUILTY** |

**FINDING:** The high-entropy trigger condition (entropy > 0.7 for 5+ min) is documented in the PRD and constitution.yaml (line 749: `"entropy>0.7 for 5min->full"`) but is **NOT implemented** in the DreamScheduler. The scheduler only monitors activity levels, not entropy.

---

### 2.2 NREM Phase (3 minutes)

**PRD Requirement:**
- Duration: 3 minutes
- Hebbian learning: delta_w = eta x pre x post
- Tight coupling (0.9)
- Replay recent memories with recency_bias: 0.8

**EVIDENCE - Implementation (`nrem.rs` lines 123-136):**
```rust
pub fn new() -> Self {
    Self {
        duration: constants::NREM_DURATION,        // 180 seconds
        coupling: constants::NREM_COUPLING,        // 0.9
        recency_bias: constants::NREM_RECENCY_BIAS, // 0.8
        learning_rate: 0.01,
        // ...
    }
}
```

**EVIDENCE - Hebbian Update (`nrem.rs` lines 237-251):**
```rust
pub fn hebbian_update(
    &self,
    current_weight: f32,
    pre_activation: f32,
    post_activation: f32,
) -> f32 {
    // Hebbian update: "neurons that fire together wire together"
    let delta_w = self.learning_rate * pre_activation * post_activation;
    let decayed = current_weight * (1.0 - self.weight_decay);
    (decayed + delta_w).clamp(self.weight_floor, self.weight_cap)
}
```

| Feature | Required | Implemented | Verdict |
|---------|----------|-------------|---------|
| Duration: 3 min | 180s | 180s | INNOCENT |
| Coupling: 0.9 | 0.9 | 0.9 | INNOCENT |
| Recency bias: 0.8 | 0.8 | 0.8 | INNOCENT |
| Hebbian formula | delta_w = eta x pre x post | YES | INNOCENT |
| Actual memory replay | YES | **STUB** | **GUILTY** |

**CRITICAL FINDING (nrem.rs line 184-188):**
```rust
// TODO: Agent 2 will implement actual processing:
// 1. Select memories with recency bias
// 2. Apply Hebbian updates: delta_w = eta * pre * post
// 3. Apply tight coupling via Kuramoto
// 4. Detect shortcut candidates
```

The NREM phase `process()` method is a **stub that simulates values** rather than performing actual memory replay and Hebbian learning.

---

### 2.3 REM Phase (2 minutes)

**PRD Requirement:**
- Duration: 2 minutes
- Synthetic queries via hyperbolic random walk
- Blind spot discovery (high semantic distance + shared causal)
- New edges with w=0.3, c=0.5
- Temperature: 2.0

**EVIDENCE - Implementation (`rem.rs` lines 124-137):**
```rust
pub fn new() -> Self {
    Self {
        duration: constants::REM_DURATION,           // 120 seconds
        temperature: constants::REM_TEMPERATURE,     // 2.0
        min_semantic_leap: constants::MIN_SEMANTIC_LEAP, // 0.7
        query_limit: constants::MAX_REM_QUERIES,     // 100
        new_edge_weight: 0.3,
        new_edge_confidence: 0.5,
        // ...
    }
}
```

| Feature | Required | Implemented | Verdict |
|---------|----------|-------------|---------|
| Duration: 2 min | 120s | 120s | INNOCENT |
| Temperature: 2.0 | 2.0 | 2.0 | INNOCENT |
| Semantic leap: 0.7 | 0.7 | 0.7 | INNOCENT |
| Query limit: 100 | 100 | 100 | INNOCENT |
| New edge w=0.3 | 0.3 | 0.3 (defined) | INNOCENT |
| New edge c=0.5 | 0.5 | 0.5 (defined) | INNOCENT |
| Hyperbolic random walk | YES | **STUB** | **GUILTY** |
| Blind spot discovery | YES | **PARTIAL** | **SUSPICIOUS** |

**CRITICAL FINDING (rem.rs lines 177-183):**
```rust
// TODO: Agent 2 will implement actual processing:
// 1. Generate synthetic queries via random walk
// 2. Search with high temperature (2.0)
// 3. Filter for semantic leap >= 0.7
// 4. Create new edges for discovered connections
// 5. Track blind spots
```

The `SyntheticQuery` struct exists but is **never populated** with actual hyperbolic random walk data. The `BlindSpot` struct exists with proper validation but discovery is simulated.

---

### 2.4 Amortized Shortcuts (Marblestone)

**PRD Requirement:**
- 3+ hop chains traversed >= 5x -> direct edge
- Confidence >= 0.7
- w = product(path weights)
- is_amortized_shortcut = true

**EVIDENCE - Implementation (`amortized.rs` lines 122-132):**
```rust
pub fn new() -> Self {
    Self {
        path_counts: HashMap::new(),
        min_hops: constants::MIN_SHORTCUT_HOPS,           // 3
        min_traversals: constants::MIN_SHORTCUT_TRAVERSALS, // 5
        confidence_threshold: constants::SHORTCUT_CONFIDENCE_THRESHOLD, // 0.7
        // ...
    }
}
```

**EVIDENCE - GraphEdge Support (`edge.rs` line 67):**
```rust
/// Whether this edge is an amortized shortcut (learned during dreams).
pub is_amortized_shortcut: bool,
```

**EVIDENCE - Quality Gate (`amortized.rs` lines 78-83):**
```rust
pub fn meets_quality_gate(&self) -> bool {
    self.hop_count >= constants::MIN_SHORTCUT_HOPS
        && self.traversal_count >= constants::MIN_SHORTCUT_TRAVERSALS
        && self.min_confidence >= constants::SHORTCUT_CONFIDENCE_THRESHOLD
}
```

| Feature | Required | Implemented | Verdict |
|---------|----------|-------------|---------|
| Min hops: 3 | 3 | 3 | INNOCENT |
| Min traversals: 5 | 5 | 5 | INNOCENT |
| Confidence >= 0.7 | 0.7 | 0.7 | INNOCENT |
| Weight = product | YES | YES (line 152) | INNOCENT |
| GraphEdge.is_amortized_shortcut | YES | YES | INNOCENT |
| Actual shortcut creation | YES | **STUB** | **GUILTY** |

**CRITICAL FINDING (amortized.rs lines 250-259):**
```rust
// TODO: Agent 2 will implement actual edge creation:
// let edge = Edge {
//     source: candidate.source,
//     target: candidate.target,
//     weight: candidate.combined_weight,
//     confidence: candidate.min_confidence,
//     is_shortcut: true,
//     original_path: Some(candidate.path_nodes.clone()),
// };
// graph.store_edge(&edge).await?;
```

---

### 2.5 Wake Behavior

**PRD Requirement:**
- Wake latency < 100ms on query
- abort_on_query: true
- GPU usage < 30%

**EVIDENCE - Implementation (`controller.rs` lines 401-439):**
```rust
pub fn abort(&mut self) -> CoreResult<Duration> {
    let abort_start = Instant::now();
    self.interrupt_flag.store(true, Ordering::SeqCst);
    self.state = DreamState::Waking;
    self.state = DreamState::Awake;
    let wake_latency = abort_start.elapsed();

    if wake_latency > self.wake_latency_budget {
        error!("Wake latency {:?} exceeded budget {:?}", wake_latency, self.wake_latency_budget);
        return Err(CoreError::LayerError { ... });
    }
    Ok(wake_latency)
}
```

**EVIDENCE - Constants (`mod.rs` lines 85-90):**
```rust
/// Maximum wake latency (Constitution: <100ms)
/// Set to 99ms to satisfy strict less-than requirement
pub const MAX_WAKE_LATENCY: Duration = Duration::from_millis(99);

/// Maximum GPU usage during dream (Constitution: <30%)
pub const MAX_GPU_USAGE: f32 = 0.30;
```

| Feature | Required | Implemented | Verdict |
|---------|----------|-------------|---------|
| Wake latency < 100ms | <100ms | 99ms max | INNOCENT |
| abort_on_query | true | Interrupt flag checked | INNOCENT |
| GPU < 30% | <30% | 0.30 constant, **stub check** | SUSPICIOUS |

**FINDING (controller.rs lines 469-475):**
```rust
fn current_gpu_usage(&self) -> f32 {
    // TODO: Integrate with actual GPU monitoring
    // For now, return a safe value
    0.0
}
```

GPU monitoring is a **stub returning 0.0**.

---

## 3. BLIND SPOT DISCOVERY ANALYSIS

The blind spot discovery system is **partially implemented** across multiple modules:

### 3.1 REM Phase BlindSpot Struct

**Location:** `rem.rs` lines 93-111
```rust
pub struct BlindSpot {
    pub node_a: Uuid,
    pub node_b: Uuid,
    pub semantic_distance: f32,
    pub confidence: f32,
}

impl BlindSpot {
    pub fn is_significant(&self) -> bool {
        self.semantic_distance >= constants::MIN_SEMANTIC_LEAP && self.confidence >= 0.5
    }
}
```

### 3.2 Johari-Based Blind Spot Discovery

**Location:** `johari/default_manager.rs` lines 369-410

A more sophisticated blind spot discovery exists in the Johari manager that:
- Analyzes external signals for Unknown/Hidden quadrants
- Calculates signal strength based on mismatches
- Creates BlindSpotCandidate objects with embedder indices

```rust
async fn discover_blind_spots(
    &self,
    id: NodeId,
    signals: &[ExternalSignal],
) -> Result<Vec<BlindSpotCandidate>, JohariError>
```

### 3.3 Fingerprint Analysis

**Location:** `types/fingerprint/johari/analysis.rs` lines 26-44

```rust
pub fn find_blind_spots(&self) -> Vec<(usize, f32)> {
    // Finds cross-space gaps where one embedder understands but another doesn't
}
```

**VERDICT ON BLIND SPOTS:**
- Structure: INNOCENT (well-designed)
- Integration with REM: **GUILTY** (not connected)

---

## 4. MEMORY CONSOLIDATION MECHANISMS

### 4.1 Phase Detection System

**Location:** `utl/phase/consolidation/`

A complete phase detection system exists:
- `ConsolidationPhase` enum (NREM/REM/Wake)
- `PhaseDetector` with EMA smoothing and hysteresis
- Phase-specific parameters (recency_bias, temperature, coupling_strength)

**This is SEPARATE from the Dream Layer** - it provides phase detection based on activity levels but does not integrate with the actual dream controller.

### 4.2 Hebbian Learning Logic

The mathematical formula is correctly implemented:
```rust
delta_w = self.learning_rate * pre_activation * post_activation
```

But it is **never called with real activation data** during dream processing.

---

## 5. CONSTITUTION COMPLIANCE MATRIX

| Mandate | Constitution | Implementation | Status |
|---------|--------------|----------------|--------|
| Activity trigger < 0.15 for 10min | line 392 | DreamScheduler | COMPLIANT |
| NREM duration 3min | line 393 | NREM_DURATION = 180s | COMPLIANT |
| REM duration 2min | line 393 | REM_DURATION = 120s | COMPLIANT |
| NREM recency_bias 0.8 | line 393 | NREM_RECENCY_BIAS = 0.8 | COMPLIANT |
| REM temp 2.0 | line 393 | REM_TEMPERATURE = 2.0 | COMPLIANT |
| Max queries 100 | line 394 | MAX_REM_QUERIES = 100 | COMPLIANT |
| Semantic leap >= 0.7 | line 394 | MIN_SEMANTIC_LEAP = 0.7 | COMPLIANT |
| abort_on_query true | line 394 | Interrupt flag | COMPLIANT |
| Wake latency < 100ms | line 394 | MAX_WAKE_LATENCY = 99ms | COMPLIANT |
| GPU < 30% | line 394 | MAX_GPU_USAGE = 0.30 | CONSTANTS ONLY |

---

## 6. THE SMOKING GUN - STUB IMPLEMENTATIONS

### Evidence Trail

Throughout the codebase, the phrase **"Agent 2 will implement"** appears as a marker for incomplete functionality:

| Location | Stub Description |
|----------|------------------|
| `nrem.rs:184-188` | Memory selection, Hebbian updates, Kuramoto coupling |
| `rem.rs:177-183` | Synthetic queries, high-temp search, edge creation |
| `amortized.rs:250-259` | Actual shortcut edge storage |
| `controller.rs:469-475` | GPU monitoring |

---

## 7. RECOMMENDATIONS

### 7.1 Critical Implementation Gaps

1. **HIGH PRIORITY - NREM Memory Replay**
   - Implement actual memory selection with recency weighting
   - Apply Hebbian updates to real edge weights
   - Integrate Kuramoto coupling for synchronization
   - Connect to storage layer for weight persistence

2. **HIGH PRIORITY - REM Exploration**
   - Implement hyperbolic random walk on Poincare ball embeddings
   - Generate real synthetic queries from walk endpoints
   - Connect to vector search with temperature=2.0 softmax
   - Create actual edges for discovered blind spots

3. **MEDIUM PRIORITY - Entropy Trigger**
   - Add entropy tracking to DreamScheduler
   - Implement 5-minute high-entropy window detection
   - Connect to CognitivePulse entropy measurements

4. **MEDIUM PRIORITY - GPU Monitoring**
   - Integrate with CUDA monitoring APIs
   - Implement actual GPU usage tracking
   - Add circuit breaker for > 30% usage

5. **LOW PRIORITY - Phase Detection Integration**
   - Connect UTL ConsolidationPhase detector with Dream layer
   - Unify activity monitoring systems

### 7.2 Architecture Observations

The codebase shows excellent architecture:
- Clear separation between phases
- Well-defined constitution constants
- Proper interrupt handling for wake latency
- Good test coverage for implemented components

The stubs are **intentional** and well-documented, indicating planned future work rather than oversight.

---

## 8. VERDICT

```
================================================================================
                           CASE CLOSED
================================================================================
```

**THE CRIME:** Dream Layer claims to enable memory consolidation and blind spot discovery, but core functionality is stub implementations.

**THE CRIMINAL:** Not malice, but intentional phased development. "Agent 2" is the designated future implementer.

**THE EVIDENCE:**
1. All constitution constants are correctly defined
2. Phase structures (NREM/REM) are properly architected
3. Hebbian learning formula exists but is never invoked with real data
4. Blind spot discovery exists in Johari but not connected to REM
5. Amortized shortcuts have logic but no storage integration
6. Entropy trigger is completely missing
7. GPU monitoring is a stub returning 0.0

**THE NARRATIVE:**
The Dream Layer was designed with constitution compliance in mind. The architecture is sound, constants are correct, and the framework is ready. However, the actual "dreaming" - memory replay, exploration, and consolidation - remains simulated placeholder code awaiting full implementation.

**THE SENTENCE:**
Full dream capability requires:
- NREM: Real memory selection + Hebbian application
- REM: Real hyperbolic random walk + blind spot creation
- Storage: Shortcut edge persistence
- Monitoring: GPU and entropy triggers

**CONFIDENCE:** HIGH

**INVESTIGATION STATUS:** COMPLETE

---

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

The truth is that this system is **ready for dreaming** but has not yet learned to dream.

---

## APPENDIX: KEY FILE LOCATIONS

```
/home/cabdru/contextgraph/
  crates/
    context-graph-core/src/
      dream/
        mod.rs              # Module root with constants
        controller.rs       # DreamController orchestration
        nrem.rs             # NREM phase with Hebbian (STUB)
        rem.rs              # REM phase with exploration (STUB)
        scheduler.rs        # Activity-based trigger (COMPLETE)
        amortized.rs        # Shortcut learner (STUB storage)
    context-graph-mcp/src/handlers/
      dream.rs              # MCP tool handlers (COMPLETE)
    context-graph-utl/src/phase/consolidation/
      mod.rs                # Phase detection root
      phase.rs              # ConsolidationPhase enum
      detector.rs           # PhaseDetector with EMA
  docs2/
    constitution.yaml       # lines 388-394 define dream mandates
    contextprd.md           # Section 7.1 Dream Layer specification
```

---

**Case File Closed: 2026-01-10**
**Sherlock Holmes, Forensic Code Investigation Agent**
