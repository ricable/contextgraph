# context-graph-core

Core domain types, traits, and business logic for the Context Graph knowledge system.

## Overview

This crate provides the foundational types for the Ultimate Context Graph - a 13-embedder memory system based on the Unified Theory of Learning (UTL). It implements the Marblestone architecture for neural-inspired knowledge organization.

## Features

- **MemoryNode**: Core knowledge node with 1536D embeddings and Ebbinghaus decay
- **GraphEdge**: Marblestone-inspired edges with 13 fields including neurotransmitter weights and steering rewards
- **CognitivePulse**: Meta-cognitive state tracking with entropy/coherence metrics
- **TeleologicalFingerprint**: 13-embedding fingerprint for multi-space retrieval
- **Marblestone Types**: Domain, EdgeType, and NeurotransmitterWeights for neural-inspired edge modulation

## Quick Start

```rust
use context_graph_core::types::{MemoryNode, CognitivePulse};
use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};

// Create a memory node
fn create_valid_embedding() -> Vec<f32> {
    const DIM: usize = 1536;
    let val = 1.0_f32 / (DIM as f32).sqrt();
    vec![val; DIM]
}

let mut node = MemoryNode::new(
    "Rust async/await patterns".to_string(),
    create_valid_embedding(),
);
node.importance = 0.8;
node.validate().expect("Node should be valid");

// Work with cognitive pulse
let pulse = CognitivePulse::from_values(0.3, 0.8);
println!("Suggested action: {:?}", pulse.suggested_action);
// Output: Ready (low entropy, high coherence)
```

## Module Structure

```
context-graph-core/
├── types/              # Core domain types
│   ├── memory_node/    # MemoryNode, NodeMetadata, ValidationError
│   ├── graph_edge/     # GraphEdge, EdgeId
│   ├── fingerprint/    # TeleologicalFingerprint, SemanticFingerprint, TopicProfile
│   ├── pulse.rs        # CognitivePulse, SuggestedAction
│   └── utl.rs          # UtlMetrics, EmotionalState
├── marblestone/        # Marblestone architecture types
│   ├── domain.rs       # Domain enum (Code, Legal, Medical, etc.)
│   ├── edge_type.rs    # EdgeType enum (Semantic, Temporal, Causal, Hierarchical)
│   └── neurotransmitter_weights.rs  # NT weights for edge modulation
├── traits/             # Trait definitions
│   └── memory_store.rs # MemoryStore async trait
├── error.rs            # CoreError, CoreResult
└── config.rs           # Config struct
```

## Key Types

### MemoryNode

The fundamental knowledge unit with:
- **content**: Text content (<=1MB)
- **embedding**: 1536D normalized vector
- **importance**: [0.0, 1.0] relevance score
- **metadata**: Tags, source, timestamps

```rust
let mut node = MemoryNode::new(content, embedding);
node.validate()?;  // Enforces constitution constraints
node.record_access();  // Track for decay calculation
let decay = node.compute_decay();  // Ebbinghaus decay factor
```

### GraphEdge (Marblestone)

Neural-inspired edges with 13 fields:
- Source/target UUIDs
- EdgeType (Semantic, Temporal, Causal, Hierarchical)
- Domain (Code, Legal, Medical, Creative, Research, General)
- Neurotransmitter weights (excitatory, inhibitory, modulatory)
- Steering reward [-1, 1] for feedback
- Amortized shortcut tracking

```rust
let edge = GraphEdge::new(source_id, target_id, EdgeType::Causal, Domain::Code);
let modulated = edge.get_modulated_weight();
// Uses internal self.weight: w_eff = base × (1 + E - I + 0.5×M)
```

### CognitivePulse

Meta-cognitive state with 6 fields:
- entropy [0, 1]: Uncertainty/novelty level
- coherence [0, 1]: Integration/understanding
- suggested_action: Ready, Continue, Explore, Consolidate, Prune, Stabilize, Review

```rust
let pulse = CognitivePulse::from_values(0.85, 0.25);
assert_eq!(pulse.suggested_action, SuggestedAction::Stabilize);
// High entropy + low coherence = needs stabilization
```

## Examples

Run the cognitive pulse example:

```bash
cargo run --package context-graph-core --example cognitive_pulse
```

## Performance Constraints

Per constitution.yaml:
- Embedding validation: < 1ms
- Node validation: < 1ms
- Memory decay calculation: < 1ms
- All operations non-blocking

## License

MIT
