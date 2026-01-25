# context-graph-storage

RocksDB-based storage layer for the Context Graph knowledge system.

## Overview

This crate provides persistent storage for MemoryNodes, GraphEdges, and embeddings using RocksDB with 12 column families optimized for different access patterns.

## Features

- **RocksDbMemex**: Main storage backend implementing the Memex trait
- **8 Column Families**: Optimized for nodes, edges, embeddings, and secondary indexes
- **CRUD Operations**: Store, get, update, delete for nodes and edges
- **Secondary Indexes**: Query by tags, source, and time range
- **Embedding Storage**: Separate optimized storage for 1536D vectors
- **Hybrid Serialization**: MessagePack for nodes/edges, bincode for embeddings

## Quick Start

```rust
use context_graph_core::types::MemoryNode;
use context_graph_storage::{RocksDbMemex, Memex};
use tempfile::TempDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create database
    let temp_dir = TempDir::new()?;
    let memex = RocksDbMemex::open(temp_dir.path())?;

    // Create and store a node
    let mut node = MemoryNode::new(
        "Rust programming concepts".to_string(),
        create_embedding(),
    );
    node.validate()?;

    memex.store_node(&node)?;

    // Retrieve by ID
    let retrieved = memex.get_node(&node.id)?;

    // Query by tag (returns Vec<NodeId>)
    node.metadata.add_tag("rust");
    memex.update_node(&node)?;
    let rust_nodes = memex.query_by_tag("rust", Some(10))?;

    // Health check
    let health = memex.health_check()?;
    println!("Nodes: {}, Edges: {}", health.node_count, health.edge_count);

    Ok(())
}

fn create_embedding() -> Vec<f32> {
    const DIM: usize = 1536;
    let val = 1.0_f32 / (DIM as f32).sqrt();
    vec![val; DIM]
}
```

## Column Families

| Family | Purpose | Key Format |
|--------|---------|------------|
| `nodes` | Primary node storage | UUID bytes |
| `edges` | Graph edges | source:target:type |
| `embeddings` | Vector storage | UUID bytes |
| `metadata` | Node metadata | UUID bytes |
| `temporal` | Time-based index | timestamp:UUID |
| `tags` | Tag index | tag:UUID |
| `sources` | Source index | source:UUID |
| `system` | System metadata | key strings |

## Memex Trait

The `Memex` trait defines the storage interface:

```rust
pub trait Memex: Send + Sync {
    // Node operations
    fn store_node(&self, node: &MemoryNode) -> StorageResult<()>;
    fn get_node(&self, id: &NodeId) -> StorageResult<MemoryNode>;
    fn update_node(&self, node: &MemoryNode) -> StorageResult<()>;
    fn delete_node(&self, id: &NodeId, soft_delete: bool) -> StorageResult<()>;

    // Edge operations
    fn store_edge(&self, edge: &GraphEdge) -> StorageResult<()>;
    fn get_edge(&self, source: &NodeId, target: &NodeId, edge_type: EdgeType) -> StorageResult<GraphEdge>;
    fn get_edges_from(&self, source: &NodeId) -> StorageResult<Vec<GraphEdge>>;
    fn get_edges_to(&self, target: &NodeId) -> StorageResult<Vec<GraphEdge>>;

    // Index queries (return NodeIds for efficiency)
    fn query_by_tag(&self, tag: &str, limit: Option<usize>) -> StorageResult<Vec<NodeId>>;

    // Health
    fn health_check(&self) -> StorageResult<StorageHealth>;
}
```

## Examples

### Basic Storage Operations

```bash
cargo run --package context-graph-storage --example basic_storage
```

Demonstrates:
- Node creation, storage, and retrieval
- Tag queries
- Update and delete operations
- Embedding storage
- Health checks

### Marblestone Edge Operations

```bash
cargo run --package context-graph-storage --example marblestone_edges
```

Demonstrates:
- Domain-specific neurotransmitter weights
- Modulated weight calculation
- Steering reward system
- Edge traversal queries
- Amortized shortcut creation

## Configuration

```rust
use context_graph_storage::RocksDbConfig;

let config = RocksDbConfig {
    cache_size_mb: 256,      // Block cache size
    max_open_files: 1000,    // File descriptor limit
    enable_statistics: true,  // Performance stats
    max_write_buffer_mb: 64, // Write buffer size
};

let memex = RocksDbMemex::open_with_config(path, config)?;
```

## Performance Constraints

Per constitution.yaml:

| Operation | Latency Target |
|-----------|----------------|
| `store_node` | < 5ms |
| `get_node` | < 1ms |
| `store_edge` | < 3ms |
| `get_edge` | < 1ms |
| `health_check` | < 1ms |
| `flush_all` | < 500ms |

## Error Handling

All operations return `StorageResult<T>` with detailed error types:

```rust
use context_graph_storage::StorageError;

match memex.get_node(&id) {
    Ok(node) => println!("Found: {}", node.content),
    Err(StorageError::NotFound { entity_type, id }) => {
        println!("Node {} not found", id);
    }
    Err(e) => eprintln!("Storage error: {}", e),
}
```

## License

MIT
