//! Trait implementations for GraphEdge.
//!
//! Implements PartialEq, Eq, and Hash based on edge ID.

use super::types::GraphEdge;

impl PartialEq for GraphEdge {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for GraphEdge {}

impl std::hash::Hash for GraphEdge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
