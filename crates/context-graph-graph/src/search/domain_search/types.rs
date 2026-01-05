//! Types and data structures for domain-aware search.
//!
//! This module contains the result types and constants used by domain search.

use crate::search::{Domain, SemanticSearchResultItem};
use uuid::Uuid;

/// Domain bonus for matching domains (from constitution)
pub const DOMAIN_MATCH_BONUS: f32 = 0.1;

/// Over-fetch multiplier for re-ranking
pub const OVERFETCH_MULTIPLIER: usize = 3;

/// Result from domain-aware search with modulation metadata.
#[derive(Debug, Clone)]
pub struct DomainSearchResult {
    /// FAISS internal ID
    pub faiss_id: i64,

    /// Node UUID (if resolved via metadata provider)
    pub node_id: Option<Uuid>,

    /// Base similarity score (before modulation)
    pub base_similarity: f32,

    /// Modulated score after NT adjustment
    pub modulated_score: f32,

    /// L2 distance from query
    pub distance: f32,

    /// Rank in result set (0 = best match)
    pub rank: usize,

    /// Domain of the result node
    pub node_domain: Option<Domain>,

    /// Query domain used for modulation
    pub query_domain: Domain,

    /// Whether domain matched (bonus applied)
    pub domain_matched: bool,

    /// Net activation from NT weights (for debugging)
    pub net_activation: f32,
}

impl DomainSearchResult {
    /// Create from semantic search result item with modulation.
    pub fn from_semantic_item(
        item: &SemanticSearchResultItem,
        modulated_score: f32,
        net_activation: f32,
        query_domain: Domain,
    ) -> Self {
        let domain_matched = item.domain.map(|d| d == query_domain).unwrap_or(false);

        Self {
            faiss_id: item.faiss_id,
            node_id: item.node_id,
            base_similarity: item.similarity,
            modulated_score,
            distance: item.distance,
            rank: 0, // Will be set after re-ranking
            node_domain: item.domain,
            query_domain,
            domain_matched,
            net_activation,
        }
    }

    /// Get the boost/penalty applied (modulated - base).
    #[inline]
    pub fn modulation_delta(&self) -> f32 {
        self.modulated_score - self.base_similarity
    }

    /// Get boost ratio (modulated / base).
    #[inline]
    pub fn boost_ratio(&self) -> f32 {
        if self.base_similarity > 1e-6 {
            self.modulated_score / self.base_similarity
        } else {
            1.0
        }
    }
}

/// Container for domain search results with metadata.
#[derive(Debug, Clone)]
pub struct DomainSearchResults {
    /// Search results ordered by modulated score
    pub items: Vec<DomainSearchResult>,

    /// Number of candidates fetched before filtering
    pub candidates_fetched: usize,

    /// Number of results after filtering
    pub results_returned: usize,

    /// Query domain used
    pub query_domain: Domain,

    /// Search latency in microseconds
    pub latency_us: u64,
}

impl DomainSearchResults {
    /// Create empty results.
    pub fn empty(query_domain: Domain) -> Self {
        Self {
            items: Vec::new(),
            candidates_fetched: 0,
            results_returned: 0,
            query_domain,
            latency_us: 0,
        }
    }

    /// Iterate over results.
    pub fn iter(&self) -> impl Iterator<Item = &DomainSearchResult> {
        self.items.iter()
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get number of results.
    pub fn len(&self) -> usize {
        self.items.len()
    }
}
