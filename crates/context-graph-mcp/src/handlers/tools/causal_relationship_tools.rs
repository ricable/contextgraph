//! Causal relationship search tool implementation.
//!
//! Provides the `search_causal_relationships` MCP tool for semantic search
//! of LLM-generated causal descriptions with full provenance.
//!
//! # E5 Asymmetric Search
//!
//! Supports directional causal search using E5 dual embeddings:
//! - `searchMode: "causes"` → "What caused X?" → query as effect, search cause vectors
//! - `searchMode: "effects"` → "What are effects of X?" → query as cause, search effect vectors
//! - `searchMode: "semantic"` → Fallback E1 semantic search (default)

use serde_json::json;
use tracing::{debug, error, info};

use crate::protocol::{JsonRpcId, JsonRpcResponse};

use super::super::Handlers;

/// Validation constants for search_causal_relationships
const MIN_TOP_K: u64 = 1;
const MAX_TOP_K: u64 = 100;
const DEFAULT_TOP_K: u64 = 10;

impl Handlers {
    /// search_causal_relationships tool implementation.
    ///
    /// Searches for causal relationships by semantic similarity to query.
    /// Returns matching causal descriptions with their source provenance.
    ///
    /// # Arguments (from JSON)
    /// * `query` - Natural language query about causal relationships
    /// * `searchMode` - Search strategy (default: "semantic"):
    ///   - "causes": Find what caused X (query as effect, search cause vectors)
    ///   - "effects": Find effects of X (query as cause, search effect vectors)
    ///   - "semantic": Fallback E1 semantic search
    /// * `topK` - Number of results (1-100, default: 10)
    /// * `includeSource` - Include original source content in results (default: true)
    ///
    /// # Returns
    /// Array of causal relationships with:
    /// - id: Causal relationship UUID
    /// - causeStatement: Brief statement of the cause
    /// - effectStatement: Brief statement of the effect
    /// - explanation: LLM-generated 1-2 paragraph explanation
    /// - mechanismType: "direct", "mediated", "feedback", or "temporal"
    /// - confidence: LLM confidence score
    /// - sourceContent: Original content (if includeSource=true)
    /// - sourceMemoryId: ID of source memory for provenance
    /// - similarity: Search similarity score
    pub(crate) async fn call_search_causal_relationships(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse query parameter (required)
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.is_empty() => q,
            Some(_) => return self.tool_error(id, "Query cannot be empty"),
            None => return self.tool_error(id, "Missing 'query' parameter"),
        };

        // Parse searchMode parameter (optional, default: "semantic")
        let search_mode = args
            .get("searchMode")
            .and_then(|v| v.as_str())
            .unwrap_or("semantic");

        // Validate search mode
        if !matches!(search_mode, "causes" | "effects" | "semantic") {
            return self.tool_error(
                id,
                &format!(
                    "Invalid searchMode '{}'. Must be 'causes', 'effects', or 'semantic'",
                    search_mode
                ),
            );
        }

        // Parse topK parameter (optional, default: 10, range: 1-100)
        let raw_top_k = args.get("topK").and_then(|v| v.as_u64());
        if let Some(k) = raw_top_k {
            if k < MIN_TOP_K {
                return self.tool_error(
                    id,
                    &format!("topK must be at least {}, got {}", MIN_TOP_K, k),
                );
            }
            if k > MAX_TOP_K {
                return self.tool_error(
                    id,
                    &format!("topK must be at most {}, got {}", MAX_TOP_K, k),
                );
            }
        }
        let top_k = raw_top_k.unwrap_or(DEFAULT_TOP_K) as usize;

        // Parse includeSource parameter (optional, default: true)
        let include_source = args
            .get("includeSource")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        info!(
            query_len = query.len(),
            search_mode = search_mode,
            top_k = top_k,
            include_source = include_source,
            "search_causal_relationships: Starting search"
        );

        // Step 1 & 2: Embed query and search based on mode
        let search_results = match search_mode {
            "causes" => {
                // "What caused X?" → query as effect, search cause vectors
                let e5_result = self.multi_array_provider.embed_e5_dual(query).await;
                let (_as_cause, as_effect) = match e5_result {
                    Ok(dual) => dual,
                    Err(e) => {
                        error!(error = %e, "search_causal_relationships: Failed to embed query for E5");
                        return self.tool_error(id, &format!("Failed to embed query: {}", e));
                    }
                };

                debug!(
                    embedding_dim = as_effect.len(),
                    mode = "causes",
                    "search_causal_relationships: Query embedded as effect"
                );

                match self
                    .teleological_store
                    .search_causal_e5(&as_effect, true, top_k)
                    .await
                {
                    Ok(results) => results,
                    Err(e) => {
                        error!(error = %e, "search_causal_relationships: E5 search failed");
                        return self.tool_error(id, &format!("Search failed: {}", e));
                    }
                }
            }
            "effects" => {
                // "What are effects of X?" → query as cause, search effect vectors
                let e5_result = self.multi_array_provider.embed_e5_dual(query).await;
                let (as_cause, _as_effect) = match e5_result {
                    Ok(dual) => dual,
                    Err(e) => {
                        error!(error = %e, "search_causal_relationships: Failed to embed query for E5");
                        return self.tool_error(id, &format!("Failed to embed query: {}", e));
                    }
                };

                debug!(
                    embedding_dim = as_cause.len(),
                    mode = "effects",
                    "search_causal_relationships: Query embedded as cause"
                );

                match self
                    .teleological_store
                    .search_causal_e5(&as_cause, false, top_k)
                    .await
                {
                    Ok(results) => results,
                    Err(e) => {
                        error!(error = %e, "search_causal_relationships: E5 search failed");
                        return self.tool_error(id, &format!("Search failed: {}", e));
                    }
                }
            }
            _ => {
                // "semantic" - Fallback E1 semantic search
                let query_embedding = match self.multi_array_provider.embed_e1_only(query).await {
                    Ok(embedding) => embedding,
                    Err(e) => {
                        error!(error = %e, "search_causal_relationships: Failed to embed query");
                        return self.tool_error(id, &format!("Failed to embed query: {}", e));
                    }
                };

                debug!(
                    embedding_dim = query_embedding.len(),
                    mode = "semantic",
                    "search_causal_relationships: Query embedded for E1 search"
                );

                match self
                    .teleological_store
                    .search_causal_relationships(&query_embedding, top_k, None)
                    .await
                {
                    Ok(results) => results,
                    Err(e) => {
                        error!(error = %e, "search_causal_relationships: Search failed");
                        return self.tool_error(id, &format!("Search failed: {}", e));
                    }
                }
            }
        };

        debug!(
            results_count = search_results.len(),
            "search_causal_relationships: Search complete"
        );

        // Step 3: Fetch full causal relationships and build response
        let mut results = Vec::with_capacity(search_results.len());

        for (causal_id, similarity) in search_results {
            match self
                .teleological_store
                .get_causal_relationship(causal_id)
                .await
            {
                Ok(Some(rel)) => {
                    let mut result = json!({
                        "id": rel.id.to_string(),
                        "causeStatement": rel.cause_statement,
                        "effectStatement": rel.effect_statement,
                        "explanation": rel.explanation,
                        "mechanismType": rel.mechanism_type,
                        "confidence": rel.confidence,
                        "sourceMemoryId": rel.source_fingerprint_id.to_string(),
                        "similarity": similarity,
                        "createdAt": rel.created_at
                    });

                    // Include source content if requested
                    if include_source {
                        result["sourceContent"] = json!(rel.source_content);
                    }

                    results.push(result);
                }
                Ok(None) => {
                    // Causal relationship not found (should be rare)
                    debug!(
                        causal_id = %causal_id,
                        "search_causal_relationships: Causal relationship not found"
                    );
                }
                Err(e) => {
                    // Log error but continue with other results
                    error!(
                        causal_id = %causal_id,
                        error = %e,
                        "search_causal_relationships: Failed to fetch causal relationship"
                    );
                }
            }
        }

        info!(
            query_preview = &query[..query.len().min(50)],
            search_mode = search_mode,
            top_k = top_k,
            results_count = results.len(),
            "search_causal_relationships: Returning results"
        );

        self.tool_result(
            id,
            json!({
                "results": results,
                "query": query,
                "searchMode": search_mode,
                "topK": top_k
            }),
        )
    }
}
