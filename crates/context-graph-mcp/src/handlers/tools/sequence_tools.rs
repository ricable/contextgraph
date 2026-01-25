//! Sequence tool implementations for E4 (V_ordering) integration.
//!
//! These tools provide first-class conversational context capabilities:
//! - `get_conversation_context`: Get memories around current turn
//! - `get_session_timeline`: Ordered session memory timeline
//! - `traverse_memory_chain`: Multi-hop memory navigation
//! - `compare_session_states`: Before/after state comparison

use serde_json::json;
use tracing::{debug, error};

use context_graph_core::teleological::matrix_search::embedder_names;
use context_graph_core::traits::{
    SequenceDirection, SequenceOptions, TeleologicalSearchOptions, TeleologicalSearchResult,
};
use context_graph_core::types::SourceMetadata;

use crate::protocol::JsonRpcId;
use crate::protocol::JsonRpcResponse;

use super::super::Handlers;

/// Input validation constants for sequence tools
mod validation {
    pub const MIN_WINDOW_SIZE: u64 = 1;
    pub const MAX_WINDOW_SIZE: u64 = 50;
    pub const DEFAULT_WINDOW_SIZE: u64 = 10;

    pub const MIN_LIMIT: u64 = 1;
    pub const MAX_LIMIT: u64 = 200;
    pub const DEFAULT_LIMIT: u64 = 50;

    pub const MIN_HOPS: u64 = 1;
    pub const MAX_HOPS: u64 = 20;
    pub const DEFAULT_HOPS: u64 = 5;

    pub const COMPARISON_BATCH_SIZE: usize = 500;
}
use validation::*;

/// Validate windowSize parameter.
/// Returns Ok(value) or Err(error_message).
fn validate_window_size(value: Option<&serde_json::Value>) -> Result<u64, String> {
    match value.and_then(|v| v.as_u64()) {
        Some(v) if v < MIN_WINDOW_SIZE => Err(
            format!("windowSize {} below minimum {}", v, MIN_WINDOW_SIZE)
        ),
        Some(v) if v > MAX_WINDOW_SIZE => Err(
            format!("windowSize {} exceeds maximum {}", v, MAX_WINDOW_SIZE)
        ),
        Some(v) => Ok(v),
        None => Ok(DEFAULT_WINDOW_SIZE),
    }
}

/// Validate limit parameter.
/// Returns Ok(value) or Err(error_message).
fn validate_limit(value: Option<&serde_json::Value>) -> Result<u64, String> {
    match value.and_then(|v| v.as_u64()) {
        Some(v) if v < MIN_LIMIT => Err(
            format!("limit {} below minimum {}", v, MIN_LIMIT)
        ),
        Some(v) if v > MAX_LIMIT => Err(
            format!("limit {} exceeds maximum {}", v, MAX_LIMIT)
        ),
        Some(v) => Ok(v),
        None => Ok(DEFAULT_LIMIT),
    }
}

/// Validate hops parameter.
/// Returns Ok(value) or Err(error_message).
fn validate_hops(value: Option<&serde_json::Value>) -> Result<u64, String> {
    match value.and_then(|v| v.as_u64()) {
        Some(v) if v < MIN_HOPS => Err(
            format!("hops {} below minimum {}", v, MIN_HOPS)
        ),
        Some(v) if v > MAX_HOPS => Err(
            format!("hops {} exceeds maximum {}", v, MAX_HOPS)
        ),
        Some(v) => Ok(v),
        None => Ok(DEFAULT_HOPS),
    }
}

impl Handlers {
    /// get_conversation_context tool implementation.
    ///
    /// Gets memories around the current conversation turn with auto-anchoring.
    /// Uses E4 (V_ordering) for sequence-based retrieval via conversation_history weight profile.
    pub(crate) async fn call_get_conversation_context(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse direction (before, after, both)
        let direction_str = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("before");
        let direction = match direction_str {
            "before" => SequenceDirection::Before,
            "after" => SequenceDirection::After,
            _ => SequenceDirection::Both,
        };

        // Parse and validate window size (FAIL FAST)
        let window_size = match validate_window_size(args.get("windowSize")) {
            Ok(v) => v as u32,
            Err(msg) => {
                error!(error = %msg, "get_conversation_context: windowSize validation FAILED");
                return self.tool_error(id, &msg);
            }
        };

        // Parse session filter
        let session_only = args
            .get("sessionOnly")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Parse include content
        let include_content = args
            .get("includeContent")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Parse optional semantic filter
        let query = args.get("query").and_then(|v| v.as_str());
        let min_similarity = args
            .get("minSimilarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        // Get current sequence number for anchoring
        let current_seq = self.current_sequence();
        let session_id = self.get_session_id();

        debug!(
            current_seq = current_seq,
            direction = ?direction,
            window_size = window_size,
            session_only = session_only,
            "get_conversation_context: Building sequence-based query"
        );

        // Build sequence options
        let seq_opts = SequenceOptions::from_sequence(current_seq, direction, window_size);

        // Build search options with conversation_history weight profile for E1+E4 balance
        let mut options = TeleologicalSearchOptions::quick(window_size as usize * 2)
            .with_min_similarity(min_similarity)
            .with_weight_profile("conversation_history");

        // Apply session filter
        if session_only {
            if let Some(ref sid) = session_id {
                options = options.with_session_filter(sid);
            }
        }

        // Apply sequence options
        options.temporal_options.sequence_options = Some(seq_opts);

        // Generate a query embedding - use provided query or a generic context query
        let query_text = query.unwrap_or("conversation context");
        let query_embedding = match self.multi_array_provider.embed_all(query_text).await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "get_conversation_context: Query embedding FAILED");
                return self.tool_error(id, &format!("Query embedding failed: {}", e));
            }
        };

        let results: Vec<TeleologicalSearchResult> = match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "get_conversation_context: Search FAILED");
                return self.tool_error(id, &format!("Search failed: {}", e));
            }
        };

        // Limit results to window size
        let filtered: Vec<_> = results.into_iter().take(window_size as usize).collect();

        // Collect IDs for batch content retrieval
        let ids: Vec<uuid::Uuid> = filtered.iter().map(|r| r.fingerprint.id).collect();

        // Batch retrieve content if requested (FAIL FAST)
        let contents: Vec<Option<String>> = if include_content && !filtered.is_empty() {
            match self.teleological_store.get_content_batch(&ids).await {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "get_conversation_context: Content retrieval FAILED");
                    return self.tool_error(id, &format!("Content retrieval failed: {}", e));
                }
            }
        } else {
            vec![None; ids.len()]
        };

        // Batch retrieve source metadata (FAIL FAST)
        let source_metadata: Vec<Option<SourceMetadata>> = if !filtered.is_empty() {
            match self.teleological_store.get_source_metadata_batch(&ids).await {
                Ok(m) => m,
                Err(e) => {
                    error!(error = %e, "get_conversation_context: Source metadata retrieval FAILED");
                    return self.tool_error(id, &format!("Source metadata retrieval failed: {}", e));
                }
            }
        } else {
            vec![]
        };

        // Build response
        let results_json: Vec<_> = filtered
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut entry = json!({
                    "fingerprintId": r.fingerprint.id.to_string(),
                    "similarity": r.similarity,
                    "dominantEmbedder": embedder_names::name(r.dominant_embedder())
                });

                if include_content {
                    entry["content"] = match contents.get(i).and_then(|c| c.as_ref()) {
                        Some(c) => json!(c),
                        None => serde_json::Value::Null,
                    };
                }

                // Add sequence info
                if let Some(Some(ref metadata)) = source_metadata.get(i) {
                    if let Some(seq) = metadata.session_sequence {
                        entry["sequenceInfo"] = json!({
                            "sessionId": metadata.session_id,
                            "sessionSequence": seq,
                            "positionLabel": compute_position_label(seq, current_seq)
                        });
                    }
                    entry["sourceType"] = json!(format!("{}", metadata.source_type));
                }

                entry
            })
            .collect();

        self.tool_result(
            id,
            json!({
                "results": results_json,
                "count": results_json.len(),
                "currentSequence": current_seq,
                "direction": direction_str,
                "windowSize": window_size
            }),
        )
    }

    /// get_session_timeline tool implementation.
    ///
    /// Returns an ordered timeline of all session memories with sequence numbers.
    pub(crate) async fn call_get_session_timeline(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse session ID (default: current)
        let session_id = args
            .get("sessionId")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| self.get_session_id());

        let session_id = match session_id {
            Some(sid) => sid,
            None => return self.tool_error(id, "No session ID available"),
        };

        // Parse and validate limit (FAIL FAST)
        let limit = match validate_limit(args.get("limit")) {
            Ok(v) => v as usize,
            Err(msg) => {
                error!(error = %msg, "get_session_timeline: limit validation FAILED");
                return self.tool_error(id, &msg);
            }
        };
        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Parse include content
        let include_content = args
            .get("includeContent")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        debug!(
            session_id = %session_id,
            limit = limit,
            offset = offset,
            "get_session_timeline: Fetching session timeline"
        );

        // Build search options for session timeline - use sequence_navigation profile
        let options = TeleologicalSearchOptions::quick(limit + offset)
            .with_session_filter(&session_id)
            .with_weight_profile("sequence_navigation");

        // Use a generic query to get all memories in the session
        let query_embedding = match self.multi_array_provider.embed_all("session timeline").await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "get_session_timeline: Query embedding FAILED");
                return self.tool_error(id, &format!("Query embedding failed: {}", e));
            }
        };

        let results: Vec<TeleologicalSearchResult> = match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "get_session_timeline: Search FAILED");
                return self.tool_error(id, &format!("Search failed: {}", e));
            }
        };

        // Get current sequence for position labels
        let current_seq = self.current_sequence();

        // Collect IDs for batch operations
        let ids: Vec<uuid::Uuid> = results.iter().map(|r| r.fingerprint.id).collect();

        // Batch retrieve content if requested (FAIL FAST)
        let contents: Vec<Option<String>> = if include_content && !results.is_empty() {
            match self.teleological_store.get_content_batch(&ids).await {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "get_session_timeline: Content retrieval FAILED");
                    return self.tool_error(id, &format!("Content retrieval failed: {}", e));
                }
            }
        } else {
            vec![None; ids.len()]
        };

        // Batch retrieve source metadata (FAIL FAST)
        let source_metadata: Vec<Option<SourceMetadata>> = if !results.is_empty() {
            match self.teleological_store.get_source_metadata_batch(&ids).await {
                Ok(m) => m,
                Err(e) => {
                    error!(error = %e, "get_session_timeline: Source metadata retrieval FAILED");
                    return self.tool_error(id, &format!("Source metadata retrieval failed: {}", e));
                }
            }
        } else {
            vec![]
        };

        // Build response - filter to those with sequence info and sort
        let mut results_with_seq: Vec<(u64, serde_json::Value)> = results
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let metadata = source_metadata.get(i).and_then(|m| m.as_ref())?;
                let seq = metadata.session_sequence?;

                let mut entry = json!({
                    "fingerprintId": r.fingerprint.id.to_string(),
                    "sessionSequence": seq,
                    "positionLabel": compute_position_label(seq, current_seq),
                    "sourceType": format!("{}", metadata.source_type)
                });

                if include_content {
                    entry["content"] = match contents.get(i).and_then(|c| c.as_ref()) {
                        Some(c) => json!(c),
                        None => serde_json::Value::Null,
                    };
                }

                Some((seq, entry))
            })
            .collect();

        // Sort by sequence ascending
        results_with_seq.sort_by_key(|(seq, _)| *seq);

        // Apply pagination
        let paginated: Vec<serde_json::Value> = results_with_seq
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(_, entry)| entry)
            .collect();

        self.tool_result(
            id,
            json!({
                "sessionId": session_id,
                "timeline": paginated,
                "count": paginated.len(),
                "currentSequence": current_seq,
                "offset": offset,
                "limit": limit
            }),
        )
    }

    /// traverse_memory_chain tool implementation.
    ///
    /// Navigates through a chain of memories starting from an anchor point.
    pub(crate) async fn call_traverse_memory_chain(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse anchor ID (required)
        let anchor_id = match args.get("anchorId").and_then(|v| v.as_str()) {
            Some(id_str) => match uuid::Uuid::parse_str(id_str) {
                Ok(uuid) => uuid,
                Err(_) => return self.tool_error(id, "Invalid anchorId UUID format"),
            },
            None => return self.tool_error(id, "Missing required 'anchorId' parameter"),
        };

        // FAIL FAST: Verify anchor exists in storage
        match self.teleological_store.get_content(anchor_id).await {
            Ok(Some(_)) => {
                debug!(anchor_id = %anchor_id, "traverse_memory_chain: Anchor exists");
            }
            Ok(None) => {
                error!(anchor_id = %anchor_id, "traverse_memory_chain: Anchor not found");
                return self.tool_error(
                    id,
                    &format!("Anchor memory {} not found in storage", anchor_id)
                );
            }
            Err(e) => {
                error!(error = %e, anchor_id = %anchor_id, "traverse_memory_chain: Anchor verification FAILED");
                return self.tool_error(id, &format!("Failed to verify anchor: {}", e));
            }
        }

        // Parse direction
        let direction_str = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("backward");
        let direction = match direction_str {
            "forward" => SequenceDirection::After,
            "backward" => SequenceDirection::Before,
            _ => SequenceDirection::Both,
        };

        // Parse and validate hops (FAIL FAST)
        let hops = match validate_hops(args.get("hops")) {
            Ok(v) => v as u32,
            Err(msg) => {
                error!(error = %msg, "traverse_memory_chain: hops validation FAILED");
                return self.tool_error(id, &msg);
            }
        };

        // Parse semantic filter
        let semantic_filter = args.get("semanticFilter").and_then(|v| v.as_str());
        let min_similarity = args
            .get("minSimilarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.3) as f32;

        // Parse include content
        let include_content = args
            .get("includeContent")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        debug!(
            anchor_id = %anchor_id,
            direction = ?direction,
            hops = hops,
            "traverse_memory_chain: Starting traversal"
        );

        // Build sequence options for traversal
        let seq_opts = SequenceOptions::around(anchor_id)
            .with_direction(direction)
            .with_max_distance(hops as usize);

        // Build search options
        let mut options = TeleologicalSearchOptions::quick(hops as usize * 2)
            .with_min_similarity(min_similarity)
            .with_weight_profile("sequence_navigation");

        options.temporal_options.sequence_options = Some(seq_opts);

        // Use semantic filter or generic query
        let query_text = semantic_filter.unwrap_or("memory chain traversal");
        let query_embedding = match self.multi_array_provider.embed_all(query_text).await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "traverse_memory_chain: Query embedding FAILED");
                return self.tool_error(id, &format!("Query embedding failed: {}", e));
            }
        };

        let results: Vec<TeleologicalSearchResult> = match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "traverse_memory_chain: Search FAILED");
                return self.tool_error(id, &format!("Search failed: {}", e));
            }
        };

        // Limit to requested hops
        let chain: Vec<_> = results.into_iter().take(hops as usize).collect();

        // Get current sequence for position labels
        let current_seq = self.current_sequence();

        // Collect IDs for batch operations
        let ids: Vec<uuid::Uuid> = chain.iter().map(|r| r.fingerprint.id).collect();

        // Batch retrieve content if requested (FAIL FAST)
        let contents: Vec<Option<String>> = if include_content && !chain.is_empty() {
            match self.teleological_store.get_content_batch(&ids).await {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "traverse_memory_chain: Content retrieval FAILED");
                    return self.tool_error(id, &format!("Content retrieval failed: {}", e));
                }
            }
        } else {
            vec![None; ids.len()]
        };

        // Batch retrieve source metadata (FAIL FAST)
        let source_metadata: Vec<Option<SourceMetadata>> = if !chain.is_empty() {
            match self.teleological_store.get_source_metadata_batch(&ids).await {
                Ok(m) => m,
                Err(e) => {
                    error!(error = %e, "traverse_memory_chain: Source metadata retrieval FAILED");
                    return self.tool_error(id, &format!("Source metadata retrieval failed: {}", e));
                }
            }
        } else {
            vec![]
        };

        // Build response
        let chain_json: Vec<_> = chain
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut entry = json!({
                    "fingerprintId": r.fingerprint.id.to_string(),
                    "hopIndex": i,
                    "similarity": r.similarity
                });

                if include_content {
                    entry["content"] = match contents.get(i).and_then(|c| c.as_ref()) {
                        Some(c) => json!(c),
                        None => serde_json::Value::Null,
                    };
                }

                if let Some(Some(ref metadata)) = source_metadata.get(i) {
                    if let Some(seq) = metadata.session_sequence {
                        entry["sequenceInfo"] = json!({
                            "sessionId": metadata.session_id,
                            "sessionSequence": seq,
                            "positionLabel": compute_position_label(seq, current_seq)
                        });
                    }
                    entry["sourceType"] = json!(format!("{}", metadata.source_type));
                }

                entry
            })
            .collect();

        self.tool_result(
            id,
            json!({
                "anchorId": anchor_id.to_string(),
                "direction": direction_str,
                "chain": chain_json,
                "chainLength": chain_json.len(),
                "requestedHops": hops
            }),
        )
    }

    /// compare_session_states tool implementation.
    ///
    /// Compares memory state at different sequence points in a session.
    pub(crate) async fn call_compare_session_states(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse session ID (default: current)
        let session_id = args
            .get("sessionId")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| self.get_session_id());

        let session_id = match session_id {
            Some(sid) => sid,
            None => return self.tool_error(id, "No session ID available"),
        };

        // Parse beforeSequence
        let current_seq = self.current_sequence();
        let before_seq = match args.get("beforeSequence") {
            Some(serde_json::Value::String(s)) if s == "start" => 0,
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(0),
            _ => 0,
        };

        // Parse afterSequence
        let after_seq = match args.get("afterSequence") {
            Some(serde_json::Value::String(s)) if s == "current" => current_seq,
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(current_seq),
            _ => current_seq,
        };

        debug!(
            session_id = %session_id,
            before_seq = before_seq,
            after_seq = after_seq,
            "compare_session_states: Comparing states"
        );

        // Build search options to get all session memories
        let options = TeleologicalSearchOptions::quick(COMPARISON_BATCH_SIZE)
            .with_session_filter(&session_id)
            .with_weight_profile("sequence_navigation");

        // Use a generic query to get all memories in the session
        let query_embedding = match self.multi_array_provider.embed_all("session state comparison").await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "compare_session_states: Query embedding FAILED");
                return self.tool_error(id, &format!("Query embedding failed: {}", e));
            }
        };

        let all_memories: Vec<TeleologicalSearchResult> = match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "compare_session_states: Search FAILED");
                return self.tool_error(id, &format!("Search failed: {}", e));
            }
        };

        // Get source metadata for sequence filtering (FAIL FAST)
        let ids: Vec<uuid::Uuid> = all_memories.iter().map(|r| r.fingerprint.id).collect();
        let source_metadata: Vec<Option<SourceMetadata>> = if !all_memories.is_empty() {
            match self.teleological_store.get_source_metadata_batch(&ids).await {
                Ok(m) => m,
                Err(e) => {
                    error!(error = %e, "compare_session_states: Source metadata retrieval FAILED");
                    return self.tool_error(id, &format!("Source metadata retrieval failed: {}", e));
                }
            }
        } else {
            vec![]
        };

        // Count memories in before and after windows
        let mut before_count = 0usize;
        let mut after_count = 0usize;
        let mut before_by_type: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut after_by_type: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

        for (i, _) in all_memories.iter().enumerate() {
            if let Some(Some(ref metadata)) = source_metadata.get(i) {
                if let Some(seq) = metadata.session_sequence {
                    let source_type = format!("{}", metadata.source_type);
                    if seq <= before_seq {
                        before_count += 1;
                        *before_by_type.entry(source_type).or_insert(0) += 1;
                    } else if seq <= after_seq {
                        after_count += 1;
                        *after_by_type.entry(source_type).or_insert(0) += 1;
                    }
                }
            }
        }

        self.tool_result(
            id,
            json!({
                "sessionId": session_id,
                "beforeState": {
                    "sequenceRange": [0, before_seq],
                    "memoryCount": before_count,
                    "sourceTypeCounts": before_by_type
                },
                "afterState": {
                    "sequenceRange": [before_seq + 1, after_seq],
                    "memoryCount": after_count,
                    "sourceTypeCounts": after_by_type
                },
                "difference": {
                    "addedMemories": after_count,
                    "sequenceSpan": after_seq.saturating_sub(before_seq)
                },
                "currentSequence": current_seq
            }),
        )
    }
}

/// Compute human-readable position label for sequence numbers.
fn compute_position_label(result_seq: u64, current_seq: u64) -> String {
    if result_seq == current_seq {
        "current turn".to_string()
    } else if result_seq < current_seq {
        let turns_ago = current_seq - result_seq;
        if turns_ago == 1 {
            "previous turn".to_string()
        } else {
            format!("{} turns ago", turns_ago)
        }
    } else {
        let turns_ahead = result_seq - current_seq;
        if turns_ahead == 1 {
            "next turn".to_string()
        } else {
            format!("{} turns ahead", turns_ahead)
        }
    }
}
