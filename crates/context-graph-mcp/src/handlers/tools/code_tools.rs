//! E7 code search tool implementation (search_code).
//!
//! # E7 Code Search (V_correctness)
//!
//! This tool leverages E7 dense embeddings (1536D) for code-specific understanding:
//! - E7 finds code patterns, function signatures, implementations
//! - Understands code constructs that E1 misses by treating code as natural language
//! - Blends E7 code scores with E1 semantic for comprehensive results
//!
//! ## Constitution Compliance
//!
//! - ARCH-12: E1 is the semantic foundation, E7 enhances with code understanding
//! - ARCH-13: Uses multi-space search with E1+E7
//! - E7 finds: "Code patterns, function signatures" that E1 misses by "Treating code as NL"
//! - AP-02: All comparisons within respective spaces (no cross-embedder)
//! - FAIL FAST: All errors propagate immediately with logging

use serde_json::json;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use context_graph_core::traits::{SearchStrategy, TeleologicalSearchOptions};

use crate::protocol::JsonRpcId;
use crate::protocol::JsonRpcResponse;

use super::code_dtos::{
    CodeSearchMetadata, CodeSearchMode, CodeSearchResult, CodeSourceInfo, DetectedLanguageInfo,
    SearchCodeRequest, SearchCodeResponse,
};

use super::super::Handlers;

impl Handlers {
    /// search_code tool implementation.
    ///
    /// Finds memories containing code patterns using E7 dense embeddings.
    /// ENHANCES E1 semantic search with code-specific understanding.
    ///
    /// # Algorithm
    ///
    /// 1. Embed the code query using all 13 embedders
    /// 2. Detect programming language indicators in query
    /// 3. Search using E1 semantic as primary (over-fetch for blending)
    /// 4. Compute E7 code similarity for each candidate
    /// 5. Blend E1 and E7 scores: (1-blend)*E1 + blend*E7
    /// 6. Filter by minScore and return top-K results
    ///
    /// # Parameters
    ///
    /// - `query`: The code query to search for (required)
    /// - `topK`: Maximum results to return (1-50, default: 10)
    /// - `minScore`: Minimum blended score threshold (0-1, default: 0.2)
    /// - `blendWithSemantic`: E7 weight in blend (0-1, default: 0.4)
    /// - `includeContent`: Include full content text (default: false)
    pub(crate) async fn call_search_code(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        // Parse and validate request
        let request: SearchCodeRequest = match serde_json::from_value(args.clone()) {
            Ok(req) => req,
            Err(e) => {
                error!(error = %e, "search_code: Failed to parse request");
                return self.tool_error_with_pulse(id, &format!("Invalid request: {}", e));
            }
        };

        if let Err(e) = request.validate() {
            error!(error = %e, "search_code: Validation failed");
            return self.tool_error_with_pulse(id, &e);
        }

        let query = &request.query;
        let top_k = request.top_k;
        let min_score = request.min_score;
        let e7_blend = request.blend_with_semantic;
        let e1_weight = 1.0 - e7_blend;
        let search_mode = request.search_mode;
        let language_hint = request.language_hint.clone();

        // Detect programming language from query (merge with hint if provided)
        let mut detected_language = detect_language(query);
        if let Some(ref hint) = language_hint {
            // Language hint overrides detection if provided
            detected_language.primary_language = Some(hint.clone());
            detected_language.confidence = 1.0;
        }

        info!(
            query_preview = %query.chars().take(50).collect::<String>(),
            top_k = top_k,
            min_score = min_score,
            search_mode = %search_mode,
            e7_blend = e7_blend,
            detected_language = ?detected_language.primary_language,
            "search_code: Starting code search"
        );

        // Step 1: Embed the code query
        let query_embedding = match self.multi_array_provider.embed_all(query).await {
            Ok(output) => output.fingerprint,
            Err(e) => {
                error!(error = %e, "search_code: Query embedding FAILED");
                return self.tool_error_with_pulse(id, &format!("Query embedding failed: {}", e));
            }
        };

        // Step 2: Search for candidates (3x over-fetch for blended reranking)
        let fetch_multiplier = 3;
        let fetch_top_k = top_k * fetch_multiplier;

        // Use code_search weight profile with E7 emphasis
        let options = TeleologicalSearchOptions::quick(fetch_top_k)
            .with_strategy(SearchStrategy::MultiSpace)
            .with_weight_profile("code_search") // E7 emphasis
            .with_min_similarity(0.0); // Get all candidates, filter later

        let candidates = match self
            .teleological_store
            .search_semantic(&query_embedding, options)
            .await
        {
            Ok(results) => results,
            Err(e) => {
                error!(error = %e, "search_code: Candidate search FAILED");
                return self.tool_error_with_pulse(id, &format!("Search failed: {}", e));
            }
        };

        let candidates_evaluated = candidates.len();
        debug!(
            candidates_evaluated = candidates_evaluated,
            "search_code: Evaluating candidates for code scoring"
        );

        // Step 3: Compute code-enhanced scores based on search mode
        let query_e7 = &query_embedding.e7_code;
        let query_e1 = &query_embedding.e1_semantic;

        let mut scored_results: Vec<(Uuid, f32, f32, f32)> = Vec::with_capacity(candidates.len());

        for candidate in &candidates {
            let cand_id = candidate.fingerprint.id;
            let cand_e1 = &candidate.fingerprint.semantic.e1_semantic;
            let cand_e7 = &candidate.fingerprint.semantic.e7_code;

            // E1 cosine similarity (THE semantic foundation per ARCH-12)
            let e1_sim = cosine_similarity(query_e1, cand_e1);

            // E7 cosine similarity (code-specific understanding)
            let e7_sim = cosine_similarity(query_e7, cand_e7);

            // Compute final score based on search mode
            let final_score = match search_mode {
                CodeSearchMode::Hybrid => {
                    // Blend scores: (1-blend)*E1 + blend*E7
                    e1_weight * e1_sim + e7_blend * e7_sim
                }
                CodeSearchMode::E7Only => {
                    // Pure E7 code search
                    e7_sim
                }
                CodeSearchMode::E1WithE7Rerank => {
                    // Use E1 as primary with E7 as tiebreaker/booster
                    // E7 provides a small boost (10%) when it agrees with E1
                    e1_sim + 0.1 * e7_sim
                }
                CodeSearchMode::Pipeline => {
                    // For full pipeline, use weighted combination
                    // E1 foundation with strong E7 enhancement
                    0.6 * e1_sim + 0.4 * e7_sim
                }
            };

            if final_score >= min_score {
                scored_results.push((cand_id, final_score, e1_sim, e7_sim));
            }
        }

        // Step 4: Sort by blended score and take top-K
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_results.truncate(top_k);

        let filtered_count = candidates_evaluated - scored_results.len();

        // Step 5: Build results with optional content
        let result_ids: Vec<Uuid> = scored_results.iter().map(|r| r.0).collect();

        // Get content if requested
        let contents: Vec<Option<String>> = if request.include_content && !result_ids.is_empty() {
            match self.teleological_store.get_content_batch(&result_ids).await {
                Ok(c) => c,
                Err(e) => {
                    warn!(error = %e, "search_code: Content retrieval failed");
                    vec![None; result_ids.len()]
                }
            }
        } else {
            vec![None; result_ids.len()]
        };

        // Get source metadata
        let source_metadata = match self
            .teleological_store
            .get_source_metadata_batch(&result_ids)
            .await
        {
            Ok(m) => m,
            Err(e) => {
                warn!(error = %e, "search_code: Source metadata retrieval failed");
                vec![None; result_ids.len()]
            }
        };

        // Build response
        let mut results: Vec<CodeSearchResult> = Vec::with_capacity(scored_results.len());

        for (i, (memory_id, score, e1_sim, e7_sim)) in scored_results.into_iter().enumerate() {
            let source = source_metadata.get(i).and_then(|m| {
                m.as_ref().map(|meta| CodeSourceInfo {
                    source_type: format!("{}", meta.source_type),
                    file_path: meta.file_path.clone(),
                    hook_type: meta.hook_type.clone(),
                    tool_name: meta.tool_name.clone(),
                })
            });

            results.push(CodeSearchResult {
                memory_id,
                score,
                e1_similarity: e1_sim,
                e7_code_score: e7_sim,
                content: contents.get(i).and_then(|c| c.clone()),
                source,
            });
        }

        let response = SearchCodeResponse {
            query: query.clone(),
            results: results.clone(),
            count: results.len(),
            metadata: CodeSearchMetadata {
                candidates_evaluated,
                filtered_by_score: filtered_count,
                search_mode,
                e7_blend_weight: e7_blend,
                e1_weight,
                language_hint,
                detected_language,
            },
        };

        info!(
            results_found = response.count,
            candidates_evaluated = candidates_evaluated,
            filtered = filtered_count,
            "search_code: Completed code-enhanced search"
        );

        self.tool_result_with_pulse(id, serde_json::to_value(response).unwrap_or_else(|_| json!({})))
    }
}

/// Compute cosine similarity between two dense vectors.
///
/// Returns 0.0 if either vector is empty or has zero norm.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Detect programming language indicators in query.
///
/// Returns language info with confidence and indicators.
fn detect_language(query: &str) -> DetectedLanguageInfo {
    let query_lower = query.to_lowercase();

    // Language detection patterns
    let patterns = [
        // Rust
        ("rust", vec!["impl ", "fn ", "let mut", "struct ", "enum ", "pub fn", "trait ", "cargo", ".rs", "unwrap(", "match "]),
        // Python
        ("python", vec!["def ", "import ", "self.", "__init__", "python", ".py", "pip ", "async def", "await ", "pytest"]),
        // JavaScript/TypeScript
        ("javascript", vec!["function ", "const ", "let ", "var ", "=>", "async ", "await ", ".js", "npm ", "node ", "react", "vue", "angular"]),
        ("typescript", vec!["interface ", ": string", ": number", ": boolean", ".ts", "type ", "<T>"]),
        // Go
        ("go", vec!["func ", "package ", "import ", "go ", "goroutine", "chan ", "defer ", ".go", "golang"]),
        // Java
        ("java", vec!["public class", "private ", "protected ", "static void", "extends ", "implements ", ".java", "maven", "gradle", "@Override"]),
        // C/C++
        ("cpp", vec!["#include", "int main", "void ", "std::", "cout <<", "cin >>", ".cpp", ".hpp", ".h", "template<"]),
        // SQL
        ("sql", vec!["select ", "from ", "where ", "join ", "insert ", "update ", "delete ", "create table", "alter ", "sql"]),
    ];

    let mut best_match: Option<(&str, Vec<String>, f32)> = None;

    for (lang, indicators) in patterns {
        let mut matched_indicators = Vec::new();

        for indicator in indicators {
            if query_lower.contains(indicator) {
                matched_indicators.push(indicator.trim().to_string());
            }
        }

        if !matched_indicators.is_empty() {
            let confidence = (matched_indicators.len() as f32 / 3.0).min(1.0);

            if best_match.is_none() || confidence > best_match.as_ref().unwrap().2 {
                best_match = Some((lang, matched_indicators, confidence));
            }
        }
    }

    match best_match {
        Some((lang, indicators, confidence)) => DetectedLanguageInfo {
            primary_language: Some(lang.to_string()),
            confidence,
            indicators,
        },
        None => DetectedLanguageInfo {
            primary_language: None,
            confidence: 0.0,
            indicators: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_detect_language_rust() {
        let query = "impl async fn handler with Result<T, Error>";
        let info = detect_language(query);
        assert_eq!(info.primary_language, Some("rust".to_string()));
        assert!(info.confidence > 0.0);
        assert!(!info.indicators.is_empty());
    }

    #[test]
    fn test_detect_language_python() {
        let query = "def process_data(self) with async def and await";
        let info = detect_language(query);
        assert_eq!(info.primary_language, Some("python".to_string()));
        assert!(info.confidence > 0.0);
    }

    #[test]
    fn test_detect_language_javascript() {
        let query = "const handler = async () => { await fetch() }";
        let info = detect_language(query);
        assert_eq!(info.primary_language, Some("javascript".to_string()));
    }

    #[test]
    fn test_detect_language_no_match() {
        let query = "hello world general text";
        let info = detect_language(query);
        assert!(info.primary_language.is_none());
        assert!((info.confidence - 0.0).abs() < 0.001);
        assert!(info.indicators.is_empty());
    }

    #[test]
    fn test_detect_language_sql() {
        let query = "SELECT * FROM users WHERE id = 1 JOIN orders";
        let info = detect_language(query);
        assert_eq!(info.primary_language, Some("sql".to_string()));
        assert!(info.confidence >= 0.66); // At least 2 indicators
    }

    #[test]
    fn test_detect_language_go() {
        let query = "func main() package main with goroutine";
        let info = detect_language(query);
        assert_eq!(info.primary_language, Some("go".to_string()));
    }
}
