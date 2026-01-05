//! Tests for domain search types (DomainSearchResult, DomainSearchResults).

#[cfg(test)]
mod tests {
    use crate::search::domain_search::{
        domain_nt_summary, DomainSearchResult, DomainSearchResults,
    };
    use crate::search::{Domain, SemanticSearchResultItem};
    use uuid::Uuid;

    // ========== DomainSearchResult Tests ==========

    #[test]
    fn test_domain_search_result_from_semantic_item() {
        let item = SemanticSearchResultItem {
            faiss_id: 42,
            node_id: Some(Uuid::from_u128(1)),
            distance: 0.4,
            similarity: 0.7,
            domain: Some(Domain::Code),
            relevance_score: None,
        };

        let result = DomainSearchResult::from_semantic_item(
            &item,
            0.9, // Modulated up
            0.5, // net_activation
            Domain::Code,
        );

        assert_eq!(result.faiss_id, 42);
        assert_eq!(result.base_similarity, 0.7);
        assert_eq!(result.modulated_score, 0.9);
        assert!(result.domain_matched);
        assert!((result.modulation_delta() - 0.2).abs() < 1e-6);
        assert!((result.boost_ratio() - (0.9 / 0.7)).abs() < 1e-6);
    }

    #[test]
    fn test_domain_search_result_no_domain_match() {
        let item = SemanticSearchResultItem {
            faiss_id: 42,
            node_id: None,
            distance: 0.4,
            similarity: 0.7,
            domain: Some(Domain::Legal),
            relevance_score: None,
        };

        let result = DomainSearchResult::from_semantic_item(
            &item,
            0.8,
            0.5,
            Domain::Code, // Different from Legal
        );

        assert!(!result.domain_matched);
    }

    #[test]
    fn test_boost_ratio_zero_base() {
        let item = SemanticSearchResultItem {
            faiss_id: 1,
            node_id: None,
            distance: 10.0,
            similarity: 0.0, // Zero base
            domain: None,
            relevance_score: None,
        };

        let result = DomainSearchResult::from_semantic_item(&item, 0.0, 0.0, Domain::General);

        // Should return 1.0 to avoid division by zero
        assert_eq!(result.boost_ratio(), 1.0);
    }

    #[test]
    fn test_modulation_delta_positive_when_boosted() {
        let item = SemanticSearchResultItem {
            faiss_id: 1,
            node_id: None,
            distance: 0.1,
            similarity: 0.5,
            domain: Some(Domain::Code),
            relevance_score: None,
        };

        // With boost
        let result = DomainSearchResult::from_semantic_item(&item, 0.8, 0.5, Domain::Code);
        assert!(
            result.modulation_delta() > 0.0,
            "Modulation delta should be positive when boosted"
        );
    }

    // ========== Domain NT Summary Tests ==========

    #[test]
    fn test_domain_nt_summary() {
        let summary = domain_nt_summary(Domain::Code);
        assert!(summary.contains("Code"));
        assert!(summary.contains("exc="));
        assert!(summary.contains("net="));
        assert!(summary.contains("boost="));
    }

    // ========== DomainSearchResults Tests ==========

    #[test]
    fn test_domain_search_results_empty() {
        let results = DomainSearchResults::empty(Domain::Code);
        assert!(results.is_empty());
        assert_eq!(results.len(), 0);
        assert_eq!(results.query_domain, Domain::Code);
    }

    #[test]
    fn test_domain_search_results_iter() {
        let item1 = SemanticSearchResultItem {
            faiss_id: 1,
            node_id: None,
            distance: 0.1,
            similarity: 0.9,
            domain: Some(Domain::Code),
            relevance_score: None,
        };
        let item2 = SemanticSearchResultItem {
            faiss_id: 2,
            node_id: None,
            distance: 0.2,
            similarity: 0.8,
            domain: Some(Domain::Code),
            relevance_score: None,
        };

        let results = DomainSearchResults {
            items: vec![
                DomainSearchResult::from_semantic_item(&item1, 1.0, 0.5, Domain::Code),
                DomainSearchResult::from_semantic_item(&item2, 0.9, 0.5, Domain::Code),
            ],
            candidates_fetched: 6,
            results_returned: 2,
            query_domain: Domain::Code,
            latency_us: 100,
        };

        assert_eq!(results.len(), 2);
        assert!(!results.is_empty());
        assert_eq!(results.iter().count(), 2);
    }
}
