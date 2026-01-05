//! Tests for the retrieval module.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use context_graph_core::types::JohariQuadrant;

    use crate::config::JohariConfig;
    use crate::johari::retrieval::{
        get_retrieval_weight, get_suggested_action, QuadrantRetrieval, SuggestedAction,
    };

    #[test]
    fn test_suggested_action_mapping() {
        assert_eq!(
            get_suggested_action(JohariQuadrant::Open),
            SuggestedAction::DirectRecall
        );
        assert_eq!(
            get_suggested_action(JohariQuadrant::Blind),
            SuggestedAction::EpistemicAction
        );
        assert_eq!(
            get_suggested_action(JohariQuadrant::Hidden),
            SuggestedAction::GetNeighborhood
        );
        assert_eq!(
            get_suggested_action(JohariQuadrant::Unknown),
            SuggestedAction::TriggerDream
        );
    }

    #[test]
    fn test_retrieval_weight() {
        assert_eq!(get_retrieval_weight(JohariQuadrant::Open), 1.0);
        assert_eq!(get_retrieval_weight(JohariQuadrant::Blind), 0.7);
        assert_eq!(get_retrieval_weight(JohariQuadrant::Hidden), 0.3);
        assert_eq!(get_retrieval_weight(JohariQuadrant::Unknown), 0.5);
    }

    #[test]
    fn test_suggested_action_description() {
        let action = SuggestedAction::DirectRecall;
        assert!(action.description().contains("Direct memory recall"));

        let action = SuggestedAction::EpistemicAction;
        assert!(action.description().contains("Epistemic"));
    }

    #[test]
    fn test_suggested_action_urgency() {
        assert_eq!(SuggestedAction::DirectRecall.urgency(), 1.0);
        assert_eq!(SuggestedAction::EpistemicAction.urgency(), 0.7);
        assert_eq!(SuggestedAction::GetNeighborhood.urgency(), 0.5);
        assert_eq!(SuggestedAction::TriggerDream.urgency(), 0.8);
    }

    #[test]
    fn test_suggested_action_display() {
        assert_eq!(format!("{}", SuggestedAction::DirectRecall), "DirectRecall");
        assert_eq!(
            format!("{}", SuggestedAction::EpistemicAction),
            "EpistemicAction"
        );
        assert_eq!(
            format!("{}", SuggestedAction::GetNeighborhood),
            "GetNeighborhood"
        );
        assert_eq!(format!("{}", SuggestedAction::TriggerDream), "TriggerDream");
    }

    #[test]
    fn test_suggested_action_all() {
        let all = SuggestedAction::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&SuggestedAction::DirectRecall));
        assert!(all.contains(&SuggestedAction::EpistemicAction));
        assert!(all.contains(&SuggestedAction::GetNeighborhood));
        assert!(all.contains(&SuggestedAction::TriggerDream));
    }

    #[test]
    fn test_quadrant_retrieval_new() {
        let config = JohariConfig::default();
        let retrieval = QuadrantRetrieval::new(&config);

        assert_eq!(
            retrieval.get_weight(JohariQuadrant::Open),
            config.open_weight
        );
        assert_eq!(
            retrieval.get_weight(JohariQuadrant::Blind),
            config.blind_weight
        );
    }

    #[test]
    fn test_quadrant_retrieval_default_weights() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert_eq!(retrieval.get_weight(JohariQuadrant::Open), 1.0);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Blind), 0.7);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Hidden), 0.3);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Unknown), 0.5);
    }

    #[test]
    fn test_quadrant_retrieval_custom_weights() {
        let retrieval =
            QuadrantRetrieval::with_custom_weights(0.9, 0.8, 0.5, 0.6).expect("Valid weights");

        assert_eq!(retrieval.get_weight(JohariQuadrant::Open), 0.9);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Blind), 0.8);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Hidden), 0.5);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Unknown), 0.6);
    }

    #[test]
    fn test_quadrant_retrieval_invalid_weights() {
        // Negative weight
        let result = QuadrantRetrieval::with_custom_weights(-0.1, 0.5, 0.5, 0.5);
        assert!(result.is_err());

        // Weight > 2.0
        let result = QuadrantRetrieval::with_custom_weights(1.0, 2.5, 0.5, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_quadrant_retrieval_get_action() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert_eq!(
            retrieval.get_action(JohariQuadrant::Open),
            SuggestedAction::DirectRecall
        );
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Blind),
            SuggestedAction::EpistemicAction
        );
    }

    #[test]
    fn test_quadrant_retrieval_apply_weight() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        // Open has weight 1.0
        assert_eq!(retrieval.apply_weight(JohariQuadrant::Open, 0.8), 0.8);

        // Hidden has weight 0.3
        let weighted = retrieval.apply_weight(JohariQuadrant::Hidden, 0.8);
        assert!((weighted - 0.24).abs() < 0.001);

        // Blind has weight 0.7
        let weighted = retrieval.apply_weight(JohariQuadrant::Blind, 1.0);
        assert!((weighted - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_quadrant_retrieval_all_weights() {
        let retrieval = QuadrantRetrieval::with_default_weights();
        let (open, blind, hidden, unknown) = retrieval.all_weights();

        assert_eq!(open, 1.0);
        assert_eq!(blind, 0.7);
        assert_eq!(hidden, 0.3);
        assert_eq!(unknown, 0.5);
    }

    #[test]
    fn test_quadrant_retrieval_should_include_by_default() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert!(retrieval.should_include_by_default(JohariQuadrant::Open));
        assert!(retrieval.should_include_by_default(JohariQuadrant::Blind));
        assert!(!retrieval.should_include_by_default(JohariQuadrant::Hidden));
        assert!(retrieval.should_include_by_default(JohariQuadrant::Unknown));
    }

    #[test]
    fn test_quadrant_retrieval_individual_weight_getters() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert_eq!(retrieval.open_weight(), 1.0);
        assert_eq!(retrieval.blind_weight(), 0.7);
        assert_eq!(retrieval.hidden_weight(), 0.3);
        assert_eq!(retrieval.unknown_weight(), 0.5);
    }

    #[test]
    fn test_quadrant_retrieval_default() {
        let retrieval = QuadrantRetrieval::default();
        assert_eq!(retrieval.open_weight(), 1.0);
    }

    #[test]
    fn test_constitution_compliance() {
        // Verify mappings match constitution.yaml specification
        let retrieval = QuadrantRetrieval::with_default_weights();

        // Open -> direct recall
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Open),
            SuggestedAction::DirectRecall
        );

        // Blind -> discovery (epistemic_action/dream)
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Blind),
            SuggestedAction::EpistemicAction
        );

        // Hidden -> private (get_neighborhood)
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Hidden),
            SuggestedAction::GetNeighborhood
        );

        // Unknown -> frontier
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Unknown),
            SuggestedAction::TriggerDream
        );
    }
}
