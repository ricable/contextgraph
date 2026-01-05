//! Tests for net activation and modulation formulas.

#[cfg(test)]
mod tests {
    use crate::search::domain_search::{compute_net_activation, expected_domain_boost};
    use crate::search::Domain;
    use context_graph_core::marblestone::NeurotransmitterWeights;

    // ========== Net Activation Formula Tests ==========

    #[test]
    fn test_canonical_formula_net_activation() {
        // Verify net_activation formula: excitatory - inhibitory + (modulatory * 0.5)
        // These values come from NeurotransmitterWeights::for_domain() in context-graph-core

        // Code: e=0.6, i=0.3, m=0.4 -> 0.6 - 0.3 + (0.4 * 0.5) = 0.5
        let code_nt = NeurotransmitterWeights::for_domain(Domain::Code);
        let net = compute_net_activation(&code_nt);
        assert!(
            (net - 0.5).abs() < 0.01,
            "Code net_activation should be 0.5, got {}",
            net
        );

        // Creative: e=0.8, i=0.1, m=0.6 -> 0.8 - 0.1 + (0.6 * 0.5) = 0.7 + 0.3 = 1.0
        let creative_nt = NeurotransmitterWeights::for_domain(Domain::Creative);
        let net = compute_net_activation(&creative_nt);
        assert!(
            (net - 1.0).abs() < 0.01,
            "Creative net_activation should be 1.0, got {}",
            net
        );

        // General: e=0.5, i=0.2, m=0.3 -> 0.5 - 0.2 + (0.3 * 0.5) = 0.3 + 0.15 = 0.45
        let general_nt = NeurotransmitterWeights::for_domain(Domain::General);
        let net = compute_net_activation(&general_nt);
        assert!(
            (net - 0.45).abs() < 0.01,
            "General net_activation should be 0.45, got {}",
            net
        );

        // Legal: e=0.4, i=0.4, m=0.2 -> 0.4 - 0.4 + (0.2 * 0.5) = 0 + 0.1 = 0.1
        let legal_nt = NeurotransmitterWeights::for_domain(Domain::Legal);
        let net = compute_net_activation(&legal_nt);
        assert!(
            (net - 0.1).abs() < 0.01,
            "Legal net_activation should be 0.1, got {}",
            net
        );

        // Medical: e=0.5, i=0.3, m=0.5 -> 0.5 - 0.3 + (0.5 * 0.5) = 0.2 + 0.25 = 0.45
        let medical_nt = NeurotransmitterWeights::for_domain(Domain::Medical);
        let net = compute_net_activation(&medical_nt);
        assert!(
            (net - 0.45).abs() < 0.01,
            "Medical net_activation should be 0.45, got {}",
            net
        );

        // Research: e=0.6, i=0.2, m=0.5 -> 0.6 - 0.2 + (0.5 * 0.5) = 0.4 + 0.25 = 0.65
        let research_nt = NeurotransmitterWeights::for_domain(Domain::Research);
        let net = compute_net_activation(&research_nt);
        assert!(
            (net - 0.65).abs() < 0.01,
            "Research net_activation should be 0.65, got {}",
            net
        );
    }

    // ========== Modulation Formula Tests ==========

    #[test]
    fn test_modulation_formula_application() {
        // Test the modulation formula: base * (1.0 + net_activation + domain_bonus)

        let base = 0.8f32;
        let net_activation = 0.5f32;
        let domain_bonus = 0.1f32;

        let modulated = base * (1.0 + net_activation + domain_bonus);
        let modulated = modulated.clamp(0.0, 1.0);

        // 0.8 * 1.6 = 1.28, clamped to 1.0
        assert!(
            (modulated - 1.0).abs() < 1e-6,
            "Modulated should be clamped to 1.0"
        );
    }

    #[test]
    fn test_modulation_no_domain_bonus() {
        let base = 0.5f32;
        let net_activation = 0.3f32;
        let domain_bonus = 0.0f32; // No match

        let modulated = base * (1.0 + net_activation + domain_bonus);
        // 0.5 * 1.3 = 0.65
        assert!((modulated - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_modulation_clamping_high() {
        // Test that high modulation values get clamped to 1.0
        let base = 0.9f32;
        let net_activation = 1.0f32; // Creative domain
        let domain_bonus = 0.1f32;

        let modulated = base * (1.0 + net_activation + domain_bonus);
        // 0.9 * 2.1 = 1.89, clamped to 1.0
        let clamped = modulated.clamp(0.0, 1.0);
        assert!(
            (clamped - 1.0).abs() < 1e-6,
            "Should be clamped to 1.0, got {}",
            clamped
        );
    }

    // ========== Expected Domain Boost Tests ==========

    #[test]
    fn test_expected_domain_boost() {
        // Code: net_activation = 0.5
        // boost = 1.0 + 0.5 + 0.1 = 1.6
        let code_boost = expected_domain_boost(Domain::Code);
        assert!(
            (code_boost - 1.6).abs() < 0.01,
            "Code boost should be 1.6, got {}",
            code_boost
        );

        // Creative: net_activation = 1.0
        // boost = 1.0 + 1.0 + 0.1 = 2.1
        let creative_boost = expected_domain_boost(Domain::Creative);
        assert!(
            (creative_boost - 2.1).abs() < 0.01,
            "Creative boost should be 2.1, got {}",
            creative_boost
        );

        // Legal: net_activation = 0.1
        // boost = 1.0 + 0.1 + 0.1 = 1.2
        let legal_boost = expected_domain_boost(Domain::Legal);
        assert!(
            (legal_boost - 1.2).abs() < 0.01,
            "Legal boost should be 1.2, got {}",
            legal_boost
        );

        // General: net_activation = 0.45
        // boost = 1.0 + 0.45 + 0.1 = 1.55
        let general_boost = expected_domain_boost(Domain::General);
        assert!(
            (general_boost - 1.55).abs() < 0.01,
            "General boost should be 1.55, got {}",
            general_boost
        );

        // General should have lower boost than Code
        assert!(
            general_boost < code_boost,
            "General boost should be lower than Code"
        );
    }

    #[test]
    fn test_all_domains_have_valid_boost() {
        // Ensure all domains produce valid boost ratios
        for domain in Domain::all() {
            let boost = expected_domain_boost(domain);
            assert!(
                boost > 1.0,
                "{:?} should have boost > 1.0, got {}",
                domain,
                boost
            );
            assert!(
                boost < 3.0,
                "{:?} should have boost < 3.0, got {}",
                domain,
                boost
            );
        }
    }
}
