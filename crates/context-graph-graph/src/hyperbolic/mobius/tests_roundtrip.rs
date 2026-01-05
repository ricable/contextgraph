//! Round-trip and edge case tests for Poincare ball operations.
//!
//! Critical tests for exp/log map inverse relationships.

#[cfg(test)]
mod tests {
    use crate::config::HyperbolicConfig;
    use crate::hyperbolic::mobius::PoincareBall;
    use crate::hyperbolic::poincare::PoincarePoint;

    fn default_ball() -> PoincareBall {
        PoincareBall::new(HyperbolicConfig::default())
    }

    fn make_point(first_coord: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = first_coord;
        PoincarePoint::from_coords(coords)
    }

    fn make_point_2d(x: f32, y: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = x;
        coords[1] = y;
        PoincarePoint::from_coords(coords)
    }

    // ========== ROUND-TRIP TESTS (CRITICAL) ==========

    #[test]
    fn test_exp_log_roundtrip_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let mut v_orig = [0.0f32; 64];
        v_orig[0] = 0.5;
        v_orig[1] = 0.3;

        // exp_map -> log_map should recover original tangent vector
        let point = ball.exp_map(&origin, &v_orig);
        let v_recovered = ball.log_map(&origin, &point);

        for i in 0..64 {
            assert!(
                (v_orig[i] - v_recovered[i]).abs() < 1e-4,
                "Roundtrip failed at index {}: {} vs {}",
                i,
                v_orig[i],
                v_recovered[i]
            );
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let ball = default_ball();
        let base = make_point(0.2);
        let target = make_point(0.5);

        // log_map -> exp_map should approximately recover target
        let v = ball.log_map(&base, &target);
        let recovered = ball.exp_map(&base, &v);

        for i in 0..64 {
            assert!(
                (target.coords[i] - recovered.coords[i]).abs() < 1e-4,
                "Roundtrip failed at index {}: {} vs {}",
                i,
                target.coords[i],
                recovered.coords[i]
            );
        }
    }

    // ========== EDGE CASES ==========

    #[test]
    fn test_distance_with_nan_coords_returns_nan() {
        let ball = default_ball();
        let mut coords = [0.0f32; 64];
        coords[0] = f32::NAN;
        let p1 = PoincarePoint::from_coords(coords);
        let p2 = PoincarePoint::origin();

        let d = ball.distance(&p1, &p2);
        assert!(d.is_nan(), "Distance with NaN input should be NaN");
    }

    #[test]
    fn test_mobius_add_handles_small_denominator() {
        let ball = default_ball();
        // Create points that might cause small denominator
        let mut coords1 = [0.0f32; 64];
        let mut coords2 = [0.0f32; 64];
        coords1[0] = 0.99;
        coords2[0] = -0.99;

        let p1 = PoincarePoint::from_coords(coords1);
        let p2 = PoincarePoint::from_coords(coords2);

        let result = ball.mobius_add(&p1, &p2);
        // Should not panic or produce NaN
        assert!(
            !result.coords[0].is_nan(),
            "Should handle small denominator"
        );
        assert!(result.is_valid());
    }

    #[test]
    fn test_custom_curvature() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        let ball = PoincareBall::new(config);

        let p1 = PoincarePoint::origin();
        let p2 = make_point(0.5);

        let d = ball.distance(&p1, &p2);
        // With lower curvature magnitude, distances should be different
        assert!(d > 0.0);
        assert!(!d.is_nan());
    }

    // ========== MATHEMATICAL PROPERTY TESTS ==========

    #[test]
    fn test_distance_formula_verification() {
        // Verify against known formula: d(0, r) = 2 * arctanh(r) for c=-1
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let r = 0.5;
        let point = make_point(r);

        let computed = ball.distance(&origin, &point);
        let expected = 2.0 * r.atanh();

        assert!(
            (computed - expected).abs() < 1e-5,
            "Distance formula mismatch: computed={}, expected={}",
            computed,
            expected
        );
    }

    // ========== PERFORMANCE SANITY TESTS ==========

    #[test]
    fn test_distance_many_calls() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(0.6);

        // Run many iterations to check for consistency
        let first_distance = ball.distance(&p1, &p2);
        for _ in 0..1000 {
            let d = ball.distance(&p1, &p2);
            assert!(
                (d - first_distance).abs() < 1e-10,
                "Distance should be deterministic"
            );
        }
    }

    // ========== 2D POINT TESTS ==========

    #[test]
    fn test_mobius_add_2d_points() {
        let ball = default_ball();
        let p1 = make_point_2d(0.2, 0.1);
        let p2 = make_point_2d(0.1, 0.2);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid());
        // Both coordinates should be non-zero
        assert!(result.coords[0].abs() > 0.01);
        assert!(result.coords[1].abs() > 0.01);
    }

    #[test]
    fn test_distance_2d_points() {
        let ball = default_ball();
        let p1 = make_point_2d(0.3, 0.4);
        let p2 = make_point_2d(-0.3, 0.4);

        let d = ball.distance(&p1, &p2);
        assert!(d > 0.0);
        assert!(!d.is_nan());
    }

    // ========== ADDITIONAL EDGE CASE TESTS ==========

    #[test]
    fn test_exp_map_large_tangent_vector() {
        let ball = default_ball();
        let base = PoincarePoint::origin();
        let mut v = [0.0f32; 64];
        v[0] = 100.0; // Very large

        let result = ball.exp_map(&base, &v);
        assert!(result.is_valid_for_config(ball.config()));
        // Should approach boundary but stay inside
        assert!(result.norm() > 0.9);
    }

    #[test]
    fn test_log_map_points_near_boundary() {
        let ball = default_ball();
        let p1 = make_point(0.1);
        let p2 = make_point(0.95);

        let v = ball.log_map(&p1, &p2);
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(v_norm > 0.0, "Log map should produce non-zero vector");
        assert!(!v_norm.is_nan());
    }
}
