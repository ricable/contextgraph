//! Tests for Poincare ball Mobius operations.
//!
//! REAL DATA ONLY, NO MOCKS (per constitution REQ-KG-TEST)

#[cfg(test)]
#[allow(dead_code, clippy::module_inception)]
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

    #[allow(dead_code)]
    fn make_point_2d(x: f32, y: f32) -> PoincarePoint {
        let mut coords = [0.0f32; 64];
        coords[0] = x;
        coords[1] = y;
        PoincarePoint::from_coords(coords)
    }

    // ========== CONSTRUCTION TESTS ==========

    #[test]
    fn test_new_creates_ball_with_config() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        let ball = PoincareBall::new(config.clone());
        assert_eq!(ball.config().curvature, -0.5);
    }

    #[test]
    fn test_config_accessor() {
        let ball = default_ball();
        assert_eq!(ball.config().dim, 64);
        assert_eq!(ball.config().curvature, -1.0);
    }

    // ========== MOBIUS ADDITION TESTS ==========

    #[test]
    fn test_mobius_add_with_origin_returns_other() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let point = make_point(0.3);

        // x + 0 = x
        let result = ball.mobius_add(&point, &origin);
        assert!((result.coords[0] - 0.3).abs() < 1e-5);

        // 0 + y = y
        let result2 = ball.mobius_add(&origin, &point);
        assert!((result2.coords[0] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_mobius_add_result_inside_ball() {
        let ball = default_ball();
        let p1 = make_point(0.5);
        let p2 = make_point(0.3);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid(), "Mobius add result must be inside ball");
    }

    #[test]
    fn test_mobius_add_near_boundary() {
        let ball = default_ball();
        // Points close to boundary
        let p1 = make_point(0.9);
        let p2 = make_point(0.8);

        let result = ball.mobius_add(&p1, &p2);
        assert!(result.is_valid_for_config(ball.config()));
    }

    #[test]
    fn test_mobius_add_opposite_directions() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(-0.3);

        // Opposite points should partially cancel
        let result = ball.mobius_add(&p1, &p2);
        assert!(result.norm() < 0.3);
    }

    // ========== DISTANCE TESTS ==========

    #[test]
    fn test_distance_same_point_is_zero() {
        let ball = default_ball();
        let point = make_point(0.5);
        assert_eq!(ball.distance(&point, &point), 0.0);
    }

    #[test]
    fn test_distance_origin_to_origin_is_zero() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        assert_eq!(ball.distance(&origin, &origin), 0.0);
    }

    #[test]
    fn test_distance_is_symmetric() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(0.6);

        let d1 = ball.distance(&p1, &p2);
        let d2 = ball.distance(&p2, &p1);
        assert!((d1 - d2).abs() < 1e-6, "Distance must be symmetric");
    }

    #[test]
    fn test_distance_is_nonnegative() {
        let ball = default_ball();
        let p1 = make_point(0.3);
        let p2 = make_point(-0.5);
        assert!(ball.distance(&p1, &p2) >= 0.0);
    }

    #[test]
    fn test_distance_triangle_inequality() {
        let ball = default_ball();
        let p1 = make_point(0.1);
        let p2 = make_point(0.3);
        let p3 = make_point(0.5);

        let d12 = ball.distance(&p1, &p2);
        let d23 = ball.distance(&p2, &p3);
        let d13 = ball.distance(&p1, &p3);

        // d(p1, p3) <= d(p1, p2) + d(p2, p3)
        assert!(d13 <= d12 + d23 + 1e-6, "Triangle inequality violated");
    }

    #[test]
    fn test_distance_from_origin_monotonic() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();

        // Distance increases as we move further from origin
        let p1 = make_point(0.1);
        let p2 = make_point(0.5);
        let p3 = make_point(0.9);

        let d1 = ball.distance(&origin, &p1);
        let d2 = ball.distance(&origin, &p2);
        let d3 = ball.distance(&origin, &p3);

        assert!(d1 < d2, "d(0, 0.1) < d(0, 0.5)");
        assert!(d2 < d3, "d(0, 0.5) < d(0, 0.9)");
    }

    #[test]
    fn test_distance_near_boundary_larger() {
        let ball = default_ball();
        // In hyperbolic space, distances near boundary are larger
        let origin = PoincarePoint::origin();
        let near_boundary = make_point(0.99);

        let d = ball.distance(&origin, &near_boundary);
        // For c=-1, d(0, r) = 2 * arctanh(r), so d(0, 0.99) ~ 5.3
        assert!(d > 4.0, "Distance near boundary should be large: {}", d);
    }

    // ========== EXP MAP TESTS ==========

    #[test]
    fn test_exp_map_zero_tangent_returns_base() {
        let ball = default_ball();
        let base = make_point(0.3);
        let zero_v = [0.0f32; 64];

        let result = ball.exp_map(&base, &zero_v);
        assert!((result.coords[0] - base.coords[0]).abs() < 1e-6);
    }

    #[test]
    fn test_exp_map_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let mut v = [0.0f32; 64];
        v[0] = 0.5;

        let result = ball.exp_map(&origin, &v);
        assert!(result.is_valid());
        assert!(result.coords[0] > 0.0, "Should move in direction of v");
    }

    #[test]
    fn test_exp_map_result_inside_ball() {
        let ball = default_ball();
        let base = make_point(0.5);
        let mut v = [0.0f32; 64];
        v[0] = 10.0; // Large tangent vector

        let result = ball.exp_map(&base, &v);
        assert!(
            result.is_valid_for_config(ball.config()),
            "exp_map result must be inside ball"
        );
    }

    // ========== LOG MAP TESTS ==========

    #[test]
    fn test_log_map_same_point_is_zero() {
        let ball = default_ball();
        let point = make_point(0.3);

        let v = ball.log_map(&point, &point);
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(v_norm < 1e-6, "log_map(x, x) should be zero vector");
    }

    #[test]
    fn test_log_map_from_origin() {
        let ball = default_ball();
        let origin = PoincarePoint::origin();
        let target = make_point(0.3);

        let v = ball.log_map(&origin, &target);
        // Should point in positive x direction
        assert!(v[0] > 0.0);
    }
}
