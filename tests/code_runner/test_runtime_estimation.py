"""Tests for runtime estimation heuristics."""

from src.code_runner import estimate_runtime


class TestRuntimeEstimation:
    """Tests for estimate_runtime function."""

    def test_estimate_runtime_heuristics(self):
        """Test that heuristics increase runtime estimate."""
        base = estimate_runtime("import meep")

        code_3d = "mp.Vector3(1,1,1)\n" * 3
        est_3d = estimate_runtime(code_3d)
        assert est_3d["estimated_minutes"] > base["estimated_minutes"]
        assert est_3d["features_detected"]["is_3d"]

        code_sweep = "for i in range(10): pass"
        est_sweep = estimate_runtime(code_sweep)
        assert est_sweep["estimated_minutes"] > base["estimated_minutes"]
        assert est_sweep["features_detected"]["has_sweep"]

    def test_estimate_runtime_timeout_buffer(self):
        """Test timeout is buffered."""
        est = estimate_runtime("import meep")
        assert est["recommended_timeout_seconds"] >= est["estimated_minutes"] * 60 * 2

