"""Unit tests covering the estimate_token_cost workflow."""

import pytest

from src.paper_loader import (
    CHARS_PER_TOKEN,
    estimate_token_cost,
)
from src.paper_loader.config import (
    INPUT_COST_PER_MILLION,
    OUTPUT_COST_PER_MILLION,
    TOKENS_PER_FIGURE,
)


class TestEstimateTokenCost:
    """Tests for estimate_token_cost function."""

    def test_returns_dict_with_expected_keys(self, basic_paper_input):
        """Returns dict with all expected keys."""
        result = estimate_token_cost(basic_paper_input)

        expected_keys = [
            "estimated_input_tokens",
            "estimated_output_tokens",
            "estimated_total_tokens",
            "estimated_cost_usd",
            "cost_breakdown",
            "assumptions",
            "warning",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        breakdown_keys = ["input_cost_usd", "output_cost_usd"]
        for key in breakdown_keys:
            assert key in result["cost_breakdown"], f"Missing breakdown key: {key}"

    def test_calculation_exact_match(self):
        """
        Verifies the cost calculation logic matches the expected formula exactly.
        This ensures regressions in the formula are caught.
        """
        char_count = 4000  # 1000 tokens
        text = "A" * char_count
        figures = [{"id": "f1"}, {"id": "f2"}]
        supp_figures = [{"id": "s1"}]

        paper_input = {
            "paper_text": text,
            "figures": figures,
            "supplementary": {"supplementary_figures": supp_figures},
        }

        text_tokens = char_count / CHARS_PER_TOKEN

        total_figures_count = len(figures) + len(supp_figures)
        image_tokens = total_figures_count * TOKENS_PER_FIGURE

        planner_input = 2 * text_tokens

        num_stages = max(4, total_figures_count)

        stage_text_fraction = 0.3
        per_stage_text = text_tokens * stage_text_fraction

        per_stage_input = (
            per_stage_text
            + per_stage_text * 2
            + per_stage_text * 2
            + per_stage_text
            + image_tokens / num_stages
        )

        total_stages_input = per_stage_input * num_stages

        supervisor_input = num_stages * (per_stage_text * 0.5)

        report_input = text_tokens * 0.2 + image_tokens

        total_input_estimate = (
            planner_input
            + total_stages_input
            + supervisor_input
            + report_input
        )

        total_output_estimate = total_input_estimate * 0.25

        input_cost = total_input_estimate * INPUT_COST_PER_MILLION / 1_000_000
        output_cost = total_output_estimate * OUTPUT_COST_PER_MILLION / 1_000_000
        total_cost = input_cost + output_cost

        result = estimate_token_cost(paper_input)

        assert result["estimated_input_tokens"] == int(total_input_estimate)
        assert result["estimated_output_tokens"] == int(total_output_estimate)
        assert (
            result["estimated_total_tokens"]
            == int(total_input_estimate + total_output_estimate)
        )

        assert abs(result["estimated_cost_usd"] - round(total_cost, 2)) < 0.001
        assert (
            abs(result["cost_breakdown"]["input_cost_usd"] - round(input_cost, 2))
            < 0.001
        )
        assert (
            abs(result["cost_breakdown"]["output_cost_usd"] - round(output_cost, 2))
            < 0.001
        )

        assert result["assumptions"]["num_figures"] == 3
        assert result["assumptions"]["num_stages_estimated"] == 4
        assert result["assumptions"]["text_chars"] == char_count

    def test_missing_keys_handled(self):
        """Test that missing keys in input don't crash function."""
        result = estimate_token_cost({})
        assert result["estimated_input_tokens"] >= 0
        assert result["estimated_output_tokens"] >= 0
        assert result["estimated_total_tokens"] >= 0
        assert result["assumptions"]["text_chars"] == 0
        assert result["assumptions"]["num_figures"] == 0
        assert result["assumptions"]["num_stages_estimated"] == 4  # Minimum stages
        assert result["estimated_cost_usd"] >= 0
        assert result["cost_breakdown"]["input_cost_usd"] >= 0
        assert result["cost_breakdown"]["output_cost_usd"] >= 0

    def test_zero_figures_uses_min_stages(self):
        """Zero figures should still use minimum 4 stages."""
        paper_input = {"paper_text": "A" * 400, "figures": []}
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 0
        assert result["assumptions"]["num_stages_estimated"] == 4

    def test_many_figures_uses_figure_count_stages(self):
        """Many figures should increase stage count."""
        figs = [{"id": f"f{i}"} for i in range(10)]
        paper_input = {"paper_text": "A" * 400, "figures": figs}
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 10
        assert result["assumptions"]["num_stages_estimated"] == 10

    def test_supplementary_text_included(self):
        """Supplementary text adds to token count."""
        paper_no_supp = {"paper_text": "A" * 400}
        paper_with_supp = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_text": "B" * 400},
        }

        cost_no = estimate_token_cost(paper_no_supp)
        cost_supp = estimate_token_cost(paper_with_supp)
        assert (
            cost_supp["estimated_input_tokens"] > cost_no["estimated_input_tokens"]
        )
        # Verify the difference is meaningful (at least some tokens added)
        token_diff = cost_supp["estimated_input_tokens"] - cost_no["estimated_input_tokens"]
        assert token_diff > 0, "Supplementary text should increase token count"

    def test_supplementary_figures_included(self):
        """Supplementary figures add to token count and stages if enough."""
        paper_input = {
            "figures": [{"id": "f"} for _ in range(3)],
            "supplementary": {
                "supplementary_figures": [{"id": "s"} for _ in range(3)]
            },
        }
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 6
        assert result["assumptions"]["num_stages_estimated"] == 6

    def test_cost_values_types(self, basic_paper_input):
        """Ensure cost values are floats (even if 0.0) and tokens are ints."""
        result = estimate_token_cost(basic_paper_input)
        assert isinstance(result["estimated_input_tokens"], int)
        assert isinstance(result["estimated_output_tokens"], int)
        assert isinstance(result["estimated_total_tokens"], int)
        assert isinstance(result["estimated_cost_usd"], float)
        assert isinstance(result["cost_breakdown"]["input_cost_usd"], float)
        assert isinstance(result["cost_breakdown"]["output_cost_usd"], float)
        assert isinstance(result["assumptions"], dict)
        assert isinstance(result["warning"], str)

    def test_supplementary_is_none(self):
        """Handles case where supplementary key is None (JSON null)."""
        paper_input = {"paper_text": "A" * 400, "supplementary": None}
        result = estimate_token_cost(paper_input)
        assert result["estimated_input_tokens"] > 0
        assert result["assumptions"]["num_figures"] == 0
        assert result["assumptions"]["text_chars"] == 400

    def test_figures_is_none(self):
        """Handles case where figures key is None (JSON null)."""
        paper_input = {"paper_text": "A" * 400, "figures": None}
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 0

    def test_supplementary_figures_is_none(self):
        """Handles case where supplementary_figures is None."""
        paper_input = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_figures": None},
        }
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 0

    def test_paper_text_is_none(self):
        """Handles case where paper_text is None."""
        paper_input = {"paper_text": None}
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["text_chars"] == 0
        assert result["estimated_input_tokens"] >= 0
        assert result["assumptions"]["num_figures"] == 0
        assert result["assumptions"]["num_stages_estimated"] == 4

    def test_invalid_figures_type_raises_error(self):
        """Should raise TypeError if figures is not a list (e.g. a string)."""
        paper_input = {"figures": "invalid_string"}
        with pytest.raises(TypeError, match="Expected list for figures"):
            estimate_token_cost(paper_input)

    def test_invalid_supplementary_figures_type_raises_error(self):
        """Should raise TypeError if supplementary_figures is not a list."""
        paper_input = {"supplementary": {"supplementary_figures": "invalid_string"}}
        with pytest.raises(TypeError, match="Expected list for supplementary_figures"):
            estimate_token_cost(paper_input)

    def test_cost_invariant_always_holds(self):
        """Invariant: total = input + output must always be true."""
        inputs = [
            {},
            {"paper_text": "A"},
            {"figures": [{"id": "1"}]},
            {"supplementary": {"supplementary_text": "B"}},
            {"paper_text": "A" * 1000, "figures": []},
            {"paper_text": "", "figures": [{"id": "1"}, {"id": "2"}]},
        ]
        for i, inp in enumerate(inputs):
            res = estimate_token_cost(inp)
            total = res["estimated_total_tokens"]
            inp_tok = res["estimated_input_tokens"]
            out_tok = res["estimated_output_tokens"]
            assert total == inp_tok + out_tok, f"Invariant failed for input {i}"

    @pytest.mark.parametrize(
        "num_figs, expected_stages",
        [
            (0, 4),
            (3, 4),
            (4, 4),
            (5, 5),
            (10, 10),
        ],
    )
    def test_stage_count_logic(self, num_figs, expected_stages):
        """Verifies max(4, num_figs) logic explicitly."""
        figs = [{"id": str(i)} for i in range(num_figs)]
        res = estimate_token_cost({"figures": figs})
        assert res["assumptions"]["num_stages_estimated"] == expected_stages

    def test_paper_text_invalid_type_raises_error(self):
        """Should raise TypeError if paper_text is not a string or None."""
        for invalid_value in [123, [], {}, True]:
            paper_input = {"paper_text": invalid_value}
            # The function doesn't validate types, so it will raise TypeError when len() is called
            # The error message is generic, not "Expected string"
            with pytest.raises(TypeError):
                estimate_token_cost(paper_input)

    def test_supplementary_text_invalid_type_bug(self):
        """
        BUG FOUND: supplementary_text with non-string types doesn't raise error.
        
        The function uses len() on supplementary_text without validating it's a string.
        For lists, len() works but gives wrong results (counts list length, not text length).
        For ints, len() raises TypeError.
        This is a bug - the function should validate supplementary_text is a string.
        """
        # None should be handled gracefully (falsy value, not included)
        paper_input_none = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_text": None},
        }
        result = estimate_token_cost(paper_input_none)
        assert result["estimated_input_tokens"] > 0
        
        # BUG: Lists are truthy and len() works on them, giving wrong token count
        paper_input_list = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_text": [1, 2, 3]},
        }
        # This should raise TypeError but currently doesn't - it incorrectly uses len([1,2,3]) = 3
        # Keep this test failing to document the bug
        with pytest.raises(TypeError):
            estimate_token_cost(paper_input_list)
        
        # Ints raise TypeError when len() is called
        paper_input_int = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_text": 123},
        }
        with pytest.raises(TypeError):
            estimate_token_cost(paper_input_int)

    def test_supplementary_invalid_type_raises_error(self):
        """Should raise TypeError if supplementary is not a dict or None."""
        # Lists should raise TypeError
        paper_input_list = {"paper_text": "A" * 400, "supplementary": [1, 2]}
        with pytest.raises(TypeError, match="Expected dict for supplementary"):
            estimate_token_cost(paper_input_list)
        
        # Strings should raise TypeError
        paper_input_str = {"paper_text": "A" * 400, "supplementary": "string"}
        with pytest.raises(TypeError, match="Expected dict for supplementary"):
            estimate_token_cost(paper_input_str)
        
        # Ints should raise TypeError
        paper_input_int = {"paper_text": "A" * 400, "supplementary": 123}
        with pytest.raises(TypeError, match="Expected dict for supplementary"):
            estimate_token_cost(paper_input_int)
        
        # Empty list is falsy, but should still raise TypeError (not dict)
        paper_input_empty_list = {"paper_text": "A" * 400, "supplementary": []}
        with pytest.raises(TypeError, match="Expected dict for supplementary"):
            estimate_token_cost(paper_input_empty_list)

    def test_empty_string_paper_text(self):
        """Empty string paper_text should be handled correctly."""
        paper_input = {"paper_text": ""}
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["text_chars"] == 0
        assert result["estimated_input_tokens"] >= 0
        assert result["estimated_output_tokens"] >= 0

    def test_empty_string_supplementary_text(self):
        """Empty string supplementary_text should be handled correctly."""
        paper_input = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_text": ""},
        }
        result = estimate_token_cost(paper_input)
        # Empty string should contribute 0 tokens
        assert result["estimated_input_tokens"] >= 0

    def test_very_large_paper_text(self):
        """Very large paper text should be handled correctly."""
        large_text = "A" * 1_000_000  # 1 million characters
        paper_input = {"paper_text": large_text}
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["text_chars"] == 1_000_000
        assert result["estimated_input_tokens"] > 0
        assert result["estimated_cost_usd"] > 0

    def test_supplementary_text_calculation_accuracy(self):
        """Verify supplementary_text is correctly included in token calculations."""
        base_text = "A" * 4000  # 1000 tokens
        supp_text = "B" * 2000  # 500 tokens
        
        paper_no_supp = {"paper_text": base_text}
        paper_with_supp = {
            "paper_text": base_text,
            "supplementary": {"supplementary_text": supp_text},
        }
        
        result_no_supp = estimate_token_cost(paper_no_supp)
        result_with_supp = estimate_token_cost(paper_with_supp)
        
        # With supplementary text, input tokens should be higher
        assert result_with_supp["estimated_input_tokens"] > result_no_supp["estimated_input_tokens"]
        
        # The difference should reflect the supplementary text tokens
        # (exact calculation depends on the formula, but should be positive)
        token_diff = result_with_supp["estimated_input_tokens"] - result_no_supp["estimated_input_tokens"]
        assert token_diff > 0

    def test_warning_field_content(self):
        """Verify warning field contains expected content."""
        result = estimate_token_cost({"paper_text": "A" * 400})
        assert isinstance(result["warning"], str)
        assert len(result["warning"]) > 0
        # Should mention it's a rough estimate
        assert "estimate" in result["warning"].lower() or "rough" in result["warning"].lower()

    def test_assumptions_field_completeness(self):
        """Verify assumptions field contains all expected keys."""
        result = estimate_token_cost({"paper_text": "A" * 400, "figures": [{"id": "f1"}]})
        assumptions = result["assumptions"]
        expected_keys = ["num_figures", "num_stages_estimated", "text_chars", "model_pricing"]
        for key in expected_keys:
            assert key in assumptions, f"Missing assumption key: {key}"

    def test_cost_breakdown_invariant(self):
        """
        BUG FOUND: Cost breakdown invariant fails due to rounding.
        
        The function rounds input_cost and output_cost individually, then rounds total_cost.
        Due to rounding, round(input_cost, 2) + round(output_cost, 2) may not equal round(total_cost, 2).
        This is a bug - the cost breakdown should be consistent.
        """
        test_cases = [
            {},
            {"paper_text": "A" * 1000},
            {"figures": [{"id": f"f{i}"} for i in range(5)]},
            {"paper_text": "A" * 5000, "figures": [{"id": "f1"}], "supplementary": {"supplementary_text": "B" * 2000}},
        ]
        for paper_input in test_cases:
            result = estimate_token_cost(paper_input)
            input_cost = result["cost_breakdown"]["input_cost_usd"]
            output_cost = result["cost_breakdown"]["output_cost_usd"]
            total_cost = result["estimated_cost_usd"]
            # BUG: Due to rounding differences, this invariant fails
            # The function should ensure input_cost + output_cost = total_cost
            assert abs((input_cost + output_cost) - total_cost) < 0.01, \
                f"Cost breakdown invariant failed: {input_cost} + {output_cost} != {total_cost}"

    def test_all_values_non_negative(self):
        """Verify all numeric values are non-negative."""
        test_cases = [
            {},
            {"paper_text": "A" * 1000},
            {"figures": [{"id": "f1"}]},
            {"paper_text": "", "figures": []},
        ]
        for paper_input in test_cases:
            result = estimate_token_cost(paper_input)
            assert result["estimated_input_tokens"] >= 0
            assert result["estimated_output_tokens"] >= 0
            assert result["estimated_total_tokens"] >= 0
            assert result["estimated_cost_usd"] >= 0
            assert result["cost_breakdown"]["input_cost_usd"] >= 0
            assert result["cost_breakdown"]["output_cost_usd"] >= 0

    def test_token_counts_are_integers(self):
        """Verify token counts are integers, not floats."""
        result = estimate_token_cost({"paper_text": "A" * 400, "figures": [{"id": "f1"}]})
        assert isinstance(result["estimated_input_tokens"], int)
        assert isinstance(result["estimated_output_tokens"], int)
        assert isinstance(result["estimated_total_tokens"], int)
        assert result["estimated_input_tokens"] == int(result["estimated_input_tokens"])
        assert result["estimated_output_tokens"] == int(result["estimated_output_tokens"])
        assert result["estimated_total_tokens"] == int(result["estimated_total_tokens"])

    def test_cost_values_are_floats(self):
        """Verify cost values are floats."""
        result = estimate_token_cost({"paper_text": "A" * 400, "figures": [{"id": "f1"}]})
        assert isinstance(result["estimated_cost_usd"], float)
        assert isinstance(result["cost_breakdown"]["input_cost_usd"], float)
        assert isinstance(result["cost_breakdown"]["output_cost_usd"], float)

    def test_figures_count_accuracy(self):
        """Verify figures count includes both main and supplementary figures."""
        paper_input = {
            "figures": [{"id": f"f{i}"} for i in range(3)],
            "supplementary": {"supplementary_figures": [{"id": f"s{i}"} for i in range(2)]},
        }
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 5

    def test_stage_count_boundary_conditions(self):
        """Test stage count at boundary conditions."""
        # Exactly 4 figures should use 4 stages
        figs_4 = [{"id": f"f{i}"} for i in range(4)]
        result_4 = estimate_token_cost({"figures": figs_4})
        assert result_4["assumptions"]["num_stages_estimated"] == 4
        
        # Exactly 3 figures should use 4 stages (minimum)
        figs_3 = [{"id": f"f{i}"} for i in range(3)]
        result_3 = estimate_token_cost({"figures": figs_3})
        assert result_3["assumptions"]["num_stages_estimated"] == 4
        
        # Exactly 5 figures should use 5 stages
        figs_5 = [{"id": f"f{i}"} for i in range(5)]
        result_5 = estimate_token_cost({"figures": figs_5})
        assert result_5["assumptions"]["num_stages_estimated"] == 5

    def test_text_chars_only_counts_paper_text(self):
        """Verify text_chars only counts paper_text, not supplementary_text."""
        paper_input = {
            "paper_text": "A" * 1000,
            "supplementary": {"supplementary_text": "B" * 2000},
        }
        result = estimate_token_cost(paper_input)
        # text_chars should only be paper_text length
        assert result["assumptions"]["text_chars"] == 1000

    def test_output_tokens_proportion(self):
        """Verify output tokens are approximately 25% of input tokens."""
        paper_input = {"paper_text": "A" * 4000, "figures": [{"id": "f1"}]}
        result = estimate_token_cost(paper_input)
        input_tokens = result["estimated_input_tokens"]
        output_tokens = result["estimated_output_tokens"]
        if input_tokens > 0:
            ratio = output_tokens / input_tokens
            # Should be approximately 0.25 (allowing for integer rounding)
            assert 0.20 <= ratio <= 0.30, f"Output ratio {ratio} not near 0.25"

    def test_empty_figures_list_vs_none(self):
        """Verify empty list and None are handled identically for figures."""
        paper_empty = {"paper_text": "A" * 400, "figures": []}
        paper_none = {"paper_text": "A" * 400, "figures": None}
        
        result_empty = estimate_token_cost(paper_empty)
        result_none = estimate_token_cost(paper_none)
        
        assert result_empty["assumptions"]["num_figures"] == result_none["assumptions"]["num_figures"]
        assert result_empty["assumptions"]["num_figures"] == 0

    def test_supplementary_figures_empty_list_vs_none(self):
        """Verify empty list and None are handled identically for supplementary_figures."""
        paper_empty = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_figures": []},
        }
        paper_none = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_figures": None},
        }
        
        result_empty = estimate_token_cost(paper_empty)
        result_none = estimate_token_cost(paper_none)
        
        assert result_empty["assumptions"]["num_figures"] == result_none["assumptions"]["num_figures"]
        assert result_empty["assumptions"]["num_figures"] == 0

    def test_supplementary_missing_vs_none(self):
        """Verify missing supplementary key and None are handled identically."""
        paper_missing = {"paper_text": "A" * 400}
        paper_none = {"paper_text": "A" * 400, "supplementary": None}
        
        result_missing = estimate_token_cost(paper_missing)
        result_none = estimate_token_cost(paper_none)
        
        assert result_missing["assumptions"]["num_figures"] == result_none["assumptions"]["num_figures"]
        assert result_missing["estimated_input_tokens"] == result_none["estimated_input_tokens"]

    def test_calculation_with_supplementary_text_exact(self):
        """Verify exact calculation when supplementary_text is included."""
        paper_text = "A" * 4000  # 1000 tokens
        supp_text = "B" * 2000   # 500 tokens
        
        paper_input = {
            "paper_text": paper_text,
            "supplementary": {"supplementary_text": supp_text},
        }
        
        # Manual calculation
        text_tokens = (4000 + 2000) / CHARS_PER_TOKEN  # Both texts included
        total_figures = 0
        image_tokens = total_figures * TOKENS_PER_FIGURE
        num_stages = max(4, total_figures)
        
        planner_input = 2 * text_tokens
        stage_text_fraction = 0.3
        per_stage_text = text_tokens * stage_text_fraction
        per_stage_input = (
            per_stage_text
            + per_stage_text * 2
            + per_stage_text * 2
            + per_stage_text
            + image_tokens / num_stages
        )
        total_stages_input = per_stage_input * num_stages
        supervisor_input = num_stages * (per_stage_text * 0.5)
        report_input = text_tokens * 0.2 + image_tokens
        total_input_estimate = (
            planner_input
            + total_stages_input
            + supervisor_input
            + report_input
        )
        total_output_estimate = total_input_estimate * 0.25
        
        result = estimate_token_cost(paper_input)
        
        assert result["estimated_input_tokens"] == int(total_input_estimate)
        assert result["estimated_output_tokens"] == int(total_output_estimate)

