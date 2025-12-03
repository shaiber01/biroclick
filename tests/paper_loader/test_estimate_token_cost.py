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
        assert result["assumptions"]["text_chars"] == 0
        assert result["assumptions"]["num_figures"] == 0

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
        assert isinstance(result["estimated_cost_usd"], float)
        assert isinstance(result["cost_breakdown"]["input_cost_usd"], float)

    def test_supplementary_is_none(self):
        """Handles case where supplementary key is None (JSON null)."""
        paper_input = {"paper_text": "A" * 400, "supplementary": None}
        result = estimate_token_cost(paper_input)
        assert result["estimated_input_tokens"] > 0

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

