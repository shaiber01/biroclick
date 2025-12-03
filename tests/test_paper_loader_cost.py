"""
Unit tests for paper_loader cost_estimation module.
"""

import pytest
from typing import Dict, Any

from src.paper_loader import (
    estimate_tokens,
    estimate_token_cost,
    check_paper_length,
    CHARS_PER_TOKEN,
    PAPER_LENGTH_NORMAL,
    PAPER_LENGTH_LONG,
    PAPER_LENGTH_VERY_LONG,
)
from src.paper_loader.config import (
    TOKENS_PER_FIGURE,
    INPUT_COST_PER_MILLION,
    OUTPUT_COST_PER_MILLION,
)


# ═══════════════════════════════════════════════════════════════════════
# estimate_tokens Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateTokens:
    """Tests for estimate_tokens function."""
    
    def test_basic_estimate(self):
        """Estimates tokens as chars / CHARS_PER_TOKEN."""
        text = "A" * 100
        tokens = estimate_tokens(text)
        # 100 / 4 = 25
        assert tokens == 25
    
    def test_empty_string_returns_zero(self):
        """Empty string returns 0 tokens."""
        assert estimate_tokens("") == 0
    
    def test_integer_division_floor(self):
        """Uses integer division (floor)."""
        # 5 chars with CHARS_PER_TOKEN=4. 5/4 = 1.25 -> floor to 1
        text = "A" * 5
        tokens = estimate_tokens(text)
        assert tokens == 1
        
        text = "A" * 7
        tokens = estimate_tokens(text)
        assert tokens == 1
        
        text = "A" * 8
        tokens = estimate_tokens(text)
        assert tokens == 2
    
    def test_large_text(self):
        """Handles large text."""
        text = "A" * 100_000
        tokens = estimate_tokens(text)
        assert tokens == 100_000 // CHARS_PER_TOKEN

    def test_unicode_characters(self):
        """Handles unicode characters (counts Python chars, not bytes)."""
        # '👍' is 1 char in Python 3 string
        text = "👍" * 10
        assert len(text) == 10
        tokens = estimate_tokens(text)
        assert tokens == 10 // CHARS_PER_TOKEN

    def test_none_input_raises_error(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            estimate_tokens(None)  # type: ignore

    def test_int_input_raises_error(self):
        """Integer input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            estimate_tokens(123)  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# check_paper_length Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCheckPaperLength:
    """Tests for check_paper_length function."""
    
    def test_normal_length_no_warnings(self):
        """Normal length paper returns no warnings."""
        text = "A" * PAPER_LENGTH_NORMAL
        warnings = check_paper_length(text)
        assert warnings == []
    
    def test_long_paper_boundary(self):
        """Test boundary conditions for long paper warning."""
        # Exact boundary: PAPER_LENGTH_LONG
        text = "A" * PAPER_LENGTH_LONG
        warnings = check_paper_length(text)
        assert warnings == []  # Should be no warning at exact limit based on implementation (> check)

        # One char over
        text = "A" * (PAPER_LENGTH_LONG + 1)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
        assert "VERY LONG" not in warnings[0]

    def test_very_long_paper_boundary(self):
        """Test boundary conditions for very long paper warning."""
        # Exact boundary
        text = "A" * PAPER_LENGTH_VERY_LONG
        warnings = check_paper_length(text)
        # Should be just "long" warning if logic is check > LONG then check > VERY_LONG?
        
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
        assert "VERY LONG" not in warnings[0] # It's just "long" at the boundary

        # One char over
        text = "A" * (PAPER_LENGTH_VERY_LONG + 1)
        warnings = check_paper_length(text)
        assert len(warnings) == 1
        assert "VERY LONG" in warnings[0]
    
    def test_custom_label(self):
        """Custom label appears in warning."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text, label="Supplementary")
        assert "Supplementary" in warnings[0]
    
    def test_default_label_is_paper(self):
        """Default label is 'Paper'."""
        text = "A" * (PAPER_LENGTH_LONG + 100)
        warnings = check_paper_length(text)
        assert warnings[0].startswith("Paper")

    def test_empty_string(self):
        """Empty string returns no warnings."""
        warnings = check_paper_length("")
        assert warnings == []

    def test_none_input_raises_error(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string"):
            check_paper_length(None) # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# estimate_token_cost Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEstimateTokenCost:
    """Tests for estimate_token_cost function."""
    
    @pytest.fixture
    def basic_paper_input(self):
        """Create basic paper input for cost estimation."""
        return {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 10_000,  # 10K chars
            "figures": [
                {"id": "Fig1", "description": "Test", "image_path": "fig1.png"},
                {"id": "Fig2", "description": "Test", "image_path": "fig2.png"},
            ]
        }
    
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
        # Setup inputs
        char_count = 4000  # 1000 tokens
        text = "A" * char_count
        figures = [{"id": "f1"}, {"id": "f2"}] # 2 figures
        supp_figures = [{"id": "s1"}] # 1 supplementary figure
        
        paper_input = {
            "paper_text": text,
            "figures": figures,
            "supplementary": {
                "supplementary_figures": supp_figures
            }
        }
        
        # Manual Calculation based on current logic
        # 1. Text Tokens
        text_tokens = char_count / CHARS_PER_TOKEN # 1000.0
        
        # 2. Image Tokens
        total_figures_count = len(figures) + len(supp_figures) # 3
        image_tokens = total_figures_count * TOKENS_PER_FIGURE # 3 * 680 = 2040
        
        # 3. Planner Input
        planner_input = 2 * text_tokens # 2000.0
        
        # 4. Stages
        # Use total figures for stages
        num_stages = max(4, total_figures_count) # max(4, 3) = 4
        
        stage_text_fraction = 0.3
        per_stage_text = text_tokens * stage_text_fraction # 300.0
        
        per_stage_input = (
            per_stage_text +                # Design
            per_stage_text * 2 +            # CodeGen
            per_stage_text * 2 +            # Review
            per_stage_text + image_tokens / num_stages # Analysis
        )
        # per_stage_input = 300 + 600 + 600 + 300 + (2040/4=510) = 1800 + 510 = 2310.0
        
        total_stages_input = per_stage_input * num_stages # 2310 * 4 = 9240.0
        
        # 5. Supervisor
        supervisor_input = num_stages * (per_stage_text * 0.5) # 4 * 150 = 600.0
        
        # 6. Report
        report_input = text_tokens * 0.2 + image_tokens # 200 + 2040 = 2240.0
        
        # 7. Totals
        total_input_estimate = (
            planner_input +     # 2000
            total_stages_input + # 9240
            supervisor_input +  # 600
            report_input        # 2240
        ) # = 14080.0
        
        total_output_estimate = total_input_estimate * 0.25 # 3520.0
        
        input_cost = total_input_estimate * INPUT_COST_PER_MILLION / 1_000_000
        output_cost = total_output_estimate * OUTPUT_COST_PER_MILLION / 1_000_000
        total_cost = input_cost + output_cost
        
        # Run function
        result = estimate_token_cost(paper_input)
        
        # Assertions
        assert result["estimated_input_tokens"] == int(total_input_estimate)
        assert result["estimated_output_tokens"] == int(total_output_estimate)
        assert result["estimated_total_tokens"] == int(total_input_estimate + total_output_estimate)
        
        # Check costs with small tolerance for float math, though rounding should match
        assert abs(result["estimated_cost_usd"] - round(total_cost, 2)) < 0.001
        assert abs(result["cost_breakdown"]["input_cost_usd"] - round(input_cost, 2)) < 0.001
        assert abs(result["cost_breakdown"]["output_cost_usd"] - round(output_cost, 2)) < 0.001
        
        # Verify assumptions
        assert result["assumptions"]["num_figures"] == 3
        assert result["assumptions"]["num_stages_estimated"] == 4
        assert result["assumptions"]["text_chars"] == char_count

    def test_missing_keys_handled(self):
        """Test that missing keys in input don't crash function."""
        # Empty dict
        result = estimate_token_cost({})
        assert result["estimated_input_tokens"] >= 0
        assert result["assumptions"]["text_chars"] == 0
        assert result["assumptions"]["num_figures"] == 0

    def test_zero_figures_uses_min_stages(self):
        """Zero figures should still use minimum 4 stages."""
        paper_input = {
            "paper_text": "A" * 400, # 100 tokens
            "figures": []
        }
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 0
        assert result["assumptions"]["num_stages_estimated"] == 4

    def test_many_figures_uses_figure_count_stages(self):
        """Many figures should increase stage count."""
        figs = [{"id": f"f{i}"} for i in range(10)]
        paper_input = {
            "paper_text": "A" * 400,
            "figures": figs
        }
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 10
        assert result["assumptions"]["num_stages_estimated"] == 10

    def test_supplementary_text_included(self):
        """Supplementary text adds to token count."""
        paper_no_supp = {"paper_text": "A" * 400}
        paper_with_supp = {
            "paper_text": "A" * 400,
            "supplementary": {"supplementary_text": "B" * 400}
        }
        
        cost_no = estimate_token_cost(paper_no_supp)
        cost_supp = estimate_token_cost(paper_with_supp)
        
        # Should be roughly double the text tokens involved
        # (exact math depends on multipliers, but definitely strictly greater)
        assert cost_supp["estimated_input_tokens"] > cost_no["estimated_input_tokens"]

    def test_supplementary_figures_included(self):
        """Supplementary figures add to token count and stages if enough."""
        # 3 main figures, 3 supp figures -> 6 total -> 6 stages
        paper_input = {
            "figures": [{"id": "f"} for _ in range(3)],
            "supplementary": {
                "supplementary_figures": [{"id": "s"} for _ in range(3)]
            }
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
        paper_input = {
            "paper_text": "A" * 400,
            "supplementary": None
        }
        # Should not raise AttributeError
        result = estimate_token_cost(paper_input)
        # Should treat as empty supplementary
        assert result["estimated_input_tokens"] > 0
        
    def test_figures_is_none(self):
        """Handles case where figures key is None (JSON null)."""
        paper_input = {
            "paper_text": "A" * 400,
            "figures": None
        }
        # Should not raise TypeError
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 0
        
    def test_supplementary_figures_is_none(self):
        """Handles case where supplementary_figures is None."""
        paper_input = {
            "paper_text": "A" * 400,
            "supplementary": {
                "supplementary_figures": None
            }
        }
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["num_figures"] == 0

    def test_paper_text_is_none(self):
        """Handles case where paper_text is None."""
        paper_input = {"paper_text": None}
        # Should not raise TypeError
        result = estimate_token_cost(paper_input)
        assert result["assumptions"]["text_chars"] == 0

    # --- New Robustness Tests ---

    def test_invalid_figures_type_raises_error(self):
        """Should raise TypeError if figures is not a list (e.g. a string)."""
        # If figures is a string "foo", len("foo") is 3.
        # The code would calculate costs for 3 figures if it doesn't validate types.
        paper_input = {"figures": "invalid_string"}
        
        # We expect robust code to reject this, not silently process characters as figures.
        with pytest.raises(TypeError, match="Expected list for figures"):
            estimate_token_cost(paper_input)

    def test_invalid_supplementary_figures_type_raises_error(self):
        """Should raise TypeError if supplementary_figures is not a list."""
        paper_input = {
            "supplementary": {
                "supplementary_figures": "invalid_string"
            }
        }
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
            {"paper_text": "", "figures": [{"id": "1"}, {"id": "2"}]}
        ]
        for i, inp in enumerate(inputs):
            res = estimate_token_cost(inp)
            total = res["estimated_total_tokens"]
            inp_tok = res["estimated_input_tokens"]
            out_tok = res["estimated_output_tokens"]
            
            # Using >= because sometimes total might include overhead, but here it is simple sum
            # The implementation computes total = input + output explicitly
            assert total == inp_tok + out_tok, f"Invariant failed for input {i}"

    @pytest.mark.parametrize("num_figs, expected_stages", [
        (0, 4),
        (3, 4),
        (4, 4),
        (5, 5),
        (10, 10)
    ])
    def test_stage_count_logic(self, num_figs, expected_stages):
        """Verifies max(4, num_figs) logic explicitly."""
        figs = [{"id": str(i)} for i in range(num_figs)]
        res = estimate_token_cost({"figures": figs})
        assert res["assumptions"]["num_stages_estimated"] == expected_stages

