"""
Unit tests for paper_loader cost_estimation module.
"""

import pytest

from src.paper_loader import (
    estimate_tokens,
    estimate_token_cost,
    check_paper_length,
    CHARS_PER_TOKEN,
    PAPER_LENGTH_NORMAL,
    PAPER_LENGTH_LONG,
    PAPER_LENGTH_VERY_LONG,
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
        
        assert tokens == 100 // CHARS_PER_TOKEN
    
    def test_empty_string_returns_zero(self):
        """Empty string returns 0 tokens."""
        assert estimate_tokens("") == 0
    
    def test_integer_division(self):
        """Uses integer division (floor)."""
        text = "A" * 5  # 5 chars with CHARS_PER_TOKEN=4 should give 1
        tokens = estimate_tokens(text)
        
        assert tokens == 1
    
    def test_large_text(self):
        """Handles large text."""
        text = "A" * 100_000
        tokens = estimate_tokens(text)
        
        assert tokens == 100_000 // CHARS_PER_TOKEN


# ═══════════════════════════════════════════════════════════════════════
# check_paper_length Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCheckPaperLength:
    """Tests for check_paper_length function."""
    
    def test_normal_length_no_warnings(self):
        """Normal length paper returns no warnings."""
        text = "A" * (PAPER_LENGTH_NORMAL - 1000)
        warnings = check_paper_length(text)
        
        assert warnings == []
    
    def test_long_paper_warns(self):
        """Long paper (>PAPER_LENGTH_LONG) returns warning."""
        text = "A" * (PAPER_LENGTH_LONG + 1000)
        warnings = check_paper_length(text)
        
        assert len(warnings) == 1
        assert "long" in warnings[0].lower()
    
    def test_very_long_paper_warns(self):
        """Very long paper (>PAPER_LENGTH_VERY_LONG) returns VERY LONG warning."""
        text = "A" * (PAPER_LENGTH_VERY_LONG + 1000)
        warnings = check_paper_length(text)
        
        assert len(warnings) == 1
        assert "VERY LONG" in warnings[0]
    
    def test_custom_label(self):
        """Custom label appears in warning."""
        text = "A" * (PAPER_LENGTH_LONG + 1000)
        warnings = check_paper_length(text, label="Supplementary")
        
        assert "Supplementary" in warnings[0]
    
    def test_default_label_is_paper(self):
        """Default label is 'Paper'."""
        text = "A" * (PAPER_LENGTH_LONG + 1000)
        warnings = check_paper_length(text)
        
        assert "Paper" in warnings[0] or warnings[0].startswith("Paper")


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
            "paper_text": "A" * 10_000,  # 10K chars ~ 2500 tokens
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
            assert key in result
    
    def test_cost_breakdown_has_input_output(self, basic_paper_input):
        """cost_breakdown includes input and output costs."""
        result = estimate_token_cost(basic_paper_input)
        
        assert "input_cost_usd" in result["cost_breakdown"]
        assert "output_cost_usd" in result["cost_breakdown"]
    
    def test_assumptions_includes_num_figures(self, basic_paper_input):
        """assumptions includes num_figures."""
        result = estimate_token_cost(basic_paper_input)
        
        assert result["assumptions"]["num_figures"] == 2
    
    def test_token_counts_are_integers(self, basic_paper_input):
        """Token counts are integers."""
        result = estimate_token_cost(basic_paper_input)
        
        assert isinstance(result["estimated_input_tokens"], int)
        assert isinstance(result["estimated_output_tokens"], int)
        assert isinstance(result["estimated_total_tokens"], int)
    
    def test_total_is_sum_of_input_output(self, basic_paper_input):
        """Total tokens is input + output."""
        result = estimate_token_cost(basic_paper_input)
        
        assert result["estimated_total_tokens"] == (
            result["estimated_input_tokens"] + result["estimated_output_tokens"]
        )
    
    def test_more_figures_increases_cost(self):
        """More figures increases estimated cost."""
        paper_2_figs = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 10_000,
            "figures": [{"id": f"Fig{i}", "description": "T", "image_path": f"f{i}.png"} for i in range(2)]
        }
        paper_10_figs = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 10_000,
            "figures": [{"id": f"Fig{i}", "description": "T", "image_path": f"f{i}.png"} for i in range(10)]
        }
        
        cost_2 = estimate_token_cost(paper_2_figs)
        cost_10 = estimate_token_cost(paper_10_figs)
        
        assert cost_10["estimated_cost_usd"] > cost_2["estimated_cost_usd"]
    
    def test_more_text_increases_cost(self):
        """More text increases estimated cost."""
        paper_short = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 5_000,
            "figures": []
        }
        paper_long = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 50_000,
            "figures": []
        }
        
        cost_short = estimate_token_cost(paper_short)
        cost_long = estimate_token_cost(paper_long)
        
        assert cost_long["estimated_cost_usd"] > cost_short["estimated_cost_usd"]
    
    def test_includes_supplementary_text(self):
        """Supplementary text is included in estimate."""
        paper_no_supp = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 10_000,
            "figures": []
        }
        paper_with_supp = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 10_000,
            "supplementary": {
                "supplementary_text": "B" * 10_000
            },
            "figures": []
        }
        
        cost_no_supp = estimate_token_cost(paper_no_supp)
        cost_with_supp = estimate_token_cost(paper_with_supp)
        
        assert cost_with_supp["estimated_input_tokens"] > cost_no_supp["estimated_input_tokens"]
    
    def test_includes_supplementary_figures(self):
        """Supplementary figures are counted."""
        paper_with_supp_figs = {
            "paper_id": "test",
            "paper_title": "Test",
            "paper_text": "A" * 10_000,
            "figures": [],
            "supplementary": {
                "supplementary_figures": [
                    {"id": "S1", "description": "T", "image_path": "s1.png"},
                    {"id": "S2", "description": "T", "image_path": "s2.png"},
                ]
            }
        }
        
        result = estimate_token_cost(paper_with_supp_figs)
        
        assert result["assumptions"]["num_figures"] == 2
    
    def test_warning_message_present(self, basic_paper_input):
        """Warning message is present and informative."""
        result = estimate_token_cost(basic_paper_input)
        
        assert len(result["warning"]) > 50  # Substantial warning
        assert "estimate" in result["warning"].lower()
    
    def test_cost_is_rounded(self, basic_paper_input):
        """Cost values are rounded to 2 decimal places."""
        result = estimate_token_cost(basic_paper_input)
        
        # Check main cost
        cost_str = str(result["estimated_cost_usd"])
        if "." in cost_str:
            decimals = len(cost_str.split(".")[1])
            assert decimals <= 2
        
        # Check breakdown costs
        for key in ["input_cost_usd", "output_cost_usd"]:
            cost_str = str(result["cost_breakdown"][key])
            if "." in cost_str:
                decimals = len(cost_str.split(".")[1])
                assert decimals <= 2

