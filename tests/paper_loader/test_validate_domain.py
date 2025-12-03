import pytest

from src.paper_loader import VALID_DOMAINS, validate_domain


class TestValidateDomain:
    """Tests for validate_domain function."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_valid_domains_pass(self, domain):
        assert validate_domain(domain) is True

    def test_invalid_domain_returns_false(self):
        assert validate_domain("invalid_domain") is False

    def test_empty_domain_returns_false(self):
        assert validate_domain("") is False

    def test_none_domain_returns_false(self):
        assert validate_domain(None) is False

