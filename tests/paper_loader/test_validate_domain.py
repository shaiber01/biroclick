import pytest

from src.paper_loader import VALID_DOMAINS, validate_domain


class TestValidateDomain:
    """Tests for validate_domain function."""

    # ═══════════════════════════════════════════════════════════════════════
    # Valid Domain Tests
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_valid_domains_return_true(self, domain):
        """Test that all valid domains return exactly True."""
        result = validate_domain(domain)
        assert result is True, f"Expected True for valid domain '{domain}', got {result!r}"
        assert isinstance(result, bool), f"Result should be bool, got {type(result).__name__}"

    def test_all_valid_domains_are_tested(self):
        """Verify that we're testing all domains from VALID_DOMAINS."""
        # This ensures if VALID_DOMAINS changes, we catch it
        assert len(VALID_DOMAINS) > 0, "VALID_DOMAINS should not be empty"
        assert isinstance(VALID_DOMAINS, list), "VALID_DOMAINS should be a list"
        # Verify each domain is a string
        for domain in VALID_DOMAINS:
            assert isinstance(domain, str), f"Domain '{domain}' should be a string, got {type(domain).__name__}"
            assert len(domain) > 0, f"Domain '{domain}' should not be empty"

    # ═══════════════════════════════════════════════════════════════════════
    # Invalid Domain Tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_invalid_domain_returns_false(self):
        """Test that clearly invalid domains return False."""
        result = validate_domain("invalid_domain")
        assert result is False, f"Expected False for invalid domain, got {result!r}"
        assert isinstance(result, bool), f"Result should be bool, got {type(result).__name__}"

    def test_empty_domain_returns_false(self):
        """Test that empty string returns False."""
        result = validate_domain("")
        assert result is False, f"Expected False for empty string, got {result!r}"
        assert isinstance(result, bool), f"Result should be bool, got {type(result).__name__}"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: Whitespace and Formatting
    # ═══════════════════════════════════════════════════════════════════════

    def test_whitespace_only_returns_false(self):
        """Test that whitespace-only strings return False."""
        assert validate_domain(" ") is False
        assert validate_domain("  ") is False
        assert validate_domain("\t") is False
        assert validate_domain("\n") is False
        assert validate_domain("\r\n") is False
        assert validate_domain(" \t\n ") is False

    def test_valid_domain_with_leading_whitespace_returns_false(self):
        """Test that valid domains with leading whitespace are rejected."""
        for domain in VALID_DOMAINS:
            assert validate_domain(f" {domain}") is False, f"Leading space should be rejected for '{domain}'"
            assert validate_domain(f"\t{domain}") is False, f"Leading tab should be rejected for '{domain}'"

    def test_valid_domain_with_trailing_whitespace_returns_false(self):
        """Test that valid domains with trailing whitespace are rejected."""
        for domain in VALID_DOMAINS:
            assert validate_domain(f"{domain} ") is False, f"Trailing space should be rejected for '{domain}'"
            assert validate_domain(f"{domain}\t") is False, f"Trailing tab should be rejected for '{domain}'"

    def test_valid_domain_with_surrounding_whitespace_returns_false(self):
        """Test that valid domains with surrounding whitespace are rejected."""
        for domain in VALID_DOMAINS:
            assert validate_domain(f" {domain} ") is False, f"Surrounding spaces should be rejected for '{domain}'"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: Case Sensitivity
    # ═══════════════════════════════════════════════════════════════════════

    def test_case_sensitivity(self):
        """Test that domain validation is case-sensitive."""
        # Test uppercase versions
        for domain in VALID_DOMAINS:
            upper_domain = domain.upper()
            assert validate_domain(upper_domain) is False, f"Uppercase '{upper_domain}' should be rejected"
        
        # Test capitalized versions
        for domain in VALID_DOMAINS:
            if domain:  # Skip empty strings
                capitalized = domain.capitalize()
                if capitalized != domain:  # Only test if capitalization changes it
                    assert validate_domain(capitalized) is False, f"Capitalized '{capitalized}' should be rejected"

    def test_mixed_case_invalid(self):
        """Test that mixed case versions of valid domains are rejected."""
        test_cases = [
            "Plasmonics",
            "PLASMONICS",
            "PlAsMoNiCs",
            "Photonic_Crystal",
            "PHOTONIC_CRYSTAL",
            "MetaMaterial",
        ]
        for domain in test_cases:
            if domain.lower() in [d.lower() for d in VALID_DOMAINS]:
                assert validate_domain(domain) is False, f"Mixed case '{domain}' should be rejected"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: Similar but Invalid Domains
    # ═══════════════════════════════════════════════════════════════════════

    def test_similar_but_invalid_domains(self):
        """Test domains that are similar to valid ones but should be rejected."""
        invalid_similar = [
            "plasmonic",  # Missing 's'
            "photonic_crystals",  # Plural
            "metamaterials",  # Plural
            "thin_films",  # Plural
            "waveguides",  # Plural
            "strong_couplings",  # Plural
            "nonlinears",  # Plural
            "others",  # Plural
            "plasmonics_extra",  # Extra suffix
            "extra_plasmonics",  # Extra prefix
            "photonic-crystal",  # Wrong separator
            "photonic crystal",  # Space instead of underscore
        ]
        for domain in invalid_similar:
            assert validate_domain(domain) is False, f"Similar domain '{domain}' should be rejected"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: Type Safety
    # ═══════════════════════════════════════════════════════════════════════

    def test_none_returns_false(self):
        """Test that None returns False (type violation but current behavior)."""
        # NOTE: Function signature says it takes str, but currently accepts None
        # This test documents current behavior - if we want strict typing, 
        # the function should raise TypeError instead
        result = validate_domain(None)
        assert result is False, f"None should return False, got {result!r}"
        assert isinstance(result, bool), f"Result should be bool, got {type(result).__name__}"

    def test_non_string_types_return_false(self):
        """Test that non-string types return False (type violation but current behavior)."""
        # NOTE: Function signature says it takes str, but currently accepts other types
        # This test documents current behavior - if we want strict typing,
        # the function should raise TypeError instead
        test_cases = [
            (123, "integer"),
            (123.45, "float"),
            ([], "list"),
            ({}, "dict"),
            (True, "bool"),
            (False, "bool"),
        ]
        for value, type_name in test_cases:
            result = validate_domain(value)
            assert result is False, f"{type_name} {value!r} should return False, got {result!r}"
            assert isinstance(result, bool), f"Result should be bool for {type_name}, got {type(result).__name__}"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: Special Characters and Unicode
    # ═══════════════════════════════════════════════════════════════════════

    def test_special_characters_invalid(self):
        """Test that domains with special characters are rejected."""
        special_char_domains = [
            "plasmonics!",
            "plasmonics@",
            "plasmonics#",
            "plasmonics$",
            "plasmonics%",
            "plasmonics&",
            "plasmonics*",
            "plasmonics+",
            "plasmonics=",
            "plasmonics?",
            "plasmonics/",
            "plasmonics\\",
            "plasmonics|",
            "plasmonics<",
            "plasmonics>",
            "plasmonics[",
            "plasmonics]",
            "plasmonics{",
            "plasmonics}",
            "plasmonics(",
            "plasmonics)",
        ]
        for domain in special_char_domains:
            assert validate_domain(domain) is False, f"Domain with special char '{domain}' should be rejected"

    def test_unicode_characters_invalid(self):
        """Test that domains with unicode characters are rejected."""
        unicode_domains = [
            "plåsmonics",  # Non-ASCII character
            "plasmonics中文",  # Chinese characters
            "plasmonics🚀",  # Emoji
            "plasmonicsα",  # Greek letter
        ]
        for domain in unicode_domains:
            assert validate_domain(domain) is False, f"Domain with unicode '{domain}' should be rejected"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases: Boundary Conditions
    # ═══════════════════════════════════════════════════════════════════════

    def test_very_long_string_returns_false(self):
        """Test that very long strings are rejected."""
        long_string = "a" * 1000
        assert validate_domain(long_string) is False, "Very long string should be rejected"
        
        # Test with valid domain prefix but extra characters
        long_invalid = "plasmonics" + "x" * 1000
        assert validate_domain(long_invalid) is False, "Long string with valid prefix should be rejected"

    # ═══════════════════════════════════════════════════════════════════════
    # Integration Tests: Verify VALID_DOMAINS Usage
    # ═══════════════════════════════════════════════════════════════════════

    def test_function_uses_valid_domains_correctly(self):
        """Test that the function correctly uses VALID_DOMAINS."""
        # Verify that every domain in VALID_DOMAINS returns True
        for domain in VALID_DOMAINS:
            assert validate_domain(domain) is True, f"Domain '{domain}' from VALID_DOMAINS should return True"
        
        # Verify that domains not in VALID_DOMAINS return False
        # Create a set of all valid domains for efficient lookup
        valid_set = set(VALID_DOMAINS)
        
        # Test some domains that definitely shouldn't be valid
        test_invalid = [
            "not_a_domain",
            "fake_domain",
            "test",
            "invalid",
            "xyz",
        ]
        for domain in test_invalid:
            if domain not in valid_set:
                assert validate_domain(domain) is False, f"Domain '{domain}' not in VALID_DOMAINS should return False"

    def test_no_domain_duplicates_in_valid_domains(self):
        """Test that VALID_DOMAINS doesn't contain duplicates."""
        # This is a sanity check on the configuration
        assert len(VALID_DOMAINS) == len(set(VALID_DOMAINS)), \
            f"VALID_DOMAINS contains duplicates: {VALID_DOMAINS}"

    def test_valid_domains_are_all_strings(self):
        """Test that VALID_DOMAINS contains only string values."""
        # This is a sanity check on the configuration
        # If VALID_DOMAINS contains non-strings, the function behavior might be unexpected
        for domain in VALID_DOMAINS:
            assert isinstance(domain, str), \
                f"VALID_DOMAINS should contain only strings, but found {type(domain).__name__}: {domain!r}"

    def test_valid_domains_are_non_empty(self):
        """Test that VALID_DOMAINS doesn't contain empty strings."""
        # Empty string would be confusing - is it valid or not?
        for domain in VALID_DOMAINS:
            assert len(domain) > 0, \
                f"VALID_DOMAINS should not contain empty strings, but found one"

    def test_valid_domains_have_no_whitespace(self):
        """Test that VALID_DOMAINS doesn't contain domains with leading/trailing whitespace."""
        # Domains with whitespace would be confusing
        for domain in VALID_DOMAINS:
            assert domain == domain.strip(), \
                f"VALID_DOMAINS should not contain domains with whitespace, but found: {domain!r}"

    # ═══════════════════════════════════════════════════════════════════════
    # Return Value Verification
    # ═══════════════════════════════════════════════════════════════════════

    def test_return_value_is_exactly_boolean(self):
        """Test that function returns exactly True or False, not truthy/falsy values."""
        # Test valid domain returns exactly True
        for domain in VALID_DOMAINS:
            result = validate_domain(domain)
            assert result is True, f"Should return exactly True, got {result!r}"
            assert result == True, f"Should equal True, got {result!r}"
            assert not (result is False), f"Should not be False, got {result!r}"
        
        # Test invalid domain returns exactly False
        invalid_domains = ["invalid", "", " "]
        for domain in invalid_domains:
            result = validate_domain(domain)
            assert result is False, f"Should return exactly False, got {result!r}"
            assert result == False, f"Should equal False, got {result!r}"
            assert not (result is True), f"Should not be True, got {result!r}"

    def test_function_is_deterministic(self):
        """Test that function returns consistent results for same input."""
        test_domains = VALID_DOMAINS + ["invalid", "", None]
        for domain in test_domains:
            result1 = validate_domain(domain)
            result2 = validate_domain(domain)
            assert result1 == result2, f"Function should be deterministic for '{domain}'"
            assert result1 is result2, f"Function should return same object for '{domain}'"

    def test_function_matches_python_in_operator(self):
        """Test that function behavior matches Python's 'in' operator."""
        # This verifies the function implementation is correct
        test_cases = VALID_DOMAINS + ["invalid", "", "plasmonics ", " Plasmonics"]
        for domain in test_cases:
            expected = domain in VALID_DOMAINS
            actual = validate_domain(domain)
            assert actual == expected, \
                f"Function should match 'in' operator: domain={domain!r}, expected={expected}, got={actual}"

    def test_substring_not_valid(self):
        """Test that substrings of valid domains are not considered valid."""
        # This ensures the function doesn't do substring matching
        for valid_domain in VALID_DOMAINS:
            if len(valid_domain) > 1:
                # Test first half
                substring = valid_domain[:len(valid_domain)//2]
                if substring not in VALID_DOMAINS:
                    assert validate_domain(substring) is False, \
                        f"Substring '{substring}' of '{valid_domain}' should not be valid"
                
                # Test last half
                substring = valid_domain[len(valid_domain)//2:]
                if substring not in VALID_DOMAINS:
                    assert validate_domain(substring) is False, \
                        f"Substring '{substring}' of '{valid_domain}' should not be valid"

    def test_concatenated_domains_not_valid(self):
        """Test that concatenating valid domains doesn't create a valid domain."""
        # This ensures the function doesn't do pattern matching
        if len(VALID_DOMAINS) >= 2:
            concatenated = VALID_DOMAINS[0] + VALID_DOMAINS[1]
            if concatenated not in VALID_DOMAINS:
                assert validate_domain(concatenated) is False, \
                    f"Concatenated domain '{concatenated}' should not be valid"

