"""Tests for platform detection utilities in `src.code_runner`."""

import sys
import os
import platform
import warnings
from unittest.mock import patch, MagicMock

import pytest

from src.code_runner import detect_platform, get_platform_capabilities, check_platform_and_warn


class TestPlatformDetection:
    """Tests for platform detection functions."""

    def test_detect_platform_returns_valid_capabilities(self):
        """Test that detect_platform returns a valid capabilities dict with all required fields."""
        caps = detect_platform()

        # Verify it's a dict
        assert isinstance(caps, dict), "detect_platform() must return a dict"
        
        # Verify all required fields exist
        required_fields = [
            "platform",
            "memory_limiting_available",
            "process_group_kill_available",
            "preexec_fn_available",
            "is_wsl",
            "warnings",
            "recommended_action"
        ]
        for field in required_fields:
            assert field in caps, f"Missing required field: {field}"
        
        # Verify field types
        assert isinstance(caps["platform"], str), "platform must be a string"
        assert isinstance(caps["memory_limiting_available"], bool), "memory_limiting_available must be a bool"
        assert isinstance(caps["process_group_kill_available"], bool), "process_group_kill_available must be a bool"
        assert isinstance(caps["preexec_fn_available"], bool), "preexec_fn_available must be a bool"
        assert isinstance(caps["is_wsl"], bool), "is_wsl must be a bool"
        assert isinstance(caps["warnings"], list), "warnings must be a list"
        assert caps["recommended_action"] is None or isinstance(caps["recommended_action"], str), \
            "recommended_action must be None or a string"
        
        # Verify platform value is one of the expected values
        valid_platforms = ["windows", "macos", "linux", "wsl"]
        assert caps["platform"] in valid_platforms, \
            f"platform must be one of {valid_platforms}, got '{caps['platform']}'"
        
        # Verify all warnings are strings
        for warning in caps["warnings"]:
            assert isinstance(warning, str), f"All warnings must be strings, got {type(warning)}"

    def test_detect_platform_identifies_current_os(self):
        """Test that platform detection matches sys.platform."""
        caps = detect_platform()

        if sys.platform == "darwin":
            assert caps["platform"] == "macos", \
                f"Expected 'macos' for darwin, got '{caps['platform']}'"
            assert caps["is_wsl"] is False, "macOS should not be detected as WSL"
        elif sys.platform == "win32":
            assert caps["platform"] == "windows", \
                f"Expected 'windows' for win32, got '{caps['platform']}'"
            assert caps["is_wsl"] is False, "Native Windows should not be detected as WSL"
        elif sys.platform.startswith("linux"):
            # WSL check might override "linux" to "wsl"
            if caps["is_wsl"]:
                assert caps["platform"] == "wsl", \
                    f"Expected 'wsl' when is_wsl=True, got '{caps['platform']}'"
            else:
                assert caps["platform"] == "linux", \
                    f"Expected 'linux' for non-WSL Linux, got '{caps['platform']}'"
        else:
            # Unknown platform - should still return valid structure
            assert caps["platform"] in ["linux", "wsl"], \
                f"Unknown platform {sys.platform} should default to linux or wsl"

    def test_windows_platform_capabilities(self):
        """Test that Windows platform has correct capabilities."""
        with patch('sys.platform', 'win32'):
            with patch('os.uname', side_effect=AttributeError("Windows doesn't have uname")):
                caps = detect_platform()
                
                assert caps["platform"] == "windows", "Should detect Windows"
                assert caps["memory_limiting_available"] is False, \
                    "Windows should not have memory limiting"
                assert caps["process_group_kill_available"] is False, \
                    "Windows should not have process group kill"
                assert caps["preexec_fn_available"] is False, \
                    "Windows should not have preexec_fn"
                assert caps["is_wsl"] is False, "Native Windows is not WSL"
                assert len(caps["warnings"]) >= 2, \
                    "Windows should have at least 2 warnings about limitations"
                assert any("Memory limiting is NOT available" in w for w in caps["warnings"]), \
                    "Windows should warn about memory limiting"
                assert any("Process group signaling is limited" in w for w in caps["warnings"]), \
                    "Windows should warn about process group signaling"
                assert caps["recommended_action"] is not None, \
                    "Windows should have a recommended action"
                assert "WSL2" in caps["recommended_action"] or "Docker" in caps["recommended_action"], \
                    "Windows recommended action should mention WSL2 or Docker"

    def test_wsl_detection_microsoft_in_release(self):
        """Test WSL detection when 'microsoft' appears in uname release."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = "5.10.16.3-microsoft-standard-WSL2"
            with patch('os.uname', return_value=mock_uname):
                caps = detect_platform()
                
                assert caps["is_wsl"] is True, "Should detect WSL when 'microsoft' in release"
                assert caps["platform"] == "wsl", "Platform should be 'wsl'"
                assert caps["memory_limiting_available"] is True, "WSL should have memory limiting"
                assert caps["process_group_kill_available"] is True, "WSL should have process group kill"
                assert caps["preexec_fn_available"] is True, "WSL should have preexec_fn"
                assert len(caps["warnings"]) == 0, "WSL should have no warnings"
                assert caps["recommended_action"] is None, "WSL should have no recommended action"

    def test_wsl_detection_wsl_in_release(self):
        """Test WSL detection when 'wsl' appears in uname release."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = "5.4.0-wsl2"
            with patch('os.uname', return_value=mock_uname):
                caps = detect_platform()
                
                assert caps["is_wsl"] is True, "Should detect WSL when 'wsl' in release"
                assert caps["platform"] == "wsl", "Platform should be 'wsl'"

    def test_wsl_detection_case_insensitive(self):
        """Test that WSL detection is case-insensitive."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = "5.10.16.3-MICROSOFT-STANDARD"
            with patch('os.uname', return_value=mock_uname):
                caps = detect_platform()
                
                assert caps["is_wsl"] is True, "WSL detection should be case-insensitive"

    def test_wsl_detection_uname_exception_handled(self):
        """Test that exceptions from os.uname() are handled gracefully."""
        with patch('sys.platform', 'linux'):
            with patch('os.uname', side_effect=OSError("uname not available")):
                caps = detect_platform()
                
                # Should fall back to regular Linux detection
                assert caps["is_wsl"] is False, "Should handle uname exception and default to False"
                assert caps["platform"] == "linux", "Should default to linux when uname fails"

    def test_wsl_detection_no_uname_attribute(self):
        """Test that missing uname attribute is handled."""
        with patch('sys.platform', 'linux'):
            # Some systems might not have os.uname
            if hasattr(os, 'uname'):
                with patch('os.uname', side_effect=AttributeError("uname not available")):
                    caps = detect_platform()
                    assert caps["is_wsl"] is False, "Should handle missing uname gracefully"

    def test_macos_platform_capabilities(self):
        """Test that macOS platform has correct capabilities."""
        with patch('sys.platform', 'darwin'):
            with patch('os.uname', side_effect=AttributeError("macOS doesn't expose uname")):
                with patch('platform.machine', return_value='x86_64'):
                    caps = detect_platform()
                    
                    assert caps["platform"] == "macos", "Should detect macOS"
                    assert caps["memory_limiting_available"] is True, \
                        "macOS should have memory limiting"
                    assert caps["process_group_kill_available"] is True, \
                        "macOS should have process group kill"
                    assert caps["preexec_fn_available"] is True, \
                        "macOS should have preexec_fn"
                    assert caps["is_wsl"] is False, "macOS is not WSL"

    def test_macos_apple_silicon_warning(self):
        """Test that Apple Silicon detection adds a warning."""
        with patch('sys.platform', 'darwin'):
            with patch('os.uname', side_effect=AttributeError("macOS doesn't expose uname")):
                with patch('platform.machine', return_value='arm64'):
                    caps = detect_platform()
                    
                    assert caps["platform"] == "macos", "Should detect macOS"
                    assert len(caps["warnings"]) >= 1, \
                        "Apple Silicon should have at least one warning"
                    assert any("Apple Silicon" in w or "ARM64" in w or "Rosetta" in w 
                              for w in caps["warnings"]), \
                        "Apple Silicon should warn about ARM64/Rosetta compatibility"

    def test_macos_platform_machine_exception_handled(self):
        """Test that exceptions from platform.machine() are handled gracefully."""
        with patch('sys.platform', 'darwin'):
            with patch('os.uname', side_effect=AttributeError("macOS doesn't expose uname")):
                with patch('platform.machine', side_effect=Exception("platform.machine failed")):
                    caps = detect_platform()
                    
                    # Should still return valid macOS capabilities
                    assert caps["platform"] == "macos", "Should detect macOS even if platform.machine fails"
                    assert caps["memory_limiting_available"] is True, \
                        "macOS capabilities should still be correct"

    def test_linux_platform_capabilities(self):
        """Test that Linux platform has correct capabilities."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = "5.15.0-generic"
            with patch('os.uname', return_value=mock_uname):
                caps = detect_platform()
                
                assert caps["platform"] == "linux", "Should detect Linux"
                assert caps["memory_limiting_available"] is True, \
                    "Linux should have memory limiting"
                assert caps["process_group_kill_available"] is True, \
                    "Linux should have process group kill"
                assert caps["preexec_fn_available"] is True, \
                    "Linux should have preexec_fn"
                assert caps["is_wsl"] is False, "Regular Linux is not WSL"
                assert len(caps["warnings"]) == 0, "Linux should have no warnings"
                assert caps["recommended_action"] is None, "Linux should have no recommended action"

    def test_linux_fallback_when_uname_fails(self):
        """Test that Linux detection works when uname fails."""
        with patch('sys.platform', 'linux'):
            with patch('os.uname', side_effect=OSError("uname failed")):
                caps = detect_platform()
                
                # Should default to Linux (not WSL) when uname fails
                assert caps["platform"] == "linux", "Should default to linux when uname fails"
                assert caps["is_wsl"] is False, "Should not detect WSL when uname fails"
                assert caps["memory_limiting_available"] is True, \
                    "Linux capabilities should still be correct"

    def test_platform_consistency_wsl_vs_linux(self):
        """Test that WSL and Linux have consistent capability differences."""
        # Test WSL
        with patch('sys.platform', 'linux'):
            mock_uname_wsl = MagicMock()
            mock_uname_wsl.release = "5.10.16.3-microsoft-standard"
            with patch('os.uname', return_value=mock_uname_wsl):
                caps_wsl = detect_platform()
                assert caps_wsl["platform"] == "wsl"
                assert caps_wsl["is_wsl"] is True
        
        # Test regular Linux
        with patch('sys.platform', 'linux'):
            mock_uname_linux = MagicMock()
            mock_uname_linux.release = "5.15.0-generic"
            with patch('os.uname', return_value=mock_uname_linux):
                caps_linux = detect_platform()
                assert caps_linux["platform"] == "linux"
                assert caps_linux["is_wsl"] is False
        
        # Both should have same capabilities (both are Unix-like)
        assert caps_wsl["memory_limiting_available"] == caps_linux["memory_limiting_available"]
        assert caps_wsl["process_group_kill_available"] == caps_linux["process_group_kill_available"]
        assert caps_wsl["preexec_fn_available"] == caps_linux["preexec_fn_available"]

    def test_get_platform_capabilities_cached(self):
        """Test that get_platform_capabilities returns cached result."""
        # Clear the cache by reloading the module would be ideal, but we'll test the behavior
        caps1 = get_platform_capabilities()
        caps2 = get_platform_capabilities()
        
        # Should return the exact same object (cached)
        assert caps1 is caps2, "get_platform_capabilities should return cached result"
        
        # Values should be identical
        assert caps1 == caps2, "Cached values should be identical"

    def test_get_platform_capabilities_structure_matches_detect_platform(self):
        """Test that get_platform_capabilities returns same structure as detect_platform."""
        caps_detect = detect_platform()
        caps_get = get_platform_capabilities()
        
        # Should have same keys
        assert set(caps_detect.keys()) == set(caps_get.keys()), \
            "Both functions should return same keys"
        
        # Platform should match (they're detecting the same system)
        assert caps_detect["platform"] == caps_get["platform"], \
            "Both should detect the same platform"
        
        # Capabilities should match (same platform = same capabilities)
        assert caps_detect["memory_limiting_available"] == caps_get["memory_limiting_available"]
        assert caps_detect["process_group_kill_available"] == caps_get["process_group_kill_available"]
        assert caps_detect["preexec_fn_available"] == caps_get["preexec_fn_available"]
        assert caps_detect["is_wsl"] == caps_get["is_wsl"]

    def test_warnings_are_non_empty_strings(self):
        """Test that all warnings are non-empty strings."""
        caps = detect_platform()
        
        for warning in caps["warnings"]:
            assert isinstance(warning, str), f"Warning must be string, got {type(warning)}"
            assert len(warning.strip()) > 0, "Warning must not be empty"

    def test_recommended_action_format(self):
        """Test that recommended_action is properly formatted when present."""
        caps = detect_platform()
        
        if caps["recommended_action"] is not None:
            assert isinstance(caps["recommended_action"], str), \
                "recommended_action must be a string when not None"
            assert len(caps["recommended_action"].strip()) > 0, \
                "recommended_action must not be empty when present"

    def test_platform_detection_idempotent(self):
        """Test that detect_platform returns consistent results on multiple calls."""
        caps1 = detect_platform()
        caps2 = detect_platform()
        
        # Should return same values (though not necessarily same object)
        assert caps1 == caps2, "detect_platform should be idempotent"
        
        # All fields should match
        for key in caps1.keys():
            assert caps1[key] == caps2[key], \
                f"Field {key} should be consistent across calls"

    def test_wsl_takes_precedence_over_linux(self):
        """Test that WSL detection takes precedence over regular Linux detection."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = "5.10.16.3-microsoft-standard-WSL2"
            with patch('os.uname', return_value=mock_uname):
                caps = detect_platform()
                
                # Even though sys.platform is 'linux', WSL should be detected
                assert caps["platform"] == "wsl", \
                    "WSL detection should take precedence over Linux"
                assert caps["is_wsl"] is True, \
                    "is_wsl should be True when WSL is detected"

    def test_unknown_platform_falls_back_to_linux(self):
        """Test that unknown platforms fall back to Linux capabilities."""
        with patch('sys.platform', 'unknown_platform'):
            with patch('os.uname', side_effect=OSError("uname not available")):
                caps = detect_platform()
                
                # Should default to Linux (last fallback)
                assert caps["platform"] == "linux", \
                    "Unknown platform should default to linux"
                assert caps["memory_limiting_available"] is True, \
                    "Should have Linux capabilities as fallback"
                assert caps["is_wsl"] is False, \
                    "Unknown platform should not be detected as WSL"

    def test_windows_always_returns_windows_even_if_wsl_detected(self):
        """Test that Windows platform always returns Windows capabilities, even if WSL check runs."""
        with patch('sys.platform', 'win32'):
            # Simulate WSL detection (though this shouldn't happen on win32)
            mock_uname = MagicMock()
            mock_uname.release = "5.10.16.3-microsoft-standard"
            with patch('os.uname', return_value=mock_uname):
                caps = detect_platform()
                
                # Should still return Windows, not WSL
                assert caps["platform"] == "windows", \
                    "win32 platform should always return 'windows', not 'wsl'"
                assert caps["is_wsl"] is False, \
                    "Native Windows should never be detected as WSL"
                assert caps["memory_limiting_available"] is False, \
                    "Windows should have Windows capabilities, not Linux"

    def test_uname_release_missing_attribute(self):
        """Test handling when uname() returns object without release attribute."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock(spec=[])  # No attributes
            # Remove release attribute if it exists
            if hasattr(mock_uname, 'release'):
                delattr(mock_uname, 'release')
            with patch('os.uname', return_value=mock_uname):
                caps = detect_platform()
                
                # Should handle gracefully and default to Linux
                assert caps["platform"] == "linux", \
                    "Should handle missing release attribute gracefully"
                assert caps["is_wsl"] is False, \
                    "Should not detect WSL without release attribute"

    def test_uname_release_not_string(self):
        """Test handling when uname().release is not a string."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = 12345  # Not a string
            with patch('os.uname', return_value=mock_uname):
                # This should raise an AttributeError when calling .lower()
                # The code catches Exception, so it should handle this
                caps = detect_platform()
                
                # Should handle gracefully
                assert caps["platform"] in ["linux", "wsl"], \
                    "Should handle non-string release gracefully"

    def test_uname_release_none(self):
        """Test handling when uname().release is None."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = None
            with patch('os.uname', return_value=mock_uname):
                # Calling .lower() on None will raise AttributeError
                # The code catches Exception, so it should handle this
                caps = detect_platform()
                
                # Should handle gracefully and default to Linux
                assert caps["platform"] == "linux", \
                    "Should handle None release gracefully"
                assert caps["is_wsl"] is False, \
                    "Should not detect WSL with None release"

    def test_cygwin_platform_handling(self):
        """Test handling of Cygwin platform (common Unix-like on Windows)."""
        with patch('sys.platform', 'cygwin'):
            with patch('os.uname', side_effect=OSError("uname not available")):
                caps = detect_platform()
                
                # Should fall back to Linux capabilities
                assert caps["platform"] == "linux", \
                    "Cygwin should fall back to linux capabilities"
                assert caps["memory_limiting_available"] is True, \
                    "Should have Unix-like capabilities"

    def test_aix_platform_handling(self):
        """Test handling of AIX platform."""
        with patch('sys.platform', 'aix'):
            with patch('os.uname', side_effect=OSError("uname not available")):
                caps = detect_platform()
                
                # Should fall back to Linux capabilities
                assert caps["platform"] == "linux", \
                    "AIX should fall back to linux capabilities"

    def test_hasattr_uname_but_call_fails(self):
        """Test that code handles case where hasattr passes but os.uname() call fails."""
        with patch('sys.platform', 'linux'):
            # hasattr(os, 'uname') returns True, but calling it raises exception
            with patch('os.uname', side_effect=RuntimeError("uname call failed")):
                caps = detect_platform()
                
                # Should handle gracefully
                assert caps["platform"] == "linux", \
                    "Should handle uname call failure gracefully"
                assert caps["is_wsl"] is False, \
                    "Should not detect WSL when uname call fails"

    def test_check_platform_and_warn_emits_warnings(self):
        """Test that check_platform_and_warn emits warnings when warnings exist."""
        with patch('sys.platform', 'win32'):
            with patch('os.uname', side_effect=AttributeError("Windows doesn't have uname")):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # Clear environment variable to ensure warnings are emitted
                    with patch.dict(os.environ, {}, clear=True):
                        caps = check_platform_and_warn()
                    
                    # Should have emitted warnings
                    assert len(w) >= 2, \
                        f"Should emit at least 2 warnings for Windows, got {len(w)}"
                    
                    # Check that warnings are RuntimeWarning
                    runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
                    assert len(runtime_warnings) >= 2, \
                        "Should emit RuntimeWarning instances"
                    
                    # Verify caps are returned correctly
                    assert caps["platform"] == "windows", \
                        "Should return correct platform capabilities"

    def test_check_platform_and_warn_suppresses_warnings_with_env_var(self):
        """Test that check_platform_and_warn suppresses warnings when env var is set."""
        with patch('sys.platform', 'win32'):
            with patch('os.uname', side_effect=AttributeError("Windows doesn't have uname")):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # Set environment variable to suppress warnings
                    with patch.dict(os.environ, {"REPROLAB_SKIP_RESOURCE_LIMITS": "1"}):
                        caps = check_platform_and_warn()
                    
                    # Should NOT have emitted warnings
                    runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
                    assert len(runtime_warnings) == 0, \
                        f"Should suppress warnings when env var is set, got {len(runtime_warnings)} warnings"
                    
                    # But should still return correct caps
                    assert caps["platform"] == "windows", \
                        "Should still return correct platform capabilities"

    def test_check_platform_and_warn_no_warnings_when_none_exist(self):
        """Test that check_platform_and_warn doesn't emit warnings when none exist."""
        with patch('sys.platform', 'linux'):
            mock_uname = MagicMock()
            mock_uname.release = "5.15.0-generic"
            with patch('os.uname', return_value=mock_uname):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    with patch.dict(os.environ, {}, clear=True):
                        caps = check_platform_and_warn()
                    
                    # Should NOT have emitted warnings (Linux has no warnings)
                    runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
                    assert len(runtime_warnings) == 0, \
                        f"Should not emit warnings when none exist, got {len(runtime_warnings)} warnings"
                    
                    # Should still return correct caps
                    assert caps["platform"] == "linux", \
                        "Should return correct platform capabilities"

    def test_check_platform_and_warn_emits_recommended_action(self):
        """Test that check_platform_and_warn emits recommended_action as warning."""
        with patch('sys.platform', 'win32'):
            with patch('os.uname', side_effect=AttributeError("Windows doesn't have uname")):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    with patch.dict(os.environ, {}, clear=True):
                        caps = check_platform_and_warn()
                    
                    # Should have emitted recommended_action warning
                    warning_messages = [str(warning.message) for warning in w]
                    recommended_warnings = [msg for msg in warning_messages if "RECOMMENDED:" in msg]
                    assert len(recommended_warnings) >= 1, \
                        f"Should emit recommended_action warning, got warnings: {warning_messages}"
                    assert any("WSL2" in msg or "Docker" in msg for msg in recommended_warnings), \
                        "Recommended action should mention WSL2 or Docker"

    def test_check_platform_and_warn_env_var_not_set_emits_warnings(self):
        """Test that warnings are emitted when env var is not set or set to '0'."""
        with patch('sys.platform', 'win32'):
            with patch('os.uname', side_effect=AttributeError("Windows doesn't have uname")):
                # Test with env var not set
                with warnings.catch_warnings(record=True) as w1:
                    warnings.simplefilter("always")
                    with patch.dict(os.environ, {}, clear=True):
                        check_platform_and_warn()
                    runtime_warnings_1 = [w for w in w1 if issubclass(w.category, RuntimeWarning)]
                
                # Test with env var set to '0'
                with warnings.catch_warnings(record=True) as w2:
                    warnings.simplefilter("always")
                    with patch.dict(os.environ, {"REPROLAB_SKIP_RESOURCE_LIMITS": "0"}):
                        check_platform_and_warn()
                    runtime_warnings_2 = [w for w in w2 if issubclass(w.category, RuntimeWarning)]
                
                # Both should emit warnings
                assert len(runtime_warnings_1) > 0, \
                    "Should emit warnings when env var is not set"
                assert len(runtime_warnings_2) > 0, \
                    "Should emit warnings when env var is set to '0'"

