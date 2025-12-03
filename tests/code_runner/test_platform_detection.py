"""Tests for platform detection utilities in `src.code_runner`."""

import sys

from src.code_runner import detect_platform, get_platform_capabilities


class TestPlatformDetection:
    """Tests for platform detection functions."""

    def test_detect_platform_returns_valid_capabilities(self):
        """Test that detect_platform returns a valid capabilities dict."""
        caps = detect_platform()

        assert isinstance(caps, dict)
        assert "platform" in caps
        assert "memory_limiting_available" in caps
        assert "process_group_kill_available" in caps
        assert "preexec_fn_available" in caps
        assert "is_wsl" in caps
        assert "warnings" in caps
        assert isinstance(caps["warnings"], list)

    def test_detect_platform_identifies_current_os(self):
        """Test that platform detection matches sys.platform."""
        caps = detect_platform()

        if sys.platform == "darwin":
            assert caps["platform"] == "macos"
        elif sys.platform == "win32":
            assert caps["platform"] == "windows"
        elif sys.platform.startswith("linux"):
            # WSL check might override "linux" to "wsl"
            if caps["is_wsl"]:
                assert caps["platform"] == "wsl"
            else:
                assert caps["platform"] == "linux"

    def test_get_platform_capabilities_cached(self):
        """Test that get_platform_capabilities returns cached result."""
        caps1 = get_platform_capabilities()
        caps2 = get_platform_capabilities()
        assert caps1 is caps2

