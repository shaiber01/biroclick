"""
Logging utilities for the biroclick project.

This module provides:
- Custom VERBOSE log level (between DEBUG and INFO)
- Two-phase logging setup functions for console and file handlers
"""

import logging
from pathlib import Path

# Custom VERBOSE level (between DEBUG=10 and INFO=20)
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")


def _verbose(self, message, *args, **kwargs):
    """Log at VERBOSE level (between DEBUG and INFO)."""
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


# Add verbose() method to all loggers
logging.Logger.verbose = _verbose


def setup_console_logging():
    """Set up console-only logging.
    
    Call this early, before the run folder is known.
    Console shows INFO and above.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything at root level
    
    # Console handler - INFO level (user sees INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Enable LangChain/LangGraph debug logging (will go to file once file handlers are added)
    for name in ["langchain", "langchain_core", "langchain_anthropic", "langgraph"]:
        logging.getLogger(name).setLevel(logging.DEBUG)


def setup_file_logging(run_output_dir: str):
    """Add file handlers once the run folder is known.
    
    Creates three log files in the run folder:
    - debug.log: Everything (DEBUG and above)
    - verbose.log: VERBOSE and above (no DEBUG)
    - info.log: INFO and above only
    
    Args:
        run_output_dir: Path to the run-specific output directory
    """
    root_logger = logging.getLogger()
    log_dir = Path(run_output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # DEBUG level log (everything)
    debug_handler = logging.FileHandler(log_dir / "debug.log", mode="w")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(file_format)
    root_logger.addHandler(debug_handler)
    
    # VERBOSE level log (VERBOSE and above, no DEBUG)
    verbose_handler = logging.FileHandler(log_dir / "verbose.log", mode="w")
    verbose_handler.setLevel(VERBOSE)
    verbose_handler.setFormatter(file_format)
    root_logger.addHandler(verbose_handler)
    
    # INFO level log (INFO and above)
    info_handler = logging.FileHandler(log_dir / "info.log", mode="w")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_format)
    root_logger.addHandler(info_handler)

