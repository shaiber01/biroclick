import logging
import sys

# =============================================================================
# FAIL-FAST: Check that meep is available in the current Python environment
# =============================================================================
try:
    import meep  # noqa: F401
except ImportError:
    print("=" * 70)
    print("ERROR: meep is not installed in the current Python environment.")
    print("=" * 70)
    print(f"Current Python: {sys.executable}")
    print()
    print("The simulation runner requires meep to be installed.")
    print("Please activate your meep conda environment and try again:")
    print()
    print("    conda activate <your-meep-env>")
    print("    python runner.py")
    print()
    print("Or install meep in your current environment:")
    print()
    print("    conda install -c conda-forge pymeep")
    print("=" * 70)
    sys.exit(1)

# Import and initialize logging utilities (must be first to register VERBOSE level)
from src.logging_utils import setup_console_logging, setup_file_logging

# Phase 1: Console logging only (before we know the run folder)
setup_console_logging()

# Get a logger for this module
logger = logging.getLogger(__name__)

from src.callbacks import GraphProgressCallback
from src.graph import create_repro_graph
from src.paper_loader import load_paper_from_markdown, create_state_from_paper_input

# Create callback instance for reuse
progress_callback = GraphProgressCallback()

# Initialize the graph
app = create_repro_graph()

# Load paper from markdown (auto-downloads figures)
paper_input = load_paper_from_markdown(
    markdown_path="./papers/Aluminum Nanoantenna Complexes for Strong Coupling between Excitons and Localized Surface Plasmons - short.md",
    output_dir="./outputs",
    paper_id="aluminum_nanoantenna_complexes",
    paper_domain="plasmonics"
)

# Phase 2: Now we know the run folder - add file logging
setup_file_logging(paper_input["run_output_dir"])
logger.info(f"📁 Run output directory: {paper_input['run_output_dir']}")

# Convert to initial state
initial_state = create_state_from_paper_input(paper_input)

# Run with interrupt handling
config = {"configurable": {"thread_id": "my_run_1"}, "callbacks": [progress_callback]}
result = app.invoke(initial_state, config)

# Handle interrupts for user input
while True:
    # Check if we're paused at ask_user
    snapshot = app.get_state(config)
    if not snapshot.next:  # No next node = finished
        print("\n✅ Graph completed!")
        break
    
    if "ask_user" in snapshot.next:
        # Show what the system is asking
        print("\n" + "="*60)
        print("🤖 SYSTEM NEEDS YOUR INPUT")
        print("="*60)
        
        state = snapshot.values
        questions = state.get("pending_user_questions", [])
        trigger = state.get("ask_user_trigger", "unknown")
        
        print(f"Trigger: {trigger}")
        for q in questions:
            print(f"  - {q}")
        
        # Get user response
        user_input = input("\nYour response (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        
        # Resume with user response
        # IMPORTANT: Use update_state() then invoke(None) to properly resume from interrupt.
        # Calling invoke(input, config) would start a NEW run from START, not resume!
        app.update_state(
            config,
            {"user_responses": {trigger: user_input}},
        )
        result = app.invoke(None, config)
    else:
        print(f"Unexpected pause at: {snapshot.next}")
        break

print("\nFinal result keys:", list(result.keys()) if result else "None")
