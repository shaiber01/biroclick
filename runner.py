import logging
import sys
from pathlib import Path

from langgraph.types import Command


def get_user_input_with_confirmation(prompt: str = "Your response (or 'quit' to exit): ") -> str:
    """
    Get user input with echo-back confirmation for longer responses.
    
    This helps catch terminal readline buffer corruption that can occur in some
    IDE terminals when users edit their input with arrow keys or backspace.
    The terminal may display one thing but Python's input() receives corrupted data.
    """
    user_input = input(prompt)
    
    # For longer responses (where corruption is more likely), echo back and offer re-entry
    if len(user_input) > 30 and user_input.lower() != 'quit':
        print(f"\n📝 Captured ({len(user_input)} chars): {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        confirm = input("Press Enter to confirm, or 'r' to re-enter: ").strip().lower()
        if confirm == 'r':
            user_input = input("Re-enter your response: ")
            # Show the re-entered response too for verification
            if len(user_input) > 30:
                print(f"📝 Captured: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
    
    return user_input

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

# Initialize the graph with persistent checkpointing
# This enables resume from interrupts after process exit
checkpoint_dir = str(Path(paper_input["run_output_dir"]) / "checkpoints")
app = create_repro_graph(checkpoint_dir=checkpoint_dir)
logger.info(f"💾 Checkpoints will be saved to: {checkpoint_dir}")

# Convert to initial state
initial_state = create_state_from_paper_input(paper_input)

# Run with interrupt handling
config = {"configurable": {"thread_id": "my_run_1"}, "callbacks": [progress_callback]}
result = app.invoke(initial_state, config)

# Handle interrupts for user input
# With the interrupt() pattern, the ask_user node pauses mid-execution via interrupt()
# and provides a payload with questions. We resume with Command(resume=user_response).
while True:
    # Check if we're paused (interrupted)
    snapshot = app.get_state(config)
    if not snapshot.next:  # No next node = finished
        print("\n✅ Graph completed!")
        break
    
    # Check for interrupt payload from ask_user node's interrupt() call
    # The interrupt payload is in snapshot.tasks[0].interrupts[0].value
    interrupt_payload = None
    if snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break
    
    if interrupt_payload:
        # Show what the system is asking
        print("\n" + "="*60)
        print("🤖 SYSTEM NEEDS YOUR INPUT")
        print("="*60)
        
        trigger = interrupt_payload.get("trigger", "unknown")
        questions = interrupt_payload.get("questions", [])
        paper_id = interrupt_payload.get("paper_id", "unknown")
        
        print(f"Paper: {paper_id}")
        print(f"Trigger: {trigger}")
        for q in questions:
            print(f"\n{q}")
        
        # Get user response with echo-back confirmation to catch terminal corruption
        print("-" * 60)
        user_input = get_user_input_with_confirmation("\nYour response (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("\n💾 State saved to checkpoint. Resume later with:")
            print(f"  python -m src --resume {checkpoint_dir}")
            break
        
        # Resume with user response using Command(resume=...)
        # This passes the response directly to the interrupt() call's return value
        result = app.invoke(Command(resume=user_input), config)
    else:
        print(f"Unexpected pause at: {snapshot.next}")
        break

print("\nFinal result keys:", list(result.keys()) if result else "None")
