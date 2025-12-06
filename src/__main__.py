"""
CLI entry point for ReproLab graph execution.

Usage:
    # Start a new run from a paper markdown file
    python -m src --paper papers/my_paper.md --output-dir ./outputs --paper-id my_paper
    
    # Resume from a checkpoint (after interrupt/timeout/Ctrl+C)
    python -m src --resume outputs/my_paper/run_20241204_123456/checkpoints/
    
    # Resume with a specific thread_id
    python -m src --resume outputs/my_paper/run_20241204_123456/checkpoints/ --thread-id my_run_1
"""

import argparse
import logging
import sys
from pathlib import Path

from langgraph.types import Command

# Set up logging before other imports
from src.logging_utils import setup_console_logging, setup_file_logging
from src.cli_utils import get_user_input_with_confirmation
setup_console_logging()

logger = logging.getLogger(__name__)


def run_new_paper(args):
    """Run the graph on a new paper."""
    from src.callbacks import GraphProgressCallback
    from src.graph import create_repro_graph
    from src.paper_loader import load_paper_from_markdown, create_state_from_paper_input
    
    # Load paper from markdown
    paper_input = load_paper_from_markdown(
        markdown_path=args.paper,
        output_dir=args.output_dir,
        paper_id=args.paper_id,
        paper_domain=args.paper_domain,
    )
    
    # Set up file logging now that we have the run folder
    setup_file_logging(paper_input["run_output_dir"])
    logger.info(f"📁 Run output directory: {paper_input['run_output_dir']}")
    
    # Convert to initial state
    initial_state = create_state_from_paper_input(paper_input)
    
    # Create graph with persistent checkpointing
    checkpoint_dir = Path(paper_input["run_output_dir"]) / "checkpoints"
    app = create_repro_graph(checkpoint_dir=str(checkpoint_dir))
    
    # Create callback instance
    progress_callback = GraphProgressCallback()
    
    # Run with interrupt handling
    thread_id = args.thread_id or "repro_run_1"
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [progress_callback]}
    result = app.invoke(initial_state, config)
    
    # Handle interrupts for user input
    # With the interrupt() pattern, ask_user node pauses mid-execution and provides
    # a payload with questions. We resume with Command(resume=user_response).
    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:  # No next node = finished
            print("\n✅ Graph completed!")
            break
        
        # Check for interrupt payload from ask_user node's interrupt() call
        interrupt_payload = None
        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    interrupt_payload = task.interrupts[0].value
                    break
        
        if interrupt_payload:
            # Show what the system is asking
            print("\n" + "=" * 60)
            print("🤖 SYSTEM NEEDS YOUR INPUT")
            print("=" * 60)
            
            trigger = interrupt_payload.get("trigger", "unknown")
            questions = interrupt_payload.get("questions", [])
            
            print(f"Trigger: {trigger}")
            for q in questions:
                print(f"\n{q}")
            
            # Get user response with echo-back confirmation to catch terminal corruption
            print("\n" + "-" * 40)
            user_input = get_user_input_with_confirmation("Your response (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                print("\n💾 State saved to checkpoint. Resume later with:")
                print(f"  python -m src --resume {checkpoint_dir}")
                break
            
            # Resume with user response using Command(resume=...)
            result = app.invoke(Command(resume=user_input), config)
        else:
            print(f"Unexpected pause at: {snapshot.next}")
            break
    
    print("\nFinal result keys:", list(result.keys()) if result else "None")
    return 0


def resume_from_checkpoint(args):
    """Resume execution from a checkpoint."""
    from src.callbacks import GraphProgressCallback
    from src.graph import create_repro_graph
    from src.persistence import JsonCheckpointSaver
    
    checkpoint_dir = Path(args.resume)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # The langgraph subdirectory contains the actual checkpoints
    langgraph_dir = checkpoint_dir / "langgraph"
    if not langgraph_dir.exists():
        print(f"❌ No LangGraph checkpoints found in: {checkpoint_dir}")
        print("   (Expected 'langgraph/' subdirectory)")
        return 1
    
    # Infer run_output_dir from checkpoint_dir (it's the parent)
    run_output_dir = checkpoint_dir.parent
    
    # Set up file logging
    if (run_output_dir / "info.log").exists():
        setup_file_logging(str(run_output_dir))
    
    logger.info(f"📂 Resuming from checkpoint: {checkpoint_dir}")
    
    # Create graph with the same checkpoint directory
    app = create_repro_graph(checkpoint_dir=str(checkpoint_dir))
    
    # Get thread_id - either from args or try to find from checkpoint
    thread_id = args.thread_id
    if not thread_id:
        # Try to find the thread_id from existing checkpoints
        checkpointer = JsonCheckpointSaver(str(checkpoint_dir))
        # Look for any checkpoint files to get thread_id
        checkpoint_files = list(langgraph_dir.glob("checkpoint_*.json"))
        if checkpoint_files:
            # Parse thread_id from filename: checkpoint_{thread_id}_{checkpoint_id}.json
            filename = checkpoint_files[0].stem
            parts = filename.split("_", 2)  # ["checkpoint", thread_id, checkpoint_id]
            if len(parts) >= 2:
                thread_id = parts[1]
    
    if not thread_id:
        print("❌ Could not determine thread_id. Please specify with --thread-id")
        return 1
    
    logger.info(f"🧵 Using thread_id: {thread_id}")
    
    # Create callback instance
    progress_callback = GraphProgressCallback()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [progress_callback]}
    
    # Get the current state to see what question was pending
    snapshot = app.get_state(config)
    
    if not snapshot or not snapshot.values:
        print("❌ No checkpoint state found. The checkpoint may be empty or corrupted.")
        return 1
    
    # Check for interrupt payload from ask_user node's interrupt() call
    interrupt_payload = None
    if snapshot.tasks:
        for task in snapshot.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break
    
    if not interrupt_payload:
        print("ℹ️  No pending interrupt in checkpoint. Resuming execution...")
        # Resume execution
        logger.info("▶️  Resuming graph execution...")
        result = app.invoke(None, config)
    else:
        # Show the pending question and prompt for response
        print("\n" + "=" * 60)
        print("🤖 RESUMING - SYSTEM NEEDS YOUR INPUT")
        print("=" * 60)
        
        trigger = interrupt_payload.get("trigger", "unknown")
        questions = interrupt_payload.get("questions", [])
        
        print(f"Trigger: {trigger}")
        for q in questions:
            print(f"\n{q}")
        
        # Get user response with echo-back confirmation to catch terminal corruption
        print("\n" + "-" * 40)
        user_input = get_user_input_with_confirmation("Your response (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("Exiting without resume.")
            return 0
        
        # Resume with user response using Command(resume=...)
        logger.info("▶️  Resuming graph execution...")
        result = app.invoke(Command(resume=user_input), config)
    
    # Continue handling any further interrupts
    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:
            print("\n✅ Graph completed!")
            break
        
        # Check for interrupt payload
        interrupt_payload = None
        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    interrupt_payload = task.interrupts[0].value
                    break
        
        if interrupt_payload:
            trigger = interrupt_payload.get("trigger", "unknown")
            questions = interrupt_payload.get("questions", [])
            
            print("\n" + "=" * 60)
            print("🤖 SYSTEM NEEDS YOUR INPUT")
            print("=" * 60)
            print(f"Trigger: {trigger}")
            for q in questions:
                print(f"\n{q}")
            
            # Get user response with echo-back confirmation to catch terminal corruption
            print("\n" + "-" * 40)
            user_input = get_user_input_with_confirmation("Your response (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                print("\n💾 State saved to checkpoint. Resume later with:")
                print(f"  python -m src --resume {checkpoint_dir}")
                break
            
            result = app.invoke(Command(resume=user_input), config)
        else:
            print(f"Unexpected pause at: {snapshot.next}")
            break
    
    print("\nFinal result keys:", list(result.keys()) if result else "None")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ReproLab - Paper Reproduction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new run
  python -m src --paper papers/my_paper.md --paper-id my_paper

  # Resume from checkpoint
  python -m src --resume outputs/my_paper/run_20241204_123456/checkpoints/
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--paper", "-p",
        help="Path to paper markdown file (starts new run)"
    )
    mode_group.add_argument(
        "--resume", "-r",
        help="Path to checkpoint directory (resumes interrupted run)"
    )
    
    # Options for new runs
    parser.add_argument(
        "--output-dir", "-o",
        default="./outputs",
        help="Output directory for run artifacts (default: ./outputs)"
    )
    parser.add_argument(
        "--paper-id",
        help="Paper identifier (default: derived from filename)"
    )
    parser.add_argument(
        "--paper-domain",
        default="plasmonics",
        help="Paper domain for prompt adaptation (default: plasmonics)"
    )
    parser.add_argument(
        "--thread-id", "-t",
        help="Thread ID for LangGraph (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.paper:
        paper_path = Path(args.paper)
        if not paper_path.exists():
            print(f"❌ Paper file not found: {args.paper}")
            return 1
        
        # Derive paper_id from filename if not provided
        if not args.paper_id:
            args.paper_id = paper_path.stem.replace(" ", "_").lower()
        
        return run_new_paper(args)
    
    elif args.resume:
        return resume_from_checkpoint(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
