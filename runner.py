import logging
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler

# Add custom VERBOSE level (between DEBUG=10 and INFO=20)
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)

logging.Logger.verbose = verbose


def setup_console_logging():
    """Set up console-only logging (before we know the run folder)."""
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
    """Add file handlers once we know the run folder.
    
    Creates three log files:
    - debug.log: Everything (DEBUG and above)
    - verbose.log: VERBOSE and above (no DEBUG)
    - info.log: INFO and above only
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


# Phase 1: Console logging only (before we know the run folder)
setup_console_logging()

# Get a logger for this module
logger = logging.getLogger(__name__)


class GraphProgressCallback(BaseCallbackHandler):
    """Log graph node transitions for visibility into execution progress."""
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain/node starts executing."""
        # Try multiple sources for the node name:
        # 1. kwargs["name"] - LangGraph often passes node name directly here
        # 2. kwargs["tags"] - node name may be in tags list
        # 3. serialized["name"] - fallback for LangChain chains
        name = kwargs.get("name")
        
        if not name:
            tags = kwargs.get("tags", [])
            # LangGraph node names are often in tags, filter out internal ones
            internal_tags = {"seq:step", "langsmith:hidden", "__start__"}
            for tag in tags:
                if tag not in internal_tags and not tag.startswith("seq:step:") and not tag.startswith("graph:step:"):
                    name = tag
                    break
        
        if not name and serialized:
            name = serialized.get("name")
        
        if not name:
            return  # Skip logging if we can't determine the name
            
        # Filter out internal LangGraph wrappers to show only meaningful nodes
        internal_names = {"RunnableSequence", "StateGraph", "CompiledStateGraph", "ChannelWrite", "ChannelRead", "RunnableLambda"}
        if name not in internal_names and not name.startswith("Runnable"):
            logger.info(f"🔄 Entering node: {name}")
    
    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain/node finishes executing."""
        pass  # Could add logging here if you want exit messages
    
    def on_chain_error(self, error, **kwargs):
        """Called when a chain/node errors."""
        logger.error(f"❌ Node error: {error}")


# Create callback instance for reuse
progress_callback = GraphProgressCallback()

from src.graph import create_repro_graph
from src.paper_loader import load_paper_from_markdown, create_state_from_paper_input

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
        result = app.invoke(
            {"user_responses": {trigger: user_input}},
            config  # callbacks already in config
        )
    else:
        print(f"Unexpected pause at: {snapshot.next}")
        break

print("\nFinal result keys:", list(result.keys()) if result else "None")
