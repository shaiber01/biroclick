import logging
from langchain_core.callbacks import BaseCallbackHandler

# Create a custom logger setup with different levels for console vs file
def setup_logging():
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything at root level
    
    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    
    # File handler - DEBUG level
    file_handler = logging.FileHandler("repro_debug.log", mode="w")  # 'w' overwrites each run
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Enable LangChain/LangGraph debug logging (goes to file only due to handler levels)
    logging.getLogger("langchain").setLevel(logging.DEBUG)
    logging.getLogger("langchain_core").setLevel(logging.DEBUG)
    logging.getLogger("langchain_anthropic").setLevel(logging.DEBUG)
    logging.getLogger("langgraph").setLevel(logging.DEBUG)

setup_logging()

# Get a logger for this module
logger = logging.getLogger(__name__)


class GraphProgressCallback(BaseCallbackHandler):
    """Log graph node transitions for visibility into execution progress."""
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain/node starts executing."""
        name = serialized.get("name", "unknown") if serialized else "unknown"
        # Filter out internal LangGraph wrappers to show only meaningful nodes
        internal_names = {"RunnableSequence", "StateGraph", "CompiledStateGraph", "ChannelWrite", "ChannelRead"}
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