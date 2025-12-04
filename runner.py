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
config = {"configurable": {"thread_id": "my_run_1"}}
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
            config
        )
    else:
        print(f"Unexpected pause at: {snapshot.next}")
        break

print("\nFinal result keys:", list(result.keys()) if result else "None")