#!/usr/bin/env python3
"""Visualize the LangGraph workflow.

This script generates visualizations of the LangGraph state machine:
- ASCII representation (prints to console) - requires grandalf package
- Mermaid diagram (saved to file and printed)

Usage:
    python visualize_graph.py

Note: For ASCII visualization, install grandalf:
    pip install grandalf
"""

from src.graph import create_repro_graph


def main():
    """Generate and display workflow visualizations."""
    print("=" * 80)
    print("LangGraph Workflow Visualization")
    print("=" * 80)
    
    # Create the compiled graph
    print("\n📊 Creating graph...")
    app = create_repro_graph()
    graph = app.get_graph()
    
    # ASCII visualization (requires grandalf)
    print("\n" + "=" * 80)
    print("ASCII Representation:")
    print("=" * 80 + "\n")
    try:
        ascii_diagram = graph.draw_ascii()
        print(ascii_diagram)
    except ImportError as e:
        print("⚠️  ASCII visualization requires 'grandalf' package.")
        print("   Install it with: pip install grandalf")
        print("   Or skip ASCII and use the Mermaid diagram below.\n")
    
    # Mermaid diagram
    print("\n" + "=" * 80)
    print("Mermaid Diagram:")
    print("=" * 80 + "\n")
    mermaid_diagram = graph.draw_mermaid()
    print(mermaid_diagram)
    
    # Save Mermaid to file
    output_file = "workflow_diagram.mmd"
    with open(output_file, "w") as f:
        f.write(mermaid_diagram)
    
    print("\n" + "=" * 80)
    print(f"✅ Mermaid diagram saved to: {output_file}")
    print("   View it at: https://mermaid.live/")
    print("   Or add it to a Markdown file with:")
    print(f"   ```mermaid")
    print(f"   {mermaid_diagram[:100]}...")
    print(f"   ```")
    print("=" * 80)


if __name__ == "__main__":
    main()

