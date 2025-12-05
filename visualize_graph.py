#!/usr/bin/env python3
"""Visualize the LangGraph workflow.

This script generates visualizations of the LangGraph state machine:
- ASCII representation (prints to console) - requires grandalf package
- Mermaid diagram (saved to file and printed)
- Enhanced Mermaid diagram with edge labels (custom generator)

Usage:
    python visualize_graph.py

Note: For ASCII visualization, install grandalf:
    pip install grandalf
"""

import ast
import inspect
import os
from typing import Set

from src.graph import create_repro_graph


def detect_llm_nodes(app) -> Set[str]:
    """
    Dynamically detect which nodes make LLM calls by analyzing source code.
    
    This function inspects each node's implementation to check if it (or any
    helper function it calls within the same module) calls 'call_agent_with_metrics'.
    
    The detection works by:
    1. Getting the source file of each node function
    2. Building a call graph of all functions in that module
    3. Checking if the node function transitively calls any function that
       invokes 'call_agent_with_metrics'
    
    Returns:
        Set of node names that make LLM calls
    """
    graph = app.get_graph()
    llm_nodes = set()
    
    # Cache analyzed modules to avoid re-parsing
    module_cache = {}  # module_file -> (functions_with_llm_calls, call_graph)
    
    for node_name, node_data in graph.nodes.items():
        # Skip special nodes
        if node_name in ("__start__", "__end__"):
            continue
        
        # Get the node's callable from the data attribute
        # LangGraph wraps node functions in RunnableCallable
        runnable = node_data.data
        
        # Get the underlying function from the RunnableCallable
        func = None
        if hasattr(runnable, 'func'):
            func = runnable.func
        elif callable(runnable):
            func = runnable
        
        if func is None:
            continue
        
        # Unwrap decorated functions to get the original
        while hasattr(func, '__wrapped__'):
            func = func.__wrapped__
            
        try:
            # Get the source file of the function
            source_file = inspect.getfile(func)
            func_name = func.__name__
            
            # Analyze module if not cached
            if source_file not in module_cache:
                module_cache[source_file] = _analyze_module_for_llm_calls(source_file)
            
            direct_llm_funcs, call_graph = module_cache[source_file]
            
            # Check if this function or any function it calls (transitively) makes LLM calls
            if _function_uses_llm(func_name, direct_llm_funcs, call_graph):
                llm_nodes.add(node_name)
                
        except (OSError, TypeError):
            # Could not get source (built-in function, etc.)
            continue
    
    return llm_nodes


def _analyze_module_for_llm_calls(source_file: str) -> tuple:
    """
    Analyze a Python source file to find LLM call patterns.
    
    Returns:
        Tuple of (functions_with_direct_llm_calls, call_graph)
        - functions_with_direct_llm_calls: Set of function names that directly call call_agent_with_metrics
        - call_graph: Dict mapping function names to sets of functions they call
    """
    with open(source_file, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    # Find all function definitions and analyze them
    direct_llm_funcs = set()
    call_graph = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            calls_in_func = set()
            has_direct_llm_call = False
            
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    # Check for call_agent_with_metrics
                    if isinstance(child.func, ast.Name):
                        calls_in_func.add(child.func.id)
                        if child.func.id == 'call_agent_with_metrics':
                            has_direct_llm_call = True
                    elif isinstance(child.func, ast.Attribute):
                        calls_in_func.add(child.func.attr)
                        if child.func.attr == 'call_agent_with_metrics':
                            has_direct_llm_call = True
            
            call_graph[func_name] = calls_in_func
            if has_direct_llm_call:
                direct_llm_funcs.add(func_name)
    
    return direct_llm_funcs, call_graph


def _function_uses_llm(func_name: str, direct_llm_funcs: Set[str], call_graph: dict, visited: Set[str] = None) -> bool:
    """
    Check if a function uses LLM (directly or transitively through called functions).
    
    Args:
        func_name: Name of function to check
        direct_llm_funcs: Set of functions that directly call call_agent_with_metrics
        call_graph: Dict mapping function names to functions they call
        visited: Set of already visited functions (to prevent infinite loops)
        
    Returns:
        True if the function (or any function it calls) uses LLM
    """
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion
    if func_name in visited:
        return False
    visited.add(func_name)
    
    # Direct LLM call
    if func_name in direct_llm_funcs:
        return True
    
    # Check called functions
    called_funcs = call_graph.get(func_name, set())
    for called_func in called_funcs:
        if _function_uses_llm(called_func, direct_llm_funcs, call_graph, visited):
            return True
    
    return False


def create_labeled_mermaid_diagram(app, llm_nodes: Set[str] = None):
    """
    Create a custom Mermaid diagram with edge labels based on routing conditions.
    
    This function reads the graph structure and routing configurations to generate
    a fully labeled Mermaid diagram that explains the conditions for each edge.
    
    Args:
        app: The compiled LangGraph application
        llm_nodes: Optional set of node names that make LLM calls. If None,
                   will be detected automatically.
    """
    graph = app.get_graph()
    
    # Detect LLM nodes if not provided
    if llm_nodes is None:
        llm_nodes = detect_llm_nodes(app)
    
    # Define routing labels based on routing.py configurations and graph.py logic
    edge_labels = {
        # Direct edges (always taken) - None means no label needed
        ("__start__", "adapt_prompts"): None,
        ("adapt_prompts", "planning"): None,
        ("design", "design_review"): None,
        ("generate_code", "code_review"): None,
        ("run_code", "execution_check"): None,
        ("analyze", "comparison_check"): None,
        ("handle_backtrack", "select_stage"): None,
        ("generate_report", "__end__"): None,
        
        # Conditional edges with labels
        # plan_review routes
        ("planning", "plan_review"): "always",
        ("plan_review", "select_stage"): "approve",
        ("plan_review", "planning"): "needs_revision",
        ("plan_review", "ask_user"): "limit_reached / error",
        
        # select_stage routes
        ("select_stage", "design"): "has next stage",
        ("select_stage", "generate_report"): "no more stages",
        
        # design_review routes
        ("design_review", "generate_code"): "approve",
        ("design_review", "design"): "needs_revision",
        ("design_review", "ask_user"): "limit_reached / error",
        
        # code_review routes
        ("code_review", "run_code"): "approve",
        ("code_review", "generate_code"): "needs_revision",
        ("code_review", "ask_user"): "limit_reached / error",
        
        # execution_check routes
        ("execution_check", "physics_check"): "pass / warning",
        ("execution_check", "generate_code"): "fail",
        ("execution_check", "ask_user"): "limit_reached / error",
        
        # physics_check routes
        ("physics_check", "analyze"): "pass / warning",
        ("physics_check", "generate_code"): "fail",
        ("physics_check", "design"): "design_flaw",
        ("physics_check", "ask_user"): "limit_reached / error",
        
        # comparison_check routes
        ("comparison_check", "supervisor"): "approve",
        ("comparison_check", "analyze"): "needs_revision",
        ("comparison_check", "ask_user"): "limit_reached / error",
        
        # supervisor routes
        ("supervisor", "material_checkpoint"): "ok_continue + Stage 0",
        ("supervisor", "select_stage"): "ok_continue + other",
        ("supervisor", "handle_backtrack"): "backtrack_to_stage",
        ("supervisor", "planning"): "replan_needed / replan_with_guidance",
        ("supervisor", "generate_report"): "all_complete / should_stop",
        ("supervisor", "ask_user"): "ask_user / error",
        
        # material_checkpoint routes
        ("material_checkpoint", "select_stage"): "materials validated",
        ("material_checkpoint", "ask_user"): "needs validation",
        
        # ask_user always routes to supervisor
        ("ask_user", "supervisor"): "user response",
    }
    
    # Extract nodes from graph.nodes (dict mapping node_id -> Node)
    nodes = list(graph.nodes.keys())
    
    # Extract edges from graph.edges (list of Edge objects)
    edges = []
    for edge in graph.edges:
        source = edge.source
        target = edge.target
        is_conditional = edge.conditional
        edges.append((source, target, is_conditional))
    
    # Build Mermaid diagram
    lines = [
        "---",
        "config:",
        "  flowchart:",
        "    curve: linear",
        "---",
        "graph TD;",
    ]
    
    # Define node type mappings for color coding
    node_types = {
        # Review nodes - Blue background
        "review": ["plan_review", "design_review", "code_review"],
        # User interaction nodes - Red background
        "user": ["ask_user"],
        # Supervisor node - Green background
        "supervisor": ["supervisor"],
        # Checkpoint nodes - Yellow background
        "checkpoint": ["material_checkpoint"],
        # Execution/validation nodes - Orange background
        "validation": ["execution_check", "physics_check", "comparison_check"],
        # Core workflow nodes - Light purple (default)
        "workflow": ["adapt_prompts", "planning", "select_stage", "design", 
                     "generate_code", "run_code", "analyze", "handle_backtrack", 
                     "generate_report"],
    }
    
    # Build reverse mapping: node -> type
    node_to_type = {}
    for node_type, node_list in node_types.items():
        for node in node_list:
            node_to_type[node] = node_type
    
    # Add node definitions with color classes and LLM indicator
    for node in sorted(nodes):
        if node == "__start__":
            lines.append(f'\t__start__([<p>__start__</p>]):::first')
        elif node == "__end__":
            lines.append(f'\t__end__([<p>__end__</p>]):::last')
        else:
            node_type = node_to_type.get(node, "workflow")
            class_name = f"{node_type}Node"
            # Add 🤖 indicator for nodes that make LLM calls
            if node in llm_nodes:
                display_name = f"🤖 {node}"
            else:
                display_name = node
            lines.append(f'\t{node}["{display_name}"]:::{class_name}')
    
    # Add edges with labels
    for source, target, is_conditional in edges:
        label = edge_labels.get((source, target))
        
        if is_conditional or label:
            # Conditional edge (dashed) with optional label
            if label:
                lines.append(f'\t{source} -.->|{label}| {target};')
            else:
                lines.append(f'\t{source} -.-> {target};')
        else:
            # Direct edge (solid)
            lines.append(f'\t{source} --> {target};')
    
    # Add styling with color classes (improved contrast)
    lines.extend([
        # Default styling for workflow nodes - Light purple with dark text
        '\tclassDef default fill:#e9d5ff,stroke:#7c3aed,stroke-width:2px,color:#1e1b4b',
        '\tclassDef workflowNode fill:#e9d5ff,stroke:#7c3aed,stroke-width:2px,color:#1e1b4b',
        # Review nodes - Dark blue with white text for high contrast
        '\tclassDef reviewNode fill:#1e40af,stroke:#1e3a8a,stroke-width:2px,color:#ffffff',
        # User interaction nodes - Dark red with white text for high contrast
        '\tclassDef userNode fill:#dc2626,stroke:#b91c1c,stroke-width:2px,color:#ffffff',
        # Supervisor node - Dark green with white text for high contrast
        '\tclassDef supervisorNode fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#ffffff',
        # Checkpoint nodes - Dark orange/yellow with white text for high contrast
        '\tclassDef checkpointNode fill:#d97706,stroke:#b45309,stroke-width:2px,color:#ffffff',
        # Validation nodes - Dark orange with white text for high contrast
        '\tclassDef validationNode fill:#ea580c,stroke:#c2410c,stroke-width:2px,color:#ffffff',
        # Start/End nodes
        '\tclassDef first fill-opacity:0,stroke:#666666,stroke-width:3px,color:#000000',
        '\tclassDef last fill:#7c3aed,stroke:#5b21b6,stroke-width:3px,color:#ffffff',
    ])
    
    return '\n'.join(lines)


def main():
    """Generate and display workflow visualizations."""
    print("=" * 80)
    print("LangGraph Workflow Visualization")
    print("=" * 80)
    
    # Create the compiled graph
    print("\n📊 Creating graph...")
    app = create_repro_graph()
    graph = app.get_graph()
    
    # Detect LLM nodes dynamically
    print("\n🔍 Detecting LLM nodes from source code...")
    llm_nodes = detect_llm_nodes(app)
    print(f"   Found {len(llm_nodes)} nodes with LLM calls: {sorted(llm_nodes)}")
    
    # Also identify non-LLM nodes for reporting
    all_nodes = {n for n in graph.nodes.keys() if n not in ("__start__", "__end__")}
    non_llm_nodes = all_nodes - llm_nodes
    print(f"   Found {len(non_llm_nodes)} nodes without LLM calls: {sorted(non_llm_nodes)}")
    
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
    
    # Standard Mermaid diagram (from LangGraph)
    print("\n" + "=" * 80)
    print("Standard Mermaid Diagram (from LangGraph):")
    print("=" * 80 + "\n")
    mermaid_diagram = graph.draw_mermaid()
    print(mermaid_diagram)
    
    # Enhanced Mermaid diagram with labels
    print("\n" + "=" * 80)
    print("Enhanced Mermaid Diagram (with edge labels):")
    print("=" * 80 + "\n")
    labeled_mermaid = create_labeled_mermaid_diagram(app, llm_nodes)
    print(labeled_mermaid)
    
    # Save both versions
    output_file = "workflow_diagram.mmd"
    with open(output_file, "w") as f:
        f.write(mermaid_diagram)
    
    labeled_output_file = "workflow_diagram_labeled.mmd"
    with open(labeled_output_file, "w") as f:
        f.write(labeled_mermaid)
    
    # Build dynamic LLM node documentation
    llm_nodes_list = ", ".join(sorted(llm_nodes))
    non_llm_nodes_list = "\n".join(f"- {node}" for node in sorted(non_llm_nodes))
    
    # Also save to GRAPH.md for easy viewing (using labeled version)
    graph_md_content = f"""# ReproLab Workflow Graph

This document contains the complete Mermaid representation of the ReproLab workflow graph, **automatically generated** from the LangGraph code.

> **Note**: This diagram is generated deterministically from `src/graph.py` and `src/routing.py`. To regenerate, run:
> ```bash
> python visualize_graph.py
> ```

## Complete Graph Structure (with Edge Labels)

```mermaid
{labeled_mermaid}
```

## Node Descriptions

### Core Workflow Nodes

- **adapt_prompts**: Customizes prompts for paper-specific needs
- **planning**: Reads paper and creates staged reproduction plan
- **plan_review**: Reviews reproduction plan before stage execution
- **select_stage**: Selects next stage based on dependencies and validation hierarchy
- **design**: Designs simulation setup for current stage
- **design_review**: Reviews simulation design before code generation
- **generate_code**: Generates Python+Meep simulation code
- **code_review**: Reviews generated code before execution
- **run_code**: Executes the simulation code
- **execution_check**: Validates simulation ran correctly
- **physics_check**: Validates physics (conservation, value ranges)
- **analyze**: Compares results to paper figures
- **comparison_check**: Validates comparison accuracy
- **supervisor**: Makes high-level decisions about workflow progression
- **ask_user**: Pauses workflow to request user input
- **generate_report**: Generates final reproduction report
- **handle_backtrack**: Handles backtracking to previous stages
- **material_checkpoint**: Mandatory checkpoint after Stage 0 (material validation)

## Edge Types

- **Solid arrows (`-->`)**: Direct edges (always taken)
- **Dashed arrows (`-.->`)**: Conditional edges (routed based on state/verdict)
- **Edge labels**: Explain the condition that causes flow to that edge (e.g., "approve", "needs_revision", "pass", "fail")

## Node Color Coding

The diagram uses color-coded nodes to help identify different node types:

- **Blue nodes** (`reviewNode`): Review nodes (plan_review, design_review, code_review)
- **Red nodes** (`userNode`): User interaction nodes (ask_user)
- **Green nodes** (`supervisorNode`): Supervisor node (supervisor)
- **Orange/Yellow nodes** (`checkpointNode`): Checkpoint nodes (material_checkpoint)
- **Orange nodes** (`validationNode`): Validation nodes (execution_check, physics_check, comparison_check)
- **Purple nodes** (`workflowNode`): Core workflow nodes (default color for other nodes)

> **Note**: All colored nodes use high-contrast color schemes (dark backgrounds with white text, or light backgrounds with dark text) to ensure readability and meet accessibility standards.

## LLM Call Indicator

Nodes marked with 🤖 make LLM (Language Model) calls. Nodes without this indicator perform pure logic operations without LLM calls.

> **Note**: LLM nodes are detected automatically by analyzing the source code for calls to `call_agent_with_metrics`.

**Nodes with LLM calls (🤖)**: {llm_nodes_list}

**Nodes without LLM calls**:
{non_llm_nodes_list}

## Key Features

- **Three-tier review system**: Plan, Design, and Code each have dedicated reviewers
- **Material checkpoint**: After Stage 0 completes, `material_checkpoint` node routes to `ask_user` for mandatory user confirmation
- **Backtracking support**: `handle_backtrack` node marks target stage as `needs_rerun` and dependent stages as `invalidated`
- **User interaction**: `ask_user` node uses LangGraph interrupts to pause workflow and request user input
- **Supervisor orchestration**: `supervisor` node makes high-level decisions and routes to appropriate next steps
"""
    
    graph_md_file = "GRAPH.md"
    with open(graph_md_file, "w") as f:
        f.write(graph_md_content)
    
    print("\n" + "=" * 80)
    print(f"✅ Standard Mermaid diagram saved to: {output_file}")
    print(f"✅ Labeled Mermaid diagram saved to: {labeled_output_file}")
    print(f"✅ Markdown documentation saved to: {graph_md_file}")
    print("   View it at: https://mermaid.live/")
    print("   Or view GRAPH.md in your markdown viewer")
    print("=" * 80)


if __name__ == "__main__":
    main()
