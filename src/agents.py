"""
Agent Node Implementations - Stubs for LangGraph Workflow

Each function is a LangGraph node that:
1. Receives the current ReproState
2. Performs agent-specific processing (TODO: implement with LLM calls)
3. Updates relevant state fields
4. Returns the updated state

These are currently stubs showing expected state mutations.
See prompts/*.md for the corresponding agent system prompts.
"""

from schemas.state import ReproState


def adapt_prompts_node(state: ReproState) -> ReproState:
    """PromptAdaptorAgent: Customize prompts for paper-specific needs."""
    state["workflow_phase"] = "adapting_prompts"
    # TODO: Implement prompt adaptation logic
    # - Analyze paper domain and techniques
    # - Generate prompt modifications
    # - Store in state["prompt_adaptations"]
    return state


def plan_node(state: ReproState) -> ReproState:
    """PlannerAgent: Analyze paper and create reproduction plan."""
    state["workflow_phase"] = "planning"
    # TODO: Implement planning logic
    # - Extract parameters from paper
    # - Classify figures
    # - Design staged reproduction plan
    # - Initialize assumptions
    return state


def select_stage_node(state: ReproState) -> ReproState:
    """Select next stage based on dependencies and validation hierarchy."""
    state["workflow_phase"] = "stage_selection"
    # TODO: Implement stage selection logic
    # - Check for backtracked/invalidated stages
    # - Check validation hierarchy
    # - Select next eligible stage
    return state


def simulation_designer_node(state: ReproState) -> ReproState:
    """SimulationDesignerAgent: Design simulation setup for current stage."""
    state["workflow_phase"] = "design"
    state["review_context"] = "design"  # Tell router this is design review
    # TODO: Implement design logic
    # - Interpret geometry from plan
    # - Select materials
    # - Configure sources, BCs, monitors
    # - Estimate performance
    return state


def code_reviewer_node(state: ReproState) -> ReproState:
    """CodeReviewerAgent: Review design or code before proceeding."""
    state["workflow_phase"] = "review"
    # Note: review_context is set by the CALLER (designer or generator), not here
    # This allows route_after_code_review to know what we're reviewing
    # TODO: Implement review logic
    # - Check design/code against checklists
    # - Identify issues
    # - Set reviewer_verdict
    return state


def code_generator_node(state: ReproState) -> ReproState:
    """CodeGeneratorAgent: Generate Python+Meep code from approved design."""
    state["workflow_phase"] = "code_generation"
    state["review_context"] = "code"  # Tell router this is code review
    # TODO: Implement code generation logic
    # - Convert design to Meep code
    # - Include progress prints
    # - Set expected outputs
    return state


def execution_validator_node(state: ReproState) -> ReproState:
    """ExecutionValidatorAgent: Validate simulation ran correctly."""
    state["workflow_phase"] = "execution_validation"
    # TODO: Implement execution validation logic
    # - Check completion status
    # - Verify output files exist
    # - Check for NaN/Inf in data
    return state


def physics_sanity_node(state: ReproState) -> ReproState:
    """PhysicsSanityAgent: Validate physics of results."""
    state["workflow_phase"] = "physics_validation"
    # TODO: Implement physics validation logic
    # - Check conservation laws (T + R + A ≈ 1)
    # - Verify value ranges
    # - Check numerical quality
    return state


def results_analyzer_node(state: ReproState) -> ReproState:
    """ResultsAnalyzerAgent: Compare results to paper figures."""
    state["workflow_phase"] = "analysis"
    # TODO: Implement analysis logic
    # - Compare simulation outputs to paper figures
    # - Compute discrepancies
    # - Classify reproduction quality
    return state


def comparison_validator_node(state: ReproState) -> ReproState:
    """ComparisonValidatorAgent: Validate comparison accuracy."""
    state["workflow_phase"] = "comparison_validation"
    # TODO: Implement comparison validation logic
    # - Verify math is correct
    # - Check classifications match numbers
    # - Validate discrepancy documentation
    return state


def supervisor_node(state: ReproState) -> ReproState:
    """SupervisorAgent: Big-picture assessment and decisions."""
    state["workflow_phase"] = "supervision"
    # TODO: Implement supervision logic
    # - Assess overall progress
    # - Check validation hierarchy
    # - Decide: continue, replan, ask_user, backtrack
    return state


def ask_user_node(state: ReproState) -> ReproState:
    """Pause for user input."""
    state["workflow_phase"] = "awaiting_user"
    state["awaiting_user_input"] = True
    # TODO: Implement user interaction logic
    # - Present pending questions
    # - Wait for responses
    # - Log interaction
    return state


def generate_report_node(state: ReproState) -> ReproState:
    """Generate final reproduction report."""
    state["workflow_phase"] = "reporting"
    # TODO: Implement report generation logic
    # - Compile figure comparisons
    # - Document assumptions
    # - Generate REPRODUCTION_REPORT.md
    return state


def handle_backtrack_node(state: ReproState) -> ReproState:
    """Process cross-stage backtracking."""
    state["workflow_phase"] = "backtracking"
    # TODO: Implement backtrack logic
    # - Mark stages as invalidated
    # - Reset target stage to needs_rerun
    # - Increment backtrack_count
    return state

