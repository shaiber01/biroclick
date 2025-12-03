"""
Metrics logging and discrepancy recording utilities.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional

from schemas.state import ReproState, get_progress_stage


def log_agent_call(agent_name: str, node_name: str, start_time: datetime):
    """
    Decorator to log agent calls to state['metrics'].
    
    Note: This is a simplified version. Ideally this would be a proper decorator,
    but for state-passing functions, we can just call a helper at the end.
    
    Usage:
        start_time = datetime.now(timezone.utc)
        # ... do work ...
        log_agent_call("AgentName", "node_name", start_time)(state, result)
    """
    def record_metric(state: ReproState, result_dict: Dict[str, Any] = None):
        if "metrics" not in state:
            state["metrics"] = {"agent_calls": [], "stage_metrics": []}
            
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Extract verdict with priority order, handling None result_dict
        verdict = None
        if result_dict:
            verdict = (result_dict.get("execution_verdict") or 
                      result_dict.get("physics_verdict") or 
                      result_dict.get("supervisor_verdict") or
                      result_dict.get("last_plan_review_verdict") or
                      None)
        
        metric = {
            "agent": agent_name,
            "node": node_name,
            "stage_id": state.get("current_stage_id"),
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "verdict": verdict,
            "error": result_dict.get("run_error") if result_dict else None
        }
        
        if "agent_calls" not in state["metrics"]:
            state["metrics"]["agent_calls"] = []
            
        state["metrics"]["agent_calls"].append(metric)
        
    return record_metric


def record_discrepancy(
    state: ReproState,
    stage_id: Optional[str],
    figure_id: str,
    quantity: str,
    paper_value: str,
    simulation_value: str,
    classification: str = "investigate",
    difference_percent: float = 100.0,
    likely_cause: str = "",
    action_taken: str = "",
    blocking: bool = True,
) -> Dict[str, Any]:
    """
    Append a discrepancy entry to both global log and the stage progress.
    
    Returns state updates dict (does not mutate state directly).
    
    Args:
        state: Current ReproState
        stage_id: Stage where discrepancy was found
        figure_id: Target figure ID
        quantity: What quantity has the discrepancy
        paper_value: Value from paper
        simulation_value: Value from simulation
        classification: "acceptable", "investigate", or "blocking"
        difference_percent: Percentage difference
        likely_cause: Explanation of likely cause
        action_taken: What action was taken
        blocking: Whether this blocks stage completion
        
    Returns:
        Dict with discrepancy entry and updated log
    """
    log = state.get("discrepancies_log", [])
    entry_id = f"D{len(log) + 1}"
    discrepancy = {
        "id": entry_id,
        "figure": figure_id,
        "quantity": quantity,
        "paper_value": paper_value,
        "simulation_value": simulation_value,
        "difference_percent": difference_percent,
        "classification": classification,
        "likely_cause": likely_cause,
        "action_taken": action_taken,
        "blocking": blocking,
    }
    
    # Return state updates instead of mutating state
    # Note: This function is called from within nodes, so callers need to merge the result
    updated_log = log + [discrepancy]
    
    # For stage-specific discrepancies, update_progress_stage_status handles it
    # But we still need to return the updated log
    result = {
        "discrepancies_log": updated_log,
    }
    
    # Also update progress stage if needed (this mutates progress, but that's handled by update_progress_stage_status)
    if stage_id:
        progress_stage = get_progress_stage(state, stage_id)
        if progress_stage is not None:
            # This will be handled by the caller merging the result
            pass
    
    return {"discrepancy": discrepancy, **result}



