"""
LangGraph callback handlers for the biroclick project.

This module provides callback handlers for monitoring and logging
LangGraph execution.
"""

import logging
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class GraphProgressCallback(BaseCallbackHandler):
    """Log graph node transitions for visibility into execution progress.
    
    This callback logs when nodes start executing, helping users
    track the progress of the graph execution.
    
    Usage:
        from src.callbacks import GraphProgressCallback
        
        callback = GraphProgressCallback()
        config = {"callbacks": [callback]}
        result = app.invoke(initial_state, config)
    """
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain/node starts executing."""
        # LangGraph passes node info in different locations depending on version:
        # 1. kwargs["name"] - direct node name
        # 2. kwargs["metadata"]["langgraph_node"] - LangGraph 0.2+ node name
        # 3. kwargs["tags"] - node name may be in tags list
        # 4. serialized["name"] - fallback for LangChain chains
        
        name = kwargs.get("name")
        
        # Check metadata for langgraph_node (LangGraph 0.2+)
        if not name:
            metadata = kwargs.get("metadata", {})
            name = metadata.get("langgraph_node")
        
        # Check tags for node name
        if not name:
            tags = kwargs.get("tags", [])
            # LangGraph node names are often in tags, filter out internal ones
            internal_tags = {"seq:step", "langsmith:hidden", "__start__", "__end__"}
            for tag in tags:
                if tag not in internal_tags and not tag.startswith("seq:step:") and not tag.startswith("graph:step:"):
                    name = tag
                    break
        
        # Fallback to serialized dict
        if not name and serialized:
            name = serialized.get("name")
        
        if not name:
            return  # Skip logging if we can't determine the name
            
        # Filter out internal LangGraph wrappers to show only meaningful nodes
        internal_names = {"RunnableSequence", "StateGraph", "CompiledStateGraph", 
                         "ChannelWrite", "ChannelRead", "RunnableLambda", "LangGraph"}
        if name not in internal_names and not name.startswith("Runnable"):
            logger.info(f"🔄 Entering node: {name}")
    
    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain/node finishes executing."""
        pass  # Could add logging here if you want exit messages
    
    def on_chain_error(self, error, **kwargs):
        """Called when a chain/node errors."""
        # LangGraph interrupts are passed as tuple of Interrupt objects.
        # These are expected control flow for human-in-the-loop, not errors.
        
        # Helper to extract trigger from Interrupt-like object
        def _extract_trigger(obj):
            """Try to extract trigger from an Interrupt object."""
            # DEBUG: Log actual structure to diagnose extraction issues
            logger.info(f"_extract_trigger: type={type(obj).__name__}, obj={obj}")
            logger.info(f"_extract_trigger: attrs={[a for a in dir(obj) if not a.startswith('_')]}")
            if hasattr(obj, 'args'):
                logger.info(f"_extract_trigger: args={obj.args}, args_type={type(obj.args)}")
            if hasattr(obj, 'value'):
                logger.info(f"_extract_trigger: value={obj.value}")
            
            try:
                # LangGraph 1.x: GraphInterrupt stores interrupts in args[0]
                # Structure: GraphInterrupt.args[0][0].value = {"trigger": "...", ...}
                if hasattr(obj, 'args') and obj.args and len(obj.args) > 0:
                    interrupts = obj.args[0]
                    if interrupts and len(interrupts) > 0:
                        first_interrupt = interrupts[0]
                        if hasattr(first_interrupt, 'value') and isinstance(first_interrupt.value, dict):
                            return first_interrupt.value.get('trigger', 'unknown')
                
                # Direct Interrupt object: has .value attribute directly
                if hasattr(obj, 'value'):
                    val = obj.value
                    if isinstance(val, dict):
                        return val.get('trigger', 'unknown')
                
                # Tuple of Interrupt objects (older LangGraph versions)
                if hasattr(obj, '__getitem__'):
                    val = obj[0] if len(obj) > 0 else None
                    if isinstance(val, dict):
                        return val.get('trigger', 'unknown')
            except (TypeError, IndexError, KeyError, AttributeError):
                pass
            return None
        
        # Helper to check if object is an Interrupt
        def _is_interrupt(obj):
            """Check if object is a LangGraph Interrupt."""
            obj_type = type(obj).__name__
            # Check type name
            if 'Interrupt' in obj_type:
                return True
            # Check module
            obj_module = getattr(type(obj), '__module__', '')
            if 'langgraph' in obj_module and 'Interrupt' in obj_type:
                return True
            # Check for Interrupt-like attributes
            if hasattr(obj, 'value') and hasattr(obj, 'id'):
                return True
            return False
        
        # Check 1: Error is a tuple containing Interrupt objects
        if isinstance(error, tuple) and len(error) > 0:
            first = error[0]
            if _is_interrupt(first):
                trigger = _extract_trigger(first) or 'unknown'
                logger.info(f"⏸️  Graph paused for user input (trigger: {trigger})")
                return
        
        # Check 2: Error itself is an Interrupt
        if _is_interrupt(error):
            trigger = _extract_trigger(error) or 'unknown'
            logger.info(f"⏸️  Graph paused for user input (trigger: {trigger})")
            return
        
        # Check 3: String-based fallback for edge cases
        error_str = str(error)
        if "Interrupt(" in error_str or "GraphInterrupt" in error_str:
            # Try to extract trigger from string representation
            # Handle both 'trigger': 'value' and "trigger": "value" formats
            import re
            trigger_match = re.search(r"['\"]trigger['\"]\s*:\s*['\"]([^'\"]+)['\"]", error_str)
            if not trigger_match:
                # Also try without quotes on value (in case it's an enum or similar)
                trigger_match = re.search(r"['\"]trigger['\"]\s*:\s*(\w+)", error_str)
            trigger = trigger_match.group(1) if trigger_match else 'unknown'
            logger.info(f"⏸️  Graph paused for user input (trigger: {trigger})")
            return
        
        # Log actual errors
        logger.error(f"❌ Node error: {error}")


