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
        logger.error(f"❌ Node error: {error}")


