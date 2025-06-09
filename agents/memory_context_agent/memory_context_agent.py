"""Memory Context Agent - Manages persistent memory and context synthesis."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

from google.adk import Agent, LlmAgent
from google.adk.tools import FunctionTool
from google.adk.callbacks import StreamingCallback

from .tools import (
    mcp_server_manager_tool,
    context_synthesis_tool,
    memory_persistence_tool,
    vector_similarity_tool,
    supabase_memory_tool
)
from .instructions import MEMORY_CONTEXT_INSTRUCTIONS
from .sub_agents.context_synthesizer_agent import context_synthesizer_agent
# Note: workflow_master_agent import removed to avoid circular dependency

logger = logging.getLogger(__name__)


class MemoryContextCallback(StreamingCallback):
    """Custom callback for memory context operations."""
    
    def __init__(self):
        self.memory_operations = []
        self.context_synthesis_results = []
    
    def on_agent_start(self, agent_name: str, input_data: Dict[str, Any]):
        """Called when agent starts processing."""
        logger.info(f"MemoryContextAgent starting: {agent_name}")
        self.memory_operations.append({
            "timestamp": datetime.now().isoformat(),
            "event": "agent_start",
            "agent": agent_name,
            "input": input_data
        })
    
    def on_tool_start(self, tool_name: str, input_data: Dict[str, Any]):
        """Called when a tool starts execution."""
        logger.info(f"Memory tool starting: {tool_name}")
        self.memory_operations.append({
            "timestamp": datetime.now().isoformat(),
            "event": "tool_start",
            "tool": tool_name,
            "input": input_data
        })
    
    def on_tool_end(self, tool_name: str, output_data: Any):
        """Called when a tool completes execution."""
        logger.info(f"Memory tool completed: {tool_name}")
        self.memory_operations.append({
            "timestamp": datetime.now().isoformat(),
            "event": "tool_end",
            "tool": tool_name,
            "output": str(output_data)[:500]  # Limit output size for logging
        })
    
    def on_agent_end(self, agent_name: str, output_data: Any):
        """Called when agent completes processing."""
        logger.info(f"MemoryContextAgent completed: {agent_name}")
        self.memory_operations.append({
            "timestamp": datetime.now().isoformat(),
            "event": "agent_end",
            "agent": agent_name,
            "output": str(output_data)[:500]
        })


class MemoryContextAgent:
    """
    Advanced memory management and context synthesis agent.
    
    Responsibilities:
    - Persistent memory management via MCP servers
    - Context synthesis from multiple sources
    - Real-time memory state management
    - Vector-based similarity search
    - Cross-session memory continuity
    """
    
    def __init__(self):
        """Initialize the MemoryContextAgent."""
        self.callback = MemoryContextCallback()
        
        # Initialize the core LLM agent with ADK
        self.agent = LlmAgent(
            agent_name="memory_context_agent",
            instructions=MEMORY_CONTEXT_INSTRUCTIONS,
            tools=[
                mcp_server_manager_tool,
                context_synthesis_tool, 
                memory_persistence_tool,
                vector_similarity_tool,
                supabase_memory_tool
            ],
            callbacks=[self.callback]
        )
        
        # Memory state management
        self.active_sessions = {}
        self.memory_cache = {}
        self.context_synthesizer = context_synthesizer_agent
        
        # MCP server connections
        self.mcp_connections = {}
        self.memory_servers = []
        
        logger.info("MemoryContextAgent initialized successfully")
    
    async def store_conversation_memory(self, session_id: str, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store conversation memory with context synthesis.
        
        Args:
            session_id: Unique session identifier
            conversation_data: Conversation data to store
            
        Returns:
            Storage result with memory ID and context synthesis
        """
        try:
            # Prepare memory storage request
            memory_request = {
                "session_id": session_id,
                "conversation_data": conversation_data,
                "timestamp": datetime.now().isoformat(),
                "operation": "store_conversation"
            }
            
            # Execute memory storage using ADK agent
            result = await self.agent.run(
                input_data=f"Store conversation memory: {json.dumps(memory_request)}"
            )
            
            # Process the result
            if result and hasattr(result, 'content'):
                try:
                    parsed_result = json.loads(result.content)
                    return {
                        "status": "success",
                        "memory_id": parsed_result.get("memory_id"),
                        "context_summary": parsed_result.get("context_summary"),
                        "storage_location": parsed_result.get("storage_location"),
                        "synthesis_score": parsed_result.get("synthesis_score", 0.0)
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "memory_id": f"mem_{session_id}_{int(datetime.now().timestamp())}",
                        "content": str(result.content),
                        "raw_result": True
                    }
            
            return {
                "status": "error",
                "error": "No valid result from memory storage",
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error storing conversation memory: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id
            }
    
    async def retrieve_context_memory(self, session_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve and synthesize context memory for a session.
        
        Args:
            session_id: Session identifier
            query: Context query for retrieval
            limit: Maximum number of memories to retrieve
            
        Returns:
            Synthesized context with relevant memories
        """
        try:
            # Prepare context retrieval request
            retrieval_request = {
                "session_id": session_id,
                "query": query,
                "limit": limit,
                "timestamp": datetime.now().isoformat(),
                "operation": "retrieve_context"
            }
            
            # Execute context retrieval using ADK agent
            result = await self.agent.run(
                input_data=f"Retrieve context memory: {json.dumps(retrieval_request)}"
            )
            
            # Process the result
            if result and hasattr(result, 'content'):
                try:
                    parsed_result = json.loads(result.content)
                    return {
                        "status": "success",
                        "context_memories": parsed_result.get("memories", []),
                        "synthesized_context": parsed_result.get("synthesized_context"),
                        "relevance_scores": parsed_result.get("relevance_scores", []),
                        "session_id": session_id,
                        "query": query
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "raw_content": str(result.content),
                        "session_id": session_id,
                        "query": query
                    }
            
            return {
                "status": "error",
                "error": "No memories found",
                "session_id": session_id,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context memory: {e}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
                "query": query
            }
    
    async def manage_mcp_servers(self, operation: str, server_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Manage MCP (Model Context Protocol) servers for persistent memory.
        
        Args:
            operation: Operation type ('start', 'stop', 'status', 'list')
            server_config: Server configuration for start operation
            
        Returns:
            MCP server management result
        """
        try:
            # Prepare MCP management request
            mcp_request = {
                "operation": operation,
                "server_config": server_config,
                "timestamp": datetime.now().isoformat(),
                "current_servers": list(self.mcp_connections.keys())
            }
            
            # Execute MCP management using ADK agent
            result = await self.agent.run(
                input_data=f"Manage MCP servers: {json.dumps(mcp_request)}"
            )
            
            # Process the result
            if result and hasattr(result, 'content'):
                try:
                    parsed_result = json.loads(result.content)
                    
                    # Update local MCP connections state
                    if operation == "start" and parsed_result.get("status") == "success":
                        server_id = parsed_result.get("server_id")
                        if server_id:
                            self.mcp_connections[server_id] = {
                                "config": server_config,
                                "status": "active",
                                "started_at": datetime.now().isoformat()
                            }
                    elif operation == "stop" and server_config and server_config.get("server_id"):
                        server_id = server_config["server_id"]
                        if server_id in self.mcp_connections:
                            del self.mcp_connections[server_id]
                    
                    return {
                        "status": "success",
                        "operation": operation,
                        "result": parsed_result,
                        "active_servers": list(self.mcp_connections.keys())
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "operation": operation,
                        "raw_content": str(result.content)
                    }
            
            return {
                "status": "error",
                "error": "No valid result from MCP management",
                "operation": operation
            }
            
        except Exception as e:
            logger.error(f"Error managing MCP servers: {e}")
            return {
                "status": "error",
                "error": str(e),
                "operation": operation
            }
    
    async def synthesize_cross_session_context(self, user_id: str, sessions: List[str], synthesis_depth: str = "deep") -> Dict[str, Any]:
        """
        Synthesize context across multiple sessions for a user.
        
        Args:
            user_id: User identifier
            sessions: List of session IDs to synthesize
            synthesis_depth: Depth of synthesis ('shallow', 'medium', 'deep')
            
        Returns:
            Cross-session context synthesis result
        """
        try:
            # Prepare cross-session synthesis request
            synthesis_request = {
                "user_id": user_id,
                "sessions": sessions,
                "synthesis_depth": synthesis_depth,
                "timestamp": datetime.now().isoformat(),
                "operation": "cross_session_synthesis"
            }
            
            # Execute synthesis using ADK agent
            result = await self.agent.run(
                input_data=f"Synthesize cross-session context: {json.dumps(synthesis_request)}"
            )
            
            # Process the result
            if result and hasattr(result, 'content'):
                try:
                    parsed_result = json.loads(result.content)
                    return {
                        "status": "success",
                        "user_id": user_id,
                        "synthesized_context": parsed_result.get("synthesized_context"),
                        "session_patterns": parsed_result.get("session_patterns", []),
                        "key_themes": parsed_result.get("key_themes", []),
                        "temporal_insights": parsed_result.get("temporal_insights", {}),
                        "synthesis_depth": synthesis_depth,
                        "sessions_analyzed": sessions
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "user_id": user_id,
                        "raw_synthesis": str(result.content),
                        "sessions_analyzed": sessions
                    }
            
            return {
                "status": "error",
                "error": "No synthesis result generated",
                "user_id": user_id,
                "sessions": sessions
            }
            
        except Exception as e:
            logger.error(f"Error in cross-session synthesis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "user_id": user_id,
                "sessions": sessions
            }
    
    async def get_memory_analytics(self, timeframe: str = "24h") -> Dict[str, Any]:
        """
        Get analytics about memory system performance.
        
        Args:
            timeframe: Analysis timeframe ('1h', '24h', '7d', '30d')
            
        Returns:
            Memory system analytics
        """
        try:
            # Prepare analytics request
            analytics_request = {
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "operation": "memory_analytics",
                "include_mcp_stats": True,
                "include_synthesis_metrics": True
            }
            
            # Execute analytics using ADK agent
            result = await self.agent.run(
                input_data=f"Generate memory analytics: {json.dumps(analytics_request)}"
            )
            
            # Process the result
            if result and hasattr(result, 'content'):
                try:
                    parsed_result = json.loads(result.content)
                    return {
                        "status": "success",
                        "timeframe": timeframe,
                        "analytics": parsed_result,
                        "generated_at": datetime.now().isoformat(),
                        "callback_operations": len(self.callback.memory_operations)
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "timeframe": timeframe,
                        "raw_analytics": str(result.content),
                        "generated_at": datetime.now().isoformat()
                    }
            
            return {
                "status": "error",
                "error": "No analytics generated",
                "timeframe": timeframe
            }
            
        except Exception as e:
            logger.error(f"Error generating memory analytics: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timeframe": timeframe
            }
    
    async def cleanup_expired_memories(self, retention_days: int = 30) -> Dict[str, Any]:
        """
        Clean up expired memories based on retention policy.
        
        Args:
            retention_days: Number of days to retain memories
            
        Returns:
            Cleanup operation result
        """
        try:
            # Prepare cleanup request
            cleanup_request = {
                "retention_days": retention_days,
                "timestamp": datetime.now().isoformat(),
                "operation": "cleanup_expired",
                "dry_run": False
            }
            
            # Execute cleanup using ADK agent
            result = await self.agent.run(
                input_data=f"Cleanup expired memories: {json.dumps(cleanup_request)}"
            )
            
            # Process the result
            if result and hasattr(result, 'content'):
                try:
                    parsed_result = json.loads(result.content)
                    return {
                        "status": "success",
                        "cleanup_result": parsed_result,
                        "retention_days": retention_days,
                        "timestamp": datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    return {
                        "status": "success",
                        "raw_cleanup_result": str(result.content),
                        "retention_days": retention_days
                    }
            
            return {
                "status": "error",
                "error": "No cleanup result",
                "retention_days": retention_days
            }
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return {
                "status": "error",
                "error": str(e),
                "retention_days": retention_days
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_name": "memory_context_agent",
            "status": "active",
            "active_sessions": len(self.active_sessions),
            "memory_cache_size": len(self.memory_cache),
            "mcp_connections": len(self.mcp_connections),
            "callback_operations": len(self.callback.memory_operations),
            "last_updated": datetime.now().isoformat(),
            "tools_available": [
                "mcp_server_manager_tool",
                "context_synthesis_tool",
                "memory_persistence_tool", 
                "vector_similarity_tool",
                "supabase_memory_tool"
            ]
        }


# Create the global memory context agent instance
memory_context_agent = MemoryContextAgent() 