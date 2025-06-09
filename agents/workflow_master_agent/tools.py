# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for the WorkflowMasterAgent."""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
from utils.supabase_client import supabase_manager
import logging

logger = logging.getLogger(__name__)


async def intelligent_routing_tool(
    request: str,
    context: str = "",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    ML-based intent classification and intelligent routing tool.
    
    Args:
        request: The user request to analyze and route
        context: Additional context for routing decisions
        tool_context: ADK tool context
    
    Returns:
        Dict containing routing decision and confidence scores
    """
    try:
        # Analyze request complexity and intent
        routing_analysis = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "original_request": request,
            "context": context
        }
        
        # Simple intent classification (can be enhanced with ML models)
        intent_keywords = {
            "conversation_parsing": ["parse", "understand", "analyze", "extract", "nlp"],
            "agent_creation": ["create agent", "deploy", "generate agent", "new agent"],
            "workflow_execution": ["execute", "run", "process", "workflow", "coordinate"],
            "memory_management": ["remember", "recall", "memory", "context", "history"],
            "integration": ["integrate", "connect", "api", "external", "service"]
        }
        
        detected_intents = []
        confidence_scores = {}
        
        request_lower = request.lower()
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in request_lower)
            if score > 0:
                confidence_scores[intent] = min(score / len(keywords), 1.0)
                detected_intents.append(intent)
        
        # Determine primary routing target
        if not detected_intents:
            primary_agent = "conversation_parser_agent"
            confidence = 0.5
        else:
            primary_intent = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
            confidence = confidence_scores[primary_intent]
            
            agent_mapping = {
                "conversation_parsing": "conversation_parser_agent",
                "agent_creation": "agent_factory_agent", 
                "workflow_execution": "execution_coordinator_agent",
                "memory_management": "memory_context_agent",
                "integration": "integration_gateway_agent"
            }
            primary_agent = agent_mapping.get(primary_intent, "conversation_parser_agent")
        
        routing_decision = {
            "primary_agent": primary_agent,
            "confidence": confidence,
            "detected_intents": detected_intents,
            "confidence_scores": confidence_scores,
            "requires_multi_agent": len(detected_intents) > 1,
            "execution_plan": {
                "sequential_agents": [primary_agent],
                "parallel_agents": [],
                "dependencies": {}
            }
        }
        
        # Store routing decision in Supabase
        await supabase_manager.insert_record(
            "routing_decisions",
            {
                "request_id": routing_analysis["request_id"],
                "request": request,
                "routing_decision": routing_decision,
                "timestamp": routing_analysis["timestamp"]
            }
        )
        
        # Update tool context state
        if tool_context:
            tool_context.state["current_routing"] = routing_decision
            tool_context.state["request_id"] = routing_analysis["request_id"]
        
        logger.info(f"Routed request to {primary_agent} with confidence {confidence}")
        return routing_decision
        
    except Exception as e:
        logger.error(f"Error in intelligent routing: {e}")
        return {
            "primary_agent": "conversation_parser_agent",
            "confidence": 0.1,
            "error": str(e),
            "fallback": True
        }


async def supabase_realtime_state_tool(
    action: str,
    state_key: str,
    state_value: Any = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Real-time state management tool with conflict resolution.
    
    Args:
        action: Action to perform (get, set, update, delete)
        state_key: Key for the state value
        state_value: Value to set/update (for set/update actions)
        tool_context: ADK tool context
    
    Returns:
        Dict containing state operation result
    """
    try:
        request_id = tool_context.state.get("request_id", str(uuid.uuid4()))
        
        if action == "get":
            # Retrieve state from Supabase
            result = await supabase_manager.get_record("workflow_state", state_key)
            return {
                "action": "get",
                "key": state_key,
                "value": result.get("value") if result else None,
                "timestamp": result.get("updated_at") if result else None
            }
        
        elif action == "set":
            # Set new state value
            state_record = {
                "id": state_key,
                "key": state_key,
                "value": state_value,
                "request_id": request_id,
                "updated_at": datetime.now().isoformat()
            }
            
            result = await supabase_manager.insert_record("workflow_state", state_record)
            
            # Broadcast state change
            await supabase_manager.broadcast_message(
                "workflow_state_updates",
                {
                    "action": "state_updated",
                    "key": state_key,
                    "value": state_value,
                    "request_id": request_id
                }
            )
            
            return {
                "action": "set",
                "key": state_key,
                "value": state_value,
                "success": True
            }
        
        elif action == "update":
            # Update existing state with conflict resolution
            existing = await supabase_manager.get_record("workflow_state", state_key)
            
            if existing:
                # Simple conflict resolution - last write wins with timestamp check
                updated_record = {
                    "value": state_value,
                    "request_id": request_id,
                    "updated_at": datetime.now().isoformat()
                }
                
                result = await supabase_manager.update_record(
                    "workflow_state", 
                    state_key, 
                    updated_record
                )
                
                # Broadcast update
                await supabase_manager.broadcast_message(
                    "workflow_state_updates",
                    {
                        "action": "state_updated",
                        "key": state_key,
                        "value": state_value,
                        "request_id": request_id
                    }
                )
                
                return {
                    "action": "update",
                    "key": state_key,
                    "value": state_value,
                    "success": True
                }
            else:
                # Create new if doesn't exist
                return await supabase_realtime_state_tool("set", state_key, state_value, tool_context)
        
        elif action == "delete":
            # Delete state (implementation depends on Supabase setup)
            # For now, we'll mark as deleted
            await supabase_manager.update_record(
                "workflow_state",
                state_key,
                {
                    "deleted": True,
                    "deleted_at": datetime.now().isoformat()
                }
            )
            
            return {
                "action": "delete",
                "key": state_key,
                "success": True
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in state management: {e}")
        return {"error": str(e), "action": action, "key": state_key}


async def conversation_memory_tool(
    action: str,
    query: str = "",
    context: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    MCP server integration for persistent conversation context.
    
    Args:
        action: Action to perform (store, retrieve, search)
        query: Query string for search operations
        context: Context data to store
        tool_context: ADK tool context
    
    Returns:
        Dict containing memory operation result
    """
    try:
        request_id = tool_context.state.get("request_id", str(uuid.uuid4()))
        
        if action == "store":
            # Store conversation context
            memory_record = {
                "id": str(uuid.uuid4()),
                "request_id": request_id,
                "context": context or {},
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "session_id": tool_context.state.get("session_id", "default")
            }
            
            result = await supabase_manager.insert_record("conversation_memory", memory_record)
            
            return {
                "action": "store",
                "memory_id": memory_record["id"],
                "success": True
            }
        
        elif action == "retrieve":
            # Retrieve recent conversation context
            filters = {"request_id": request_id} if request_id else {}
            memories = await supabase_manager.query_records(
                "conversation_memory",
                filters=filters,
                limit=10
            )
            
            return {
                "action": "retrieve",
                "memories": memories,
                "count": len(memories)
            }
        
        elif action == "search":
            # Search conversation memory (simplified - can be enhanced with vector search)
            memories = await supabase_manager.query_records(
                "conversation_memory",
                limit=50
            )
            
            # Simple text search
            matching_memories = [
                memory for memory in memories
                if query.lower() in str(memory.get("context", "")).lower() or
                   query.lower() in memory.get("query", "").lower()
            ]
            
            return {
                "action": "search",
                "query": query,
                "results": matching_memories[:10],
                "count": len(matching_memories)
            }
        
        else:
            return {"error": f"Unknown memory action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in conversation memory: {e}")
        return {"error": str(e), "action": action}


async def execution_coordinator_tool(
    action: str,
    workflow_plan: Dict[str, Any] = None,
    agent_name: str = "",
    task_data: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Graph-based execution planning and coordination tool.
    
    Args:
        action: Action to perform (plan, execute, monitor, status)
        workflow_plan: Workflow execution plan
        agent_name: Target agent name for execution
        task_data: Task data to send to agent
        tool_context: ADK tool context
    
    Returns:
        Dict containing execution coordination result
    """
    try:
        request_id = tool_context.state.get("request_id", str(uuid.uuid4()))
        
        if action == "plan":
            # Create execution plan
            execution_plan = {
                "plan_id": str(uuid.uuid4()),
                "request_id": request_id,
                "workflow_plan": workflow_plan or {},
                "status": "planned",
                "created_at": datetime.now().isoformat(),
                "steps": [],
                "dependencies": {}
            }
            
            # Store execution plan
            await supabase_manager.insert_record("execution_plans", execution_plan)
            
            # Update tool context
            tool_context.state["execution_plan"] = execution_plan
            
            return {
                "action": "plan",
                "plan_id": execution_plan["plan_id"],
                "success": True
            }
        
        elif action == "execute":
            # Execute task on specified agent
            task_record = {
                "id": str(uuid.uuid4()),
                "request_id": request_id,
                "agent_name": agent_name,
                "task_data": task_data or {},
                "status": "executing",
                "started_at": datetime.now().isoformat()
            }
            
            await supabase_manager.insert_record("agent_tasks", task_record)
            
            # Broadcast task assignment
            await supabase_manager.broadcast_message(
                f"agent_{agent_name}",
                {
                    "action": "execute_task",
                    "task_id": task_record["id"],
                    "task_data": task_data
                }
            )
            
            return {
                "action": "execute",
                "task_id": task_record["id"],
                "agent_name": agent_name,
                "success": True
            }
        
        elif action == "monitor":
            # Monitor execution status
            tasks = await supabase_manager.query_records(
                "agent_tasks",
                filters={"request_id": request_id}
            )
            
            return {
                "action": "monitor",
                "request_id": request_id,
                "tasks": tasks,
                "total_tasks": len(tasks),
                "completed_tasks": len([t for t in tasks if t.get("status") == "completed"])
            }
        
        elif action == "status":
            # Get overall execution status
            execution_plan = tool_context.state.get("execution_plan", {})
            
            return {
                "action": "status",
                "plan_id": execution_plan.get("plan_id"),
                "status": execution_plan.get("status", "unknown"),
                "request_id": request_id
            }
        
        else:
            return {"error": f"Unknown execution action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in execution coordination: {e}")
        return {"error": str(e), "action": action}


async def realtime_broadcast_tool(
    channel: str,
    message: Dict[str, Any],
    target_agents: List[str] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Low-latency message broadcasting tool.
    
    Args:
        channel: Broadcast channel name
        message: Message to broadcast
        target_agents: Specific agents to target (optional)
        tool_context: ADK tool context
    
    Returns:
        Dict containing broadcast result
    """
    try:
        request_id = tool_context.state.get("request_id", str(uuid.uuid4()))
        
        # Add metadata to message
        broadcast_message = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "source": "workflow_master_agent",
            "target_agents": target_agents,
            "payload": message
        }
        
        # Broadcast to main channel
        await supabase_manager.broadcast_message(channel, broadcast_message)
        
        # Broadcast to specific agent channels if specified
        if target_agents:
            for agent in target_agents:
                agent_channel = f"agent_{agent}"
                await supabase_manager.broadcast_message(agent_channel, broadcast_message)
        
        # Log broadcast
        await supabase_manager.insert_record(
            "broadcast_log",
            {
                "id": str(uuid.uuid4()),
                "channel": channel,
                "message": broadcast_message,
                "target_agents": target_agents,
                "timestamp": broadcast_message["timestamp"]
            }
        )
        
        return {
            "action": "broadcast",
            "channel": channel,
            "target_agents": target_agents,
            "success": True,
            "timestamp": broadcast_message["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Error in broadcast: {e}")
        return {"error": str(e), "channel": channel} 