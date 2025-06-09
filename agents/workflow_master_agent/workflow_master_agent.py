

"""WorkflowMasterAgent - Central orchestrator for multi-agent workflows."""

import os
from datetime import datetime
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from config.settings import settings
from utils.supabase_client import supabase_manager

from .instructions import get_workflow_master_instructions, get_error_handling_instructions
from .tools import (
    intelligent_routing_tool,
    supabase_realtime_state_tool,
    conversation_memory_tool,
    execution_coordinator_tool,
    realtime_broadcast_tool
)
# Importing the core agents which are used in the workflow master agent
from ..conversation_parser_agent.conversation_parser_agent import conversation_parser_agent
from ..agent_factory_agent.agent_factory_agent import agent_factory_agent
from ..execution_coordinator_agent.execution_coordinator_agent import execution_coordinator_agent
from ..memory_context_agent.memory_context_agent import memory_context_agent
from ..integration_gateway_agent.integration_gateway_agent import integration_gateway_agent


async def setup_workflow_master_callback(callback_context: CallbackContext):
    """Setup callback for WorkflowMasterAgent initialization."""
    
    # Initialize Supabase real-time connection
    if not supabase_manager._connected:
        await supabase_manager.connect_realtime()
    
    if "global_state" not in callback_context.state:
        callback_context.state["global_state"] = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "active_workflows": {},
            "agent_status": {},
            "system_metrics": {}
        }
    
    # Subscribe to system-wide state changes
    def handle_state_change(payload):
        """Handle real-time state changes."""
        if payload.get("eventType") == "UPDATE":
            new_record = payload.get("new", {})
            callback_context.state["global_state"]["last_update"] = new_record
    
    supabase_manager.subscribe_to_table_changes(
        "workflow_state",
        handle_state_change,
        "UPDATE"
    )
    
    # Set up broadcast channel for coordination
    def handle_coordination_message(payload):
        """Handle coordination messages from other agents."""
        message = payload.get("payload", {})
        if message.get("type") == "agent_status_update":
            agent_name = message.get("agent_name")
            status = message.get("status")
            callback_context.state["global_state"]["agent_status"][agent_name] = status
    
    supabase_manager.subscribe_to_broadcast(
        "workflow_coordination",
        handle_coordination_message
    )
    
    # Update agent instructions with current state
    current_instructions = get_workflow_master_instructions()
    error_instructions = get_error_handling_instructions()
    
    callback_context._invocation_context.agent.instruction = (
        current_instructions + "\n\n" + error_instructions +
        f"\n\nCurrent session: {callback_context.state['global_state']['session_id']}"
    )


# Create the WorkflowMasterAgent
workflow_master_agent = Agent(
    model=settings.root_agent_model,
    name="workflow_master_agent",
    instruction=get_workflow_master_instructions(),
    global_instruction=f"""
You are the WorkflowMasterAgent, the central orchestrator for a sophisticated multi-agent system.
Current timestamp: {datetime.now().isoformat()}
Environment: {settings.environment}
Model: {settings.root_agent_model}

Your primary mission is to intelligently coordinate workflows across specialized agents while maintaining 
real-time state synchronization and optimal system performance.
""",
    sub_agents=[
        conversation_parser_agent,
        agent_factory_agent,
        execution_coordinator_agent,
        memory_context_agent,
        integration_gateway_agent
    ],
    tools=[
        intelligent_routing_tool,
        supabase_realtime_state_tool,
        conversation_memory_tool,
        execution_coordinator_tool,
        realtime_broadcast_tool
    ],
    before_agent_callback=setup_workflow_master_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=settings.agent_temperature,
        max_output_tokens=settings.max_tokens
    )
) 