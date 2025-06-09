"""ExecutionCoordinatorAgent - Multi-agent workflow execution with real-time monitoring."""

from datetime import datetime
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from config.settings import settings
from utils.supabase_client import supabase_manager

from .instructions import get_execution_coordinator_instructions, get_task_distributor_instructions
from .tools import (
    langgraph_executor_tool,
    supabase_task_queue_tool,
    realtime_state_tool,
    health_monitoring_tool,
    presence_tracking_tool
)
from .sub_agents.task_distributor_agent import task_distributor_agent


async def setup_execution_coordinator_callback(callback_context: CallbackContext):
    """Setup callback for ExecutionCoordinatorAgent initialization."""
    
    # Initialize coordinator state
    if "coordinator_state" not in callback_context.state:
        callback_context.state["coordinator_state"] = {
            "session_id": f"coordinator_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "active_workflows": {},
            "agent_registry": {},
            "execution_metrics": {
                "workflows_executed": 0,
                "tasks_completed": 0,
                "average_execution_time": 0
            }
        }
    
    # Subscribe to workflow state updates
    def handle_workflow_update(payload):
        """Handle workflow state updates."""
        if payload.get("eventType") == "UPDATE":
            workflow_record = payload.get("new", {})
            workflow_id = workflow_record.get("id")
            if workflow_id:
                callback_context.state["coordinator_state"]["active_workflows"][workflow_id] = workflow_record
    
    supabase_manager.subscribe_to_table_changes(
        "workflow_executions",
        handle_workflow_update,
        "UPDATE"
    )
    
    # Subscribe to agent health updates
    def handle_health_update(payload):
        """Handle agent health updates."""
        if payload.get("eventType") in ["INSERT", "UPDATE"]:
            health_record = payload.get("new", {})
            agent_id = health_record.get("agent_id")
            if agent_id:
                callback_context.state["coordinator_state"]["agent_registry"][agent_id] = health_record
    
    supabase_manager.subscribe_to_table_changes(
        "agent_health",
        handle_health_update,
        "*"
    )
    
    # Subscribe to task queue updates
    def handle_task_update(payload):
        """Handle task queue updates."""
        message = payload.get("payload", {})
        if message.get("type") == "task_available":
            # Log task availability for coordination
            task_id = message.get("task_id")
            if task_id:
                callback_context.state["coordinator_state"].setdefault("pending_tasks", []).append(task_id)
    
    supabase_manager.subscribe_to_broadcast(
        "queue_default",
        handle_task_update
    )
    
    # Update instructions with current state
    current_instructions = get_execution_coordinator_instructions()
    task_distributor_instructions = get_task_distributor_instructions()
    
    callback_context._invocation_context.agent.instruction = (
        current_instructions + "\n\n" + task_distributor_instructions +
        f"\n\nCurrent session: {callback_context.state['coordinator_state']['session_id']}"
    )


# Create the ExecutionCoordinatorAgent
execution_coordinator_agent = Agent(
    model=settings.root_agent_model,
    name="execution_coordinator_agent",
    instruction=get_execution_coordinator_instructions(),
    global_instruction=f"""
You are the ExecutionCoordinatorAgent, the specialized orchestrator for multi-agent workflow execution 
and real-time monitoring in the agentflow system.

Current timestamp: {datetime.now().isoformat()}
Environment: {settings.environment}
Model: {settings.root_agent_model}

Your mission is to ensure flawless workflow execution while maintaining optimal performance, 
complete observability, and robust fault tolerance across the entire multi-agent ecosystem.

Key capabilities:
- LangGraph workflow execution
- Real-time task distribution
- Agent health monitoring
- Load balancing and optimization
- Fault tolerance and recovery
""",
    sub_agents=[task_distributor_agent],
    tools=[
        langgraph_executor_tool,
        supabase_task_queue_tool,
        realtime_state_tool,
        health_monitoring_tool,
        presence_tracking_tool
    ],
    before_agent_callback=setup_execution_coordinator_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=settings.agent_temperature,
        max_output_tokens=settings.max_tokens
    )
) 