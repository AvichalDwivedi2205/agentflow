"""AgentFactoryAgent - Dynamic ADK agent generation with Google Cloud deployment."""

from datetime import datetime
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from config.settings import settings
from utils.supabase_client import supabase_manager

from .instructions import get_agent_factory_instructions, get_specialization_instructions
from .tools import (
    adk_template_tool,
    gemini_configurator_tool,
    cloud_deployment_tool,
    supabase_registry_tool,
    realtime_deployment_tool
)
from .sub_agents.specialization_agent import specialization_agent


async def setup_agent_factory_callback(callback_context: CallbackContext):
    """Setup callback for AgentFactoryAgent initialization."""
    
    # Initialize factory state
    if "factory_state" not in callback_context.state:
        callback_context.state["factory_state"] = {
            "session_id": f"factory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "active_deployments": {},
            "agent_templates": {},
            "deployment_queue": []
        }
    
    # Subscribe to deployment status updates
    def handle_deployment_update(payload):
        """Handle deployment status updates."""
        if payload.get("eventType") == "UPDATE":
            deployment_record = payload.get("new", {})
            deployment_id = deployment_record.get("id")
            if deployment_id:
                callback_context.state["factory_state"]["active_deployments"][deployment_id] = deployment_record
    
    supabase_manager.subscribe_to_table_changes(
        "agent_deployments",
        handle_deployment_update,
        "UPDATE"
    )
    
    # Subscribe to template updates
    def handle_template_update(payload):
        """Handle template updates."""
        if payload.get("eventType") in ["INSERT", "UPDATE"]:
            template_record = payload.get("new", {})
            template_name = template_record.get("name")
            if template_name:
                callback_context.state["factory_state"]["agent_templates"][template_name] = template_record
    
    supabase_manager.subscribe_to_table_changes(
        "agent_templates",
        handle_template_update,
        "*"
    )
    
    # Subscribe to deployment requests
    def handle_deployment_request(payload):
        """Handle deployment requests via broadcast."""
        message = payload.get("payload", {})
        if message.get("type") == "deployment_request":
            deployment_request = message.get("request", {})
            callback_context.state["factory_state"]["deployment_queue"].append(deployment_request)
    
    supabase_manager.subscribe_to_broadcast(
        "deployment_requests",
        handle_deployment_request
    )
    
    # Update instructions with current capabilities
    current_instructions = get_agent_factory_instructions()
    specialization_instructions = get_specialization_instructions()
    
    callback_context._invocation_context.agent.instruction = (
        current_instructions + "\n\n" + specialization_instructions +
        f"\n\nCurrent session: {callback_context.state['factory_state']['session_id']}"
    )


# Create the AgentFactoryAgent
agent_factory_agent = Agent(
    model=settings.root_agent_model,
    name="agent_factory_agent",
    instruction=get_agent_factory_instructions(),
    global_instruction=f"""
You are the AgentFactoryAgent, the dynamic agent generation and deployment specialist for the multi-agent system.

Current timestamp: {datetime.now().isoformat()}
Environment: {settings.environment}
Model: {settings.root_agent_model}

Your mission is to create, configure, and deploy specialized AI agents optimized for Google Cloud infrastructure
and Gemini models, ensuring production-ready performance and scalability.

Key capabilities:
- Dynamic ADK agent generation
- Google Cloud Run deployment
- Gemini model optimization
- Agent registry management
- Real-time deployment monitoring
""",
    sub_agents=[specialization_agent],
    tools=[
        adk_template_tool,
        gemini_configurator_tool,
        cloud_deployment_tool,
        supabase_registry_tool,
        realtime_deployment_tool
    ],
    before_agent_callback=setup_agent_factory_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=settings.agent_temperature,
        max_output_tokens=settings.max_tokens
    )
) 