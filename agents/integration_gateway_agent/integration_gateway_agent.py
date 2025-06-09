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

"""IntegrationGatewayAgent - External service integration with MCP protocol support."""

from datetime import datetime
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from config.settings import settings
from utils.supabase_client import supabase_manager

from .instructions import get_integration_gateway_instructions, get_compatibility_checker_instructions
from .tools import (
    mcp_server_registry_tool,
    supabase_api_gateway_tool,
    realtime_auth_tool
)
from .sub_agents.compatibility_checker_agent import compatibility_checker_agent


async def setup_integration_gateway_callback(callback_context: CallbackContext):
    """Setup callback for IntegrationGatewayAgent initialization."""
    
    # Initialize gateway state
    if "gateway_state" not in callback_context.state:
        callback_context.state["gateway_state"] = {
            "session_id": f"gateway_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "registered_services": {},
            "active_routes": {},
            "auth_sessions": {}
        }
    
    # Subscribe to MCP server updates
    def handle_mcp_server_update(payload):
        """Handle MCP server registry updates."""
        if payload.get("eventType") in ["INSERT", "UPDATE"]:
            server_record = payload.get("new", {})
            server_id = server_record.get("id")
            if server_id:
                callback_context.state["gateway_state"]["registered_services"][server_id] = server_record
    
    supabase_manager.subscribe_to_table_changes(
        "mcp_servers",
        handle_mcp_server_update,
        "*"
    )
    
    # Subscribe to API route updates
    def handle_route_update(payload):
        """Handle API route updates."""
        if payload.get("eventType") in ["INSERT", "UPDATE"]:
            route_record = payload.get("new", {})
            route_id = route_record.get("id")
            if route_id:
                callback_context.state["gateway_state"]["active_routes"][route_id] = route_record
    
    supabase_manager.subscribe_to_table_changes(
        "api_routes",
        handle_route_update,
        "*"
    )
    
    # Update instructions
    current_instructions = get_integration_gateway_instructions()
    checker_instructions = get_compatibility_checker_instructions()
    
    callback_context._invocation_context.agent.instruction = (
        current_instructions + "\n\n" + checker_instructions +
        f"\n\nCurrent session: {callback_context.state['gateway_state']['session_id']}"
    )


# Create the IntegrationGatewayAgent
integration_gateway_agent = Agent(
    model=settings.root_agent_model,
    name="integration_gateway_agent",
    instruction=get_integration_gateway_instructions(),
    global_instruction=f"""
You are the IntegrationGatewayAgent, the specialized integration orchestrator for external 
service connectivity and MCP protocol support in the agentflow system.

Current timestamp: {datetime.now().isoformat()}
Environment: {settings.environment}
Model: {settings.root_agent_model}

Your mission is to provide seamless, secure, and reliable integration capabilities that enable 
the multi-agent system to leverage external services effectively while maintaining high 
performance and security standards.

Key capabilities:
- MCP server discovery and management
- API gateway with intelligent routing
- Authentication and authorization
- Service compatibility analysis
- Protocol adaptation and transformation
""",
    sub_agents=[compatibility_checker_agent],
    tools=[
        mcp_server_registry_tool,
        supabase_api_gateway_tool,
        realtime_auth_tool
    ],
    before_agent_callback=setup_integration_gateway_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=settings.agent_temperature,
        max_output_tokens=settings.max_tokens
    )
) 