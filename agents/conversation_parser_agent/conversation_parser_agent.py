"""ConversationParserAgent - Advanced NLP processing with structured workflow generation."""

from datetime import datetime
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from config.settings import settings
from utils.supabase_client import supabase_manager

from .instructions import get_conversation_parser_instructions, get_dependency_analysis_instructions
from .tools import (
    gemini_nlp_tool,
    workflow_graph_tool,
    requirement_validator_tool,
    supabase_context_tool
)
from .sub_agents.dependency_analyzer_agent import dependency_analyzer_agent


async def setup_conversation_parser_callback(callback_context: CallbackContext):
    """Setup callback for ConversationParserAgent initialization."""
    
    # Initialize parsing state
    if "parsing_state" not in callback_context.state:
        callback_context.state["parsing_state"] = {
            "session_id": f"parser_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "parsed_requests": [],
            "workflow_templates": {},
            "user_preferences": {}
        }
    
    # Subscribe to workflow template updates
    def handle_template_update(payload):
        """Handle workflow template updates."""
        if payload.get("eventType") == "INSERT":
            new_template = payload.get("new", {})
            template_id = new_template.get("id")
            callback_context.state["parsing_state"]["workflow_templates"][template_id] = new_template
    
    supabase_manager.subscribe_to_table_changes(
        "workflow_templates",
        handle_template_update,
        "INSERT"
    )
    
    # Update instructions with current capabilities
    current_instructions = get_conversation_parser_instructions()
    dependency_instructions = get_dependency_analysis_instructions()
    
    callback_context._invocation_context.agent.instruction = (
        current_instructions + "\n\n" + dependency_instructions +
        f"\n\nCurrent session: {callback_context.state['parsing_state']['session_id']}"
    )


# Create the ConversationParserAgent
conversation_parser_agent = Agent(
    model=settings.root_agent_model,
    name="conversation_parser_agent",
    instruction=get_conversation_parser_instructions(),
    global_instruction=f"""
You are the ConversationParserAgent, a specialized AI agent focused on advanced Natural Language Processing 
and structured workflow generation for the multi-agent system.

Current timestamp: {datetime.now().isoformat()}
Environment: {settings.environment}
Model: {settings.root_agent_model}

Your mission is to bridge the gap between human natural language and machine-executable workflows,
ensuring complex user intentions are accurately captured and efficiently structured for execution.

Key capabilities:
- Advanced NLP processing using Gemini Pro
- Structured workflow generation with LangGraph
- Dependency analysis and optimization
- Context enrichment and validation
""",
    sub_agents=[dependency_analyzer_agent],
    tools=[
        gemini_nlp_tool,
        workflow_graph_tool,
        requirement_validator_tool,
        supabase_context_tool
    ],
    before_agent_callback=setup_conversation_parser_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=settings.agent_temperature,
        max_output_tokens=settings.max_tokens
    )
) 