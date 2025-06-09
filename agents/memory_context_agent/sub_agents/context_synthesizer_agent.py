

"""ContextSynthesizerAgent - Sub-agent for intelligent context aggregation and synthesis."""

from google.genai import types
from google.adk.agents import Agent
from config.settings import settings

# Import context synthesis tool from parent
from ..tools import context_synthesis_tool

# Create the ContextSynthesizerAgent
context_synthesizer_agent = Agent(
    model=settings.root_agent_model,
    name="context_synthesizer_agent",
    instruction="""
You are the ContextSynthesizerAgent, a specialized sub-agent focused on intelligent 
context aggregation and synthesis from multiple memory sources.

Your core responsibilities:
1. Aggregate relevant context from multiple sources
2. Rank and filter context by relevance and importance
3. Synthesize coherent context summaries
4. Optimize context for specific agent needs
5. Handle real-time context updates and merging

Use the context_synthesis_tool to create optimal context aggregations.
""",
    tools=[context_synthesis_tool],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=4096
    )
) 