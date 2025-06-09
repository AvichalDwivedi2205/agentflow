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