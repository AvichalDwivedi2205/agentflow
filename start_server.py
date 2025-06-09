#!/usr/bin/env python3
"""
Startup script for the Multi-Agent Workflow System with Google ADK.

This script properly initializes all agents and starts the FastAPI server
with comprehensive error handling and logging.
"""

import os
import sys
import asyncio
import logging
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agentflow.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check if all required environment variables are set."""
    required_vars = [
        'GOOGLE_API_KEY',
        'SUPABASE_URL', 
        'SUPABASE_ANON_KEY',
        'SUPABASE_SERVICE_ROLE_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please check your .env file or set these variables directly")
        return False
    
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    try:
        import google.adk
        import supabase
        import fastapi
        import uvicorn
        import langchain
        import langgraph
        logger.info("All core dependencies found")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False


async def test_agent_imports():
    """Test importing all agents to catch any issues early."""
    agents_status = {}
    
    try:
        from agents.workflow_master_agent.workflow_master_agent import workflow_master_agent
        agents_status['workflow_master'] = '✅ OK'
        logger.info("WorkflowMasterAgent imported successfully")
    except Exception as e:
        agents_status['workflow_master'] = f'❌ Error: {e}'
        logger.error(f"Failed to import WorkflowMasterAgent: {e}")
    
    try:
        from agents.conversation_parser_agent.conversation_parser_agent import conversation_parser_agent
        agents_status['conversation_parser'] = '✅ OK'
        logger.info("ConversationParserAgent imported successfully")
    except Exception as e:
        agents_status['conversation_parser'] = f'❌ Error: {e}'
        logger.error(f"Failed to import ConversationParserAgent: {e}")
    
    try:
        from agents.agent_factory_agent.agent_factory_agent import agent_factory_agent
        agents_status['agent_factory'] = '✅ OK'
        logger.info("AgentFactoryAgent imported successfully")
    except Exception as e:
        agents_status['agent_factory'] = f'❌ Error: {e}'
        logger.error(f"Failed to import AgentFactoryAgent: {e}")
    
    try:
        from agents.execution_coordinator_agent.execution_coordinator_agent import execution_coordinator_agent
        agents_status['execution_coordinator'] = '✅ OK'
        logger.info("ExecutionCoordinatorAgent imported successfully")
    except Exception as e:
        agents_status['execution_coordinator'] = f'❌ Error: {e}'
        logger.error(f"Failed to import ExecutionCoordinatorAgent: {e}")
    
    try:
        from agents.memory_context_agent.memory_context_agent import memory_context_agent
        agents_status['memory_context'] = '✅ OK'
        logger.info("MemoryContextAgent imported successfully")
    except Exception as e:
        agents_status['memory_context'] = f'❌ Error: {e}'
        logger.error(f"Failed to import MemoryContextAgent: {e}")
    
    try:
        from agents.integration_gateway_agent.integration_gateway_agent import integration_gateway_agent
        agents_status['integration_gateway'] = '✅ OK'
        logger.info("IntegrationGatewayAgent imported successfully")
    except Exception as e:
        agents_status['integration_gateway'] = f'❌ Error: {e}'
        logger.error(f"Failed to import IntegrationGatewayAgent: {e}")
    
    return agents_status


async def test_supabase_connection():
    """Test connection to Supabase."""
    try:
        from utils.supabase_client import supabase_manager
        await supabase_manager.connect_realtime()
        logger.info("✅ Supabase connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return False


def print_startup_banner():
    """Print a nice startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    🤖 AGENTFLOW SYSTEM 🤖                     ║
║              Multi-Agent Workflow System with ADK             ║
║                                                               ║
║  🎯 WorkflowMasterAgent     - Central orchestration          ║
║  💬 ConversationParserAgent - NLP & workflow generation       ║
║  🏭 AgentFactoryAgent       - Dynamic agent creation          ║
║  ⚡ ExecutionCoordinatorAgent- Multi-agent execution         ║
║  🧠 MemoryContextAgent      - Persistent memory & MCP        ║
║  🌐 IntegrationGatewayAgent - External service integration   ║
║                                                               ║
║  📊 FastAPI Server + Google ADK + Supabase Realtime         ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


async def main():
    """Main startup function."""
    print_startup_banner()
    
    logger.info("🚀 Starting AgentFlow Multi-Agent System...")
    
    # Environment checks
    logger.info("🔍 Checking environment...")
    if not check_environment():
        sys.exit(1)
    
    logger.info("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Test agent imports
    logger.info("🤖 Testing agent imports...")
    agents_status = await test_agent_imports()
    
    print("\n📋 Agent Status Report:")
    for agent_name, status in agents_status.items():
        print(f"  {agent_name}: {status}")
    
    # Test Supabase connection
    logger.info("🔌 Testing Supabase connection...")
    supabase_ok = await test_supabase_connection()
    
    # Count successful agents
    successful_agents = sum(1 for status in agents_status.values() if '✅' in status)
    total_agents = len(agents_status)
    
    print(f"\n📊 System Status:")
    print(f"  Agents: {successful_agents}/{total_agents} successfully loaded")
    print(f"  Supabase: {'✅ Connected' if supabase_ok else '❌ Failed'}")
    
    if successful_agents < total_agents:
        logger.warning(f"⚠️  Some agents failed to load. System will run in limited mode.")
    
    # Start the server
    logger.info("🌐 Starting FastAPI server...")
    print(f"\n🎯 Server starting at: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🔥 WebSocket endpoint: ws://localhost:8000/ws/{{client_id}}")
    print(f"❤️  Health check: http://localhost:8000/health")
    print(f"\n🔄 Use Ctrl+C to stop the server")
    
    # Configure uvicorn
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["agents", "config", "utils"],
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server crashed: {e}")
        sys.exit(1)