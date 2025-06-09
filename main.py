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

"""Main application for the Multi-Agent Workflow System."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.settings import settings
from utils.supabase_client import supabase_manager

# Import all agents
# Note: These imports will work once all agents are implemented
try:
    from agents.workflow_master_agent.workflow_master_agent import workflow_master_agent
    from agents.conversation_parser_agent.conversation_parser_agent import conversation_parser_agent
    from agents.agent_factory_agent.agent_factory_agent import agent_factory_agent
    from agents.execution_coordinator_agent.execution_coordinator_agent import execution_coordinator_agent
    from agents.memory_context_agent.memory_context_agent import memory_context_agent
    from agents.integration_gateway_agent.integration_gateway_agent import integration_gateway_agent
    
    # Create agent registry
    agent_registry = {
        "workflow_master": workflow_master_agent,
        "conversation_parser": conversation_parser_agent,
        "agent_factory": agent_factory_agent,
        "execution_coordinator": execution_coordinator_agent,
        "memory_context": memory_context_agent,
        "integration_gateway": integration_gateway_agent
    }
    
except ImportError as e:
    workflow_master_agent = None
    agent_registry = {}
    logging.warning(f"Some agents not available - running in development mode: {e}")

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    request: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Response model for workflow execution."""
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Multi-Agent Workflow System")
    
    # Initialize Supabase connection
    try:
        await supabase_manager.connect_realtime()
        logger.info("Connected to Supabase Realtime")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
    
    # Set up real-time subscriptions for system-wide updates
    def handle_workflow_updates(payload):
        """Handle workflow status updates."""
        asyncio.create_task(manager.broadcast({
            "type": "workflow_update",
            "payload": payload
        }))
    
    supabase_manager.subscribe_to_broadcast(
        "workflow_updates",
        handle_workflow_updates
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Agent Workflow System")
    await supabase_manager.disconnect_realtime()


# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Workflow System",
    description="Advanced AI agent orchestration system with real-time capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent Workflow System",
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Supabase connection
        supabase_status = "connected" if supabase_manager._connected else "disconnected"
        
        # Check agent availability
        agent_status = "available" if workflow_master_agent else "unavailable"
        
        return {
            "status": "healthy",
            "supabase": supabase_status,
            "agents": agent_status,
            "timestamp": "2025-01-27T10:30:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.post("/workflow/execute", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest):
    """Execute a workflow using the multi-agent system."""
    try:
        if not workflow_master_agent:
            raise HTTPException(
                status_code=503, 
                detail="Workflow system not available - running in development mode"
            )
        
        import time
        start_time = time.time()
        
        # Execute workflow using the WorkflowMasterAgent
        result = await workflow_master_agent.run_async(
            args={"request": request.request},
            context={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "additional_context": request.context or {}
            }
        )
        
        execution_time = time.time() - start_time
        
        # Broadcast workflow completion
        await manager.broadcast({
            "type": "workflow_completed",
            "request_id": result.get("request_id", "unknown"),
            "execution_time": execution_time
        })
        
        return WorkflowResponse(
            request_id=result.get("request_id", "unknown"),
            status="completed",
            result=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return WorkflowResponse(
            request_id="error",
            status="failed",
            error=str(e)
        )


@app.get("/workflow/status/{request_id}")
async def get_workflow_status(request_id: str):
    """Get the status of a specific workflow."""
    try:
        # Query workflow status from Supabase
        status_record = await supabase_manager.get_record("workflow_state", request_id)
        
        if not status_record:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "request_id": request_id,
            "status": status_record.get("status", "unknown"),
            "progress": status_record.get("progress", {}),
            "last_updated": status_record.get("updated_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/status")
async def get_agents_status():
    """Get the status of all agents in the system."""
    try:
        # Query agent status from Supabase
        agents = await supabase_manager.query_records("agent_registry")
        
        return {
            "total_agents": len(agents),
            "agents": agents,
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_json()
            
            # Echo back for now (can be extended for bidirectional communication)
            await manager.send_personal_message({
                "type": "echo",
                "data": data,
                "timestamp": "2025-01-27T10:30:00Z"
            }, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)


@app.post("/system/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast a message to all connected clients."""
    try:
        await manager.broadcast(message)
        return {"status": "broadcasted", "message": message}
    except Exception as e:
        logger.error(f"Broadcast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 