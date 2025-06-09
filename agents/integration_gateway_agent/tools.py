

"""Tools for the IntegrationGatewayAgent."""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
from utils.supabase_client import supabase_manager
import logging

logger = logging.getLogger(__name__)


async def mcp_server_registry_tool(
    action: str,
    server_config: Dict[str, Any] = None,
    server_id: str = "",
    filters: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Discovery and management of MCP (Model Context Protocol) servers.
    
    Args:
        action: Action to perform (register, discover, update, remove, list)
        server_config: Server configuration for registration
        server_id: Server identifier
        filters: Filters for discovery/listing
        tool_context: ADK tool context
    
    Returns:
        Dict containing MCP server operation result
    """
    try:
        if action == "register":
            if not server_config:
                return {"error": "Server configuration required for registration"}
            
            server_record = {
                "id": server_id or str(uuid.uuid4()),
                "name": server_config.get("name", "unnamed_server"),
                "endpoint": server_config.get("endpoint", ""),
                "protocol_version": server_config.get("protocol_version", "1.0"),
                "capabilities": server_config.get("capabilities", []),
                "authentication": server_config.get("authentication", {}),
                "health_status": "unknown",
                "registered_at": datetime.now().isoformat(),
                "last_health_check": None
            }
            
            await supabase_manager.insert_record("mcp_servers", server_record)
            
            # Perform initial health check
            health_result = await perform_health_check(server_record)
            await supabase_manager.update_record(
                "mcp_servers", 
                server_record["id"], 
                {"health_status": health_result["status"]}
            )
            
            return {
                "action": "register",
                "server_id": server_record["id"],
                "health_status": health_result["status"],
                "success": True
            }
        
        elif action == "discover":
            # Auto-discover MCP servers
            discovery_results = await auto_discover_mcp_servers()
            
            # Register discovered servers
            registered_servers = []
            for server in discovery_results:
                if not await server_exists(server["endpoint"]):
                    server["id"] = str(uuid.uuid4())
                    server["registered_at"] = datetime.now().isoformat()
                    await supabase_manager.insert_record("mcp_servers", server)
                    registered_servers.append(server["id"])
            
            return {
                "action": "discover",
                "servers_found": len(discovery_results),
                "servers_registered": len(registered_servers),
                "registered_ids": registered_servers
            }
        
        elif action == "list":
            # List registered servers
            servers = await supabase_manager.query_records("mcp_servers", filters=filters)
            
            return {
                "action": "list",
                "servers": servers,
                "count": len(servers)
            }
        
        elif action == "health_check":
            # Perform health check on servers
            if server_id:
                servers = [await supabase_manager.get_record("mcp_servers", server_id)]
            else:
                servers = await supabase_manager.query_records("mcp_servers")
            
            health_results = []
            for server in servers:
                if server:
                    health_result = await perform_health_check(server)
                    await supabase_manager.update_record(
                        "mcp_servers",
                        server["id"],
                        {
                            "health_status": health_result["status"],
                            "last_health_check": datetime.now().isoformat()
                        }
                    )
                    health_results.append({
                        "server_id": server["id"],
                        "status": health_result["status"]
                    })
            
            return {
                "action": "health_check",
                "results": health_results,
                "total_checked": len(health_results)
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in MCP server registry tool: {e}")
        return {"error": str(e), "action": action}


async def auto_discover_mcp_servers() -> List[Dict[str, Any]]:
    """Auto-discover MCP servers using common discovery methods."""
    
    discovered_servers = []
    
    # Common MCP server ports and endpoints
    common_endpoints = [
        "http://localhost:8080/mcp",
        "http://localhost:3000/mcp",
        "ws://localhost:8080/mcp",
        "ws://localhost:3000/mcp"
    ]
    
    for endpoint in common_endpoints:
        try:
            # Simulate discovery (in production, make actual HTTP/WebSocket requests)
            discovered_servers.append({
                "name": f"mcp_server_{endpoint.split(':')[-1].split('/')[0]}",
                "endpoint": endpoint,
                "protocol_version": "1.0",
                "capabilities": ["read", "write", "subscribe"],
                "health_status": "healthy"
            })
        except Exception:
            continue
    
    return discovered_servers


async def server_exists(endpoint: str) -> bool:
    """Check if a server with the given endpoint already exists."""
    
    existing = await supabase_manager.query_records(
        "mcp_servers",
        filters={"endpoint": endpoint}
    )
    return len(existing) > 0


async def perform_health_check(server: Dict[str, Any]) -> Dict[str, Any]:
    """Perform health check on MCP server."""
    
    try:
        # Simulate health check (in production, make actual request)
        endpoint = server["endpoint"]
        
        # Mock health check logic
        if "localhost" in endpoint:
            return {"status": "healthy", "response_time": 50}
        else:
            return {"status": "unknown", "response_time": None}
            
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def supabase_api_gateway_tool(
    action: str,
    route_config: Dict[str, Any] = None,
    request_data: Dict[str, Any] = None,
    policies: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Centralized API gateway with intelligent routing and policy enforcement.
    
    Args:
        action: Action to perform (route, configure, policy, metrics)
        route_config: Route configuration
        request_data: Request data for routing
        policies: Policy configuration
        tool_context: ADK tool context
    
    Returns:
        Dict containing API gateway operation result
    """
    try:
        if action == "route":
            # Route API request
            if not request_data:
                return {"error": "Request data required for routing"}
            
            # Find appropriate route
            route = await find_best_route(request_data)
            
            if not route:
                return {"error": "No suitable route found"}
            
            # Apply policies
            policy_result = await apply_policies(request_data, route)
            
            if not policy_result["allowed"]:
                return {
                    "action": "route",
                    "status": "denied",
                    "reason": policy_result["reason"]
                }
            
            # Execute request
            response = await execute_api_request(request_data, route)
            
            # Log request
            await log_api_request(request_data, route, response)
            
            return {
                "action": "route",
                "route_id": route["id"],
                "status": "success",
                "response": response
            }
        
        elif action == "configure":
            # Configure route
            if not route_config:
                return {"error": "Route configuration required"}
            
            route_record = {
                "id": str(uuid.uuid4()),
                "path_pattern": route_config.get("path_pattern", "/"),
                "target_service": route_config.get("target_service", ""),
                "method": route_config.get("method", "GET"),
                "policies": route_config.get("policies", []),
                "rate_limit": route_config.get("rate_limit", {}),
                "timeout": route_config.get("timeout", 30),
                "created_at": datetime.now().isoformat()
            }
            
            await supabase_manager.insert_record("api_routes", route_record)
            
            return {
                "action": "configure",
                "route_id": route_record["id"],
                "success": True
            }
        
        elif action == "metrics":
            # Get API metrics
            metrics = await get_api_metrics()
            
            return {
                "action": "metrics",
                "metrics": metrics
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in API gateway tool: {e}")
        return {"error": str(e), "action": action}


async def find_best_route(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find the best route for the given request."""
    
    path = request_data.get("path", "/")
    method = request_data.get("method", "GET")
    
    routes = await supabase_manager.query_records("api_routes")
    
    # Simple pattern matching (in production, use more sophisticated routing)
    for route in routes:
        if (route["method"] == method and 
            path.startswith(route["path_pattern"].rstrip("*"))):
            return route
    
    return None


async def apply_policies(request_data: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, bool]:
    """Apply policies to the request."""
    
    policies = route.get("policies", [])
    
    for policy in policies:
        if policy == "rate_limit":
            # Check rate limit
            if not await check_rate_limit(request_data, route):
                return {"allowed": False, "reason": "Rate limit exceeded"}
        
        elif policy == "auth_required":
            # Check authentication
            if not request_data.get("authorization"):
                return {"allowed": False, "reason": "Authentication required"}
    
    return {"allowed": True, "reason": "Policies passed"}


async def check_rate_limit(request_data: Dict[str, Any], route: Dict[str, Any]) -> bool:
    """Check if request is within rate limits."""
    
    # Simple rate limiting logic
    client_id = request_data.get("client_id", "anonymous")
    rate_limit = route.get("rate_limit", {})
    
    if not rate_limit:
        return True
    
    # Check request count in time window
    window_minutes = rate_limit.get("window_minutes", 60)
    max_requests = rate_limit.get("max_requests", 1000)
    
    # In production, use Redis or similar for distributed rate limiting
    return True  # Simplified for demo


async def execute_api_request(request_data: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the API request to the target service."""
    
    # Simulate API request execution
    target_service = route["target_service"]
    
    return {
        "status": "success",
        "target_service": target_service,
        "response_time": 150,
        "data": {"message": f"Response from {target_service}"}
    }


async def log_api_request(request_data: Dict[str, Any], route: Dict[str, Any], response: Dict[str, Any]):
    """Log API request for monitoring and analytics."""
    
    log_record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "route_id": route["id"],
        "path": request_data.get("path"),
        "method": request_data.get("method"),
        "client_id": request_data.get("client_id"),
        "response_status": response.get("status"),
        "response_time": response.get("response_time"),
        "target_service": route["target_service"]
    }
    
    await supabase_manager.insert_record("api_logs", log_record)


async def get_api_metrics() -> Dict[str, Any]:
    """Get API gateway metrics."""
    
    # Get recent logs for metrics calculation
    recent_logs = await supabase_manager.query_records("api_logs", limit=1000)
    
    if not recent_logs:
        return {"total_requests": 0}
    
    # Calculate basic metrics
    total_requests = len(recent_logs)
    successful_requests = len([log for log in recent_logs if log.get("response_status") == "success"])
    
    # Calculate average response time
    response_times = [log.get("response_time", 0) for log in recent_logs if log.get("response_time")]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    return {
        "total_requests": total_requests,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        "average_response_time": avg_response_time,
        "error_rate": (total_requests - successful_requests) / total_requests if total_requests > 0 else 0
    }


async def realtime_auth_tool(
    action: str,
    auth_request: Dict[str, Any] = None,
    token_data: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Real-time authentication and authorization management.
    
    Args:
        action: Action to perform (authenticate, authorize, validate, revoke)
        auth_request: Authentication request data
        token_data: Token data for validation
        tool_context: ADK tool context
    
    Returns:
        Dict containing auth operation result
    """
    try:
        if action == "authenticate":
            # Authenticate user/service
            if not auth_request:
                return {"error": "Authentication request required"}
            
            # Validate credentials
            auth_result = await validate_credentials(auth_request)
            
            if auth_result["valid"]:
                # Generate access token
                token = await generate_access_token(auth_result["principal"])
                
                return {
                    "action": "authenticate",
                    "success": True,
                    "token": token,
                    "principal": auth_result["principal"]
                }
            else:
                return {
                    "action": "authenticate",
                    "success": False,
                    "error": "Invalid credentials"
                }
        
        elif action == "validate":
            # Validate token
            if not token_data:
                return {"error": "Token data required for validation"}
            
            validation_result = await validate_token(token_data.get("token", ""))
            
            return {
                "action": "validate",
                "valid": validation_result["valid"],
                "principal": validation_result.get("principal"),
                "expires_at": validation_result.get("expires_at")
            }
        
        elif action == "authorize":
            # Check authorization
            if not auth_request:
                return {"error": "Authorization request required"}
            
            principal = auth_request.get("principal", "")
            resource = auth_request.get("resource", "")
            action_requested = auth_request.get("action", "")
            
            authorized = await check_authorization(principal, resource, action_requested)
            
            return {
                "action": "authorize",
                "authorized": authorized,
                "principal": principal,
                "resource": resource
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in auth tool: {e}")
        return {"error": str(e), "action": action}


async def validate_credentials(auth_request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user credentials."""
    
    auth_type = auth_request.get("type", "api_key")
    
    if auth_type == "api_key":
        api_key = auth_request.get("api_key", "")
        # Simulate API key validation
        if api_key.startswith("ak_"):
            return {"valid": True, "principal": f"api_key:{api_key[:10]}"}
    
    elif auth_type == "bearer_token":
        token = auth_request.get("token", "")
        # Simulate token validation
        if len(token) > 20:
            return {"valid": True, "principal": f"user:token_user"}
    
    return {"valid": False}


async def generate_access_token(principal: str) -> str:
    """Generate access token for authenticated principal."""
    
    # Simple token generation (in production, use JWT or similar)
    token = f"at_{uuid.uuid4().hex}"
    
    # Store token with expiration
    token_record = {
        "token": token,
        "principal": principal,
        "issued_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
    }
    
    await supabase_manager.insert_record("auth_tokens", token_record)
    
    return token


async def validate_token(token: str) -> Dict[str, Any]:
    """Validate access token."""
    
    tokens = await supabase_manager.query_records(
        "auth_tokens",
        filters={"token": token}
    )
    
    if not tokens:
        return {"valid": False, "reason": "Token not found"}
    
    token_record = tokens[0]
    expires_at = datetime.fromisoformat(token_record["expires_at"])
    
    if datetime.now() > expires_at:
        return {"valid": False, "reason": "Token expired"}
    
    return {
        "valid": True,
        "principal": token_record["principal"],
        "expires_at": token_record["expires_at"]
    }


async def check_authorization(principal: str, resource: str, action: str) -> bool:
    """Check if principal is authorized for action on resource."""
    
    # Simple authorization logic (in production, use RBAC/ABAC)
    
    # Admin principals have access to everything
    if "admin" in principal:
        return True
    
    # API keys have limited access
    if principal.startswith("api_key:"):
        return action in ["read", "list"]
    
    # Default deny
    return False 