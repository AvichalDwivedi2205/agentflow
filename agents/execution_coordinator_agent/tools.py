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

"""Tools for the ExecutionCoordinatorAgent."""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from google.adk.tools import ToolContext
from utils.supabase_client import supabase_manager
import logging

logger = logging.getLogger(__name__)


async def langgraph_executor_tool(
    workflow_graph: Dict[str, Any],
    execution_config: Dict[str, Any] = None,
    checkpoint_interval: int = 30,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Execute workflow graphs with state management and checkpointing.
    
    Args:
        workflow_graph: LangGraph workflow to execute
        execution_config: Configuration for execution
        checkpoint_interval: Interval for state checkpointing (seconds)
        tool_context: ADK tool context
    
    Returns:
        Dict containing execution result
    """
    try:
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        if not execution_config:
            execution_config = {
                "timeout": 300,  # 5 minutes
                "retry_count": 3,
                "parallel_execution": True,
                "state_persistence": True
            }
        
        # Initialize execution state
        execution_state = {
            "execution_id": execution_id,
            "workflow_id": workflow_graph.get("workflow_id"),
            "status": "initializing",
            "start_time": start_time.isoformat(),
            "nodes": workflow_graph.get("nodes", []),
            "edges": workflow_graph.get("edges", []),
            "completed_nodes": [],
            "failed_nodes": [],
            "current_nodes": [],
            "node_results": {},
            "checkpoints": []
        }
        
        # Store initial execution state
        await supabase_manager.insert_record(
            "workflow_executions",
            {
                "id": execution_id,
                "execution_state": execution_state,
                "created_at": start_time.isoformat()
            }
        )
        
        # Execute workflow
        execution_result = await execute_workflow_graph(
            workflow_graph, 
            execution_state, 
            execution_config,
            checkpoint_interval
        )
        
        # Calculate execution metrics
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        final_result = {
            "execution_id": execution_id,
            "workflow_id": workflow_graph.get("workflow_id"),
            "status": execution_result["status"],
            "execution_time": execution_time,
            "completed_nodes": execution_result["completed_nodes"],
            "failed_nodes": execution_result["failed_nodes"],
            "node_results": execution_result["node_results"],
            "metrics": {
                "total_nodes": len(workflow_graph.get("nodes", [])),
                "success_rate": len(execution_result["completed_nodes"]) / max(len(workflow_graph.get("nodes", [])), 1),
                "average_node_time": execution_time / max(len(execution_result["completed_nodes"]), 1),
                "checkpoints_created": len(execution_result.get("checkpoints", []))
            },
            "end_time": end_time.isoformat()
        }
        
        # Update execution record
        await supabase_manager.update_record(
            "workflow_executions",
            execution_id,
            {
                "execution_state": final_result,
                "status": execution_result["status"],
                "completed_at": end_time.isoformat()
            }
        )
        
        # Broadcast completion
        await supabase_manager.broadcast_message(
            "workflow_executions",
            {
                "type": "execution_completed",
                "execution_id": execution_id,
                "status": execution_result["status"],
                "execution_time": execution_time
            }
        )
        
        # Update tool context
        if tool_context:
            tool_context.state["execution_result"] = final_result
            tool_context.state["execution_id"] = execution_id
        
        logger.info(f"Completed workflow execution {execution_id} in {execution_time:.2f}s")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        return {"error": str(e), "execution_id": execution_id}


async def execute_workflow_graph(
    workflow_graph: Dict[str, Any],
    execution_state: Dict[str, Any],
    execution_config: Dict[str, Any],
    checkpoint_interval: int
) -> Dict[str, Any]:
    """Execute the workflow graph with proper state management."""
    
    nodes = workflow_graph.get("nodes", [])
    edges = workflow_graph.get("edges", [])
    
    # Build dependency graph
    dependencies = build_dependency_graph(nodes, edges)
    
    # Find entry points (nodes with no dependencies)
    entry_points = find_entry_points(dependencies)
    
    # Initialize execution queues
    ready_queue = list(entry_points)
    executing_nodes = set()
    completed_nodes = set()
    failed_nodes = set()
    node_results = {}
    
    execution_state["status"] = "executing"
    execution_state["current_nodes"] = ready_queue.copy()
    
    last_checkpoint = datetime.now()
    
    try:
        while ready_queue or executing_nodes:
            # Execute ready nodes
            if ready_queue:
                current_batch = []
                batch_size = min(len(ready_queue), execution_config.get("max_parallel", 5))
                
                for _ in range(batch_size):
                    if ready_queue:
                        node_id = ready_queue.pop(0)
                        current_batch.append(node_id)
                        executing_nodes.add(node_id)
                
                # Execute batch of nodes
                batch_results = await execute_node_batch(current_batch, nodes, node_results, execution_config)
                
                # Process results
                for node_id, result in batch_results.items():
                    executing_nodes.discard(node_id)
                    
                    if result.get("success", False):
                        completed_nodes.add(node_id)
                        node_results[node_id] = result
                        
                        # Find newly ready nodes
                        newly_ready = find_newly_ready_nodes(
                            node_id, dependencies, completed_nodes, failed_nodes
                        )
                        ready_queue.extend(newly_ready)
                    else:
                        failed_nodes.add(node_id)
                        node_results[node_id] = result
                        
                        # Handle failure
                        if not execution_config.get("continue_on_failure", True):
                            break
            
            # Create checkpoint if needed
            if (datetime.now() - last_checkpoint).seconds >= checkpoint_interval:
                await create_execution_checkpoint(execution_state, completed_nodes, failed_nodes, node_results)
                last_checkpoint = datetime.now()
            
            # Small delay to prevent tight loops
            if not ready_queue and executing_nodes:
                await asyncio.sleep(0.1)
        
        # Determine final status
        if failed_nodes and not execution_config.get("continue_on_failure", True):
            final_status = "failed"
        elif failed_nodes:
            final_status = "completed_with_errors"
        else:
            final_status = "completed"
        
        return {
            "status": final_status,
            "completed_nodes": list(completed_nodes),
            "failed_nodes": list(failed_nodes),
            "node_results": node_results,
            "checkpoints": execution_state.get("checkpoints", [])
        }
        
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        return {
            "status": "error",
            "error": str(e),
            "completed_nodes": list(completed_nodes),
            "failed_nodes": list(failed_nodes),
            "node_results": node_results
        }


def build_dependency_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Build dependency graph from nodes and edges."""
    dependencies = {node["id"]: set() for node in nodes}
    
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        dependencies[to_node].add(from_node)
    
    return dependencies


def find_entry_points(dependencies: Dict[str, Set[str]]) -> List[str]:
    """Find nodes with no dependencies (entry points)."""
    return [node_id for node_id, deps in dependencies.items() if not deps]


def find_newly_ready_nodes(
    completed_node: str,
    dependencies: Dict[str, Set[str]],
    completed_nodes: Set[str],
    failed_nodes: Set[str]
) -> List[str]:
    """Find nodes that are newly ready after a node completes."""
    newly_ready = []
    
    for node_id, deps in dependencies.items():
        if node_id not in completed_nodes and node_id not in failed_nodes:
            if deps.issubset(completed_nodes):
                newly_ready.append(node_id)
    
    return newly_ready


async def execute_node_batch(
    node_ids: List[str],
    nodes: List[Dict[str, Any]],
    node_results: Dict[str, Any],
    execution_config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Execute a batch of nodes in parallel."""
    
    batch_results = {}
    
    # Create tasks for parallel execution
    tasks = []
    for node_id in node_ids:
        node = next((n for n in nodes if n["id"] == node_id), None)
        if node:
            task = execute_single_node(node, node_results, execution_config)
            tasks.append((node_id, task))
    
    # Execute tasks
    if tasks:
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (node_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                batch_results[node_id] = {
                    "success": False,
                    "error": str(result),
                    "node_id": node_id
                }
            else:
                batch_results[node_id] = result
    
    return batch_results


async def execute_single_node(
    node: Dict[str, Any],
    node_results: Dict[str, Any],
    execution_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a single workflow node."""
    
    node_id = node["id"]
    start_time = datetime.now()
    
    try:
        # Simulate node execution (in production, this would call actual agents)
        agent_name = node.get("agent", "default_agent")
        action = node.get("action", "process")
        inputs = node.get("inputs", {})
        
        # Add results from previous nodes as inputs
        for input_key, input_source in inputs.items():
            if input_source in node_results:
                inputs[input_key] = node_results[input_source].get("result")
        
        # Simulate execution time based on node complexity
        execution_time = node.get("metadata", {}).get("estimated_duration", 5)
        await asyncio.sleep(min(execution_time / 10, 2))  # Scale down for simulation
        
        # Generate mock result
        result = {
            "success": True,
            "node_id": node_id,
            "agent": agent_name,
            "action": action,
            "result": f"Processed by {agent_name}: {action}",
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "outputs": node.get("outputs", {})
        }
        
        logger.debug(f"Executed node {node_id} successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error executing node {node_id}: {e}")
        return {
            "success": False,
            "node_id": node_id,
            "error": str(e),
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }


async def create_execution_checkpoint(
    execution_state: Dict[str, Any],
    completed_nodes: Set[str],
    failed_nodes: Set[str],
    node_results: Dict[str, Any]
):
    """Create execution checkpoint for recovery."""
    
    checkpoint = {
        "checkpoint_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "completed_nodes": list(completed_nodes),
        "failed_nodes": list(failed_nodes),
        "node_results": node_results,
        "execution_state": execution_state
    }
    
    execution_state.setdefault("checkpoints", []).append(checkpoint)
    
    # Store checkpoint
    await supabase_manager.insert_record(
        "execution_checkpoints",
        {
            "id": checkpoint["checkpoint_id"],
            "execution_id": execution_state["execution_id"],
            "checkpoint_data": checkpoint,
            "created_at": checkpoint["timestamp"]
        }
    )


async def supabase_task_queue_tool(
    action: str,
    task_data: Dict[str, Any] = None,
    queue_name: str = "default",
    priority: int = 5,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Manage task distribution using Supabase with real-time queue updates.
    
    Args:
        action: Action to perform (enqueue, dequeue, status, clear)
        task_data: Task data to enqueue
        queue_name: Queue name
        priority: Task priority (1-10, 10 is highest)
        tool_context: ADK tool context
    
    Returns:
        Dict containing queue operation result
    """
    try:
        if action == "enqueue":
            # Add task to queue
            if not task_data:
                return {"error": "Task data required for enqueue"}
            
            task_record = {
                "id": str(uuid.uuid4()),
                "queue_name": queue_name,
                "task_data": task_data,
                "priority": priority,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "attempts": 0
            }
            
            await supabase_manager.insert_record("task_queue", task_record)
            
            # Broadcast task availability
            await supabase_manager.broadcast_message(
                f"queue_{queue_name}",
                {
                    "type": "task_available",
                    "task_id": task_record["id"],
                    "priority": priority
                }
            )
            
            return {
                "action": "enqueue",
                "task_id": task_record["id"],
                "queue_name": queue_name,
                "success": True
            }
        
        elif action == "dequeue":
            # Get highest priority pending task
            tasks = await supabase_manager.query_records(
                "task_queue",
                filters={"queue_name": queue_name, "status": "pending"}
            )
            
            if not tasks:
                return {
                    "action": "dequeue",
                    "queue_name": queue_name,
                    "task": None,
                    "found": False
                }
            
            # Sort by priority (highest first) then by creation time
            tasks.sort(key=lambda t: (-t.get("priority", 5), t.get("created_at", "")))
            selected_task = tasks[0]
            
            # Mark task as executing
            await supabase_manager.update_record(
                "task_queue",
                selected_task["id"],
                {
                    "status": "executing",
                    "started_at": datetime.now().isoformat(),
                    "attempts": selected_task.get("attempts", 0) + 1
                }
            )
            
            return {
                "action": "dequeue",
                "queue_name": queue_name,
                "task": selected_task,
                "found": True
            }
        
        elif action == "complete":
            # Mark task as completed
            task_id = task_data.get("task_id") if task_data else ""
            if not task_id:
                return {"error": "Task ID required for completion"}
            
            await supabase_manager.update_record(
                "task_queue",
                task_id,
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": task_data.get("result", {})
                }
            )
            
            return {
                "action": "complete",
                "task_id": task_id,
                "success": True
            }
        
        elif action == "fail":
            # Mark task as failed
            task_id = task_data.get("task_id") if task_data else ""
            error_message = task_data.get("error", "") if task_data else ""
            
            if not task_id:
                return {"error": "Task ID required for failure"}
            
            # Get current task to check retry count
            current_task = await supabase_manager.get_record("task_queue", task_id)
            if not current_task:
                return {"error": "Task not found"}
            
            attempts = current_task.get("attempts", 0)
            max_attempts = 3
            
            if attempts < max_attempts:
                # Retry task
                await supabase_manager.update_record(
                    "task_queue",
                    task_id,
                    {
                        "status": "pending",
                        "error": error_message,
                        "last_attempt_at": datetime.now().isoformat()
                    }
                )
                status = "retrying"
            else:
                # Mark as permanently failed
                await supabase_manager.update_record(
                    "task_queue",
                    task_id,
                    {
                        "status": "failed",
                        "error": error_message,
                        "failed_at": datetime.now().isoformat()
                    }
                )
                status = "failed"
            
            return {
                "action": "fail",
                "task_id": task_id,
                "status": status,
                "attempts": attempts,
                "success": True
            }
        
        elif action == "status":
            # Get queue status
            all_tasks = await supabase_manager.query_records(
                "task_queue",
                filters={"queue_name": queue_name}
            )
            
            status_summary = {
                "pending": len([t for t in all_tasks if t.get("status") == "pending"]),
                "executing": len([t for t in all_tasks if t.get("status") == "executing"]),
                "completed": len([t for t in all_tasks if t.get("status") == "completed"]),
                "failed": len([t for t in all_tasks if t.get("status") == "failed"])
            }
            
            return {
                "action": "status",
                "queue_name": queue_name,
                "status_summary": status_summary,
                "total_tasks": len(all_tasks)
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in task queue tool: {e}")
        return {"error": str(e), "action": action}


async def realtime_state_tool(
    workflow_id: str,
    state_update: Dict[str, Any],
    update_type: str = "progress",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Real-time execution state updates with automatic conflict resolution.
    
    Args:
        workflow_id: Workflow identifier
        state_update: State update data
        update_type: Type of update (progress, status, error)
        tool_context: ADK tool context
    
    Returns:
        Dict containing state update result
    """
    try:
        timestamp = datetime.now().isoformat()
        
        # Get current state
        current_state = await supabase_manager.get_record("workflow_state", workflow_id)
        
        if current_state:
            # Merge state updates
            updated_state = merge_state_updates(current_state.get("value", {}), state_update)
            
            # Update with conflict resolution
            await supabase_manager.update_record(
                "workflow_state",
                workflow_id,
                {
                    "value": updated_state,
                    "update_type": update_type,
                    "updated_at": timestamp
                }
            )
        else:
            # Create new state record
            await supabase_manager.insert_record(
                "workflow_state",
                {
                    "id": workflow_id,
                    "key": workflow_id,
                    "value": state_update,
                    "update_type": update_type,
                    "updated_at": timestamp
                }
            )
        
        # Broadcast state update
        await supabase_manager.broadcast_message(
            "workflow_state_updates",
            {
                "type": "state_updated",
                "workflow_id": workflow_id,
                "update_type": update_type,
                "timestamp": timestamp,
                "state_update": state_update
            }
        )
        
        return {
            "workflow_id": workflow_id,
            "update_type": update_type,
            "timestamp": timestamp,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in realtime state tool: {e}")
        return {"error": str(e), "workflow_id": workflow_id}


def merge_state_updates(current_state: Dict[str, Any], state_update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge state updates with conflict resolution."""
    
    merged_state = current_state.copy()
    
    for key, value in state_update.items():
        if key in merged_state:
            # Handle conflicts based on key type
            if key == "progress":
                # Take the maximum progress value
                merged_state[key] = max(merged_state[key], value)
            elif key == "status":
                # Status transitions follow a hierarchy
                status_priority = {
                    "pending": 1,
                    "executing": 2,
                    "completed": 3,
                    "failed": 4,
                    "error": 5
                }
                current_priority = status_priority.get(merged_state[key], 0)
                new_priority = status_priority.get(value, 0)
                if new_priority > current_priority:
                    merged_state[key] = value
            elif key == "timestamp":
                # Always use the latest timestamp
                merged_state[key] = value
            else:
                # For other keys, use the new value
                merged_state[key] = value
        else:
            merged_state[key] = value
    
    return merged_state


async def health_monitoring_tool(
    action: str,
    agent_id: str = "",
    health_data: Dict[str, Any] = None,
    alert_thresholds: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Monitor agent health with instant notifications via Broadcast channels.
    
    Args:
        action: Action to perform (check, report, alert, status)
        agent_id: Agent identifier
        health_data: Health metrics data
        alert_thresholds: Threshold values for alerts
        tool_context: ADK tool context
    
    Returns:
        Dict containing health monitoring result
    """
    try:
        if action == "report":
            # Report agent health metrics
            if not health_data:
                return {"error": "Health data required for reporting"}
            
            health_record = {
                "id": str(uuid.uuid4()),
                "agent_id": agent_id,
                "health_data": health_data,
                "timestamp": datetime.now().isoformat(),
                "status": determine_health_status(health_data, alert_thresholds)
            }
            
            await supabase_manager.insert_record("agent_health", health_record)
            
            # Check for alerts
            alerts = check_health_alerts(health_data, alert_thresholds or {})
            
            if alerts:
                # Broadcast alerts
                await supabase_manager.broadcast_message(
                    "health_alerts",
                    {
                        "type": "health_alert",
                        "agent_id": agent_id,
                        "alerts": alerts,
                        "timestamp": health_record["timestamp"]
                    }
                )
            
            return {
                "action": "report",
                "agent_id": agent_id,
                "status": health_record["status"],
                "alerts": alerts,
                "success": True
            }
        
        elif action == "check":
            # Check specific agent health
            recent_health = await supabase_manager.query_records(
                "agent_health",
                filters={"agent_id": agent_id},
                limit=1
            )
            
            if recent_health:
                return {
                    "action": "check",
                    "agent_id": agent_id,
                    "health_data": recent_health[0],
                    "found": True
                }
            else:
                return {
                    "action": "check",
                    "agent_id": agent_id,
                    "health_data": None,
                    "found": False
                }
        
        elif action == "status":
            # Get overall system health status
            all_health = await supabase_manager.query_records("agent_health", limit=100)
            
            # Group by agent and get latest status
            agent_status = {}
            for record in all_health:
                agent = record["agent_id"]
                timestamp = record["timestamp"]
                
                if agent not in agent_status or timestamp > agent_status[agent]["timestamp"]:
                    agent_status[agent] = record
            
            # Calculate overall status
            statuses = [record["status"] for record in agent_status.values()]
            overall_status = determine_overall_status(statuses)
            
            return {
                "action": "status",
                "overall_status": overall_status,
                "agent_count": len(agent_status),
                "agent_statuses": agent_status,
                "healthy_agents": len([s for s in statuses if s == "healthy"]),
                "unhealthy_agents": len([s for s in statuses if s != "healthy"])
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in health monitoring tool: {e}")
        return {"error": str(e), "action": action}


def determine_health_status(health_data: Dict[str, Any], alert_thresholds: Dict[str, Any] = None) -> str:
    """Determine health status based on metrics."""
    
    if not alert_thresholds:
        alert_thresholds = {
            "cpu_usage": 80,
            "memory_usage": 80,
            "error_rate": 5,
            "response_time": 5000  # milliseconds
        }
    
    # Check critical metrics
    cpu_usage = health_data.get("cpu_usage", 0)
    memory_usage = health_data.get("memory_usage", 0)
    error_rate = health_data.get("error_rate", 0)
    response_time = health_data.get("response_time", 0)
    
    if (cpu_usage > alert_thresholds.get("cpu_usage", 80) or
        memory_usage > alert_thresholds.get("memory_usage", 80) or
        error_rate > alert_thresholds.get("error_rate", 5) or
        response_time > alert_thresholds.get("response_time", 5000)):
        return "unhealthy"
    elif (cpu_usage > alert_thresholds.get("cpu_usage", 80) * 0.8 or
          memory_usage > alert_thresholds.get("memory_usage", 80) * 0.8):
        return "warning"
    else:
        return "healthy"


def check_health_alerts(health_data: Dict[str, Any], alert_thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for health alerts based on thresholds."""
    
    alerts = []
    
    for metric, threshold in alert_thresholds.items():
        value = health_data.get(metric, 0)
        
        if value > threshold:
            alerts.append({
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "severity": "critical" if value > threshold * 1.2 else "warning"
            })
    
    return alerts


def determine_overall_status(statuses: List[str]) -> str:
    """Determine overall system status from individual agent statuses."""
    
    if not statuses:
        return "unknown"
    
    unhealthy_count = len([s for s in statuses if s == "unhealthy"])
    warning_count = len([s for s in statuses if s == "warning"])
    
    if unhealthy_count > len(statuses) * 0.3:  # More than 30% unhealthy
        return "critical"
    elif unhealthy_count > 0 or warning_count > len(statuses) * 0.5:  # Any unhealthy or >50% warning
        return "degraded"
    elif warning_count > 0:
        return "warning"
    else:
        return "healthy"


async def presence_tracking_tool(
    action: str,
    agent_id: str = "",
    presence_data: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Track active agents and users using Supabase Presence for coordination.
    
    Args:
        action: Action to perform (join, leave, update, list)
        agent_id: Agent identifier
        presence_data: Presence data (status, metadata, etc.)
        tool_context: ADK tool context
    
    Returns:
        Dict containing presence tracking result
    """
    try:
        if action == "join":
            # Join presence tracking
            if not presence_data:
                presence_data = {
                    "status": "online",
                    "capabilities": [],
                    "load": 0
                }
            
            presence_record = {
                "agent_id": agent_id,
                "presence_data": presence_data,
                "last_seen": datetime.now().isoformat(),
                "status": "online"
            }
            
            # Update or insert presence
            existing = await supabase_manager.query_records(
                "agent_presence",
                filters={"agent_id": agent_id}
            )
            
            if existing:
                await supabase_manager.update_record(
                    "agent_presence",
                    existing[0]["id"],
                    presence_record
                )
            else:
                presence_record["id"] = str(uuid.uuid4())
                await supabase_manager.insert_record("agent_presence", presence_record)
            
            # Broadcast presence update
            await supabase_manager.broadcast_message(
                "agent_presence",
                {
                    "type": "agent_joined",
                    "agent_id": agent_id,
                    "presence_data": presence_data
                }
            )
            
            return {
                "action": "join",
                "agent_id": agent_id,
                "success": True
            }
        
        elif action == "leave":
            # Leave presence tracking
            await supabase_manager.update_record(
                "agent_presence",
                agent_id,
                {
                    "status": "offline",
                    "last_seen": datetime.now().isoformat()
                }
            )
            
            # Broadcast presence update
            await supabase_manager.broadcast_message(
                "agent_presence",
                {
                    "type": "agent_left",
                    "agent_id": agent_id
                }
            )
            
            return {
                "action": "leave",
                "agent_id": agent_id,
                "success": True
            }
        
        elif action == "update":
            # Update presence data
            if not presence_data:
                return {"error": "Presence data required for update"}
            
            await supabase_manager.update_record(
                "agent_presence",
                agent_id,
                {
                    "presence_data": presence_data,
                    "last_seen": datetime.now().isoformat()
                }
            )
            
            return {
                "action": "update",
                "agent_id": agent_id,
                "success": True
            }
        
        elif action == "list":
            # List all active agents
            current_time = datetime.now()
            timeout_threshold = current_time - timedelta(minutes=5)  # 5 minute timeout
            
            all_presence = await supabase_manager.query_records("agent_presence")
            
            active_agents = []
            for record in all_presence:
                last_seen = datetime.fromisoformat(record["last_seen"].replace('Z', '+00:00'))
                if last_seen > timeout_threshold and record.get("status") == "online":
                    active_agents.append(record)
            
            return {
                "action": "list",
                "active_agents": active_agents,
                "total_active": len(active_agents)
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in presence tracking tool: {e}")
        return {"error": str(e), "action": action} 