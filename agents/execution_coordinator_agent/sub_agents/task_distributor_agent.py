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

"""TaskDistributorAgent - Sub-agent for optimal task distribution and load balancing."""

import uuid
from datetime import datetime
from typing import Dict, Any, List
from google.genai import types
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from config.settings import settings
from utils.supabase_client import supabase_manager
import logging

logger = logging.getLogger(__name__)


async def distribute_tasks_optimally(
    tasks: List[Dict[str, Any]],
    available_agents: List[Dict[str, Any]],
    distribution_strategy: str = "balanced",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Optimally distribute tasks across available agents based on load and capability.
    
    Args:
        tasks: List of tasks to distribute
        available_agents: List of available agents with their capabilities
        distribution_strategy: Strategy to use (balanced, capability, load, geographic)
        tool_context: ADK tool context
    
    Returns:
        Dict containing task distribution plan
    """
    try:
        distribution_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Analyze task requirements
        task_analysis = analyze_task_requirements(tasks)
        
        # Analyze agent capabilities and current load
        agent_analysis = analyze_agent_capabilities(available_agents)
        
        # Generate optimal distribution plan
        distribution_plan = generate_distribution_plan(
            tasks, available_agents, task_analysis, agent_analysis, distribution_strategy
        )
        
        # Validate distribution plan
        validation_result = validate_distribution_plan(distribution_plan, tasks, available_agents)
        
        # Calculate optimization metrics
        optimization_metrics = calculate_optimization_metrics(distribution_plan, task_analysis, agent_analysis)
        
        result = {
            "distribution_id": distribution_id,
            "timestamp": timestamp,
            "strategy": distribution_strategy,
            "task_count": len(tasks),
            "agent_count": len(available_agents),
            "distribution_plan": distribution_plan,
            "task_analysis": task_analysis,
            "agent_analysis": agent_analysis,
            "validation": validation_result,
            "optimization_metrics": optimization_metrics
        }
        
        # Store distribution plan
        await supabase_manager.insert_record(
            "task_distributions",
            {
                "id": distribution_id,
                "distribution_plan": result,
                "timestamp": timestamp
            }
        )
        
        # Update tool context
        if tool_context:
            tool_context.state["distribution_plan"] = result
            tool_context.state["distribution_id"] = distribution_id
        
        logger.info(f"Generated task distribution plan {distribution_id} for {len(tasks)} tasks")
        return result
        
    except Exception as e:
        logger.error(f"Error in task distribution: {e}")
        return {"error": str(e), "distribution_id": distribution_id}


def analyze_task_requirements(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze requirements across all tasks."""
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        return {"total_tasks": 0}
    
    # Aggregate requirements
    capabilities_needed = {}
    priority_distribution = {"low": 0, "medium": 0, "high": 0}
    estimated_total_time = 0
    resource_requirements = {"cpu": [], "memory": [], "network": []}
    
    for task in tasks:
        # Capabilities
        required_capabilities = task.get("required_capabilities", [])
        for capability in required_capabilities:
            capabilities_needed[capability] = capabilities_needed.get(capability, 0) + 1
        
        # Priority
        priority = task.get("priority", "medium")
        if priority in priority_distribution:
            priority_distribution[priority] += 1
        
        # Time estimation
        estimated_time = task.get("estimated_time", 30)
        estimated_total_time += estimated_time
        
        # Resources
        resources = task.get("resource_requirements", {})
        for resource_type in ["cpu", "memory", "network"]:
            if resource_type in resources:
                resource_requirements[resource_type].append(resources[resource_type])
    
    # Calculate resource averages
    avg_resources = {}
    for resource_type, values in resource_requirements.items():
        if values:
            level_weights = {"low": 1, "medium": 2, "high": 3}
            avg_level = sum(level_weights.get(v, 2) for v in values) / len(values)
            if avg_level <= 1.5:
                avg_resources[resource_type] = "low"
            elif avg_level <= 2.5:
                avg_resources[resource_type] = "medium"
            else:
                avg_resources[resource_type] = "high"
    
    return {
        "total_tasks": total_tasks,
        "capabilities_needed": capabilities_needed,
        "priority_distribution": priority_distribution,
        "estimated_total_time": estimated_total_time,
        "average_task_time": estimated_total_time / total_tasks,
        "resource_requirements": avg_resources,
        "complexity_score": calculate_task_complexity(tasks)
    }


def analyze_agent_capabilities(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze capabilities and current load of available agents."""
    
    total_agents = len(agents)
    if total_agents == 0:
        return {"total_agents": 0}
    
    # Aggregate agent data
    all_capabilities = set()
    load_distribution = []
    availability_count = {"online": 0, "busy": 0, "offline": 0}
    performance_metrics = {"response_times": [], "success_rates": [], "throughput": []}
    
    for agent in agents:
        # Capabilities
        capabilities = agent.get("capabilities", [])
        all_capabilities.update(capabilities)
        
        # Load
        current_load = agent.get("current_load", 0)
        load_distribution.append(current_load)
        
        # Availability
        status = agent.get("status", "online")
        if status in availability_count:
            availability_count[status] += 1
        
        # Performance
        metrics = agent.get("performance_metrics", {})
        if "response_time" in metrics:
            performance_metrics["response_times"].append(metrics["response_time"])
        if "success_rate" in metrics:
            performance_metrics["success_rates"].append(metrics["success_rate"])
        if "throughput" in metrics:
            performance_metrics["throughput"].append(metrics["throughput"])
    
    # Calculate averages
    avg_load = sum(load_distribution) / total_agents if load_distribution else 0
    avg_performance = {}
    for metric, values in performance_metrics.items():
        if values:
            avg_performance[metric] = sum(values) / len(values)
    
    return {
        "total_agents": total_agents,
        "available_capabilities": list(all_capabilities),
        "average_load": avg_load,
        "load_distribution": load_distribution,
        "availability": availability_count,
        "performance_metrics": avg_performance,
        "capacity_score": calculate_system_capacity(agents)
    }


def generate_distribution_plan(
    tasks: List[Dict[str, Any]],
    agents: List[Dict[str, Any]],
    task_analysis: Dict[str, Any],
    agent_analysis: Dict[str, Any],
    strategy: str
) -> Dict[str, Any]:
    """Generate optimal distribution plan based on strategy."""
    
    if strategy == "balanced":
        return generate_balanced_distribution(tasks, agents)
    elif strategy == "capability":
        return generate_capability_based_distribution(tasks, agents)
    elif strategy == "load":
        return generate_load_based_distribution(tasks, agents)
    elif strategy == "priority":
        return generate_priority_based_distribution(tasks, agents)
    else:
        return generate_balanced_distribution(tasks, agents)


def generate_balanced_distribution(tasks: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate balanced distribution across all agents."""
    
    if not agents:
        return {"assignments": [], "strategy": "balanced", "error": "No agents available"}
    
    assignments = []
    agent_loads = {agent["id"]: agent.get("current_load", 0) for agent in agents}
    
    # Sort tasks by priority
    sorted_tasks = sorted(tasks, key=lambda t: {"high": 3, "medium": 2, "low": 1}.get(t.get("priority", "medium"), 2), reverse=True)
    
    for task in sorted_tasks:
        # Find agent with lowest current load
        best_agent_id = min(agent_loads.keys(), key=lambda aid: agent_loads[aid])
        best_agent = next(agent for agent in agents if agent["id"] == best_agent_id)
        
        # Create assignment
        assignment = {
            "task_id": task["id"],
            "agent_id": best_agent_id,
            "agent_name": best_agent.get("name", best_agent_id),
            "estimated_time": task.get("estimated_time", 30),
            "priority": task.get("priority", "medium"),
            "assignment_reason": "load_balancing"
        }
        assignments.append(assignment)
        
        # Update agent load
        agent_loads[best_agent_id] += task.get("estimated_time", 30)
    
    return {
        "assignments": assignments,
        "strategy": "balanced",
        "final_loads": agent_loads
    }


def generate_capability_based_distribution(tasks: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate distribution based on agent capabilities."""
    
    assignments = []
    unassigned_tasks = []
    
    for task in tasks:
        required_capabilities = set(task.get("required_capabilities", []))
        best_agent = None
        best_match_score = 0
        
        for agent in agents:
            agent_capabilities = set(agent.get("capabilities", []))
            
            # Check if agent can handle the task
            if required_capabilities.issubset(agent_capabilities):
                # Calculate match score (more capabilities = higher score)
                match_score = len(agent_capabilities.intersection(required_capabilities))
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_agent = agent
        
        if best_agent:
            assignment = {
                "task_id": task["id"],
                "agent_id": best_agent["id"],
                "agent_name": best_agent.get("name", best_agent["id"]),
                "match_score": best_match_score,
                "required_capabilities": list(required_capabilities),
                "assignment_reason": "capability_match"
            }
            assignments.append(assignment)
        else:
            unassigned_tasks.append(task["id"])
    
    return {
        "assignments": assignments,
        "strategy": "capability",
        "unassigned_tasks": unassigned_tasks
    }


def generate_load_based_distribution(tasks: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate distribution based on current agent load."""
    
    assignments = []
    
    # Sort agents by current load (ascending)
    sorted_agents = sorted(agents, key=lambda a: a.get("current_load", 0))
    
    # Sort tasks by estimated time (descending) to balance load better
    sorted_tasks = sorted(tasks, key=lambda t: t.get("estimated_time", 30), reverse=True)
    
    agent_loads = {agent["id"]: agent.get("current_load", 0) for agent in agents}
    
    for task in sorted_tasks:
        # Find agent with lowest current load
        best_agent_id = min(agent_loads.keys(), key=lambda aid: agent_loads[aid])
        best_agent = next(agent for agent in agents if agent["id"] == best_agent_id)
        
        assignment = {
            "task_id": task["id"],
            "agent_id": best_agent_id,
            "agent_name": best_agent.get("name", best_agent_id),
            "current_load_before": agent_loads[best_agent_id],
            "estimated_time": task.get("estimated_time", 30),
            "assignment_reason": "load_optimization"
        }
        assignments.append(assignment)
        
        # Update load
        agent_loads[best_agent_id] += task.get("estimated_time", 30)
    
    return {
        "assignments": assignments,
        "strategy": "load",
        "final_loads": agent_loads
    }


def generate_priority_based_distribution(tasks: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate distribution based on task priority."""
    
    assignments = []
    
    # Sort tasks by priority
    priority_order = {"high": 3, "medium": 2, "low": 1}
    sorted_tasks = sorted(tasks, key=lambda t: priority_order.get(t.get("priority", "medium"), 2), reverse=True)
    
    # Sort agents by performance (best performers get high priority tasks)
    sorted_agents = sorted(agents, key=lambda a: a.get("performance_metrics", {}).get("success_rate", 0.8), reverse=True)
    
    agent_index = 0
    for task in sorted_tasks:
        if not sorted_agents:
            break
        
        # Assign high priority tasks to best performing agents
        agent = sorted_agents[agent_index % len(sorted_agents)]
        
        assignment = {
            "task_id": task["id"],
            "agent_id": agent["id"],
            "agent_name": agent.get("name", agent["id"]),
            "task_priority": task.get("priority", "medium"),
            "agent_performance": agent.get("performance_metrics", {}).get("success_rate", 0.8),
            "assignment_reason": "priority_optimization"
        }
        assignments.append(assignment)
        
        agent_index += 1
    
    return {
        "assignments": assignments,
        "strategy": "priority"
    }


def validate_distribution_plan(distribution_plan: Dict[str, Any], tasks: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the distribution plan for feasibility and completeness."""
    
    assignments = distribution_plan.get("assignments", [])
    
    # Check if all tasks are assigned
    assigned_task_ids = {assignment["task_id"] for assignment in assignments}
    all_task_ids = {task["id"] for task in tasks}
    unassigned_tasks = all_task_ids - assigned_task_ids
    
    # Check if all agents exist
    agent_ids = {agent["id"] for agent in agents}
    assigned_agent_ids = {assignment["agent_id"] for assignment in assignments}
    invalid_agents = assigned_agent_ids - agent_ids
    
    # Check for overloading
    agent_loads = {}
    for assignment in assignments:
        agent_id = assignment["agent_id"]
        estimated_time = assignment.get("estimated_time", 30)
        agent_loads[agent_id] = agent_loads.get(agent_id, 0) + estimated_time
    
    overloaded_agents = [agent_id for agent_id, load in agent_loads.items() if load > 300]  # 5 minutes max
    
    # Calculate validation score
    total_issues = len(unassigned_tasks) + len(invalid_agents) + len(overloaded_agents)
    validation_score = max(0, 1 - (total_issues / max(len(tasks), 1)))
    
    return {
        "is_valid": total_issues == 0,
        "validation_score": validation_score,
        "unassigned_tasks": list(unassigned_tasks),
        "invalid_agents": list(invalid_agents),
        "overloaded_agents": overloaded_agents,
        "agent_loads": agent_loads,
        "total_issues": total_issues
    }


def calculate_optimization_metrics(distribution_plan: Dict[str, Any], task_analysis: Dict[str, Any], agent_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimization metrics for the distribution plan."""
    
    assignments = distribution_plan.get("assignments", [])
    
    if not assignments:
        return {"load_balance_score": 0, "efficiency_score": 0, "utilization_score": 0}
    
    # Load balance score
    agent_loads = {}
    for assignment in assignments:
        agent_id = assignment["agent_id"]
        estimated_time = assignment.get("estimated_time", 30)
        agent_loads[agent_id] = agent_loads.get(agent_id, 0) + estimated_time
    
    load_values = list(agent_loads.values())
    if load_values:
        load_variance = sum((load - sum(load_values)/len(load_values))**2 for load in load_values) / len(load_values)
        load_balance_score = max(0, 1 - (load_variance / 10000))  # Normalize variance
    else:
        load_balance_score = 1.0
    
    # Efficiency score (based on capability matching)
    capability_matches = sum(1 for assignment in assignments if assignment.get("match_score", 0) > 0)
    efficiency_score = capability_matches / len(assignments) if assignments else 0
    
    # Utilization score
    total_estimated_time = sum(assignment.get("estimated_time", 30) for assignment in assignments)
    total_agent_capacity = agent_analysis.get("total_agents", 1) * 300  # Assume 5 min capacity per agent
    utilization_score = min(1.0, total_estimated_time / total_agent_capacity) if total_agent_capacity > 0 else 0
    
    return {
        "load_balance_score": round(load_balance_score, 3),
        "efficiency_score": round(efficiency_score, 3),
        "utilization_score": round(utilization_score, 3),
        "overall_score": round((load_balance_score + efficiency_score + utilization_score) / 3, 3)
    }


def calculate_task_complexity(tasks: List[Dict[str, Any]]) -> float:
    """Calculate overall task complexity score."""
    
    if not tasks:
        return 0.0
    
    complexity_sum = 0
    for task in tasks:
        # Base complexity
        complexity = 1.0
        
        # Add complexity for capabilities
        capabilities_count = len(task.get("required_capabilities", []))
        complexity += capabilities_count * 0.5
        
        # Add complexity for estimated time
        estimated_time = task.get("estimated_time", 30)
        if estimated_time > 60:
            complexity += 1.0
        elif estimated_time > 120:
            complexity += 2.0
        
        # Add complexity for priority
        if task.get("priority") == "high":
            complexity += 0.5
        
        complexity_sum += complexity
    
    return complexity_sum / len(tasks)


def calculate_system_capacity(agents: List[Dict[str, Any]]) -> float:
    """Calculate overall system capacity score."""
    
    if not agents:
        return 0.0
    
    capacity_sum = 0
    for agent in agents:
        # Base capacity
        capacity = 1.0
        
        # Add capacity for capabilities
        capabilities_count = len(agent.get("capabilities", []))
        capacity += capabilities_count * 0.3
        
        # Add capacity for performance
        performance = agent.get("performance_metrics", {})
        success_rate = performance.get("success_rate", 0.8)
        capacity += success_rate * 0.5
        
        # Reduce capacity for current load
        current_load = agent.get("current_load", 0)
        load_factor = max(0, 1 - (current_load / 300))  # Assume 300s max capacity
        capacity *= load_factor
        
        capacity_sum += capacity
    
    return capacity_sum / len(agents)


# Create the TaskDistributorAgent
task_distributor_agent = Agent(
    model=settings.root_agent_model,
    name="task_distributor_agent",
    instruction="""
You are the TaskDistributorAgent, a specialized sub-agent focused on optimal task distribution 
and load balancing across the multi-agent system.

Your core responsibilities:
1. Analyze task requirements and agent capabilities
2. Generate optimal task distribution plans
3. Implement intelligent load balancing algorithms
4. Optimize resource utilization and performance
5. Validate distribution plans for feasibility

Use the distribute_tasks_optimally tool to create efficient task distribution strategies.
""",
    tools=[distribute_tasks_optimally],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=4096
    )
) 