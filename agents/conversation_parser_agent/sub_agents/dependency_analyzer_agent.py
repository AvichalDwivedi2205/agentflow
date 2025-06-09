"""DependencyAnalyzerAgent - Sub-agent for workflow dependency analysis."""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple
from google.genai import types
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


async def analyze_workflow_dependencies(
    workflow_graph: Dict[str, Any],
    optimization_goals: List[str] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Analyze workflow dependencies and identify optimization opportunities.
    
    Args:
        workflow_graph: Workflow graph to analyze
        optimization_goals: List of optimization goals (speed, resource, reliability)
        tool_context: ADK tool context
    
    Returns:
        Dict containing dependency analysis results
    """
    try:
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        nodes = workflow_graph.get("nodes", [])
        edges = workflow_graph.get("edges", [])
        
        if not optimization_goals:
            optimization_goals = ["speed", "resource"]
        
        # Build dependency graph
        dependency_graph = build_dependency_graph(nodes, edges)
        
        # Analyze data flow patterns
        data_flow = analyze_data_flow_patterns(nodes, edges)
        
        # Identify bottlenecks
        bottlenecks = identify_bottlenecks(dependency_graph, nodes)
        
        # Find parallel execution opportunities
        parallel_opportunities = find_parallel_opportunities(dependency_graph, nodes)
        
        # Optimize execution order
        optimized_order = optimize_execution_order(dependency_graph, nodes, optimization_goals)
        
        # Calculate critical path
        critical_path = calculate_critical_path(dependency_graph, nodes)
        
        # Generate optimization recommendations
        recommendations = generate_optimization_recommendations(
            dependency_graph, bottlenecks, parallel_opportunities, optimization_goals
        )
        
        analysis_results = {
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "workflow_id": workflow_graph.get("workflow_id"),
            "dependency_graph": dependency_graph,
            "data_flow": data_flow,
            "bottlenecks": bottlenecks,
            "parallel_opportunities": parallel_opportunities,
            "optimized_execution_order": optimized_order,
            "critical_path": critical_path,
            "recommendations": recommendations,
            "metrics": {
                "total_nodes": len(nodes),
                "total_dependencies": len(edges),
                "parallelization_potential": len(parallel_opportunities),
                "bottleneck_count": len(bottlenecks),
                "critical_path_length": len(critical_path)
            }
        }
        
        # Update tool context
        if tool_context:
            tool_context.state["dependency_analysis"] = analysis_results
            tool_context.state["dependency_analysis_id"] = analysis_id
        
        logger.info(f"Completed dependency analysis {analysis_id}")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error in dependency analysis: {e}")
        return {"error": str(e), "analysis_id": analysis_id}


def build_dependency_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a comprehensive dependency graph."""
    graph = {
        "nodes": {node["id"]: node for node in nodes},
        "adjacency_list": {},
        "reverse_adjacency_list": {},
        "dependency_levels": {},
        "resource_dependencies": {}
    }
    
    # Initialize adjacency lists
    for node in nodes:
        node_id = node["id"]
        graph["adjacency_list"][node_id] = []
        graph["reverse_adjacency_list"][node_id] = []
    
    # Build adjacency lists from edges
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        
        graph["adjacency_list"][from_node].append({
            "target": to_node,
            "type": edge.get("type", "sequential"),
            "condition": edge.get("condition"),
            "weight": edge.get("weight", 1.0)
        })
        
        graph["reverse_adjacency_list"][to_node].append({
            "source": from_node,
            "type": edge.get("type", "sequential"),
            "condition": edge.get("condition"),
            "weight": edge.get("weight", 1.0)
        })
    
    # Calculate dependency levels
    graph["dependency_levels"] = calculate_dependency_levels(graph["adjacency_list"], nodes)
    
    # Analyze resource dependencies
    graph["resource_dependencies"] = analyze_resource_dependencies(nodes)
    
    return graph


def calculate_dependency_levels(adjacency_list: Dict[str, List], nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate the dependency level for each node."""
    levels = {}
    
    # Find nodes with no dependencies (level 0)
    for node in nodes:
        node_id = node["id"]
        if not any(node_id in [dep["target"] for deps in adjacency_list.values() for dep in deps]):
            levels[node_id] = 0
    
    # Calculate levels iteratively
    max_iterations = len(nodes)
    for iteration in range(max_iterations):
        changed = False
        for node in nodes:
            node_id = node["id"]
            if node_id not in levels:
                # Check if all dependencies have levels assigned
                dependencies = [dep["source"] for dep_list in adjacency_list.values() 
                              for dep in dep_list if dep["target"] == node_id]
                
                if all(dep in levels for dep in dependencies):
                    if dependencies:
                        levels[node_id] = max(levels[dep] for dep in dependencies) + 1
                    else:
                        levels[node_id] = 0
                    changed = True
        
        if not changed:
            break
    
    # Assign remaining nodes to level 0 (fallback)
    for node in nodes:
        if node["id"] not in levels:
            levels[node["id"]] = 0
    
    return levels


def analyze_resource_dependencies(nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Analyze resource dependencies between nodes."""
    resource_deps = {}
    
    # Group nodes by resource requirements
    resource_groups = {}
    for node in nodes:
        requirements = node.get("metadata", {}).get("resource_requirements", {})
        for resource, level in requirements.items():
            if resource not in resource_groups:
                resource_groups[resource] = {}
            if level not in resource_groups[resource]:
                resource_groups[resource][level] = []
            resource_groups[resource][level].append(node["id"])
    
    # Identify potential resource conflicts
    for resource, levels in resource_groups.items():
        high_usage_nodes = levels.get("high", [])
        if len(high_usage_nodes) > 1:
            resource_deps[f"{resource}_conflict"] = high_usage_nodes
    
    return resource_deps


def analyze_data_flow_patterns(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze data flow patterns between nodes."""
    data_flow = {
        "input_nodes": [],
        "output_nodes": [],
        "transformation_nodes": [],
        "data_dependencies": [],
        "flow_patterns": {}
    }
    
    # Identify node types based on inputs/outputs
    for node in nodes:
        node_id = node["id"]
        inputs = node.get("inputs", {})
        outputs = node.get("outputs", {})
        
        if not inputs or len(inputs) == 0:
            data_flow["input_nodes"].append(node_id)
        elif not outputs or len(outputs) == 0:
            data_flow["output_nodes"].append(node_id)
        else:
            data_flow["transformation_nodes"].append(node_id)
    
    # Analyze data dependencies
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        
        # Check if there's data flow between nodes
        from_outputs = next((n["outputs"] for n in nodes if n["id"] == from_node), {})
        to_inputs = next((n["inputs"] for n in nodes if n["id"] == to_node), {})
        
        # Simple heuristic: if output keys match input keys, there's data dependency
        data_keys = set(from_outputs.keys()) & set(to_inputs.keys())
        if data_keys:
            data_flow["data_dependencies"].append({
                "from": from_node,
                "to": to_node,
                "data_keys": list(data_keys)
            })
    
    # Identify flow patterns
    data_flow["flow_patterns"] = {
        "linear_chains": find_linear_chains(edges),
        "fan_out_patterns": find_fan_out_patterns(edges),
        "fan_in_patterns": find_fan_in_patterns(edges),
        "cycles": detect_cycles(edges)
    }
    
    return data_flow


def find_linear_chains(edges: List[Dict[str, Any]]) -> List[List[str]]:
    """Find linear chains in the workflow."""
    chains = []
    
    # Build adjacency map
    adjacency = {}
    reverse_adjacency = {}
    
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        
        if from_node not in adjacency:
            adjacency[from_node] = []
        adjacency[from_node].append(to_node)
        
        if to_node not in reverse_adjacency:
            reverse_adjacency[to_node] = []
        reverse_adjacency[to_node].append(from_node)
    
    # Find chain starts (nodes with no predecessors or multiple predecessors)
    visited = set()
    
    for node in adjacency.keys():
        if node not in visited and (node not in reverse_adjacency or len(reverse_adjacency[node]) != 1):
            chain = build_chain(node, adjacency, reverse_adjacency)
            if len(chain) > 1:
                chains.append(chain)
                visited.update(chain)
    
    return chains


def build_chain(start_node: str, adjacency: Dict[str, List[str]], reverse_adjacency: Dict[str, List[str]]) -> List[str]:
    """Build a linear chain starting from a node."""
    chain = [start_node]
    current = start_node
    
    while current in adjacency and len(adjacency[current]) == 1:
        next_node = adjacency[current][0]
        if next_node in reverse_adjacency and len(reverse_adjacency[next_node]) == 1:
            chain.append(next_node)
            current = next_node
        else:
            break
    
    return chain


def find_fan_out_patterns(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find fan-out patterns (one node feeding multiple nodes)."""
    fan_outs = []
    
    # Count outgoing edges for each node
    outgoing_counts = {}
    for edge in edges:
        from_node = edge["from"]
        if from_node not in outgoing_counts:
            outgoing_counts[from_node] = []
        outgoing_counts[from_node].append(edge["to"])
    
    # Identify fan-out patterns
    for node, targets in outgoing_counts.items():
        if len(targets) > 1:
            fan_outs.append({
                "source": node,
                "targets": targets,
                "fan_out_degree": len(targets)
            })
    
    return fan_outs


def find_fan_in_patterns(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find fan-in patterns (multiple nodes feeding one node)."""
    fan_ins = []
    
    # Count incoming edges for each node
    incoming_counts = {}
    for edge in edges:
        to_node = edge["to"]
        if to_node not in incoming_counts:
            incoming_counts[to_node] = []
        incoming_counts[to_node].append(edge["from"])
    
    # Identify fan-in patterns
    for node, sources in incoming_counts.items():
        if len(sources) > 1:
            fan_ins.append({
                "target": node,
                "sources": sources,
                "fan_in_degree": len(sources)
            })
    
    return fan_ins


def detect_cycles(edges: List[Dict[str, Any]]) -> List[List[str]]:
    """Detect cycles in the workflow graph."""
    # Build adjacency list
    adjacency = {}
    for edge in edges:
        from_node = edge["from"]
        if from_node not in adjacency:
            adjacency[from_node] = []
        adjacency[from_node].append(edge["to"])
    
    # DFS-based cycle detection
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node, path):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in adjacency.get(node, []):
            dfs(neighbor, path + [node])
        
        rec_stack.remove(node)
    
    for node in adjacency.keys():
        if node not in visited:
            dfs(node, [])
    
    return cycles


def identify_bottlenecks(dependency_graph: Dict[str, Any], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify potential bottlenecks in the workflow."""
    bottlenecks = []
    
    adjacency_list = dependency_graph["adjacency_list"]
    reverse_adjacency_list = dependency_graph["reverse_adjacency_list"]
    
    for node in nodes:
        node_id = node["id"]
        
        # Check for high fan-in (many dependencies)
        incoming_count = len(reverse_adjacency_list.get(node_id, []))
        
        # Check for high fan-out (many dependents)
        outgoing_count = len(adjacency_list.get(node_id, []))
        
        # Check for high resource requirements
        resource_requirements = node.get("metadata", {}).get("resource_requirements", {})
        high_resource_usage = any(level == "high" for level in resource_requirements.values())
        
        # Check for long estimated duration
        estimated_duration = node.get("metadata", {}).get("estimated_duration", 0)
        
        bottleneck_score = 0
        reasons = []
        
        if incoming_count > 2:
            bottleneck_score += incoming_count * 0.2
            reasons.append(f"High fan-in: {incoming_count} dependencies")
        
        if outgoing_count > 2:
            bottleneck_score += outgoing_count * 0.2
            reasons.append(f"High fan-out: {outgoing_count} dependents")
        
        if high_resource_usage:
            bottleneck_score += 0.5
            reasons.append("High resource requirements")
        
        if estimated_duration > 30:  # More than 30 seconds
            bottleneck_score += estimated_duration / 60  # Convert to minutes
            reasons.append(f"Long duration: {estimated_duration}s")
        
        if bottleneck_score > 1.0:
            bottlenecks.append({
                "node_id": node_id,
                "bottleneck_score": bottleneck_score,
                "reasons": reasons,
                "incoming_count": incoming_count,
                "outgoing_count": outgoing_count,
                "estimated_duration": estimated_duration
            })
    
    return sorted(bottlenecks, key=lambda x: x["bottleneck_score"], reverse=True)


def find_parallel_opportunities(dependency_graph: Dict[str, Any], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find opportunities for parallel execution."""
    opportunities = []
    
    dependency_levels = dependency_graph["dependency_levels"]
    resource_dependencies = dependency_graph["resource_dependencies"]
    
    # Group nodes by dependency level
    level_groups = {}
    for node_id, level in dependency_levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node_id)
    
    # Find parallel opportunities within each level
    for level, node_ids in level_groups.items():
        if len(node_ids) > 1:
            # Check for resource conflicts
            conflicting_groups = []
            for resource_conflict, conflicting_nodes in resource_dependencies.items():
                conflict_in_level = [n for n in node_ids if n in conflicting_nodes]
                if len(conflict_in_level) > 1:
                    conflicting_groups.append(conflict_in_level)
            
            # Create parallel groups avoiding conflicts
            if not conflicting_groups:
                # No conflicts, all nodes can run in parallel
                opportunities.append({
                    "type": "level_parallelization",
                    "level": level,
                    "parallel_nodes": node_ids,
                    "estimated_speedup": len(node_ids),
                    "conflicts": []
                })
            else:
                # Create conflict-free groups
                remaining_nodes = set(node_ids)
                parallel_groups = []
                
                for conflict_group in conflicting_groups:
                    # Remove conflicting nodes from remaining
                    remaining_nodes -= set(conflict_group)
                    # Add individual nodes from conflict group
                    for node in conflict_group:
                        parallel_groups.append([node])
                
                # Add remaining nodes as a parallel group
                if remaining_nodes:
                    parallel_groups.append(list(remaining_nodes))
                
                for i, group in enumerate(parallel_groups):
                    if len(group) > 1:
                        opportunities.append({
                            "type": "conflict_aware_parallelization",
                            "level": level,
                            "group_id": i,
                            "parallel_nodes": group,
                            "estimated_speedup": len(group),
                            "conflicts": [c for c in conflicting_groups if any(n in group for n in c)]
                        })
    
    return opportunities


def optimize_execution_order(
    dependency_graph: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    optimization_goals: List[str]
) -> List[Dict[str, Any]]:
    """Optimize execution order based on goals."""
    adjacency_list = dependency_graph["adjacency_list"]
    dependency_levels = dependency_graph["dependency_levels"]
    
    # Create execution phases based on dependency levels
    phases = {}
    for node_id, level in dependency_levels.items():
        if level not in phases:
            phases[level] = []
        phases[level].append(node_id)
    
    optimized_order = []
    
    for level in sorted(phases.keys()):
        level_nodes = phases[level]
        
        if "speed" in optimization_goals:
            # Optimize for speed - prioritize nodes with most dependents
            level_nodes.sort(key=lambda n: len(adjacency_list.get(n, [])), reverse=True)
        
        if "resource" in optimization_goals:
            # Optimize for resources - group by resource requirements
            node_resources = {}
            for node_id in level_nodes:
                node = next(n for n in nodes if n["id"] == node_id)
                requirements = node.get("metadata", {}).get("resource_requirements", {})
                resource_key = tuple(sorted(requirements.items()))
                if resource_key not in node_resources:
                    node_resources[resource_key] = []
                node_resources[resource_key].append(node_id)
            
            # Flatten grouped nodes
            level_nodes = [node for group in node_resources.values() for node in group]
        
        optimized_order.append({
            "phase": level,
            "nodes": level_nodes,
            "execution_type": "parallel" if len(level_nodes) > 1 else "sequential",
            "estimated_duration": max(
                next(n for n in nodes if n["id"] == node_id).get("metadata", {}).get("estimated_duration", 0)
                for node_id in level_nodes
            ) if level_nodes else 0
        })
    
    return optimized_order


def calculate_critical_path(dependency_graph: Dict[str, Any], nodes: List[Dict[str, Any]]) -> List[str]:
    """Calculate the critical path through the workflow."""
    adjacency_list = dependency_graph["adjacency_list"]
    
    # Create node duration map
    node_durations = {}
    for node in nodes:
        node_durations[node["id"]] = node.get("metadata", {}).get("estimated_duration", 0)
    
    # Find longest path using topological sort and dynamic programming
    # First, find nodes with no dependencies
    all_nodes = set(node["id"] for node in nodes)
    dependent_nodes = set()
    for deps in adjacency_list.values():
        for dep in deps:
            dependent_nodes.add(dep["target"])
    
    start_nodes = all_nodes - dependent_nodes
    
    # Calculate longest path from each start node
    longest_paths = {}
    
    def calculate_longest_path(node_id, visited=None):
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return 0, []  # Cycle detected
        
        if node_id in longest_paths:
            return longest_paths[node_id]
        
        visited.add(node_id)
        
        max_path_length = 0
        max_path = []
        
        for dep in adjacency_list.get(node_id, []):
            target = dep["target"]
            path_length, path = calculate_longest_path(target, visited.copy())
            total_length = node_durations[node_id] + path_length
            
            if total_length > max_path_length:
                max_path_length = total_length
                max_path = [node_id] + path
        
        if not max_path:
            max_path = [node_id]
            max_path_length = node_durations[node_id]
        
        longest_paths[node_id] = (max_path_length, max_path)
        return max_path_length, max_path
    
    # Find the overall critical path
    critical_path = []
    max_duration = 0
    
    for start_node in start_nodes:
        duration, path = calculate_longest_path(start_node)
        if duration > max_duration:
            max_duration = duration
            critical_path = path
    
    return critical_path


def generate_optimization_recommendations(
    dependency_graph: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    parallel_opportunities: List[Dict[str, Any]],
    optimization_goals: List[str]
) -> List[Dict[str, Any]]:
    """Generate optimization recommendations."""
    recommendations = []
    
    # Bottleneck recommendations
    if bottlenecks:
        top_bottleneck = bottlenecks[0]
        recommendations.append({
            "type": "bottleneck_optimization",
            "priority": "high",
            "description": f"Optimize bottleneck node {top_bottleneck['node_id']}",
            "actions": [
                "Consider breaking down the task into smaller components",
                "Optimize resource allocation",
                "Add caching or preprocessing"
            ],
            "expected_impact": "Reduce overall execution time by 20-40%"
        })
    
    # Parallelization recommendations
    if parallel_opportunities:
        total_speedup = sum(opp["estimated_speedup"] for opp in parallel_opportunities)
        recommendations.append({
            "type": "parallelization",
            "priority": "medium",
            "description": f"Implement parallel execution for {len(parallel_opportunities)} opportunity groups",
            "actions": [
                "Configure parallel execution for identified node groups",
                "Ensure adequate resource allocation",
                "Add synchronization points"
            ],
            "expected_impact": f"Potential speedup of {total_speedup:.1f}x"
        })
    
    # Resource optimization recommendations
    if "resource" in optimization_goals:
        recommendations.append({
            "type": "resource_optimization",
            "priority": "medium",
            "description": "Optimize resource usage across workflow",
            "actions": [
                "Balance resource requirements across parallel tasks",
                "Implement resource pooling",
                "Add resource monitoring"
            ],
            "expected_impact": "Reduce resource contention and improve stability"
        })
    
    # Reliability recommendations
    if "reliability" in optimization_goals:
        recommendations.append({
            "type": "reliability_improvement",
            "priority": "low",
            "description": "Improve workflow reliability and error handling",
            "actions": [
                "Add checkpointing for long-running tasks",
                "Implement retry logic",
                "Add monitoring and alerting"
            ],
            "expected_impact": "Improve workflow success rate and recovery"
        })
    
    return recommendations


# Create the DependencyAnalyzerAgent
dependency_analyzer_agent = Agent(
    model=settings.root_agent_model,
    name="dependency_analyzer_agent",
    instruction="""
You are the DependencyAnalyzerAgent, a specialized sub-agent focused on analyzing workflow dependencies 
and identifying optimization opportunities.

Your core responsibilities:
1. Map complex workflow dependencies and relationships
2. Identify bottlenecks and performance constraints
3. Find parallel execution opportunities
4. Optimize execution order for maximum efficiency
5. Generate actionable optimization recommendations

Use the analyze_workflow_dependencies tool to perform comprehensive dependency analysis.
""",
    tools=[analyze_workflow_dependencies],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=4096
    )
) 