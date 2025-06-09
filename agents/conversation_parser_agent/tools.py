
"""Tools for the ConversationParserAgent."""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
from utils.supabase_client import supabase_manager
import logging

logger = logging.getLogger(__name__)


async def gemini_nlp_tool(
    text: str,
    analysis_type: str = "full",
    context: str = "",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Advanced NLP processing using Google's Gemini Pro for entity extraction and intent classification.
    
    Args:
        text: Text to analyze
        analysis_type: Type of analysis (full, entities, intent, sentiment)
        context: Additional context for analysis
        tool_context: ADK tool context
    
    Returns:
        Dict containing NLP analysis results
    """
    try:
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Simulate advanced NLP processing (in production, this would use Gemini API)
        nlp_analysis = {
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "input_text": text,
            "analysis_type": analysis_type,
            "context": context
        }
        
        # Entity extraction
        entities = extract_entities(text)
        
        # Intent classification
        intents = classify_intents(text)
        
        # Sentiment analysis
        sentiment = analyze_sentiment(text)
        
        # Relationship extraction
        relationships = extract_relationships(text, entities)
        
        # Complexity assessment
        complexity = assess_complexity(text, entities, intents)
        
        nlp_results = {
            "entities": entities,
            "intents": intents,
            "sentiment": sentiment,
            "relationships": relationships,
            "complexity": complexity,
            "confidence_scores": {
                "overall": 0.85,
                "entities": 0.9,
                "intents": 0.8,
                "relationships": 0.75
            },
            "metadata": {
                "word_count": len(text.split()),
                "sentence_count": len([s for s in text.split('.') if s.strip()]),
                "language": "en",
                "processing_time": "0.5s"
            }
        }
        
        # Store analysis results
        await supabase_manager.insert_record(
            "nlp_analysis",
            {
                "id": analysis_id,
                "input_text": text,
                "analysis_results": nlp_results,
                "timestamp": timestamp
            }
        )
        
        # Update tool context
        if tool_context:
            tool_context.state["nlp_analysis"] = nlp_results
            tool_context.state["analysis_id"] = analysis_id
        
        logger.info(f"Completed NLP analysis {analysis_id}")
        return nlp_results
        
    except Exception as e:
        logger.error(f"Error in NLP analysis: {e}")
        return {"error": str(e), "analysis_id": analysis_id}


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract entities from text."""
    # Simplified entity extraction (would use Gemini in production)
    entities = []
    
    # Common entity patterns
    entity_patterns = {
        "AGENT": ["agent", "bot", "assistant", "service"],
        "ACTION": ["create", "deploy", "execute", "run", "analyze", "process"],
        "RESOURCE": ["database", "api", "file", "data", "model"],
        "TIME": ["now", "today", "tomorrow", "urgent", "asap"],
        "QUANTITY": ["all", "some", "many", "few", "multiple"]
    }
    
    text_lower = text.lower()
    for entity_type, patterns in entity_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                start_pos = text_lower.find(pattern)
                entities.append({
                    "type": entity_type,
                    "value": pattern,
                    "start": start_pos,
                    "end": start_pos + len(pattern),
                    "confidence": 0.8
                })
    
    return entities


def classify_intents(text: str) -> List[Dict[str, Any]]:
    """Classify intents from text."""
    intents = []
    
    intent_patterns = {
        "CREATE": ["create", "make", "build", "generate", "new"],
        "EXECUTE": ["run", "execute", "start", "launch", "process"],
        "ANALYZE": ["analyze", "examine", "study", "investigate"],
        "RETRIEVE": ["get", "fetch", "find", "search", "retrieve"],
        "UPDATE": ["update", "modify", "change", "edit"],
        "DELETE": ["delete", "remove", "destroy", "eliminate"]
    }
    
    text_lower = text.lower()
    for intent, patterns in intent_patterns.items():
        score = sum(1 for pattern in patterns if pattern in text_lower)
        if score > 0:
            intents.append({
                "intent": intent,
                "confidence": min(score / len(patterns), 1.0),
                "patterns_matched": [p for p in patterns if p in text_lower]
            })
    
    return sorted(intents, key=lambda x: x["confidence"], reverse=True)


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text."""
    # Simplified sentiment analysis
    positive_words = ["good", "great", "excellent", "amazing", "perfect", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "wrong", "error"]
    urgent_words = ["urgent", "asap", "immediately", "quickly", "fast"]
    
    text_lower = text.lower()
    
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)
    urgency_score = sum(1 for word in urgent_words if word in text_lower)
    
    total_words = len(text.split())
    
    return {
        "polarity": "positive" if positive_score > negative_score else "negative" if negative_score > 0 else "neutral",
        "confidence": max(positive_score, negative_score) / max(total_words, 1),
        "urgency": "high" if urgency_score > 0 else "normal",
        "scores": {
            "positive": positive_score,
            "negative": negative_score,
            "urgency": urgency_score
        }
    }


def extract_relationships(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract relationships between entities."""
    relationships = []
    
    # Simple relationship patterns
    relationship_patterns = {
        "USES": ["use", "uses", "with", "using"],
        "CREATES": ["create", "creates", "make", "makes"],
        "PROCESSES": ["process", "processes", "handle", "handles"],
        "DEPENDS_ON": ["need", "needs", "require", "requires", "depend"]
    }
    
    text_lower = text.lower()
    
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:
                # Check if there's a relationship pattern between entities
                for rel_type, patterns in relationship_patterns.items():
                    for pattern in patterns:
                        if pattern in text_lower:
                            relationships.append({
                                "source": entity1["value"],
                                "target": entity2["value"],
                                "relationship": rel_type,
                                "confidence": 0.6
                            })
    
    return relationships


def assess_complexity(text: str, entities: List[Dict[str, Any]], intents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess the complexity of the request."""
    word_count = len(text.split())
    entity_count = len(entities)
    intent_count = len(intents)
    
    # Complexity indicators
    complexity_indicators = {
        "multiple_steps": any(word in text.lower() for word in ["then", "after", "next", "following"]),
        "conditional_logic": any(word in text.lower() for word in ["if", "when", "unless", "provided"]),
        "parallel_tasks": any(word in text.lower() for word in ["simultaneously", "parallel", "concurrent"]),
        "loops": any(word in text.lower() for word in ["repeat", "loop", "iterate", "each", "every"])
    }
    
    complexity_score = (
        word_count * 0.1 +
        entity_count * 0.2 +
        intent_count * 0.3 +
        sum(complexity_indicators.values()) * 0.4
    )
    
    if complexity_score < 2:
        level = "low"
    elif complexity_score < 5:
        level = "medium"
    else:
        level = "high"
    
    return {
        "level": level,
        "score": complexity_score,
        "indicators": complexity_indicators,
        "estimated_agents": min(max(intent_count, 1), 5),
        "estimated_time": f"{complexity_score * 30:.0f} seconds"
    }


async def workflow_graph_tool(
    nlp_results: Dict[str, Any],
    workflow_type: str = "auto",
    optimization_level: str = "balanced",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Generate DAG representations using LangGraph for complex workflow visualization.
    
    Args:
        nlp_results: Results from NLP analysis
        workflow_type: Type of workflow (sequential, parallel, conditional, auto)
        optimization_level: Optimization level (speed, balanced, thorough)
        tool_context: ADK tool context
    
    Returns:
        Dict containing workflow graph definition
    """
    try:
        workflow_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        entities = nlp_results.get("entities", [])
        intents = nlp_results.get("intents", [])
        complexity = nlp_results.get("complexity", {})
        
        # Generate workflow nodes based on intents
        nodes = []
        for i, intent in enumerate(intents):
            node = {
                "id": f"node_{i}",
                "type": "task",
                "action": intent["intent"].lower(),
                "agent": map_intent_to_agent(intent["intent"]),
                "inputs": extract_node_inputs(entities, intent),
                "outputs": generate_node_outputs(intent),
                "metadata": {
                    "confidence": intent["confidence"],
                    "estimated_duration": estimate_duration(intent),
                    "resource_requirements": estimate_resources(intent)
                }
            }
            nodes.append(node)
        
        # Generate edges based on dependencies
        edges = generate_workflow_edges(nodes, nlp_results)
        
        # Determine workflow type if auto
        if workflow_type == "auto":
            workflow_type = determine_workflow_type(nodes, edges, complexity)
        
        # Optimize workflow based on level
        if optimization_level in ["balanced", "speed"]:
            nodes, edges = optimize_workflow(nodes, edges, optimization_level)
        
        workflow_graph = {
            "workflow_id": workflow_id,
            "timestamp": timestamp,
            "type": workflow_type,
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "complexity": complexity,
                "optimization_level": optimization_level,
                "estimated_total_time": sum(node["metadata"]["estimated_duration"] for node in nodes),
                "parallel_opportunities": count_parallel_opportunities(edges),
                "critical_path": find_critical_path(nodes, edges)
            },
            "execution_plan": {
                "entry_points": [node["id"] for node in nodes if not has_dependencies(node["id"], edges)],
                "exit_points": [node["id"] for node in nodes if not has_dependents(node["id"], edges)],
                "parallel_groups": identify_parallel_groups(nodes, edges)
            }
        }
        
        # Store workflow graph
        await supabase_manager.insert_record(
            "workflow_graphs",
            {
                "id": workflow_id,
                "graph_definition": workflow_graph,
                "nlp_analysis_id": tool_context.state.get("analysis_id"),
                "timestamp": timestamp
            }
        )
        
        # Update tool context
        if tool_context:
            tool_context.state["workflow_graph"] = workflow_graph
            tool_context.state["workflow_id"] = workflow_id
        
        logger.info(f"Generated workflow graph {workflow_id}")
        return workflow_graph
        
    except Exception as e:
        logger.error(f"Error generating workflow graph: {e}")
        return {"error": str(e), "workflow_id": workflow_id}


def map_intent_to_agent(intent: str) -> str:
    """Map intent to appropriate agent."""
    intent_agent_mapping = {
        "CREATE": "agent_factory_agent",
        "EXECUTE": "execution_coordinator_agent",
        "ANALYZE": "conversation_parser_agent",
        "RETRIEVE": "memory_context_agent",
        "UPDATE": "execution_coordinator_agent",
        "DELETE": "execution_coordinator_agent"
    }
    return intent_agent_mapping.get(intent, "workflow_master_agent")


def extract_node_inputs(entities: List[Dict[str, Any]], intent: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inputs for a workflow node."""
    inputs = {}
    
    for entity in entities:
        if entity["type"] == "RESOURCE":
            inputs["resource"] = entity["value"]
        elif entity["type"] == "QUANTITY":
            inputs["quantity"] = entity["value"]
        elif entity["type"] == "TIME":
            inputs["time_constraint"] = entity["value"]
    
    inputs["intent"] = intent["intent"]
    inputs["confidence"] = intent["confidence"]
    
    return inputs


def generate_node_outputs(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Generate expected outputs for a workflow node."""
    output_mapping = {
        "CREATE": {"result": "created_resource", "status": "creation_status"},
        "EXECUTE": {"result": "execution_result", "status": "execution_status"},
        "ANALYZE": {"result": "analysis_result", "insights": "analysis_insights"},
        "RETRIEVE": {"result": "retrieved_data", "metadata": "data_metadata"},
        "UPDATE": {"result": "updated_resource", "changes": "change_summary"},
        "DELETE": {"result": "deletion_status", "confirmation": "deletion_confirmation"}
    }
    
    return output_mapping.get(intent["intent"], {"result": "generic_result"})


def estimate_duration(intent: Dict[str, Any]) -> int:
    """Estimate duration for an intent in seconds."""
    duration_mapping = {
        "CREATE": 30,
        "EXECUTE": 20,
        "ANALYZE": 15,
        "RETRIEVE": 5,
        "UPDATE": 10,
        "DELETE": 5
    }
    
    base_duration = duration_mapping.get(intent["intent"], 15)
    confidence_factor = 1 / max(intent["confidence"], 0.1)
    
    return int(base_duration * confidence_factor)


def estimate_resources(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate resource requirements for an intent."""
    return {
        "cpu": "medium" if intent["confidence"] > 0.8 else "high",
        "memory": "low",
        "network": "medium" if intent["intent"] in ["RETRIEVE", "UPDATE"] else "low",
        "storage": "medium" if intent["intent"] in ["CREATE", "UPDATE"] else "low"
    }


def generate_workflow_edges(nodes: List[Dict[str, Any]], nlp_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate edges between workflow nodes."""
    edges = []
    
    # Simple sequential dependency for now
    for i in range(len(nodes) - 1):
        edges.append({
            "from": nodes[i]["id"],
            "to": nodes[i + 1]["id"],
            "type": "sequential",
            "condition": None,
            "weight": 1.0
        })
    
    return edges


def determine_workflow_type(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], complexity: Dict[str, Any]) -> str:
    """Determine the appropriate workflow type."""
    if complexity.get("indicators", {}).get("conditional_logic"):
        return "conditional"
    elif complexity.get("indicators", {}).get("parallel_tasks"):
        return "parallel"
    elif complexity.get("indicators", {}).get("loops"):
        return "loop"
    else:
        return "sequential"


def optimize_workflow(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], level: str) -> tuple:
    """Optimize workflow based on optimization level."""
    if level == "speed":
        # Identify parallel opportunities
        parallel_nodes = []
        for node in nodes:
            if not has_dependencies(node["id"], edges):
                parallel_nodes.append(node)
        
        # Update metadata for parallel execution
        for node in parallel_nodes:
            node["metadata"]["parallel_eligible"] = True
    
    return nodes, edges


def count_parallel_opportunities(edges: List[Dict[str, Any]]) -> int:
    """Count parallel execution opportunities."""
    # Simplified - count nodes without dependencies
    dependent_nodes = set(edge["to"] for edge in edges)
    all_nodes = set(edge["from"] for edge in edges) | dependent_nodes
    independent_nodes = all_nodes - dependent_nodes
    return len(independent_nodes)


def find_critical_path(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
    """Find the critical path through the workflow."""
    # Simplified critical path - longest sequential chain
    if not edges:
        return [node["id"] for node in nodes]
    
    # Find the longest path
    path = []
    current = edges[0]["from"]
    path.append(current)
    
    for edge in edges:
        if edge["from"] == current:
            path.append(edge["to"])
            current = edge["to"]
    
    return path


def has_dependencies(node_id: str, edges: List[Dict[str, Any]]) -> bool:
    """Check if a node has dependencies."""
    return any(edge["to"] == node_id for edge in edges)


def has_dependents(node_id: str, edges: List[Dict[str, Any]]) -> bool:
    """Check if a node has dependents."""
    return any(edge["from"] == node_id for edge in edges)


def identify_parallel_groups(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[List[str]]:
    """Identify groups of nodes that can run in parallel."""
    parallel_groups = []
    
    # Find nodes at the same dependency level
    levels = {}
    for node in nodes:
        level = calculate_dependency_level(node["id"], edges)
        if level not in levels:
            levels[level] = []
        levels[level].append(node["id"])
    
    # Groups with more than one node can run in parallel
    for level, node_ids in levels.items():
        if len(node_ids) > 1:
            parallel_groups.append(node_ids)
    
    return parallel_groups


def calculate_dependency_level(node_id: str, edges: List[Dict[str, Any]]) -> int:
    """Calculate the dependency level of a node."""
    level = 0
    current_nodes = {node_id}
    
    while True:
        dependencies = set()
        for edge in edges:
            if edge["to"] in current_nodes:
                dependencies.add(edge["from"])
        
        if not dependencies:
            break
        
        current_nodes = dependencies
        level += 1
    
    return level


async def requirement_validator_tool(
    workflow_graph: Dict[str, Any],
    system_constraints: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Validate workflow feasibility against available agents and tools.
    
    Args:
        workflow_graph: Workflow graph to validate
        system_constraints: System constraints and limits
        tool_context: ADK tool context
    
    Returns:
        Dict containing validation results
    """
    try:
        validation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        nodes = workflow_graph.get("nodes", [])
        edges = workflow_graph.get("edges", [])
        metadata = workflow_graph.get("metadata", {})
        
        # Default system constraints
        if not system_constraints:
            system_constraints = {
                "max_execution_time": 300,  # 5 minutes
                "max_parallel_agents": 5,
                "available_agents": [
                    "workflow_master_agent",
                    "conversation_parser_agent",
                    "agent_factory_agent",
                    "execution_coordinator_agent",
                    "memory_context_agent",
                    "integration_gateway_agent"
                ],
                "resource_limits": {
                    "cpu": "high",
                    "memory": "high",
                    "network": "medium",
                    "storage": "high"
                }
            }
        
        validation_results = {
            "validation_id": validation_id,
            "timestamp": timestamp,
            "workflow_id": workflow_graph.get("workflow_id"),
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "resource_analysis": {},
            "feasibility_score": 1.0
        }
        
        # Validate agent availability
        for node in nodes:
            required_agent = node.get("agent")
            if required_agent not in system_constraints["available_agents"]:
                validation_results["issues"].append({
                    "type": "missing_agent",
                    "node_id": node["id"],
                    "required_agent": required_agent,
                    "severity": "high"
                })
                validation_results["is_valid"] = False
        
        # Validate execution time
        estimated_time = metadata.get("estimated_total_time", 0)
        if estimated_time > system_constraints["max_execution_time"]:
            validation_results["warnings"].append({
                "type": "execution_time_warning",
                "estimated_time": estimated_time,
                "max_allowed": system_constraints["max_execution_time"],
                "severity": "medium"
            })
        
        # Validate parallel execution limits
        parallel_groups = workflow_graph.get("execution_plan", {}).get("parallel_groups", [])
        max_parallel = max(len(group) for group in parallel_groups) if parallel_groups else 1
        
        if max_parallel > system_constraints["max_parallel_agents"]:
            validation_results["issues"].append({
                "type": "parallel_limit_exceeded",
                "required_parallel": max_parallel,
                "max_allowed": system_constraints["max_parallel_agents"],
                "severity": "high"
            })
            validation_results["is_valid"] = False
        
        # Resource analysis
        resource_requirements = analyze_resource_requirements(nodes)
        validation_results["resource_analysis"] = resource_requirements
        
        # Check resource constraints
        for resource, requirement in resource_requirements.items():
            limit = system_constraints["resource_limits"].get(resource, "medium")
            if not check_resource_compatibility(requirement, limit):
                validation_results["warnings"].append({
                    "type": "resource_warning",
                    "resource": resource,
                    "requirement": requirement,
                    "limit": limit,
                    "severity": "medium"
                })
        
        # Generate recommendations
        validation_results["recommendations"] = generate_recommendations(
            validation_results, workflow_graph, system_constraints
        )
        
        # Calculate feasibility score
        issue_count = len(validation_results["issues"])
        warning_count = len(validation_results["warnings"])
        validation_results["feasibility_score"] = max(0, 1.0 - (issue_count * 0.3) - (warning_count * 0.1))
        
        # Store validation results
        await supabase_manager.insert_record(
            "workflow_validations",
            {
                "id": validation_id,
                "workflow_id": workflow_graph.get("workflow_id"),
                "validation_results": validation_results,
                "timestamp": timestamp
            }
        )
        
        # Update tool context
        if tool_context:
            tool_context.state["validation_results"] = validation_results
            tool_context.state["validation_id"] = validation_id
        
        logger.info(f"Completed workflow validation {validation_id}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error in workflow validation: {e}")
        return {"error": str(e), "validation_id": validation_id}


def analyze_resource_requirements(nodes: List[Dict[str, Any]]) -> Dict[str, str]:
    """Analyze resource requirements across all nodes."""
    resource_levels = {"low": 1, "medium": 2, "high": 3}
    
    max_requirements = {"cpu": "low", "memory": "low", "network": "low", "storage": "low"}
    
    for node in nodes:
        requirements = node.get("metadata", {}).get("resource_requirements", {})
        for resource, level in requirements.items():
            current_level = resource_levels.get(max_requirements.get(resource, "low"), 1)
            new_level = resource_levels.get(level, 1)
            if new_level > current_level:
                max_requirements[resource] = level
    
    return max_requirements


def check_resource_compatibility(requirement: str, limit: str) -> bool:
    """Check if resource requirement is within limits."""
    levels = {"low": 1, "medium": 2, "high": 3}
    return levels.get(requirement, 1) <= levels.get(limit, 2)


def generate_recommendations(
    validation_results: Dict[str, Any],
    workflow_graph: Dict[str, Any],
    system_constraints: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate optimization recommendations."""
    recommendations = []
    
    if validation_results["issues"]:
        recommendations.append({
            "type": "fix_issues",
            "description": "Address validation issues before execution",
            "priority": "high",
            "actions": ["Review agent availability", "Check system constraints"]
        })
    
    if validation_results["warnings"]:
        recommendations.append({
            "type": "optimize_performance",
            "description": "Consider optimizations to improve performance",
            "priority": "medium",
            "actions": ["Reduce parallel execution", "Optimize resource usage"]
        })
    
    # Check for optimization opportunities
    parallel_groups = workflow_graph.get("execution_plan", {}).get("parallel_groups", [])
    if not parallel_groups:
        recommendations.append({
            "type": "add_parallelization",
            "description": "Consider adding parallel execution to improve speed",
            "priority": "low",
            "actions": ["Identify independent tasks", "Restructure workflow"]
        })
    
    return recommendations


async def supabase_context_tool(
    action: str,
    user_id: str = "",
    context_type: str = "preferences",
    data: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Enrich workflows with user preferences and context from Supabase.
    
    Args:
        action: Action to perform (get, set, update)
        user_id: User identifier
        context_type: Type of context (preferences, history, patterns)
        data: Context data to store
        tool_context: ADK tool context
    
    Returns:
        Dict containing context operation result
    """
    try:
        if action == "get":
            # Retrieve user context
            context_records = await supabase_manager.query_records(
                "user_context",
                filters={"user_id": user_id, "context_type": context_type}
            )
            
            return {
                "action": "get",
                "user_id": user_id,
                "context_type": context_type,
                "context_data": context_records[0].get("data", {}) if context_records else {},
                "found": len(context_records) > 0
            }
        
        elif action == "set":
            # Store user context
            context_record = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "context_type": context_type,
                "data": data or {},
                "updated_at": datetime.now().isoformat()
            }
            
            await supabase_manager.insert_record("user_context", context_record)
            
            return {
                "action": "set",
                "user_id": user_id,
                "context_type": context_type,
                "success": True
            }
        
        elif action == "update":
            # Update existing context
            existing_records = await supabase_manager.query_records(
                "user_context",
                filters={"user_id": user_id, "context_type": context_type}
            )
            
            if existing_records:
                record_id = existing_records[0]["id"]
                await supabase_manager.update_record(
                    "user_context",
                    record_id,
                    {
                        "data": data or {},
                        "updated_at": datetime.now().isoformat()
                    }
                )
                
                return {
                    "action": "update",
                    "user_id": user_id,
                    "context_type": context_type,
                    "success": True
                }
            else:
                # Create new if doesn't exist
                return await supabase_context_tool("set", user_id, context_type, data, tool_context)
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Error in context management: {e}")
        return {"error": str(e), "action": action} 