

"""CompatibilityCheckerAgent - Sub-agent for service compatibility analysis."""

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


async def check_service_compatibility(
    service_spec: Dict[str, Any],
    integration_requirements: Dict[str, Any] = None,
    compatibility_mode: str = "comprehensive",
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Analyze service compatibility for integration feasibility.
    
    Args:
        service_spec: Service specification to analyze
        integration_requirements: Integration requirements
        compatibility_mode: Analysis mode (quick, comprehensive, deep)
        tool_context: ADK tool context
    
    Returns:
        Dict containing compatibility analysis result
    """
    try:
        check_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        if not integration_requirements:
            integration_requirements = {
                "protocols": ["HTTP", "WebSocket"],
                "data_formats": ["JSON", "XML"],
                "authentication": ["OAuth2", "API_Key"],
                "performance": {"max_latency": 1000, "min_throughput": 100}
            }
        
        # Analyze different compatibility aspects
        compatibility_results = {
            "check_id": check_id,
            "timestamp": timestamp,
            "service_name": service_spec.get("name", "unknown"),
            "compatibility_mode": compatibility_mode,
            "protocol_compatibility": analyze_protocol_compatibility(service_spec, integration_requirements),
            "data_format_compatibility": analyze_data_format_compatibility(service_spec, integration_requirements),
            "authentication_compatibility": analyze_auth_compatibility(service_spec, integration_requirements),
            "performance_compatibility": analyze_performance_compatibility(service_spec, integration_requirements),
            "feature_compatibility": analyze_feature_compatibility(service_spec, integration_requirements)
        }
        
        # Calculate overall compatibility score
        compatibility_results["overall_score"] = calculate_overall_compatibility_score(compatibility_results)
        compatibility_results["recommendations"] = generate_compatibility_recommendations(compatibility_results)
        compatibility_results["integration_effort"] = estimate_integration_effort(compatibility_results)
        
        # Store compatibility analysis
        await supabase_manager.insert_record(
            "compatibility_checks",
            {
                "id": check_id,
                "service_name": service_spec.get("name"),
                "compatibility_result": compatibility_results,
                "timestamp": timestamp
            }
        )
        
        logger.info(f"Completed compatibility check {check_id} with score {compatibility_results['overall_score']}")
        return compatibility_results
        
    except Exception as e:
        logger.error(f"Error in compatibility check: {e}")
        return {"error": str(e), "check_id": check_id}


def analyze_protocol_compatibility(service_spec: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze protocol compatibility."""
    
    service_protocols = service_spec.get("protocols", [])
    required_protocols = requirements.get("protocols", [])
    
    compatible_protocols = list(set(service_protocols) & set(required_protocols))
    incompatible_protocols = list(set(required_protocols) - set(service_protocols))
    
    compatibility_score = len(compatible_protocols) / max(len(required_protocols), 1)
    
    return {
        "score": compatibility_score,
        "compatible_protocols": compatible_protocols,
        "incompatible_protocols": incompatible_protocols,
        "service_protocols": service_protocols,
        "adaptation_needed": len(incompatible_protocols) > 0
    }


def analyze_data_format_compatibility(service_spec: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data format compatibility."""
    
    service_formats = service_spec.get("data_formats", [])
    required_formats = requirements.get("data_formats", [])
    
    compatible_formats = list(set(service_formats) & set(required_formats))
    incompatible_formats = list(set(required_formats) - set(service_formats))
    
    compatibility_score = len(compatible_formats) / max(len(required_formats), 1)
    
    return {
        "score": compatibility_score,
        "compatible_formats": compatible_formats,
        "incompatible_formats": incompatible_formats,
        "service_formats": service_formats,
        "transformation_needed": len(incompatible_formats) > 0
    }


def analyze_auth_compatibility(service_spec: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze authentication compatibility."""
    
    service_auth = service_spec.get("authentication", [])
    required_auth = requirements.get("authentication", [])
    
    compatible_auth = list(set(service_auth) & set(required_auth))
    incompatible_auth = list(set(required_auth) - set(service_auth))
    
    compatibility_score = len(compatible_auth) / max(len(required_auth), 1)
    
    return {
        "score": compatibility_score,
        "compatible_methods": compatible_auth,
        "incompatible_methods": incompatible_auth,
        "service_methods": service_auth,
        "auth_adaptation_needed": len(incompatible_auth) > 0
    }


def analyze_performance_compatibility(service_spec: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance compatibility."""
    
    service_perf = service_spec.get("performance", {})
    required_perf = requirements.get("performance", {})
    
    latency_compatible = True
    throughput_compatible = True
    
    if "max_latency" in required_perf and "latency" in service_perf:
        latency_compatible = service_perf["latency"] <= required_perf["max_latency"]
    
    if "min_throughput" in required_perf and "throughput" in service_perf:
        throughput_compatible = service_perf["throughput"] >= required_perf["min_throughput"]
    
    # Calculate performance score
    performance_score = (
        (1.0 if latency_compatible else 0.5) + 
        (1.0 if throughput_compatible else 0.5)
    ) / 2
    
    return {
        "score": performance_score,
        "latency_compatible": latency_compatible,
        "throughput_compatible": throughput_compatible,
        "service_performance": service_perf,
        "required_performance": required_perf,
        "performance_optimization_needed": not (latency_compatible and throughput_compatible)
    }


def analyze_feature_compatibility(service_spec: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze feature compatibility."""
    
    service_features = service_spec.get("features", [])
    required_features = requirements.get("features", [])
    
    compatible_features = list(set(service_features) & set(required_features))
    missing_features = list(set(required_features) - set(service_features))
    
    compatibility_score = len(compatible_features) / max(len(required_features), 1) if required_features else 1.0
    
    return {
        "score": compatibility_score,
        "compatible_features": compatible_features,
        "missing_features": missing_features,
        "service_features": service_features,
        "feature_gap_exists": len(missing_features) > 0
    }


def calculate_overall_compatibility_score(results: Dict[str, Any]) -> float:
    """Calculate overall compatibility score."""
    
    scores = []
    weights = {
        "protocol_compatibility": 0.3,
        "data_format_compatibility": 0.25,
        "authentication_compatibility": 0.2,
        "performance_compatibility": 0.15,
        "feature_compatibility": 0.1
    }
    
    for aspect, weight in weights.items():
        if aspect in results and "score" in results[aspect]:
            scores.append(results[aspect]["score"] * weight)
    
    return sum(scores)


def generate_compatibility_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on compatibility analysis."""
    
    recommendations = []
    
    # Protocol recommendations
    if results["protocol_compatibility"]["adaptation_needed"]:
        recommendations.append("Implement protocol adapters for unsupported protocols")
    
    # Data format recommendations
    if results["data_format_compatibility"]["transformation_needed"]:
        recommendations.append("Add data transformation layer for format conversion")
    
    # Authentication recommendations
    if results["authentication_compatibility"]["auth_adaptation_needed"]:
        recommendations.append("Implement authentication wrapper or proxy")
    
    # Performance recommendations
    if results["performance_compatibility"]["performance_optimization_needed"]:
        recommendations.append("Consider performance optimization or caching layer")
    
    # Feature recommendations
    if results["feature_compatibility"]["feature_gap_exists"]:
        recommendations.append("Evaluate alternative services or implement missing features")
    
    # Overall score recommendations
    overall_score = results["overall_score"]
    if overall_score < 0.5:
        recommendations.append("Consider alternative services due to low compatibility")
    elif overall_score < 0.8:
        recommendations.append("Plan for significant integration effort and testing")
    
    return recommendations


def estimate_integration_effort(results: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate integration effort based on compatibility analysis."""
    
    effort_points = 0
    
    # Add effort for each incompatibility
    if results["protocol_compatibility"]["adaptation_needed"]:
        effort_points += len(results["protocol_compatibility"]["incompatible_protocols"]) * 3
    
    if results["data_format_compatibility"]["transformation_needed"]:
        effort_points += len(results["data_format_compatibility"]["incompatible_formats"]) * 2
    
    if results["authentication_compatibility"]["auth_adaptation_needed"]:
        effort_points += len(results["authentication_compatibility"]["incompatible_methods"]) * 2
    
    if results["performance_compatibility"]["performance_optimization_needed"]:
        effort_points += 3
    
    if results["feature_compatibility"]["feature_gap_exists"]:
        effort_points += len(results["feature_compatibility"]["missing_features"]) * 1
    
    # Convert effort points to time estimates
    if effort_points <= 5:
        effort_level = "low"
        estimated_days = "1-3 days"
    elif effort_points <= 15:
        effort_level = "medium"
        estimated_days = "1-2 weeks"
    elif effort_points <= 30:
        effort_level = "high"
        estimated_days = "2-4 weeks"
    else:
        effort_level = "very_high"
        estimated_days = "4+ weeks"
    
    return {
        "effort_points": effort_points,
        "effort_level": effort_level,
        "estimated_duration": estimated_days,
        "complexity_factors": [
            "Protocol adaptation" if results["protocol_compatibility"]["adaptation_needed"] else None,
            "Data transformation" if results["data_format_compatibility"]["transformation_needed"] else None,
            "Authentication wrapper" if results["authentication_compatibility"]["auth_adaptation_needed"] else None,
            "Performance optimization" if results["performance_compatibility"]["performance_optimization_needed"] else None,
            "Feature implementation" if results["feature_compatibility"]["feature_gap_exists"] else None
        ]
    }


# Create the CompatibilityCheckerAgent
compatibility_checker_agent = Agent(
    model=settings.root_agent_model,
    name="compatibility_checker_agent",
    instruction="""
You are the CompatibilityCheckerAgent, a specialized sub-agent focused on analyzing 
service compatibility and integration feasibility.

Your core responsibilities:
1. Analyze API specifications and service capabilities
2. Check protocol and data format compatibility
3. Assess authentication and security compatibility
4. Evaluate performance and scalability requirements
5. Provide integration effort estimates and recommendations

Use the check_service_compatibility tool to perform comprehensive compatibility analysis.
""",
    tools=[check_service_compatibility],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=4096
    )
) 