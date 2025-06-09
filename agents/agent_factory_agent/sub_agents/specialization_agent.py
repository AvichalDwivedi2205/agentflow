

"""SpecializationAgent - Sub-agent for domain-specific agent configurations."""

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


async def create_specialized_configuration(
    domain: str,
    requirements: Dict[str, Any],
    base_template: Dict[str, Any] = None,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Create specialized agent configuration for specific domains.
    
    Args:
        domain: Target domain (finance, healthcare, retail, etc.)
        requirements: Domain-specific requirements
        base_template: Base template to customize
        tool_context: ADK tool context
    
    Returns:
        Dict containing specialized configuration
    """
    try:
        config_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        if not base_template:
            base_template = get_base_template()
        
        # Generate domain-specific configurations
        specialized_config = {
            "config_id": config_id,
            "timestamp": timestamp,
            "domain": domain,
            "requirements": requirements,
            "base_template": base_template
        }
        
        # Apply domain specializations
        specialized_config["instructions"] = generate_domain_instructions(domain, requirements)
        specialized_config["tools"] = select_domain_tools(domain, requirements)
        specialized_config["performance_config"] = optimize_for_domain(domain, requirements)
        specialized_config["validation_rules"] = create_domain_validation(domain, requirements)
        specialized_config["monitoring_config"] = setup_domain_monitoring(domain, requirements)
        
        # Store specialized configuration
        await supabase_manager.insert_record(
            "specialized_configurations",
            {
                "id": config_id,
                "domain": domain,
                "configuration": specialized_config,
                "timestamp": timestamp
            }
        )
        
        # Update tool context
        if tool_context:
            tool_context.state["specialized_config"] = specialized_config
            tool_context.state["specialization_id"] = config_id
        
        logger.info(f"Created specialized configuration {config_id} for domain {domain}")
        return specialized_config
        
    except Exception as e:
        logger.error(f"Error creating specialized configuration: {e}")
        return {"error": str(e), "config_id": config_id}


def get_base_template() -> Dict[str, Any]:
    """Get base template for specialization."""
    return {
        "name": "specialized_agent",
        "model": settings.root_agent_model,
        "instruction": "You are a specialized AI agent.",
        "tools": [],
        "generate_content_config": {
            "temperature": settings.agent_temperature,
            "max_output_tokens": settings.max_tokens
        }
    }


def generate_domain_instructions(domain: str, requirements: Dict[str, Any]) -> str:
    """Generate domain-specific instructions."""
    
    domain_instructions = {
        "finance": f"""
You are a specialized Financial AI Agent designed for {requirements.get('use_case', 'financial analysis')}.

Core Responsibilities:
- Financial data analysis and risk assessment
- Regulatory compliance (SOX, Basel III, GDPR, MiFID II)
- Investment portfolio optimization
- Market trend analysis and forecasting
- Credit risk evaluation and fraud detection

Operational Guidelines:
- Always provide confidence intervals for numerical predictions
- Include risk assessments with all recommendations
- Maintain detailed audit trails for regulatory compliance
- Follow financial calculation standards (GAAP, IFRS)
- Implement appropriate data lineage and governance

Security Requirements:
- Ensure PCI DSS compliance for payment data
- Implement strong data encryption and access controls
- Follow KYC (Know Your Customer) and AML (Anti-Money Laundering) procedures
- Maintain comprehensive activity logging

Performance Standards:
- Response time: <2 seconds for standard queries
- Accuracy: >99% for numerical calculations
- Availability: 99.9% uptime SLA
- Compliance: 100% adherence to regulatory requirements
""",

        "healthcare": f"""
You are a specialized Healthcare AI Agent designed for {requirements.get('use_case', 'clinical decision support')}.

Core Responsibilities:
- Clinical data analysis and patient care optimization
- Medical diagnosis assistance and treatment recommendations
- Drug interaction checking and dosage optimization
- Medical imaging analysis and interpretation
- Health outcomes prediction and risk stratification

Operational Guidelines:
- Follow evidence-based medicine principles
- Provide confidence scores for all medical recommendations
- Include appropriate medical disclaimers and limitations
- Maintain patient privacy and confidentiality
- Support clinical workflow integration

Compliance Requirements:
- HIPAA compliance for patient data protection
- FDA regulations for medical device software
- Clinical guidelines (HL7, FHIR standards)
- Medical coding standards (ICD-10, CPT, SNOMED)
- Quality assurance and validation protocols

Safety Protocols:
- Never provide definitive medical diagnoses
- Always recommend consulting healthcare professionals
- Include contraindications and side effects
- Implement medication safety checks
- Maintain detailed clinical decision logs
""",

        "retail": f"""
You are a specialized Retail AI Agent designed for {requirements.get('use_case', 'customer experience optimization')}.

Core Responsibilities:
- Customer behavior analysis and segmentation
- Inventory optimization and demand forecasting
- Price optimization and promotion management
- Supply chain optimization and logistics
- Personalized recommendation systems

Operational Guidelines:
- Focus on customer experience and satisfaction
- Provide actionable business insights and KPIs
- Consider seasonal patterns and market trends
- Optimize for conversion and revenue growth
- Maintain real-time inventory accuracy

Performance Metrics:
- Customer satisfaction scores
- Conversion rate optimization
- Inventory turnover rates
- Revenue per customer
- Supply chain efficiency

Technology Integration:
- E-commerce platform APIs
- Payment processing systems
- CRM and marketing automation
- Analytics and business intelligence
- Mobile and omnichannel support
""",

        "manufacturing": f"""
You are a specialized Manufacturing AI Agent designed for {requirements.get('use_case', 'production optimization')}.

Core Responsibilities:
- Production planning and optimization
- Quality control and defect prediction
- Predictive maintenance and equipment monitoring
- Supply chain and logistics optimization
- Safety compliance and risk management

Operational Guidelines:
- Focus on operational efficiency and quality metrics
- Consider safety protocols and regulatory requirements
- Provide predictive insights for maintenance
- Optimize resource allocation and scheduling
- Maintain comprehensive production records

Quality Standards:
- Six Sigma quality methodologies
- ISO 9001 quality management
- Lean manufacturing principles
- Statistical process control
- Continuous improvement processes

Safety Requirements:
- OSHA compliance and safety protocols
- Environmental regulations (EPA, ISO 14001)
- Worker safety and ergonomics
- Hazardous material handling
- Emergency response procedures
"""
    }
    
    base_instruction = domain_instructions.get(domain, f"""
You are a specialized AI agent for {domain} operations.
Focus on domain-specific best practices and industry standards.
Provide accurate, relevant, and actionable insights.
""")
    
    # Add requirement-specific customizations
    customizations = []
    
    if requirements.get("real_time", False):
        customizations.append("- Prioritize real-time processing and low-latency responses")
    
    if requirements.get("high_accuracy", False):
        customizations.append("- Implement additional validation and accuracy checks")
    
    if requirements.get("compliance_critical", False):
        customizations.append("- Maintain strict compliance and audit trail requirements")
    
    if requirements.get("scalability", False):
        customizations.append("- Design for high scalability and concurrent processing")
    
    if customizations:
        base_instruction += f"\n\nAdditional Requirements:\n" + "\n".join(customizations)
    
    return base_instruction


def select_domain_tools(domain: str, requirements: Dict[str, Any]) -> List[str]:
    """Select appropriate tools for the domain."""
    
    domain_tools = {
        "finance": [
            "financial_data_analyzer",
            "risk_assessment_tool",
            "compliance_checker",
            "market_data_connector",
            "portfolio_optimizer"
        ],
        "healthcare": [
            "clinical_data_processor",
            "medical_knowledge_base",
            "drug_interaction_checker",
            "hipaa_compliance_tool",
            "clinical_decision_support"
        ],
        "retail": [
            "customer_analytics_tool",
            "inventory_optimizer",
            "price_optimization_tool",
            "recommendation_engine",
            "sales_forecaster"
        ],
        "manufacturing": [
            "production_planner",
            "quality_control_monitor",
            "predictive_maintenance_tool",
            "supply_chain_optimizer",
            "safety_compliance_checker"
        ]
    }
    
    base_tools = ["data_validator", "logging_tool", "error_handler"]
    domain_specific_tools = domain_tools.get(domain, [])
    
    # Add requirement-specific tools
    if requirements.get("real_time", False):
        domain_specific_tools.append("real_time_processor")
    
    if requirements.get("ml_capabilities", False):
        domain_specific_tools.extend(["ml_model_runner", "feature_processor"])
    
    if requirements.get("external_apis", False):
        domain_specific_tools.append("api_connector")
    
    return base_tools + domain_specific_tools


def optimize_for_domain(domain: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize performance configuration for domain."""
    
    base_config = {
        "temperature": 0.1,
        "max_output_tokens": 4096,
        "timeout": 30,
        "retry_count": 3
    }
    
    domain_optimizations = {
        "finance": {
            "temperature": 0.05,  # Very low for accuracy
            "max_output_tokens": 2048,  # Concise responses
            "timeout": 15,  # Fast responses for trading
            "retry_count": 5  # High reliability
        },
        "healthcare": {
            "temperature": 0.1,  # Low for medical accuracy
            "max_output_tokens": 6144,  # Detailed explanations
            "timeout": 45,  # Allow time for complex analysis
            "retry_count": 3
        },
        "retail": {
            "temperature": 0.2,  # Slightly higher for recommendations
            "max_output_tokens": 3072,
            "timeout": 20,  # Fast for customer experience
            "retry_count": 3
        },
        "manufacturing": {
            "temperature": 0.1,  # Low for precision
            "max_output_tokens": 4096,
            "timeout": 60,  # Complex calculations may take time
            "retry_count": 4
        }
    }
    
    config = {**base_config, **domain_optimizations.get(domain, {})}
    
    # Apply requirement-specific optimizations
    if requirements.get("real_time", False):
        config["timeout"] = min(config["timeout"], 10)
        config["max_output_tokens"] = min(config["max_output_tokens"], 1024)
    
    if requirements.get("high_accuracy", False):
        config["temperature"] = min(config["temperature"], 0.05)
        config["retry_count"] = max(config["retry_count"], 5)
    
    return config


def create_domain_validation(domain: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Create domain-specific validation rules."""
    
    validation_rules = {
        "input_validation": True,
        "output_validation": True,
        "format_checking": True,
        "range_validation": True
    }
    
    domain_validations = {
        "finance": {
            **validation_rules,
            "currency_validation": True,
            "calculation_verification": True,
            "regulatory_compliance_check": True,
            "risk_assessment_validation": True
        },
        "healthcare": {
            **validation_rules,
            "medical_code_validation": True,
            "dosage_safety_check": True,
            "hipaa_compliance_check": True,
            "clinical_guideline_validation": True
        },
        "retail": {
            **validation_rules,
            "product_data_validation": True,
            "price_range_validation": True,
            "inventory_consistency_check": True,
            "customer_data_validation": True
        },
        "manufacturing": {
            **validation_rules,
            "measurement_validation": True,
            "safety_parameter_check": True,
            "quality_standard_validation": True,
            "production_constraint_check": True
        }
    }
    
    return domain_validations.get(domain, validation_rules)


def setup_domain_monitoring(domain: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Setup domain-specific monitoring configuration."""
    
    base_monitoring = {
        "performance_metrics": ["latency", "throughput", "error_rate"],
        "alert_thresholds": {
            "latency": 2000,  # 2 seconds
            "error_rate": 0.05,  # 5%
            "throughput": 100  # requests per minute
        },
        "logging_level": "INFO",
        "metric_collection_interval": 60  # seconds
    }
    
    domain_monitoring = {
        "finance": {
            **base_monitoring,
            "performance_metrics": base_monitoring["performance_metrics"] + [
                "calculation_accuracy", "compliance_score", "risk_assessment_time"
            ],
            "alert_thresholds": {
                **base_monitoring["alert_thresholds"],
                "latency": 1000,  # 1 second for financial data
                "calculation_accuracy": 0.999  # 99.9% accuracy
            },
            "compliance_monitoring": True,
            "audit_logging": True
        },
        "healthcare": {
            **base_monitoring,
            "performance_metrics": base_monitoring["performance_metrics"] + [
                "clinical_accuracy", "safety_score", "hipaa_compliance"
            ],
            "alert_thresholds": {
                **base_monitoring["alert_thresholds"],
                "clinical_accuracy": 0.95,  # 95% accuracy
                "safety_score": 0.99  # 99% safety compliance
            },
            "privacy_monitoring": True,
            "clinical_audit_logging": True
        },
        "retail": {
            **base_monitoring,
            "performance_metrics": base_monitoring["performance_metrics"] + [
                "recommendation_relevance", "customer_satisfaction", "conversion_rate"
            ],
            "alert_thresholds": {
                **base_monitoring["alert_thresholds"],
                "recommendation_relevance": 0.8,  # 80% relevance
                "customer_satisfaction": 4.0  # 4.0/5.0 rating
            },
            "customer_experience_monitoring": True,
            "business_metrics_tracking": True
        },
        "manufacturing": {
            **base_monitoring,
            "performance_metrics": base_monitoring["performance_metrics"] + [
                "quality_score", "safety_compliance", "production_efficiency"
            ],
            "alert_thresholds": {
                **base_monitoring["alert_thresholds"],
                "quality_score": 0.98,  # 98% quality
                "safety_compliance": 1.0,  # 100% safety compliance
                "production_efficiency": 0.85  # 85% efficiency
            },
            "safety_monitoring": True,
            "production_tracking": True
        }
    }
    
    return domain_monitoring.get(domain, base_monitoring)


# Create the SpecializationAgent
specialization_agent = Agent(
    model=settings.root_agent_model,
    name="specialization_agent",
    instruction="""
You are the SpecializationAgent, a sub-agent focused on creating domain-specific agent configurations.

Your core responsibilities:
1. Analyze domain requirements and constraints
2. Generate specialized system prompts and instructions
3. Configure domain-specific tools and integrations
4. Optimize for domain-specific performance metrics
5. Create validation rules and monitoring configurations

Use the create_specialized_configuration tool to generate comprehensive domain specializations.
""",
    tools=[create_specialized_configuration],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=4096
    )
) 