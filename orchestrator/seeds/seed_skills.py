
"""
Seed Skills and Patterns Data
============================

Seeds the database with all 32 skills across 4 categories and sample patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import sessionmaker
from database import engine
from models import Skill, Pattern

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def seed_skills():
    """Seed all 32 skills across 4 categories"""
    db = SessionLocal()
    
    try:
        # Development Category Skills (8 skills)
        development_skills = [
            {
                "name": "Code Review",
                "description": "Automated code review and quality assessment",
                "skill_type": "technical",
                "category": "development",
                "implementation": "def code_review(code): return analyze_code_quality(code)",
                "parameters": {"languages": ["python", "javascript", "java"], "metrics": ["complexity", "coverage", "style"]}
            },
            {
                "name": "Testing Automation",
                "description": "Automated test generation and execution",
                "skill_type": "technical", 
                "category": "development",
                "implementation": "def automated_testing(code): return generate_and_run_tests(code)",
                "parameters": {"test_types": ["unit", "integration", "e2e"], "frameworks": ["pytest", "jest", "junit"]}
            },
            {
                "name": "Best Practices",
                "description": "Code best practices analysis and recommendations",
                "skill_type": "cognitive",
                "category": "development", 
                "implementation": "def best_practices_analysis(code): return analyze_best_practices(code)",
                "parameters": {"standards": ["PEP8", "ESLint", "SonarQube"], "severity_levels": ["info", "warning", "error"]}
            },
            {
                "name": "Design Patterns",
                "description": "Design pattern recognition and implementation",
                "skill_type": "cognitive",
                "category": "development",
                "implementation": "def design_patterns(requirements): return suggest_patterns(requirements)",
                "parameters": {"patterns": ["singleton", "factory", "observer", "strategy"], "languages": ["python", "java", "csharp"]}
            },
            {
                "name": "API Development",
                "description": "RESTful API design and implementation",
                "skill_type": "technical",
                "category": "development",
                "implementation": "def api_development(spec): return generate_api_code(spec)",
                "parameters": {"frameworks": ["fastapi", "express", "spring"], "standards": ["OpenAPI", "GraphQL"]}
            },
            {
                "name": "Database Design",
                "description": "Database schema design and optimization",
                "skill_type": "technical",
                "category": "development",
                "implementation": "def database_design(requirements): return create_schema(requirements)",
                "parameters": {"databases": ["postgresql", "mysql", "mongodb"], "optimization": ["indexing", "normalization"]}
            },
            {
                "name": "Version Control",
                "description": "Git workflow management and branching strategies",
                "skill_type": "technical",
                "category": "development",
                "implementation": "def version_control(project): return manage_git_workflow(project)",
                "parameters": {"strategies": ["gitflow", "github-flow"], "tools": ["git", "github", "gitlab"]}
            },
            {
                "name": "Documentation",
                "description": "Automated documentation generation and maintenance",
                "skill_type": "communication",
                "category": "development",
                "implementation": "def generate_documentation(code): return create_docs(code)",
                "parameters": {"formats": ["markdown", "sphinx", "jsdoc"], "types": ["api", "user", "technical"]}
            }
        ]
        
        # Security Category Skills (8 skills)
        security_skills = [
            {
                "name": "Vulnerability Scanning",
                "description": "Automated security vulnerability detection and assessment",
                "skill_type": "technical",
                "category": "security",
                "implementation": "def vulnerability_scan(target): return scan_for_vulnerabilities(target)",
                "parameters": {"scan_types": ["static", "dynamic", "interactive"], "severity": ["low", "medium", "high", "critical"]}
            },
            {
                "name": "Threat Modeling",
                "description": "Security threat analysis and modeling",
                "skill_type": "cognitive",
                "category": "security",
                "implementation": "def threat_modeling(system): return analyze_threats(system)",
                "parameters": {"methodologies": ["STRIDE", "PASTA", "OCTAVE"], "asset_types": ["data", "systems", "processes"]}
            },
            {
                "name": "Penetration Testing",
                "description": "Automated penetration testing and security assessment",
                "skill_type": "technical",
                "category": "security",
                "implementation": "def penetration_test(target): return conduct_pentest(target)",
                "parameters": {"test_types": ["network", "web", "mobile"], "tools": ["nmap", "burp", "metasploit"]}
            },
            {
                "name": "Compliance Auditing",
                "description": "Security compliance checking and reporting",
                "skill_type": "cognitive",
                "category": "security",
                "implementation": "def compliance_audit(system): return check_compliance(system)",
                "parameters": {"standards": ["SOC2", "ISO27001", "GDPR", "HIPAA"], "audit_types": ["technical", "administrative"]}
            },
            {
                "name": "Incident Response",
                "description": "Security incident detection and response automation",
                "skill_type": "technical",
                "category": "security",
                "implementation": "def incident_response(alert): return handle_security_incident(alert)",
                "parameters": {"incident_types": ["malware", "breach", "ddos"], "response_actions": ["isolate", "analyze", "remediate"]}
            },
            {
                "name": "Access Control",
                "description": "Identity and access management automation",
                "skill_type": "technical",
                "category": "security",
                "implementation": "def access_control(user, resource): return manage_access(user, resource)",
                "parameters": {"auth_methods": ["oauth", "saml", "ldap"], "policies": ["rbac", "abac", "mac"]}
            },
            {
                "name": "Encryption Management",
                "description": "Cryptographic key and encryption management",
                "skill_type": "technical",
                "category": "security",
                "implementation": "def encryption_management(data): return encrypt_and_manage_keys(data)",
                "parameters": {"algorithms": ["AES", "RSA", "ECC"], "key_management": ["rotation", "escrow", "recovery"]}
            },
            {
                "name": "Security Monitoring",
                "description": "Continuous security monitoring and alerting",
                "skill_type": "technical",
                "category": "security",
                "implementation": "def security_monitoring(logs): return analyze_security_events(logs)",
                "parameters": {"log_sources": ["firewall", "ids", "application"], "alert_types": ["anomaly", "signature", "behavioral"]}
            }
        ]
        
        # Infrastructure Category Skills (8 skills)
        infrastructure_skills = [
            {
                "name": "Deployment Automation",
                "description": "Automated application deployment and release management",
                "skill_type": "technical",
                "category": "infrastructure",
                "implementation": "def deployment_automation(app): return deploy_application(app)",
                "parameters": {"platforms": ["kubernetes", "docker", "aws"], "strategies": ["blue-green", "canary", "rolling"]}
            },
            {
                "name": "Scaling Management",
                "description": "Automatic scaling and resource optimization",
                "skill_type": "technical",
                "category": "infrastructure",
                "implementation": "def scaling_management(metrics): return auto_scale_resources(metrics)",
                "parameters": {"scaling_types": ["horizontal", "vertical"], "triggers": ["cpu", "memory", "requests"]}
            },
            {
                "name": "Monitoring Setup",
                "description": "Infrastructure monitoring and alerting configuration",
                "skill_type": "technical",
                "category": "infrastructure",
                "implementation": "def monitoring_setup(infrastructure): return configure_monitoring(infrastructure)",
                "parameters": {"tools": ["prometheus", "grafana", "datadog"], "metrics": ["system", "application", "business"]}
            },
            {
                "name": "Backup Management",
                "description": "Automated backup and disaster recovery",
                "skill_type": "technical",
                "category": "infrastructure",
                "implementation": "def backup_management(data): return manage_backups(data)",
                "parameters": {"backup_types": ["full", "incremental", "differential"], "storage": ["s3", "azure", "gcp"]}
            },
            {
                "name": "Network Configuration",
                "description": "Network infrastructure setup and management",
                "skill_type": "technical",
                "category": "infrastructure",
                "implementation": "def network_configuration(requirements): return configure_network(requirements)",
                "parameters": {"components": ["vpc", "subnets", "security_groups"], "protocols": ["tcp", "udp", "http"]}
            },
            {
                "name": "Container Orchestration",
                "description": "Container deployment and orchestration management",
                "skill_type": "technical",
                "category": "infrastructure",
                "implementation": "def container_orchestration(containers): return orchestrate_containers(containers)",
                "parameters": {"platforms": ["kubernetes", "docker-swarm", "ecs"], "features": ["service-mesh", "ingress", "storage"]}
            },
            {
                "name": "Infrastructure as Code",
                "description": "Infrastructure provisioning through code",
                "skill_type": "technical",
                "category": "infrastructure",
                "implementation": "def infrastructure_as_code(spec): return provision_infrastructure(spec)",
                "parameters": {"tools": ["terraform", "cloudformation", "ansible"], "providers": ["aws", "azure", "gcp"]}
            },
            {
                "name": "Performance Optimization",
                "description": "System performance analysis and optimization",
                "skill_type": "cognitive",
                "category": "infrastructure",
                "implementation": "def performance_optimization(system): return optimize_performance(system)",
                "parameters": {"optimization_areas": ["cpu", "memory", "io", "network"], "tools": ["profilers", "benchmarks"]}
            }
        ]
        
        # Analytics Category Skills (8 skills)
        analytics_skills = [
            {
                "name": "Data Processing",
                "description": "Large-scale data processing and transformation",
                "skill_type": "technical",
                "category": "analytics",
                "implementation": "def data_processing(data): return process_and_transform(data)",
                "parameters": {"frameworks": ["spark", "pandas", "dask"], "formats": ["csv", "json", "parquet"]}
            },
            {
                "name": "Pattern Recognition",
                "description": "Automated pattern detection in data",
                "skill_type": "cognitive",
                "category": "analytics",
                "implementation": "def pattern_recognition(data): return detect_patterns(data)",
                "parameters": {"algorithms": ["clustering", "classification", "regression"], "pattern_types": ["temporal", "spatial", "behavioral"]}
            },
            {
                "name": "Predictive Modeling",
                "description": "Machine learning model development and deployment",
                "skill_type": "cognitive",
                "category": "analytics",
                "implementation": "def predictive_modeling(data): return build_predictive_model(data)",
                "parameters": {"algorithms": ["random_forest", "neural_networks", "svm"], "validation": ["cross_validation", "holdout"]}
            },
            {
                "name": "Data Visualization",
                "description": "Automated chart and dashboard generation",
                "skill_type": "communication",
                "category": "analytics",
                "implementation": "def data_visualization(data): return create_visualizations(data)",
                "parameters": {"chart_types": ["bar", "line", "scatter", "heatmap"], "tools": ["plotly", "d3", "tableau"]}
            },
            {
                "name": "Statistical Analysis",
                "description": "Statistical analysis and hypothesis testing",
                "skill_type": "cognitive",
                "category": "analytics",
                "implementation": "def statistical_analysis(data): return perform_statistical_tests(data)",
                "parameters": {"tests": ["t_test", "chi_square", "anova"], "confidence_levels": [0.90, 0.95, 0.99]}
            },
            {
                "name": "Real-time Analytics",
                "description": "Stream processing and real-time data analysis",
                "skill_type": "technical",
                "category": "analytics",
                "implementation": "def realtime_analytics(stream): return analyze_stream(stream)",
                "parameters": {"frameworks": ["kafka", "storm", "flink"], "window_types": ["tumbling", "sliding", "session"]}
            },
            {
                "name": "Data Quality Assessment",
                "description": "Data quality monitoring and validation",
                "skill_type": "cognitive",
                "category": "analytics",
                "implementation": "def data_quality_assessment(data): return assess_data_quality(data)",
                "parameters": {"quality_dimensions": ["completeness", "accuracy", "consistency"], "validation_rules": ["format", "range", "uniqueness"]}
            },
            {
                "name": "Business Intelligence",
                "description": "Business metrics calculation and reporting",
                "skill_type": "communication",
                "category": "analytics",
                "implementation": "def business_intelligence(data): return generate_business_insights(data)",
                "parameters": {"metrics": ["kpi", "roi", "conversion"], "report_types": ["executive", "operational", "analytical"]}
            }
        ]
        
        # Combine all skills
        all_skills = development_skills + security_skills + infrastructure_skills + analytics_skills
        
        # Check if skills already exist
        existing_skills = db.query(Skill).count()
        if existing_skills > 0:
            print(f"Skills already exist ({existing_skills} found). Skipping skill seeding.")
        else:
            # Create skill records
            for skill_data in all_skills:
                skill = Skill(**skill_data)
                db.add(skill)
            
            db.commit()
            print(f"Successfully seeded {len(all_skills)} skills across 4 categories")
        
    except Exception as e:
        print(f"Error seeding skills: {e}")
        db.rollback()
    finally:
        db.close()

def seed_patterns():
    """Seed sample patterns"""
    db = SessionLocal()
    
    try:
        sample_patterns = [
            {
                "name": "Multi-Agent Coordination",
                "description": "Pattern for coordinating multiple agents on complex tasks",
                "pattern_type": "coordination",
                "pattern_data": {
                    "steps": [
                        "Task decomposition",
                        "Agent assignment",
                        "Progress monitoring",
                        "Result aggregation"
                    ],
                    "roles": ["coordinator", "worker", "validator"],
                    "communication_protocol": "message_passing"
                }
            },
            {
                "name": "Error Recovery",
                "description": "Pattern for handling and recovering from agent errors",
                "pattern_type": "decision",
                "pattern_data": {
                    "error_types": ["timeout", "resource_unavailable", "invalid_input"],
                    "recovery_strategies": ["retry", "fallback", "escalate"],
                    "max_retries": 3,
                    "backoff_strategy": "exponential"
                }
            },
            {
                "name": "Load Balancing",
                "description": "Pattern for distributing work across multiple agents",
                "pattern_type": "coordination",
                "pattern_data": {
                    "balancing_algorithms": ["round_robin", "least_connections", "weighted"],
                    "health_checks": True,
                    "failover_enabled": True,
                    "metrics": ["response_time", "success_rate", "resource_usage"]
                }
            },
            {
                "name": "Pipeline Processing",
                "description": "Pattern for sequential processing through multiple agents",
                "pattern_type": "coordination",
                "pattern_data": {
                    "stages": ["input_validation", "processing", "output_formatting"],
                    "parallel_processing": False,
                    "checkpoint_enabled": True,
                    "rollback_strategy": "stage_level"
                }
            },
            {
                "name": "Consensus Decision",
                "description": "Pattern for making decisions based on multiple agent inputs",
                "pattern_type": "decision",
                "pattern_data": {
                    "voting_mechanism": "majority",
                    "minimum_participants": 3,
                    "confidence_threshold": 0.7,
                    "tie_breaking": "random"
                }
            }
        ]
        
        # Check if patterns already exist
        existing_patterns = db.query(Pattern).count()
        if existing_patterns > 0:
            print(f"Patterns already exist ({existing_patterns} found). Skipping pattern seeding.")
        else:
            # Create pattern records
            for pattern_data in sample_patterns:
                pattern = Pattern(**pattern_data)
                db.add(pattern)
            
            db.commit()
            print(f"Successfully seeded {len(sample_patterns)} sample patterns")
        
    except Exception as e:
        print(f"Error seeding patterns: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Seeding skills and patterns...")
    seed_skills()
    seed_patterns()
    print("Seeding completed!")
