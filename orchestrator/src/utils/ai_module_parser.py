
"""
AI Module Parser for YAML Configuration
=======================================

Parses and validates ai-module.yaml files for self-contained repositories
in the two-tiered orchestration system.
"""

import yaml
import logging
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import jsonschema
from pathlib import Path

logger = logging.getLogger(__name__)

class ModuleType(Enum):
    WEB_APP = "web_app"
    API = "api"
    MICROSERVICE = "microservice"
    DATA_PIPELINE = "data_pipeline"
    ML_MODEL = "ml_model"
    AUTOMATION = "automation"
    INFRASTRUCTURE = "infrastructure"

class DeploymentTarget(Enum):
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD_RUN = "cloud_run"
    LAMBDA = "lambda"
    VERCEL = "vercel"

@dataclass
class ResourceRequirements:
    cpu: str = "100m"
    memory: str = "128Mi"
    storage: str = "1Gi"
    gpu: bool = False

@dataclass
class HealthCheck:
    enabled: bool = True
    endpoint: str = "/health"
    interval: int = 30
    timeout: int = 10
    retries: int = 3

@dataclass
class SecurityConfig:
    enable_auth: bool = False
    auth_type: str = "jwt"
    rate_limiting: bool = True
    input_validation: bool = True
    cors_enabled: bool = True
    allowed_origins: List[str] = field(default_factory=list)

@dataclass
class MonitoringConfig:
    metrics_enabled: bool = True
    logging_level: str = "INFO"
    tracing_enabled: bool = False
    alerts_enabled: bool = False

@dataclass
class AIModuleConfig:
    """Complete AI module configuration"""
    name: str
    version: str
    description: str
    module_type: ModuleType
    build_command: str
    start_command: str
    
    # Build configuration
    test_command: Optional[str] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    system_dependencies: List[str] = field(default_factory=list)
    
    # Environment
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    
    # Deployment
    deployment_target: DeploymentTarget = DeploymentTarget.DOCKER
    port: int = 8000
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Configuration
    health_check: HealthCheck = field(default_factory=HealthCheck)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Advanced features
    auto_scaling: bool = False
    backup_enabled: bool = False
    rollback_enabled: bool = True
    
    # Custom configurations
    custom_config: Dict[str, Any] = field(default_factory=dict)

class AIModuleParser:
    """Parser for ai-module.yaml configurations"""
    
    def __init__(self):
        self.schema = self._get_validation_schema()
    
    def _get_validation_schema(self) -> Dict[str, Any]:
        """Get JSON schema for validation"""
        return {
            "type": "object",
            "required": ["name", "version", "description", "module_type", "build_command", "start_command"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                "description": {"type": "string", "minLength": 1},
                "module_type": {
                    "type": "string",
                    "enum": [t.value for t in ModuleType]
                },
                "build_command": {"type": "string", "minLength": 1},
                "test_command": {"type": ["string", "null"]},
                "start_command": {"type": "string", "minLength": 1},
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "dev_dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "system_dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "environment_variables": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                },
                "secrets": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "deployment_target": {
                    "type": "string",
                    "enum": [t.value for t in DeploymentTarget]
                },
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "resources": {
                    "type": "object",
                    "properties": {
                        "cpu": {"type": "string"},
                        "memory": {"type": "string"},
                        "storage": {"type": "string"},
                        "gpu": {"type": "boolean"}
                    }
                },
                "health_check": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "endpoint": {"type": "string"},
                        "interval": {"type": "integer", "minimum": 1},
                        "timeout": {"type": "integer", "minimum": 1},
                        "retries": {"type": "integer", "minimum": 1}
                    }
                },
                "security": {
                    "type": "object",
                    "properties": {
                        "enable_auth": {"type": "boolean"},
                        "auth_type": {"type": "string"},
                        "rate_limiting": {"type": "boolean"},
                        "input_validation": {"type": "boolean"},
                        "cors_enabled": {"type": "boolean"},
                        "allowed_origins": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "monitoring": {
                    "type": "object",
                    "properties": {
                        "metrics_enabled": {"type": "boolean"},
                        "logging_level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        },
                        "tracing_enabled": {"type": "boolean"},
                        "alerts_enabled": {"type": "boolean"}
                    }
                },
                "auto_scaling": {"type": "boolean"},
                "backup_enabled": {"type": "boolean"},
                "rollback_enabled": {"type": "boolean"},
                "custom_config": {"type": "object"}
            }
        }
    
    def parse_file(self, file_path: str) -> AIModuleConfig:
        """Parse ai-module.yaml file"""
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            
            return self.parse_dict(data)
            
        except FileNotFoundError:
            raise ValueError(f"AI module file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def parse_dict(self, data: Dict[str, Any]) -> AIModuleConfig:
        """Parse configuration from dictionary"""
        try:
            # Validate against schema
            jsonschema.validate(data, self.schema)
            
            # Parse module type
            module_type = ModuleType(data["module_type"])
            
            # Parse deployment target
            deployment_target = DeploymentTarget(
                data.get("deployment_target", "docker")
            )
            
            # Parse resources
            resources_data = data.get("resources", {})
            resources = ResourceRequirements(
                cpu=resources_data.get("cpu", "100m"),
                memory=resources_data.get("memory", "128Mi"),
                storage=resources_data.get("storage", "1Gi"),
                gpu=resources_data.get("gpu", False)
            )
            
            # Parse health check
            health_data = data.get("health_check", {})
            health_check = HealthCheck(
                enabled=health_data.get("enabled", True),
                endpoint=health_data.get("endpoint", "/health"),
                interval=health_data.get("interval", 30),
                timeout=health_data.get("timeout", 10),
                retries=health_data.get("retries", 3)
            )
            
            # Parse security
            security_data = data.get("security", {})
            security = SecurityConfig(
                enable_auth=security_data.get("enable_auth", False),
                auth_type=security_data.get("auth_type", "jwt"),
                rate_limiting=security_data.get("rate_limiting", True),
                input_validation=security_data.get("input_validation", True),
                cors_enabled=security_data.get("cors_enabled", True),
                allowed_origins=security_data.get("allowed_origins", [])
            )
            
            # Parse monitoring
            monitoring_data = data.get("monitoring", {})
            monitoring = MonitoringConfig(
                metrics_enabled=monitoring_data.get("metrics_enabled", True),
                logging_level=monitoring_data.get("logging_level", "INFO"),
                tracing_enabled=monitoring_data.get("tracing_enabled", False),
                alerts_enabled=monitoring_data.get("alerts_enabled", False)
            )
            
            # Create configuration object
            config = AIModuleConfig(
                name=data["name"],
                version=data["version"],
                description=data["description"],
                module_type=module_type,
                build_command=data["build_command"],
                test_command=data.get("test_command"),
                start_command=data["start_command"],
                dependencies=data.get("dependencies", []),
                dev_dependencies=data.get("dev_dependencies", []),
                system_dependencies=data.get("system_dependencies", []),
                environment_variables=data.get("environment_variables", {}),
                secrets=data.get("secrets", []),
                deployment_target=deployment_target,
                port=data.get("port", 8000),
                resources=resources,
                health_check=health_check,
                security=security,
                monitoring=monitoring,
                auto_scaling=data.get("auto_scaling", False),
                backup_enabled=data.get("backup_enabled", False),
                rollback_enabled=data.get("rollback_enabled", True),
                custom_config=data.get("custom_config", {})
            )
            
            logger.info(f"Successfully parsed AI module: {config.name} v{config.version}")
            return config
            
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
        except Exception as e:
            raise ValueError(f"Failed to parse configuration: {e}")
    
    def validate_file(self, file_path: str) -> bool:
        """Validate ai-module.yaml file without parsing"""
        try:
            self.parse_file(file_path)
            return True
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            return False
    
    def generate_template(self, module_type: ModuleType) -> str:
        """Generate template ai-module.yaml for given module type"""
        
        templates = {
            ModuleType.WEB_APP: {
                "name": "my-web-app",
                "version": "1.0.0",
                "description": "A modern web application",
                "module_type": "web_app",
                "build_command": "npm run build",
                "test_command": "npm test",
                "start_command": "npm start",
                "dependencies": ["react", "express"],
                "dev_dependencies": ["jest", "eslint"],
                "port": 3000,
                "security": {
                    "cors_enabled": True,
                    "rate_limiting": True
                }
            },
            ModuleType.API: {
                "name": "my-api",
                "version": "1.0.0",
                "description": "RESTful API service",
                "module_type": "api",
                "build_command": "pip install -r requirements.txt",
                "test_command": "pytest",
                "start_command": "uvicorn main:app --host 0.0.0.0 --port 8000",
                "dependencies": ["fastapi", "uvicorn"],
                "dev_dependencies": ["pytest", "black"],
                "port": 8000,
                "security": {
                    "enable_auth": True,
                    "auth_type": "jwt",
                    "rate_limiting": True,
                    "input_validation": True
                }
            },
            ModuleType.ML_MODEL: {
                "name": "my-ml-model",
                "version": "1.0.0",
                "description": "Machine learning model service",
                "module_type": "ml_model",
                "build_command": "pip install -r requirements.txt",
                "test_command": "pytest tests/",
                "start_command": "python serve.py",
                "dependencies": ["scikit-learn", "pandas", "numpy"],
                "dev_dependencies": ["pytest", "jupyter"],
                "port": 8000,
                "resources": {
                    "cpu": "500m",
                    "memory": "1Gi",
                    "gpu": False
                }
            }
        }
        
        template = templates.get(module_type, templates[ModuleType.API])
        return yaml.dump(template, default_flow_style=False, sort_keys=False)
    
    def detect_module_type(self, repo_path: str) -> Optional[ModuleType]:
        """Auto-detect module type from repository structure"""
        repo_path = Path(repo_path)
        
        # Check for specific files/patterns
        if (repo_path / "package.json").exists():
            package_json = yaml.safe_load((repo_path / "package.json").read_text())
            if "react" in str(package_json.get("dependencies", {})):
                return ModuleType.WEB_APP
            return ModuleType.WEB_APP
        
        if (repo_path / "requirements.txt").exists():
            requirements = (repo_path / "requirements.txt").read_text()
            if "fastapi" in requirements or "flask" in requirements:
                return ModuleType.API
            if "scikit-learn" in requirements or "tensorflow" in requirements:
                return ModuleType.ML_MODEL
            return ModuleType.API
        
        if (repo_path / "Dockerfile").exists():
            dockerfile = (repo_path / "Dockerfile").read_text()
            if "node" in dockerfile.lower():
                return ModuleType.WEB_APP
            if "python" in dockerfile.lower():
                return ModuleType.API
        
        return None

# Example usage
if __name__ == "__main__":
    parser = AIModuleParser()
    
    # Generate template
    template = parser.generate_template(ModuleType.API)
    print("API Template:")
    print(template)
    
    # Test validation
    test_config = {
        "name": "test-api",
        "version": "1.0.0",
        "description": "Test API",
        "module_type": "api",
        "build_command": "pip install -r requirements.txt",
        "start_command": "uvicorn main:app"
    }
    
    try:
        config = parser.parse_dict(test_config)
        print(f"\nParsed config: {config.name}")
    except Exception as e:
        print(f"Error: {e}")
