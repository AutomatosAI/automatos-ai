#!/usr/bin/env python3
"""
System Validation Script for Enhanced Two-Tiered Multi-Agent Orchestration System
"""

import os
import sys
from pathlib import Path

def validate_files():
    """Validate all required files are present"""
    required_files = [
        'orchestrator.py',
        'mcp_bridge.py', 
        'ssh_manager.py',
        'ai_module_parser.py',
        'security/__init__.py',
        'security/audit.py',
        'requirements.txt',
        'Dockerfile',
        'docker compose.yml',
        '.env.example',
        'init.sql',
        'deploy.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def validate_templates():
    """Validate template structure"""
    ai_module_template = Path('templates/ai-module-examples/web-app/ai-module.yaml')
    task_prompt_template = Path('templates/task-prompt-examples/simple-web-server/app.py')
    
    if not ai_module_template.exists():
        print("❌ AI module template missing")
        return False
        
    if not task_prompt_template.exists():
        print("❌ Task prompt template missing")
        return False
    
    print("✅ Templates validated")
    return True

def validate_python_syntax():
    """Validate Python syntax"""
    python_files = [
        'orchestrator.py',
        'mcp_bridge.py',
        'ssh_manager.py', 
        'ai_module_parser.py',
        'security/audit.py'
    ]
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            return False
    
    print("✅ Python syntax validated")
    return True

def validate_ai_module_parser():
    """Test AI module parser"""
    try:
        from ai_module_parser import AIModuleParser
        parser = AIModuleParser()
        config = parser.parse_file('templates/ai-module-examples/web-app/ai-module.yaml')
        print(f"✅ AI Module Parser: {config.name} v{config.version}")
        return True
    except Exception as e:
        print(f"❌ AI Module Parser failed: {e}")
        return False

def validate_security_system():
    """Test security audit system"""
    try:
        from security.audit import SecurityAuditLogger, create_security_event, EventType, SecurityLevel
        audit_logger = SecurityAuditLogger()
        event = create_security_event(
            event_type=EventType.AUTHENTICATION,
            user_id='test',
            source_ip='127.0.0.1',
            resource='/test',
            action='test',
            result='success'
        )
        audit_logger.log_event(event)
        print("✅ Security system validated")
        return True
    except Exception as e:
        print(f"❌ Security system failed: {e}")
        return False

def main():
    """Run all validations"""
    print("🔍 Validating Enhanced Two-Tiered Multi-Agent Orchestration System")
    print("=" * 70)
    
    validations = [
        validate_files,
        validate_templates,
        validate_python_syntax,
        validate_ai_module_parser,
        validate_security_system
    ]
    
    results = []
    for validation in validations:
        try:
            result = validation()
            results.append(result)
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    if all(results):
        print("🎉 All validations passed! System ready for deployment.")
        print("\nNext steps:")
        print("1. Configure .env file with your credentials")
        print("2. Run: ./deploy.sh --host mcp.xplaincrypto.ai")
        print("3. Test API: curl -H 'X-API-Key: your_key' http://mcp.xplaincrypto.ai:8001/health")
        return 0
    else:
        print("❌ Some validations failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
