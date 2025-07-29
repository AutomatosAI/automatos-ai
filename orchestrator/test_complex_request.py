
"""
Complex Development Task Test Script
===================================

This script tests the enhanced Automotas AI system with a complex development request
to demonstrate the system's capabilities with full workflow visibility.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the orchestrator directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_orchestrator import EnhancedTwoTierOrchestrator

class ComplexTaskTester:
    """Test runner for complex development tasks"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.end_time = None
    
    async def run_complex_api_test(self):
        """Test: Build a comprehensive REST API with authentication"""
        
        print("🚀 TEST 1: Complex REST API with JWT Authentication")
        print("=" * 60)
        
        task_description = """
        Build a comprehensive REST API with the following requirements:
        
        CORE FEATURES:
        - FastAPI framework with async support
        - JWT token-based authentication system
        - User registration and login endpoints
        - Protected routes requiring authentication
        - User profile management (CRUD operations)
        - Password hashing with bcrypt
        - Input validation with Pydantic models
        - SQLite database with SQLAlchemy ORM
        
        API ENDPOINTS:
        - POST /auth/register - User registration
        - POST /auth/login - User login (returns JWT token)
        - GET /auth/me - Get current user profile (protected)
        - PUT /auth/me - Update user profile (protected)
        - GET /users - List all users (admin only)
        - DELETE /users/{user_id} - Delete user (admin only)
        - GET /health - Health check endpoint
        
        SECURITY FEATURES:
        - Password strength validation
        - Rate limiting on auth endpoints
        - CORS configuration
        - Request/response logging
        - Error handling with proper HTTP status codes
        
        TESTING & DOCUMENTATION:
        - Comprehensive unit tests with pytest
        - Integration tests for all endpoints
        - API documentation with OpenAPI/Swagger
        - Test coverage reporting
        
        DEPLOYMENT:
        - Docker configuration with multi-stage build
        - Docker Compose for development
        - Environment variable configuration
        - Production-ready logging setup
        - Health checks and monitoring
        
        ADDITIONAL FILES:
        - requirements.txt with all dependencies
        - .env.example for environment variables
        - README.md with setup and usage instructions
        - .gitignore for Python projects
        - Makefile for common development tasks
        """
        
        orchestrator = EnhancedTwoTierOrchestrator()
        
        try:
            result = await orchestrator.execute_complex_development_task(
                task_description=task_description,
                project_path="/tmp/complex_api_test"
            )
            
            self.test_results.append({
                "test_name": "Complex REST API",
                "success": result["success"],
                "result": result
            })
            
            if result["success"]:
                print("✅ Complex API test completed successfully!")
                self._display_test_results(result)
            else:
                print(f"❌ Complex API test failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            self.test_results.append({
                "test_name": "Complex REST API",
                "success": False,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    async def run_fullstack_webapp_test(self):
        """Test: Build a full-stack web application"""
        
        print("\n🚀 TEST 2: Full-Stack Web Application")
        print("=" * 60)
        
        task_description = """
        Build a full-stack web application with the following specifications:
        
        BACKEND (Python/FastAPI):
        - RESTful API with CRUD operations
        - User authentication and authorization
        - Database integration with PostgreSQL
        - File upload and management
        - Real-time notifications with WebSockets
        - Background task processing with Celery
        - API rate limiting and caching
        
        FRONTEND (React/TypeScript):
        - Modern React application with TypeScript
        - User authentication flow (login/register/logout)
        - Dashboard with data visualization
        - File upload interface with progress tracking
        - Real-time updates via WebSocket connection
        - Responsive design with Tailwind CSS
        - Form validation and error handling
        
        DATABASE DESIGN:
        - User management tables
        - Content/data tables with relationships
        - File metadata storage
        - Audit logging tables
        - Database migrations and seeding
        
        INFRASTRUCTURE:
        - Docker Compose for multi-service setup
        - Nginx reverse proxy configuration
        - Redis for caching and session storage
        - PostgreSQL database with persistent volumes
        - Environment-based configuration
        
        TESTING & QUALITY:
        - Backend unit and integration tests
        - Frontend component and E2E tests
        - Code quality tools (ESLint, Prettier, Black)
        - CI/CD pipeline configuration
        - Performance monitoring setup
        
        DOCUMENTATION:
        - API documentation with interactive examples
        - Frontend component documentation
        - Deployment guide
        - Development setup instructions
        - Architecture overview
        """
        
        orchestrator = EnhancedTwoTierOrchestrator()
        
        try:
            result = await orchestrator.execute_complex_development_task(
                task_description=task_description,
                project_path="/tmp/fullstack_webapp_test"
            )
            
            self.test_results.append({
                "test_name": "Full-Stack Web Application",
                "success": result["success"],
                "result": result
            })
            
            if result["success"]:
                print("✅ Full-stack webapp test completed successfully!")
                self._display_test_results(result)
            else:
                print(f"❌ Full-stack webapp test failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            self.test_results.append({
                "test_name": "Full-Stack Web Application",
                "success": False,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    async def run_microservices_test(self):
        """Test: Build a microservices architecture"""
        
        print("\n🚀 TEST 3: Microservices Architecture")
        print("=" * 60)
        
        task_description = """
        Build a microservices architecture with the following components:
        
        SERVICE ARCHITECTURE:
        - API Gateway service (FastAPI/Kong)
        - User Authentication service
        - Product Catalog service
        - Order Management service
        - Payment Processing service
        - Notification service
        - File Storage service
        
        INTER-SERVICE COMMUNICATION:
        - REST APIs for synchronous communication
        - Message queues (RabbitMQ/Redis) for async communication
        - Service discovery and registration
        - Circuit breaker pattern implementation
        - Distributed tracing with OpenTelemetry
        
        DATA MANAGEMENT:
        - Database per service pattern
        - Event sourcing for critical operations
        - CQRS implementation where appropriate
        - Data consistency strategies
        - Database migration management
        
        INFRASTRUCTURE:
        - Docker containers for each service
        - Kubernetes deployment manifests
        - Service mesh configuration (Istio)
        - Load balancing and auto-scaling
        - Monitoring and logging stack
        
        SECURITY:
        - JWT-based authentication across services
        - API rate limiting and throttling
        - Service-to-service authentication
        - Secrets management
        - Security scanning and compliance
        
        OBSERVABILITY:
        - Centralized logging with ELK stack
        - Metrics collection with Prometheus
        - Distributed tracing
        - Health checks and monitoring
        - Alerting and notification setup
        
        TESTING:
        - Unit tests for each service
        - Integration tests between services
        - Contract testing with Pact
        - End-to-end testing scenarios
        - Performance and load testing
        """
        
        orchestrator = EnhancedTwoTierOrchestrator()
        
        try:
            result = await orchestrator.execute_complex_development_task(
                task_description=task_description,
                project_path="/tmp/microservices_test"
            )
            
            self.test_results.append({
                "test_name": "Microservices Architecture",
                "success": result["success"],
                "result": result
            })
            
            if result["success"]:
                print("✅ Microservices test completed successfully!")
                self._display_test_results(result)
            else:
                print(f"❌ Microservices test failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            self.test_results.append({
                "test_name": "Microservices Architecture",
                "success": False,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    def _display_test_results(self, result):
        """Display detailed test results"""
        if not result.get("success"):
            return
        
        print(f"\n📊 Test Results Summary:")
        print(f"  📁 Project Path: {result.get('project_path', 'N/A')}")
        print(f"  📄 Files Generated: {result.get('files_generated', 0)}")
        print(f"  🔧 Subtasks Processed: {result.get('subtasks_processed', 0)}")
        
        # Display generated files
        generated_files = result.get("generated_files", [])
        if generated_files:
            print(f"\n📋 Generated Files ({len(generated_files)}):")
            for file_info in generated_files[:10]:  # Show first 10 files
                size_kb = file_info.get('size_bytes', 0) / 1024
                print(f"  - {file_info.get('file_path', 'Unknown')} ({file_info.get('content_type', 'Unknown')}, {size_kb:.1f} KB)")
            
            if len(generated_files) > 10:
                print(f"  ... and {len(generated_files) - 10} more files")
        
        # Display execution summary
        summary = result.get("execution_summary", {})
        if summary:
            print(f"\n⏱️  Execution Metrics:")
            print(f"  - Duration: {summary.get('total_duration_seconds', 0):.2f} seconds")
            print(f"  - Steps: {summary.get('completed_steps', 0)}/{summary.get('total_steps', 0)}")
            print(f"  - Agents: {summary.get('total_agents', 0)} ({summary.get('active_agents', 0)} active)")
            print(f"  - Tokens: {summary.get('total_tokens_used', 0):,}")
            print(f"  - Cost: ${summary.get('total_estimated_cost', 0):.4f}")
            print(f"  - Phase: {summary.get('current_phase', 'Unknown')}")
            print(f"  - Progress: {summary.get('overall_progress', 0):.1f}%")
    
    async def run_all_tests(self):
        """Run all complex development tests"""
        
        print("🎯 AUTOMOTAS AI COMPLEX TASK TESTING")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = datetime.now()
        
        # Run tests
        await self.run_complex_api_test()
        # await self.run_fullstack_webapp_test()  # Uncomment for additional tests
        # await self.run_microservices_test()     # Uncomment for additional tests
        
        self.end_time = datetime.now()
        
        # Display final summary
        self._display_final_summary()
    
    def _display_final_summary(self):
        """Display final test summary"""
        
        print("\n" + "=" * 80)
        print("🏁 FINAL TEST SUMMARY")
        print("=" * 80)
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        successful_tests = sum(1 for test in self.test_results if test.get("success", False))
        total_tests = len(self.test_results)
        
        print(f"📊 Overall Results:")
        print(f"  - Tests Run: {total_tests}")
        print(f"  - Successful: {successful_tests}")
        print(f"  - Failed: {total_tests - successful_tests}")
        print(f"  - Success Rate: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "  - Success Rate: N/A")
        print(f"  - Total Duration: {total_duration:.2f} seconds")
        
        print(f"\n📋 Test Details:")
        for i, test in enumerate(self.test_results, 1):
            status = "✅ PASS" if test.get("success", False) else "❌ FAIL"
            print(f"  {i}. {test['test_name']}: {status}")
            if not test.get("success", False) and "error" in test:
                print(f"     Error: {test['error']}")
        
        # Calculate aggregate metrics
        total_files = 0
        total_subtasks = 0
        total_tokens = 0
        total_cost = 0.0
        
        for test in self.test_results:
            if test.get("success", False) and "result" in test:
                result = test["result"]
                total_files += result.get("files_generated", 0)
                total_subtasks += result.get("subtasks_processed", 0)
                
                summary = result.get("execution_summary", {})
                total_tokens += summary.get("total_tokens_used", 0)
                total_cost += summary.get("total_estimated_cost", 0.0)
        
        if successful_tests > 0:
            print(f"\n📈 Aggregate Metrics:")
            print(f"  - Total Files Generated: {total_files}")
            print(f"  - Total Subtasks Processed: {total_subtasks}")
            print(f"  - Total Tokens Used: {total_tokens:,}")
            print(f"  - Total Estimated Cost: ${total_cost:.4f}")
        
        print(f"\n🎉 Testing completed at {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save results to file
        self._save_results_to_file()
    
    def _save_results_to_file(self):
        """Save test results to a JSON file"""
        
        results_data = {
            "test_session": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": sum(1 for test in self.test_results if test.get("success", False)),
                "failed_tests": sum(1 for test in self.test_results if not test.get("success", False))
            }
        }
        
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"📄 Test results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results to file: {e}")

async def main():
    """Main test execution function"""
    
    # Check environment setup
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the tests.")
        return
    
    # Run tests
    tester = ComplexTaskTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
