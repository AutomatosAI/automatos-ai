
# 🤝 Contributing to Automatos AI

*Join the revolution in AI-powered automation! We welcome contributors of all skill levels.*

---

## 🌟 Why Contribute?

### **🚀 Be Part of Something Revolutionary**
- **Cutting-Edge AI**: Work on advanced multi-agent systems and context engineering
- **Real Impact**: Your contributions power enterprise automation worldwide
- **Learn & Grow**: Gain expertise in AI, distributed systems, and modern software architecture
- **Global Recognition**: Build your reputation in the open source community

### **🏆 Contributor Benefits**
- **🎯 Exclusive Access**: Early access to new features and beta releases
- **🎁 Swag & Rewards**: Exclusive contributor merchandise and digital badges
- **📺 Speaking Opportunities**: Present at conferences and community meetups
- **🤝 Networking**: Connect with top developers and AI researchers globally
- **💼 Career Opportunities**: Direct pipeline to job opportunities at leading tech companies

---

## 🛠️ Ways to Contribute

### **💻 Code Contributions**

#### **🐛 Bug Fixes**
- **Difficulty**: Beginner to Intermediate
- **Impact**: High - Keep the platform stable and reliable
- **Process**: Find issues in GitHub, reproduce locally, fix, and submit PR

#### **✨ Feature Development**
- **Difficulty**: Intermediate to Advanced
- **Impact**: Very High - Add new capabilities to the platform
- **Process**: Discuss in issues/Discord, design, implement, test, document

#### **🏗️ Architecture Improvements**
- **Difficulty**: Advanced
- **Impact**: Very High - Improve scalability and performance
- **Process**: Propose RFC, get community feedback, implement gradually

#### **🔧 Developer Tools**
- **Difficulty**: Beginner to Intermediate
- **Impact**: Medium - Improve developer experience
- **Process**: Identify pain points, create solutions, share with community

### **📚 Documentation**

#### **📖 User Guides**
- **Skills Needed**: Technical writing, platform knowledge
- **Impact**: High - Help users adopt the platform
- **Examples**: Tutorials, how-to guides, best practices

#### **👩‍💻 Developer Documentation**
- **Skills Needed**: Programming knowledge, clear communication
- **Impact**: High - Enable other contributors
- **Examples**: API docs, architecture guides, code examples

#### **🌍 Internationalization**
- **Skills Needed**: Language skills, cultural understanding
- **Impact**: Medium - Make platform accessible globally
- **Examples**: UI translations, localized documentation

### **🧪 Quality Assurance**

#### **🔍 Manual Testing**
- **Skills Needed**: Attention to detail, systematic thinking
- **Impact**: High - Ensure platform reliability
- **Process**: Test new features, report bugs, verify fixes

#### **⚡ Automated Testing**  
- **Skills Needed**: Test automation, programming
- **Impact**: Very High - Scale quality assurance
- **Examples**: Unit tests, integration tests, E2E tests

#### **📊 Performance Testing**
- **Skills Needed**: Performance engineering, monitoring
- **Impact**: High - Ensure scalability
- **Examples**: Load testing, benchmarking, optimization

### **🎨 Design & UX**

#### **💡 User Experience Design**
- **Skills Needed**: UX design, user research
- **Impact**: Very High - Make platform intuitive
- **Process**: User research, wireframes, prototypes, usability testing

#### **🎨 Visual Design**
- **Skills Needed**: Graphic design, brand understanding
- **Impact**: Medium - Improve aesthetic appeal
- **Examples**: Icons, illustrations, marketing materials

#### **♿ Accessibility**
- **Skills Needed**: Accessibility standards, inclusive design
- **Impact**: High - Make platform usable by everyone
- **Examples**: WCAG compliance, screen reader support

### **🌐 Community Building**

#### **💬 Community Support**
- **Skills Needed**: Platform knowledge, patience, communication
- **Impact**: High - Help users succeed
- **Process**: Answer questions in Discord/GitHub, create helpful content

#### **📢 Advocacy & Outreach**
- **Skills Needed**: Public speaking, content creation
- **Impact**: Medium - Grow the community
- **Examples**: Blog posts, conference talks, social media

#### **🎓 Education & Training**
- **Skills Needed**: Teaching, curriculum development
- **Impact**: High - Enable platform adoption
- **Examples**: Video tutorials, workshops, certification programs

---

## 🚀 Getting Started

### **Step 1: Choose Your Adventure**

#### **🔰 First-Time Contributors**
1. **Browse [Good First Issues](https://github.com/automotas-ai/automotas/labels/good%20first%20issue)**
2. **Join our [Discord community](https://discord.gg/automotas)**
3. **Read this contributing guide completely**
4. **Set up your development environment**

#### **🏃‍♂️ Experienced Contributors**
1. **Review our [architecture documentation](architecture.md)**
2. **Check current [project roadmap](../PHASE_3_ROADMAP.md)**
3. **Identify high-impact areas for contribution**
4. **Propose significant changes through RFCs**

### **Step 2: Development Environment Setup**

#### **⚡ Quick Setup (Recommended)**
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/automotas.git
cd automotas

# 3. Set up development environment
./scripts/dev-setup.sh  # Automated setup script

# 4. Verify everything works
make test
make dev  # Start development servers
```

#### **🔧 Manual Setup**

##### **Backend Development**
```bash
cd orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up database
createdb automotas_dev
python -c "from context_manager import init_database; init_database()"

# Install pre-commit hooks
pre-commit install

# Start development server
python mcp_bridge.py --reload
```

##### **Frontend Development**
```bash
cd frontend

# Install dependencies
npm install

# Set up environment
cp .env.example .env.local
# Edit .env.local with your configuration

# Start development server
npm run dev
```

##### **Development Tools**
```bash
# Install global development tools
npm install -g typescript eslint prettier
pip install black flake8 mypy pytest

# VS Code extensions (recommended)
code --install-extension ms-python.python
code --install-extension esbenp.prettier-vscode
code --install-extension ms-vscode.vscode-typescript-next
```

### **Step 3: Make Your First Contribution**

#### **📝 Documentation Fix (Great First Contribution)**
```bash
# 1. Find a documentation issue
# 2. Make your changes
git checkout -b docs/fix-installation-guide
# Edit the documentation
git add docs/installation.md
git commit -m "docs: fix Docker installation command"
git push origin docs/fix-installation-guide
# 3. Create pull request on GitHub
```

#### **🐛 Bug Fix**
```bash
# 1. Reproduce the bug locally
# 2. Create a branch
git checkout -b fix/agent-coordination-issue-123

# 3. Fix the bug with tests
# Edit code and add tests
python -m pytest tests/test_agent_coordination.py -v

# 4. Commit with descriptive message
git add .
git commit -m "fix: resolve agent coordination race condition

- Add mutex to prevent concurrent access to shared state
- Include integration test for multi-agent scenarios
- Fixes #123"

# 5. Push and create PR
git push origin fix/agent-coordination-issue-123
```

#### **✨ Feature Implementation**
```bash
# 1. Discuss feature in GitHub issue first
# 2. Create feature branch
git checkout -b feature/advanced-workflow-templates

# 3. Implement with comprehensive tests
# Add feature implementation
# Add unit tests
# Add integration tests
# Update documentation

# 4. Ensure all tests pass
make test-all

# 5. Commit and push
git add .
git commit -m "feat: add advanced workflow template system

- Implement template parser with YAML/JSON support
- Add template validation and error handling  
- Include comprehensive test suite
- Update API documentation
- Add usage examples

Closes #456"

git push origin feature/advanced-workflow-templates
```

---

## 📋 Contribution Standards

### **📝 Commit Message Guidelines**

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear and semantic commit messages:

#### **Format**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### **Types**
- **feat**: New feature
- **fix**: Bug fix  
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring without feature/bug changes
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks, dependency updates
- **ci**: CI/CD pipeline changes

#### **Examples**
```bash
# Good commit messages
feat(agents): add collaborative reasoning capability
fix(api): resolve race condition in workflow execution
docs(readme): update installation instructions for Docker
refactor(context): extract embedding logic to separate module
test(integration): add end-to-end workflow testing

# Bad commit messages (avoid these)
"Fixed bug"
"Updated code" 
"Changes"
"WIP"
```

### **🔄 Pull Request Guidelines**

#### **📋 PR Template**
Every pull request should include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass (if applicable)
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Changelog entry added

## Screenshots (if applicable)
Include screenshots for UI changes.

## Breaking Changes
List any breaking changes and migration steps.

## Additional Notes
Any additional context or notes for reviewers.
```

#### **✅ PR Requirements**

##### **Must Have**
- ✅ **Descriptive title and description**
- ✅ **All tests passing**
- ✅ **Code review from at least one maintainer**
- ✅ **Documentation updated (if applicable)**
- ✅ **No merge conflicts**

##### **Code Quality**
- ✅ **Follows established code style**
- ✅ **Includes appropriate tests**
- ✅ **Has clear, readable code**
- ✅ **Minimal, focused changes**
- ✅ **No debugging code or console logs**

##### **Performance**
- ✅ **No significant performance regressions**
- ✅ **Efficient algorithms and data structures**
- ✅ **Proper resource management**
- ✅ **Memory leaks prevented**

### **🧪 Testing Standards**

#### **📊 Test Coverage Requirements**
- **New Features**: 90%+ test coverage
- **Bug Fixes**: Must include reproduction test
- **Critical Paths**: 100% test coverage
- **Integration Points**: Full integration testing

#### **🔧 Test Types**

##### **Unit Tests**
```python
# Example: Good unit test
import pytest
from unittest.mock import Mock, patch
from orchestrator.agents.strategy_agent import StrategyAgent

class TestStrategyAgent:
    
    @pytest.fixture
    def strategy_agent(self):
        return StrategyAgent(
            name="test_strategy",
            capabilities=["infrastructure_analysis", "deployment_planning"]
        )
    
    @pytest.mark.asyncio
    async def test_can_handle_task_with_matching_capabilities(self, strategy_agent):
        """Test agent can handle tasks matching its capabilities."""
        task = {
            "type": "analyze_infrastructure",
            "required_capabilities": ["infrastructure_analysis"]
        }
        
        result = await strategy_agent.can_handle_task(task)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cannot_handle_task_without_required_capabilities(self, strategy_agent):
        """Test agent rejects tasks requiring missing capabilities."""
        task = {
            "type": "analyze_code",
            "required_capabilities": ["code_analysis"]
        }
        
        result = await strategy_agent.can_handle_task(task)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_infrastructure_analysis_task(self, strategy_agent):
        """Test successful execution of infrastructure analysis."""
        task = {
            "type": "analyze_infrastructure",
            "target_environment": "production",
            "infrastructure_config": {"servers": ["web1", "web2"]}
        }
        
        with patch.object(strategy_agent, '_analyze_infrastructure') as mock_analyze:
            mock_analyze.return_value = {
                "analysis": "Infrastructure analysis complete",
                "recommendations": ["Add load balancer", "Scale web servers"]
            }
            
            result = await strategy_agent.execute_task(task)
            
            assert result.success is True
            assert "recommendations" in result.data
            mock_analyze.assert_called_once_with(task)
```

##### **Integration Tests**
```python
# Example: Good integration test
import pytest
import asyncio
from orchestrator.core.orchestrator import Orchestrator
from orchestrator.agents.strategy_agent import StrategyAgent
from orchestrator.agents.execution_agent import ExecutionAgent

class TestMultiAgentWorkflow:
    
    @pytest.fixture
    async def orchestrator(self):
        orchestrator = Orchestrator()
        
        # Add test agents
        strategy_agent = StrategyAgent(name="strategy", capabilities=["deployment_planning"])
        execution_agent = ExecutionAgent(name="execution", capabilities=["code_deployment"])
        
        await orchestrator.add_agent(strategy_agent)
        await orchestrator.add_agent(execution_agent)
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_complete_deployment_workflow(self, orchestrator):
        """Test complete workflow from planning to execution."""
        workflow_request = {
            "repository_url": "https://github.com/test/test-repo.git",
            "workflow_type": "ai_module",
            "environment": "staging"
        }
        
        # Start workflow
        workflow_id = await orchestrator.create_workflow(workflow_request)
        
        # Wait for workflow completion (with timeout)
        workflow_status = await asyncio.wait_for(
            orchestrator.wait_for_completion(workflow_id),
            timeout=60.0
        )
        
        # Verify workflow completed successfully
        assert workflow_status["status"] == "completed"
        assert workflow_status["agents_involved"] >= 2
        assert "strategy" in [agent["name"] for agent in workflow_status["agent_history"]]
        assert "execution" in [agent["name"] for agent in workflow_status["agent_history"]]
        
        # Verify deployment artifacts were created
        artifacts = await orchestrator.get_workflow_artifacts(workflow_id)
        assert len(artifacts) > 0
        assert any("deployment_plan" in artifact["type"] for artifact in artifacts)
```

##### **End-to-End Tests**
```typescript
// Example: Good E2E test using Playwright
import { test, expect } from '@playwright/test';

test.describe('Workflow Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login and navigate to workflows page
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="password"]', 'testpassword123');
    await page.click('[data-testid="login-button"]');
    await page.waitForURL('/dashboard');
  });

  test('should create and monitor workflow successfully', async ({ page }) => {
    // Navigate to workflows
    await page.click('[data-testid="workflows-nav"]');
    await page.waitForURL('/workflows');

    // Create new workflow
    await page.click('[data-testid="create-workflow"]');
    
    await page.fill('[data-testid="repository-url"]', 'https://github.com/test/sample-app.git');
    await page.selectOption('[data-testid="workflow-type"]', 'ai_module');
    await page.selectOption('[data-testid="environment"]', 'staging');
    
    await page.click('[data-testid="create-workflow-submit"]');
    
    // Wait for workflow to appear in list
    await expect(page.locator('[data-testid="workflow-card"]')).toBeVisible({ timeout: 10000 });
    
    // Verify workflow status updates
    const workflowCard = page.locator('[data-testid="workflow-card"]').first();
    
    // Should start as "queued"
    await expect(workflowCard.locator('[data-testid="status-badge"]')).toContainText('QUEUED');
    
    // Should transition to "running"  
    await expect(workflowCard.locator('[data-testid="status-badge"]')).toContainText('RUNNING', { timeout: 30000 });
    
    // View workflow details
    await workflowCard.click();
    await page.waitForURL(/\/workflows\/[a-f0-9-]+$/);
    
    // Verify workflow details page
    await expect(page.locator('[data-testid="workflow-title"]')).toBeVisible();
    await expect(page.locator('[data-testid="agent-activity"]')).toBeVisible();
    await expect(page.locator('[data-testid="real-time-logs"]')).toBeVisible();
    
    // Verify logs are streaming
    const logsContainer = page.locator('[data-testid="real-time-logs"]');
    await expect(logsContainer.locator('.log-entry')).toHaveCount({ min: 1 });
  });

  test('should handle workflow errors gracefully', async ({ page }) => {
    // Create workflow with invalid repository
    await page.click('[data-testid="workflows-nav"]');
    await page.click('[data-testid="create-workflow"]');
    
    await page.fill('[data-testid="repository-url"]', 'https://github.com/nonexistent/repo.git');
    await page.selectOption('[data-testid="workflow-type"]', 'task_prompt');
    
    await page.click('[data-testid="create-workflow-submit"]');
    
    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Repository not found');
    
    // Workflow should be marked as failed
    const workflowCard = page.locator('[data-testid="workflow-card"]').first();
    await expect(workflowCard.locator('[data-testid="status-badge"]')).toContainText('FAILED');
  });
});
```

### **📚 Documentation Standards**

#### **📖 Documentation Types**

##### **Code Documentation**
```python
# Example: Well-documented Python code
class ContextEngineering:
    """
    Advanced context engineering system for intelligent information retrieval.
    
    This class implements sophisticated retrieval-augmented generation (RAG)
    capabilities with mathematical foundations including Bayesian inference
    and information theory principles.
    
    Attributes:
        retrieval_engine (AdvancedRAGEngine): Core retrieval system
        vector_store (VectorDatabase): Semantic search database
        learning_engine (ContinuousLearner): Adaptive learning system
        
    Example:
        >>> context_engine = ContextEngineering()
        >>> await context_engine.initialize()
        >>> response = await context_engine.process_query("deployment best practices")
        >>> print(f"Quality: {response.quality_score:.2f}")
        Quality: 0.87
    """
    
    def __init__(self, config: ContextConfig = None):
        """
        Initialize the context engineering system.
        
        Args:
            config (ContextConfig, optional): Configuration parameters.
                If None, default configuration is used.
                
        Raises:
            ConfigurationError: If configuration is invalid
            DatabaseConnectionError: If vector database is unavailable
        """
        self.config = config or ContextConfig.default()
        self._validate_config()
        
        # Initialize core components
        self.retrieval_engine = AdvancedRAGEngine(self.config.retrieval)
        self.vector_store = VectorDatabase(self.config.database)
        self.learning_engine = ContinuousLearner(self.config.learning)
    
    async def process_query(
        self,
        query: str,
        context_filters: Dict[str, Any] = None,
        max_results: int = 10
    ) -> ContextResponse:
        """
        Process a natural language query and return relevant context.
        
        This method implements the complete context engineering pipeline:
        1. Query understanding and embedding generation
        2. Semantic search with filtering
        3. Context assembly and quality assessment
        4. Continuous learning from interaction
        
        Args:
            query (str): Natural language query to process
            context_filters (Dict[str, Any], optional): Additional filters
                for context retrieval. Supported keys:
                - 'categories': List of document categories to include
                - 'date_range': Tuple of (start_date, end_date)
                - 'relevance_threshold': Minimum relevance score (0.0-1.0)
            max_results (int, optional): Maximum number of context chunks
                to return. Defaults to 10. Must be between 1 and 50.
        
        Returns:
            ContextResponse: Complete context response including:
                - context: Assembled context string
                - sources: List of source documents with metadata
                - quality_score: Overall quality assessment (0.0-1.0)
                - processing_time: Time taken for processing in seconds
        
        Raises:
            QueryProcessingError: If query cannot be processed
            DatabaseError: If database operations fail
            ValidationError: If parameters are invalid
        
        Example:
            >>> response = await context_engine.process_query(
            ...     "How to deploy microservices securely?",
            ...     context_filters={"categories": ["security", "deployment"]},
            ...     max_results=15
            ... )
            >>> print(f"Found {len(response.sources)} relevant sources")
            Found 12 relevant sources
        """
        # Input validation
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
            
        if max_results < 1 or max_results > 50:
            raise ValidationError("max_results must be between 1 and 50")
        
        start_time = time.time()
        
        try:
            # Step 1: Generate query embedding
            logger.debug(f"Processing query: {query[:50]}...")
            query_embedding = await self._generate_embedding(query)
            
            # Step 2: Perform semantic search
            search_results = await self.retrieval_engine.search(
                embedding=query_embedding,
                filters=context_filters,
                max_results=max_results
            )
            
            # Step 3: Assemble context
            context = await self._assemble_context(search_results, query)
            
            # Step 4: Assess quality
            quality_score = await self._assess_context_quality(context, query)
            
            # Step 5: Learn from interaction
            await self.learning_engine.record_interaction(
                query=query,
                context=context,
                quality_score=quality_score
            )
            
            processing_time = time.time() - start_time
            
            return ContextResponse(
                context=context.assembled_text,
                sources=context.source_documents,
                quality_score=quality_score,
                processing_time=processing_time,
                metadata={
                    "embedding_model": self.config.embedding_model,
                    "search_results_count": len(search_results),
                    "filters_applied": context_filters or {}
                }
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise QueryProcessingError(f"Failed to process query: {str(e)}") from e
```

##### **API Documentation**
```yaml
# Example: OpenAPI specification
openapi: 3.0.3
info:
  title: Automatos AI API
  description: |
    ## Advanced Multi-Agent Orchestration Platform API
    
    The Automatos AI API provides comprehensive endpoints for managing
    workflows, agents, documents, and system configuration in our
    intelligent automation platform.
    
    ### Key Features
    - **Multi-Agent Coordination**: Orchestrate complex workflows across multiple AI agents
    - **Context Engineering**: Advanced RAG system with semantic search
    - **Real-time Monitoring**: WebSocket-based live updates and monitoring
    - **Enterprise Security**: Banking-grade security with comprehensive audit trails
    
    ### Authentication
    All API endpoints require authentication via API key in the `X-API-Key` header.
    
    ### Rate Limiting
    API requests are limited to 1000 requests per minute per API key.
    
    ### Error Handling
    The API uses standard HTTP status codes and returns detailed error information
    in a consistent JSON format.
  version: 2.0.0
  contact:
    name: Automatos AI Support
    email: api-support@automotas.ai
    url: https://docs.automotas.ai
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.automotas.ai/v2
    description: Production server
  - url: https://staging-api.automotas.ai/v2
    description: Staging server
  - url: http://localhost:8002/v2
    description: Development server

paths:
  /workflows:
    post:
      summary: Create new workflow
      description: |
        Create a new workflow for automated deployment and orchestration.
        
        This endpoint supports two types of workflows:
        1. **AI Module Workflows**: For repositories with `ai-module.yaml` configuration
        2. **Task Prompt Workflows**: For repositories using natural language task descriptions
        
        ### Workflow Types
        
        #### AI Module Workflow
        Automatically detects and processes repositories with `ai-module.yaml` configuration files.
        
        #### Task Prompt Workflow  
        Uses natural language processing to understand deployment requirements from
        free-form task descriptions.
        
        ### Processing Pipeline
        1. **Repository Analysis**: Clone and analyze repository structure
        2. **Agent Assignment**: Automatically assign appropriate agents based on requirements
        3. **Strategy Planning**: Generate optimal deployment strategy
        4. **Security Validation**: Comprehensive security and compliance checking
        5. **Execution**: Automated deployment with real-time monitoring
        6. **Optimization**: Continuous optimization based on performance metrics
      operationId: createWorkflow
      tags:
        - Workflows
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateWorkflowRequest'
            examples:
              ai_module_workflow:
                summary: AI Module Workflow
                description: Workflow for repository with ai-module.yaml configuration
                value:
                  repository_url: "https://github.com/example/my-ai-app.git"
                  workflow_type: "ai_module"
                  environment: "production"
                  priority: "high"
                  notification_channels: ["slack", "email"]
              task_prompt_workflow:
                summary: Task Prompt Workflow
                description: Workflow with natural language task description
                value:
                  repository_url: "https://github.com/example/legacy-app.git"
                  workflow_type: "task_prompt"
                  task_prompt: "Deploy Python Flask application with Redis cache to AWS ECS with auto-scaling and monitoring"
                  environment: "staging"
                  priority: "normal"
      responses:
        '201':
          description: Workflow created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/WorkflowResponse'
              example:
                success: true
                data:
                  workflow_id: "wf_2024_abc123def456"
                  status: "queued"
                  repository_url: "https://github.com/example/my-app.git"
                  workflow_type: "ai_module"
                  environment: "production"
                  estimated_duration: "15-20 minutes"
                  assigned_agents:
                    - id: "agent_strategy_001"
                      name: "Primary Strategy Agent"
                      type: "strategy"
                    - id: "agent_security_001"
                      name: "Security Validator"
                      type: "security"
                  created_at: "2024-01-15T10:30:00Z"
                  updated_at: "2024-01-15T10:30:00Z"
                timestamp: "2024-01-15T10:30:00Z"
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '422':
          $ref: '#/components/responses/ValidationError'
        '429':
          $ref: '#/components/responses/RateLimit'
        '500':
          $ref: '#/components/responses/InternalError'

components:
  schemas:
    CreateWorkflowRequest:
      type: object
      required:
        - repository_url
        - workflow_type
      properties:
        repository_url:
          type: string
          format: uri
          description: |
            Git repository URL to process. Supported protocols:
            - HTTPS: `https://github.com/user/repo.git`
            - SSH: `git@github.com:user/repo.git`
            
            Supported platforms:
            - GitHub, GitLab, Bitbucket
            - Private repositories (with appropriate credentials)
          example: "https://github.com/example/my-application.git"
        workflow_type:
          type: string
          enum: [ai_module, task_prompt]
          description: |
            Type of workflow to create:
            - `ai_module`: Repository with ai-module.yaml configuration
            - `task_prompt`: Natural language task description
        task_prompt:
          type: string
          maxLength: 2000
          description: |
            Natural language description of deployment requirements.
            Required for task_prompt workflows, ignored for ai_module workflows.
            
            Should include:
            - Application type and technology stack
            - Target environment and platform
            - Specific requirements (scaling, monitoring, etc.)
          example: "Deploy Node.js Express API with PostgreSQL database to AWS ECS with auto-scaling, SSL termination, and CloudWatch monitoring"
        environment:
          type: string
          enum: [development, staging, production]
          default: development
          description: Target deployment environment
        priority:
          type: string
          enum: [low, normal, high, critical]
          default: normal
          description: |
            Workflow execution priority:
            - `low`: Best effort execution
            - `normal`: Standard priority (default)
            - `high`: Prioritized execution
            - `critical`: Immediate execution with resource allocation
        notification_channels:
          type: array
          items:
            type: string
            enum: [email, slack, teams, webhook]
          description: Channels for workflow status notifications
          example: ["slack", "email"]
        metadata:
          type: object
          additionalProperties: true
          description: Additional metadata for workflow tracking
          example:
            project: "web-platform"
            team: "backend-team"
            cost_center: "engineering"

  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: |
        API key authentication. Include your API key in the `X-API-Key` header.
        
        Example: `X-API-Key: ak_1234567890abcdef`
        
        To obtain an API key:
        1. Log into the Automatos AI dashboard
        2. Navigate to Settings > API Keys
        3. Generate a new API key with appropriate permissions

security:
  - ApiKeyAuth: []
```

##### **User Documentation**
````markdown
# Creating Your First Workflow

Learn how to create and manage automated deployment workflows using Automatos AI.

## What You'll Learn

By the end of this guide, you'll know how to:
- ✅ Create different types of workflows
- ✅ Monitor workflow execution in real-time
- ✅ Troubleshoot common issues
- ✅ Optimize workflow performance

## Prerequisites

Before starting, ensure you have:
- ✅ Automatos AI account with API access
- ✅ Git repository with your application code
- ✅ Basic understanding of your application's deployment needs

## Step 1: Choose Your Workflow Type

Automatos AI supports two types of workflows:

### 🤖 AI Module Workflows (Recommended)

**Best for**: Modern applications with structured configuration

AI Module workflows use a `ai-module.yaml` file to define deployment specifications.

#### Create ai-module.yaml

Create this file in your repository root:

```yaml
# ai-module.yaml
name: "my-web-application"
version: "1.2.0"
description: "High-performance web application with API backend"

# Application configuration
module_type: "web_app"
framework: "nodejs"
runtime_version: "18"

# Build configuration
build:
  command: "npm install && npm run build"
  output_dir: "dist"
  environment:
    NODE_ENV: "production"

# Runtime configuration
runtime:
  start_command: "npm start"
  port: 3000
  health_check: "/health"

# Infrastructure requirements
infrastructure:
  cpu: "512m"
  memory: "1Gi"
  replicas:
    min: 2
    max: 10
  autoscaling:
    target_cpu: 70
    target_memory: 80

# Dependencies
dependencies:
  database:
    type: "postgresql"
    version: "14"
  cache:
    type: "redis"
    version: "7"

# Security configuration
security:
  https_only: true
  security_headers: true
  rate_limiting:
    requests_per_minute: 1000

# Monitoring
monitoring:
  enabled: true
  metrics: ["response_time", "error_rate", "throughput"]
  alerts:
    - condition: "response_time > 1000ms"
      severity: "warning"
    - condition: "error_rate > 5%"
      severity: "critical"
```

#### Deploy AI Module Workflow

```bash
# Using CLI
automatos workflow create \
  --repository https://github.com/yourusername/your-app.git \
  --type ai_module \
  --environment production

# Using API
curl -X POST https://api.automotas.ai/v2/workflows \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "repository_url": "https://github.com/yourusername/your-app.git",
    "workflow_type": "ai_module",
    "environment": "production",
    "priority": "high"
  }'
```

### 📝 Task Prompt Workflows

**Best for**: Legacy applications or when you prefer natural language descriptions

Task Prompt workflows use AI to understand deployment requirements from natural language.

#### Example Task Prompts

**Simple Web Application**:
```
Deploy a React frontend application with nginx reverse proxy to AWS ECS. 
Enable HTTPS with automatic SSL certificate management and CloudFront CDN.
Set up auto-scaling based on CPU usage with minimum 2 instances.
```

**Complex Microservices**:
```
Deploy a microservices architecture with:
- Node.js API gateway on port 3000
- Python FastAPI user service with PostgreSQL database
- Redis cache for session management
- RabbitMQ for async messaging
- Prometheus monitoring with Grafana dashboards
- Deploy to Kubernetes with Helm charts
- Enable horizontal pod autoscaling
- Set up ingress with SSL termination
```

**Data Pipeline**:
```
Create a data processing pipeline:
- Python ETL service reading from AWS S3
- Apache Airflow for workflow orchestration  
- Data validation and transformation
- Output to PostgreSQL data warehouse
- Monitoring with custom metrics
- Error handling and retry logic
- Deploy with Docker containers to ECS
```

#### Deploy Task Prompt Workflow

```bash
# Using CLI
automatos workflow create \
  --repository https://github.com/yourusername/legacy-app.git \
  --type task_prompt \
  --prompt "Deploy PHP Laravel application with MySQL database and Redis cache to AWS EC2 with load balancer and SSL" \
  --environment staging

# Using Dashboard
# 1. Navigate to Workflows > Create New
# 2. Select "Task Prompt" workflow type
# 3. Enter repository URL
# 4. Write detailed task description
# 5. Choose environment and priority
# 6. Click "Create Workflow"
```

## Step 2: Monitor Workflow Execution

### Real-time Dashboard

1. **Navigate to Workflows**: Go to the workflows section in your dashboard
2. **Find Your Workflow**: Locate your newly created workflow
3. **View Details**: Click on the workflow to see detailed progress

### Live Updates

The dashboard provides real-time updates including:
- **🎯 Current Stage**: Which agents are currently active
- **📊 Progress**: Percentage completion and estimated time remaining
- **📝 Live Logs**: Streaming logs from all agents
- **🔍 Agent Activity**: What each agent is currently doing
- **⚡ Performance**: Resource usage and optimization metrics

### WebSocket Integration

For custom integrations, connect to live updates:

```javascript
// Connect to workflow updates
const ws = new WebSocket(`wss://api.automotas.ai/v2/ws/workflows/${workflowId}`);

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Workflow update:', update);
  
  switch (update.type) {
    case 'status_change':
      console.log(`Status: ${update.status}`);
      break;
    case 'agent_activity':
      console.log(`${update.agent}: ${update.activity}`);
      break;
    case 'log_entry':
      console.log(`Log: ${update.message}`);
      break;
  }
};
```

## Step 3: Handle Common Scenarios

### ✅ Successful Deployment

When your workflow completes successfully:

1. **Verify Deployment**: Check that your application is running correctly
2. **Review Metrics**: Examine performance and resource usage
3. **Save Configuration**: The platform automatically saves successful configurations for future use
4. **Monitor Health**: Set up ongoing monitoring and alerting

### ⚠️ Partial Success

If some components deploy successfully but others fail:

1. **Review Logs**: Check the detailed logs for failed components
2. **Fix Issues**: Address the root cause (code bugs, configuration errors, etc.)
3. **Retry Failed Components**: Use selective retry to fix only failed parts
4. **Update Configuration**: Modify ai-module.yaml or task prompt as needed

### ❌ Deployment Failure

If the entire workflow fails:

1. **Analyze Error Messages**: Review the comprehensive error reports
2. **Check Prerequisites**: Verify repository access, environment configuration
3. **Validate Configuration**: Ensure ai-module.yaml is valid or task prompt is clear
4. **Consult Documentation**: Review specific error code documentation
5. **Contact Support**: Reach out if you need additional help

## Step 4: Optimize Performance

### Configuration Optimization

Based on deployment results, optimize your configuration:

```yaml
# Optimized ai-module.yaml based on performance data
infrastructure:
  # Increase resources if CPU/memory usage is high
  cpu: "1"      # Increased from 512m
  memory: "2Gi" # Increased from 1Gi
  
  # Adjust scaling parameters based on traffic patterns
  replicas:
    min: 3      # Increased from 2 for better availability
    max: 20     # Increased from 10 for peak traffic
    
  autoscaling:
    target_cpu: 60     # Reduced from 70 for faster scaling
    target_memory: 70  # Reduced from 80 for memory headroom

# Add caching for better performance
caching:
  enabled: true
  type: "application"
  ttl: 300
  redis_url: "redis://cache-cluster:6379"

# Optimize database connections
database:
  connection_pool:
    min: 5
    max: 50
  query_timeout: 30000
```

### Monitoring and Alerting

Set up comprehensive monitoring:

```yaml
# Enhanced monitoring configuration
monitoring:
  enabled: true
  metrics:
    - "response_time"
    - "error_rate"
    - "throughput"
    - "database_connections"
    - "cache_hit_rate"
    - "memory_usage"
    - "cpu_usage"
  
  alerts:
    # Performance alerts
    - name: "High Response Time"
      condition: "response_time > 500ms for 5 minutes"
      severity: "warning"
      channels: ["slack", "email"]
    
    - name: "Critical Response Time"
      condition: "response_time > 2000ms for 2 minutes"
      severity: "critical"
      channels: ["slack", "email", "pagerduty"]
    
    # Error alerts
    - name: "Elevated Error Rate"
      condition: "error_rate > 1% for 10 minutes"
      severity: "warning"
    
    - name: "High Error Rate"
      condition: "error_rate > 5% for 5 minutes"
      severity: "critical"
    
    # Resource alerts
    - name: "High CPU Usage"
      condition: "cpu_usage > 80% for 15 minutes"
      severity: "warning"
    
    - name: "Memory Leak Detection"
      condition: "memory_usage increasing for 30 minutes"
      severity: "warning"
```

## Best Practices

### 🎯 Configuration Management

1. **Version Control**: Keep ai-module.yaml in version control
2. **Environment Separation**: Use different configurations for dev/staging/prod
3. **Secrets Management**: Use secure secret management for sensitive data
4. **Documentation**: Document configuration decisions and trade-offs

### 🚀 Performance Optimization

1. **Right-sizing**: Start conservative, scale based on actual usage
2. **Caching Strategy**: Implement multi-layer caching where appropriate
3. **Database Optimization**: Index frequently queried fields
4. **CDN Usage**: Use CDN for static assets and global distribution

### 🛡️ Security Best Practices

1. **HTTPS Everywhere**: Always use HTTPS in production
2. **Security Headers**: Enable security headers for web applications
3. **Access Control**: Implement proper authentication and authorization
4. **Regular Updates**: Keep dependencies and base images updated

### 📊 Monitoring Strategy

1. **Comprehensive Metrics**: Monitor all critical system components
2. **Meaningful Alerts**: Set alerts that indicate real problems
3. **Performance Baselines**: Establish baselines for normal operation
4. **Regular Review**: Regularly review and update monitoring configuration

## Next Steps

Now that you've created your first workflow:

1. **📚 [Advanced Features](advanced-workflows.md)**: Learn about advanced workflow capabilities
2. **🔧 [Agent Management](agent-management.md)**: Customize agent behavior and capabilities  
3. **📊 [Analytics Guide](analytics-guide.md)**: Deep dive into performance analytics
4. **🛡️ [Security Configuration](security-guide.md)**: Advanced security and compliance setup
5. **🤝 [Team Collaboration](team-guide.md)**: Set up team workflows and permissions

## Getting Help

If you encounter any issues:

- **📖 [Documentation](https://docs.automotas.ai)**: Comprehensive guides and references
- **💬 [Community Discord](https://discord.gg/automotas)**: Get help from the community
- **🐛 [GitHub Issues](https://github.com/automotas-ai/automotas/issues)**: Report bugs or request features
- **📧 [Support Email](mailto:support@automotas.ai)**: Direct support for complex issues
- **📞 [Enterprise Support](https://automotas.ai/enterprise)**: Priority support for enterprise customers

---

**🎉 Congratulations!** You've successfully created your first Automatos AI workflow. You're now ready to automate complex deployments with confidence.
````

---

## 🎖️ Recognition & Rewards

### **🏆 Contributor Levels**

#### **🌱 New Contributor**
- **Requirements**: First merged PR
- **Benefits**: 
  - Welcome package with stickers
  - Discord contributor badge
  - Listed in contributor credits

#### **⭐ Regular Contributor**  
- **Requirements**: 5+ merged PRs, active participation
- **Benefits**:
  - Exclusive contributor t-shirt
  - Early access to new features
  - Monthly contributor call invitations

#### **🚀 Core Contributor**
- **Requirements**: 25+ merged PRs, significant feature contributions
- **Benefits**:
  - Core contributor hoodie
  - Beta testing opportunities
  - Input on product roadmap
  - Speaking opportunities at events

#### **💎 Maintainer**
- **Requirements**: Long-term commitment, technical leadership
- **Benefits**:
  - Maintainer compensation
  - Conference speaking opportunities
  - Direct input on strategic decisions
  - Recognition on website and materials

### **🎁 Reward System**

#### **Monthly Recognition**
- **Top Contributor**: Featured on website and social media
- **Best Bug Fix**: Recognition for most impactful bug resolution  
- **Documentation Hero**: Best documentation contribution
- **Community Helper**: Most helpful in community support

#### **Annual Awards**
- **Outstanding Contributor Award**: Significant long-term impact
- **Innovation Award**: Most creative or innovative contribution
- **Community Impact Award**: Greatest positive impact on community
- **Rising Star Award**: Most promising new contributor

### **🌟 Special Programs**

#### **Mentorship Program**
- **For New Contributors**: Paired with experienced maintainers
- **For Students**: Special student contributor track with learning resources
- **For Open Source Newcomers**: Introduction to open source contribution

#### **Speaker Program**
- **Conference Sponsorship**: Travel and accommodation for speaking
- **Workshop Development**: Create and deliver technical workshops
- **Content Creation**: Blog posts, videos, and educational materials

---

## 📞 Getting Help

### **🤝 Community Support**

#### **Discord Community**
- **#general**: General discussions and questions
- **#development**: Technical development discussions  
- **#documentation**: Documentation improvements and questions
- **#feature-requests**: Propose and discuss new features
- **#help**: Get help from the community
- **#showcase**: Show off your contributions and projects

#### **GitHub Discussions**
- **Ideas**: Brainstorm new features and improvements
- **Q&A**: Technical questions and answers
- **Show and Tell**: Demonstrate your work and projects
- **General**: Broader discussions about the project

### **🆘 Direct Support**

#### **For Contributors**
- **Email**: [contributors@automotas.ai](mailto:contributors@automotas.ai)
- **Office Hours**: Weekly contributor office hours (Thursdays 3-4 PM UTC)
- **Mentorship**: Request a mentor for guidance and support

#### **For Maintainers**
- **Maintainer Chat**: Private Discord channel for maintainers
- **Monthly Sync**: Regular video calls for coordination
- **Emergency Contact**: Direct contact for urgent issues

### **📚 Learning Resources**

#### **Development Resources**
- **[Development Setup Guide](DEVELOPMENT.md)**: Complete development environment setup
- **[Architecture Overview](architecture.md)**: Understanding the system architecture
- **[API Documentation](API_REFERENCE.md)**: Complete API reference
- **[Code Examples](examples/)**: Practical code examples and patterns

#### **Open Source Resources**
- **[First Contributions](https://firstcontributions.github.io/)**: Learn Git and GitHub basics
- **[How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)**: General open source contribution guide
- **[Code Review Best Practices](https://google.github.io/eng-practices/review/)**: Google's code review guidelines

---

## 📈 Roadmap & Future Plans

### **🎯 Immediate Goals (Next 3 Months)**
- **Growing Community**: Reach 1000+ contributors
- **Documentation Expansion**: Complete all missing documentation
- **Testing Infrastructure**: Achieve 95%+ test coverage
- **Performance Optimization**: 50% improvement in core metrics

### **🚀 Medium-term Goals (6-12 Months)**
- **Multi-language Support**: Add support for 5+ programming languages
- **Advanced Features**: Implement Phase 3 roadmap features
- **Global Community**: Establish regional contributor communities
- **Educational Content**: Comprehensive learning track for contributors

### **🌟 Long-term Vision (1-2 Years)**
- **Industry Leadership**: Become the go-to platform for AI orchestration
- **Academic Partnerships**: Collaborate with universities on research
- **Enterprise Adoption**: Power automation for Fortune 500 companies
- **Open Source Ecosystem**: Thriving ecosystem of plugins and extensions

---

**Ready to make your mark on the future of AI automation?**

🚀 **[Start Contributing Today](https://github.com/automotas-ai/automotas/issues/good%20first%20issue)**

*Join thousands of developers building the future of intelligent automation.*

---

*This contributing guide is a living document. Help us improve it by submitting suggestions and updates!*
