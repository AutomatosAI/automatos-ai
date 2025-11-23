# Automatos AI 🤖

**Advanced AI Agent Management Platform for Enterprise Automation**

Automatos AI is a powerful, enterprise-grade platform for creating, managing, and orchestrating AI agents across your organization. Built with modern technologies and designed for scalability, security, and performance.

## 🚀 Features

### 🎯 **Agent Management**
- **Multi-type AI Agents**: Code architects, security experts, performance optimizers, data analysts
- **Dynamic Agent Orchestration**: Auto-scaling and load balancing
- **Real-time Status Monitoring**: Live agent health and performance metrics
- **Bulk Operations**: Create and manage multiple agents efficiently

### 🧠 **Context Engineering**
- **RAG (Retrieval Augmented Generation)**: Advanced document processing and retrieval
- **Vector Embeddings**: Semantic search and knowledge extraction
- **Document Processing**: PDF, DOCX, and text analysis
- **Intelligent Chunking**: Optimized content segmentation

### 🏗️ **Enterprise Architecture**
- **FastAPI Backend**: High-performance async API
- **PostgreSQL + pgvector**: Vector database for AI operations
- **Redis**: High-speed caching and session management
- **Docker**: Containerized deployment
- **Next.js Frontend**: Modern, responsive web interface

### 🔧 **Developer Experience**
- **OpenAPI Documentation**: Auto-generated API docs
- **Type Safety**: Full TypeScript/Python type coverage
- **Database Migrations**: Alembic-powered schema management
- **Testing Framework**: Comprehensive test suite
- **Code Quality**: Black, isort, pytest integration

## 🛠️ Quick Start

Get Automatos AI running in 3 simple steps:

### Prerequisites
- **Docker** & **Docker Compose** (latest versions)
- **Git**
- Optional: OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1️⃣ Clone & Configure
```bash
git clone https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai
cp .env.example .env
# Optional: Edit .env and add your OPENAI_API_KEY
```

### 2️⃣ Start Everything
```bash
docker-compose up
```
⏳ First startup takes 2-3 minutes (building images, loading seed data)

### 3️⃣ Access the Platform
- **🌐 Frontend**: http://localhost:3000
- **📚 API Docs**: http://localhost:8000/docs
- **❤️ Health Check**: http://localhost:8000/health

**That's it!** 🎉 The platform is ready to use.

### Optional: Monitoring & Admin Tools
```bash
# Add admin tools (Adminer for database management)
docker-compose --profile all up

# Add everything (includes Adminer for database management)
docker-compose --profile all up
```

### Troubleshooting
See [Quick Start Guide](docs/QUICKSTART.md) for detailed setup instructions and troubleshooting.

## 📁 Project Structure

```
automatos-ai/
├── orchestrator/          # Backend API & Services
│   ├── src/              # Source code
│   │   ├── api/          # FastAPI routes
│   │   ├── database/     # Models & database
│   │   └── services/     # Business logic
│   ├── alembic/          # Database migrations
│   ├── tests/            # Test suite
│   └── main.py           # Application entry point
├── frontend/             # Next.js web interface
├── docs/                 # Documentation
└── docker-compose.yml    # Container orchestration
```

## 🔌 API Endpoints

### Core Operations
- `GET /health` - System health check
- `GET /api/agents` - List all agents
- `POST /api/agents` - Create new agent
- `GET /api/agents/{id}/status` - Agent status
- `POST /api/agents/{id}/execute` - Execute agent

### Management
- `GET /api/agents/types` - Available agent types
- `GET /api/agents/stats` - System statistics
- `POST /api/agents/bulk` - Bulk operations

### Context Engineering
- `GET /api/context/stats` - RAG system metrics
- `POST /api/documents` - Upload documents
- `GET /api/skills` - Available skills

## 🧪 Testing

```bash
# Run all tests
cd orchestrator
python -m pytest

# Run with coverage
python -m pytest --cov=orchestrator

# Async tests
python -m pytest tests/test_agents.py -v
```

## 🚀 Deployment

### Production Docker
```bash
# Build optimized images
docker compose -f docker-compose.prod.yml build

# Deploy with environment config
docker compose -f docker-compose.prod.yml up -d
```

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/automatos_ai
POSTGRES_DB=automatos_ai
POSTGRES_USER=automatos_user
POSTGRES_PASSWORD=your_secure_password

# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your_jwt_secret
API_KEY=your_internal_api_key
```

## 📖 Documentation

### Quick Start Guides
- **[Quick Start Guide](docs/quickstart.md)**: Get started in 5 minutes
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Development setup and workflows
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: Production deployment

### Core Platform Guides
- **[Agent System Guide](docs/AGENT_SYSTEM_GUIDE.md)**: Agent creation, multi-model support, LLM-driven orchestration
- **[Workflow Orchestration Guide](docs/WORKFLOW_SYSTEM_GUIDE.md)**: 9-stage intelligent workflow pipeline
- **[Context Engineering Guide](docs/CONTEXT_ENGINEERING_GUIDE.md)**: RAG, token optimization, mathematical foundations
- **[Tools & Integration Guide](docs/TOOLS_INTEGRATION_GUIDE.md)**: 400+ MCP integrations, credential management
- **[Memory & Knowledge Guide](docs/MEMORY_KNOWLEDGE_GUIDE.md)**: Hierarchical memory, knowledge graphs, multimodal KB

### Advanced Features
- **[Agent Communication Guide](docs/AGENT_COMMUNICATION_MONITORING_GUIDE.md)**: Inter-agent messaging and shared context
- **[Playbooks Guide](docs/PLAYBOOKS_GUIDE.md)**: Automated pattern discovery and learning
- **[CodeGraph Guide](docs/CODEGRAPH_GUIDE.md)**: Code understanding and semantic analysis
- **[Credential System Guide](docs/CREDENTIAL_SYSTEM_GUIDE.md)**: Secure credential management

### Reference
- **[API Documentation](http://localhost:8000/docs)**: Interactive OpenAPI docs
- **[Comprehensive Guide](docs/COMPREHENSIVE_GUIDE.md)**: Complete platform overview
- **[Architecture](docs/architecture.md)**: System design and components

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

Copyright (c) 2025 Automatos AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 🏢 Enterprise Support

For enterprise licensing, commercial support, and custom development:
- **Website**: [https://automatos.ai](https://automatos.ai)
- **Email**: enterprise@automatos.ai
- **Documentation**: [https://docs.automatos.ai](https://docs.automatos.ai)

---

**Built with ❤️ by the Automatos AI Team**
