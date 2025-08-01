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

### Prerequisites
- Docker & Docker Compose
- Git
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)

### 1. Clone and Setup
```bash
git clone git@github.com:AutomatosAI/automatos-ai.git
cd automatos-ai/orchestrator
cp .env.example .env  # Configure your environment
```

### 2. Start Services
```bash
# Start all services
docker compose up -d

# Backend only (for API development)
docker compose up -d postgres redis backend_api
```

### 3. Access the Platform
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000 (when built)

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
python -m pytest --cov=src

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

- **[Architecture Guide](./orchestrator/ARCHITECTURE.md)**: System design and components
- **[Developer Guide](./orchestrator/DEVELOPER_GUIDE.md)**: Development setup and workflows
- **[API Documentation](http://localhost:8000/docs)**: Interactive OpenAPI docs

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
