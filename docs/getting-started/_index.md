# Getting Started

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/test.yml](.github/workflows/test.yml)
- [docker-compose.yml](docker-compose.yml)
- [frontend/.dockerignore](frontend/.dockerignore)
- [frontend/Dockerfile](frontend/Dockerfile)
- [infrastructure/.env.example](infrastructure/.env.example)
- [infrastructure/railway-manifest.json](infrastructure/railway-manifest.json)
- [orchestrator/Dockerfile](orchestrator/Dockerfile)
- [orchestrator/core/redis/client.py](orchestrator/core/redis/client.py)
- [orchestrator/requirements.txt](orchestrator/requirements.txt)
- [orchestrator/tests/test_dockerfile_prod_parity.py](orchestrator/tests/test_dockerfile_prod_parity.py)

</details>



This page provides a high-level overview of installing, configuring, and launching Automatos AI. It guides new users through the initial setup process, system architecture, core services, and onboarding flows. 

Automatos AI operates as an agentic operating system combining multi-agent orchestration, intelligent message routing, layered memory, and autonomous workspace capabilities [docker-compose.yml:1-24]().

For detailed instructions, see the child pages:
- **[Installation & Setup](#2.1)** — Docker Compose configuration, multi-stage container builds, environment variables, and dependency initialization [docker-compose.yml:1-159](), [orchestrator/Dockerfile:1-120](), [frontend/Dockerfile:1-50]().
- **[Configuration Guide](#2.2)** — Configuring LLM providers, Redis, PostgreSQL, MinIO/S3, and core service parameters via `orchestrator/config.py` and environment seeds [infrastructure/.env.example:1-174]().
- **[Quick Start Tutorial](#2.3)** — Step-by-step workflows for creating agents, connecting tools, executing chats, and running playbooks.
- **[Business Intake Wizard](#2.4)** — PRD-130 onboarding wizard covering domain entry, Firecrawl website scanning, RAG ingestion, Graphify knowledge graph building, and Mission Zero plan generation.

---

## System Architecture Overview

Automatos AI relies on a containerized multi-tier architecture powered by FastAPI on the backend and Next.js on the frontend, supported by PostgreSQL with `pgvector`, Redis, and MinIO [docker-compose.yml:26-159](). 

### System Entity Map (Natural Language to Code Entity Space)

The following diagram bridges user interaction points to their corresponding backend services, database models, and container endpoints.

```mermaid
graph TB
    subgraph "NaturalLanguageSpace"
        UI[""UserBrowser"<br/>frontend:3000<br/>apiClient.ts""]
    end
    
    subgraph "CodeEntitySpace"
        API[""FastAPIBackend"<br/>backend:8000<br/>orchestrator/main.py""]
        Worker[""WorkspaceWorker"<br/>agent-workspace-worker<br/>WorkspaceWorker_ARQ""]
        DB[""PostgresDatabase"<br/>postgres:5432<br/>SQLAlchemy_Models""]
        Cache[""RedisCache"<br/>redis:6379<br/>core/redis/client.py""]
        Storage[""S3Storage"<br/>minio:9000<br/>Object_Storage""]
    end
    
    UI -->|"ClerkJWT/APIRequest"| API
    API -->|"SessionLocal"| DB
    API -->|"RedisClient"| Cache
    API -->|"S3Endpoint"| Storage
    API -->|"ARQQueue"| Worker
    Worker -->|"DBConnection"| DB
```

Sources: [docker-compose.yml:26-159](), [orchestrator/core/redis/client.py:14-36](), [frontend/Dockerfile:14-48]()

---

## 1. Installation & Setup

Automatos AI is deployed primarily via Docker Compose, coordinating core infrastructure services including the FastAPI orchestrator, Next.js frontend, PostgreSQL (`pgvector`), Redis, and MinIO [docker-compose.yml:26-159](). Multi-stage Dockerfiles ensure minimal production footprints while preserving development hot-reload features [orchestrator/Dockerfile:1-132](), [frontend/Dockerfile:1-49]().

For detailed instructions on local container orchestration, environment variables, database schema migrations via Alembic, and volume management, see **[Installation & Setup](#2.1)**.

Sources: [docker-compose.yml:1-159](), [orchestrator/Dockerfile:1-132](), [frontend/Dockerfile:1-49]()

---

## 2. Configuration Guide

Platform services are governed by centralized environment variables and configuration files [infrastructure/.env.example:1-174](). Essential configuration parameters include database connection strings, Redis channels, encryption keys for credentials, LLM provider API keys (OpenAI, Anthropic, OpenRouter), and tool integrations like Composio [infrastructure/.env.example:17-116]().

### Service Configuration Flow

```mermaid
graph LR
    Env["".env.example / .env"<br/>infrastructure/.env.example""] --> Config[""ConfigManager"<br/>orchestrator/config.py""]
    Config --> DB[""PostgreSQL / pgvector"<br/>orchestrator_db""]
    Config --> Redis[""Redis Cache & PubSub"<br/>orchestrator/core/redis/client.py""]
    Config --> LLM[""LLM Managers"<br/>OpenAI / Anthropic / OpenRouter""]
```

For detailed setup instructions covering provider resolution tiers, vector store configurations (Qdrant/pgvector), and system settings seeds, see **[Configuration Guide](#2.2)**.

Sources: [infrastructure/.env.example:1-174](), [orchestrator/core/redis/client.py:141-194]()

---

## 3. Quick Start Tutorial

Once services are running, new users can immediately interact with the platform. Every fresh workspace seeds a primary orchestration agent named `Auto` which acts as the core conversational partner and default router [orchestrator/core/seeds/seed_auto_agent.py:5-16](). 

Users can prompt agents, connect external tool integrations, test streaming chat responses, and trigger multi-agent workflows through the web UI [docker-compose.yml:4-10]().

For a step-by-step walkthrough covering agent creation, tool linking, and execution playbooks, see **[Quick Start Tutorial](#2.3)**.

Sources: [docker-compose.yml:4-10](), [orchestrator/core/seeds/seed_auto_agent.py:5-16]()

---

## 4. Business Intake Wizard

To streamline tenant onboarding, Automatos AI includes the **Business Intake Wizard** (PRD-130) [frontend/components/onboarding/welcome-modal.tsx:51-141](). When a user inputs their domain, the onboarding flow executes a Firecrawl website scan, ingests documentation into the RAG pipeline, builds a knowledge graph via Graphify, and generates a "Mission Zero" bootstrap plan.

For comprehensive details on the onboarding state machine, wizard components, and workspace initialization logic, see **[Business Intake Wizard](#2.4)**.

Sources: [frontend/components/onboarding/welcome-modal.tsx:51-141](), [orchestrator/main.py:61-61]()

---