# Getting Started

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/test.yml](../../.github/workflows/test.yml)
- [docker-compose.yml](../../docker-compose.yml)
- [frontend/.dockerignore](../../frontend/.dockerignore)
- [frontend/Dockerfile](../../frontend/Dockerfile)
- [infrastructure/.env.example](../../infrastructure/.env.example)
- [infrastructure/railway-manifest.json](../../infrastructure/railway-manifest.json)
- [orchestrator/Dockerfile](../../orchestrator/Dockerfile)
- [orchestrator/core/redis/client.py](../../orchestrator/core/redis/client.py)
- [orchestrator/requirements.txt](../../orchestrator/requirements.txt)
- [orchestrator/tests/test_dockerfile_prod_parity.py](../../orchestrator/tests/test_dockerfile_prod_parity.py)

</details>



This page provides a high-level overview of installing, configuring, and launching Automatos AI. It guides new users through the initial setup process, system architecture, core services, and onboarding flows. 

Automatos AI operates as an agentic operating system combining multi-agent orchestration, intelligent message routing, layered memory, and autonomous workspace capabilities [docker-compose.yml:1-24](../../docker-compose.yml#L1-L24).

For detailed instructions, see the child pages:
- **[Installation & Setup](installation-setup.md)** — Docker Compose configuration, multi-stage container builds, environment variables, and dependency initialization [docker-compose.yml:1-159](../../docker-compose.yml#L1-L159), [orchestrator/Dockerfile:1-120](../../orchestrator/Dockerfile#L1-L120), [frontend/Dockerfile:1-50](../../frontend/Dockerfile#L1-L50).
- **[Configuration Guide](configuration-guide.md)** — Configuring LLM providers, Redis, PostgreSQL, MinIO/S3, and core service parameters via `orchestrator/config.py` and environment seeds [infrastructure/.env.example:1-174](../../infrastructure/.env.example#L1-L174).
- **[Quick Start Tutorial](quick-start-tutorial.md)** — Step-by-step workflows for creating agents, connecting tools, executing chats, and running Playbooks.
- **[Business Intake Wizard](business-intake-wizard.md)** — Auto-led onboarding entry point and workspace setup through chat.

---

## System Architecture Overview

Automatos AI relies on a containerized multi-tier architecture powered by FastAPI on the backend and Next.js on the frontend, supported by PostgreSQL with `pgvector`, Redis, and MinIO [docker-compose.yml:26-159](../../docker-compose.yml#L26-L159).

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

Sources: [docker-compose.yml:26-159](../../docker-compose.yml#L26-L159), [orchestrator/core/redis/client.py:14-36](../../orchestrator/core/redis/client.py#L14-L36), [frontend/Dockerfile:14-48](../../frontend/Dockerfile#L14-L48)

---

## 1. Installation & Setup

Automatos AI is deployed primarily via Docker Compose, coordinating core infrastructure services including the FastAPI orchestrator, Next.js frontend, PostgreSQL (`pgvector`), Redis, and MinIO [docker-compose.yml:26-159](../../docker-compose.yml#L26-L159). Multi-stage Dockerfiles ensure minimal production footprints while preserving development hot-reload features [orchestrator/Dockerfile:1-132](../../orchestrator/Dockerfile#L1-L132), [frontend/Dockerfile:1-49](../../frontend/Dockerfile#L1-L49).

For detailed instructions on local container orchestration, environment variables, database schema migrations via Alembic, and volume management, see **[Installation & Setup](installation-setup.md)**.

Sources: [docker-compose.yml:1-159](../../docker-compose.yml#L1-L159), [orchestrator/Dockerfile:1-132](../../orchestrator/Dockerfile#L1-L132), [frontend/Dockerfile:1-49](../../frontend/Dockerfile#L1-L49)

---

## 2. Configuration Guide

Platform services are governed by centralized environment variables and configuration files [infrastructure/.env.example:1-174](../../infrastructure/.env.example#L1-L174). Essential configuration parameters include database connection strings, Redis channels, encryption keys for credentials, LLM provider API keys (OpenAI, Anthropic, OpenRouter), and tool integrations like Composio [infrastructure/.env.example:17-116](../../infrastructure/.env.example#L17-L116).

### Service Configuration Flow

```mermaid
graph LR
    Env["".env.example / .env"<br/>infrastructure/.env.example""] --> Config[""ConfigManager"<br/>orchestrator/config.py""]
    Config --> DB[""PostgreSQL / pgvector"<br/>orchestrator_db""]
    Config --> Redis[""Redis Cache & PubSub"<br/>orchestrator/core/redis/client.py""]
    Config --> LLM[""LLM Managers"<br/>OpenAI / Anthropic / OpenRouter""]
```

For detailed setup instructions covering provider resolution tiers, vector store configurations (Qdrant/pgvector), and system settings seeds, see **[Configuration Guide](configuration-guide.md)**.

Sources: [infrastructure/.env.example:1-174](../../infrastructure/.env.example#L1-L174), [orchestrator/core/redis/client.py:141-194](../../orchestrator/core/redis/client.py#L141-L194)

---

## 3. Quick Start Tutorial

Once services are running, new users can immediately interact with the platform. Every fresh workspace seeds a primary orchestration agent named `Auto` which acts as the core conversational partner and default router [orchestrator/core/seeds/seed_auto_agent.py:5-16](../../orchestrator/core/seeds/seed_auto_agent.py#L5-L16).

Users can prompt agents, connect external tool integrations, test streaming chat responses, and trigger multi-agent workflows through the web UI [docker-compose.yml:4-10](../../docker-compose.yml#L4-L10).

For a step-by-step walkthrough covering agent creation, tool linking, and running Playbooks, see **[Quick Start Tutorial](quick-start-tutorial.md)**.

Sources: [docker-compose.yml:4-10](../../docker-compose.yml#L4-L10), [orchestrator/core/seeds/seed_auto_agent.py:5-16](../../orchestrator/core/seeds/seed_auto_agent.py#L5-L16)

---

## 4. Business Intake Wizard

Onboarding starts with Auto's greeting in the empty chat. `OnboardingOpener` renders only when `workspace.onboarding.stage === 'not_started'`, asks “what's your business?”, and invites the user to reply in chat. It reads the server-provided workspace snapshot and never writes onboarding state; once the stage advances or the snapshot is absent, it renders nothing [frontend/components/onboarding/onboarding-opener.tsx:5-43](../../frontend/components/onboarding/onboarding-opener.tsx#L5-L43).

For further onboarding and workspace setup documentation, see **[Business Intake Wizard](business-intake-wizard.md)**.

Sources: [frontend/components/onboarding/onboarding-opener.tsx:5-43](../../frontend/components/onboarding/onboarding-opener.tsx#L5-L43), [frontend/components/chatbot/chat.tsx:1375-1381](../../frontend/components/chatbot/chat.tsx#L1375-L1381), [frontend/components/onboarding/__tests__/onboarding-opener.test.tsx:30-69](../../frontend/components/onboarding/__tests__/onboarding-opener.test.tsx#L30-L69)

---
