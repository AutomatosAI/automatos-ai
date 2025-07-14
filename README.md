# Multi-Agent Orchestration System for Autonomous DevOps Workflows

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

## Table of Contents
- [Project Overview](#project-overview)
- [Core End Goal](#core-end-goal)
- [Design and Architecture](#design-and-architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project is a Python-based framework that creates a "virtual development team" using AI agents to automate the entire process of building, testing, and deploying software or workflows. Think of it as a smart robot crew that handles DevOps tasks (development + operations) without you having to do everything manually. It's inspired by tools like CrewAI (for agents) and LangGraph (for workflows), but customized for reliability in high-stakes fields like banking or IT.

If you've never coded before, don't worry—this README explains everything like you're starting from zero. Coding is like giving instructions to a computer, and this system does most of that for you. The "agents" are like digital workers: one builds code, another tests it (and fixes if broken), and another deploys it (makes it live). Everything runs on a cloud server (called a Droplet, e.g., from DigitalOcean), wrapped in secure "containers" (using Docker) so it's isolated and safe.

The system is modular (easy to add/remove parts) and uses a "bridge" to run real computer commands securely. It's designed to be self-healing—if tests fail, it loops back to fix automatically.

## Core End Goal
The ultimate goal is to create an autonomous (self-running) system where you can give a simple task—like "Build a secure API for a financial app"—and the AI team handles everything:
- Breaks down the task.
- Generates/writes code (using AI like DeepAgent).
- Tests and fixes bugs in loops.
- Deploys to "live" (e.g., pushes to GitHub or copies to a server).
- Stores "memories" (past tips/code) in a database for smarter future runs.

This turns manual coding/DevOps into automated pipelines, saving time and reducing errors. For beginners: It's like a magic factory—you input an idea, it outputs working software. For developers: It's extensible for complex workflows, with integration points for tools like Cursor (code editor) or n8n (automation workflows).

In the future, it could build things like the financial platform in the design prompt (ai-builder-design-prompt.md), using AI to generate code based on specs.

## Design and Architecture
### High-Level Design
The system is built like a team:
- **Orchestrator (Boss)**: Plans the work, wakes up agents, and manages the flow (using LangGraph for decision trees, e.g., "If test fails, go back").
- **Agents (Workers)**: Role-based AI helpers from CrewAI.
  - Dev Agent: Builds/generates code.
  - Tester Agent: Runs tests, loops fixes.
  - Deployer Agent: Pushes to live (e.g., GitHub).
- **MCP Bridge (Messenger)**: A secure web server (FastAPI) that runs real commands (e.g., "git push") on the server without risks.
- **Context Manager (Memory Helper)**: Pulls smart tips from the database (using Hugging Face for searches, pgvector for storage).
- **Database (Memory Box)**: PostgreSQL with pgvector to store/retrieve data like code snippets.

Everything runs in Docker containers on your cloud server for easy setup/scaling. No local Mac install needed—control from anywhere via API calls (like curl commands).

### How It Works (Beginner-Friendly Flow)
1. You send a task via API (e.g., from your Mac: curl ... with task and GitHub repo URL).
2. Orchestrator breaks it down and pulls memories from DB.
3. Dev Agent clones the repo, generates code (mock for now; DeepAgent later), writes files.
4. Tester runs tests; if fail, loops back to Dev.
5. Deployer commits/pushes changes to GitHub.
6. Results returned to you.

Technical Details for Developers:
- Built with Python 3.11.
- Frameworks: CrewAI (agents), LangGraph (workflows), FastAPI (API/bridge), psycopg2 (DB).
- Security: Allowlist for commands, API keys for access.
- Persistence: Docker volumes for files/DB.

## Features
- Task decomposition and agent roles.
- Self-fixing test loops.
- Secure command execution via bridge.
- RAG (memory) from DB for smarter agents.
- API-triggered (no SSH for runs).
- Git integration: Clone, edit, push.
- Extensible for DeepAgent/Cursor/n8n.

## Prerequisites
- **Hardware/Cloud**: A cloud server (e.g., DigitalOcean Droplet, Ubuntu 22.04+, 2GB RAM min). $6/month starter.
- **Software**:
  - Docker & Docker Compose (install guide below).
  - Git (for repo handling).
  - API Keys: Hugging Face (free for models), GitHub PAT (for push), Abacus.AI (for DeepAgent, optional).
- **Knowledge**: Basic terminal commands (we explain). No prior coding needed for setup.

## Installation and Setup
### Step 1: Set Up Your Cloud Server (If New)
1. Create a DigitalOcean Droplet (or similar): Choose Ubuntu, add SSH key from your Mac (ssh-keygen if none).
2. SSH in: `ssh root@your-droplet-ip` from Mac.
3. Update system: `sudo apt update && sudo apt upgrade -y`.

### Step 2: Install Docker
From SSH:
```
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo usermod -aG docker $USER  # Log out/in
```

### Step 3: Clone the Repo
```
git clone https://github.com/youruser/multi-agent-orchestrator.git  # Replace with your repo URL
cd multi-agent-orchestrator
```

### Step 4: Create .env File
Create `.env` (nano .env):
```
API_KEY=your_secure_api_key  # Invent a strong password
POSTGRES_USER=yourdbuser
POSTGRES_PASSWORD=yourdbpass
POSTGRES_DB=yourdbname
HUGGINGFACE_TOKEN=hf_yourtoken  # From huggingface.co/settings/tokens
GITHUB_TOKEN=ghp_yourgithubtoken  # From github.com/settings/tokens (repo scope)
# Add ABACUS_API_KEY=yourabacuskey later for DeepAgent
```

### Step 5: Build and Start
```
docker compose up -d --build
```
- Checks: `docker ps` (see mcp-bridge and db up).
- Logs: `docker logs orchestrator-bridge-mcp-bridge-1`

### Step 6: Set Up NGINX (For HTTPS/Domain)
1. Install: `sudo apt install nginx certbot python3-certbot-nginx -y`
2. Config: nano /etc/nginx/sites-available/mcp.yourdomain.com (replace with your domain):
   ```
   server {
       listen 80;
       server_name mcp.yourdomain.com;
       location / {
           proxy_pass http://localhost:5679;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```
3. Link/Reload: `sudo ln -s /etc/nginx/sites-available/mcp.yourdomain.com /etc/nginx/sites-enabled/ && sudo nginx -t && sudo systemctl reload nginx`
4. SSL: `sudo certbot --nginx -d mcp.yourdomain.com`

## Usage Guide
### For Beginners (Run a Task)
From Mac (no SSH):
```
curl -X POST https://mcp.yourdomain.com/run-task -H "Content-Type: application/json" -H "X-API-Key: your_secure_api_key" -d '{"task": "Build a simple API", "github_url": "https://github.com/youruser/test-repo.git"}'
```
- What Happens: Clones repo, builds (mock code), tests, pushes back.
- Check GitHub: See new commits.

### For Developers (Extend)
- Edit agents in orchestrator.py (e.g., add DeepAgent: Uncomment tool, use in dev_node).
- Run locally: `python orchestrator.py` (tests hardcoded task).

## Contributing
- Fork repo on GitHub.
- Branch: `git checkout -b feature/yourfeature`
- Commit/Push: `git commit -m "Add feature" && git push`
- PR to main.

## Troubleshooting
- Container down? `docker logs <container-id>`
- API error? Check .env keys.
- Git push fail? Ensure GITHUB_TOKEN has repo scope.

## License
MIT License. See LICENSE file.

## Acknowledgments
Inspired by CrewAI, LangGraph, and your banking/IT needs. Thanks to xAI for guidance!