# Automatos AI v2.5 - Local Development Setup Guide

## Overview
This guide provides instructions for setting up the Automatos AI v2.5 project locally for development and testing.

## Prerequisites
- Ubuntu 22.04 or similar Linux distribution
- Python 3.11+
- Node.js 22.14.0+
- npm 10.9.2+
- PostgreSQL 14+
- Redis 6+

## Quick Start

### 1. Database Setup
```bash
# Install PostgreSQL and Redis
sudo apt update
sudo apt install -y postgresql postgresql-contrib redis-server

# Start services
sudo service postgresql start
sudo service redis-server start

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE orchestrator_db;"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'secure_password_123';"

# Initialize database schema
cd ~/automatos-ai-v2.5/automatos-ai/automatos-ai/orchestrator
sudo -u postgres psql -d orchestrator_db -f init.sql
```

### 2. Backend Setup
```bash
# Navigate to orchestrator directory
cd ~/automatos-ai-v2.5/automatos-ai/automatos-ai/orchestrator

# Install Python dependencies
pip3 install -r requirements.txt
pip3 install redis pdfplumber chromadb openai anthropic

# Start backend server
python3 main.py
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd ~/automatos-ai-v2.5/automatos-ai/automatos-ai/frontend/app

# Install Node.js dependencies
npm install --legacy-peer-deps

# Start development server
npm run dev
```

## Configuration Files

### Backend Environment (.env)
Location: `~/automatos-ai-v2.5/automatos-ai/automatos-ai/.env`
```env
# Database Configuration
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password_123
POSTGRES_PORT=5432

# Redis Configuration
REDIS_PASSWORD=redis_password_123
REDIS_PORT=6379

# API Configuration
API_KEY=automatos_demo_key_2024
MCP_PORT=8001

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000

# API Keys (Replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Development Configuration
DEBUG=false
LOG_LEVEL=INFO
```

### Frontend Environment (.env.local)
Location: `~/automatos-ai-v2.5/automatos-ai/automatos-ai/frontend/app/.env.local`
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=automatos_nextauth_secret_2024_local_dev
```

## Service Ports
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000 (when working)
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Docker Compose Issues Fixed

### Main Issues Resolved:
1. **Port Conflicts**: Fixed Grafana port conflict (3000 → 3002)
2. **Environment Files**: Added proper env_file references to all services
3. **API URL Consistency**: Aligned frontend API URL to point to backend on port 8000
4. **Database Initialization**: Ensured proper PostgreSQL setup with required extensions

### Updated Docker Compose Configuration:
- Main compose file: `~/automatos-ai-v2.5/automatos-ai/automatos-ai/docker-compose.yml`
- Orchestrator compose file: `~/automatos-ai-v2.5/automatos-ai/automatos-ai/orchestrator/docker-compose.yml`

## Testing the Setup

### Backend Testing
```bash
# Test API health
curl http://localhost:8000/docs

# Should return Swagger UI HTML
```

### Frontend Testing
```bash
# Test frontend (when working)
curl http://localhost:3000

# Should return Next.js application HTML
```

## Troubleshooting

### Common Issues:

1. **Backend Import Errors**
   - Install missing Python packages: `pip3 install pdfplumber chromadb openai anthropic redis`

2. **Frontend Dependency Conflicts**
   - Use legacy peer deps: `npm install --legacy-peer-deps`

3. **Database Connection Issues**
   - Ensure PostgreSQL is running: `sudo service postgresql status`
   - Check database exists: `sudo -u postgres psql -l | grep orchestrator_db`

4. **Redis Connection Issues**
   - Ensure Redis is running: `sudo service redis-server status`
   - Test connection: `redis-cli ping`

5. **Docker Issues**
   - If Docker fails, use direct installation as shown above
   - Docker may have permission issues in some environments

### Service Status Check
```bash
# Check running processes
ps aux | grep -E "(python3 main.py|npm run dev)"

# Check port usage
netstat -tlnp | grep -E "(8000|3000|5432|6379)"
```

## Development Workflow

1. **Start Database Services**
   ```bash
   sudo service postgresql start
   sudo service redis-server start
   ```

2. **Start Backend**
   ```bash
   cd ~/automatos-ai-v2.5/automatos-ai/automatos-ai/orchestrator
   python3 main.py
   ```

3. **Start Frontend** (in separate terminal)
   ```bash
   cd ~/automatos-ai-v2.5/automatos-ai/automatos-ai/frontend/app
   npm run dev
   ```

4. **Access Services**
   - Backend API: http://localhost:8000/docs
   - Frontend: http://localhost:3000

## Notes

- The backend is fully functional and accessible at http://localhost:8000
- Frontend may need additional configuration or dependency resolution
- All environment files have been properly configured with the provided API keys
- Database schema has been successfully initialized
- Docker Compose files have been fixed for future use

## Next Steps

1. Resolve frontend startup issues (likely dependency-related)
2. Test full integration between frontend and backend
3. Verify all API endpoints are working correctly
4. Test authentication and authorization flows
