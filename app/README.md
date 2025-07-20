# Automatos AI Frontend

Next.js frontend for the Automatos AI orchestration platform.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Backend API running (default: http://localhost:8001)

### Deployment

1. **Clone and navigate to the frontend directory:**
   ```bash
   cd app
   ```

2. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Deploy:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - API Base URL: Configured in .env

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FRONTEND_PORT` | Frontend port | 3000 |
| `API_BASE_URL` | Backend API URL | http://localhost:8001 |
| `NEXTAUTH_URL` | NextAuth callback URL | http://localhost:3000 |
| `NEXTAUTH_SECRET` | NextAuth secret | Required |
| `DATABASE_URL` | Database connection | Optional |
| `REDIS_CACHE_PORT` | Redis cache port | 6380 |
| `REDIS_CACHE_PASSWORD` | Redis cache password | redis_cache_123 |

### Optional Redis Cache

To enable Redis caching for the frontend:

```bash
docker-compose --profile cache up -d
```

## Development

### Local Development
```bash
npm install
npm run dev
```

### Build for Production
```bash
npm run build
npm start
```

## Docker Commands

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose build --no-cache

# Enable Redis cache
docker-compose --profile cache up -d
```

## Architecture

- **Frontend**: Next.js 14 with TypeScript
- **UI**: Radix UI components with Tailwind CSS
- **State**: Zustand for state management
- **Auth**: NextAuth.js
- **Cache**: Optional Redis for performance
- **Deployment**: Docker with multi-stage builds

## Backend Integration

The frontend connects to the Automatos AI backend API. Make sure the backend is running and accessible at the URL specified in `API_BASE_URL`.

## License

Open source - see main project license. 