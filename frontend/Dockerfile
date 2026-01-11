# =============================================================================
# Automatos AI - Frontend Dockerfile (Multi-stage)
# =============================================================================
# Stages:
#   - base: Dependencies installation
#   - development: Hot-reload development server
#   - builder: Production build
#   - production: Optimized production image
# =============================================================================

# =============================================================================
# Base Stage - Dependencies
# =============================================================================
FROM node:18-alpine as base

WORKDIR /app

# Install dependencies for node-gyp and other native modules
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    curl

# Copy package files (works when Root Directory is set to automatos-ai)
COPY frontend/package*.json ./

# =============================================================================
# Development Stage - Hot Reload
# =============================================================================
FROM base as development

# Install all dependencies (including devDependencies)
# Use npm install instead of npm ci for development (more forgiving with lock file)
RUN npm install --legacy-peer-deps

# Copy source code
COPY frontend/ .

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Development server with hot-reload
CMD ["npm", "run", "dev"]

# =============================================================================
# Builder Stage - Production Build
# =============================================================================
FROM base as builder

# Install ALL dependencies (including devDependencies for build)
RUN npm install --legacy-peer-deps

# Copy source code
COPY frontend/ .

# Build application
RUN npm run build

# =============================================================================
# Production Stage - Optimized
# =============================================================================
FROM node:18-alpine as production

WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy package files and install production dependencies only
COPY frontend/package*.json ./
RUN npm ci --legacy-peer-deps --only=production && npm cache clean --force

# Copy built application from builder
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./next.config.js
COPY --from=builder /app/package.json ./package.json

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001 && \
    chown -R nextjs:nodejs /app

USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Production server
ENV NODE_ENV=production
CMD ["npm", "start"]

