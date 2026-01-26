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

# Copy package files (Dockerfile is in frontend/, so copy from current dir)
COPY package*.json ./

# =============================================================================
# Development Stage - Hot Reload
# =============================================================================
FROM base as development

# Install all dependencies (including devDependencies)
# Use npm install instead of npm ci for development (more forgiving with lock file)
RUN npm install --legacy-peer-deps

# Copy source code (Dockerfile is in frontend/, so copy from current dir)
COPY . .

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

# Accept build arguments for environment variables
# NOTE: NEXT_PUBLIC_* variables are embedded in the client bundle and must NOT contain secrets
# API keys should be handled server-side only (see frontend/app/api/chat/route.ts)
ARG NEXT_PUBLIC_API_URL

# Set as environment variables for Next.js build
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Install ALL dependencies (including devDependencies for build)
RUN npm install --legacy-peer-deps

# Copy source code (Dockerfile is in frontend/, so copy from current dir)
COPY . .

# Build application (NEXT_PUBLIC_* vars will be baked into the bundle)
RUN npm run build

# =============================================================================
# Production Stage - Optimized
# =============================================================================
FROM node:18-alpine as production

WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy package files and install production dependencies only
COPY package*.json ./
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

