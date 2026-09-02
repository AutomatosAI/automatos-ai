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
FROM node:20-alpine as base

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
# PRD-233: NEXT_PUBLIC_* are inlined at BUILD time, so the edition and the
# API/WS origins must be build args for the standalone image. The DEFAULT is
# the hosted product (saas) — Railway DOES build this Dockerfile (it prefers a
# Dockerfile over Railpack when one exists in the service root), and a `local`
# default here shipped the Clerk-less edition to production on 2026-08-30 (#650
# → #668). The local edition passes NEXT_PUBLIC_AUTH_EDITION=local explicitly
# via docker-compose.yml build args.
ARG NEXT_PUBLIC_AUTH_EDITION=saas
ARG NEXT_PUBLIC_WS_URL
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
ARG NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
ARG NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard
ARG NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard

# Set as environment variables for Next.js build
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_AUTH_EDITION=${NEXT_PUBLIC_AUTH_EDITION}
ENV NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL}
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV NEXT_PUBLIC_CLERK_SIGN_IN_URL=${NEXT_PUBLIC_CLERK_SIGN_IN_URL}
ENV NEXT_PUBLIC_CLERK_SIGN_UP_URL=${NEXT_PUBLIC_CLERK_SIGN_UP_URL}
ENV NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=${NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL}
ENV NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=${NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL}

# Install ALL dependencies (including devDependencies for build)
RUN npm install --legacy-peer-deps

# Cache bust marker — bump this value (or pass --build-arg FRONTEND_CACHE_BUST=...)
# to force a clean rebuild of the COPY + npm run build layers. Used when
# Railway's Docker layer cache produces stale or inconsistent output, e.g.
# .next/static missing chunks the standalone server HTML references.
ARG FRONTEND_CACHE_BUST=2026-05-28T13-00-00Z
RUN echo "Frontend cache bust: $FRONTEND_CACHE_BUST"

# Copy source code (Dockerfile is in frontend/, so copy from current dir)
COPY . .

# Build application (NEXT_PUBLIC_* vars will be baked into the bundle)
RUN npm run build

# =============================================================================
# Production Stage - Optimized (Next.js standalone)
# =============================================================================
FROM node:20-alpine as production

WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Copy standalone build (includes only traced dependencies — no full node_modules)
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Standalone server uses server.js directly — no npm needed
ENV NODE_ENV=production
ENV HOSTNAME="0.0.0.0"
ENV PORT=3000
CMD ["node", "server.js"]

