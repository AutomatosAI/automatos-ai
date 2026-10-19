# =============================================================================
# Automatos AI - Frontend Dockerfile (Multi-stage)
# =============================================================================
# Stages:
#   - base: Node 24 on Alpine, with the package files
#   - development: Hot-reload development server
#   - builder: Production build
#   - production: Alpine, the node binary and the standalone build
# =============================================================================

# =============================================================================
# Base Stage - Dependencies
# =============================================================================
# PRD-262 US-007: Node 24 (LTS until April 2028), pinned as tag@digest so an
# updater moves both together. No apk build tools: no package's install script
# compiles on Linux (each native one ships a prebuilt binary).
# The production stage copies this stage's node binary, which is built against
# musl for Alpine 3.24, so the alpine3.24 suffix here and alpine:3.24 below name
# the same release: bump them together.
FROM node:24-alpine3.24@sha256:44fce696f38ba8c987c4db89954051000bbd05ffd8d8659ba15ed4a8602e937f AS base

WORKDIR /app

# Copy package files (Dockerfile is in frontend/, so copy from current dir)
COPY package*.json ./

# =============================================================================
# Development Stage - Hot Reload
# =============================================================================
FROM base AS development

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
FROM base AS builder

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
# F237: where presigned media loads from (CSP img-src/media-src, frontend/csp-sources.js). The DEFAULT is
# the hosted product's AWS S3; the local edition passes its MinIO origin via docker-compose.yml.
ARG NEXT_PUBLIC_MEDIA_ORIGINS=https://*.amazonaws.com

# Set as environment variables for Next.js build
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_AUTH_EDITION=${NEXT_PUBLIC_AUTH_EDITION}
ENV NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL}
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV NEXT_PUBLIC_CLERK_SIGN_IN_URL=${NEXT_PUBLIC_CLERK_SIGN_IN_URL}
ENV NEXT_PUBLIC_CLERK_SIGN_UP_URL=${NEXT_PUBLIC_CLERK_SIGN_UP_URL}
ENV NEXT_PUBLIC_MEDIA_ORIGINS=${NEXT_PUBLIC_MEDIA_ORIGINS}
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
# PRD-262 US-007: plain Alpine plus the one file the standalone server needs from
# Node, /usr/local/bin/node. No npm, yarn, corepack, C headers or curl.
# libstdc++ (which brings libgcc) is the binary's only library beyond musl.
FROM alpine:3.24@sha256:294b683cb724975bec92580e1e685676bd4b50bda910ddb8c51d4cabeaec77e6 AS production

WORKDIR /app

# PRD-262 US-011: images.yml's weekly run rebuilds this stage without the cache, and
# `apk upgrade` then takes Alpine's fixes (OpenSSL and the rest) before the pinned digest moves.
RUN apk upgrade --no-cache && \
    apk add --no-cache libstdc++

COPY --from=base /usr/local/bin/node /usr/local/bin/node

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

# Health check: busybox wget on a static file. /favicon.ico skips the middleware
# (its matcher ignores paths with an extension) and server rendering, so a probe
# never renders a page or calls Clerk. 127.0.0.1, not localhost: the server
# listens on IPv4 (HOSTNAME=0.0.0.0) and localhost can resolve to ::1.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -q -O /dev/null "http://127.0.0.1:${PORT:-3000}/favicon.ico" || exit 1

# Standalone server uses server.js directly — no npm needed
ENV NODE_ENV=production
ENV HOSTNAME="0.0.0.0"
ENV PORT=3000
CMD ["node", "server.js"]

