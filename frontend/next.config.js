/** @type {import('next').NextConfig} */
const path = require('path')

const nextConfig = {
  reactStrictMode: false,
  typescript: {
    // TODO: Set to false once ~400 TS errors are resolved (separate PR)
    // Security audit 2026-02-04: 798 lines of TS errors found across dozens of components
    ignoreBuildErrors: true
  },
  // Next 14.x Turbopack config lives under `experimental.turbo`.
  // Keep the newer `turbopack` key as well for forward-compat.
  experimental: {
    turbo: {
      // Prevent Turbopack inferring `app/` as root in monorepo setups
      root: __dirname,
    },
  },
  turbopack: {
    // Prevent Turbopack inferring `app/` as root in monorepo setups
    root: __dirname,
  },
  // Rewrites disabled - using absolute URLs from client instead
  // Next.js rewrites require env vars at build time, which is problematic on Railway
  // The API client will use NEXT_PUBLIC_API_URL for direct calls to backend
}

module.exports = nextConfig
