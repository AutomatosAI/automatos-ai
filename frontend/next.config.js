/** @type {import('next').NextConfig} */
const path = require('path')

const nextConfig = {
  output: 'standalone',
  reactStrictMode: false,
  // SECURITY: Disable X-Powered-By header to reduce information leakage (OWASP A05:2021)
  poweredByHeader: false,
  typescript: {
    // TODO: Set to false once ~400 TS errors are resolved (separate PR)
    // Security audit 2026-02-04: 798 lines of TS errors found across dozens of components
    ignoreBuildErrors: true
  },
  typedRoutes: true,
  turbopack: {
    root: __dirname,
  },
  // Rewrites disabled - using absolute URLs from client instead
  // Next.js rewrites require env vars at build time, which is problematic on Railway
  // The API client will use NEXT_PUBLIC_API_URL for direct calls to backend

  // Cluster 1A rehouse: permanent redirects from old routes to new IA
  async redirects() {
    return [
      { source: '/workspace', destination: '/deliverables', permanent: true },
      { source: '/workspace/explorer', destination: '/deliverables/explorer', permanent: true },
      { source: '/workspace/templates', destination: '/deliverables?tab=templates', permanent: true },
      { source: '/workspace/blog', destination: '/deliverables?tab=blogs', permanent: true },
      { source: '/activity', destination: '/command-center', permanent: true },
      { source: '/activity/blog', destination: '/deliverables?tab=blogs', permanent: true },
      { source: '/activity/board', destination: '/command-center?tab=board', permanent: true },
    ]
  },

  // SECURITY: HTTP response headers for all routes (OWASP A05:2021 - Security Misconfiguration)
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(self), geolocation=(), interest-cohort=()',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=31536000; includeSubDomains',
          },
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          // PRD-70 FIX-08: Content-Security-Policy
          // Restricts script/style/image sources to prevent XSS.
          // 'unsafe-inline' needed for Next.js styled-jsx and Clerk components.
          // 'unsafe-eval' needed for Next.js dev mode — remove in production hardening phase.
          // PRD-70 FIX-08: Content-Security-Policy
          // Restricts script/style/image sources to prevent XSS.
          // 'unsafe-inline' needed for Next.js styled-jsx and Clerk components.
          // 'unsafe-eval' needed for Next.js dev mode — remove in production hardening phase.
          // connect-src includes api.automatos.app (cross-origin API) and Clerk auth endpoints.
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.clerk.accounts.dev https://challenges.cloudflare.com https://cdn.jsdelivr.net",
              "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
              "img-src 'self' data: blob: https://*.clerk.accounts.dev https://img.clerk.com https://*.googleusercontent.com https://logos.composio.dev",
              "font-src 'self' data: https://cdn.jsdelivr.net",
              "connect-src 'self' https://*.automatos.app https://*.clerk.accounts.dev https://api.clerk.com https://cdn.jsdelivr.net wss: ws:",
              "frame-src 'self' https://*.clerk.accounts.dev https://challenges.cloudflare.com",
              "worker-src 'self' blob:",
              "object-src 'none'",
              "base-uri 'self'",
              "form-action 'self'",
              "frame-ancestors 'none'",
            ].join('; '),
          },
        ],
      },
    ]
  },
}

module.exports = nextConfig
