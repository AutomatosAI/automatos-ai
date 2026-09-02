/** @type {import('next').NextConfig} */
const path = require('path')

// PRD-209 (local edition): the API's origin must be an allowed connect-src.
const apiOrigin = (() => {
  try {
    return process.env.NEXT_PUBLIC_API_URL ? new URL(process.env.NEXT_PUBLIC_API_URL).origin : ''
  } catch {
    return ''
  }
})()

const nextConfig = {
  output: 'standalone',
  reactStrictMode: false,
  // SECURITY: Disable X-Powered-By header to reduce information leakage (OWASP A05:2021)
  poweredByHeader: false,
  typescript: {
    // Kept true on PURPOSE: flipping to false would break `next build` (deploy)
    // on the ~hundreds of pre-existing TS errors. PRD-182 W12-S1 (F034) adds the
    // enforcement lane separately — the `frontend-ci` job in
    // .github/workflows/test.yml runs `tsc --noEmit` via
    // scripts/tsc-baseline-check.js and fails only when the error count REGRESSES
    // above frontend/.tsc-baseline.json (measured floor: 554). Type errors are
    // gated in CI without blocking the build. Ratchet the baseline down as errors
    // are fixed; never re-raise it. (Prior note: security audit 2026-02-04 found
    // 798 lines of TS errors across dozens of components.)
    ignoreBuildErrors: true
  },
  eslint: {
    // PRD-154 S10 added .eslintrc.json (no config existed before, so `next
    // build` previously skipped linting). The new no-restricted-syntax rule
    // banning raw fetch('/api…') fires on pre-existing call sites; lint runs in
    // CI / `next lint`, not as a deploy gate — same posture as ignoreBuildErrors.
    // PRD-182 W12-S1 also runs eslint report-only in the frontend-ci job.
    ignoreDuringBuilds: true
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
              // The API origin is derived from NEXT_PUBLIC_API_URL so the local edition
              // (http://localhost:8000) is allowed too — without it the browser refuses
              // every API call ("Failed to fetch") before it leaves the page. In SaaS the
              // value is https://api.automatos.app, already covered by the wildcard.
              `connect-src 'self' ${apiOrigin} https://*.automatos.app https://*.clerk.accounts.dev https://api.clerk.com https://cdn.jsdelivr.net wss: ws:`,
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
