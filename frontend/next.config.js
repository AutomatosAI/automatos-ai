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
            value: 'camera=(), microphone=(), geolocation=(), interest-cohort=()',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=31536000; includeSubDomains',
          },
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
        ],
      },
    ]
  },
}

module.exports = nextConfig
