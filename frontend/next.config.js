/** @type {import('next').NextConfig} */
const path = require('path')
const {
  mediaOrigins,
  clerkFrontendApiOrigin,
  CLERK_PROTECT_SOURCE,
  CLERK_PROTECT_CONNECT_SOURCE,
} = require('./csp-sources')

// PRD-209 (local edition): the API's origin must be an allowed connect-src.
const apiOrigin = (() => {
  try {
    return process.env.NEXT_PUBLIC_API_URL ? new URL(process.env.NEXT_PUBLIC_API_URL).origin : ''
  } catch {
    return ''
  }
})()

// F237: presigned media (Socials thumbnails, a post's files) loads from object storage's public origin.
const mediaSources = mediaOrigins(process.env.NEXT_PUBLIC_MEDIA_ORIGINS).join(' ')

// Clerk production readiness: a pk_live_ key loads clerk-js from the instance's own Frontend API
// (clerk.automatos.app), not *.clerk.accounts.dev, so the sign-in page would not render without it.
// Derived from the publishable key the build ships with; '' for the local edition (no key).
const clerkFrontendApi = clerkFrontendApiOrigin(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY)

/** One CSP directive: the given parts, empty ones dropped, space-separated. */
const directive = (...parts) => parts.filter(Boolean).join(' ')

// SECURITY: HTTP response headers for all routes (OWASP A05:2021 - Security Misconfiguration)
const SECURITY_HEADERS_RULE = {
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
    // connect-src includes api.automatos.app (cross-origin API) and Clerk auth endpoints.
    {
      key: 'Content-Security-Policy',
      value: [
        "default-src 'self'",
        directive(
          "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.clerk.accounts.dev",
          clerkFrontendApi,
          CLERK_PROTECT_SOURCE,
          'https://challenges.cloudflare.com https://cdn.jsdelivr.net',
        ),
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        `img-src 'self' data: blob: https://*.clerk.accounts.dev https://img.clerk.com https://*.googleusercontent.com https://logos.composio.dev ${mediaSources}`.trim(),
        `media-src 'self' blob: ${mediaSources}`.trim(),
        "font-src 'self' data: https://cdn.jsdelivr.net",
        // The API origin is derived from NEXT_PUBLIC_API_URL so the local edition
        // (http://localhost:8000) is allowed too — without it the browser refuses
        // every API call ("Failed to fetch") before it leaves the page. In SaaS the
        // value is https://api.automatos.app, already covered by the wildcard.
        directive(
          `connect-src 'self' ${apiOrigin} https://*.automatos.app https://*.clerk.accounts.dev`,
          clerkFrontendApi,
          CLERK_PROTECT_CONNECT_SOURCE,
          'https://api.clerk.com https://cdn.jsdelivr.net wss: ws:',
        ),
        // F352 (night 10b): a PDF preview (Deliverables side panel, Explorer, chat file card)
        // is fetched with the Bearer header and shown as a blob: URL in an <iframe>
        // (useAuthenticatedBlobUrl). 'self' never matches blob:, so frame-src names it. The
        // blob document inherits this policy, and the browser's PDF viewer is a plugin, so
        // object-src allows blob: too. Only this page's own script can mint a blob: URL.
        directive("frame-src 'self' blob: https://*.clerk.accounts.dev", CLERK_PROTECT_SOURCE, 'https://challenges.cloudflare.com'),
        "worker-src 'self' blob:",
        "object-src blob:",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
      ].join('; '),
    },
  ],
}

// PRD-260 (P260-FX-15): Auto's clips and stills (public/auto/, lib/mascot/media.ts MASCOT_DIR).
// Next serves public files with max-age=0, so every state swap waited a revalidation round trip:
// they stay fresh for a day, then a stale copy plays while the browser revalidates in the
// background. Not `immutable`: the file names carry no version, so a changed clip must still arrive.
const SECONDS_PER_DAY = 24 * 60 * 60
const MASCOT_MAX_AGE_S = SECONDS_PER_DAY
const MASCOT_SWR_S = 7 * SECONDS_PER_DAY
const MASCOT_CACHE_RULE = {
  source: '/auto/:file*',
  headers: [
    {
      key: 'Cache-Control',
      value: `public, max-age=${MASCOT_MAX_AGE_S}, stale-while-revalidate=${MASCOT_SWR_S}`,
    },
  ],
}

// PRD-262 US-046: what the standalone server never loads stays out of its trace, and so out of
// the image: sharp and its libvips builds (image optimisation is off, below), TypeScript (this
// config is .js and only `next build` reads TypeScript) and Next's AMP validator (no page is AMP).
// PRD-262 #29: Next's own build and dev internals, which the trace reaches through requires the
// production server never runs: next/font's loader and the capsize metrics only it reads, Babel,
// browserslist and caniuse-lite, the CSS tools, webpack's crypto polyfill, Next's file tracer, the
// dev overlay, and the route runtimes Next picks only for experimental React or Turbopack (turning
// either on needs those two globs gone first, or every page fails with MODULE_NOT_FOUND). The
// server does load compiled/babel/code-frame.js, babel-code-frame, next/dist/build and the edge
// runtime, so they stay. PRD-262 #47: React's development builds and Next's React Refresh, which
// the server never loads: the standalone server.js sets NODE_ENV=production, and the Dockerfile
// does too. frontend/scripts/check-standalone-image.sh looks for each glob in every built image
// (test.yml's prod-image-parity job, images.yml) and requests every page.
const STANDALONE_TRACE_EXCLUDES = [
  'node_modules/@img/**',
  'node_modules/sharp/**',
  'node_modules/typescript/**',
  'node_modules/next/dist/compiled/amphtml-validator/**',
  'node_modules/next/dist/server/capsize-font-metrics.json',
  'node_modules/next/dist/server/font-utils.js',
  'node_modules/next/dist/compiled/@next/font/**',
  'node_modules/next/dist/compiled/babel/bundle.js',
  'node_modules/next/dist/compiled/babel-packages/**',
  'node_modules/next/dist/compiled/browserslist/**',
  'node_modules/caniuse-lite/**',
  'node_modules/next/dist/compiled/postcss-preset-env/**',
  'node_modules/next/dist/compiled/cssnano-simple/**',
  'node_modules/next/dist/compiled/crypto-browserify/**',
  'node_modules/next/dist/compiled/@vercel/nft/**',
  'node_modules/next/dist/compiled/next-devtools/**',
  'node_modules/next/dist/compiled/next-server/*-experimental.runtime.prod.js',
  'node_modules/next/dist/compiled/next-server/*-turbo.runtime.prod.js',
  'node_modules/react/cjs/*.development.js',
  'node_modules/react-dom/cjs/*.development.js',
  'node_modules/scheduler/cjs/*.development.js',
  'node_modules/next/dist/compiled/react-is/cjs/*.development.js',
  'node_modules/next/dist/compiled/react-refresh/**',
]

// PRD-262 #47: packages only the browser runs, loaded on demand inside a handler or an effect
// (hooks/use-retell-call.ts's voice call, which brings LiveKit, and FilePreview's spreadsheet
// preview), never during a server render. The server builds resolve them to an empty module, so
// their code (0.96 MB of server chunks) stays out of the image; the browser bundle keeps them.
// Not `externals`: the trace would then copy each package into node_modules.
const CLIENT_ONLY_PACKAGES = ['retell-client-js-sdk', 'livekit-client', 'xlsx']

/** The webpack config with each client-only package resolved to an empty module on the server. */
function withoutClientOnlyPackagesOnTheServer(config, { isServer }) {
  if (!isServer) return config
  const absent = Object.fromEntries(CLIENT_ONLY_PACKAGES.map((name) => [name, false]))
  return { ...config, resolve: { ...config.resolve, alias: { ...config.resolve?.alias, ...absent } } }
}

const nextConfig = {
  output: 'standalone',
  // PRD-262 US-046 (§9-Q12): no image optimisation, so the server never loads sharp and
  // /_next/image answers 404. The only optimised image was the 48 px brand mark on the hosted
  // sign-in pages, which public/brand/ now holds at that size (96 px wide, for 2x screens).
  images: { unoptimized: true },
  // '*' matches every route and the server's own trace ('next-server').
  outputFileTracingExcludes: { '*': STANDALONE_TRACE_EXCLUDES },
  webpack: withoutClientOnlyPackagesOnTheServer,
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

  // SECURITY_HEADERS_RULE for every route; MASCOT_CACHE_RULE for Auto's clips and stills.
  async headers() {
    return [SECURITY_HEADERS_RULE, MASCOT_CACHE_RULE]
  },
}

module.exports = nextConfig
