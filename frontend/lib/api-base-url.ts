/**
 * PRD-277 US-505 — the address apiClient puts in front of every API path.
 *
 *   enterprise    → '' : every call is relative (`/api/…`), on the page's own origin. The web app
 *                   and the API answer on one host (PRD-277 §5.3); the session cookie, the
 *                   cross-site header wrapper (lib/enterprise/session-fetch.ts) and the CSRF check
 *                   all assume it, and the enterprise build carries no API address
 *                   (enterprise-build.js refuses one). Nothing on the page can point it elsewhere.
 *   local, saas   → as before: a runtime-injected `window.__NEXT_PUBLIC_API_URL__`, else the
 *                   build-time NEXT_PUBLIC_API_URL, else `window.NEXT_PUBLIC_API_URL`, else ''
 *                   (with the same console errors when there is none in the browser).
 */
import { isEnterprise } from '@/lib/edition-name'

type ApiUrlWindow = Window & { __NEXT_PUBLIC_API_URL__?: string; NEXT_PUBLIC_API_URL?: string }

function configuredApiUrl(): string {
  const page = typeof window !== 'undefined' ? (window as ApiUrlWindow) : null
  return (
    (page !== null && page.__NEXT_PUBLIC_API_URL__) || // Runtime injection
    process.env.NEXT_PUBLIC_API_URL || // Build-time env var
    (page !== null && page.NEXT_PUBLIC_API_URL) || // Runtime fallback
    ''
  )
}

/** apiClient's base address for this build: '' (relative) in enterprise, the configured one otherwise. */
export function apiBaseUrl(enterprise: boolean = isEnterprise): string {
  if (enterprise) return ''
  const baseUrl = configuredApiUrl()
  // Warn if baseUrl is not set (will cause 404s)
  if (!baseUrl && typeof window !== 'undefined') {
    console.error('❌ NEXT_PUBLIC_API_URL is not set! API calls will fail.')
    console.error('Set NEXT_PUBLIC_API_URL in Railway frontend service variables')
    console.error('Current env:', process.env.NEXT_PUBLIC_API_URL || 'NOT SET')
  }
  return baseUrl
}
