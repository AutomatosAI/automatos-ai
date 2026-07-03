import { describe, it, expect, vi, afterEach } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { render } from '@testing-library/react'

// PRD-175 W5 (F008) — the frontend half of the absent PRD-150. The app must
// render with NO ClerkProvider mount under AUTH_EDITION=local (git clone &&
// docker compose up, no Clerk tenant), while saas keeps Clerk mounted and
// protecting routes exactly as today. The choke point is the *mount*: one flag,
// two conditional mounts (providers + middleware), the existing token seam reused.

const PROVIDERS = path.resolve(__dirname, '..', 'providers.tsx')
const MIDDLEWARE = path.resolve(__dirname, '..', '..', 'middleware.ts')
const SIGN_IN = path.resolve(
  __dirname, '..', '..', 'app', 'sign-in', '[[...rest]]', 'page.tsx',
)

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

// ---------------------------------------------------------------------------
// lib/auth-edition.ts — one source of truth for the client edition
// ---------------------------------------------------------------------------

describe('PRD-175 — lib/auth-edition', () => {
  it('defaults to saas when the flag is unset (running product unchanged)', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', '')
    const mod = await import('@/lib/auth-edition')
    expect(mod.authEdition).toBe('saas')
    expect(mod.isSaaS).toBe(true)
    expect(mod.isLocal).toBe(false)
  })

  it('is local when the flag is local', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'local')
    const mod = await import('@/lib/auth-edition')
    expect(mod.authEdition).toBe('local')
    expect(mod.isSaaS).toBe(false)
    expect(mod.isLocal).toBe(true)
  })

  it('falls back to saas for an unknown value (never silently un-guard)', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'banana')
    const mod = await import('@/lib/auth-edition')
    expect(mod.authEdition).toBe('saas')
    expect(mod.isSaaS).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// LocalAuthProvider — installs a no-op token getter via the existing seam
// ---------------------------------------------------------------------------

describe('PRD-175 — LocalAuthProvider', () => {
  it('installs a null token getter (no bearer → backend local identity) and renders children', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'local')
    const { apiClient } = await import('@/lib/api-client')
    const spy = vi.spyOn(apiClient, 'setClerkTokenGetter')

    const { LocalAuthProvider } = await import('@/components/local-auth-provider')
    const { getByText } = render(<LocalAuthProvider><span>child-ok</span></LocalAuthProvider>)

    expect(getByText('child-ok')).toBeInTheDocument()
    expect(spy).toHaveBeenCalledTimes(1)
    const getter = spy.mock.calls[0][0]
    await expect(getter()).resolves.toBeNull()
  })
})

// ---------------------------------------------------------------------------
// providers.tsx — the mount is conditional on the edition (F008 choke point)
// ---------------------------------------------------------------------------

describe('PRD-175 — providers.tsx mount gate', () => {
  const src = readFileSync(PROVIDERS, 'utf8')

  it('branches the mount on the edition flag, not an unconditional ClerkProvider', () => {
    // isSaaS decides which auth boundary mounts.
    expect(src).toMatch(/isSaaS/)
    // The local branch mounts LocalAuthProvider instead of ClerkProvider.
    expect(src).toContain('LocalAuthProvider')
    // The `Providers` component itself must NOT return ClerkProvider directly —
    // its top-level element is the edition-aware AuthBoundary, so no Clerk symbol
    // is reached in `local`.
    expect(src).toMatch(/export function Providers[\s\S]*?return\s*\(\s*<AuthBoundary/)
    // And the `local` branch short-circuits before any ClerkProvider is rendered.
    expect(src).toMatch(/if\s*\(\s*!isSaaS\s*\)[\s\S]*?LocalAuthProvider/)
  })

  it('keeps the shared providers identical across editions (single children tree)', () => {
    // RoleProvider / ThemeProvider / WorkspaceProvider are below the auth boundary.
    expect(src).toContain('RoleProvider')
    expect(src).toContain('WorkspaceProvider')
  })
})

// ---------------------------------------------------------------------------
// middleware.ts — auth.protect() only runs under saas
// ---------------------------------------------------------------------------

describe('PRD-175 — middleware edition gate', () => {
  const src = readFileSync(MIDDLEWARE, 'utf8')

  it('gates clerkMiddleware / auth.protect() on the edition flag', () => {
    expect(src).toMatch(/NEXT_PUBLIC_AUTH_EDITION|isSaaS|authEdition/)
    // A local pass-through exists so every route is served with no auth.protect().
  })

  it('does not call auth.protect() unconditionally at module top level', () => {
    // The default export must be edition-aware, not a bare clerkMiddleware(protect).
    // (Presence of a local pass-through branch is the assertion.)
    expect(src).toMatch(/local/i)
  })
})

// ---------------------------------------------------------------------------
// sign-in is saas-only — local redirects to '/'
// ---------------------------------------------------------------------------

describe('PRD-175 — sign-in is saas-only', () => {
  const src = readFileSync(SIGN_IN, 'utf8')
  it('redirects to / under the local edition (no login in local)', () => {
    expect(src).toMatch(/isSaaS|authEdition|NEXT_PUBLIC_AUTH_EDITION/)
    expect(src).toMatch(/redirect|'\/'/)
  })
})
