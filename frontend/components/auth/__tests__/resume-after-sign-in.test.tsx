/**
 * PRD-270 US-270-09a (§5.7, K21) — ResumeAfterSignIn, back on the remembered Marketplace item.
 *
 * - signed in with a workspace: router.replace() to the remembered path, once, and the path is
 *   forgotten (a re-render, a navigation or a second mount resumes nothing);
 * - it waits while the person is signed out, has no workspace yet (a new sign-up's is still being
 *   made) or is still on a page that navigates on its own (sign-in, sign-up, the SSO callback, an
 *   invitation, a password reset), then resumes once that ends; on the home page it gives the
 *   page's own forward to Chat time to happen, and resumes there if the page stays (a theme's home);
 * - nothing when no path was remembered, or it expired or is unsafe, and no navigation when the
 *   person already stands on it;
 * - the local edition renders nothing and never reads the store;
 * - Providers mounts it once, inside WorkspaceProvider: a structural check of providers.tsx's
 *   source, as the marketplace-page test checks app/marketplace/page.tsx (rendering Providers
 *   would mean doubling Clerk, the theme host and every provider under it).
 *
 * The edition is a build-time constant, so each case loads the component under its own edition
 * (the marketplace-page test's pattern).
 */
import { readFileSync } from 'fs'
import path from 'path'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { RETURN_PATH_STORAGE_KEY, RETURN_PATH_TTL_MS, remember } from '@/lib/marketplace/return-after-sign-in'

const session = vi.hoisted(() => ({
  isSignedIn: false as boolean | undefined,
  workspace: null as { id: string } | null,
  pathname: '/chat',
}))
const router = vi.hoisted(() => ({ replace: vi.fn(), push: vi.fn() }))

vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ isSignedIn: session.isSignedIn }) }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ({ workspace: session.workspace }) }))
vi.mock('next/navigation', () => ({ usePathname: () => session.pathname, useRouter: () => router }))

const ARCADE = '/marketplace?tab=themes&item=automatos.arcade&add=1'
const WORKSPACE = { id: 'workspace-1' }
const stored = () => window.sessionStorage.getItem(RETURN_PATH_STORAGE_KEY)

async function load(edition: 'local' | 'saas') {
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  return import('../resume-after-sign-in')
}

function signIn(pathname = '/chat') {
  session.isSignedIn = true
  session.workspace = WORKSPACE
  session.pathname = pathname
}

beforeEach(() => {
  window.sessionStorage.clear()
  window.history.replaceState(null, '', '/')
  session.isSignedIn = false
  session.workspace = null
  session.pathname = '/chat'
  router.replace.mockReset()
  router.push.mockReset()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  window.sessionStorage.clear()
  window.history.replaceState(null, '', '/')
  vi.restoreAllMocks()
  vi.doUnmock('@/lib/auth-edition')
  vi.resetModules()
})

describe('ResumeAfterSignIn (PRD-270 US-270-09a)', () => {
  it('signed in with a workspace: replaces to the remembered path once, and forgets it', async () => {
    const { ResumeAfterSignIn } = await load('saas')
    remember(ARCADE)
    signIn()
    const { rerender } = render(<ResumeAfterSignIn />)

    expect(router.replace).toHaveBeenCalledTimes(1)
    expect(router.replace).toHaveBeenCalledWith(ARCADE)
    expect(stored()).toBeNull()

    session.pathname = '/marketplace'
    rerender(<ResumeAfterSignIn />)
    expect(router.replace).toHaveBeenCalledTimes(1)
    expect(router.push).not.toHaveBeenCalled()
  })

  it('waits while there is no workspace, then resumes once it arrives', async () => {
    const { ResumeAfterSignIn } = await load('saas')
    remember(ARCADE)
    session.isSignedIn = true
    const { rerender } = render(<ResumeAfterSignIn />)

    expect(router.replace).not.toHaveBeenCalled()
    expect(stored()).not.toBeNull()

    session.workspace = WORKSPACE
    rerender(<ResumeAfterSignIn />)
    expect(router.replace).toHaveBeenCalledTimes(1)
    expect(router.replace).toHaveBeenCalledWith(ARCADE)
  })

  it('waits while signed out, or while Clerk loads, then resumes once signed in', async () => {
    const { ResumeAfterSignIn } = await load('saas')
    remember(ARCADE)
    session.workspace = WORKSPACE
    const { rerender } = render(<ResumeAfterSignIn />)
    session.isSignedIn = undefined
    rerender(<ResumeAfterSignIn />)

    expect(router.replace).not.toHaveBeenCalled()
    expect(stored()).not.toBeNull()

    session.isSignedIn = true
    rerender(<ResumeAfterSignIn />)
    expect(router.replace).toHaveBeenCalledTimes(1)
    expect(router.replace).toHaveBeenCalledWith(ARCADE)
  })

  it.each([
    '/sign-in',
    '/sign-in/factor-one',
    '/sign-up',
    '/sign-up/verify-email-address',
    '/sso-callback',
    '/accept-invitation',
    '/reset-password',
  ])(
    'waits on %s, which navigates on its own, then resumes where the person lands',
    async (pathname) => {
      const { ResumeAfterSignIn } = await load('saas')
      remember(ARCADE)
      signIn(pathname)
      const { rerender } = render(<ResumeAfterSignIn />)

      expect(router.replace).not.toHaveBeenCalled()
      expect(stored()).not.toBeNull()

      session.pathname = '/chat'
      rerender(<ResumeAfterSignIn />)
      expect(router.replace).toHaveBeenCalledTimes(1)
      expect(router.replace).toHaveBeenCalledWith(ARCADE)
    },
  )

  it('on the home page: waits for its own forward, then resumes on Chat at once', async () => {
    const { ResumeAfterSignIn, HOME_SETTLE_MS } = await load('saas')
    vi.useFakeTimers()
    remember(ARCADE)
    signIn('/')
    const { rerender } = render(<ResumeAfterSignIn />)

    act(() => {
      vi.advanceTimersByTime(HOME_SETTLE_MS - 1)
    })
    expect(router.replace).not.toHaveBeenCalled()
    expect(stored()).not.toBeNull()

    session.pathname = '/chat'
    rerender(<ResumeAfterSignIn />)
    expect(router.replace).toHaveBeenCalledTimes(1)
    expect(router.replace).toHaveBeenCalledWith(ARCADE)

    act(() => {
      vi.advanceTimersByTime(HOME_SETTLE_MS)
    })
    expect(router.replace).toHaveBeenCalledTimes(1)
  })

  it("on a theme's own home page, which stays: resumes there once it has settled", async () => {
    const { ResumeAfterSignIn, HOME_SETTLE_MS } = await load('saas')
    vi.useFakeTimers()
    remember(ARCADE)
    signIn('/')
    render(<ResumeAfterSignIn />)

    act(() => {
      vi.advanceTimersByTime(HOME_SETTLE_MS)
    })
    expect(router.replace).toHaveBeenCalledTimes(1)
    expect(router.replace).toHaveBeenCalledWith(ARCADE)
    expect(stored()).toBeNull()
  })

  it('a path taken is forgotten: a second mount resumes nothing', async () => {
    const { ResumeAfterSignIn } = await load('saas')
    remember(ARCADE)
    signIn()
    const first = render(<ResumeAfterSignIn />)
    first.unmount()
    render(<ResumeAfterSignIn />)

    expect(router.replace).toHaveBeenCalledTimes(1)
  })

  it('nothing when no path was remembered', async () => {
    const { ResumeAfterSignIn } = await load('saas')
    signIn()
    render(<ResumeAfterSignIn />)

    expect(router.replace).not.toHaveBeenCalled()
  })

  it('nothing when the path expired, and it is forgotten', async () => {
    const { ResumeAfterSignIn } = await load('saas')
    remember(ARCADE, window.sessionStorage, Date.now() - RETURN_PATH_TTL_MS - 1)
    signIn()
    render(<ResumeAfterSignIn />)

    expect(router.replace).not.toHaveBeenCalled()
    expect(stored()).toBeNull()
  })

  it.each(['https://evil.example', '//evil.example', '/chat', '/marketplace/publish', 'javascript:alert(1)'])(
    'nothing when the store holds %s, and it is forgotten',
    async (unsafe) => {
      const { ResumeAfterSignIn } = await load('saas')
      window.sessionStorage.setItem(RETURN_PATH_STORAGE_KEY, JSON.stringify({ path: unsafe, expiresAt: Date.now() + 60_000 }))
      signIn()
      render(<ResumeAfterSignIn />)

      expect(router.replace).not.toHaveBeenCalled()
      expect(router.push).not.toHaveBeenCalled()
      expect(stored()).toBeNull()
    },
  )

  it('already on the remembered path: forgets it without navigating', async () => {
    const { ResumeAfterSignIn } = await load('saas')
    window.history.replaceState(null, '', ARCADE)
    remember(ARCADE)
    signIn('/marketplace')
    render(<ResumeAfterSignIn />)

    expect(router.replace).not.toHaveBeenCalled()
    expect(stored()).toBeNull()
  })

  it('local edition: renders nothing and never reads the store', async () => {
    const { ResumeAfterSignIn } = await load('local')
    remember(ARCADE)
    signIn()
    const { container } = render(<ResumeAfterSignIn />)

    expect(container).toBeEmptyDOMElement()
    expect(router.replace).not.toHaveBeenCalled()
    expect(stored()).not.toBeNull()
  })
})

describe('Providers (PRD-270 US-270-09a)', () => {
  it('mounts ResumeAfterSignIn once, inside WorkspaceProvider', () => {
    const source = readFileSync(path.resolve(__dirname, '..', '..', 'providers.tsx'), 'utf8')
    const mounts = source.match(/<ResumeAfterSignIn \/>/g) ?? []
    const opens = source.indexOf('<WorkspaceProvider>')
    const mount = source.indexOf('<ResumeAfterSignIn />')
    const closes = source.indexOf('</WorkspaceProvider>')

    expect(mounts).toHaveLength(1)
    expect(opens).toBeGreaterThan(-1)
    expect(mount).toBeGreaterThan(opens)
    expect(mount).toBeLessThan(closes)
  })
})
