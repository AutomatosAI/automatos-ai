/**
 * PRD-270 (§5.7, one Marketplace page for everyone) — the /marketplace page: the same homepage for
 * everyone, in a different frame.
 *
 *   SaaS, signed out → the visitor header (logo and Studio back to the product site, Sign in, Start
 *                      free) over MarketplaceHomepage with the visitor's access, and no MainLayout.
 *   SaaS, signed in  → today's page: MainLayout + MarketplaceHomepage, the member's access.
 *   SaaS, loading    → a quiet line, neither.
 *   local            → today's page, and Clerk's useAuth is never called.
 *
 * The edition is a build-time constant, so each case loads the module under its own edition. The
 * app shell and the homepage are stubbed (the homepage says which access it was given; the signed-
 * out page itself is one-page.test.tsx's).
 */
import { readFileSync } from 'fs'
import path from 'path'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'

const { clerkUseAuth, usePageAPI } = vi.hoisted(() => ({ clerkUseAuth: vi.fn(), usePageAPI: vi.fn() }))

vi.mock('@clerk/nextjs', () => ({
  useAuth: clerkUseAuth,
  useUser: vi.fn(),
  useSession: vi.fn(),
  useOrganization: vi.fn(),
  useClerk: vi.fn(),
}))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))
vi.mock('next/navigation', () => ({
  usePathname: () => '/marketplace',
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="main-layout">{children}</div>,
}))
vi.mock('@/components/marketplace/marketplace-homepage', async () => {
  const { useMarketplaceAccess } = await import('@/components/marketplace/access/marketplace-access')
  return {
    MarketplaceHomepage: () => {
      const { signedIn, source } = useMarketplaceAccess()
      return <div data-testid="marketplace-homepage" data-signed-in={String(signedIn)} data-source={source.kind} />
    },
  }
})

import { STUDIO_SITE_URL } from '@/lib/marketplace/return-after-sign-in'

function renderPage(Page: () => React.ReactElement) {
  const client = new QueryClient()
  return render(
    <QueryClientProvider client={client}>
      <Page />
    </QueryClientProvider>,
  )
}

async function loadPage(edition: 'local' | 'saas') {
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  return import('../marketplace-page')
}

beforeEach(() => {
  clerkUseAuth.mockReset()
  usePageAPI.mockReset()
})

afterEach(() => {
  cleanup()
  vi.doUnmock('@/lib/auth-edition')
  vi.resetModules()
})

describe('MarketplacePage (PRD-270 §5.7, one page for everyone)', () => {
  it('SaaS signed out: the visitor header over the same homepage, with the visitor’s access, and no MainLayout', async () => {
    clerkUseAuth.mockReturnValue({ isLoaded: true, isSignedIn: false })
    const { MarketplacePage } = await loadPage('saas')
    renderPage(MarketplacePage)

    expect(screen.getByRole('img', { name: 'Automatos logo' })).toHaveAttribute('src', '/brand/automatos-mark-hi.png')
    expect(screen.getByRole('link', { name: 'Sign in' })).toHaveAttribute('href', '/sign-in?redirect_url=%2Fmarketplace')
    expect(screen.getByRole('link', { name: 'Start free' })).toHaveAttribute('href', '/sign-up?redirect_url=%2Fmarketplace')
    const homepage = screen.getByTestId('marketplace-homepage')
    expect(homepage).toHaveAttribute('data-signed-in', 'false')
    expect(homepage).toHaveAttribute('data-source', 'public')
    expect(screen.queryByTestId('main-layout')).not.toBeInTheDocument()
    expect(usePageAPI).toHaveBeenCalledWith('marketplace')
  })

  it('SaaS signed out: the logo and the Studio link go back to the Studio product site', async () => {
    clerkUseAuth.mockReturnValue({ isLoaded: true, isSignedIn: false })
    const { MarketplacePage } = await loadPage('saas')
    renderPage(MarketplacePage)

    expect(STUDIO_SITE_URL).toBe('https://studio.automatos.app')
    expect(screen.getByRole('link', { name: 'Automatos Studio' })).toHaveAttribute('href', STUDIO_SITE_URL)
    expect(screen.getByRole('link', { name: 'Automatos Studio' })).toContainElement(screen.getByRole('img', { name: 'Automatos logo' }))
    expect(screen.getByRole('link', { name: 'Studio' })).toHaveAttribute('href', STUDIO_SITE_URL)
  })

  it("SaaS signed in: today's page, MainLayout around MarketplaceHomepage with the member's access, no visitor header", async () => {
    clerkUseAuth.mockReturnValue({ isLoaded: true, isSignedIn: true })
    const { MarketplacePage } = await loadPage('saas')
    renderPage(MarketplacePage)

    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('marketplace-homepage'))
    expect(screen.getByTestId('marketplace-homepage')).toHaveAttribute('data-signed-in', 'true')
    expect(screen.getByTestId('marketplace-homepage')).toHaveAttribute('data-source', 'member')
    expect(screen.queryByRole('link', { name: 'Start free' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Studio' })).not.toBeInTheDocument()
    expect(usePageAPI).toHaveBeenCalledWith('marketplace')
  })

  it('SaaS while Clerk loads: a quiet loading line, never the member page or the visitor view', async () => {
    clerkUseAuth.mockReturnValue({ isLoaded: false, isSignedIn: undefined })
    const { MarketplacePage, MARKETPLACE_LOADING } = await loadPage('saas')
    renderPage(MarketplacePage)

    expect(screen.getByRole('status')).toHaveTextContent(MARKETPLACE_LOADING)
    expect(screen.queryByTestId('main-layout')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Start free' })).not.toBeInTheDocument()
  })

  it("local: today's page, with no Clerk call", async () => {
    const { MarketplacePage } = await loadPage('local')
    renderPage(MarketplacePage)

    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('marketplace-homepage'))
    expect(screen.getByTestId('marketplace-homepage')).toHaveAttribute('data-signed-in', 'true')
    expect(screen.queryByRole('link', { name: 'Sign in' })).not.toBeInTheDocument()
    expect(clerkUseAuth).not.toHaveBeenCalled()
    expect(usePageAPI).toHaveBeenCalledWith('marketplace')
  })
})

describe('app/marketplace/page.tsx (PRD-270 US-270-08a)', () => {
  it('is a server component that renders the client switch', () => {
    const source = readFileSync(path.resolve(__dirname, '..', '..', '..', '..', 'app', 'marketplace', 'page.tsx'), 'utf8')
    expect(source).not.toMatch(/^\s*['"]use client['"]/m)
    expect(source).toContain("from '@/components/marketplace/public/marketplace-page'")
  })
})
