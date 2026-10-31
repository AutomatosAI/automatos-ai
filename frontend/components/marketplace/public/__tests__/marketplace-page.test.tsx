/**
 * PRD-270 US-270-08a (K19) — the /marketplace client switch.
 *
 *   SaaS, signed out → the visitor header (logo, Sign in, Start free) and no MainLayout.
 *   SaaS, signed in  → today's page: MainLayout + MarketplaceHomepage.
 *   SaaS, loading    → a quiet line, neither page.
 *   local            → today's page, and Clerk's useAuth is never called.
 *
 * The edition is a build-time constant, so each case loads the module under its own edition.
 * The app shell and the homepage are stubbed (they are today's, unchanged); the page header is
 * stubbed as the /verify test does (its theme slots are not under test); the visitor's tabs are
 * stubbed too (US-270-08b, visitor-tabs.test.tsx covers them).
 */
import { readFileSync } from 'fs'
import path from 'path'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="main-layout">{children}</div>,
}))
vi.mock('@/components/marketplace/marketplace-homepage', () => ({
  MarketplaceHomepage: () => <div data-testid="marketplace-homepage" />,
}))
vi.mock('@/components/marketplace/public/visitor-tabs', () => ({
  VisitorTabs: () => <div data-testid="visitor-tabs" />,
}))
vi.mock('@/components/shared/page-header', () => ({
  PageHeader: ({ title, titleAccent }: { title: string; titleAccent: string }) => (
    <h1>
      {title} {titleAccent}
    </h1>
  ),
}))

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

describe('MarketplacePage (PRD-270 US-270-08a)', () => {
  it('SaaS signed out: the visitor header with the logo, Sign in and Start free, and no MainLayout', async () => {
    clerkUseAuth.mockReturnValue({ isLoaded: true, isSignedIn: false })
    const { MarketplacePage } = await loadPage('saas')
    render(<MarketplacePage />)

    expect(screen.getByRole('img', { name: 'Automatos logo' })).toHaveAttribute('src', '/brand/automatos-mark-hi.png')
    expect(screen.getByRole('link', { name: 'Sign in' })).toHaveAttribute('href', '/sign-in')
    expect(screen.getByRole('link', { name: 'Start free' })).toHaveAttribute('href', '/sign-up')
    expect(screen.getByRole('heading', { name: 'Community Marketplace' })).toBeInTheDocument()
    expect(screen.getByTestId('visitor-tabs')).toBeInTheDocument()
    expect(screen.queryByTestId('main-layout')).not.toBeInTheDocument()
    expect(screen.queryByTestId('marketplace-homepage')).not.toBeInTheDocument()
    expect(usePageAPI).toHaveBeenCalledWith('marketplace')
  })

  it("SaaS signed in: today's page, MainLayout around MarketplaceHomepage, and no visitor header", async () => {
    clerkUseAuth.mockReturnValue({ isLoaded: true, isSignedIn: true })
    const { MarketplacePage } = await loadPage('saas')
    render(<MarketplacePage />)

    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('marketplace-homepage'))
    expect(screen.queryByRole('link', { name: 'Start free' })).not.toBeInTheDocument()
    expect(screen.queryByTestId('visitor-tabs')).not.toBeInTheDocument()
    expect(usePageAPI).toHaveBeenCalledWith('marketplace')
  })

  it('SaaS while Clerk loads: a quiet loading line, never the member page or the visitor view', async () => {
    clerkUseAuth.mockReturnValue({ isLoaded: false, isSignedIn: undefined })
    const { MarketplacePage, MARKETPLACE_LOADING } = await loadPage('saas')
    render(<MarketplacePage />)

    expect(screen.getByRole('status')).toHaveTextContent(MARKETPLACE_LOADING)
    expect(screen.queryByTestId('main-layout')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Start free' })).not.toBeInTheDocument()
  })

  it("local: today's page, with no Clerk call", async () => {
    const { MarketplacePage } = await loadPage('local')
    render(<MarketplacePage />)

    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('marketplace-homepage'))
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
