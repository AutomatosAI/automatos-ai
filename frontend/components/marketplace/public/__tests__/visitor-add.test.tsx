/**
 * PRD-270 US-270-09a (§5.7, K21) — the signed-out visitor's ways in, each remembering the way back.
 *
 * - Add to my workspace goes to `/sign-in?redirect_url=<the item's Marketplace link>` and remembers
 *   that link (`/marketplace?tab=<tab>&item=<key>&add=1`), for a theme and for any other tab;
 * - the header's Sign in goes to sign-in, and Start free to `/sign-up?redirect_url=<encoded>`, each
 *   carrying and remembering the Marketplace page the visitor is on; an unsafe page rides as
 *   '/marketplace';
 * - no install request, and no request at all: every click only remembers and navigates.
 *
 * Navigation itself is the browser's: a click's default is cancelled at the document, after the
 * link's own handler has run.
 */
import type { ReactNode } from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ pathname: '/marketplace', params: new URLSearchParams() }))
const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn(), getBaseUrl: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('next/navigation', () => ({ usePathname: () => nav.pathname, useSearchParams: () => nav.params }))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))

import type { ThemeListing } from '@/lib/api-client'
import { RETURN_PATH_STORAGE_KEY, take } from '@/lib/marketplace/return-after-sign-in'

import { ADD_TO_WORKSPACE } from '../../themes/listing-format'
import { VisitorAdd, VisitorThemeAdd } from '../visitor-add'
import { VISITOR_SIGN_IN, VISITOR_START_FREE, VisitorHeader } from '../visitor-header'

const ARCADE = '/marketplace?tab=themes&item=automatos.arcade&add=1'
const WRITER = '/marketplace?tab=agents&item=writer&add=1'
const encoded = (path: string) => encodeURIComponent(path)
const cancelNavigation = (event: Event) => event.preventDefault()
const ARCADE_LISTING = { listingId: 'automatos.arcade', themeId: 'automatos/arcade' } as ThemeListing

const fetchSpy = vi.fn()

beforeEach(() => {
  window.sessionStorage.clear()
  nav.pathname = '/marketplace'
  nav.params = new URLSearchParams()
  document.addEventListener('click', cancelNavigation)
  fetchSpy.mockReset()
  vi.stubGlobal('fetch', fetchSpy)
})

afterEach(() => {
  for (const call of [api.get, api.post, api.put, api.delete]) expect(call).not.toHaveBeenCalled()
  expect(fetchSpy).not.toHaveBeenCalled()
  cleanup()
  document.removeEventListener('click', cancelNavigation)
  window.sessionStorage.clear()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Add to my workspace, signed out (PRD-270 US-270-09a)', () => {
  it('a theme: goes to sign-in carrying the item, and remembers it', () => {
    render(<VisitorThemeAdd listing={ARCADE_LISTING} canManage={false} />)
    const add = screen.getByRole('link', { name: ADD_TO_WORKSPACE })

    expect(add).toHaveAttribute('href', `/sign-in?redirect_url=${encoded(ARCADE)}`)
    expect(window.sessionStorage.getItem(RETURN_PATH_STORAGE_KEY)).toBeNull()

    fireEvent.click(add)
    expect(take()).toBe(ARCADE)
  })

  it("another tab's item: its tab and key ride the link and the remembered path", () => {
    render(<VisitorAdd tab="agents" item="writer" />)
    const add = screen.getByRole('link', { name: ADD_TO_WORKSPACE })

    expect(add).toHaveAttribute('href', `/sign-in?redirect_url=${encoded(WRITER)}`)
    fireEvent.click(add)
    expect(take()).toBe(WRITER)
  })
})

describe('the visitor header (PRD-270 US-270-09a)', () => {
  it('Sign in carries and remembers the Marketplace page the visitor is on', () => {
    nav.params = new URLSearchParams({ tab: 'agents', item: 'writer' })
    render(<VisitorHeader />)
    const signIn = screen.getByRole('link', { name: VISITOR_SIGN_IN })
    const page = '/marketplace?tab=agents&item=writer'

    expect(signIn).toHaveAttribute('href', `/sign-in?redirect_url=${encoded(page)}`)
    fireEvent.click(signIn)
    expect(take()).toBe(page)
  })

  it('Start free goes to sign-up carrying the Marketplace page, and remembers it', () => {
    nav.params = new URLSearchParams({ tab: 'themes', item: 'automatos.arcade', add: '1' })
    render(<VisitorHeader />)
    const startFree = screen.getByRole('link', { name: VISITOR_START_FREE })

    expect(startFree).toHaveAttribute('href', `/sign-up?redirect_url=${encoded(ARCADE)}`)
    fireEvent.click(startFree)
    expect(take()).toBe(ARCADE)
  })

  it('the bare Marketplace: both carry /marketplace', () => {
    render(<VisitorHeader />)

    expect(screen.getByRole('link', { name: VISITOR_SIGN_IN })).toHaveAttribute('href', '/sign-in?redirect_url=%2Fmarketplace')
    expect(screen.getByRole('link', { name: VISITOR_START_FREE })).toHaveAttribute('href', '/sign-up?redirect_url=%2Fmarketplace')
  })

  it('only the tab, item and add ride along; anything else is dropped', () => {
    nav.params = new URLSearchParams({ tab: 'agents', next: 'https://evil.example', redirect_url: '//evil.example' })
    render(<VisitorHeader />)
    const startFree = screen.getByRole('link', { name: VISITOR_START_FREE })

    expect(startFree).toHaveAttribute('href', `/sign-up?redirect_url=${encoded('/marketplace?tab=agents')}`)
    fireEvent.click(startFree)
    expect(take()).toBe('/marketplace?tab=agents')
  })

  it('a page that is not the Marketplace rides as /marketplace', () => {
    nav.pathname = '/marketplace/publish'
    render(<VisitorHeader />)
    const signIn = screen.getByRole('link', { name: VISITOR_SIGN_IN })

    expect(signIn).toHaveAttribute('href', '/sign-in?redirect_url=%2Fmarketplace')
    fireEvent.click(signIn)
    expect(take()).toBe('/marketplace')
  })
})
