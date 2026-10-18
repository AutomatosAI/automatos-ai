/**
 * PRD-261 A-10a — SDK area `navigation`. The router and HostLink take a path inside the app or,
 * for a link, https out of it, and refuse everything else; the core nav is the Studio menu after
 * the host's role and plan filters, with no icons and no second route list.
 */
import { render, renderHook, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({
  router: { push: vi.fn(), replace: vi.fn(), back: vi.fn(), refresh: vi.fn(), prefetch: vi.fn() },
  pathname: '/agents' as string | null,
}))

vi.mock('next/navigation', () => ({ useRouter: () => nav.router, usePathname: () => nav.pathname }))

import { filterNavForEdition } from '@/lib/auth-edition'
import { STUDIO_MENU_FOOTER, STUDIO_MENU_PRIMARY } from '@/lib/studio-menu'
import { buildCoreNav } from '../internal/core-nav'
import { isExternalHref, isInternalHref } from '../internal/href'
import { HostLink, resolveActiveNavId, useHostPathname, useHostRouter } from '../navigation'

const REFUSED = [
  '//evil.example',
  '/\\evil.example',
  'javascript:alert(1)',
  'http://plain.example',
  'chat',
  '',
  ' /chat',
  // A browser strips tab/LF/CR, so '/\t/evil.com' is '//evil.com'; backslash or space anywhere too.
  '/\t/evil.com',
  '/\n/evil.com',
  '/\r\\evil.com',
  '/ /x',
  '/a\\b',
  '/chat\u0000',
  '/chat\u007f',
]
const REFUSED_EXTERNAL = [
  'https://docs.automatos.app/\tx',
  'https://docs.automatos.app/ x',
  'https://docs\u007f.app',
]

afterEach(() => {
  Object.values(nav.router).forEach((fn) => fn.mockClear())
  nav.pathname = '/agents'
})

describe('PRD-261 A-10a — hrefs', () => {
  it('accepts paths inside the app and https links out', () => {
    expect(['/', '/chat', '/missions/7?tab=log'].every(isInternalHref)).toBe(true)
    expect(isExternalHref('https://docs.automatos.app')).toBe(true)
  })

  it.each(REFUSED)('refuses %j as a path inside the app', (href) => {
    expect(isInternalHref(href)).toBe(false)
  })

  it.each(REFUSED_EXTERNAL)('refuses %j as an https link out', (href) => {
    expect(isExternalHref(href)).toBe(false)
  })
})

describe('PRD-261 A-10a — useHostRouter and useHostPathname', () => {
  it("pushes, replaces and prefetches paths on the App Router's own router", () => {
    const router = renderHook(() => useHostRouter()).result.current
    router.push('/chat')
    router.replace('/deliverables')
    router.prefetch('/agents')
    router.back()
    router.refresh()
    expect(nav.router.push).toHaveBeenCalledWith('/chat')
    expect(nav.router.replace).toHaveBeenCalledWith('/deliverables')
    expect(nav.router.prefetch).toHaveBeenCalledWith('/agents')
    expect(nav.router.back).toHaveBeenCalledTimes(1)
    expect(nav.router.refresh).toHaveBeenCalledTimes(1)
  })

  it.each(REFUSED)('throws for %j and never navigates', (href) => {
    const router = renderHook(() => useHostRouter()).result.current
    expect(() => router.push(href)).toThrow(TypeError)
    expect(() => router.replace(href)).toThrow(TypeError)
    expect(nav.router.push).not.toHaveBeenCalled()
    expect(nav.router.replace).not.toHaveBeenCalled()
  })

  it('reads the path, and / when the router has none', () => {
    expect(renderHook(() => useHostPathname()).result.current).toBe('/agents')
    nav.pathname = null
    expect(renderHook(() => useHostPathname()).result.current).toBe('/')
  })
})

describe('PRD-261 A-10a — HostLink', () => {
  it('links a path inside the app in place', () => {
    render(<HostLink href="/chat" className="pill">Chat</HostLink>)
    const link = screen.getByRole('link', { name: 'Chat' })
    expect(link).toHaveAttribute('href', '/chat')
    expect(link).toHaveClass('pill')
    expect(link).not.toHaveAttribute('target')
  })

  it('opens an https link in a new tab, without an opener', () => {
    render(<HostLink href="https://docs.automatos.app">Docs</HostLink>)
    const link = screen.getByRole('link', { name: 'Docs' })
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('throws for any other href', () => {
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    try {
      expect(() => render(<HostLink href="javascript:alert(1)">Bad</HostLink>)).toThrow(TypeError)
    } finally {
      quiet.mockRestore()
    }
  })
})

describe('PRD-261 A-10a — the core nav', () => {
  it("is the Studio menu's items for this edition, in order, for an admin", () => {
    const coreNav = buildCoreNav({ isAdmin: true, exposure: null })
    expect(coreNav.primary.map((item) => item.href)).toEqual(filterNavForEdition(STUDIO_MENU_PRIMARY).map((item) => item.href))
    expect(coreNav.footer.map((item) => item.id)).toEqual(filterNavForEdition(STUDIO_MENU_FOOTER).map((item) => item.id))
    expect(coreNav.primary[0]).toEqual({ id: 'chat', label: 'Chat', description: 'Your AI workspace', href: '/chat', group: 'OPERATIONS', external: false })
    expect(coreNav.footer.find((item) => item.id === 'docs')).toMatchObject({ external: true, group: null, description: null })
  })

  it('hides admin-only items from a member and items the plan turns off', () => {
    const exposure = { plan: 'basic', families: {}, marketplace_depth: 0, nav: { team: false } }
    const coreNav = buildCoreNav({ isAdmin: false, exposure })
    const ids = [...coreNav.primary, ...coreNav.footer].map((item) => item.id)
    expect(ids).not.toContain('admin')
    expect(ids).not.toContain('settings')
    expect(ids).not.toContain('team')
    expect(ids).toContain('analytics')
  })

  it('hands out frozen items with no icon', () => {
    const coreNav = buildCoreNav({ isAdmin: true, exposure: null })
    expect(Object.isFrozen(coreNav.primary)).toBe(true)
    expect(coreNav.primary.every((item) => Object.isFrozen(item) && !('icon' in item))).toBe(true)
  })

  it("resolves a path to its nav item with the host's own resolver", () => {
    expect(resolveActiveNavId('/chat/123')).toBe('chat')
    expect(resolveActiveNavId('/missions/7')).toBe('assign')
    expect(resolveActiveNavId('/nowhere')).toBeNull()
  })
})
