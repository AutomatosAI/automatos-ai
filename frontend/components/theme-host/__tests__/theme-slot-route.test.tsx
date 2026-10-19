/**
 * PRD-261 A-04b — a theme's own routes at /x/<route> (registerRoute). The server page answers the
 * 404 for an address no theme of the style can serve (a bad route, Classic, Studio, a builtin
 * theme without the route) and otherwise renders ThemeSlotRoute full bleed inside MainLayout.
 * ThemeSlotRoute renders the active theme's route (builtin Play's Conversations through its lazy
 * slot, findBy*; an installed theme's from the theme host), nothing while an installed theme
 * loads, and calls notFound() when nothing is registered by that name.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'

const state = vi.hoisted(() => ({ entry: null as ThemeEntry | null, layout: 'theme' as ThemeLayout, style: undefined as string | undefined }))

vi.mock('next/headers', () => ({
  cookies: async () => ({
    get: (name: string) => (name === 'automatos-style' && state.style !== undefined ? { name, value: state.style } : undefined),
  }),
}))
vi.mock('@/hooks/use-studio-theme', () => ({
  useIsStudio: () => true,
  useActiveTheme: () => state.entry,
  useThemeLayout: () => state.layout,
}))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children, fullBleed }: { children: React.ReactNode; fullBleed?: boolean }) => (
    <div data-testid="main-layout" data-full-bleed={fullBleed ? 'true' : 'false'}>
      {children}
    </div>
  ),
}))
vi.mock('@/components/play/conversations/play-conversations', () => ({
  PlayConversations: () => <div data-testid="play-conversations" />,
}))

import ThemeRoutePage from '@/app/x/[...path]/page'
import { MainLayout } from '@/components/layout/main-layout'
import { BUILTIN_PLAY_ENTRY } from '@/lib/theme-host/builtin/play-entry'

import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'
import { ThemeSlotRoute } from '../theme-slot-route'
import { OuterBoundary } from './next-router-errors'

const INSTALLED_STYLE = 'theme:install:00000000-0000-4000-8000-0000000a04b1'
const NOT_FOUND_DIGEST = /^NEXT_HTTP_ERROR_FALLBACK;404/

function HelloFrame({ children }: { children?: React.ReactNode }) {
  return <>{children}</>
}

function HelloRoute() {
  return <div data-testid="hello-route">Hello</div>
}

type HostValue = ThemeHostContextValue & { readonly routes: ReadonlyMap<string, unknown> }

function installedHost(status: 'active' | 'loading'): HostValue {
  const active = status === 'active'
  return {
    status,
    reason: null,
    frame: active ? HelloFrame : null,
    slots: null,
    routes: new Map<string, unknown>(active ? [['hello', HelloRoute]] : []),
    reportRenderFailure: vi.fn(),
  }
}

/** ThemeSlotRoute for `route`, under `host` when given, with a boundary standing in for Next's 404. */
function renderRoute(route: string, host: HostValue | null = null) {
  const onCaught = vi.fn()
  const tree = (
    <OuterBoundary onCaught={onCaught}>
      <ThemeSlotRoute route={route} />
    </OuterBoundary>
  )
  render(host === null ? tree : <ThemeHostContext.Provider value={host}>{tree}</ThemeHostContext.Provider>)
  return onCaught
}

function params(...path: string[]) {
  return { params: Promise.resolve({ path }) }
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  Object.assign(state, { entry: null, layout: 'theme', style: undefined })
})

describe('ThemeSlotRoute', () => {
  it("renders builtin Play's route 'conversations' through its lazy slot", async () => {
    state.entry = BUILTIN_PLAY_ENTRY
    renderRoute('conversations')
    expect(await screen.findByTestId('play-conversations')).toBeInTheDocument()
  })

  it("renders builtin Play's route below its width too: the route decides itself (Conversations sends a phone to /chat)", async () => {
    state.entry = BUILTIN_PLAY_ENTRY
    state.layout = 'fallback'
    renderRoute('conversations')
    expect(await screen.findByTestId('play-conversations')).toBeInTheDocument()
  })

  it("renders an installed theme's route from the theme host", () => {
    expect(renderRoute('hello', installedHost('active'))).not.toHaveBeenCalled()
    expect(screen.getByTestId('hello-route')).toBeInTheDocument()
  })

  it('renders nothing while an installed theme loads', () => {
    const onCaught = renderRoute('hello', installedHost('loading'))
    expect(onCaught).not.toHaveBeenCalled()
    expect(screen.queryByTestId('hello-route')).toBeNull()
  })

  it.each([
    ['a route the theme did not register', BUILTIN_PLAY_ENTRY, 'hello'],
    ['no theme (Classic or Studio)', null, 'conversations'],
  ])('is the 404 for %s', (_label, entry, route) => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    state.entry = entry
    const onCaught = renderRoute(route)
    expect(onCaught).toHaveBeenCalledWith(expect.objectContaining({ digest: expect.stringMatching(NOT_FOUND_DIGEST) }))
  })
})

describe('app/x/[...path]/page.tsx', () => {
  it.each([
    ['the style builtin Play is', 'theme:builtin:play', ['conversations']],
    ['a legacy play cookie', 'play', ['conversations']],
    ["an installed theme's style (its routes are known in the browser)", INSTALLED_STYLE, ['hello']],
    ['an installed theme, a nested route', INSTALLED_STYLE, ['hello', 'there']],
  ])('renders the route full bleed in MainLayout under %s', async (_label, style, path) => {
    state.style = style
    const page = await ThemeRoutePage(params(...path))
    expect(page.type).toBe(MainLayout)
    expect(page.props.fullBleed).toBe(true)
    expect(page.props.children).toEqual(<ThemeSlotRoute route={path.join('/')} />)
  })

  it.each([
    ['Studio', 'studio', ['conversations']],
    ['Classic', 'classic', ['conversations']],
    ['no cookie (the default style)', undefined, ['conversations']],
    ['builtin Play, a route it does not serve', 'theme:builtin:play', ['hello']],
    ['an installed theme, a route outside the pattern', INSTALLED_STYLE, ['Hello']],
    ['an installed theme, five segments', INSTALLED_STYLE, ['a', 'b', 'c', 'd', 'e']],
  ])('answers the 404 on the server under %s', async (_label, style, path) => {
    state.style = style
    await expect(ThemeRoutePage(params(...path))).rejects.toMatchObject({ digest: expect.stringMatching(NOT_FOUND_DIGEST) })
  })
})
