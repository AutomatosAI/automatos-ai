/**
 * PRD-261 A-04b — a theme's own routes at /x/<route> (registerRoute). The server page answers the
 * 404 for an address no theme of the style can serve (a route outside the pattern, Classic, Studio,
 * a style cookie that names no theme this build knows, which reads as the default style) and
 * otherwise, under an installed theme, renders ThemeSlotRoute full bleed inside MainLayout, which
 * sits inside the host's AI label gate (PRD-264 EUAI-11a). So /x/conversations is a 404 under
 * Classic and Studio: no theme is compiled into the app (A-19f1).
 * ThemeSlotRoute renders the active theme's route from the theme host (the fixture theme's
 * `fixture-route`, PRD-261 A-19b, and an installed theme's), at any width, nothing while an
 * installed theme loads, and calls notFound() when nothing is registered by that name.
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

import ThemeRoutePage from '@/app/x/[...path]/page'
import { AiLabelGate } from '@/components/ai-label/ai-label'
import { MainLayout } from '@/components/layout/main-layout'
import {
  FIXTURE_THEME_ENTRY,
  FIXTURE_THEME_ROUTE,
  FIXTURE_THEME_STYLE,
  FIXTURE_THEME_TEST_IDS,
  fixtureThemeHost,
} from '@/lib/theme-host/__tests__/fixture-theme-entry'

import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'
import { ThemeSlotRoute } from '../theme-slot-route'
import { OuterBoundary } from './next-router-errors'

const INSTALLED_STYLE = 'theme:install:00000000-0000-4000-8000-0000000a04b1'
const NOT_FOUND_DIGEST = /^NEXT_HTTP_ERROR_FALLBACK;404/
/** A style that names no theme this build knows: it reads as the default style, Studio. */
const NO_THEME_STYLE = 'theme:install:not-a-uuid'

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
function renderRoute(route: string, host: ThemeHostContextValue | null = null) {
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
  it("renders the theme's route 'fixture-route' from its registrations", async () => {
    state.entry = FIXTURE_THEME_ENTRY
    renderRoute(FIXTURE_THEME_ROUTE, fixtureThemeHost())
    expect(await screen.findByTestId(FIXTURE_THEME_TEST_IDS.route)).toBeInTheDocument()
  })

  it("renders the theme's route below its width too: the route decides itself what a phone gets", async () => {
    state.entry = FIXTURE_THEME_ENTRY
    state.layout = 'fallback'
    renderRoute(FIXTURE_THEME_ROUTE, fixtureThemeHost())
    expect(await screen.findByTestId(FIXTURE_THEME_TEST_IDS.route)).toBeInTheDocument()
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
    ['a route the theme did not register', FIXTURE_THEME_ENTRY, true, 'hello'],
    ['no theme (Classic or Studio)', null, false, 'conversations'],
  ])('is the 404 for %s', (_label, entry, themed, route) => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    state.entry = entry
    const onCaught = renderRoute(route, themed ? fixtureThemeHost() : null)
    expect(onCaught).toHaveBeenCalledWith(expect.objectContaining({ digest: expect.stringMatching(NOT_FOUND_DIGEST) }))
  })
})

describe('app/x/[...path]/page.tsx', () => {
  it.each([
    ["an installed theme's style (its routes are known in the browser)", INSTALLED_STYLE, ['hello']],
    ["the fixture theme's style, its own route", FIXTURE_THEME_STYLE, [FIXTURE_THEME_ROUTE]],
    ['an installed theme, a nested route', INSTALLED_STYLE, ['hello', 'there']],
  ])('renders the route full bleed in MainLayout, inside the host AI label gate, under %s', async (_label, style, path) => {
    state.style = style
    const gate = await ThemeRoutePage(params(...path))
    // PRD-264 EUAI-11a: the gate wraps the theme's frame, so the label sits outside it, on the left
    expect(gate.type).toBe(AiLabelGate)
    expect(gate.props.placement).toBe('bottom-left')
    const page = gate.props.children
    expect(page.type).toBe(MainLayout)
    expect(page.props.fullBleed).toBe(true)
    expect(page.props.children).toEqual(<ThemeSlotRoute route={path.join('/')} />)
  })

  it.each([
    ['Studio', 'studio', ['conversations']],
    ['Classic', 'classic', ['conversations']],
    ['no cookie (the default style)', undefined, ['conversations']],
    ['a style that names no theme this build knows', NO_THEME_STYLE, [FIXTURE_THEME_ROUTE]],
    ['another style that names no theme', 'theme:other', ['conversations']],
    ['an installed theme, a route outside the pattern', INSTALLED_STYLE, ['Hello']],
    ['an installed theme, five segments', INSTALLED_STYLE, ['a', 'b', 'c', 'd', 'e']],
  ])('answers the 404 on the server under %s', async (_label, style, path) => {
    state.style = style
    await expect(ThemeRoutePage(params(...path))).rejects.toMatchObject({ digest: expect.stringMatching(NOT_FOUND_DIGEST) })
  })
})
