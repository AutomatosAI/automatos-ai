/**
 * PRD-261 A-19e2 — the core's chrome below a theme's width, on the fixture theme
 * (lib/theme-host/__tests__/fixture-theme-entry.ts, A-19b: an installed theme, minWidth 1024 px,
 * which registers a frame, /chat and /deliverables pages and a 404). The real width source runs: the
 * UiStyleProvider under the fixture's style and entry, hooks/use-studio-theme.ts's useThemeLayout and
 * hooks/use-mobile.ts over a window of a given width (hooks/__tests__/window-width.ts), with the
 * theme host's value as its provider hands it over. Each registered component is wrapped in a spy.
 *
 * On a phone: MainLayout draws the core's fallback chrome with the page in it from its first render,
 * and the theme's frame never renders, not even for one render (a client navigation: a fresh mount
 * under a host that already has the theme loaded; useIsTabletOrBelow reads the width on the first
 * render, W-L1); the chat and Deliverables layouts and the 404 show the core's own versions, and the
 * theme's page or 404 never mounts, not even for a render; widening the window past the theme's width
 * brings the frame in. On a laptop the same trees are the theme's (the frame around the page, which
 * mounts once; the theme's chat in it; its 404). On the server, where no width is known, they render
 * nothing of the theme and no core page.
 *
 * Replaces, in the core test ledger (plan §2.2.8), the core assertions of
 * components/play/__tests__/play-layout-phone.test.tsx (the core's chrome below 1024 px under a theme,
 * the plain 404, the chat layout's core chat, nothing of the theme on the server) and the "below
 * 1024 px renders the existing fallback chrome" case of components/play/shell/__tests__/play-frame.test.tsx.
 * Play's own screens (its Conversations redirect, its look pages' note) move to the kit (B-05f).
 */
import React, { useEffect, type ComponentType, type ReactNode } from 'react'
import { renderToString } from 'react-dom/server'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ pathname: '/command-center', search: '' }))
vi.mock('next/navigation', async (importOriginal) => ({
  ...(await importOriginal<typeof import('next/navigation')>()),
  usePathname: () => nav.pathname,
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}))
// The core's chrome, its parts left out (their own tests cover them): the fallback header marks it.
vi.mock('@/components/layout/sidebar', () => ({ Sidebar: () => null }))
vi.mock('@/components/layout/mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('@/components/layout/header', () => ({ Header: () => <header data-testid="fallback-header" /> }))
vi.mock('@/components/layout/studio-sidebar', () => ({ StudioSidebar: () => null }))
vi.mock('@/components/layout/studio-header', () => ({ StudioHeader: () => null, StudioHeaderUtilities: () => null }))
vi.mock('@/components/chatbot/chat-widget', () => ({ AutoWidget: () => null }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children?: ReactNode }) => <>{children}</>,
  SheetContent: () => null,
}))
// The chat layout's AI label and Discuss bar have their own tests.
vi.mock('@/components/ai-label/ai-label', () => ({ AiLabelGate: ({ children }: { children?: ReactNode }) => <>{children}</> }))
vi.mock('@/components/chatbot/discussion-bar', () => ({ DiscussionBar: () => null }))
vi.mock('@/components/deliverables/deliverable-deep-link', () => ({ DeliverableDeepLink: () => <div data-testid="deep-link" /> }))

import ChatLayout from '@/app/chat/layout'
import DeliverablesLayout from '@/app/deliverables/layout'
import NotFound from '@/app/not-found'
import { MainLayout } from '@/components/layout/main-layout'
import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { LAPTOP_WIDTH_PX, PHONE_WIDTH_PX, stubWindowWidth, type WindowWidth } from '@/hooks/__tests__/window-width'
import { FRAME_SLOT } from '@/lib/theme-host/constants'
import {
  FIXTURE_THEME_ENTRY,
  FIXTURE_THEME_REGISTRATIONS,
  FIXTURE_THEME_STYLE,
  FIXTURE_THEME_TEST_IDS,
  fixtureThemeHost,
} from '@/lib/theme-host/__tests__/fixture-theme-entry'
import type { SlotId } from '@/lib/theme-host/registry-types'
import type { FrameProps } from '@/lib/theme-host/types'
import { CHAT_HREF, DELIVERABLES_HREF } from '@/lib/ticket-links'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'

const ids = FIXTURE_THEME_TEST_IDS
const NOT_FOUND_SLOT = 'notFound@1'
const CORE_CHAT = 'The core chat page'
const CORE_DELIVERABLES = 'The core Deliverables page'
const STUDIO_PAGE = 'A Studio page the theme has no design for'
const PLAIN_404 = 'This page could not be found.'

/** Every registered component that rendered, by its slot or page, render by render. */
const rendered = vi.hoisted(() => ({ names: [] as string[], pageMounts: 0 }))

/** A registered screen (it takes no props) that says when it renders. */
function spiedScreen(name: string, component: unknown): ComponentType {
  const Screen = component as ComponentType
  function SpiedScreen() {
    rendered.names.push(name)
    return <Screen />
  }
  return SpiedScreen
}

function spiedMap<K extends string>(map: ReadonlyMap<K, unknown>): Map<K, ComponentType> {
  return new Map(Array.from(map, ([key, component]) => [key, spiedScreen(key, component)] as const))
}

const FixtureFrame = FIXTURE_THEME_REGISTRATIONS.frame

/** The fixture's frame, saying when it renders. */
function SpiedFrame(props: FrameProps) {
  rendered.names.push(FRAME_SLOT)
  return <FixtureFrame {...props} />
}

const HOST = fixtureThemeHost('active', {
  frame: SpiedFrame,
  slots: new Map<SlotId, unknown>([...spiedMap(FIXTURE_THEME_REGISTRATIONS.slots), [FRAME_SLOT, SpiedFrame]]),
  pages: spiedMap(FIXTURE_THEME_REGISTRATIONS.pages),
})

/** A Studio page inside MainLayout, counting its mounts. */
function CountedPage() {
  useEffect(() => {
    rendered.pageMounts += 1
  }, [])
  return <p>{STUDIO_PAGE}</p>
}

/** Under the fixture theme's style and entry, inside the theme host that loaded it. */
function underTheme(tree: ReactNode) {
  return (
    <UiStyleProvider initialStyle={FIXTURE_THEME_STYLE} installedTheme={FIXTURE_THEME_ENTRY}>
      <ThemeHostContext.Provider value={HOST}>{tree}</ThemeHostContext.Provider>
    </UiStyleProvider>
  )
}

const mainLayout = () => (
  <MainLayout>
    <CountedPage />
  </MainLayout>
)
const chatRoute = () => (
  <ChatLayout>
    <p>{CORE_CHAT}</p>
  </ChatLayout>
)
const deliverablesRoute = () => (
  <DeliverablesLayout>
    <p>{CORE_DELIVERABLES}</p>
  </DeliverablesLayout>
)

let width: WindowWidth | null = null

beforeEach(() => {
  // A browser already moved to the current defaults, so the provider keeps the fixture's style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
})

afterEach(() => {
  cleanup()
  width?.restore()
  width = null
  window.localStorage.clear()
  Object.assign(nav, { pathname: '/command-center', search: '' })
  Object.assign(rendered, { names: [], pageMounts: 0 })
})

describe("below the theme's width (a phone), the core's chrome and pages", () => {
  beforeEach(() => {
    width = stubWindowWidth(PHONE_WIDTH_PX)
  })

  it("MainLayout settles on the fallback chrome with the page in it, never the theme's frame", () => {
    render(underTheme(mainLayout()))
    expect(screen.getByTestId('fallback-header')).toBeInTheDocument()
    expect(screen.getByText(STUDIO_PAGE)).toBeInTheDocument()
    expect(screen.queryByTestId(ids.frame)).toBeNull()
  })

  it("MainLayout never renders the theme's frame on a client navigation, not even for one render; the page mounts once", () => {
    render(underTheme(mainLayout()))
    expect(rendered.names, "the theme's frame rendered on a phone").not.toContain(FRAME_SLOT)
    expect(rendered.pageMounts, 'the page mounted in one chrome and again in another').toBe(1)
  })

  it("the chat layout shows the core's chat; the theme's chat never mounts", () => {
    Object.assign(nav, { pathname: CHAT_HREF, search: 'chatId=c1' })
    render(underTheme(chatRoute()))
    expect(screen.getByText(CORE_CHAT)).toBeInTheDocument()
    expect(rendered.names).not.toContain(CHAT_HREF)
    expect(screen.queryByTestId(ids.chat)).toBeNull()
  })

  it("the Deliverables layout keeps the core page and its deep link for a ?deliverable= link; the theme's page never mounts", () => {
    Object.assign(nav, { pathname: DELIVERABLES_HREF, search: 'deliverable=d-1' })
    render(underTheme(deliverablesRoute()))
    expect(screen.getByText(CORE_DELIVERABLES)).toBeInTheDocument()
    expect(screen.getByTestId('deep-link')).toBeInTheDocument()
    expect(rendered.names).not.toContain(DELIVERABLES_HREF)
  })

  it("the 404 is the plain page; the theme's 404 never mounts", () => {
    nav.pathname = '/nowhere'
    render(underTheme(<NotFound />))
    expect(screen.getByText(PLAIN_404)).toBeInTheDocument()
    expect(rendered.names).not.toContain(NOT_FOUND_SLOT)
  })

  it("widening the window past the theme's width brings the theme's frame in around the page", () => {
    render(underTheme(mainLayout()))
    expect(screen.queryByTestId(ids.frame)).toBeNull()
    act(() => width?.resize(LAPTOP_WIDTH_PX))
    expect(screen.getByTestId(ids.frame)).toContainElement(screen.getByText(STUDIO_PAGE))
    expect(screen.queryByTestId('fallback-header')).toBeNull()
  })
})

describe("at the theme's width (a laptop), the same trees are the theme's", () => {
  beforeEach(() => {
    width = stubWindowWidth(LAPTOP_WIDTH_PX)
  })

  it("MainLayout draws the theme's frame around the page, which mounts once (M69)", () => {
    render(underTheme(mainLayout()))
    expect(screen.getByTestId(ids.frame)).toContainElement(screen.getByText(STUDIO_PAGE))
    expect(screen.queryByTestId('fallback-header')).toBeNull()
    expect(rendered.pageMounts).toBe(1)
  })

  it("MainLayout renders the theme's frame (the spy the phone case relies on sees it here)", () => {
    render(underTheme(mainLayout()))
    expect(rendered.names).toContain(FRAME_SLOT)
  })

  it("the chat layout draws the theme's chat inside its frame, not the core's", () => {
    nav.pathname = CHAT_HREF
    render(underTheme(chatRoute()))
    expect(screen.getByTestId(ids.frame)).toContainElement(screen.getByTestId(ids.chat))
    expect(screen.queryByText(CORE_CHAT)).toBeNull()
  })

  it("the 404 is the theme's", () => {
    nav.pathname = '/nowhere'
    render(underTheme(<NotFound />))
    expect(screen.getByTestId(ids.notFound)).toBeInTheDocument()
    expect(screen.queryByText(PLAIN_404)).toBeNull()
  })
})

describe('on the server, where no width is known', () => {
  it('MainLayout, the Deliverables layout and the 404 send nothing', () => {
    const trees: ReadonlyArray<readonly [string, ReactNode]> = [
      ['/command-center', mainLayout()],
      [DELIVERABLES_HREF, deliverablesRoute()],
      ['/nowhere', <NotFound key="404" />],
    ]
    for (const [pathname, tree] of trees) {
      nav.pathname = pathname
      expect(renderToString(underTheme(tree)), pathname).toBe('')
    }
    expect(rendered.names).toEqual([])
  })

  it("the chat layout sends neither the theme's chat nor the core's", () => {
    nav.pathname = CHAT_HREF
    const html = renderToString(underTheme(chatRoute()))
    expect(html).not.toContain(CORE_CHAT)
    expect(html).not.toContain(ids.chat)
    expect(rendered.names).toEqual([])
  })
})
