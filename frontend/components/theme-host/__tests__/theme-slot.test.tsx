/**
 * PRD-261 A-04b — ThemeSlot, the seam a theme's page or slot mounts through (ported from PRD-260
 * K7's play-route-switch.test.tsx, which it replaces), and the chat layout that uses it.
 *
 * A page entry: the theme's page on its route at the theme layout; the core page with no theme,
 * below the theme's width ('narrow'), on another path, for a link into a surface only the core
 * page has (`keepCorePage`, P260-FX-11 M13/M14: a value the theme's page serves keeps it), and
 * where the theme registered no such page. While the width isn't known, or an installed theme is
 * still loading, nothing renders where the theme decides (P260-FX-25, M69). The page entry reads
 * the search params with no Suspense boundary of its own, so the core page is never a fallback
 * (theme-slot-server.test.tsx proves the server's HTML), and the core page sits at one place in the
 * tree: it mounts once when the width becomes known, an installed theme settles or fails, or the
 * style switches in place (M69). A slot: the theme's version where it registered one, else the core's.
 *
 * An installed theme's registrations come from the theme host (provided here as the provider
 * provides it, with its pages); builtin Play's from its module constants, through the chat layout
 * (its chat comes through its lazy slot, so it arrives a moment after the first render: findBy*).
 * A-04d: an installed theme's page draws full bleed inside MainLayout (the theme's frame, stood in
 * for here); builtin Play's pages draw MainLayout themselves, so ThemeSlot never frames them, and
 * a slot is drawn where it is placed, never framed. A-12a: a page entry hands the core page down
 * to the theme's page, so the theme's `cards.CorePage` (re-exported from host-cards/core-page)
 * draws the real core page there; a slot has none, so CorePage in it draws its own children.
 */
import React, { Suspense, useEffect } from 'react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SlotId, ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'

const state = vi.hoisted(() => ({
  entry: null as ThemeEntry | null,
  layout: 'fallback' as ThemeLayout,
  pathname: '/chat',
  search: '',
  /** While set, reading the search params suspends on it (Next's static-prerender bailout). */
  pending: null as Promise<void> | null,
  /** How many times the counted core page mounted. */
  mounts: 0,
}))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => {
    if (state.pending !== null) throw state.pending
    return new URLSearchParams(state.search)
  },
}))
vi.mock('@/hooks/use-studio-theme', () => ({
  useIsStudio: () => true,
  useActiveTheme: () => state.entry,
  useThemeLayout: () => state.layout,
}))
// Builtin Play's /chat slot, which draws Play's chat in its frame (PRD-261 A-14a: MainLayout moved out of Play's chat into it).
vi.mock('@/lib/theme-host/builtin/play-chat-lazy', () => ({ PlayChat: () => <div data-testid="play-chat">Play chat</div> }))
vi.mock('@/components/chatbot/discussion-bar', () => ({ DiscussionBar: () => <div data-testid="discussion-bar" /> }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children, fullBleed }: { children: React.ReactNode; fullBleed?: boolean }) => (
    <div data-testid="main-layout" data-full-bleed={fullBleed ? 'true' : 'false'}>
      {children}
    </div>
  ),
}))

import ChatLayout from '@/app/chat/layout'
import { BUILTIN_PLAY_ENTRY } from '@/lib/theme-host/builtin/play-entry'
import { CorePage } from '@/lib/theme-sdk/host-cards/core-page'
import type { ThemeHostStatus } from '@/lib/theme-host/types'

import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'
import { corePageAsked, routeMatches, ThemeSlot, type ThemeServedValues } from '../theme-slot'
import { INSTALLED_FRAME_MIN_WIDTH_PX } from '../theme-slot-registrations'

/** An installed theme's registry entry (A-02c): its own width, the theme layout from 1024 px. */
const INSTALLED_ENTRY: ThemeEntry = {
  ...BUILTIN_PLAY_ENTRY,
  ref: 'install:00000000-0000-4000-8000-0000000a04b0',
  kind: 'installed',
  ownsPages: ['/chat'],
}

function HelloFrame({ children }: { children?: React.ReactNode }) {
  return <>{children}</>
}

function ThemePage() {
  return <div data-testid="theme">The theme's page</div>
}

function ThemeLost() {
  return <div data-testid="theme">The theme's 404</div>
}

type HostValue = ThemeHostContextValue & { readonly pages: ReadonlyMap<string, unknown>; readonly routes: ReadonlyMap<string, unknown> }

/** The theme host as the provider gives it for an installed theme that registered a page on /chat and on /missions/[id]. */
function installedHost(status: ThemeHostStatus = 'active'): HostValue {
  const active = status === 'active'
  return {
    status,
    reason: status === 'failed' ? 'load' : null,
    frame: active ? HelloFrame : null,
    slots: active ? new Map<SlotId, unknown>([['notFound@1', ThemeLost]]) : null,
    pages: new Map<string, unknown>(active ? [['/chat', ThemePage], ['/missions/[id]', ThemePage]] : []),
    routes: new Map(),
    reportRenderFailure: vi.fn(),
  }
}

let host: HostValue | null = null

interface PageProps {
  readonly page?: string
  readonly keepCorePage?: readonly string[]
  readonly themeServes?: ThemeServedValues
}

/** A page entry around `core`, under the theme host when one is set. */
function pageTree(props: PageProps = {}, core: React.ReactNode = <div data-testid="page">The core page</div>) {
  const slot = (
    <ThemeSlot page={props.page ?? '/chat'} keepCorePage={props.keepCorePage} themeServes={props.themeServes}>
      {core}
    </ThemeSlot>
  )
  return host === null ? slot : <ThemeHostContext.Provider value={host}>{slot}</ThemeHostContext.Provider>
}

function renderPage(props: PageProps = {}) {
  return render(pageTree(props))
}

function shows(testId: 'theme' | 'page') {
  expect(screen.getByTestId(testId)).toBeInTheDocument()
  expect(screen.queryByTestId(testId === 'theme' ? 'page' : 'theme')).toBeNull()
}

function showsNothing() {
  expect(screen.queryByTestId('theme')).toBeNull()
  expect(screen.queryByTestId('page')).toBeNull()
}

/** An installed theme at the theme layout, active. */
function underInstalledTheme(status: ThemeHostStatus = 'active') {
  state.entry = INSTALLED_ENTRY
  state.layout = 'theme'
  host = installedHost(status)
}

afterEach(() => {
  cleanup()
  host = null
  Object.assign(state, { entry: null, layout: 'fallback', pathname: '/chat', search: '', pending: null, mounts: 0 })
})

describe('routeMatches and corePageAsked', () => {
  it('matches the same segments, a [param] matching any one non-empty segment', () => {
    expect(routeMatches('/chat', '/chat')).toBe(true)
    expect(routeMatches('/chat', '/chat/elsewhere')).toBe(false)
    expect(routeMatches('/', '/')).toBe(true)
    expect(routeMatches('/', '/chat')).toBe(false)
    expect(routeMatches('/missions/[id]', '/missions/m-1')).toBe(true)
    expect(routeMatches('/missions/[id]', '/missions')).toBe(false)
    expect(routeMatches('/missions/[id]', '/missions/')).toBe(false)
    expect(routeMatches('/chat', null)).toBe(false)
  })

  it('asks for the core page only for a parameter the theme does not serve', () => {
    const served = { tab: ['outputs'] }
    expect(corePageAsked(new URLSearchParams('tab=brand'), ['tab'], served)).toBe(true)
    expect(corePageAsked(new URLSearchParams('tab=outputs'), ['tab'], served)).toBe(false)
    expect(corePageAsked(new URLSearchParams('chatId=c1'), ['repo'], {})).toBe(false)
    expect(corePageAsked(null, ['repo'], {})).toBe(false)
  })
})

describe('ThemeSlot, a page entry', () => {
  it("renders the theme's page at the theme layout on its route", () => {
    underInstalledTheme()
    renderPage()
    shows('theme')
  })

  it("draws an installed theme's page full bleed inside MainLayout, the theme's frame (A-04d)", () => {
    underInstalledTheme()
    renderPage()
    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('theme'))
    expect(screen.getByTestId('main-layout')).toHaveAttribute('data-full-bleed', 'true')
  })

  it('never frames the core page: the core page draws its own MainLayout', () => {
    renderPage()
    shows('page')
    expect(screen.queryByTestId('main-layout')).toBeNull()
  })

  it('renders the core page with no theme', () => {
    renderPage()
    shows('page')
  })

  it("renders the core page below the theme's width (D7)", () => {
    underInstalledTheme()
    state.layout = 'fallback'
    renderPage()
    shows('page')
  })

  it('renders neither while the width is not known, so neither mounts for a render it then loses (M69)', () => {
    underInstalledTheme()
    state.layout = 'pending'
    renderPage()
    showsNothing()
  })

  it('renders the core page while the width is not known where the theme does not decide', () => {
    underInstalledTheme()
    state.layout = 'pending'
    state.pathname = '/chat/elsewhere'
    renderPage()
    shows('page')
    cleanup()
    state.pathname = '/chat'
    state.search = 'repo=projects%2Fsite'
    renderPage({ keepCorePage: ['repo'] })
    shows('page')
    cleanup()
    state.pathname = '/deliverables'
    renderPage({ page: '/deliverables' })
    shows('page')
  })

  it('renders nothing while an installed theme loads, whose pages are not known yet (M69)', () => {
    underInstalledTheme('loading')
    renderPage()
    showsNothing()
  })

  it('renders the core page when the installed theme failed, or registered no page here', () => {
    underInstalledTheme('failed')
    renderPage()
    shows('page')
    cleanup()
    underInstalledTheme()
    state.pathname = '/deliverables'
    renderPage({ page: '/deliverables' })
    shows('page')
  })

  it("renders an installed theme's page from the desktop width up while the registry has no entry for it (A-02c)", () => {
    underInstalledTheme()
    state.entry = null
    state.layout = 'fallback'
    renderPage()
    shows('theme')
  })

  it("renders the core page on a phone, where MainLayout draws no installed theme's frame (W-A2 review L3)", () => {
    const queries: string[] = []
    vi.stubGlobal('matchMedia', (query: string) => {
      queries.push(query)
      return { matches: true, media: query, addEventListener: vi.fn(), removeEventListener: vi.fn() }
    })
    try {
      underInstalledTheme()
      state.entry = null
      state.layout = 'fallback'
      renderPage()
      shows('page')
      expect(queries).toContain(`(max-width: ${INSTALLED_FRAME_MIN_WIDTH_PX - 1}px)`)
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('matches a page entry with a dynamic segment on any one value of it', () => {
    underInstalledTheme()
    state.pathname = '/missions/m-1'
    renderPage({ page: '/missions/[id]' })
    shows('theme')
  })

  it('renders the core page when the link asks for a surface only the core page has', () => {
    underInstalledTheme()
    state.search = 'repo=projects%2Fsite&ticket=42'
    renderPage({ keepCorePage: ['repo', 'runtime', 'mode'] })
    shows('page')
  })

  it("keeps the theme's page for any other query (a deep link to a conversation)", () => {
    underInstalledTheme()
    state.search = 'chatId=c1'
    renderPage({ keepCorePage: ['repo', 'runtime', 'mode'] })
    shows('theme')
  })

  describe('themeServes: values of a core-page parameter that the theme still serves', () => {
    const served = { keepCorePage: ['tab', 'post'], themeServes: { tab: ['outputs'] } }

    it("keeps the theme's page for a value it serves", () => {
      underInstalledTheme()
      state.search = 'tab=outputs'
      renderPage(served)
      shows('theme')
    })

    it.each(['tab=brand', 'tab=', 'tab=outputs&post=p1'])('renders the core page for ?%s', (search) => {
      underInstalledTheme()
      state.search = search
      renderPage(served)
      shows('page')
    })
  })

  it("adds no Suspense boundary of its own: a read that suspends reaches the route's, the core page never a fallback", async () => {
    underInstalledTheme()
    let release = () => {}
    state.pending = new Promise<void>((resolve) => {
      release = resolve
    })
    render(<Suspense fallback={<div data-testid="route-fallback" />}>{pageTree()}</Suspense>)
    expect(screen.getByTestId('route-fallback')).toBeInTheDocument()
    showsNothing()
    state.pending = null
    await act(async () => {
      release()
    })
    expect(await screen.findByTestId('theme')).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('route-fallback')).toBeNull()
  })
})

/** The core page, counting its mounts: a page mounted twice ran its effects twice (M69). */
function CountedPage() {
  useEffect(() => {
    state.mounts += 1
  }, [])
  return <div data-testid="page">The core page</div>
}

/** The page entry around the counted core page; `rerender` renders it again with the state as it is now. */
function renderCounted(props: PageProps) {
  const view = render(pageTree(props, <CountedPage />))
  return { rerender: () => view.rerender(pageTree(props, <CountedPage />)) }
}

describe('ThemeSlot keeps the core page in one place: it mounts once (P260-FX-25, M69)', () => {
  it('when the width becomes known, on a link into a surface only the core page has', () => {
    state.entry = BUILTIN_PLAY_ENTRY
    state.layout = 'pending'
    state.search = 'repo=projects%2Fsite'
    const view = renderCounted({ keepCorePage: ['repo'] })
    shows('page')
    for (const layout of ['fallback', 'theme'] as const) {
      state.layout = layout
      view.rerender()
      shows('page')
    }
    expect(state.mounts).toBe(1)
  })

  it('when an installed theme settles with no page here, or fails, on a link the core page keeps', () => {
    underInstalledTheme('loading')
    state.pathname = '/deliverables'
    state.search = 'tab=brand'
    const view = renderCounted({ page: '/deliverables', keepCorePage: ['tab'] })
    shows('page')
    for (const status of ['active', 'failed'] as const) {
      host = installedHost(status)
      view.rerender()
      shows('page')
    }
    expect(state.mounts).toBe(1)
  })

  // A style switch reaches ThemeSlot as the active theme's entry (useActiveTheme) and the host's status.
  it("when the style switches in place between Studio and builtin Play, below Play's width", () => {
    const view = renderCounted({})
    shows('page')
    for (const entry of [BUILTIN_PLAY_ENTRY, null]) {
      state.entry = entry
      view.rerender()
      shows('page')
    }
    expect(state.mounts).toBe(1)
  })

  it('when the style moves off a failed installed theme to Studio, on a link the core page keeps', () => {
    underInstalledTheme('failed')
    state.pathname = '/deliverables'
    state.search = 'tab=brand'
    const view = renderCounted({ page: '/deliverables', keepCorePage: ['tab'] })
    shows('page')
    state.entry = null
    state.layout = 'fallback'
    host = installedHost('off')
    view.rerender()
    shows('page')
    expect(state.mounts).toBe(1)
  })
})

describe("ThemeSlot hands the core page down to the theme's page (A-12a cards.CorePage)", () => {
  function ThemeKeepsCore() {
    return (
      <div data-testid="theme">
        <CorePage>
          <div data-testid="own-fallback">The theme's own fallback</div>
        </CorePage>
      </div>
    )
  }

  function ThemeLostKeepsCore() {
    return (
      <div data-testid="theme">
        <CorePage>
          <div data-testid="own-fallback">The theme's own 404</div>
        </CorePage>
      </div>
    )
  }

  it("renders the real core page where the theme's page places CorePage, not the theme's fallback", () => {
    underInstalledTheme()
    host = { ...installedHost(), pages: new Map<string, unknown>([['/chat', ThemeKeepsCore]]) }
    renderPage()
    expect(screen.getByTestId('theme')).toContainElement(screen.getByTestId('page'))
    expect(screen.queryByTestId('own-fallback')).toBeNull()
  })

  it("draws the handed-down core page inside the installed theme's frame, once (A-04d)", () => {
    underInstalledTheme()
    host = { ...installedHost(), pages: new Map<string, unknown>([['/chat', ThemeKeepsCore]]) }
    renderPage()
    expect(screen.getByTestId('main-layout')).toContainElement(screen.getByTestId('page'))
    expect(screen.getAllByTestId('page')).toHaveLength(1)
  })

  it('renders CorePage\'s own children in a slot, which has no core page to hand down', () => {
    underInstalledTheme()
    host = { ...installedHost(), slots: new Map<SlotId, unknown>([['notFound@1', ThemeLostKeepsCore]]) }
    render(
      <ThemeHostContext.Provider value={host}>
        <ThemeSlot slot="notFound@1">
          <div data-testid="page">The core 404</div>
        </ThemeSlot>
      </ThemeHostContext.Provider>,
    )
    expect(screen.getByTestId('own-fallback')).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
  })
})

describe('ThemeSlot, a slot', () => {
  function renderSlot() {
    const slot = (
      <ThemeSlot slot="notFound@1">
        <div data-testid="page">The core 404</div>
      </ThemeSlot>
    )
    return render(host === null ? slot : <ThemeHostContext.Provider value={host}>{slot}</ThemeHostContext.Provider>)
  }

  it("renders the theme's version where it registered one, at the theme layout, never framed", () => {
    underInstalledTheme()
    renderSlot()
    shows('theme')
    expect(screen.queryByTestId('main-layout')).toBeNull()
  })

  it("renders the core's version with no theme, below the theme's width, and for a slot the theme left out", () => {
    renderSlot()
    shows('page')
    cleanup()
    underInstalledTheme()
    state.layout = 'fallback'
    renderSlot()
    shows('page')
    cleanup()
    underInstalledTheme()
    host = { ...installedHost(), slots: new Map<SlotId, unknown>() }
    renderSlot()
    shows('page')
  })

  it('renders nothing while the width is not known, where the theme registered one (M69)', () => {
    underInstalledTheme()
    state.layout = 'pending'
    renderSlot()
    showsNothing()
  })
})

describe('app/chat/layout.tsx under builtin Play (its chat registered by lib/theme-host/builtin/play-slots.tsx)', () => {
  function underPlay(layout: ThemeLayout = 'theme') {
    state.entry = BUILTIN_PLAY_ENTRY
    state.layout = layout
  }

  function renderLayout() {
    return render(
      <ChatLayout>
        <div data-testid="page">The chat page</div>
      </ChatLayout>,
    )
  }

  it("renders Play's chat instead of the chat page under Play at desktop width, with the DiscussionBar", async () => {
    underPlay()
    renderLayout()
    expect(await screen.findByTestId('play-chat')).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.getByTestId('discussion-bar')).toBeInTheDocument()
  })

  it("never frames Play's chat a second time: PlayChat draws MainLayout itself (A-04d)", async () => {
    underPlay()
    renderLayout()
    expect(await screen.findByTestId('play-chat')).toBeInTheDocument()
    expect(screen.queryByTestId('main-layout')).toBeNull()
  })

  it('renders neither chat while the width is not known, the DiscussionBar still there (M69)', () => {
    underPlay('pending')
    renderLayout()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('play-chat')).toBeNull()
    expect(screen.getByTestId('discussion-bar')).toBeInTheDocument()
  })

  it('renders the existing chat page under Classic or Studio, and under Play below 1024 px', () => {
    renderLayout()
    expect(screen.getByTestId('page')).toHaveTextContent('The chat page')
    expect(screen.queryByTestId('play-chat')).toBeNull()
    expect(screen.getByTestId('discussion-bar')).toBeInTheDocument()
    cleanup()
    underPlay('fallback')
    renderLayout()
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(screen.queryByTestId('play-chat')).toBeNull()
  })

  it.each(['repo=projects%2Fsite', 'ticket=42&repo=x&runtime=1', 'mode=plan&from=assignments'])(
    'keeps the existing chat page for ?%s under Play',
    (search) => {
      underPlay()
      state.search = search
      renderLayout()
      expect(screen.getByTestId('page')).toBeInTheDocument()
      expect(screen.queryByTestId('play-chat')).toBeNull()
    },
  )
})
